"""Write a versioned regional data pool without modifying source artifacts."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from src.ai.rain_stations import station_alias
from .preprocessing import check_hourly, make_samples, fit_normalizers, calendar_features


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def make_rain_groups(standard, strict, windows, resolution_minutes=60):
    if not standard.index.equals(strict.index) or list(standard) != list(strict):
        raise ValueError('Standard/strict rainfall grids differ')
    columns, groups, values = {}, {}, []
    for sid in standard:
        block = []
        for hours in windows:
            name = f'rain_{sid}_sum_{hours}h'
            if 60 % resolution_minutes:
                raise ValueError('Resolution must divide one hour')
            stride = 60 // resolution_minutes
            series = standard[sid].copy()
            for offset in range(stride):
                series.iloc[offset::stride] = standard[sid].iloc[offset::stride].rolling(hours, min_periods=hours).sum().to_numpy()
            columns[name] = series
            columns[name+'_missing'] = series.isna().astype('uint8')
            block.extend([name, name+'_missing'])
            values.append(name)
        name = f'rain_{sid}_hour_estimated'
        columns[name] = (standard[sid].notna() & strict[sid].isna()).astype('uint8')
        block.append(name)
        groups[sid] = block
    return pd.DataFrame(columns, index=standard.index), groups, values


def build_dataset(output, *, level, rain_standard, rain_strict, weather, stations,
                  clusters, config, source_paths, rain_catalog):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refuse to overwrite {output}; select a new directory')
    output.mkdir(parents=True)
    write_json(output/'BUILDING.json', {'status': 'incomplete'})
    resolution_minutes = config.get('resolution_minutes', 60)
    check_hourly(level.index, resolution_minutes)
    if not level.index.equals(rain_standard.index) or not level.index.equals(weather.index):
        raise ValueError('Water, rainfall and weather must share the same complete hourly grid')
    if list(level) != list(stations):
        raise ValueError('Station mapping and water columns have different order')
    if len(stations) != config['expected_station_count']:
        raise ValueError('Unexpected station count')
    group_ids = set(v['cluster'] for v in stations.values())
    allowed = {0, 1, 2, -1} if config.get('allow_unclustered_stations') else {0, 1, 2}
    if ((config.get('require_all_eda_groups', True) and not {0, 1, 2}.issubset(group_ids))
            or not group_ids.issubset(allowed)):
        raise ValueError('All three EDA1 groups are required')
    if np.isinf(level.to_numpy()).any() or np.isinf(rain_standard.to_numpy()).any():
        raise ValueError('Infinite source values')
    rain, groups, rain_values = make_rain_groups(rain_standard, rain_strict, config['rain_windows'], resolution_minutes)
    calendar = calendar_features(level.index)
    rain_ids = list(rain_standard)
    aliases = {station_alias(r['name']): str(r['station_id']) for r in rain_catalog.to_dict('records')}
    aliases.update(nham='750007', bach_ma='750041', ho_hoa_my='682816', hong_thai='750062',
                   phu_bai='750014', khe_mo_rang='750042', huong_son='750028')
    aliases = {name: sid for name, sid in aliases.items() if sid in rain_ids}
    weather_values = [f'weather_forecast_rain_{a}_{b}h' for a, b in [(0, 3), (3, 6), (6, 12), (12, 24)]]+['weather_forecast_age_hours']
    weather_columns = [col for col in weather if col.startswith('weather_forecast_')]
    schema = dict(water_columns=['water_level_depth', 'water_level_missing'],
        calendar_columns=list(calendar.columns),
        calendar_groups={'base': ['flood_season'], 'hour': ['hour_sin', 'hour_cos'],
                         'day_of_year': ['day_sin', 'day_cos']},
        time_feature_modes={'none': ['flood_season'],
                            'hour': ['flood_season', 'hour_sin', 'hour_cos'],
                            'day_of_year': ['flood_season', 'day_sin', 'day_cos'],
                            'all': list(calendar.columns)},
        time_feature_formulas={'timezone': 'Asia/Ho_Chi_Minh',
            'hour': 'sin/cos(2*pi*local_hour/24)',
            'day_of_year': 'sin/cos(2*pi*(local_dayofyear-1)/(366 if local year is leap else 365))',
            'normalization': 'none; preserve cyclic values in [-1,1]',
            'default_mode': 'none'},
        flood_season='1 in local months 9,10,11,12; 0 otherwise; Asia/Ho_Chi_Minh; calendar feature, not an observed flood label',
        rain_groups=groups, rain_aliases=aliases, rain_value_columns=rain_values,
        weather_columns=weather_columns, weather_value_columns=weather_values,
        weather_mask_mapping={v: v+'_missing' for v in weather_values},
        target_columns=['target_h6_mm', 'target_h12_mm'], horizons_hours=[6, 12],
        station_id='separate integer station_index; never a numeric dynamic channel',
        sequence_order='oldest to newest: t-95h through t', sample_stride=1,
        raw_missing='NaN plus explicit masks; no raw zero imputation',
        normalized_missing='0 in standardized value channels, with masks preserved',
        forecast_history_policy='anchor_only; past weather values/provenance=0 and missing masks=1',
        normalization='per-station water and per-station/per-horizon absolute targets; shared rain/weather scalers',
        delta_normalization='delta_mm / target_scale; persistence affine coefficients stored per station/fold',
        units={'water': 'mm relative to station-specific unverified datum', 'rain': 'mm', 'age': 'hours'},
        physical_datum_warning='Do not equate station zero, DEM elevation or public alarm thresholds')
    samples = make_samples(level, stations, config['folds'], config['cutoff'], config['sequence_length'],
        config['minimum_water_history_fraction'], config['purge_hours'], resolution_minutes)
    normalizers = fit_normalizers(samples, level, rain, weather, schema, config['folds'], config['sequence_length'])
    default = config['folds'][-1]['id']
    constant_target_stations = []
    for sid in stations:
        if sid not in normalizers[default]['stations']:
            raise ValueError(f'No default training data for {sid}')
        train = samples[samples.station_id.eq(sid) & samples[default+'_train']]
        if not any(train[f'delta_h{h}_mm'].std() > 0 for h in [6, 12]):
            if sid not in config.get('allow_constant_target_station_ids', []):
                raise ValueError(f'No training target dynamics: {sid}')
            constant_target_stations.append(sid)
    for filename, frame in [('water.parquet', level), ('rainfall.parquet', rain),
                             ('rainfall_hourly_standard.parquet', rain_standard),
                             ('rainfall_hourly_strict.parquet', rain_strict), ('weather.parquet', weather), ('calendar.parquet', calendar)]:
        frame.rename_axis('timestamp_utc').reset_index().to_parquet(output/filename, index=False)
    samples.to_parquet(output/'samples.parquet', index=False)
    rain_catalog.to_parquet(output/'rain_station_catalog.parquet', index=False)
    clusters.to_csv(output/'eda1_station_groups.csv', index=False)
    write_json(output/'station_mapping.json', stations)
    write_json(output/'feature_schema.json', schema)
    write_json(output/'normalizers.json', normalizers)
    write_json(output/'config.json', config)
    # Public report with all necessary per-station sample counts and missingness.
    summary = []
    for sid, station in stations.items():
        part = samples[samples.station_id.eq(sid)]
        summary.append(dict(station_id=sid, name=station['name'], cluster=station['cluster'],
            samples=len(part), train=int(part.split.eq('train').sum()),
            validation=int(part.split.eq('validation').sum()), test_exploratory=int(part.split.eq('test_exploratory').sum()),
            purged=int(part.split.eq('purged').sum()), missing_water_pct=float(100*level[sid].isna().mean()),
            first_anchor=str(part.anchor_utc.min()), last_anchor=str(part.anchor_utc.max())))
    pd.DataFrame(summary).to_csv(output/'station_summary.csv', index=False)
    rain_summary = rain_catalog[['station_id', 'name', 'latitude', 'longitude']].copy()
    rain_summary['hour_missing_pct'] = rain_summary.station_id.map(rain_standard.isna().mean()*100)
    rain_summary['estimated_hour_pct'] = rain_summary.station_id.map((rain_standard.notna() & rain_strict.isna()).mean()*100)
    train_positions = np.unique(samples.loc[samples[default+'_train'], 'anchor_row'])
    rain_summary['train_anchor_missing_pct'] = rain_summary.station_id.map(rain_standard.iloc[train_positions].isna().mean()*100)
    rain_summary.to_csv(output/'rain_summary.csv', index=False)
    manifest = dict(status='complete', format_version=2, created_at=pd.Timestamp.now(tz='UTC').isoformat(),
        sequence_length=config['sequence_length'], resolution_minutes=resolution_minutes, output_size=2, horizons_hours=[6, 12],
        station_count=len(stations), rain_station_count=len(groups), sample_count=len(samples),
        hourly_rows=len(level), default_fold=default, default_rain_stations='nham+bach_ma+ho_hoa_my+hong_thai',
        default_input_size=3+sum(len(groups[s]) for s in ['750007', '750041', '682816', '750062'])+len(weather_columns),
        default_time_features='none', time_feature_modes=['none', 'hour', 'day_of_year', 'all'],
        stored_calendar_features=list(calendar.columns),
        cutoff=config['cutoff'], test_status='exploratory; post-cutoff data have been examined previously',
        weather_grid_id=config['weather_grid_id'], forecast_release_lag_hours=6, max_forecast_age_hours=24,
        forecast_availability='retrospective assumed availability +6h unless observed operational evidence exists; hindcasts stay labelled',
        observation_availability='retrospective hourly observations; receipt by anchor not proven for all observations',
        selection=config.get('station_selection_description', 'EDA1 clustered and Stage4 eligible; groups derived before cutoff but not before every inner validation fold'),
        constant_training_target_stations=constant_target_stations,
        source_sha256={str(Path(p).resolve()): sha(p) for p in source_paths},
        source_code_sha256={str(p.resolve()): sha(p) for p in Path(__file__).parent.glob('*.py')})
    write_json(output/'manifest.json', manifest)
    report = ['# Regional dataset: chuẩn bị dữ liệu cho hai horizon +6h và +12h', '',
        f'Trạm không có biến động target trong train: {", ".join(constant_target_stations) if constant_target_stations else "không có"}. Đây là ghi chú chất lượng, không sửa dữ liệu.', '',
        'Dataset này chưa được dùng huấn luyện model. Mỗi sample thuộc đúng một trạm nước, có lịch sử 96 giờ và hai nhãn theo timestamp chính xác. Mưa và forecast chỉ lưu một lần theo giờ, dùng chung cho mọi trạm.', '',
        '## Các trạm và số mẫu', '',
        '| Trạm | ID | Nhóm EDA1 | Train | Validation | Đánh giá thăm dò | Purged | Tổng | Thiếu nước (%) |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in summary:
        report.append(f"| {row['name']} | {row['station_id']} | {row['cluster']} | {row['train']} | {row['validation']} | {row['test_exploratory']} | {row['purged']} | {row['samples']} | {row['missing_water_pct']:.2f} |")
    report += ['', '## Phạm vi và cách chia thời gian', '',
        f"Dữ liệu giờ: {level.index.min().tz_convert('Asia/Ho_Chi_Minh')} đến {level.index.max().tz_convert('Asia/Ho_Chi_Minh')}. Tổng {len(samples):,} sample. Mốc giờ trong bảng dưới là UTC+7; khoảng kết thúc là mốc cuối có sample.", '',
        '| Split | Anchor đầu | Anchor cuối | Số sample |', '|---|---|---|---:|']
    for split, part in samples.groupby('split'):
        report.append(f'| {split} | {part.anchor_utc.min().tz_convert("Asia/Ho_Chi_Minh")} | {part.anchor_utc.max().tz_convert("Asia/Ho_Chi_Minh")} | {len(part)} |')
    report += ['', 'Validation mặc định là fold summer của EDA2. Train anchor < 30/06/2026 00:00; validation từ 01/07/2026 00:00 và anchor +24h < 01/08/2026 00:00. Phần sau 01/08/2026 00:00 được gọi test_exploratory vì đã được xem nhiều lần. Purge 24h bảo đảm cả nhãn +6h/+12h không vượt sang tập tiếp theo. Lịch sử đầu vào của validation có thể chứa quan trắc từ train vì chúng đã xảy ra trước anchor. Không yêu cầu hai cửa sổ lịch sử liên tiếp không chồng nhau.', '',
        'Bốn fold rainy/winter/spring/summer của EDA2 được lưu thành cột membership riêng; mỗi fold có normalizer riêng chỉ fit trên train của fold đó. Trạm bắt đầu muộn có thể không có normalizer ở fold sớm và reader sẽ báo rõ, không mượn scaler tương lai.', '',
        '## Input, target và lựa chọn feature', '',
        f"Mặc định x có shape [96, {manifest['default_input_size']}], y có shape [2], station_index là số nguyên riêng. Hai water channels là mực nước chuẩn hóa riêng từng trạm và cờ thiếu. Mỗi trạm mưa có {len(config['rain_windows'])*2+1} channels: tổng mưa quá khứ {config['rain_windows']} giờ, cờ thiếu cho từng tổng và cờ giờ mưa ước lượng 5/6 mẫu. Có thể chọn bất kỳ tập con nào trong 55 trạm mưa bằng alias hoặc ID, không thay đổi danh sách sample.", '',
        'Forecast chỉ gồm ô ECMWF A Lưới 16.274164°N, 107.264960°E, bốn tổng 0–3/3–6/6–12/12–24h, cờ thiếu, tuổi forecast, cờ thiếu tuổi và ba cờ observed/assumed/hindcast. Các giá trị của một anchor đến từ cùng một run. Metadata chứa run ID, initialization, retrieval, available_at, grid, model version và provenance. Input chỉ dùng forecast tại timestep cuối (anchor_only).', '',
        'Raw NaN được giữ trong Parquet. Reader chuẩn hóa bằng thống kê train, điền 0 trong không gian chuẩn hóa và giữ cờ thiếu. Mask và cờ provenance không được chuẩn hóa. Target là mực nước quan trắc giờ tại đúng t+6h/t+12h, kèm delta để kiểm tra. Target thiếu bị loại, không nội suy. Sample có H(t) hợp lệ, đủ 96 giờ trên lưới và ít nhất 80% lịch sử nước có số đo. Không loại thêm sample do thay tổ hợp mưa hoặc tắt forecast.', '',
        'Persistence dùng hệ số riêng theo station và horizon: scale = water_std / target_std; offset = (water_mean − target_mean) / target_std. Không lấy cao độ DEM làm mốc nước và không trộn số đo các trạm thành một mốc tuyệt đối.', '',
        'Mỗi timestep có thêm flood_season dùng chung cho mọi trạm: 1 trong tháng 9–12, 0 trong các tháng còn lại theo giờ Việt Nam. Quy ước này giống dataset Dã Viên hiện tại; đây là lịch mùa, không phải nhãn xác nhận lũ. Đặc trưng giữ nguyên 0/1 và không chuẩn hóa.', '',
        'Calendar lưu đủ hour_sin/hour_cos theo chu kỳ 24 giờ và day_sin/day_cos theo chu kỳ năm. Công thức ngày dùng (dayofyear−1)/365 hoặc /366 theo năm địa phương, nên xử lý đúng ngày 29/02 và ranh giới năm. Tất cả lấy giờ Việt Nam; sin/cos giữ nguyên trong [-1,1], không fit scaler. Reader cho chọn time_features=none/hour/day_of_year/all mà không đổi sample hoặc rebuild: tương ứng mặc định 68/70/70/72 channels khi dùng bốn trạm mưa và forecast. flood_season luôn được giữ.', '',
        '## Mưa quan trắc và tỷ lệ thiếu', '',
        'Giữ đầy đủ 55 trạm trong pool để có thể đổi tổ hợp sau này; việc lưu một trạm không có nghĩa khuyến nghị dùng nó. Lựa chọn mặc định gồm Nhâm (A Lưới, phía tây), Hồng Thái 750062 (A Lưới), Bạch Mã 750041 (phía đông nam, khác VQG/Đỉnh Bạch Mã) và Hồ Hòa Mỹ 682816 (phía tây bắc). Các trạm Khe Mỏ Rang, Hương Sơn và Phú Bài được giữ để kiểm chứng các tín hiệu đã thấy trong EDA2. Một số trạm thiếu toàn bộ train; phải xem bảng dưới trước khi chọn.', '',
        'Mưa giờ dùng đúng kết quả EDA2: tổng 6/6 mẫu 10 phút; 5/6 mẫu được nhân 6/5 và gắn cờ ước lượng; ít hơn 5 mẫu giữ thiếu. Tổng nhiều giờ yêu cầu đầy đủ mọi giờ trong cửa sổ. Không biến thiếu thành không mưa.', '',
        '| Trạm mưa | ID | Thiếu giờ toàn kỳ (%) | Giờ ước lượng (%) | Thiếu tại train anchors (%) |', '|---|---|---:|---:|---:|']
    for r in rain_summary.itertuples():
        report.append(f'| {r.name} | {r.station_id} | {r.hour_missing_pct:.2f} | {r.estimated_hour_pct:.2f} | {r.train_anchor_missing_pct:.2f} |')
    report += ['', '## Độ đầy đủ và nguồn gốc forecast', '',
        f"Có {int(weather.run_id.notna().sum())}/{len(weather)} anchor giờ có đủ forecast. Số anchor có bằng chứng observed availability là {int(weather.weather_forecast_observed_availability.sum())}; số dùng assumed availability là {int(weather.weather_forecast_assumed_availability.sum())}. Số anchor gắn cờ hindcast là {int(weather.weather_forecast_hindcast.sum())}.", '',
        '| Loại archive | Số anchor giờ |', '|---|---:|']
    for kind, count in weather.archive_kind.value_counts(dropna=False).items():
        report.append(f'| {kind} | {count} |')
    report += ['', '## Giới hạn nghiên cứu và leakage', '',
        'Forecast có cả hindcast/tái dựng và archive chưa xác minh thời điểm phát hành. Chính sách +6h là giả định nghiên cứu; không coi initialization hoặc retrieval muộn là bằng chứng forecast từng có sẵn trong quá khứ. Các cờ và metadata bảo toàn sự phân biệt này. Quan trắc cũng chưa chứng minh đầy đủ receipt trước anchor. Dataset phù hợp nghiên cứu hồi cứu, chưa chứng minh chất lượng vận hành thời gian thực.', '',
        'Nhóm EDA1 và danh sách trạm được xác định từ dữ liệu trước cutoff, nhưng đã xem một phần inner validation; đây là giới hạn selection-informed của nghiên cứu, không phải phép chọn độc lập ở mọi fold. Không tối ưu trạm mưa theo phần sau cutoff. Các nhóm chỉ là metadata nghiên cứu, không tự gán ý nghĩa vật lý hoặc ngưỡng báo động.', '',
        'Không ghi đè dataset/model cũ hoặc sửa PostgreSQL. Các hash nguồn và cấu hình nằm trong manifest. Xem validation.json trong cùng thư mục để biết các phép kiểm tra đã thực sự chạy. README của src/ai/regional_data mô tả cách build và đọc dữ liệu.']
    content = '\n'.join(report)+'\n'
    if config.get('split_policy') == 'fixed_train_validation_test':
        first = config['folds'][0]
        boundary = pd.Timestamp(first['validation_start']).tz_convert('Asia/Ho_Chi_Minh')
        cutoff = pd.Timestamp(config['cutoff']).tz_convert('Asia/Ho_Chi_Minh')
        replacement = (
            f'Dataset dùng duy nhất một bộ train/validation/test cố định. Train là phần trước {boundary}; '
            f'validation từ {boundary} đến trước {cutoff}; test từ {cutoff} đến hết quan trắc '
            f'{config["observation_end"]}. Các mốc này là ranh giới khoảng dữ liệu, không phải mọi giờ đều đủ điều kiện làm anchor. '
            'Purge 24h loại anchor ở cuối train và cuối validation, không đẩy ngày bắt đầu validation/test sang ngày khác. '
            'Nhãn +6h/+12h đều phải nằm trước ranh giới tập tiếp theo. Anchor test cuối cách quan trắc cuối ít nhất 12 giờ. '
            'Lịch sử đầu vào được phép dùng quan trắc ở tập trước nếu chúng đã xảy ra trước anchor. '
            'Scaler được fit lại chỉ trên train; không có nhiều fold trong dataset này. '
            'Test vẫn là đánh giá thăm dò vì dữ liệu giai đoạn này đã từng được nghiên cứu.')
        a = content.index('Validation mặc định là fold summer')
        b = content.index('## Input, target', a)
        content = content[:a]+replacement+'\n\n'+content[b:]
    (output/'REPORT.md').write_text(content)
    return manifest
