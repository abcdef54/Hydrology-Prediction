"""Evidence-linked Vietnamese report, per-station summaries and local SQLite export."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .config import write_json
from .database import store_analysis_database
from .metadata import load_sources


def _number(value, precision=3):
    return f'{value:,.{precision}f}'.replace(',', '_').replace('.', ',').replace('_', '.')


def markdown(frame):
    if frame.empty:
        return '_Không có hàng đủ điều kiện._'
    def cell(value):
        if value is None or value is pd.NA:
            return '—'
        if isinstance(value, float):
            return _number(value) if np.isfinite(value) else '—'
        if isinstance(value, int) and not isinstance(value, bool):
            return _number(value, 0)
        return str(value).replace('|', '/').replace('\n', ' ')
    rows = ['| ' + ' | '.join(map(str, frame.columns)) + ' |',
            '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
    rows += ['| ' + ' | '.join(cell(v) for v in row) + ' |'
             for row in frame.itertuples(index=False, name=None)]
    return '\n'.join(rows)


def create_report(hourly, metadata, tables, config):
    work, output = Path(config['work_dir']), Path(config['output_dir'])
    (output / 'stations').mkdir(exist_ok=True)
    (output / 'tables').mkdir(exist_ok=True)
    snapshot = json.loads((work / 'snapshot/manifest.json').read_text())
    sources = load_sources()
    raw = pd.read_parquet(work / 'snapshot/raw_summary.parquet')
    stats = tables['station_statistics'].merge(raw, on='station_id').merge(
        metadata[['station_id', 'name', 'static_group_hypothesis', 'measured_quantity',
                  'water_level_datum', 'alarm_basis']], on='station_id')
    terrain = tables['terrain']
    preferred_radius = 200 if 200 in config['radii_m'] else config['radii_m'][0]
    selected = terrain.loc[terrain.dem_source.eq('FABDEM') & terrain.radius_m.eq(preferred_radius)]
    stats = stats.merge(selected[['station_id', 'median_m', 'p10_m', 'p90_m', 'valid_fraction']],
                        on='station_id', how='left')
    tables = {**tables, 'station_metadata': metadata, 'raw_summary': raw,
              'station_overview': stats, 'hourly': hourly.reset_index()}
    for name, frame in tables.items():
        if name in {'hourly', 'changes'}:
            frame.to_csv(output / 'tables' / f'{name}.csv.gz', index=False, compression='gzip')
        else:
            frame.to_csv(output / 'tables' / f'{name}.csv', index=False)
    store_analysis_database(work / 'analysis.sqlite', tables)
    write_json(output / 'configuration.json', config)
    write_json(output / 'metadata_sources.json', sources)
    shutil.copyfile(work / 'snapshot/manifest.json', output / 'snapshot_manifest.json')
    boundary_meta = Path(config['boundary_file']).with_suffix('.source.json')
    boundary = json.loads(boundary_meta.read_text()) if boundary_meta.exists() else {}
    (output / 'report.md').write_text(
        render_report(hourly, metadata, tables, stats, selected, snapshot, boundary, sources, config),
        encoding='utf-8',
    )
    changes = tables['change_statistics']
    for row in stats.itertuples():
        sid = row.station_id
        meta = metadata.loc[metadata.station_id.eq(sid)]
        station_tables = {
            'Metadata và nguồn': meta.T.reset_index().rename(columns={'index': 'Trường', meta.index[0]: 'Giá trị'}),
            'Thống kê và khả dụng': stats.loc[stats.station_id.eq(sid)].T.reset_index(),
            'Địa hình': terrain.loc[terrain.station_id.eq(sid)],
            'Thay đổi 6/12h': changes.loc[changes.station_id.eq(sid)],
            'Thiếu liên tục': tables['missing_gaps'].loc[tables['missing_gaps'].station_id.eq(sid)],
            'Tháng': tables['monthly_statistics'].loc[tables['monthly_statistics'].station_id.eq(sid)],
        }
        content = [f'# {row.name} ({sid})', '', '[Báo cáo tổng hợp](../report.md)', '']
        for title, frame in station_tables.items():
            content += [f'## {title}', '', markdown(frame), '']
        (output / 'stations' / f'{sid}.md').write_text('\n'.join(content), encoding='utf-8')


def _local_time(value):
    if pd.isna(value):
        return 'chưa có dữ liệu'
    return pd.Timestamp(value).tz_convert('Asia/Ho_Chi_Minh').strftime('%H:%M ngày %d/%m/%Y')


def _display(frame, columns):
    return markdown(frame[list(columns)].rename(columns=columns))


def render_report(hourly, metadata, tables, stats, selected, snapshot, boundary, sources, config):
    """Explain the saved evidence for readers who did not perform the analysis."""
    names = metadata.set_index('station_id').name.to_dict()
    radius = selected.radius_m.iloc[0] if len(selected) else config['radii_m'][0]
    radius = f'{radius:g}'
    overview = stats.copy()
    overview['missing_percent'] = overview.missing_rate * 100
    overview['station_link'] = overview.apply(
        lambda row: f"[{row['name']}](stations/{row['station_id']}.md)", axis=1)
    availability_table = _display(overview, {
        'station_link': 'Trạm quan trắc', 'raw_records': 'Số bản ghi gốc',
        'valid_hours': 'Số giờ có số đo', 'missing_percent': 'Tỷ lệ thiếu toàn kỳ (%)',
        'longest_missing_hours': 'Khoảng thiếu dài nhất (giờ)',
        'strict_water_windows': 'Số mẫu đủ dữ liệu nước',
    })
    terrain_table = _display(selected.merge(metadata[['station_id', 'name']], on='station_id'), {
        'name': 'Trạm quan trắc', 'point_m': 'Cao độ ô tại tọa độ trạm (m)',
        'median_m': f'Cao độ trung vị trong {radius} m (m)',
        'p10_m': 'Phân vị 10% (m)', 'p90_m': 'Phân vị 90% (m)',
    })
    registry = tables['dem_registry'].copy()
    registry['coverage_percent'] = registry.roi_bounds_coverage_fraction * 100
    registry_table = _display(registry, {
        'dem_source': 'Nguồn địa hình', 'height': 'Số hàng ô lưới', 'width': 'Số cột ô lưới',
        'coverage_percent': 'Phạm vi nguồn phủ vùng phân tích (%)',
    })
    changes = tables['change_statistics'].copy()
    changes['name'] = changes.station_id.map(names)
    change_tables = []
    for horizon in config['horizons']:
        part = changes.loc[changes.horizon_hours.eq(horizon)]
        change_tables.append(f'### Thay đổi sau {horizon} giờ\n\n' + _display(part, {
            'name': 'Trạm quan trắc', 'valid_hours': 'Số cặp thời điểm',
            'std_mm': 'Độ lệch chuẩn của thay đổi (mm)',
            'abs_change_p95_mm': 'Phân vị 95% độ lớn thay đổi (mm)',
            'rise_above_threshold_hours': f"Số mốc tăng >{config['rapid_threshold_mm']:g} mm",
            'fall_below_threshold_hours': f"Số mốc giảm >{config['rapid_threshold_mm']:g} mm",
        }))
    group_names = {
        'coastal_lagoon': 'Cửa biển và đầm phá',
        'huong_bo_rivers': 'Hệ thống sông Hương và sông Bồ',
        'o_lau_river': 'Sông Ô Lâu',
        'southern_rivers_review_separately': 'Sông Bù Lu và sông Truồi',
        'urban_inundation_provisional': 'Các điểm theo dõi ngập đô thị, phân loại sơ bộ',
        'floodplain_inundation_provisional': 'Các điểm theo dõi ngập đồng bằng, phân loại sơ bộ',
    }
    physical_groups = []
    for group, part in stats.groupby('static_group_hypothesis', sort=False):
        physical_groups.append({'Nhóm đề xuất': group_names.get(group, group),
                                'Các trạm trong nhóm': '; '.join(part.name),
                                'Số trạm': len(part)})
    physical_table = markdown(pd.DataFrame(physical_groups))
    correlations = tables['grouping_correlations']
    pairs = correlations.loc[
        (correlations.station_a.eq('0074647403') | correlations.station_b.eq('0074647403'))
        & correlations.kind.isin(['change_6h', 'change_12h']) & correlations.lag_hours.eq(0)
    ].copy()
    pairs['Trạm đối chiếu với Dã Viên'] = pairs.apply(lambda row: names.get(
        row.station_b if row.station_a == '0074647403' else row.station_a, 'Chưa rõ'), axis=1)
    correlation_rows = []
    for name, part in pairs.groupby('Trạm đối chiếu với Dã Viên', sort=False):
        result = {'Trạm đối chiếu với Dã Viên': name}
        for horizon in [6, 12]:
            entry = part.loc[part.kind.eq(f'change_{horizon}h')]
            result[f'Tương quan thay đổi {horizon}h'] = entry.iloc[0].pearson_r if len(entry) else np.nan
            result[f'Số cặp {horizon}h'] = int(entry.iloc[0].paired_hours) if len(entry) else 0
        correlation_rows.append(result)
    correlation_table = markdown(pd.DataFrame(correlation_rows))
    assignments = tables['cluster_assignments']
    cluster_rows = []
    for cluster, part in assignments.dropna(subset=['cluster']).groupby('cluster'):
        cluster_rows.append({'Nhóm thống kê': f'Nhóm {int(cluster)+1}',
                             'Các trạm được xếp cùng nhóm': '; '.join(part.name), 'Số trạm': len(part)})
    cluster_table = markdown(pd.DataFrame(cluster_rows))
    scores = tables['cluster_scores'].copy()
    scores['chosen_vi'] = scores.selected.map({True: 'Có', False: 'Không'})
    score_table = _display(scores, {'k': 'Số nhóm được thử', 'silhouette': 'Điểm silhouette',
                                   'chosen_vi': 'Được lựa chọn trong lần phân tích này'})
    selected_scores = scores.loc[scores.selected]
    cluster_summary = (
        f"Trong {len(assignments)} trạm được xem xét, {assignments.cluster.notna().sum()} trạm "
        f"có đủ đặc trưng để phân nhóm. Phương án có điểm cao nhất chia các trạm thành "
        f"{int(selected_scores.iloc[0].k)} nhóm, với điểm silhouette bằng "
        f"**{_number(selected_scores.iloc[0].silhouette)}**."
        if len(selected_scores) else
        'Dữ liệu chưa đáp ứng điều kiện để chọn một phương án phân nhóm thống kê.'
    )
    sensitivity = tables['cluster_sensitivity']
    sensitivity_text = (
        f"Khi bỏ các đặc trưng chu kỳ, thuật toán chọn {int(sensitivity.iloc[0].without_spectral_k)} "
        f"nhóm thay vì {int(sensitivity.iloc[0].selected_k)} nhóm. Chỉ số so sánh hai cách phân nhóm "
        f"Adjusted Rand Index (ARI) bằng **{_number(sensitivity.iloc[0].adjusted_rand_index)}**. "
        'Chỉ số này bằng 1 khi hai cách phân nhóm hoàn toàn trùng nhau và gần 0 khi mức trùng '
        'khớp tương đương ngẫu nhiên. Kết quả cho thấy cách chia nhóm phụ thuộc vào bộ đặc trưng được chọn.'
        if len(sensitivity) and sensitivity.iloc[0].spectral_features_used else
        'Trong lần phân tích này chưa có đủ đặc trưng chu kỳ để thực hiện một đối chứng riêng về ảnh hưởng của chúng.'
    )
    excluded = assignments.loc[assignments.cluster.isna()].copy()
    reasons = {
        'insufficient_observed_hours': 'Chưa có đủ số giờ quan trắc trong giai đoạn phân nhóm',
        'constant_or_no_robust_scale': 'Phần lớn giá trị quá ít thay đổi nên không tính được thang chuẩn hóa đã chọn',
        'incomplete_dynamic_features': 'Chưa tính được đầy đủ các đặc trưng biến động',
        'measurement_unit_not_project_mm': 'Đơn vị nguồn chưa phù hợp với điều kiện phân nhóm hiện tại',
        'training_period_cutoff_required': 'Chưa xác định giai đoạn dùng để phân nhóm',
        'grouping_cutoff_mismatch': 'Mốc thời gian của đặc trưng chưa khớp cấu hình',
    }
    excluded['Giải thích'] = excluded.exclusion_reason.map(
        lambda value: '; '.join(reasons.get(reason, reason) for reason in value.split(';') if reason)
        or 'Chưa có đủ cấu trúc dữ liệu để thuật toán tạo nhóm')
    excluded_table = _display(excluded, {'name': 'Trạm chưa được phân nhóm', 'Giải thích': 'Lý do'})
    associations = tables['terrain_associations']
    association_rows = []
    for horizon in [6, 12]:
        for normalized in [False, True]:
            feature = f"change_{horizon}h_iqr_{'ratio' if normalized else 'mm'}"
            part = associations.loc[associations.terrain_feature.eq('median_m')
                                    & associations.dynamic_feature.eq(feature)]
            if len(part):
                association_rows.append({
                    'Đặc trưng so với cao độ trung vị': f'Độ phân tán thay đổi {horizon}h'
                    + (' sau chuẩn hóa' if normalized else ' theo đơn vị mm'),
                    'Số trạm': int(part.iloc[0].n), 'Tương quan Spearman': part.iloc[0].spearman_rho,
                })
    association_table = markdown(pd.DataFrame(association_rows))
    quality = tables.get('quality_summary', pd.DataFrame())
    quality_table, quality_intro, quality_examples = '', '', ''
    if len(quality):
        quality = quality.merge(metadata[['station_id', 'name']], on='station_id')
        quality_intro = (
            f"Quy trình kiểm tra đã đánh dấu **{int(quality.hourly_jump_flags.sum())} bước nhảy theo giờ** "
            f"có độ lớn trên {config.get('quality_jump_mm', 1000):g} mm, "
            f"**{int(quality.isolated_spike_flags.sum())} trường hợp tăng hoặc giảm đột ngột rồi trở về ngay**, "
            f"và **{int(quality.raw_within_hour_large_ranges.sum())} giờ** có chênh lệch lớn giữa các số đo gốc trong cùng giờ. "
            'Một thời điểm có thể thỏa mãn nhiều điều kiện, vì vậy không cộng các con số này để suy ra tổng số lỗi.'
        )
        quality_table = _display(quality, {
            'name': 'Trạm quan trắc', 'hourly_jump_flags': 'Bước nhảy giờ cần kiểm tra',
            'isolated_spike_flags': 'Biến động đột ngột rồi trở về',
            'raw_within_hour_large_ranges': 'Giờ có chênh lệch lớn giữa các số đo gốc',
        })
        flags = tables['quality_flags']
        examples = []
        for row in flags.loc[flags.kind.eq('isolated_reverting_spike')].itertuples():
            examples.append(
                f"Tại {names.get(row.station_id, row.station_id)}, giá trị ở {_local_time(row.start_utc)} "
                f"là {_number(row.value_mm, 2)} mm, trong khi hai giờ liền trước và sau lần lượt là "
                f"{_number(row.previous_mm, 2)} mm và {_number(row.next_mm, 2)} mm. Đây là trường hợp nên kiểm tra "
                'lại số đo gốc và tình trạng thiết bị trước khi dùng làm mẫu học về nước dâng nhanh.'
            )
        plateaus = flags.loc[flags.kind.eq('low_variability_rolling_plateau')
                             & flags.station_id.eq('750043')].sort_values('hours', ascending=False)
        if len(plateaus):
            row = plateaus.iloc[0]
            examples.append(
                f"Tại Phú Xuân, một đoạn kéo dài {int(row.hours)} giờ, từ {_local_time(row.start_utc)} "
                f"đến {_local_time(row.end_utc)}, có giá trị trung vị khoảng {_number(row.value_mm, 2)} mm "
                f"và khoảng biến thiên của cả đoạn là {_number(row.magnitude_mm, 2)} mm. "
                'Các đoạn gần như nằm ngang này cần được đối chiếu với đặc điểm vận hành của cảm biến; '
                'chỉ nhìn chuỗi số chưa đủ để khẳng định thiết bị bị lỗi.'
            )
        quality_examples = '\n\n'.join(examples)
    alarm_count = metadata[[f'alarm_level_{level}_m' for level in [1, 2, 3]]].notna().any(axis=1).sum()
    alarm_text = (
        f"Cơ sở dữ liệu hiện chưa lưu ngưỡng báo động riêng cho cả {len(metadata)} trạm được phân tích."
        if alarm_count == 0 else
        f"Có {alarm_count} trạm lưu ít nhất một ngưỡng báo động, nhưng vẫn cần xác minh nguồn và mốc áp dụng."
    )
    start, end = _local_time(hourly.index.min()), _local_time(hourly.index.max())
    grouping_end = _local_time(config['grouping_end'])
    source_links = {
        'copernicus': 'Tài liệu sản phẩm Copernicus DEM',
        'fabdem_vietnam': 'Bộ dữ liệu FABDEM Vietnam của Đại học Bristol',
        'fabdem_readme': 'Tài liệu mô tả và hệ quy chiếu của FABDEM Vietnam',
        'official_alerts': 'Văn bản trả lời của cơ quan thủy lợi Huế về các ngưỡng báo động lũ',
        'coastal_geography': 'Cổng thông tin Huế: địa hình đầm phá và vùng biển ven bờ',
        'coastal_tides': 'Cổng thông tin Huế: chế độ hải văn vùng biển',
        'sinh_confluence': 'Cổng thông tin du lịch Huế: vị trí làng Sình',
        'truoi_aqueduct': 'Phản hồi của đơn vị quản lý thủy lợi về Cầu Máng',
        'flood_towers': 'Thông tin địa phương về các tháp báo lũ và báo ngập',
    }
    source_list = '\n'.join(
        f"- [{source_links.get(key, source['title'])}]({source['url']}). "
        + ('Nguồn này đã được đối chiếu với tài liệu gốc.' if source['status'] == 'verified_primary'
           else 'Thông tin đã được đối chiếu ở mức nội dung được công cụ tìm kiếm trích dẫn; việc áp dụng cho từng thiết bị vẫn cần xác minh.')
        for key, source in sources['sources'].items()
    )
    content = f"""# Phân tích dữ liệu và địa hình các trạm mực nước tại Huế

## 1. Mục tiêu nghiên cứu và kết luận chính

Nghiên cứu này được thực hiện để chuẩn bị cho việc xây dựng **Regional ResidualLSTM**, tức là một mô hình dùng chung trọng số để dự báo mực nước tại nhiều trạm ở Huế. Mỗi mẫu dữ liệu vẫn thuộc về một trạm cụ thể, nhưng mô hình có thể học từ các chuỗi quan trắc của nhiều trạm thay vì chỉ học từ Dã Viên. Câu hỏi cần trả lời ở bước này là liệu dữ liệu giữa các trạm có đủ chất lượng và đủ tương đồng để huấn luyện chung hay không.

Báo cáo sử dụng **{len(metadata)} trạm**, **{_number(snapshot['raw_records'], 0)} bản ghi quan trắc gốc** và một lịch phân tích gồm **{_number(len(hourly), 0)} mốc giờ**, từ **{start} đến {end}**. Toàn bộ thời gian trình bày trong báo cáo sử dụng giờ Việt Nam (UTC+7). Tài liệu tham khảo ban đầu liệt kê 17 trạm; danh mục trong cơ sở dữ liệu còn có Cầu Rào Bại trên sông Bù Lu và Cầu Máng trên sông Truồi, nên phân tích toàn bộ hiện gồm 19 trạm.

Kết quả cho thấy có thể tiếp tục chuẩn bị thí nghiệm Regional Model, nhưng **chưa nên gộp tất cả các trạm thành một tập huấn luyện chỉ vì chúng cùng nằm tại Huế hoặc có cao độ gần nhau**. Một số trạm có lịch sử quan trắc dài và biến động tương đồng, trong khi một số trạm khác mới có dữ liệu ngắn, có những đoạn cần kiểm tra chất lượng hoặc chưa xác minh rõ ý nghĩa đại lượng đo. Kết quả phân nhóm thống kê cũng chưa tạo ra những nhóm tách biệt rõ ràng.

Đề xuất của báo cáo là kiểm tra dữ liệu và ý nghĩa phép đo trước, sau đó huấn luyện một Regional Model trên các trạm đáp ứng điều kiện. Mô hình này cần được đánh giá riêng tại từng trạm; nếu kết quả chưa tốt thì tiếp tục tinh chỉnh mô hình chung cho từng trạm. **Báo cáo này chưa bao gồm kết quả huấn luyện Regional Model và chưa cung cấp bằng chứng rằng mô hình đó sẽ tốt hơn Model B.**

## 2. Dữ liệu được phân tích và cách đọc các bảng

Dữ liệu được lấy từ một bản chụp nhất quán của cơ sở dữ liệu PostgreSQL. Quá trình trích xuất chỉ đọc dữ liệu, còn kết quả tính toán được lưu riêng để có thể kiểm tra và chạy lại. Các bản ghi quan trắc gốc không bị chỉnh sửa.

Các số đo gốc thường xuất hiện cách nhau khoảng 10 phút. Để phù hợp với hướng dự báo theo giờ, các số đo hợp lệ trong mỗi giờ được lấy trung bình thành một giá trị đại diện cho giờ đó. Nếu một giờ không có số đo hợp lệ thì giờ đó được giữ là thiếu dữ liệu; quy trình không tự điền bằng giá trị của giờ trước hoặc nội suy từ các giờ lân cận. Giờ cuối chưa thu thập đầy đủ cũng không được đưa vào chuỗi phân tích.

Đơn vị mực nước đang được hệ thống lưu và xử lý là milimét (mm). Tuy nhiên, việc cùng sử dụng đơn vị mm chưa chứng minh rằng các trạm đo cùng một đại lượng hoặc có chung mốc cao độ. Ví dụ, một thiết bị có thể đo mực nước so với mốc riêng của trạm, trong khi một thiết bị theo dõi ngập có thể đo độ sâu nước so với mặt nền. Sự khác biệt này phải được xác minh trước khi dùng chung cách chuẩn hóa và diễn giải đầu ra mô hình.

Trong các bảng dưới đây, “thay đổi sau 6 giờ” được tính bằng mực nước đo được sau 6 giờ trừ mực nước ở thời điểm ban đầu. Cách tính sau 12 giờ cũng tương tự. Các cặp giá trị được ghép đúng theo thời gian, nên khi thiếu số đo ở thời điểm cần đối chiếu, quy trình không lấy một bản ghi gần đó để thay thế.

## 3. Mức độ đầy đủ và khả năng tạo mẫu huấn luyện

Bảng dưới cho biết số lượng quan trắc và mức độ thiếu dữ liệu của từng trạm. Tỷ lệ thiếu được tính trên cùng khoảng thời gian phân tích để người đọc có thể so sánh. Với một trạm mới lắp đặt, những giờ trước khi trạm bắt đầu có dữ liệu vẫn được tính là thiếu trong bảng này; vì vậy, tỷ lệ thiếu cao không đồng nghĩa với việc thiết bị đã ngừng hoạt động trong toàn bộ khoảng đó. Các bảng chi tiết có thêm tỷ lệ thiếu chỉ trong giai đoạn trạm thực sự có quan trắc.

Một “mẫu đủ dữ liệu nước” cần có {config['sequence_length']} giờ lịch sử liên tục và có giá trị quan trắc ở các mốc dự báo {', '.join(str(h) for h in config['horizons'])} giờ sau đó. Số mẫu này mới phản ánh khả năng sử dụng chuỗi mực nước. Khi tạo dataset huấn luyện thực tế, vẫn phải kiểm tra thêm dữ liệu mưa, dự báo thời tiết và ranh giới chia tập huấn luyện, xác thực, kiểm thử.

{availability_table}

Các trạm có lịch sử dữ liệu dài trong bộ số liệu này, như Dã Viên, Tứ Phú, Ngã Ba Sình và hai cảng, có chuỗi theo giờ gần như đầy đủ trong giai đoạn khảo sát. Ngược lại, Rào Bại và Cầu Máng mới có một khoảng quan trắc ngắn nên chưa đại diện được cho nhiều tình huống thủy văn. Hòa Xuân và Thủy Thanh cũng cần được xem riêng vì phần thời gian có dữ liệu ngắn hơn phần lớn các trạm còn lại.

Số mẫu lớn không có nghĩa là số trận lũ độc lập cũng lớn. Hai mẫu được tạo ở hai giờ liền nhau sẽ dùng phần lớn cùng một chuỗi lịch sử. Vì vậy, việc đánh giá mô hình sau này phải xét cả số đợt nước biến động mạnh, thay vì chỉ dựa vào tổng số cửa sổ dữ liệu.

![Tỷ lệ có dữ liệu của từng trạm theo tháng](figures/01_monthly_coverage.png)

*Hình 1 cho thấy mức độ có dữ liệu theo từng tháng. Hình này giúp phân biệt những trạm có lịch sử dài với những trạm chỉ bắt đầu ghi nhận ở cuối giai đoạn khảo sát.*

## 4. Địa hình quanh các trạm và giới hạn của dữ liệu DEM

DEM là mô hình số biểu diễn độ cao của bề mặt dưới dạng các ô lưới. Nghiên cứu sử dụng hai nguồn đã được cung cấp: FABDEM Vietnam và Copernicus DEM. FABDEM được ưu tiên vì đã xử lý để giảm ảnh hưởng của cây và công trình; Copernicus được dùng để đối chiếu vì sản phẩm này vẫn phản ánh cả các đối tượng nằm trên mặt đất.

Để giới hạn dung lượng xử lý, hai raster được cắt theo đường bao Thừa Thiên Huế do geoBoundaries cung cấp, sau đó mở rộng thêm {config['boundary_buffer_m']:g} m và bổ sung vùng quanh từng trạm. Bộ đường bao này biểu diễn địa giới năm **{boundary.get('boundaryYearRepresented', 'chưa xác định')}**. Vì vậy, phạm vi cắt được sử dụng để bao phủ địa hình phục vụ nghiên cứu, chứ chưa được xem là bản xác nhận địa giới hành chính đang áp dụng. Phạm vi này cũng không thay thế cho việc xác định lưu vực và mạng lưới kết nối sông.

{registry_table}

Tỷ lệ bao phủ trong bảng mô tả phần vùng phân tích nằm trong phạm vi không gian của raster nguồn. FABDEM bao phủ toàn bộ phạm vi đã chọn, còn ô Copernicus được cung cấp chưa phủ hết vùng này. Tuy vậy, cả hai nguồn đều có dữ liệu tại các vùng lân cận được khảo sát quanh toàn bộ trạm trong lần chạy hiện tại. Những phần nguồn không bao phủ được giữ là thiếu dữ liệu, thay vì tạo thêm cao độ giả định.

Cao độ được lấy tại tọa độ từng trạm và trong các vùng tròn có bán kính {', '.join(f'{r:g}' for r in config['radii_m'])} m. Việc dùng thêm vùng lân cận giúp giảm nguy cơ diễn giải sai từ một ô đơn lẻ nằm trên mặt sông, cầu hoặc công trình. Bảng sau trình bày cao độ trung vị và các phân vị trong bán kính {radius} m. Phân vị 10% và 90% cho biết khoảng chứa phần lớn các cao độ trong vùng, qua đó thể hiện mức độ chênh lệch địa hình quanh trạm.

{terrain_table}

Đối với các điểm đặt gần sông hoặc cảng, những ô có cao độ rất thấp hoặc bằng nhau có thể phản ánh mặt nước đã được làm phẳng trong sản phẩm DEM. Do đó, cao độ lấy tại tọa độ trạm không nhất thiết là cao độ nền lắp thiết bị. Chênh lệch giữa giá trị tại điểm và thống kê vùng lân cận cần được xem cùng bản đồ trước khi chọn làm đặc trưng cho mô hình.

Hai DEM sử dụng hệ độ cao EGM2008 theo tài liệu nguồn. Trong khi đó, tài liệu hiện có chưa xác nhận mốc cao độ của từng cảm biến quan trắc. **Vì chưa xác định được quan hệ giữa hai loại mốc này, không thể lấy mực nước tại trạm trừ trực tiếp cao độ DEM để tính độ sâu ngập.** DEM cũng không cung cấp độ sâu đáy sông hoặc cao độ số 0 của thước đo tại trạm.

![Địa hình FABDEM và vị trí các trạm](figures/06_station_terrain_map.png)

*Hình 2 thể hiện vị trí các trạm trên nền địa hình FABDEM. Các số thứ tự tương ứng với thứ tự trạm trong bảng danh mục. Bản đồ được dùng để xem bối cảnh địa hình, chưa phải bản đồ lưu vực hoặc mô phỏng hướng dòng chảy.*

![Đối chiếu địa hình từ hai nguồn DEM](figures/07_terrain_comparison.png)

*Hình 3 so sánh cao độ trung vị và khoảng phân vị 10–90% của hai nguồn DEM quanh từng trạm. Chênh lệch giữa hai nguồn là thông tin để xem xét độ nhạy của đặc trưng địa hình, chưa phải sai số đã được xác định bằng đo đạc thực địa.*

## 5. Biến động mực nước và những trường hợp cần kiểm tra chất lượng

![Chuỗi mực nước theo giờ tại từng trạm](figures/02_station_time_series.png)

*Hình 4 cho phép quan sát trực tiếp các đợt nước lên, xuống và các khoảng thiếu. Mỗi trạm có một trục tung riêng để tránh che mất biến động nhỏ; vì vậy không nên so độ cao của đường vẽ giữa hai ô mà bỏ qua đơn vị và thang đo.*

Chuỗi quan trắc giữa các trạm có những khác biệt rõ rệt. Các trạm sông và cảng thường có biến động liên tục, trong khi nhiều điểm theo dõi ngập có phần lớn giá trị gần như không thay đổi và chỉ biến động trong một số khoảng ngắn. Một số chuỗi còn có bước nhảy hoặc đoạn nằm ngang kéo dài cần được đối chiếu với hoạt động thiết bị.

{quality_intro}

{quality_table}

{quality_examples}

Các điều kiện kiểm tra trên chỉ giúp xác định những đoạn nên xem xét kỹ hơn, chứ không đủ để quyết định bản ghi nào sai. Một khoảng ít thay đổi có thể là trạng thái nước ổn định thực sự, còn giá trị âm có thể hợp lệ nếu số 0 của trạm nằm cao hơn mặt nước. Quy trình hiện tại giữ nguyên tất cả các giá trị này, đồng thời lưu thời gian và lý do đánh dấu để phục vụ kiểm tra sau.

Điều này có ý nghĩa trực tiếp đối với Regional Model: nếu biến động do thiết bị hoặc thay đổi mốc đo bị đưa vào huấn luyện như một đợt lũ, mô hình có thể học sai cách nước phản ứng. Khi xử lý chất lượng dữ liệu, cần lưu một lớp dữ liệu dẫn xuất kèm lý do xử lý để vẫn truy nguyên được về quan trắc ban đầu.

## 6. Mực nước thay đổi thế nào sau 6 giờ và 12 giờ?

Độ lệch chuẩn trong các bảng dưới mô tả mức phân tán của thay đổi mực nước. Phân vị 95% của độ lớn thay đổi cho biết có khoảng 95% cặp quan trắc có mức thay đổi không vượt quá giá trị đó. Chỉ tiêu này giúp phân biệt những trạm thường ít biến động với những trạm có biên độ thay đổi lớn hơn trong khoảng dự báo.

Hai cột cuối đếm những mốc mà mực nước thực tế tăng hoặc giảm vượt {config['rapid_threshold_mm']:g} mm. **Đây là số mốc biến động của quan trắc, không phải số dự báo có sai số trên 200 mm của một mô hình.** Các số đếm vẫn có thể bao gồm những đoạn đang được đánh dấu kiểm tra chất lượng và các mốc liên tiếp thuộc cùng một sự kiện.

{chr(10).join(change_tables)}

Sự khác nhau giữa hai khoảng dự báo cho thấy không thể giả định rằng trạm biến động mạnh ở 6 giờ cũng sẽ biến động mạnh tương ứng ở 12 giờ. Biến động có chu kỳ, dòng chảy lũ và các bất thường của thiết bị có thể tạo ra các phân phối khác nhau. Khi đánh giá mô hình, cần báo cáo riêng từng trạm và từng mốc dự báo để tránh che mất những khác biệt này trong một chỉ số trung bình chung.

![Phân phối thay đổi mực nước sau 6 giờ và 12 giờ](figures/04_horizon_changes.png)

*Hình 5 cho thấy độ phân tán và chiều tăng, giảm của thay đổi mực nước. Các đường tham chiếu thể hiện ngưỡng biến động được dùng để đếm trong bảng, không phải ngưỡng báo động lũ.*

![Phân phối mực nước sau khi trừ trung vị của từng trạm](figures/03_centered_distributions.png)

*Hình 6 đã trừ trung vị riêng của từng trạm để dễ quan sát độ phân tán. Thao tác này chỉ phục vụ biểu đồ và không chứng minh rằng các trạm đã có chung mốc cao độ hoặc chung đại lượng đo.*

## 7. Các trạm nào có biến động tương đồng với Dã Viên?

Để tìm hiểu khả năng chia sẻ thông tin giữa các trạm, nghiên cứu so sánh thay đổi mực nước trong cùng khoảng 6 giờ hoặc 12 giờ tại những thời điểm trùng nhau. Bảng sử dụng hệ số tương quan Pearson: giá trị gần 1 cho thấy hai chuỗi thường tăng, giảm cùng nhau; giá trị gần 0 cho thấy quan hệ tuyến tính yếu; giá trị âm cho thấy xu hướng biến động trái chiều.

Phân tích phục vụ đề xuất nhóm chỉ dùng dữ liệu **trước {grouping_end}**, là mốc bắt đầu tập xác thực của dataset Dã Viên hiện có trong cấu hình mặc định. Mỗi cặp trạm cần có ít nhất {config['min_pair_hours']} cặp giờ đủ số đo để tính tương quan. Cách giới hạn thời gian này giúp tách việc đề xuất nhóm khỏi phần dữ liệu sẽ được dùng để đánh giá sau đó. Các bảng thống kê mô tả ở những phần trước vẫn sử dụng toàn bộ khoảng khảo sát và không được coi là kết quả lựa chọn đặc trưng chỉ từ tập huấn luyện.

{correlation_table}

Dã Viên và Ngã Ba Sình có sự tương đồng rõ nhất trong các chuỗi thay đổi được so sánh. Tứ Phú cũng có tương quan dương với Dã Viên, nhưng mức tương quan khác nhau giữa khoảng 6 giờ và 12 giờ. Đây là những quan sát đáng kiểm chứng trong thí nghiệm huấn luyện chung, vì tên trạm và bối cảnh địa lý cho thấy các trạm này thuộc hệ thống sông có liên quan.

Tuy nhiên, tương quan không đủ để xác định trạm nào gây ảnh hưởng đến trạm nào, và cũng không bảo đảm rằng thêm dữ liệu của một trạm sẽ cải thiện dự báo ở trạm khác. Hai chuỗi có thể cùng phản ứng với mưa, thủy triều hoặc một tác động chung. Những ô có dấu “—” biểu thị trường hợp chưa tính được hệ số đáng tin cậy theo điều kiện đã đặt ra, chẳng hạn thiếu thời gian trùng nhau hoặc một chuỗi gần như không đổi.

![Tương quan thay đổi 6 giờ giữa các trạm trên toàn kỳ mô tả](figures/05_change_correlation.png)

*Hình 7 thể hiện tương quan thay đổi 6 giờ trên toàn kỳ quan trắc để người đọc xem tổng quan. Vì hình dùng khoảng thời gian rộng hơn bảng đề xuất nhóm ở trên, hai nơi có thể cho hệ số khác nhau; không dùng trực tiếp hình này để lựa chọn nhóm trên tập huấn luyện.*

## 8. Nên phân nhóm trạm theo tiêu chí nào?

### 8.1. Nhóm dựa trên địa lý và loại phép đo

Các trạm được đề xuất xem xét theo những nhóm sau để thuận tiện cho việc kiểm chứng. Các nhóm sông dựa vào tên sông trong danh mục và thông tin địa lý đã đối chiếu. Những nhóm theo dõi ngập vẫn là phân loại sơ bộ vì cần xác nhận vị trí lắp đặt và cách thiết bị quy định số 0.

{physical_table}

{alarm_text} Các tài liệu công khai có ngưỡng chính thức cho Kim Long, Phú Ốc và Thượng Nhật, nhưng chưa có căn cứ để áp dụng các ngưỡng đó cho thiết bị khác trong danh sách này. Vì vậy, báo cáo không sử dụng ngưỡng Kim Long cho Dã Viên hoặc ngưỡng Phú Ốc cho Tứ Phú.

Thông tin địa lý cũng cho thấy cần xem xét ảnh hưởng thủy triều tại hai cửa biển. Tuy nhiên, tài liệu về chế độ triều của khu vực không thay thế cho việc xác nhận đáp ứng của từng cảm biến. Tương tự, thiết bị được ghi là “tháp báo lũ” hoặc “tháp báo ngập” chưa đủ để kết luận rằng giá trị lưu trong cơ sở dữ liệu chính là độ sâu nước trên mặt đường.

### 8.2. Kết quả phân nhóm bằng thống kê

Phương pháp KMeans được thử để xem các trạm có tự hình thành nhóm khi chỉ xét đặc điểm biến động hay không. Đầu vào gồm độ phân tán của thay đổi 6 giờ và 12 giờ, mức giống nhau của chuỗi với chính nó sau các khoảng trễ, cùng mức độ nổi bật của dao động khoảng 12 giờ và 24 giờ. Các đặc trưng được chuẩn hóa theo độ phân tán riêng của trạm; mực nước tuyệt đối, tọa độ và cao độ DEM không được dùng để quyết định nhóm.

{cluster_summary} Điểm silhouette có giá trị càng gần 1 thì các nhóm càng tách biệt rõ theo những đặc trưng đang xét; giá trị gần 0 cho thấy các nhóm còn chồng lấn. Vì thế, kết quả hiện tại nên được dùng để đặt giả thuyết nghiên cứu, chưa nên coi là phân loại thủy văn đã được xác nhận.

{score_table}

{cluster_table}

{sensitivity_text}

Các trạm chưa đủ điều kiện được giữ trong danh mục và được giải thích ở bảng dưới. Việc không được phân nhóm ở bước này không đồng nghĩa với việc trạm không có giá trị sử dụng. Đặc biệt, những chuỗi có phần lớn giá trị gần bằng nhau có thể vẫn chứa các đợt ngập hiếm nhưng quan trọng mà cách chuẩn hóa hiện tại chưa mô tả tốt.

{excluded_table}

Một số nhóm thống kê còn chứa cả trạm sông và điểm theo dõi ngập. Điều này cho thấy thuật toán chỉ nhận biết sự giống nhau của các đặc trưng được cung cấp; nó không xác nhận các thiết bị đo cùng một đại lượng hoặc có cùng cơ chế thủy văn. Việc chia model theo kết quả này ngay lập tức có thể tạo ra những nhóm không phù hợp về vật lý.

### 8.3. Cao độ địa hình có đủ để quyết định nhóm không?

Để kiểm tra giả thuyết phân nhóm theo cao độ, nghiên cứu so sánh cao độ trung vị quanh trạm với độ phân tán của thay đổi mực nước. Hệ số Spearman được dùng để xem các trạm có xu hướng được sắp xếp giống hay ngược nhau theo hai đặc trưng này. Độ phân tán trong bảng được tính bằng khoảng giữa phân vị 25% và 75%; phiên bản sau chuẩn hóa còn chia cho khoảng phân tán mực nước riêng của từng trạm.

{association_table}

Các kết quả cho thấy có những quan hệ đáng chú ý, nhưng chưa đủ để kết luận rằng cao độ quyết định cách phân nhóm. Phân tích chỉ dựa trên một số ít trạm đủ đặc trưng, trong khi các ô DEM quanh cảng và sông có thể bị chi phối bởi mặt nước. Đại lượng đo và chất lượng dữ liệu của các trạm cũng chưa được xác nhận đồng nhất. Ngoài ra, nhiều quan hệ đã được khảo sát cùng lúc nên không nên chọn riêng một hệ số lớn rồi coi đó là bằng chứng chắc chắn.

Do đó, **cao độ DEM nên được giữ lại như thông tin bối cảnh để kiểm chứng trong mô hình**, còn quyết định chia nhóm cần xem đồng thời loại phép đo, hệ sông, ảnh hưởng triều, chất lượng quan trắc và kết quả đánh giá dự báo.

## 9. Đề xuất thí nghiệm Regional Model

Trước khi huấn luyện, cần xác nhận ý nghĩa phép đo và rà soát những đoạn đã được đánh dấu chất lượng. Nhóm trạm có lịch sử dài như Dã Viên, Tứ Phú, Ngã Ba Sình, hai cảng và Huỳnh Liên là các ứng viên để xem xét trước. Hòa Xuân cần được kiểm tra thêm về thời gian thiếu và những bước chuyển giá trị. Các điểm theo dõi ngập nên được nghiên cứu riêng cho đến khi xác nhận được cách đo; hai trạm phía nam cần thêm thời gian quan trắc trước khi đánh giá khả năng tổng quát hóa.

Sau bước đó, thí nghiệm đầu tiên nên huấn luyện **một Regional ResidualLSTM dùng chung trọng số cho các trạm đủ điều kiện**, với lịch sử {config['sequence_length']} giờ và các mốc dự báo {', '.join(f'+{h}h' for h in config['horizons'])}. Mỗi mẫu phải thuộc về một trạm duy nhất, có mã nhận diện trạm và các đặc trưng mưa phù hợp. Cách chuẩn hóa và phép dự báo giữ nguyên mực nước hiện tại làm đường cơ sở cần được tính đúng cho từng trạm, chỉ từ dữ liệu huấn luyện.

Các trạm phải dùng cùng các ranh giới thời gian khi chia tập huấn luyện, xác thực và kiểm thử. Dữ liệu tương lai của một trạm khác cũng không được dùng để huấn luyện mô hình chung trước khi đánh giá tại trạm đích. Giữa các tập cần có khoảng cách bảo vệ phù hợp với mốc dự báo dài nhất, đồng thời phải thống nhất quy định về phần lịch sử được phép dùng tại ranh giới. Khi lấy mẫu, nên kiểm tra cách cân bằng đóng góp giữa các trạm để những trạm có nhiều bản ghi không chi phối toàn bộ quá trình học.

Kết quả phải được báo cáo riêng theo trạm, mốc dự báo và các đợt nước tăng hoặc giảm mạnh. Mô hình chung cần được so sánh với cách dự báo giữ nguyên mực nước hiện tại và với mô hình được huấn luyện riêng từ đầu trên từng trạm. Model B là đối chứng sẵn có cho Dã Viên, không phải đối chứng đã được huấn luyện cho toàn bộ các trạm khác.

**Nếu Regional Model chưa tốt, bước tiếp theo là lấy chính trọng số của mô hình đó để tinh chỉnh riêng cho từng trạm**, rồi đánh giá lại trên cùng các mốc kiểm thử. Các model riêng theo nhóm địa lý hoặc thủy văn có thể được thử sau, khi có bằng chứng rằng việc chia nhóm giúp ích. Kết quả clustering hiện tại chưa đủ mạnh để quyết định tạo nhiều model theo nhóm ngay từ đầu.

![Biến động theo mùa và khả năng tạo mẫu dữ liệu](figures/08_seasonality_and_windows.png)

*Hình 8 đối chiếu biến động trong các tháng được cấu hình là mùa lũ với các tháng còn lại, đồng thời thể hiện số mẫu đủ dữ liệu nước. Việc chia theo tháng chỉ mô tả lịch mùa; không có nghĩa mọi giờ trong mùa lũ đều thuộc một trận lũ.*

## 10. Tài liệu chi tiết, khả năng sử dụng lại và nguồn tham khảo

Các bảng đầy đủ được giữ riêng để báo cáo chính tập trung vào kết quả và cách diễn giải:

- [Bảng tổng hợp từng trạm](tables/station_overview.csv) chứa số lượng quan trắc, mức độ thiếu, thống kê mực nước và số mẫu đủ dữ liệu.
- [Bảng địa hình quanh trạm](tables/terrain.csv) chứa kết quả của cả hai DEM ở các bán kính đã khảo sát.
- [Bảng thay đổi mực nước](tables/change_statistics.csv) và [bảng những đoạn cần kiểm tra](tables/quality_flags.csv) cho phép đối chiếu các phát hiện trong báo cáo.
- [Bảng tương quan dùng để đề xuất nhóm](tables/grouping_correlations.csv) ghi cả hệ số và số cặp quan trắc thực tế.
- [Bảng quan hệ địa hình với biến động nước](tables/terrain_associations.csv) giữ các phép so sánh chi tiết, bao gồm những quan hệ không nổi bật.

Tên từng trạm trong bảng ở mục 3 dẫn tới trang thống kê riêng. Các trang này và các tệp CSV phục vụ tra cứu chi tiết; khi sử dụng số liệu để thiết kế model, cần giữ những giới hạn về chất lượng và ý nghĩa phép đo đã nêu trong báo cáo.

Các raster đã xử lý được lưu tại `{Path(config['work_dir']) / 'dem'}`, còn metadata và các bảng phân tích được lưu tại `{Path(config['work_dir']) / 'analysis.sqlite'}`. Mô hình lưu trữ này cho phép dùng lại địa hình và thống kê trong những lần chuẩn bị dataset sau mà chưa cần đưa toàn bộ raster vào PostgreSQL. Cấu hình, bản chụp dữ liệu và nguồn tham khảo cũng được lưu cùng kết quả để có thể kiểm tra lại cách tạo ra các con số.

Các nhận định về sản phẩm DEM, bối cảnh địa lý và ngưỡng báo động được đối chiếu với những nguồn sau. Mức kiểm chứng của từng nguồn được ghi rõ để phân biệt thông tin đã đọc từ tài liệu gốc với thông tin mới được xác nhận qua phần trích dẫn tìm kiếm.

{source_list}

Đường bao phục vụ cắt địa hình được lấy từ [geoBoundaries]({config['boundary_api']}), với nguồn gốc và năm biểu diễn được lưu trong metadata đi kèm. Tài liệu `hue_17_water_level_stations_web_research.md` do người dùng cung cấp được dùng làm điểm khởi đầu để đối chiếu địa danh; số bản ghi trong tài liệu đó không được dùng thay cho số liệu trích từ cơ sở dữ liệu trong lần phân tích này.

Bộ FABDEM Vietnam V1-0 được cung cấp theo giấy phép CC BY 4.0. Theo yêu cầu ghi công của dữ liệu, các thông tin sau được giữ nguyên:

> FABDEM is produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.
>
> The organisations in charge of the Copernicus programme by law or by delegation do not incur any liability for any use of the Copernicus WorldDEM-30.
"""
    return content
