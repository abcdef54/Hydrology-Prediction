"""Build the hourly top-four-rain dataset and chronological training splits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAIN_STATIONS = {
    '750062': 'Hồng Thái (A Lưới)',
    '750007': 'Nhâm (A Lưới)',
    '750003': 'Thị trấn A Lưới (A Lưới)',
    '750011': 'Hồ A Lá (A Lưới)',
}
WIND_STATIONS = ('Cảng Thuận An', 'Cảng Tư Hiền')
RAIN_WINDOWS = (3, 6, 12, 24)
TARGET_HOURS = (1, 3, 6, 12, 24)


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_observations(data_dir: Path, name: str, identifier: str, values: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.read_csv(data_dir / f'{name}.csv.gz',
                        usecols=[identifier, 'time_point', *values], dtype={identifier: str})
    frame['time'] = pd.to_datetime(frame['time_point'], utc=True, errors='coerce')
    frame = frame.dropna(subset=['time']).sort_values('time')
    for column in values:
        frame[column] = pd.to_numeric(frame[column], errors='coerce').replace([np.inf, -np.inf], np.nan)
    return frame


def station_hourly(frame: pd.DataFrame, value: str, aggregation: str) -> pd.Series:
    # The label t represents (t - 1 hour, t], matching the analysis notebook.
    grouped = frame.set_index('time')[value].resample('1h', closed='right', label='right')
    if aggregation == 'sum':
        return grouped.sum(min_count=1)
    if aggregation == 'mean':
        return grouped.mean()
    return grouped.last()


def build_table(data_dir: Path) -> pd.DataFrame:
    water = read_observations(data_dir, 'water_level_observations', 'station_id', ('depth',))
    water = water.loc[water['station_id'].str.lstrip('0').eq('74647403')]
    if water.empty:
        raise ValueError('No Dã Viên water-level observations')
    water_hourly = station_hourly(water, 'depth', 'mean')
    water_hourly.index.name = 'timestamp_utc'
    table = water_hourly.rename('water_level_depth').to_frame()
    for lag in (1, 3):
        table[f'water_level_lag_{lag * 60}m'] = water_hourly.shift(lag)
    table['water_level_missing'] = water_hourly.isna().astype('int8')

    rain = read_observations(data_dir, 'rain_observations', 'station_id', ('depth',))
    for code in RAIN_STATIONS:
        station_rain = rain.loc[rain['station_id'].eq(code)]
        if station_rain.empty:
            raise ValueError(f'No rainfall observations for {code}')
        # Compute windows before reindexing, retaining rain before the water series starts.
        hourly = station_hourly(station_rain, 'depth', 'sum')
        table[f'rain_{code}_depth'] = hourly.reindex(table.index)
        for hours in RAIN_WINDOWS:
            table[f'rain_{code}_sum_{hours}h'] = hourly.rolling(hours, min_periods=hours).sum().reindex(table.index)
        table[f'rain_{code}_missing'] = table[f'rain_{code}_depth'].isna().astype('int8')

    wind_info = pd.read_csv(data_dir / 'wind_stations.csv.gz', dtype=str)
    wind = read_observations(data_dir, 'wind_observations', 'sid', ('ws', 'wsg', 'wd', 'wdg'))
    availability = []
    for name in WIND_STATIONS:
        match = wind_info.loc[wind_info['name'].eq(name), 'id']
        if len(match) != 1:
            raise ValueError(f'Wind station name must resolve exactly once: {name}')
        sid = match.iloc[0]
        code = sid.replace('-', '_')
        observations = wind.loc[wind['sid'].eq(sid)]
        if observations.empty:
            raise ValueError(f'No wind observations for {name}')
        values = []
        for variable in ('ws', 'wsg', 'wd', 'wdg'):
            aggregation = 'mean' if variable in ('ws', 'wsg') else 'last'
            column = f'wind_{code}_{variable}'
            table[column] = station_hourly(observations, variable, aggregation).reindex(table.index)
            table[f'{column}_missing'] = table[column].isna().astype('int8')
            values.append(column)
        availability.append(table[values].notna().any(axis=1))
    available = pd.concat(availability, axis=1).sum(axis=1).astype('int8')
    table['wind_network_available_station_count'] = available
    table['wind_network_missing'] = available.eq(0).astype('int8')
    table['flood_season'] = table.index.tz_convert('Asia/Ho_Chi_Minh').month.isin([9, 10, 11, 12]).astype('int8')
    for hours in TARGET_HOURS:
        table[f'target_water_level_plus_{hours * 60}m'] = water_hourly.shift(-hours)
    return table


def split_table(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    # Retain the hourly grid and NaN labels for sequences; the loader excludes
    # invalid sequence endpoints, and tree training drops only missing labels.
    train_end = int(len(table) * 0.70)
    val_end = int(len(table) * 0.85)
    val_start, test_start = table.index[train_end], table.index[val_end]
    purge = pd.Timedelta(hours=max(TARGET_HOURS))
    train = table.iloc[:train_end]
    val = table.iloc[train_end:val_end]
    return {
        'train': train.loc[train.index + purge < val_start],
        'val': val.loc[val.index + purge < test_start],
        'test': table.iloc[val_end:],
    }


def audit(table: pd.DataFrame, splits: dict[str, pd.DataFrame]) -> dict[str, bool]:
    checks = {}
    def same(left, right):
        return bool(np.allclose(left, right, equal_nan=True))
    checks['hourly_cadence'] = bool(table.index.to_series().diff().dropna().eq(pd.Timedelta(hours=1)).all())
    checks['timestamps_unique'] = table.index.is_unique
    checks['no_reservoir_columns'] = not any(c.startswith('reservoir_') for c in table)
    checks['exactly_four_rain_stations'] = {c.split('_')[1] for c in table if c.startswith('rain_')} == set(RAIN_STATIONS)
    for lag in (1, 3):
        checks[f'water_lag_{lag}h'] = same(table[f'water_level_lag_{lag * 60}m'], table['water_level_depth'].shift(lag))
    for hours in TARGET_HOURS:
        checks[f'target_{hours}h'] = same(table[f'target_water_level_plus_{hours * 60}m'], table['water_level_depth'].shift(-hours))
    for code in RAIN_STATIONS:
        current = table[f'rain_{code}_depth']
        checks[f'rain_{code}_missing_flag'] = same(table[f'rain_{code}_missing'], current.isna())
        for hours in RAIN_WINDOWS:
            # Early rows can legitimately include rain observed before the first water hour.
            actual = table[f'rain_{code}_sum_{hours}h'].iloc[hours - 1:]
            expected = current.rolling(hours, min_periods=hours).sum().iloc[hours - 1:]
            checks[f'rain_{code}_past_only_{hours}h'] = same(actual, expected)
    expected_season = table.index.tz_convert('Asia/Ho_Chi_Minh').month.isin([9, 10, 11, 12])
    checks['local_month_flood_season'] = same(table['flood_season'], expected_season)
    checks['train_target_before_val'] = bool(splits['train'].index.max() + pd.Timedelta(hours=24) < splits['val'].index.min())
    checks['val_target_before_test'] = bool(splits['val'].index.max() + pd.Timedelta(hours=24) < splits['test'].index.min())
    for name, frame in splits.items():
        checks[f'{name}_hourly_cadence'] = bool(frame.index.to_series().diff().dropna().eq(pd.Timedelta(hours=1)).all())
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f'Dataset audit failed: {failed}')
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'train_data/1h_top4')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/hydrology_training_1h_top4.csv')
    parser.add_argument('--force', action='store_true', help='Replace only the requested derived outputs')
    args = parser.parse_args()
    paths = [args.output, *(args.output_dir / f'{name}.csv' for name in ('train', 'val', 'test')),
             args.output_dir / 'manifest.json', args.output.with_suffix('.md'),
             args.output.with_name(args.output.stem + '_missingness.csv')]
    if not args.force and any(path.exists() for path in paths):
        raise FileExistsError('Derived output already exists; use --force to rebuild')
    data_root = args.data_dir.resolve()
    if any(data_root == path.resolve() or data_root in path.resolve().parents for path in paths):
        raise ValueError('Derived outputs must be outside the raw data directory')
    table = build_table(args.data_dir)
    splits = split_table(table)
    checks = audit(table, splits)
    targets = [c for c in table if c.startswith('target_')]
    features = [c for c in table if c not in targets]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=True)
    for name, frame in splits.items():
        frame.to_csv(args.output_dir / f'{name}.csv', index=True)
    missingness = table[features].isna().mean().rename('missing_rate').to_frame()
    missingness.index.name = 'feature'
    missingness.to_csv(paths[-1])
    split_summary = {}
    for name, frame in splits.items():
        valid = frame[targets].notna().all(axis=1) & frame['water_level_depth'].notna()
        split_summary[name] = {
            'rows': len(frame), 'complete_label_and_current_level_rows': int(valid.sum()),
            'start_utc': str(frame.index.min()), 'end_utc': str(frame.index.max()),
            'flood_season_fraction': float(frame['flood_season'].mean()),
        }
    source_paths = [args.data_dir / f'{name}.csv.gz' for name in (
        'water_level_observations', 'rain_observations', 'wind_observations', 'wind_stations')]
    manifest = {
        'dataset': '1h_top4', 'resolution_minutes': 60,
        'rain_stations': RAIN_STATIONS, 'water_station': 'Dã Viên', 'wind_stations': WIND_STATIONS,
        'rain_windows_hours': [1, *RAIN_WINDOWS], 'target_horizons_hours': TARGET_HOURS,
        'aggregation': 'right-closed bins (t-1h,t], labelled t; mean water, sum rain; mean wind speed, last wind direction',
        'flood_season': '1 for months 9–12, otherwise 0, Asia/Ho_Chi_Minh',
        'missing_data': 'NaN preserved, no filling; missing indicators retained; all hourly rows retained in split CSVs',
        'split': '70/15/15 by chronological hourly grid; purge 24h before Val and Test',
        'station_selection': 'Four stations chosen from exploratory notebook correlations on the available period; not a train-only selection',
        'units': 'Original sensor units; no physical-unit conversion',
        'feature_columns': features, 'target_columns': targets,
        'rows': len(table), 'feature_count': len(features), 'splits': split_summary,
        'audit_checks': checks,
        'source_sha256': {str(p.relative_to(args.data_dir)): checksum(p) for p in source_paths},
        'split_sha256': {name: checksum(args.output_dir / f'{name}.csv') for name in splits},
    }
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    lines = [
        '# Dataset 1h: mưa 4 trạm, mực nước, gió và mùa lũ', '',
        f'- Tổng số dòng theo giờ: {len(table):,}; đầu vào: {len(features)}; đầu ra: {len(targets)}.',
        '- Mưa: Hồng Thái, Nhâm, Thị trấn A Lưới, Hồ A Lá; tổng 1/3/6/12/24h và cờ thiếu.',
        '- Mực nước: Dã Viên, giá trị hiện tại, lag 1h/3h, cờ thiếu.',
        '- Gió: Cảng Thuận An và Cảng Tư Hiền; các biến ws/wsg/wd/wdg và cờ thiếu.',
        '- flood_season: 1 trong tháng 9–12 theo giờ Việt Nam, 0 trong các tháng khác.',
        '- Không có dữ liệu hồ chứa; không chuyển đổi đơn vị đo.',
        '- Timestamp t là cuối giờ (t-1h, t], đồng nhất với notebook phân tích; khác cách gắn nhãn đầu giờ ở dataset cũ.',
        '- Chia 70/15/15 theo thời gian, purge 24h trước Val và Test; không lọc dòng theo độ đầy đủ của feature.',
        '- File split giữ dòng thiếu nhãn để chuỗi LSTM liên tục. LSTM bỏ điểm kết thúc thiếu nhãn; cây bỏ dòng thiếu nhãn khi train.',
        '- Các lựa chọn feature_set dùng cùng split để so sánh. flood_season là lịch mùa, không phải nhãn ngập lụt.',
        '- Bốn trạm được chọn từ phân tích toàn kỳ; các thử nghiệm mới vẫn là khảo sát, cần dữ liệu tương lai để xác nhận.', '',
        '| Tập | Dòng theo giờ | Đủ nhãn và mực nước hiện tại | Tỷ lệ giờ mùa lũ |',
        '|---|---:|---:|---:|',
    ]
    for name, summary in split_summary.items():
        lines.append(f"| {name} | {summary['rows']} | {summary['complete_label_and_current_level_rows']} | {summary['flood_season_fraction']:.1%} |")
    lines += ['', f'Kiểm tra: {sum(checks.values())}/{len(checks)} đạt.', '',
              'Tạo lại: `.venv/bin/python scripts/build_training_dataset_1h_top4.py --force`.',
              'Chọn khi train: `--horizon 1h --dataset-dir train_data/1h_top4 --feature-set water+rain+flood_season`.']
    args.output.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print(f'Wrote {args.output}: {len(table)} rows, {len(features)} features, {len(targets)} targets')
    print(json.dumps(split_summary, ensure_ascii=False, indent=2))
    print(f'Audit passed: {len(checks)}/{len(checks)} checks')


if __name__ == '__main__':
    main()
