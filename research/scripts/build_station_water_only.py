"""Single-station dataset with the station's own water level as the only observed input (+3/+6/+12 h targets).

Default station: Cảng Cá Thuận An (0074647402). Samples, anchors and the fixed split are taken from the regional pool
(train_data/hue_regional_20261015_h6_h12_fixed_split_v1): same 96 h history rule, same train/validation/test membership.
Inputs per hour: water_level_depth, water_level_missing (reader-built) and the calendar flag flood_season. No rain, forecast
or other-station groups. Normalisers are fitted on train only (inputs over unique training-window hours, targets over
training samples), exactly as for the Dã Viên revision datasets.

Run: OPENBLAS_NUM_THREADS=1 .venv/bin/python -m research.scripts.build_station_water_only
Train: see models/thuan_an/REPRODUCE.md
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from src.ai.regional_data.preprocessing import context_rows, moments

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'train_data/hue_regional_20261015_h6_h12_fixed_split_v1'
STATION = '0074647402'
OUTPUT = ROOT / 'train_data/thuan_an'
HORIZONS = [3, 6, 12]
CALENDAR = ['flood_season', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def build(station=STATION, name='ta_water_only', output=OUTPUT):
    d = Path(output) / name
    if d.exists():
        raise FileExistsError(d)
    src_manifest = json.loads((SOURCE / 'manifest.json').read_text())
    config = json.loads((SOURCE / 'config.json').read_text())
    water = pd.read_parquet(SOURCE / 'water.parquet').set_index('timestamp_utc')[[station]]
    index = water.index
    calendar = pd.read_parquet(SOURCE / 'calendar.parquet').set_index('timestamp_utc')
    if not calendar.index.equals(index):
        raise ValueError('source grids differ')
    mapping_src = json.loads((SOURCE / 'station_mapping.json').read_text())
    mapping = {station: {**mapping_src[station], 'station_index': 0, 'source_station_index': mapping_src[station]['station_index']}}

    samples = pd.read_parquet(SOURCE / 'samples.parquet')
    samples = samples[samples.station_id.eq(station)].copy()
    if samples.empty:
        raise ValueError(f'no samples for {station}')
    level = water[station]
    for h in HORIZONS:
        samples[f'target_h{h}_mm'] = level.reindex(pd.DatetimeIndex(samples.anchor_utc) + pd.Timedelta(hours=h)).to_numpy()
        samples[f'target_h{h}_utc'] = samples.anchor_utc + pd.Timedelta(hours=h)
        samples[f'delta_h{h}_mm'] = samples[f'target_h{h}_mm'] - samples.water_now_mm
    complete = np.all([np.isfinite(samples[f'target_h{h}_mm']) for h in HORIZONS], axis=0)
    dropped = int((~complete).sum())
    samples = samples[complete].reset_index(drop=True)
    samples['station_index'] = 0
    samples['sample_id'] = np.arange(len(samples))
    order = ['sample_id', 'station_id', 'station_index', 'anchor_utc', 'anchor_row', 'history_start_row', 'history_coverage',
             'water_now_mm'] + [f'{p}{h}{s}' for h in HORIZONS for p, s in (('target_h', '_mm'), ('target_h', '_utc'), ('delta_h', '_mm'))] \
        + ['fixed_train', 'fixed_validation', 'split', 'refit_train']
    samples = samples[order]

    train = samples[samples.fixed_train]
    L = config['sequence_length']
    rows = context_rows(train.anchor_row, L, len(index))
    w = moments(level.iloc[rows])
    targets = {f'h{h}': moments(train[f'target_h{h}_mm']) for h in HORIZONS}
    station_norm = dict(water=w, targets=targets,
                        persistence_scale=[w['scale'] / targets[f'h{h}']['scale'] for h in (6, 12)],
                        persistence_offset=[(w['mean'] - targets[f'h{h}']['mean']) / targets[f'h{h}']['scale'] for h in (6, 12)],
                        persistence_horizons=[6, 12], training_samples=len(train),
                        input_fit_first=str(index[rows.min()]), input_fit_last=str(index[rows.max()]),
                        target_fit_last=str(train.target_h12_utc.max()))
    norm = dict(stations={station: station_norm}, shared={},
                train_anchor_first=str(train.anchor_utc.min()), train_anchor_last=str(train.anchor_utc.max()),
                shared_rain_fit_last=str(index[rows.max()]), shared_weather_fit_last=str(index[rows.max()]),
                fit_policy='train-only: water over unique training-window input hours; targets over training samples')
    if pd.Timestamp(station_norm['input_fit_last']) >= pd.Timestamp(config['folds'][0]['validation_start']):
        raise ValueError('normaliser fit reaches the validation period')

    d.mkdir(parents=True)
    write_json(d / 'BUILDING.json', {'status': 'incomplete'})
    reset = lambda f: f.rename_axis('timestamp_utc').reset_index()
    reset(water).to_parquet(d / 'water.parquet', index=False)
    reset(calendar[CALENDAR]).to_parquet(d / 'calendar.parquet', index=False)
    pd.DataFrame({'timestamp_utc': index}).to_parquet(d / 'rainfall.parquet', index=False)
    pd.DataFrame({'timestamp_utc': index}).to_parquet(d / 'weather.parquet', index=False)
    samples.to_parquet(d / 'samples.parquet', index=False)
    write_json(d / 'station_mapping.json', mapping)
    write_json(d / 'normalizers.json', {'fixed': norm})
    write_json(d / 'config.json', {**config, 'expected_station_count': 1, 'horizons_hours': HORIZONS, 'groups': []})
    write_json(d / 'feature_schema.json', dict(
        water_columns=['water_level_depth', 'water_level_missing'], calendar_columns=CALENDAR,
        calendar_groups={'base': ['flood_season'], 'hour': ['hour_sin', 'hour_cos'], 'day_of_year': ['day_sin', 'day_cos']},
        time_feature_modes={'none': ['flood_season'], 'hour': ['flood_season', 'hour_sin', 'hour_cos'],
                            'day_of_year': ['flood_season', 'day_sin', 'day_cos'], 'all': CALENDAR},
        flood_season='1 in local months 9,10,11,12; 0 otherwise; Asia/Ho_Chi_Minh',
        rain_groups={}, rain_aliases={}, rain_value_columns=[], weather_columns=[], weather_value_columns=[],
        target_columns=[f'target_h{h}_mm' for h in HORIZONS], horizons_hours=HORIZONS,
        sequence_order=f'oldest to newest: t-{L - 1}h through t', sample_stride=1,
        raw_missing='NaN in parquet; standardized NaN -> 0 by the reader with water_level_missing = 1',
        units={'water': 'mm relative to station datum'}))
    persistence = {sp: {f'h{h}': int((samples[samples.split == sp][f'delta_h{h}_mm'].abs() > 200).sum()) for h in HORIZONS}
                   for sp in ('train', 'validation', 'test_exploratory')}
    write_json(d / 'manifest.json', dict(
        status='complete', format_version=2, dataset=name, created_at=pd.Timestamp.now(tz='UTC').isoformat(),
        scope=f'local single-station ({mapping[station].get("name", station)} {station}), own water level only',
        sequence_length=L, horizons_hours=HORIZONS, output_size=len(HORIZONS), station_count=1, groups=[],
        input_features=3, input_feature_breakdown={'water': 2, 'flood_season': 1},
        sample_count=len(samples), split_counts=samples.split.value_counts().to_dict(),
        samples_dropped_missing_label=dropped, hourly_rows=len(index), default_fold='fixed',
        default_rain_stations='none', default_time_features='none', cutoff=config['cutoff'],
        test_status=src_manifest.get('test_status'), observation_availability=src_manifest.get('observation_availability'),
        persistence_large_errors_200mm=persistence,
        train_command='--dataset-dir <this dir> --horizon 1h --target-horizons 3 6 12 --rain-stations none '
                      '--use-weather-forecast false --feature-set water_level+flood_season',
        source_sha256={str(SOURCE / f): sha(SOURCE / f) for f in ('manifest.json', 'samples.parquet', 'water.parquet', 'calendar.parquet')},
        source_code_sha256={str(Path(__file__).resolve()): sha(Path(__file__).resolve())}))
    (d / 'BUILDING.json').unlink()
    return d, samples, persistence


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--station', default=STATION)
    ap.add_argument('--name', default='ta_water_only')
    ap.add_argument('--output-dir', type=Path, default=OUTPUT)
    a = ap.parse_args()
    d, s, p = build(a.station, a.name, a.output_dir)
    print(d, s.split.value_counts().to_dict())
    print('persistence |error| > 200 mm:', p)
