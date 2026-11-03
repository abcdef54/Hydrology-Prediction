"""Adapt frozen EDA1/EDA2 artifacts into a selectable regional sequence pool."""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from research.station_analysis.phase2.rainfall_preprocessing import preprocess_canonical_water_levels
from research.station_analysis.phase2.spatial_weather.processing import synchronized_features
from src.ai.regional_data.builder import build_dataset
from src.ai.regional_data.validation import validate_dataset

ROOT = Path(__file__).resolve().parents[2]
GRID = 'ecmwf_ifs:16.274164:107.264960'


def prepare_weather(index, audit, points, grid_id=GRID):
    """Single-cell vintage selection under the same +6h/max24h EDA2 rules."""
    audit = audit[audit.location_id.eq(grid_id)].copy()
    if audit.empty:
        raise ValueError('A Lưới grid is absent')
    points = points[points.run_id.isin(audit.run_id)]
    selected = synchronized_features(pd.DataFrame(), [grid_id], index, 6, 24, audit, points)
    if selected.empty:
        raise ValueError('No usable forecast run')
    result = selected.set_index('anchor_utc').reindex(index)
    result = result.rename(columns={f'{grid_id}|run_id': 'run_id', f'{grid_id}|archive_kind': 'archive_kind'})
    for a, b in [(0, 3), (3, 6), (6, 12), (12, 24)]:
        key = f'weather_forecast_rain_{a}_{b}h'
        result[key] = result.pop(f'{grid_id}|rain_{a}_{b}h_mm')
        result[key+'_missing'] = result[key].isna().astype('uint8')
    result['weather_forecast_age_hours'] = result.pop('age_hours')
    result['weather_forecast_age_hours_missing'] = result.weather_forecast_age_hours.isna().astype('uint8')
    metadata = audit.drop_duplicates('run_id').set_index('run_id')
    for column in ['retrieved_at', 'available_at', 'availability_basis', 'provenance_class',
                   'strict_eligible_run', 'model', 'grid_latitude', 'grid_longitude']:
        result[column] = result.run_id.map(metadata[column])
    valid = result.run_id.notna()
    for column in ['initialized_at', 'retrieved_at', 'available_at']:
        result[column] = pd.to_datetime(result[column], utc=True)
    observed = valid & result.strict_eligible_run.fillna(False).astype(bool) & result.available_at.notna()
    result['weather_forecast_observed_availability'] = observed.astype('uint8')
    result['weather_forecast_assumed_availability'] = (valid & ~observed).astype('uint8')
    result['weather_forecast_hindcast'] = result.archive_kind.fillna('').str.contains('hindcast|reconstruct', case=False).astype('uint8')
    result['assumed_available_at'] = result.initialized_at+pd.Timedelta(hours=6)
    result['model_version'] = result.archive_kind.str.extract(r'(\d+r\d+)', expand=False)
    result['grid_id'] = grid_id
    result['cell_selection'] = 'land'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution-manifest', type=Path, default=ROOT/'data/processed/station_analysis/phase2/feature_ablation/comprehensive_v2/execution_manifest.json')
    parser.add_argument('--groups', type=Path, default=ROOT/'reports/station_analysis/tables/cluster_assignments.csv')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--station-ids', nargs='+', help='Explicit station IDs; can include unclustered stations with training data')
    parser.add_argument('--allow-constant-targets', action='store_true', help='Include explicitly selected stations with no training target changes')
    parser.add_argument('--start', help='Optional inclusive timestamp with explicit timezone')
    parser.add_argument('--end', help='Optional inclusive timestamp with explicit timezone')
    parser.add_argument('--validation-start', default='2026-07-31T00:00:00+07:00')
    parser.add_argument('--test-start', default='2026-09-15T00:00:00+07:00')
    args = parser.parse_args()
    if args.allow_constant_targets and not args.station_ids:
        parser.error('--allow-constant-targets requires explicit --station-ids')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    execution = json.loads(args.execution_manifest.read_text())
    config = json.loads(json.dumps(execution['config']))
    source_root = ROOT/'data/processed/station_analysis/phase2'
    paths = [args.execution_manifest, args.groups]
    availability_path = source_root/'inventory/tables/observation_availability.parquet'
    standard_path = source_root/'rainfall/tables/hourly_rainfall_standard.parquet'
    strict_path = source_root/'rainfall/tables/hourly_rainfall_strict.parquet'
    station_path = next(Path(p) for p in execution['inputs'] if p.endswith('/stations.parquet'))
    spatial_report = Path(execution['stage3b_report'])
    spec_path = spatial_report/'stage4_spatial_ablation_spec.json'
    spatial = Path(json.loads(spec_path.read_text())['synchronized_forecast_path']).parent
    audit_path, points_path = spatial/'tables/forecast_run_integrity.parquet', spatial/'points.parquet'
    eligibility_path = args.execution_manifest.parent/'eligibility.json'
    paths.extend([availability_path, standard_path, strict_path, station_path, spec_path, audit_path, points_path, eligibility_path])
    stations = pd.read_parquet(station_path)
    groups = pd.read_csv(args.groups, dtype={'station_id': str})
    eligible = pd.DataFrame(json.loads(eligibility_path.read_text()))
    default_ids = set(groups.loc[groups.status.eq('clustered_exploratory'), 'station_id']) & set(eligible.loc[eligible.eligible, 'station_id'])
    if args.station_ids:
        if len(args.station_ids) != len(set(args.station_ids)):
            raise ValueError('Duplicate station IDs')
        missing = set(args.station_ids) - set(stations.station_id)
        missing_groups = set(args.station_ids) - set(groups.station_id)
        ineligible = set(args.station_ids) - set(eligible.loc[eligible.eligible, 'station_id'])
        if missing or missing_groups or ineligible:
            raise ValueError(f'Unknown, missing metadata or ineligible stations: {sorted(missing | missing_groups | ineligible)}')
        ids = sorted(args.station_ids)
    else:
        ids = sorted(default_ids)
    logging.info('Selected %s stations%s', len(ids), ' explicitly' if args.station_ids else ' from EDA1 and Stage4 eligibility')
    mapping = {}
    meta = stations.set_index('station_id')
    for i, sid in enumerate(ids):
        cluster = groups.set_index('station_id').loc[sid]
        mapping[sid] = dict(station_index=i, name=str(meta.loc[sid, 'name']), cluster=int(cluster['cluster']) if pd.notna(cluster['cluster']) else -1,
            latitude=float(meta.loc[sid, 'latitude']), longitude=float(meta.loc[sid, 'longitude']),
            unit='mm', water_level_datum='unknown; not tied to DEM',
            hydrological_type=str(cluster.hydrological_type), measured_quantity=str(cluster.measured_quantity))
    standard, strict = pd.read_parquet(standard_path), pd.read_parquet(strict_path)
    selected_index = standard.index
    for value in [args.start, args.end]:
        if value and pd.Timestamp(value).tz is None:
            raise ValueError('--start/--end require explicit timezone')
    if args.start:
        selected_index = selected_index[selected_index >= pd.Timestamp(args.start)]
    # Match the retained local-dataset observation snapshot, rather than silently
    # extending its requested October evaluation period to October 21.
    end = args.end or '2026-10-15T14:00:00+07:00'
    selected_index = selected_index[selected_index <= pd.Timestamp(end)]
    if selected_index.empty:
        raise ValueError('Empty time range')
    standard, strict = standard.loc[selected_index], strict.loc[selected_index]
    availability = pd.read_parquet(availability_path)
    water_catalog = stations[stations.station_id.isin(ids)]
    level, _ = preprocess_canonical_water_levels(availability, water_catalog, selected_index)
    level = level[ids]
    logging.info('Selecting A Lưới forecast vintages for %s hours', len(selected_index))
    weather = prepare_weather(selected_index, pd.read_parquet(audit_path), pd.read_parquet(points_path))
    config = {k: config[k] for k in ['folds', 'cutoff', 'sequence_length', 'minimum_water_history_fraction', 'purge_hours']}
    validation_start, test_start = pd.Timestamp(args.validation_start), pd.Timestamp(args.test_start)
    if validation_start.tz is None or test_start.tz is None or not selected_index.min() < validation_start < test_start < selected_index.max():
        raise ValueError('Require ordered timezone-aware validation/test boundaries within the source range')
    config['folds'] = [dict(id='fixed', validation_start=validation_start.isoformat(), validation_end=test_start.isoformat())]
    config['cutoff'] = test_start.isoformat()
    config['split_policy'] = 'fixed_train_validation_test'
    config['observation_end'] = pd.Timestamp(end).isoformat()
    config.update(expected_station_count=len(ids) if args.station_ids else 11, rain_windows=[1, 3, 6, 12, 24, 48], weather_grid_id=GRID)
    if args.station_ids:
        config['allow_unclustered_stations'] = True
        config['allow_constant_target_station_ids'] = sorted(set(ids)-default_ids) if args.allow_constant_targets else []
        config['station_selection_description'] = ('Explicit user-selected IDs; cluster -1 is unclassified. '
            'Flood-monitoring measurements and station datum may require verification; '
            'standard quality and sample selection rules apply without manual correction.')
    output = args.output_dir or ROOT/'train_data'/('hue_regional_h6_h12_'+pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%SZ'))
    logging.info('Writing dataset to %s', output)
    build_dataset(output, level=level, rain_standard=standard, rain_strict=strict, weather=weather,
        stations=mapping, clusters=groups[groups.station_id.isin(ids)], config=config, source_paths=paths,
        rain_catalog=stations[stations.station_type.eq('rainfall')].sort_values('station_id').reset_index(drop=True))
    result = validate_dataset(output)
    logging.info('Dataset validated: %s', result)
    print(output.resolve())


if __name__ == '__main__':
    main()
