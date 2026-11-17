"""Execute reviewed Phase 2 inventory and rainfall stages."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import shutil
import sys
import tempfile
from uuid import uuid4

import pandas as pd

from research.station_analysis.config import ROOT, digest, sha256, write_json
from research.station_analysis.database import store_analysis_database
from .config import load_config
from .snapshot import analysis_anchors, extract_snapshot, load_frames, validate_snapshot

LOG = logging.getLogger(__name__)
STAGES = ('inventory', 'rainfall', 'weather', 'spatial-weather', 'features')


def _safe_paths(config, stage='inventory'):
    work = Path(config['work_dir'])
    output = Path(config['output_dir'])
    if stage == 'spatial-weather':
        return work / 'spatial_weather', output / 'stage3b_spatial_weather'
    if stage == 'weather':
        return work / 'weather', output / 'stage3_weather'
    if stage == 'rainfall':
        return work / 'rainfall', output / 'stage2_rainfall'
    return work / 'inventory', output / 'stage1_inventory'


def _code_hashes():
    return {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')}


def _inputs(config, snapshot_folder, directories):
    paths = [Path(config['master_plan'])]
    for directory in directories:
        paths.extend(p for p in directory.glob('*') if p.suffix in ['.json','.csv','.gz'])
    phase1 = Path(config['phase1_work_dir'])
    paths.extend((phase1/'tables').glob('*.parquet'))
    registry=phase1/'tables/dem_registry.parquet'
    if registry.exists():
        paths.extend(Path(p) for p in pd.read_parquet(registry).processed_path)
    return {
        'config': config, 'code': _code_hashes(),
        'snapshot': sha256(Path(snapshot_folder)/'manifest.json'),
        'artifacts': {str(p): sha256(p) for p in sorted(paths) if p.is_file()},
    }


def _cached(cache, fingerprint):
    if not cache.exists():
        return False
    record = json.loads(cache.read_text())
    return record['fingerprint'] == fingerprint and all(
        Path(p).is_file() and sha256(p)==expected for p,expected in record['outputs'].items())


def _snapshot(config, work, *, refresh=False):
    pointer = work/'current_snapshot.json'
    if pointer.exists() and not refresh:
        record=json.loads(pointer.read_text())
        folder = Path(record['directory'])
        if sha256(folder/'manifest.json')!=record['manifest_sha256']:
            raise ValueError('Frozen snapshot manifest changed; investigate before reusing it')
        validate_snapshot(folder, config)
        LOG.info('Reuse frozen source snapshot %s', folder.name)
        return folder
    snapshots = work/'snapshots'
    snapshots.mkdir(parents=True, exist_ok=True)
    # Extraction occurs into a new directory; a failed query cannot replace a prior snapshot.
    with tempfile.TemporaryDirectory(dir=snapshots, prefix='.extract-') as temp:
        staged = Path(temp)/'snapshot'
        extract_snapshot(config, staged)
        final = snapshots/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:8])
        staged.replace(final)
    write_json(pointer, {'directory': str(final), 'manifest_sha256': sha256(final/'manifest.json')})
    return final


def run_inventory(config, *, force=False, refresh_snapshot=False):
    from .artifacts import audit_datasets, dataset_directories, reuse_phase1
    from .observations import audit_observations
    from .forecasts import audit_forecasts
    from .provenance import audit_dataset_provenance
    work, output = _safe_paths(config)
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    source = _snapshot(config, work, refresh=refresh_snapshot)
    directories = dataset_directories(config)
    fingerprint = digest(_inputs(config, source, directories))
    cache = work/'cache.json'
    if not force and _cached(cache, fingerprint):
        LOG.info('Reuse completed Stage 1; no source query or analysis repeated')
        return output/'report.md'
    frames = load_frames(source)
    anchors = analysis_anchors(frames['observations'], frames['stations'], config)
    effective = {**config,
                 'observation_expected_interval_minutes': config['expected_interval_minutes'],
                 'rainfall_record_screen_mm': config['rain_outlier_threshold_mm']}
    LOG.info('Audit observations for %s stations / %s anchors',len(frames['stations']),len(anchors))
    tables = audit_observations(frames['observations'], frames['stations'], anchors, effective)
    LOG.info('Audit %s forecast vintages',len(frames['forecast_runs']))
    tables.update(audit_forecasts(frames['forecast_runs'],frames['forecast_points'],
                                 frames['weather_locations'],anchors,effective))
    LOG.info('Audit retained dataset alignment and forecast provenance')
    tables.update(audit_datasets(directories))
    tables.update(audit_dataset_provenance(directories,frames['forecast_runs'],effective,
                                          points=frames['forecast_points']))
    tables.update(reuse_phase1(config,frames['stations']))
    # Long tables remain usable by later stages; no rain/water correlation is computed here.
    table_dir = work/'tables'
    table_dir.mkdir(exist_ok=True)
    for name,frame in tables.items():
        frame.to_parquet(table_dir/f'{name}.parquet',index=False)
    tables['station_catalog'] = frames['stations']
    tables['forecast_locations'] = frames['weather_locations']
    LOG.info('Write inventory store, availability matrices, figures and report')
    from .reporting import create_inventory_report
    create_inventory_report(tables,frames,anchors,config,work,output,source)
    store_analysis_database(work/'inventory.sqlite', tables)
    write_json(work/'run_manifest.json', {
        'stage':'inventory','master_plan':str(config['master_plan']),
        'master_plan_sha256':sha256(config['master_plan']),
        'snapshot_directory':str(source),'snapshot_manifest_sha256':sha256(source/'manifest.json'),
        'code_sha256':_code_hashes(),'configuration':config,
        'analysis_start_utc':str(anchors.min()),'analysis_end_utc':str(anchors.max()),
        'rows_by_table':{name:len(frame) for name,frame in tables.items()},
        'completed_at_utc':datetime.now(timezone.utc).isoformat(),
        'executed_stages':['inventory'],'not_executed':['rainfall','weather','features','regional_training'],
    })
    outputs = [p for p in table_dir.glob('*.parquet')]
    outputs += [work/'inventory.sqlite',work/'run_manifest.json']
    outputs += [p for p in output.rglob('*') if p.is_file() and p.name!='pipeline.log']
    write_json(cache, {'fingerprint':fingerprint,'outputs':{str(p):sha256(p) for p in outputs}})
    LOG.info('Stage 1 complete: %s',output/'report.md')
    return output/'report.md'


def run_rainfall(config, *, force=False, refresh_snapshot=False):
    """Execute Stage 2: Rainfall EDA & Rainfall–Water Level Relationship Analysis."""
    from .rainfall_preprocessing import (
        preprocess_hourly_rainfall,
        preprocess_canonical_water_levels,
        compute_multi_scale_accumulations,
        extract_rain_station_dem,
        calculate_station_distance_matrix
    )
    from .rainfall_eda import (
        compute_station_rainfall_summary,
        compute_multi_scale_accumulation_summary,
        compute_seasonal_rainfall_summary,
        compute_spatial_correlation_matrix,
        analyze_spatial_distance_decay,
        cluster_rain_stations,
        extract_independent_rainfall_events
    )
    from .water_relationship import (
        extract_flood_episodes,
        compute_conditional_relationships,
        compute_leave_one_out_stability,
        compare_policy_sensitivity
    )
    from .station_selection import (
        evaluate_single_station_candidates,
        build_exclusion_list,
        evaluate_multistation_combinations
    )
    from .stage2_reporting import generate_stage2_figures, write_stage2_report

    work, output = _safe_paths(config, stage='rainfall')
    inv_work, _ = _safe_paths(config, stage='inventory')
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)

    source = _snapshot(config, inv_work, refresh=refresh_snapshot)
    frames = load_frames(source)
    anchors = analysis_anchors(frames['observations'], frames['stations'], config)

    rain_stations = frames['stations'][frames['stations']['station_type'] == 'rainfall']
    water_stations = frames['stations'][frames['stations']['station_type'] == 'water_level']

    # Load observation availability for water levels
    avail_path = inv_work / 'tables/observation_availability.parquet'
    if not avail_path.exists():
        LOG.info('Running inventory first to generate availability table...')
        run_inventory(config, force=False)
    avail_df = pd.read_parquet(avail_path)

    LOG.info('1. Extract DEM elevation and planar distance matrix...')
    dem_df = extract_rain_station_dem(rain_stations)
    dist_matrix = calculate_station_distance_matrix(rain_stations, water_stations)

    LOG.info('2. Preprocess hourly rainfall under 4-tier policy...')
    p_std, p_str, p_q, records = preprocess_hourly_rainfall(frames['observations'], rain_stations, anchors)

    LOG.info('3. Extract canonical water levels and changes...')
    pivot_h, targets = preprocess_canonical_water_levels(avail_df, water_stations, anchors)

    LOG.info('4. Compute multi-scale rolling accumulations...')
    accum_std, accum_str = compute_multi_scale_accumulations(p_std, p_str)

    LOG.info('5. Compute Stage 2A Rainfall EDA metrics...')
    summary_df = compute_station_rainfall_summary(p_std, rain_stations, dem_df)
    accum_summary_df = compute_multi_scale_accumulation_summary(accum_std)
    seasonal_df = compute_seasonal_rainfall_summary(p_std, rain_stations)
    p_corr, s_corr = compute_spatial_correlation_matrix(p_std)
    pairs_df, decay_df = analyze_spatial_distance_decay(p_corr, rain_stations)
    clusters_df = cluster_rain_stations(p_corr, rain_stations, dem_df)
    events_df = extract_independent_rainfall_events(p_std)

    LOG.info('6. Compute Stage 2B Relationships and Event Stability...')
    episodes = extract_flood_episodes(pivot_h, targets)
    key_pairs = [
        ('0074647403', '750062', 6, 0),  # Dã Viên - Hồng Thái
        ('0074647403', '750007', 6, 0),  # Dã Viên - Nhâm
        ('0074647403', '750041', 6, 0),  # Dã Viên - Bạch Mã (750041)
        ('0074647403', '750030', 6, 0),  # Dã Viên - VQG Bạch Mã
        ('0074647403', '750014', 6, 0),  # Dã Viên - Hồ Phú Bài
        ('0074647403', '750061', 6, 0),  # Dã Viên - Bình Điền
    ]
    cond_df = compute_conditional_relationships(accum_std, targets, water_stations, rain_stations, key_pairs)
    loo_df = compute_leave_one_out_stability(accum_std, targets, water_stations, rain_stations, episodes, key_pairs)
    sens_df = compare_policy_sensitivity(accum_std, accum_str, targets, key_pairs)

    cutoff = config['future_analysis']['grouping_end']
    LOG.info('7. Select stations with +12h target purge at %s', cutoff)
    eval_df = evaluate_single_station_candidates(accum_std, targets, water_stations, rain_stations, dist_matrix, dem_df, train_end_utc=cutoff)
    excl_df = build_exclusion_list(p_std, rain_stations, train_end_utc=cutoff)

    candidate_subsets = {
        'Single: Hồng Thái': ['750062'],
        'Single: Bạch Mã (750041)': ['750041'],
        'Model B (2 stations): Hồng Thái + Nhâm': ['750062', '750007'],
        'Pair (Upstream + Ridge): Hồng Thái + Bạch Mã': ['750062', '750041'],
        '3-Station Arm: Hồng Thái + Bạch Mã + Phú Bài': ['750062', '750041', '750014'],
        '3-Station Alt: Hồng Thái + Nhâm + Bạch Mã': ['750062', '750007', '750041'],
        '4-Station: Hồng Thái + Bạch Mã + Bình Điền + VP': ['750062', '750041', '750061', '750033'],
        '4-Station Prod: Hồng Thái + Nhâm + Bạch Mã + Phú Bài': ['750062', '750007', '750041', '750014']
    }
    multi_eval_df = evaluate_multistation_combinations(accum_std, targets, water_stations, candidate_subsets, train_end_utc=cutoff)

    LOG.info('8. Generate figures, CSV tables, and master report...')
    figures = generate_stage2_figures(
        dem_df, rain_stations, water_stations, p_corr, clusters_df,
        accum_std, targets, cond_df, loo_df, multi_eval_df, output
    )
    report_file = write_stage2_report(
        summary_df, seasonal_df, decay_df, clusters_df, events_df,
        eval_df, cond_df, loo_df, sens_df, multi_eval_df, excl_df,
        figures, output, anchors=anchors, cutoff=cutoff,
        rainfall=p_std, targets=targets, episodes=episodes, snapshot=source, stations=frames['stations']
    )

    # Save processed intermediate Parquet tables in work/tables
    table_dir = work / 'tables'
    table_dir.mkdir(exist_ok=True)
    summary_df.to_parquet(table_dir / 'station_rainfall_summary.parquet', index=False)
    p_std.to_parquet(table_dir / 'hourly_rainfall_standard.parquet')
    p_str.to_parquet(table_dir / 'hourly_rainfall_strict.parquet')
    p_corr.to_parquet(table_dir / 'spatial_correlation_pearson.parquet')
    clusters_df.to_parquet(table_dir / 'rainfall_clusters.parquet', index=False)
    eval_df.to_parquet(table_dir / 'single_station_candidates.parquet', index=False)
    multi_eval_df.to_parquet(table_dir / 'multistation_combinations.parquet', index=False)

    write_json(output / 'run_manifest.json', {
        'stage': 'rainfall',
        'master_plan': str(config['master_plan']),
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'figures_count': len(list((output/'figures').glob('*.png'))),
        'tables_count': len(list((output/'tables').rglob('*.csv'))),
        'configuration': config,
        'code_sha256': _code_hashes(),
        'snapshot_directory': str(source),
        'snapshot_manifest_sha256': sha256(source/'manifest.json'),
        'train_cutoff_utc': pd.Timestamp(cutoff).tz_convert('UTC').isoformat(),
        'target_purge_hours': 12,
        'selection_windows': [1, 3, 6, 12, 24, 48],
        'selection_lags_hours': list(range(0, 73, 2)),
        'analysis_start_utc': str(anchors.min()),
        'analysis_end_utc': str(anchors.max()),
        'output_sha256': {str(p.relative_to(output)): sha256(p)
                          for p in output.rglob('*')
                          if p.is_file() and 'audit_history' not in p.parts and p.suffix in ('.csv', '.md', '.png')},
        'latest_clarification_audit': 'clarification_manifest.json',
        'report_path': str(report_file)
    })
    LOG.info('Stage 2 complete: %s', report_file)
    return report_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.yaml'))
    parser.add_argument('--stage', choices=STAGES, default='inventory')
    parser.add_argument('--start')
    parser.add_argument('--end')
    parser.add_argument('--water-station-ids', nargs='+')
    parser.add_argument('--rain-station-ids', nargs='+')
    parser.add_argument('--work-dir')
    parser.add_argument('--output-dir')
    parser.add_argument('--refresh-snapshot', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--feature-config', type=Path,
                        help='Explicit reviewed Stage 4 configuration; keeps feature fitting opt-in')
    args = parser.parse_args()
    if args.stage == 'features':
        if args.feature_config is None:
            parser.error('Stage features requires reviewed --feature-config before fitting models.')
        if args.refresh_snapshot or args.force:
            parser.error('Stage 4 reuses frozen inputs and resumable fits; snapshot refresh/force is not supported.')
        from .feature_ablation.pipeline import execute
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
        print(execute(args.feature_config, args.water_station_ids), flush=True)
        return
    if args.stage not in ('inventory', 'rainfall', 'weather', 'spatial-weather'):
        parser.error(f'Stage {args.stage} requires review of Stage 3 deliverables before execution.')
    config = load_config(args.config, {key: getattr(args, key) for key in (
        'start', 'end', 'water_station_ids', 'rain_station_ids', 'work_dir', 'output_dir')})
    _, output = _safe_paths(config, stage=args.stage)
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(output / 'pipeline.log')])
    try:
        if args.stage == 'inventory':
            run_inventory(config, force=args.force, refresh_snapshot=args.refresh_snapshot)
        elif args.stage == 'rainfall':
            run_rainfall(config, force=args.force, refresh_snapshot=args.refresh_snapshot)
        elif args.stage == 'spatial-weather':
            from .spatial_weather.pipeline import run_spatial_weather
            run_spatial_weather(config, force=args.force, refresh_snapshot=args.refresh_snapshot)
        elif args.stage == 'weather':
            from .weather import run_weather
            run_weather(config, force=args.force, refresh_snapshot=args.refresh_snapshot)
    except Exception as exc:
        # Never dump a DB exception that could contain connection parameters.
        LOG.error('Pipeline failed (%s). Frozen inputs were not modified.', type(exc).__name__)
        if isinstance(exc, (ValueError, FileNotFoundError, FileExistsError)):
            LOG.error('%s', str(exc))
        raise SystemExit(1) from None


if __name__=='__main__':
    main()
