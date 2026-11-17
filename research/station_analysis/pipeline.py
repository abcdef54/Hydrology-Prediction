"""Run all analysis stages or reuse a validated, frozen source snapshot."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import pandas as pd

from .config import ROOT, digest, file_identity, load_config, sha256, write_json

STAGES = ['extract', 'dem', 'statistics', 'grouping', 'plots', 'report']
DEPENDENCIES = {'extract': [], 'dem': ['extract'], 'statistics': ['extract'],
                'grouping': ['dem', 'statistics'], 'plots': ['grouping'],
                'report': ['plots']}
LOG = logging.getLogger(__name__)


def _write_tables(folder, tables):
    folder.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_parquet(folder / f'{name}.parquet', index=False)


def _read_tables(folder):
    return {path.stem: pd.read_parquet(path) for path in folder.glob('*.parquet')}


def _fingerprint(stage, config, work):
    source_dir = Path(__file__).parent
    modules = {'dem': ['dem.py', 'metadata.py'],
               'statistics': ['statistics.py', 'quality.py'],
               'grouping': ['grouping.py'], 'plots': ['visualization.py'],
               'report': ['reporting.py', 'database.py', 'metadata.py']}
    code = {name: sha256(source_dir / name)
            for name in modules.get(stage, []) + ['config.py', 'pipeline.py']}
    identity = {'config': config, 'code': code,
                'metadata_sources': sha256(source_dir / 'metadata_sources.yaml')}
    if stage == 'extract':
        # DB is deliberately frozen until explicit --refresh-snapshot.
        return digest({'selection': {key: config.get(key) for key in
                                    ('start', 'end', 'station_ids')},
                       'extractor': sha256(source_dir / 'database.py')})
    files = list((work / 'snapshot').glob('*'))
    if stage == 'dem':
        files += [Path(config['fabdem']), Path(config['copernicus'])]
        if Path(config['boundary_file']).exists():
            files.append(Path(config['boundary_file']))
    if stage in ['grouping', 'plots', 'report']:
        files += list((work / 'tables').glob('*.parquet'))
    if stage in ['plots', 'report']:
        files += list((work / 'dem').glob('*.tif'))
    identity['inputs'] = [file_identity(p) for p in sorted(files) if p.is_file()]
    # Own output tables must not invalidate a stage on every subsequent run.
    produced = {'grouping': {'cluster_assignments', 'cluster_scores', 'cluster_features',
                            'cluster_sensitivity', 'terrain_associations'}}
    if stage in produced:
        identity['inputs'] = [i for i in identity['inputs']
                              if Path(i['path']).stem not in produced[stage]]
    return digest(identity)


def _validate_prerequisites(stage, config, work):
    for prerequisite in DEPENDENCIES[stage]:
        _validate_prerequisites(prerequisite, config, work)
        cache = work / 'cache' / f'{prerequisite}.json'
        if not cache.exists():
            raise ValueError(f'Run prerequisite stage {prerequisite} before {stage}')
        saved = json.loads(cache.read_text())
        if saved['fingerprint'] != _fingerprint(prerequisite, config, work):
            raise ValueError(f'Stale {prerequisite} settings or inputs; rerun it and downstream stages')
        if any(not Path(path).is_file() or sha256(path) != expected
               for path, expected in saved['outputs'].items()):
            raise ValueError(f'Changed or missing {prerequisite} artifact; rerun prerequisite stages')


def run(config, stages=STAGES, *, force=False, refresh_snapshot=False):
    from .metadata import enrich_stations
    work = Path(config['work_dir'])
    output = Path(config['output_dir'])
    protected = (ROOT / 'data/DEM').resolve()
    if work == output or any(directory == protected or protected in directory.parents
                             for directory in (work, output)):
        raise ValueError('Use a separate processed directory outside original DEM data')
    for directory in [work / 'cache', work / 'tables', output / 'tables', output / 'figures']:
        directory.mkdir(parents=True, exist_ok=True)
    config['correlation_min_pairs'] = config['min_pair_hours']
    for stage in stages:
        LOG.info('Stage %s', stage)
        _validate_prerequisites(stage, config, work)
        cache = work / 'cache' / f'{stage}.json'
        fingerprint = _fingerprint(stage, config, work)
        cached = json.loads(cache.read_text()) if cache.exists() else None
        cache_valid = cached and cached['fingerprint'] == fingerprint
        outputs_valid = cached and all(Path(path).is_file() and sha256(path) == expected
                                       for path, expected in cached.get('outputs', {}).items())
        if cache_valid and outputs_valid and not force and not (stage == 'extract' and refresh_snapshot):
            LOG.info('Reuse %s', stage)
            continue
        if stage != 'extract' and not (work / 'snapshot/hourly.parquet').exists():
            raise ValueError('Run extract first; individual stages never silently query PostgreSQL')
        before = {str(p): file_identity(p) for p in work.rglob('*') if p.is_file() and 'cache' not in p.parts}
        if stage == 'extract':
            from .database import extract
            extract(config)
        else:
            stations = pd.read_parquet(work / 'snapshot/stations.parquet')
            metadata = enrich_stations(stations)
            hourly = pd.read_parquet(work / 'snapshot/hourly.parquet')
            tables = _read_tables(work / 'tables')
            if stage == 'dem':
                from .dem import process
                registry, terrain = process(config, stations)
                _write_tables(work / 'tables', {'dem_registry': registry, 'terrain': terrain,
                                               'station_metadata': metadata})
            elif stage == 'statistics':
                from .statistics import analyze
                from .quality import audit_quality
                raw = pd.read_parquet(work / 'snapshot/raw_observations.parquet')
                _write_tables(work / 'tables', {
                    **analyze(hourly, stations, config),
                    **audit_quality(hourly, raw, stations, config),
                })
            elif stage == 'grouping':
                from .grouping import group_stations
                if 'grouping_features' not in tables or 'terrain' not in tables:
                    raise ValueError('Run statistics and dem before grouping')
                _write_tables(work / 'tables', group_stations(
                    tables['grouping_features'], metadata, tables['terrain'], config))
            elif stage == 'plots':
                from .visualization import plot_results
                plot_results(hourly, metadata, tables, {
                    'FABDEM': str(work / 'dem/hue_fabdem.tif'),
                    'Copernicus': str(work / 'dem/hue_copernicus.tif')}, config)
            elif stage == 'report':
                from .reporting import create_report
                create_report(hourly, metadata, tables, config)
        outputs = {}
        for p in work.rglob('*'):
            if p.is_file() and 'cache' not in p.parts and before.get(str(p)) != file_identity(p):
                outputs[str(p)] = sha256(p)
        if stage == 'plots':
            outputs.update({str(p): sha256(p) for p in (output / 'figures').glob('*.png')})
        if stage == 'report':
            outputs.update({str(p): sha256(p) for p in output.rglob('*') if p.is_file() and p.name != 'pipeline.log'})
        # Boundary may first be downloaded during DEM; fingerprint it after processing.
        write_json(cache, {'fingerprint': _fingerprint(stage, config, work), 'outputs': outputs})
    LOG.info('Requested stages completed; report directory: %s', output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('default.yaml'))
    parser.add_argument('--stages', nargs='+', choices=STAGES, default=STAGES)
    parser.add_argument('--station-ids', nargs='+')
    parser.add_argument('--start')
    parser.add_argument('--end')
    parser.add_argument('--grouping-end')
    parser.add_argument('--fabdem')
    parser.add_argument('--copernicus')
    parser.add_argument('--radii-m', nargs='+', type=float)
    parser.add_argument('--work-dir')
    parser.add_argument('--output-dir')
    parser.add_argument('--force', action='store_true', help='Recompute requested stages on same snapshot')
    parser.add_argument('--refresh-snapshot', action='store_true', help='Explicitly re-query read-only source DB')
    args = parser.parse_args()
    config = load_config(args.config, {key: getattr(args, key) for key in [
        'station_ids', 'start', 'end', 'grouping_end', 'fabdem', 'copernicus', 'radii_m',
        'work_dir', 'output_dir']})
    Path(config['output_dir']).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(
                            Path(config['output_dir']) / 'pipeline.log', encoding='utf-8')])
    try:
        run(config, args.stages, force=args.force, refresh_snapshot=args.refresh_snapshot)
    except Exception:
        LOG.exception('Analysis failed; originals and source observations were not written')
        raise


if __name__ == '__main__':
    main()
