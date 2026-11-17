"""Audit retained dataset contracts and reuse existing DEM/metadata artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.station_analysis.config import ROOT, file_identity, sha256


def dataset_directories(config):
    if config['dataset_dirs']:
        return [Path(p) for p in config['dataset_dirs']]
    return sorted(p.parent for p in (ROOT/'train_data').glob('*/manifest.json'))


def _times(frame):
    if 'timestamp_utc' not in frame:
        raise ValueError('Dataset requires timestamp_utc')
    values = frame.timestamp_utc
    if any(pd.notna(v) and pd.Timestamp(v).tzinfo is None for v in values):
        raise ValueError('Dataset timestamps require explicit timezones')
    stamps = pd.DatetimeIndex(pd.to_datetime(values, utc=True))
    if stamps.hasnans or stamps.has_duplicates or not stamps.is_monotonic_increasing:
        raise ValueError('Dataset timestamps must be finite, unique and ordered')
    if not stamps.equals(stamps.floor('h')):
        raise ValueError('Dataset timestamps must be on hourly boundaries')
    return stamps


def audit_datasets(directories):
    inventories, splits, targets, issues = [], [], [], []
    for directory in directories:
        manifest_path = directory/'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        spec_path = directory/'dataset_spec.json'
        spec = json.loads(spec_path.read_text()) if spec_path.exists() else {}
        feature_columns = spec.get('feature_columns', manifest.get('feature_columns', []))
        target_columns = spec.get('target_columns', manifest.get('target_columns', []))
        hourly = pd.read_csv(directory/'hourly.csv')
        try:
            index = _times(hourly)
        except ValueError as exc:
            issues.append({'dataset': directory.name, 'file': 'hourly.csv',
                           'issue': str(exc), 'severity': 'blocking'})
            continue
        hourly.index = index
        inventories.append({
            'dataset': directory.name, 'path': str(directory), 'hourly_rows': len(hourly),
            'start_utc': index.min(), 'end_utc': index.max(),
            'input_features': len(feature_columns), 'output_targets': len(target_columns),
            'sequence_length': manifest.get('sequence_length', spec.get('sequence_length', 96)),
            'purge_hours': manifest.get('purge_hours', spec.get('purge_hours', np.nan)),
            'manifest_sha256': sha256(manifest_path),
            'weather_provenance_present': (directory/'weather_provenance.csv.gz').exists(),
            'parent_dataset': manifest.get('parent_dataset'),
            'independent_test_claim': False,
            'test_status': 'historical exploratory periods; expanded validation reuses former test where declared',
        })
        parts = {}
        for split in ['train', 'val', 'test']:
            path = directory/f'{split}.csv'
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            try:
                timestamps = _times(frame)
            except ValueError as exc:
                issues.append({'dataset': directory.name, 'file': path.name,
                               'issue': str(exc), 'severity': 'blocking'})
                continue
            parts[split] = timestamps
            splits.append({'dataset': directory.name, 'split': split, 'rows': len(frame),
                           'start_utc': timestamps.min(), 'end_utc': timestamps.max(),
                           'missing_in_hourly': int((~timestamps.isin(index)).sum()),
                           'internal_timestamp_gap_count': int((timestamps.to_series().diff().dropna()!=pd.Timedelta(hours=1)).sum())})
        for left, right in [('train','val'),('val','test')]:
            if left not in parts or right not in parts:
                continue
            overlap = parts[left].intersection(parts[right])
            purge = int(manifest.get('purge_hours', 24))
            distance = (parts[right].min()-parts[left].max())/pd.Timedelta(hours=1)
            if len(overlap) or distance <= purge:
                issues.append({'dataset': directory.name, 'file': f'{left}/{right}',
                               'issue': f'overlap={len(overlap)}; boundary separation={distance}h, required>{purge}h',
                               'severity': 'blocking'})
        for column in [c for c in hourly if c.startswith('target_water_level_plus_')]:
            minutes = int(column.removeprefix('target_water_level_plus_').removesuffix('m'))
            if 'water_level_depth' not in hourly:
                continue
            expected = hourly.water_level_depth.reindex(index+pd.Timedelta(minutes=minutes)).to_numpy()
            actual = pd.to_numeric(hourly[column],errors='coerce').to_numpy(dtype=float)
            finite = np.isfinite(actual) & np.isfinite(expected)
            mismatch = finite & ~np.isclose(actual,expected,rtol=1e-6,atol=1e-3)
            targets.append({'dataset': directory.name,'target_column': column,
                            'horizon_hours': minutes/60,'paired_finite': int(finite.sum()),
                            'value_mismatches': int(mismatch.sum()),
                            'finite_labels_without_own_hourly_reference': int((np.isfinite(actual)&~np.isfinite(expected)).sum()),
                            'validation_scope': 'own frozen hourly values, not current revised DB'})
    return {
        'dataset_inventory': pd.DataFrame(inventories,columns=[
            'dataset','path','hourly_rows','start_utc','end_utc','input_features','output_targets',
            'sequence_length','purge_hours','manifest_sha256','weather_provenance_present',
            'parent_dataset','independent_test_claim','test_status']),
        'dataset_splits': pd.DataFrame(splits,columns=[
            'dataset','split','rows','start_utc','end_utc','missing_in_hourly','internal_timestamp_gap_count']),
        'dataset_target_alignment': pd.DataFrame(targets,columns=[
            'dataset','target_column','horizon_hours','paired_finite','value_mismatches',
            'finite_labels_without_own_hourly_reference','validation_scope']),
        'dataset_integrity_issues': pd.DataFrame(issues, columns=['dataset','file','issue','severity']),
    }


def reuse_phase1(config, stations):
    """Inventory existing outputs; do not rerun DEM processing or prior EDA."""
    phase1 = Path(config['phase1_work_dir'])
    rows, spatial = [], []
    for name in ['station_metadata','terrain','dem_registry','station_statistics',
                 'quality_flags','cluster_assignments']:
        path = phase1/'tables'/f'{name}.parquet'
        rows.append({'artifact': name,'path': str(path),'exists': path.is_file(),
                     'sha256': sha256(path) if path.is_file() else None,
                     'reuse_policy': 'read-only existing Phase1 product; original label convention retained'})
    registry_path = phase1/'tables/dem_registry.parquet'
    terrain_path=phase1/'tables/terrain.parquet'
    extracted_ids=set(pd.read_parquet(terrain_path).station_id) if terrain_path.exists() else set()
    if registry_path.exists():
        import rasterio
        from pyproj import Transformer
        for dem in pd.read_parquet(registry_path).itertuples():
            path = Path(dem.processed_path)
            if not path.exists():
                continue
            with rasterio.open(path) as source:
                convert = Transformer.from_crs('EPSG:4326',source.crs,always_xy=True)
                for station in stations.itertuples():
                    lon,lat = station.longitude,station.latitude
                    valid = pd.notna(lon) and pd.notna(lat) and -180<=lon<=180 and -90<=lat<=90
                    covered=False
                    if valid:
                        x,y=convert.transform(lon,lat)
                        if source.bounds.left<=x<source.bounds.right and source.bounds.bottom<y<=source.bounds.top:
                            value=next(source.sample([(x,y)],masked=True))[0]
                            covered=not np.ma.is_masked(value) and np.isfinite(value)
                    spatial.append({'station_id':station.station_id,'station_type':station.station_type,
                                    'dem_source':dem.dem_source,'valid_registry_coordinate': bool(valid),
                                    'processed_dem_point_covered':bool(covered),
                                    'terrain_neighborhood_extracted_in_phase1':station.station_id in extracted_ids,
                                    'interpretation':'point footprint check only; not a new terrain feature or catchment assignment'})
    return {'phase1_artifacts':pd.DataFrame(rows),'station_dem_coverage':pd.DataFrame(spatial,columns=[
        'station_id','station_type','dem_source','valid_registry_coordinate',
        'processed_dem_point_covered','terrain_neighborhood_extracted_in_phase1','interpretation'])}
