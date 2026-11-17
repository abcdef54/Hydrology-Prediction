"""Validate Phase 2 configuration without connecting or mutating source data."""
from pathlib import Path

import pandas as pd
import yaml

from research.station_analysis.config import ROOT


def load_config(path, overrides=None):
    result = yaml.safe_load(Path(path).read_text())
    if not isinstance(result, dict):
        raise ValueError('Configuration must be a mapping')
    result.update({key: value for key, value in (overrides or {}).items() if value is not None})
    for name in ('master_plan', 'work_dir', 'output_dir', 'phase1_work_dir'):
        result[name] = str((ROOT / result[name]).resolve())
    protected = [(ROOT / 'data/DEM').resolve(), (ROOT / 'train_data').resolve(),
                 Path(result['phase1_work_dir']).resolve()]
    for key in ('work_dir', 'output_dir'):
        folder = Path(result[key])
        for source in protected:
            # The default Phase2 work folder is a separate child of Phase1 work.
            if source == Path(result['phase1_work_dir']) and folder == source / 'phase2':
                continue
            if folder == source or source in folder.parents:
                raise ValueError(f'{key} overlaps a protected source directory')
        if folder == ROOT:
            raise ValueError('Use a separate analysis directory')
    if result['work_dir'] == result['output_dir']:
        raise ValueError('work_dir and output_dir must be distinct')
    for name in ('start', 'end'):
        if result.get(name):
            stamp = pd.Timestamp(result[name])
            if stamp.tzinfo is None:
                raise ValueError(f'{name} must include a timezone')
            result[name] = stamp.tz_convert('UTC').isoformat()
    if result.get('end') and pd.Timestamp(result['start']) >= pd.Timestamp(result['end']):
        raise ValueError('start must precede exclusive end')
    for name in ('water_station_ids', 'rain_station_ids'):
        if not isinstance(result[name], list) or any(not isinstance(s, str) for s in result[name]):
            raise ValueError('Station IDs must be strings, preserving leading zeros')
    for name in ('expected_interval_minutes', 'max_forecast_age_hours', 'rain_outlier_threshold_mm'):
        if result[name] <= 0:
            raise ValueError(f'{name} must be positive')
    if 60 % result['expected_interval_minutes']:
        raise ValueError('Expected sampling interval must divide one hour')
    if result['assumed_release_lag_hours'] < 0:
        raise ValueError('Assumed release lag cannot be negative')
    windows = result['forecast_windows']
    if windows != [[0, 3], [3, 6], [6, 12], [12, 24]]:
        raise ValueError('Inventory currently audits the four existing non-overlapping forecast windows')
    cutoff = pd.Timestamp(result['future_analysis']['grouping_end'])
    if cutoff.tzinfo is None:
        raise ValueError('future_analysis.grouping_end must include a timezone')
    weather = result.get('weather_analysis')
    if weather is not None:
        if weather['wet_threshold_mm'] <= 0 or weather['heavy_mean_rate_mm_h'] <= 0:
            raise ValueError('Weather detection thresholds must be positive')
        if isinstance(weather['nearby_gauges'], bool) or int(weather['nearby_gauges']) != weather['nearby_gauges'] or weather['nearby_gauges'] < 0:
            raise ValueError('nearby_gauges must be a non-negative integer')
        if not weather['reference_gauge_ids'] or any(not isinstance(s, str) for s in weather['reference_gauge_ids']):
            raise ValueError('Weather reference gauge IDs must be nonempty strings')
    spatial=result.get('spatial_weather')
    if spatial is not None:
        version=spatial['collection_version']
        if not isinstance(version,str) or Path(version).name!=version or version in ('','.','..'):
            raise ValueError('collection_version must be a single directory name')
        if spatial['model']!='ecmwf_ifs' or spatial['cell_selection']!='land':
            raise ValueError('Stage 3B baseline must keep ecmwf_ifs / land')
        if any(not isinstance(s,str) for s in spatial['requested_station_ids']):
            raise ValueError('Requested spatial station IDs must be strings')
    result['dataset_dirs'] = [str((ROOT / p).resolve()) for p in result['dataset_dirs']]
    return result
