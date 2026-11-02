"""Runtime history/target selection over immutable regional pool artifacts."""
from copy import deepcopy
import json
import numpy as np
import pandas as pd
from .preprocessing import moments


def select_samples(frame, water, length, horizons, config, fold):
    """Retain saved anchors/memberships; filter only invalid requested contexts/labels."""
    frame = frame.copy()
    frame['history_start_row'] = frame.anchor_row-length+1
    valid = frame.history_start_row.ge(0).to_numpy(copy=True)
    minimum = config.get('minimum_water_history_fraction', .8)
    original_length = config['sequence_length']
    for sid, part in frame.groupby('station_id'):
        positions = part.index.to_numpy()
        if length != original_length:
            coverage = water[sid].notna().rolling(length, min_periods=length).mean()
            actual = coverage.iloc[part.anchor_row.to_numpy(dtype=int)].to_numpy()
            valid[positions] &= actual >= minimum
            frame.loc[positions, 'history_coverage'] = actual
        for h in horizons:
            col = f'target_h{h}_mm'
            if col not in frame:
                frame[col] = np.nan
            # Exact timestamp lookup: no interpolation, resampling or row shift.
            values = water[sid].reindex(pd.DatetimeIndex(part.anchor_utc)+pd.Timedelta(hours=h)).to_numpy()
            frame.loc[positions, col] = values
            valid[positions] &= np.isfinite(values)
    purge = max(config.get('purge_hours', 24), max(horizons))
    bounds = next(item for item in config['folds'] if item['id'] == fold)
    start, end = pd.Timestamp(bounds['validation_start']), pd.Timestamp(bounds['validation_end'])
    valid &= (~frame[fold+'_train'] | (frame.anchor_utc+pd.Timedelta(hours=purge) < start)).to_numpy()
    valid &= (~frame[fold+'_validation'] | (frame.anchor_utc+pd.Timedelta(hours=purge) < end)).to_numpy()
    for h in horizons:
        frame[f'target_h{h}_utc'] = frame.anchor_utc+pd.Timedelta(hours=h)
        frame[f'delta_h{h}_mm'] = frame[f'target_h{h}_mm']-frame.water_now_mm
    return frame.loc[valid].reset_index(drop=True), purge


def configure_view(view, sequence_length=None, target_horizons=None, target_normalizers=None):
    length = view.length if sequence_length is None else sequence_length
    horizons = [6, 12] if target_horizons is None else sorted(target_horizons)
    if isinstance(length, bool) or not isinstance(length, int) or length < 1:
        raise ValueError('Sequence length must be a positive integer')
    if (not horizons or len(set(horizons)) != len(horizons)
            or any(isinstance(h, bool) or not isinstance(h, int) or h < 1 for h in horizons)):
        raise ValueError('Target horizons must be distinct positive integer hours')
    original_length = view.length
    view.target_horizons = horizons
    view.length = length
    default = length == original_length and horizons == [6, 12]
    if default:
        return dict(history_policy='saved anchor pool, unchanged', target_policy='saved train-only scalers',
                    sequence_length=length, target_horizons=horizons, dropped_samples=0)
    config = json.loads((view.path/'config.json').read_text())
    before = len(view.samples)
    view.samples, purge = select_samples(view.samples, view.water, length, horizons, config, view.fold)
    if target_normalizers is not None:
        view.normalizers = deepcopy(target_normalizers)
    else:
        stats = deepcopy(view.normalizers)
        new_horizons = set(horizons)-{6, 12}
        train = None
        if new_horizons:
            all_samples = pd.read_parquet(view.path/'samples.parquet')
            train = all_samples.loc[all_samples[view.fold+'_train']].reset_index(drop=True)
            train, _ = select_samples(train, view.water, length, horizons, config, view.fold)
        for sid, station in stats['stations'].items():
            targets = {}
            for h in horizons:
                key = f'h{h}'
                if key in station['targets']:
                    targets[key] = station['targets'][key]
                else:
                    values = train.loc[train.station_id.eq(sid), f'target_h{h}_mm']
                    if values.empty:
                        raise ValueError(f'No training labels for station {sid}, horizon +{h}h')
                    targets[key] = moments(values)
            station['targets'] = targets
            station['persistence_scale'] = [station['water']['scale']/targets[f'h{h}']['scale'] for h in horizons]
            station['persistence_offset'] = [(station['water']['mean']-targets[f'h{h}']['mean'])/targets[f'h{h}']['scale'] for h in horizons]
        view.normalizers = stats
    return dict(sequence_length=length, target_horizons=horizons, dropped_samples=before-len(view.samples),
        effective_purge_hours=purge,
        history_policy='saved anchor pool; require requested history coverage; no new anchors',
        target_policy='exact water.parquet timestamps; retain saved horizon scalers; fit new horizons on selected train only',
        feature_scaler_policy='reuse immutable train-only input scalers')
