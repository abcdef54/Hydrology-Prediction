"""Timestamp-exact samples, chronological memberships and train-only scaling."""
import numpy as np
import pandas as pd


def check_hourly(index, resolution_minutes=60):
    index = pd.DatetimeIndex(index)
    if (index.tz is None or index.has_duplicates or not index.is_monotonic_increasing
            or not index.equals(pd.date_range(index.min(), index.max(), freq=f'{resolution_minutes}min'))):
        raise ValueError('Require a unique, timezone-aware complete time grid')


def calendar_features(index):
    local = index.tz_convert('Asia/Ho_Chi_Minh')
    hour_angle = 2*np.pi*(local.hour.to_numpy()+local.minute.to_numpy()/60)/24
    days_in_year = np.where(local.is_leap_year, 366, 365)
    day_angle = 2*np.pi*(local.dayofyear.to_numpy()-1)/days_in_year
    return pd.DataFrame(dict(
        flood_season=local.month.isin([9, 10, 11, 12]).astype('uint8'),
        hour_sin=np.sin(hour_angle), hour_cos=np.cos(hour_angle),
        day_sin=np.sin(day_angle), day_cos=np.cos(day_angle)), index=index)


def make_samples(level, station_mapping, folds, cutoff, sequence_length=96,
                 minimum_coverage=.8, purge_hours=24, resolution_minutes=60):
    """Window validity is independent of rain selection and weather availability."""
    check_hourly(level.index, resolution_minutes)
    result = []
    for sid, station in station_mapping.items():
        h = level[sid]
        future = {z: h.reindex(h.index + pd.Timedelta(hours=z)).to_numpy() for z in [6, 12]}
        coverage = h.notna().rolling(sequence_length, min_periods=sequence_length).mean()
        valid = (np.isfinite(h) & np.isfinite(future[6]) & np.isfinite(future[12])
                 & coverage.ge(minimum_coverage))
        positions = np.flatnonzero(valid)
        frame = pd.DataFrame(dict(station_id=sid, station_index=station['station_index'],
            anchor_utc=h.index[positions], anchor_row=positions,
            history_start_row=positions-sequence_length+1,
            history_coverage=coverage.iloc[positions].to_numpy(),
            water_now_mm=h.iloc[positions].to_numpy(),
            target_h6_mm=future[6][positions], target_h12_mm=future[12][positions]))
        for z in [6, 12]:
            frame[f'target_h{z}_utc'] = frame.anchor_utc + pd.Timedelta(hours=z)
            frame[f'delta_h{z}_mm'] = frame[f'target_h{z}_mm'] - frame.water_now_mm
        for fold in folds:
            start, end = pd.Timestamp(fold['validation_start']), pd.Timestamp(fold['validation_end'])
            frame[fold['id']+'_train'] = frame.anchor_utc < start-pd.Timedelta(hours=purge_hours)
            frame[fold['id']+'_validation'] = ((frame.anchor_utc >= start)
                & (frame.anchor_utc+pd.Timedelta(hours=purge_hours) < end))
        default = folds[-1]['id']
        frame['split'] = np.select([frame[default+'_train'], frame[default+'_validation'],
            frame.anchor_utc >= pd.Timestamp(cutoff)], ['train', 'validation', 'test_exploratory'], default='purged')
        frame['refit_train'] = frame.anchor_utc < pd.Timestamp(cutoff)-pd.Timedelta(hours=purge_hours)
        result.append(frame)
    samples = pd.concat(result, ignore_index=True).sort_values(['station_index', 'anchor_utc']).reset_index(drop=True)
    samples.insert(0, 'sample_id', np.arange(len(samples)))
    return samples


def context_rows(anchors, length, size):
    """Union of actual input timestamps, without weighting overlapping windows."""
    difference = np.zeros(size+1, dtype=np.int64)
    end = np.asarray(anchors, dtype=int)
    if len(end):
        np.add.at(difference, end-length+1, 1)
        np.add.at(difference, end+1, -1)
    return np.flatnonzero(np.cumsum(difference[:-1]) > 0)


def moments(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    mean = float(values.mean()) if len(values) else 0.
    std = float(values.std()) if len(values) else 0.
    return dict(mean=mean, scale=std if std > 0 else 1., count=len(values),
                constant=bool(len(values) and std == 0), all_missing=not len(values))


def fit_normalizers(samples, level, rain, weather, schema, folds, sequence_length):
    output = {}
    for fold in folds:
        selected = samples[samples[fold['id']+'_train']]
        if selected.empty:
            raise ValueError(f'No training samples in fold {fold["id"]}')
        stations, all_rows = {}, []
        for sid, part in selected.groupby('station_id'):
            rows = context_rows(part.anchor_row, sequence_length, len(level))
            all_rows.extend(rows.tolist())
            water = moments(level[sid].iloc[rows])
            targets = {f'h{z}': moments(part[f'target_h{z}_mm']) for z in [6, 12]}
            stations[sid] = dict(water=water, targets=targets,
                persistence_scale=[water['scale']/targets[f'h{z}']['scale'] for z in [6, 12]],
                persistence_offset=[(water['mean']-targets[f'h{z}']['mean'])/targets[f'h{z}']['scale'] for z in [6, 12]],
                training_samples=len(part), input_fit_first=str(level.index[rows.min()]),
                input_fit_last=str(level.index[rows.max()]),
                target_fit_last=str(part.target_h12_utc.max()))
        rows = np.unique(all_rows)
        anchor_rows = np.unique(selected.anchor_row)
        shared = {}
        for frame, columns, fit_rows in [(rain, schema['rain_value_columns'], rows),
                                         (weather, schema['weather_value_columns'], anchor_rows)]:
            for col in columns:
                shared[col] = moments(frame[col].iloc[fit_rows])
        output[fold['id']] = dict(stations=stations, shared=shared,
            train_anchor_first=str(selected.anchor_utc.min()), train_anchor_last=str(selected.anchor_utc.max()),
            shared_rain_fit_last=str(level.index[rows.max()]),
            shared_weather_fit_last=str(level.index[anchor_rows.max()]),
            fit_policy='unique training-window input timestamps; forecast only at training anchors; targets only training samples')
    return output
