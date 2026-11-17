"""Build a single-station training dataset (regional-dataset format read by src/ai/regional_data/reader.py).

Inputs: the station's water level in PostgreSQL (hydrology.observations, filled by the worker / api_backfill) and,
optionally, Open-Meteo downloads (src/ai/station_dataset/open_meteo.py) for the wind and ECMWF rain groups.
All settings come from a JSON config, e.g. models/thuan_an/dataset_config.json:

    .venv/bin/python -m src.ai.station_dataset.build --config models/thuan_an/dataset_config.json \
        --name ta_water_only_wind_tide30_fc --groups fc_ecmwf wind tide --output-dir train_data/thuan_an

Written files (all times UTC):
  water.parquet      timestamp_utc + one column per station: hourly level (mm) = mean of valid observations in (t-1h, t]
  calendar.parquet   flood_season (1 in local months Sep-Dec), hour/day sin-cos (unused by default)
  rainfall.parquet   the extra input groups (name kept for the reader): wind_*, tide_*, fc_* columns
  weather.parquet    empty (timestamp only)
  samples.parquet    one row per valid issue time with targets, history window and split membership
  normalizers.json   train-only mean/scale for water, targets and every non-flag extra column
  feature_schema.json, station_mapping.json, config.json, manifest.json
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.ai.regional_data.preprocessing import calendar_features, context_rows, make_samples, moments
from src.ai.station_dataset import features as F
from src.ai.station_dataset.open_meteo import load_single_runs, load_single_runs_vars

CALENDAR = ['flood_season', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def hourly_level_from_db(station_id, index, engine=None, despike=None, return_flagged=False):
    """Hourly level on `index` from hydrology.observations: value at t = mean of the station's observations with
    observed_at in (t-1h, t], quality_status 'observed' and finite depth; an hour containing two records with the
    same timestamp is left missing. Units: the station's measurement_unit must be mm.

    despike: optional config dict (see despike_observations); flagged 10-min readings are dropped before the hourly
    mean, so the hour keeps the mean of its remaining readings. The database is not modified."""
    import sqlalchemy as sa
    from src.backend.db.session import build_engine
    own = engine is None
    engine = engine or build_engine()
    pad = pd.Timedelta(hours=1) if despike else pd.Timedelta(0)        # context for the centred median at the edges
    try:
        with engine.connect() as c:
            unit = c.execute(sa.text('select measurement_unit from hydrology.stations where id=:s'), {'s': station_id}).scalar()
            if unit not in (None, 'mm'):
                raise ValueError(f'{station_id}: expected unit mm, got {unit}')
            obs = pd.read_sql(sa.text('select observed_at, depth, quality_status from hydrology.observations '
                                      'where station_id=:s and observed_at > :a and observed_at <= :b'), c,
                              params={'s': station_id, 'a': index.min() - pd.Timedelta(hours=1) - pad, 'b': index.max() + pad})
    finally:
        if own:
            engine.dispose()
    flagged = pd.DataFrame(columns=['observed_at', 'depth', 'rolling_median', 'deviation_mm'])
    if despike:
        obs, flagged = despike_observations(obs, **{k: v for k, v in despike.items() if k != 'note'})
    level = hourly_level(obs, index)
    return (level, flagged) if return_flagged else level


def despike_observations(obs, window_minutes=121, threshold_mm=150, keep=()):
    """Mark isolated sensor spikes (single readings or short plateaus) as quality_status 'spike'.

    A reading is a spike when it deviates more than `threshold_mm` from the median of the station's valid readings
    within +-window_minutes/2 (centred, so a plateau shorter than half the window is still caught). `keep`: list of
    {start_utc, end_utc, note} ranges reviewed as possibly real and left untouched. Returns (obs, flagged rows)."""
    obs = obs.copy()
    obs['observed_at'] = pd.to_datetime(obs.observed_at, utc=True)
    obs['depth'] = pd.to_numeric(obs.depth, errors='coerce').astype(float)
    ok = obs.quality_status.eq('observed') & np.isfinite(obs.depth)
    s = obs.loc[ok].set_index('observed_at').depth.sort_index()
    med = s.rolling(f'{int(window_minutes)}min', center=True, min_periods=5).median()
    dev = s - med
    hit = dev.abs() > threshold_mm
    for k in keep:
        hit &= ~((s.index >= pd.Timestamp(k['start_utc'])) & (s.index <= pd.Timestamp(k['end_utc'])))
    times = s.index[hit]
    obs.loc[ok & obs.observed_at.isin(times), 'quality_status'] = 'spike'
    flagged = pd.DataFrame({'observed_at': times, 'depth': s[hit].to_numpy(), 'rolling_median': med[hit].to_numpy(),
                            'deviation_mm': dev[hit].to_numpy()})
    return obs, flagged


def gauge_rain_from_db(station_ids, index, engine=None):
    """Hourly rain per gauge on `index` (features.hourly_gauge_rain) from hydrology.observations."""
    import sqlalchemy as sa
    from src.backend.db.session import build_engine
    own = engine is None
    engine = engine or build_engine()
    try:
        with engine.connect() as c:
            obs = pd.read_sql(sa.text('select station_id, observed_at, depth, quality_status from hydrology.observations '
                                      'where station_id = any(:s) and observed_at > :a and observed_at <= :b'), c,
                              params={'s': list(station_ids), 'a': index.min() - pd.Timedelta(hours=1), 'b': index.max()})
    finally:
        if own:
            engine.dispose()
    return F.hourly_gauge_rain(obs, index)


def hourly_level(obs, index):
    obs = obs.copy()
    obs['observed_at'] = pd.to_datetime(obs.observed_at, utc=True)
    obs['depth'] = pd.to_numeric(obs.depth, errors='coerce').astype(float)
    obs['hour'] = obs.observed_at.dt.ceil('h')
    dup_hours = set(obs.loc[obs.observed_at.duplicated(keep=False), 'hour'])
    valid = obs[obs.quality_status.eq('observed') & np.isfinite(obs.depth)]
    level = valid.groupby('hour').depth.mean().reindex(index)
    level[level.index.isin(dup_hours)] = np.nan
    return level.astype(float)


def build(cfg, name, groups, output_dir, level=None, normalizers_from=None):
    """Create <output_dir>/<name>. `level` (hourly Series on the config grid) defaults to PostgreSQL.

    normalizers_from: an existing (training) dataset directory whose normalizers.json is copied instead of refitting,
    e.g. to evaluate trained models on a grid extended past the training data (same train split, same scaling).
    """
    sid = cfg['station_id']
    d = Path(output_dir) / name
    if d.exists():
        raise FileExistsError(d)
    index = pd.date_range(pd.Timestamp(cfg['grid_start_utc']), pd.Timestamp(cfg['grid_end_utc']), freq='h')
    flagged = None
    if level is None:
        level, flagged = hourly_level_from_db(sid, index, despike=cfg.get('despike'), return_flagged=True)
    else:
        level = level.reindex(index).astype(float)
    L, horizons = cfg['sequence_length'], cfg['horizons_hours']
    folds = [dict(id='fixed', validation_start=cfg['validation_start'], validation_end=cfg['validation_end'])]
    mapping = {sid: dict(station_index=0, name=cfg.get('station_name', sid), latitude=cfg.get('latitude'),
                         longitude=cfg.get('longitude'), unit='mm')}

    # ---- samples: history coverage >= 80 % over L hours, level(t) and all targets present; fixed split
    samples = make_samples(level.to_frame(sid), mapping, folds, cfg['cutoff'], L,
                           minimum_coverage=cfg['minimum_water_history_fraction'], purge_hours=cfg['purge_hours'])
    for h in horizons:
        samples[f'target_h{h}_mm'] = level.reindex(pd.DatetimeIndex(samples.anchor_utc) + pd.Timedelta(hours=h)).to_numpy()
        samples[f'target_h{h}_utc'] = samples.anchor_utc + pd.Timedelta(hours=h)
        samples[f'delta_h{h}_mm'] = samples[f'target_h{h}_mm'] - samples.water_now_mm
    complete = np.all([np.isfinite(samples[f'target_h{h}_mm']) for h in horizons], axis=0)
    samples = samples[complete].reset_index(drop=True)
    # ---- sensor steps (gauge datum changes, cfg['sensor_steps']): a sample whose current hour or any target hour
    # lies on the other side of the step has a label jump with no hydrological cause -> excluded from every split.
    # Hourly value at x covers (x-1h, x], so affected anchors a satisfy a-1h < step <= a+max(horizons).
    # Raw levels are kept (no offset correction); history windows that contain a step are allowed.
    anchors = pd.DatetimeIndex(samples.anchor_utc)
    excluded = np.zeros(len(samples), dtype=bool)
    for s in cfg.get('sensor_steps', []):
        t = pd.Timestamp(s['time_utc'])
        excluded |= np.asarray((anchors - pd.Timedelta(hours=1) < t) & (t <= anchors + pd.Timedelta(hours=max(horizons))))
    samples.loc[excluded, ['fixed_train', 'fixed_validation', 'refit_train']] = False
    samples.loc[excluded, 'split'] = 'excluded_sensor_step'
    samples['station_index'] = 0
    samples['sample_id'] = np.arange(len(samples))
    order = ['sample_id', 'station_id', 'station_index', 'anchor_utc', 'anchor_row', 'history_start_row', 'history_coverage',
             'water_now_mm'] + [f'{p}{h}{s}' for h in horizons for p, s in (('target_h', '_mm'), ('target_h', '_utc'), ('delta_h', '_mm'))] \
        + ['fixed_train', 'fixed_validation', 'split', 'refit_train']
    samples = samples[order]
    train = samples[samples.fixed_train]
    rows = context_rows(train.anchor_row, L, len(index))         # every hour inside any training window

    # ---- extra input groups (order = group order in the dataset)
    train_rows = np.isin(np.arange(len(index)), train.anchor_row.to_numpy())
    frames = {}
    for g in groups:
        if g == 'wind':
            # onshore_pos: hourly onshore component clipped at 0 (offshore wind -> 0) before the window means
            wind = F.wind_features(cfg['wind']['previous_runs_json'], index, onshore_pos=cfg['wind'].get('onshore_pos', False))
            if cfg['wind'].get('channels'):             # optional subset, e.g. the 4 wind_onshore_* channels
                unknown = set(cfg['wind']['channels']) - set(wind.columns)
                if unknown:
                    raise ValueError(f'unknown wind channels {sorted(unknown)}')
                wind = wind[[c for c in wind.columns if c in cfg['wind']['channels'] or c == 'wind_missing_frac']]
            frames['wind'] = wind
        elif g == 'tide':
            frames['tide'] = F.tide_features(level, train_rows, window_days=cfg['tide']['window_days'])
        elif g == 'fc_ecmwf':
            fc = cfg['fc_ecmwf']
            runs = load_single_runs(fc['single_runs_dir'])
            all_cells = sorted(fc['all_cells'])
            sync = F.synchronized_rain(runs, index, all_cells, fc.get('lag_hours', 6), fc.get('max_age_hours', 24))
            rain = F.forecast_rain_features(sync, index, fc['named_cells'], all_cells, fc.get('transform', 'log1p'))
            if fc.get('channels'):                      # optional subset, e.g. the 4 fc_thuan_an_rain_* channels
                unknown = set(fc['channels']) - set(rain.columns)
                if unknown:
                    raise ValueError(f'unknown fc_ecmwf channels {sorted(unknown)}')
                rain = rain[[c for c in rain.columns if c in fc['channels'] or c == 'fc_missing_frac']]
            frames['fc_ecmwf'] = rain
        elif g == 'rain_obs':
            ro = cfg['rain_obs']
            windows = ro.get('windows_hours', [1, 3, 6, 12, 24, 48])
            clusters = {k: v for k, v in ro['clusters'].items() if not ro.get('use') or k in ro['use']}
            ext = pd.date_range(index.min() - pd.Timedelta(hours=max(windows)), index.max(), freq='h')   # history for the sums
            hourly = gauge_rain_from_db(sorted({s for v in clusters.values() for s in v}), ext)
            frames['rain_obs'] = F.rain_cluster_features(hourly, clusters, windows, ro.get('transform', 'log1p')).reindex(index)
        elif g == 'sea':
            frames['sea'] = F.sea_level_features(cfg['sea']['marine_json'], index, cfg['sea'].get('week_hours', 168),
                                                 cfg['sea'].get('direct', False))
        elif g == 'sea_multi':
            sm = cfg['sea_multi']
            frames['sea_multi'] = F.sea_multi_features(sm['marine_multi_json'], index, sm['mean_points'], sm['north'], sm['south'],
                                                       sm['offshore'], sm['nearshore'], sm.get('week_hours', 168))
        elif g == 'wave':
            wv = cfg['wave']
            frames['wave'] = F.wave_features(wv['marine_multi_json'], index, wv['point'], wv.get('onshore_from_deg', 45))
        elif g == 'soil':
            so = cfg['soil']
            layers = so.get('layers', ['soil_moisture_0_to_7cm'])
            inter = so.get('interaction', True)
            runs_v = load_single_runs_vars(so['single_runs_dir'], layers + (['precipitation'] if inter else []))
            frames['soil'] = F.soil_moisture_features(runs_v, index, sorted(so.get('cells', cfg['fc_ecmwf']['all_cells'])), layers, inter,
                                                      so.get('lag_hours', 6), so.get('max_age_hours', 24))
        elif g == 'lin_fc':
            # cross-fitted linear forecast of the change, built from groups listed BEFORE it in --groups
            lc = cfg.get('lin_fc', {})
            inputs = lc.get('inputs', [k for k in frames])
            missing = [k for k in inputs if k not in frames]
            if missing:
                raise ValueError(f'lin_fc needs groups {missing} earlier in --groups')
            frames['lin_fc'] = F.linear_forecast_features(level, [frames[k] for k in inputs], train.anchor_row.to_numpy(),
                                                          horizons, lc.get('history_hours', 12), lc.get('folds', 5))
        else:
            raise ValueError(f'unknown group {g}')
    extra = pd.concat(frames.values(), axis=1) if frames else pd.DataFrame(index=index)
    value_cols = [c for c in extra.columns if not c.endswith('_missing_frac')]

    # ---- train-only normalisers
    w = moments(level.iloc[rows])
    targets = {f'h{h}': moments(train[f'target_h{h}_mm']) for h in horizons}
    station_norm = dict(water=w, targets=targets,
                        persistence_scale=[w['scale'] / targets[f'h{h}']['scale'] for h in horizons],
                        persistence_offset=[(w['mean'] - targets[f'h{h}']['mean']) / targets[f'h{h}']['scale'] for h in horizons],
                        persistence_horizons=horizons, training_samples=len(train),
                        input_fit_first=str(index[rows.min()]), input_fit_last=str(index[rows.max()]),
                        target_fit_last=str(train[f'target_h{max(horizons)}_utc'].max()))
    shared = {c: moments(extra[c].iloc[rows]) for c in value_cols}
    norm = dict(stations={sid: station_norm}, shared=shared, train_anchor_first=str(train.anchor_utc.min()),
                train_anchor_last=str(train.anchor_utc.max()), fit_policy='train-only: inputs over unique training-window hours; targets over training samples')
    if pd.Timestamp(station_norm['input_fit_last']) >= pd.Timestamp(cfg['validation_start']):
        raise ValueError('normaliser fit reaches the validation period')
    if normalizers_from:
        saved = json.loads((Path(normalizers_from) / 'normalizers.json').read_text())['fixed']
        if set(saved['shared']) != set(shared) or set(saved['stations']) != {sid}:
            raise ValueError('normalizers_from has different input columns or stations')
        norm = saved

    # ---- write
    d.mkdir(parents=True)
    reset = lambda f: f.rename_axis('timestamp_utc').reset_index()
    reset(level.rename(sid).to_frame()).to_parquet(d / 'water.parquet', index=False)
    reset(calendar_features(index)[CALENDAR]).to_parquet(d / 'calendar.parquet', index=False)
    reset(extra).to_parquet(d / 'rainfall.parquet', index=False)
    pd.DataFrame({'timestamp_utc': index}).to_parquet(d / 'weather.parquet', index=False)
    samples.to_parquet(d / 'samples.parquet', index=False)
    if flagged is not None and len(flagged):
        flagged.to_csv(d / 'despiked_readings.csv', index=False)     # 10-min readings dropped by cfg['despike']
    write_json(d / 'station_mapping.json', mapping)
    write_json(d / 'normalizers.json', {'fixed': norm})
    write_json(d / 'config.json', dict(folds=folds, cutoff=cfg['cutoff'], sequence_length=L,
               minimum_water_history_fraction=cfg['minimum_water_history_fraction'], purge_hours=cfg['purge_hours'],
               split_policy='fixed_train_validation_test', expected_station_count=1, horizons_hours=horizons, groups=list(frames)))
    group_cols = {g: list(f.columns) for g, f in frames.items()}
    write_json(d / 'feature_schema.json', dict(
        water_columns=['water_level_depth', 'water_level_missing'], calendar_columns=CALENDAR,
        calendar_groups={'base': ['flood_season'], 'hour': ['hour_sin', 'hour_cos'], 'day_of_year': ['day_sin', 'day_cos']},
        rain_groups=group_cols, rain_aliases={g: g for g in group_cols}, rain_value_columns=value_cols,
        weather_columns=[], weather_value_columns=[], target_columns=[f'target_h{h}_mm' for h in horizons],
        horizons_hours=horizons, sequence_order=f'oldest to newest: t-{L - 1}h through t',
        raw_missing='NaN in parquet; standardized NaN -> 0 by the reader with water_level_missing = 1'))
    write_json(d / 'manifest.json', dict(
        status='complete', format_version=2, dataset=name, created_at=pd.Timestamp.now(tz='UTC').isoformat(),
        sequence_length=L, horizons_hours=horizons, output_size=len(horizons), station_count=1, groups=list(frames),
        input_features=3 + len(extra.columns), sample_count=len(samples), split_counts=samples.split.value_counts().to_dict(),
        hourly_rows=len(index), default_fold='fixed', cutoff=cfg['cutoff'],
        train_command='--dataset-dir <this dir> --target-horizons ' + ' '.join(map(str, horizons)) +
                      (' --rain-stations all --feature-set water_level+rain+flood_season' if frames else
                       ' --feature-set water_level+flood_season') + ' --use-weather-forecast false',
        config=cfg))
    return d, samples


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--groups', nargs='*', default=[], choices=['fc_ecmwf', 'wind', 'tide', 'rain_obs', 'lin_fc', 'soil', 'sea', 'sea_multi', 'wave'])
    p.add_argument('--output-dir', required=True)
    p.add_argument('--grid-end-utc', help='override grid_end_utc of the config (e.g. evaluation on newer data)')
    p.add_argument('--normalizers-from', help='copy normalizers.json from this (training) dataset instead of refitting')
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    if a.grid_end_utc:
        cfg['grid_end_utc'] = a.grid_end_utc
    d, s = build(cfg, a.name, a.groups, a.output_dir, normalizers_from=a.normalizers_from)
    print(d, s.split.value_counts().to_dict())


if __name__ == '__main__':
    main()
