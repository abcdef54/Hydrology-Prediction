"""Causal input groups of the station datasets (every value at issue time t uses only information available at t).

* wind   (15 channels): ECMWF day-ahead 10 m wind / pressure at one offshore point (Open-Meteo previous_day1).
* tide   (5 channels):  astronomical tide from a trailing harmonic fit of the observed level.
* fc_ecmwf (18 channels): ECMWF rain forecast for the next 0-24 h from the latest available run.
* rain_obs (7 per cluster): observed gauge rain, cluster mean, trailing sums over 1..48 h.
Each frame is indexed by the hourly UTC grid of the dataset; `<group>_missing_frac` columns are raw 0..1 flags,
all other columns are standardised later with train-only statistics.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = (3, 6, 12)
ORIGIN = pd.Timestamp('2025-01-01', tz='UTC')            # time origin of the harmonic fits (hours)
TIDE_PERIODS_7 = {'M2': 12.4206012, 'S2': 12.0, 'N2': 12.65834751, 'K1': 23.93447213, 'O1': 25.81934166,
                  'M4': 6.210300601, 'MS4': 6.103339275}
TIDE_PERIODS_10 = {**{k: TIDE_PERIODS_7[k] for k in ('M2', 'S2', 'N2', 'K1', 'O1')}, 'Q1': 26.868350,
                   'M4': 6.210300601, 'MS4': 6.103339275, 'MN4': 6.269173, 'M6': 4.140200}
FC_WINDOWS = ((0, 3, '0_3h'), (3, 6, '3_6h'), (6, 12, '6_12h'), (12, 24, '12_24h'))


# ----------------------------------------------------------------------------------------------- wind
def hourly_wind(previous_runs_json):
    """Onshore/along-shore/speed/pressure from the *_previous_day1 variables of a Previous Runs response.

    Direction is meteorological (where the wind comes FROM). u = -s*sin(d), v = -s*cos(d) (towards east/north);
    onshore = -(u+v)/sqrt(2) is the component blowing from the north-east towards the south-west (perpendicular to
    the Thuận An coast), along = (u-v)/sqrt(2).
    """
    payload = json.loads(Path(previous_runs_json).read_text()) if not isinstance(previous_runs_json, dict) else previous_runs_json
    x = pd.DataFrame(payload['hourly'])
    x['time'] = pd.to_datetime(x.time, utc=True)
    x = x.set_index('time')
    sp, th = x.wind_speed_10m_previous_day1, np.deg2rad(x.wind_direction_10m_previous_day1)
    u, v = -sp * np.sin(th), -sp * np.cos(th)
    return pd.DataFrame({'onshore': -(u + v) / np.sqrt(2), 'along': (u - v) / np.sqrt(2), 'speed': sp,
                         'p': x.pressure_msl_previous_day1})


def wind_features(previous_runs_json, index, onshore_pos=False):
    """15 wind channels at each issue time t of `index`.

    For c in onshore, along, speed: past24 = mean of hours t-23..t, next6/next12 = mean of hours t+1..t+6/12
    (each needs at least half of its hours), chg12 = next12 - past24. wind_p_now = pressure at t,
    wind_p_chg12 = mean(t+1..t+12) - mean(t-11..t). wind_missing_frac = 1 if any channel is missing.
    """
    h = hourly_wind(previous_runs_json)
    if onshore_pos:
        h['onshore'] = h.onshore.clip(lower=0)
    h = h.reindex(pd.date_range(index.min() - pd.Timedelta(hours=48), index.max() + pd.Timedelta(hours=24), freq='h'))
    past = lambda s, k: s.rolling(k, min_periods=k // 2).mean()
    nxt = lambda s, k: s[::-1].rolling(k, min_periods=k // 2).mean()[::-1].shift(-1)
    out = {}
    for c in ('onshore', 'along', 'speed'):
        out[f'wind_{c}_past24'] = past(h[c], 24)
        out[f'wind_{c}_next6'] = nxt(h[c], 6)
        out[f'wind_{c}_next12'] = nxt(h[c], 12)
        out[f'wind_{c}_chg12'] = out[f'wind_{c}_next12'] - out[f'wind_{c}_past24']
    out['wind_p_now'] = h.p
    out['wind_p_chg12'] = nxt(h.p, 12) - past(h.p, 12)
    f = pd.DataFrame(out).reindex(index)
    f['wind_missing_frac'] = f.isna().any(axis=1).astype(float)
    return f


# ----------------------------------------------------------------------------------------------- tide
def _design(periods):
    per = list(periods.values())
    return lambda tt: np.column_stack([fn(2 * np.pi * tt / p) for p in per for fn in (np.cos, np.sin)])


def tide_features(level, train_mask, window_days=30, min_days=7, min_fraction=0.7, horizons=HORIZONS):
    """Astronomical tide known at t: tide_now = T(t), tide_d{h} = T(t+h) - T(t), tide_missing_frac.

    window_days == 7 reproduces the original 7-constituent, fixed 7-day fit; any other value uses 10 constituents
    (adds Q1, MN4, M6) and the trailing min(window_days, available history) days (at least min_days).
    At each hour t: least squares of level on [1, (t'-t), cos/sin(2*pi*t'/P) for each period P] over the window of
    observed hours ending at t (needs >= min_fraction observed), then T = harmonic part only (no constant, no trend).
    Hours without a valid window use one global harmonic fit of the high-passed level (level minus its centred 25 h
    mean) on TRAIN anchor hours only, and get tide_missing_frac = 1.
    """
    long = window_days != 7
    X = _design(TIDE_PERIODS_10 if long else TIDE_PERIODS_7)
    t = (level.index - ORIGIN).total_seconds().to_numpy() / 3600
    y = level.to_numpy(dtype=float)
    hp = level - level.rolling(25, center=True, min_periods=20).mean()
    m = np.asarray(train_mask) & hp.notna().to_numpy()
    gcoef, *_ = np.linalg.lstsq(X(t[m]), hp.to_numpy()[m], rcond=None)
    now = X(t) @ gcoef
    inc = {h: X(t + h) @ gcoef - now for h in horizons}
    fallback = np.ones(len(y))
    W = window_days * 24
    first = min_days * 24 if long else W
    for i in range(first, len(y)):
        lo = max(0, i - W) if long else i - W
        seg = y[lo:i + 1]; ok = np.isfinite(seg)
        need = max(min_fraction * len(seg), min_days * 24 * min_fraction) if long else min_fraction * (W + 1)
        if ok.sum() < need or not np.isfinite(y[i]):
            continue
        tt = t[lo:i + 1]
        A = np.column_stack([np.ones(len(tt)), tt - t[i], X(tt)])
        c, *_ = np.linalg.lstsq(A[ok], seg[ok], rcond=None)
        base = (X(t[i:i + 1]) @ c[2:])[0]
        now[i] = base
        for h in horizons:
            inc[h][i] = (X(np.array([t[i] + h])) @ c[2:])[0] - base
        fallback[i] = 0.
    return pd.DataFrame({'tide_now': now, **{f'tide_d{h}': inc[h] for h in horizons}, 'tide_missing_frac': fallback},
                        index=level.index)


# ----------------------------------------------------------------------------------------------- observed rain
def hourly_gauge_rain(obs, index):
    """Hourly rain (mm) per gauge on `index` from 10-minute records (columns station_id, observed_at, depth,
    quality_status). Value at t = rain in (t-1h, t]: sum of the 6 valid records; 5 records -> sum * 6/5;
    fewer -> missing (missing is never turned into 0). Same rule as the regional rainfall_hourly_standard."""
    obs = obs.copy()
    obs['observed_at'] = pd.to_datetime(obs.observed_at, utc=True)
    obs['depth'] = pd.to_numeric(obs.depth, errors='coerce').astype(float)
    obs = obs[obs.quality_status.eq('observed') & np.isfinite(obs.depth)]
    obs['hour'] = obs.observed_at.dt.ceil('h')
    g = obs.groupby(['station_id', 'hour']).depth.agg(['sum', 'size']).reset_index()
    g['v'] = np.where(g['size'] == 6, g['sum'], np.where(g['size'] == 5, g['sum'] * 6 / 5, np.nan))
    return g.pivot(index='hour', columns='station_id', values='v').reindex(index)


def rain_cluster_features(hourly, clusters, windows=(1, 3, 6, 12, 24, 48), transform='log1p'):
    """For each cluster: rainobs_<name>_sum_<k>h = trailing k-hour sum (hours t-k+1..t) of the cluster mean
    (mean over the gauges reporting that hour; a window with any hour without a reporting gauge is missing),
    log1p(mm) when transform='log1p'; rainobs_<name>_station_missing_frac = share of the cluster's gauges missing at t."""
    f = (lambda v: np.log1p(v.clip(lower=0))) if transform == 'log1p' else (lambda v: v)
    out = {}
    for name, stations in clusters.items():
        block = hourly.reindex(columns=stations)
        mean = block.mean(axis=1)
        for k in windows:
            out[f'rainobs_{name}_sum_{k}h'] = f(mean.rolling(k, min_periods=k).sum())
        out[f'rainobs_{name}_station_missing_frac'] = block.isna().mean(axis=1)
    return pd.DataFrame(out, index=hourly.index)


# ----------------------------------------------------------------------------------------------- modelled sea level
def sea_level_features(marine_json, index, week_hours=168, direct=False):
    """Offshore sea level known at t, from the Marine API *_previous_day1 series (forecast 24 h before each hour), mm.

    sea_setup_chg   = mean(t+1..t+12) - mean(t-23..t)          (coming change of the sub-tidal level: surge, setup)
    Only hours up to t+12 are used: a previous_day1 value for hour H was issued about 24 h before H, so for H <= t+12
    it is at least 12 h old at issue time t (same rule as the wind group).
    sea_anom        = level(t) - mean of the trailing week       (current setup relative to the recent normal)
    sea_next12_anom = mean(t+1..t+12) - mean of the trailing week
    sea_missing_frac = 1 if any channel is missing. Means need at least half of their hours (the week: 5 days).
    direct=True adds the forecast values themselves: sea_now = level(t), sea_p<h> = level(t+h) - level(t) for h = 3, 6, 12.
    """
    payload = json.loads(Path(marine_json).read_text()) if not isinstance(marine_json, dict) else marine_json
    x = pd.DataFrame(payload['hourly']); x['time'] = pd.to_datetime(x.time, utc=True)
    sl = x.set_index('time').sea_level_height_msl_previous_day1.astype(float) * 1000
    sl = sl.reindex(pd.date_range(index.min() - pd.Timedelta(hours=week_hours + 24), index.max() + pd.Timedelta(hours=24), freq='h'))
    past = lambda q, k: q.rolling(k, min_periods=k // 2).mean()
    nxt = lambda q, k: q[::-1].rolling(k, min_periods=k // 2).mean()[::-1].shift(-1)
    week = sl.rolling(week_hours, min_periods=week_hours * 5 // 7).mean()
    f = pd.DataFrame({'sea_setup_chg': nxt(sl, 12) - past(sl, 24), 'sea_anom': sl - week,
                      'sea_next12_anom': nxt(sl, 12) - week}).reindex(index)
    if direct:
        for h in (3, 6, 12):
            f[f'sea_p{h}'] = (sl.shift(-h) - sl).reindex(index)
        f['sea_now'] = sl.reindex(index)
    f['sea_missing_frac'] = f.isna().any(axis=1).astype(float)
    return f


def _marine_series(item, var, grid):
    x = pd.DataFrame(item['hourly']); x['time'] = pd.to_datetime(x.time, utc=True)
    return x.set_index('time')[var].astype(float).reindex(grid)


def sea_multi_features(marine_multi_json, index, mean_points, north, south, offshore, nearshore, week_hours=168):
    """Sea level forecast (previous_day1, mm) at several offshore points; per point the 3 quantities of
    sea_level_features (setup = mean(t+1..t+12) - mean(t-23..t), anom = level(t) - trailing-week mean,
    n12 = mean(t+1..t+12) - trailing-week mean), combined into 9 channels:
    seam_mean_* (average over mean_points), seam_ns_* (north minus south: along-shore slope),
    seam_off_* (offshore minus nearshore: cross-shore slope); seam_missing_frac. Only hours up to t+12 are used."""
    payload = json.loads(Path(marine_multi_json).read_text())
    grid = pd.date_range(index.min() - pd.Timedelta(hours=week_hours + 24), index.max() + pd.Timedelta(hours=24), freq='h')
    past = lambda q, k: q.rolling(k, min_periods=k // 2).mean()
    nxt = lambda q, k: q[::-1].rolling(k, min_periods=k // 2).mean()[::-1].shift(-1)

    def three(name):
        sl = _marine_series(payload[name], 'sea_level_height_msl_previous_day1', grid) * 1000
        wk = sl.rolling(week_hours, min_periods=week_hours * 5 // 7).mean()
        return pd.DataFrame({'setup': nxt(sl, 12) - past(sl, 24), 'anom': sl - wk, 'n12': nxt(sl, 12) - wk})
    p = {n: three(n) for n in set(mean_points) | {north, south, offshore, nearshore}}
    mean = sum(p[n] for n in mean_points) / len(mean_points)
    f = pd.concat([mean.add_prefix('seam_mean_'), (p[north] - p[south]).add_prefix('seam_ns_'),
                   (p[offshore] - p[nearshore]).add_prefix('seam_off_')], axis=1).reindex(index)
    f['seam_missing_frac'] = f.isna().any(axis=1).astype(float)
    return f


def wave_features(marine_multi_json, index, point, onshore_from_deg=45):
    """Wave height forecast (previous_day1, m) at one offshore point: wave_h_next12 = mean(t+1..t+12),
    wave_h_past24 = mean(t-23..t), wave_h_chg = next12 - past24, wave_on_next12 = mean over t+1..t+12 of
    height * cos(direction - onshore_from_deg) (waves coming from the open sea); wave_missing_frac."""
    payload = json.loads(Path(marine_multi_json).read_text())
    grid = pd.date_range(index.min() - pd.Timedelta(hours=48), index.max() + pd.Timedelta(hours=24), freq='h')
    past = lambda q, k: q.rolling(k, min_periods=k // 2).mean()
    nxt = lambda q, k: q[::-1].rolling(k, min_periods=k // 2).mean()[::-1].shift(-1)
    wh = _marine_series(payload[point], 'wave_height_previous_day1', grid)
    wd = np.deg2rad(_marine_series(payload[point], 'wave_direction_previous_day1', grid))
    won = wh * np.cos(wd - np.deg2rad(onshore_from_deg))
    f = pd.DataFrame({'wave_h_next12': nxt(wh, 12), 'wave_h_past24': past(wh, 24), 'wave_h_chg': nxt(wh, 12) - past(wh, 24),
                      'wave_on_next12': nxt(won, 12)}).reindex(index)
    f['wave_missing_frac'] = f.isna().any(axis=1).astype(float)
    return f


# ----------------------------------------------------------------------------------------------- ECMWF soil moisture
def soil_moisture_features(runs, index, cells, layers=('soil_moisture_0_to_7cm',), interaction=True,
                           lag_hours=6, max_age_hours=24):
    """ECMWF soil moisture (m3/m3) known at issue time t, from ONE run per t (same rule as synchronized_rain:
    newest run with t-24h <= init <= t-6h having finite values for every cell).

    runs: {init: DataFrame indexed by valid time (UTC) with columns '<grid_id>|<variable>'} (open_meteo.load_single_runs_vars),
    needing every layer and, with interaction=True, 'precipitation'.
    Channels: soil_<layer>_basin = mean over `cells` of the layer at valid time t;
    soil_x_fcrain = soil_<first layer>_basin * log1p(basin-mean ECMWF rain over t+1..t+24 from the same run)
    (wet soil and coming rain together); soil_missing_frac = 1 when no run qualifies.
    """
    cycles = pd.DatetimeIndex(sorted(runs))
    recs = {}
    for t in index:
        eligible = cycles[(cycles <= t - pd.Timedelta(hours=lag_hours)) & (cycles >= t - pd.Timedelta(hours=max_age_hours))]
        future = [t + pd.Timedelta(hours=h) for h in range(1, 25)]
        for cycle in reversed(eligible):
            df = runs[cycle]
            if t not in df.index:
                continue
            vals = {L: df.loc[t, [f'{g}|{L}' for g in cells]].to_numpy(float) for L in layers}
            if not all(np.isfinite(v).all() for v in vals.values()):
                continue
            r = {f'soil_{L.replace("soil_moisture_", "")}_basin': v.mean() for L, v in vals.items()}
            if interaction:
                rain = df.reindex(future)[[f'{g}|precipitation' for g in cells]].to_numpy(float)
                if not (np.isfinite(rain).all() and (rain >= 0).all()):
                    continue
                r['soil_x_fcrain'] = vals[layers[0]].mean() * np.log1p(rain.sum(axis=0).mean())
            recs[t] = r
            break
    f = pd.DataFrame.from_dict(recs, orient='index').reindex(index)
    f['soil_missing_frac'] = f.isna().any(axis=1).astype(float)
    return f


# ----------------------------------------------------------------------------------------------- linear forecast
def linear_forecast_features(level, blocks, train_anchor_rows, horizons=HORIZONS, history_hours=12, folds=5):
    """Cross-fitted linear forecast of the level change, one channel per horizon (lin_d<h>, mm) + lin_missing_frac.

    Per horizon h, least squares of level(t+h) - level(t) on: the last `history_hours` levels (t-k minus t, and t),
    and every non-flag column of `blocks` (frames on the same hourly index; tide_* columns are used only for the
    matching horizon: tide_d<h>, tide_now). Inputs are standardised with train-anchor statistics, NaN -> 0.
    Hours up to the last train anchor get out-of-fold values: the train period is cut into `folds` contiguous
    blocks and each block is predicted by a fit on the train anchors outside the block widened by
    history_hours + max(horizons) on both sides. All later hours (validation, test) use the fit on all train anchors.
    """
    n = len(level)
    idx = level.index
    wf = level.ffill()
    lags = np.column_stack([wf.shift(k).to_numpy(float) for k in range(history_hours)])
    hist = np.column_stack([lags - lags[:, :1], lags[:, :1]])
    cur = level.to_numpy(float)
    frame = pd.concat(blocks, axis=1)
    train_anchor_rows = np.asarray(sorted(train_anchor_rows))
    out = {}
    for h in horizons:
        cols = [c for c in frame.columns if not c.endswith('_missing_frac')
                and (not c.startswith('tide_') or c in (f'tide_d{h}', 'tide_now'))]
        X = np.column_stack([hist, frame[cols].to_numpy(float)])
        mu = np.nanmean(X[train_anchor_rows], axis=0)
        sd = np.nanstd(X[train_anchor_rows], axis=0)
        X = np.nan_to_num((X - mu) / np.where(sd > 0, sd, 1.0))
        X = np.column_stack([np.ones(n), X])
        y = np.full(n, np.nan)
        y[:n - h] = cur[h:] - cur[:n - h]
        fit_ok = train_anchor_rows[np.isfinite(y[train_anchor_rows])]

        def fit(rows):
            c, *_ = np.linalg.lstsq(X[rows], y[rows], rcond=None)
            return c
        pred = X @ fit(fit_ok)                              # validation / test and later hours
        last = train_anchor_rows.max()
        edges = np.linspace(0, last + 1, folds + 1).astype(int)
        gap = history_hours + max(horizons)
        for lo, hi in zip(edges[:-1], edges[1:]):
            keep = fit_ok[(fit_ok < lo - gap) | (fit_ok >= hi + gap)]
            pred[lo:hi] = X[lo:hi] @ fit(keep)
        pred[~np.isfinite(lags).all(axis=1) | ~np.isfinite(cur)] = np.nan
        out[f'lin_d{h}'] = pred
    f = pd.DataFrame(out, index=idx)
    f['lin_missing_frac'] = f.isna().any(axis=1).astype(float)
    return f


# ----------------------------------------------------------------------------------------------- ECMWF rain
def synchronized_rain(runs, anchors, cells, lag_hours=6, max_age_hours=24):
    """Rain sums of ONE ECMWF run per issue time, identical cycle for all cells (never mixes runs).

    runs: {initialized_at: {grid_id: hourly precipitation Series}} (open_meteo.load_single_runs).
    For each anchor t take the newest run with t-24h <= init <= t-6h (assumed publication delay 6 h) present for
    every cell with finite, non-negative values for t+1..t+24; sum hourly precipitation valid at t+1..t+3, t+4..t+6,
    t+7..t+12, t+13..t+24 per cell. Anchors without such a run are absent (-> missing).
    """
    cycles = pd.DatetimeIndex(sorted(c for c, per in runs.items() if set(cells) <= set(per)))
    recs = []
    for anchor in anchors:
        eligible = cycles[(cycles <= anchor - pd.Timedelta(hours=lag_hours)) & (cycles >= anchor - pd.Timedelta(hours=max_age_hours))]
        valid_times = [anchor + pd.Timedelta(hours=h) for h in range(1, 25)]
        for cycle in reversed(eligible):
            vec, ok = {}, True
            for g in cells:
                hourly = runs[cycle][g].reindex(valid_times).to_numpy()
                if not (np.isfinite(hourly).all() and (hourly >= 0).all()):
                    ok = False; break
                for a, b, w in FC_WINDOWS:
                    # Python floats: sum() is compensated (Python >= 3.12), bit-identical to the training archive
                    vec[f'{g}|rain_{w}_mm'] = sum(hourly[a:b].tolist())
            if ok:
                recs.append({'anchor_utc': anchor, 'initialized_at': cycle, 'age_hours': (anchor - cycle) / pd.Timedelta(hours=1), **vec})
                break
    return pd.DataFrame(recs).set_index('anchor_utc') if recs else pd.DataFrame()


def forecast_rain_features(sync, index, named_cells, all_cells, transform='log1p'):
    """18 channels: fc_<name>_rain_<window> for each named cell, fc_basin_rain_<window> (mean of all cells),
    fc_age_hours (t - run initialisation) and fc_missing_frac. Rain values are log1p(mm) when transform='log1p'."""
    sf = sync.reindex(index)
    f = (lambda v: np.log1p(v.clip(lower=0))) if transform == 'log1p' else (lambda v: v)
    out = {}
    for name, g in named_cells.items():
        for *_, w in FC_WINDOWS:
            out[f'fc_{name}_rain_{w}'] = f(sf[f'{g}|rain_{w}_mm'].astype(float))
    for *_, w in FC_WINDOWS:
        out[f'fc_basin_rain_{w}'] = f(pd.concat([sf[f'{g}|rain_{w}_mm'].astype(float) for g in all_cells], axis=1).mean(axis=1))
    frame = pd.DataFrame(out, index=index)
    frame['fc_age_hours'] = sf.age_hours.astype(float)
    frame['fc_missing_frac'] = frame.drop(columns='fc_age_hours').isna().any(axis=1).astype(float)
    return frame
