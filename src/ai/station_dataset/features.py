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


# ----------------------------------------------------------------------------------------------- neighbour levels
def station_obs_features(others, level, lags_hours=12, surge=(), train_mask=None, surge_lags=(1, 3, 6)):
    """Observed level of neighbour stations, known at t: obs_<name>_minus_self = other(t) - level(t),
    obs_<name>_d<k> = other(t) - other(t-k) for k = 1..lags_hours (mm), obs_<name>_missing_frac = 1 when any of them
    is missing. `others`: {name: hourly Series on level.index} (already corrected for datum changes).
    surge: names whose non-tidal part is added: obs_<name>_surge = other(t) - its own 7-day tide_now(t) (tide_features,
    trailing window, known at t; needs train_mask for the fallback fit), obs_<name>_surge_d<k> = change over k hours,
    obs_<name>_tide_d<h> for h in HORIZONS (its expected astronomical change)."""
    out = {}
    for name, other in others.items():
        other = other.reindex(level.index).astype(float)
        block = {f'obs_{name}_minus_self': other - level}
        for k in range(1, lags_hours + 1):
            block[f'obs_{name}_d{k}'] = other - other.shift(k)
        if name in surge:
            if train_mask is None:
                raise ValueError('surge features need train_mask')
            td = tide_features(other, train_mask, window_days=7)
            sg = other - td.tide_now
            block[f'obs_{name}_surge'] = sg
            for k in surge_lags:
                block[f'obs_{name}_surge_d{k}'] = sg - sg.shift(k)
            for h in HORIZONS:
                block[f'obs_{name}_tide_d{h}'] = td[f'tide_d{h}']
        block = pd.DataFrame(block, index=level.index)
        block[f'obs_{name}_missing_frac'] = block.isna().any(axis=1).astype(float)
        out.update(block)
    return pd.DataFrame(out, index=level.index)


def flow_features(level, other, train_mask, lags=(1, 3, 6, 12), lowpass_hours=25, lowpass_changes=(6, 12, 24)):
    """River-flow proxies from the station and a downstream neighbour (both observed, known at t), 18 channels + flag.

    flow_dvres = level - own 7-day tide_now (non-tidal level); flow_grad = flow_dvres - (other - other's 7-day tide_now),
    i.e. the surface slope between the two gauges with the astronomical tide of each removed; flow_grad_d<k> and
    flow_dvres_d<k> = change over k hours. Low-pass (tidal-day mean over the trailing `lowpass_hours`, >= 80 % present):
    flow_lp_dv = mean(level) - level(t), flow_lp_grad = mean(level - other), and their changes flow_lp_*_d<k>.
    flow_missing_frac = 1 when any channel is missing. `other` must already be on the station's old datum (offsets)."""
    other = other.reindex(level.index).astype(float)
    res = level - tide_features(level, train_mask, window_days=7).tide_now
    grad = res - (other - tide_features(other, train_mask, window_days=7).tide_now)
    b = {'flow_dvres': res, 'flow_grad': grad}
    for k in lags:
        b[f'flow_grad_d{k}'] = grad - grad.shift(k)
        b[f'flow_dvres_d{k}'] = res - res.shift(k)
    mp = int(np.ceil(0.8 * lowpass_hours))
    lp = level.rolling(lowpass_hours, min_periods=mp).mean()
    lpg = (level - other).rolling(lowpass_hours, min_periods=mp).mean()
    b['flow_lp_dv'] = lp - level
    b['flow_lp_grad'] = lpg
    for k in lowpass_changes:
        b[f'flow_lp_dv_d{k}'] = lp - lp.shift(k)
        b[f'flow_lp_grad_d{k}'] = lpg - lpg.shift(k)
    f = pd.DataFrame(b, index=level.index)
    f['flow_missing_frac'] = f.isna().any(axis=1).astype(float)
    return f


def tide_phase_features(level, max_lead=14, min_fraction=0.7):
    """Shape of the predicted astronomical tide over the next hours (same fixed 7-day, 7-constituent trailing fit as
    tide_features(window_days=7), evaluated at t+0..t+max_lead), 9 channels + flag. With T(k) = tide at t+k:
    tph_hw_hours / tph_lw_hours = lead (1..13 h, 13 if none) of the next high / low water, tph_hw_rise / tph_lw_rise =
    T(that lead) - T(0), tph_range12 = max - min of T(0..12), tph_slope_now = T(1) - T(0), tph_slope6 = T(7) - T(5),
    tph_slope12 = T(13) - T(11), tph_curv6 = T(7) - 2 T(6) + T(5). Hours without a valid window are NaN with
    tph_missing_frac = 1 (no global fallback). Names avoid the tide_ prefix (linear_forecast_features filters it)."""
    if max_lead < 14:
        raise ValueError('max_lead must be >= 14')
    X = _design(TIDE_PERIODS_7)
    t = (level.index - ORIGIN).total_seconds().to_numpy() / 3600
    y = level.to_numpy(dtype=float)
    K = np.arange(max_lead + 1)
    curve = np.full((len(y), len(K)), np.nan)
    W = 7 * 24
    for i in range(W, len(y)):
        seg = y[i - W:i + 1]; ok = np.isfinite(seg)
        if ok.sum() < min_fraction * (W + 1) or not np.isfinite(y[i]):
            continue
        tt = t[i - W:i + 1]
        A = np.column_stack([np.ones(len(tt)), tt - t[i], X(tt)])
        c, *_ = np.linalg.lstsq(A[ok], seg[ok], rcond=None)
        curve[i] = X(t[i] + K) @ c[2:]
    rel = curve - curve[:, :1]
    rows = np.arange(len(y))

    def next_extreme(sign):
        d = np.diff(curve[:, :15], axis=1) * sign
        ext = (d[:, :-1] > 0) & (d[:, 1:] <= 0)                     # extreme at lead 1..13
        k = np.where(ext.any(axis=1), ext.argmax(axis=1) + 1, 13)
        return k.astype(float), rel[rows, k]
    f = pd.DataFrame(index=level.index)
    f['tph_hw_hours'], f['tph_hw_rise'] = next_extreme(1)
    f['tph_lw_hours'], f['tph_lw_rise'] = next_extreme(-1)
    f['tph_range12'] = np.max(curve[:, :13], axis=1) - np.min(curve[:, :13], axis=1)
    f['tph_slope_now'] = curve[:, 1] - curve[:, 0]
    f['tph_slope6'] = curve[:, 7] - curve[:, 5]
    f['tph_slope12'] = curve[:, 13] - curve[:, 11]
    f['tph_curv6'] = curve[:, 7] - 2 * curve[:, 6] + curve[:, 5]
    bad = ~np.isfinite(curve[:, 0])
    f.loc[bad, :] = np.nan
    f['tph_missing_frac'] = bad.astype(float)
    return f


def self_forecast_features(level, forecasts, issue_lag_hours=1):
    """The station model's own earlier forecasts as inputs (`self_fc` group), known at t.

    forecasts: DataFrame indexed by issue time (UTC hours) with columns p<h> = forecast level (mm) for issue + h.
    With L = issue_lag_hours, channel selffc_t<k> at hour t = forecast issued at t-L for hour t+k, minus the observed
    level at t, for every k with k + L in the exported horizons (k = 0..H-L). selffc_t0 (L = 1) is minus the known
    error of the +1 h forecast valid at t. selffc_missing_frac = 1 when any channel is missing.
    Caution: on train hours these forecasts are in-sample for the source model (better than in real use)."""
    hs = sorted(int(c[1:]) for c in forecasts.columns if c.startswith('p'))
    f = forecasts.reindex(level.index)
    out = {}
    for h in hs:
        k = h - issue_lag_hours
        if k >= 0:
            out[f'selffc_t{k}'] = f[f'p{h}'].shift(issue_lag_hours) - level
    frame = pd.DataFrame(out, index=level.index)
    frame['selffc_missing_frac'] = frame.isna().any(axis=1).astype(float)
    return frame


def spring_neap_features(index):
    """Astronomical spring-neap and declination cycles (known for any time, past or future), 4 channels + flag.

    D = mean elongation of the Moon from the Sun, s = mean longitude of the Moon (Meeus, degrees, T = Julian centuries
    from J2000.0; UTC is used for TT, an error of ~1 min). Spring tides follow new and full moon (D = 0, 180 deg):
    sn_cos = cos 2D (+1 near springs, -1 near neaps), sn_sin = sin 2D (sign = approaching / leaving springs), period
    14.77 d (the M2-S2 beat). sn_decl_cos / sn_decl_sin = cos / sin 2s, period 13.66 d (the K1-O1 beat; the diurnal
    inequality is largest at maximum lunar declination). The station's age of the tide (springs ~1-2 days after the
    moon phase) is a phase shift that a model can learn as a combination of the sin and cos channels.
    sn_missing_frac is always 0."""
    t = (index - pd.Timestamp('2000-01-01 12:00', tz='UTC')).total_seconds().to_numpy() / 86400 / 36525
    d = np.deg2rad((297.8501921 + 445267.1114034 * t) % 360)
    s = np.deg2rad((218.3164477 + 481267.88123421 * t) % 360)
    return pd.DataFrame({'sn_cos': np.cos(2 * d), 'sn_sin': np.sin(2 * d), 'sn_decl_cos': np.cos(2 * s),
                         'sn_decl_sin': np.sin(2 * s), 'sn_missing_frac': 0.}, index=index)


def rain_api_features(hourly, clusters, index, taus=(12, 24, 72), per_cluster=False, fc_next=None, transform='log1p'):
    """Antecedent precipitation index of observed gauge rain: API_tau(t) = sum_i r(t-i) exp(-i/tau) (recursive form,
    exponentially weighted mean x tau), r = hourly rain of the basin = mean of the cluster means (cluster mean = mean of
    reporting gauges; an hour with no reporting gauge in a cluster counts as 0 for that cluster).
    rain_api_basin_<tau>; per_cluster adds rain_api_<cluster>_<tau>; fc_next (mm Series on `index`, e.g. ECMWF rain of
    the next 6 h, NaN -> 0) adds rain_api_basin_<tau>_plusfc = API + fc_next. log1p when transform='log1p'.
    rain_api_missing_frac = share of the last 24 h with no reporting gauge in any cluster.
    `hourly` should start well before index[0] (>= 5 x max(taus)) so the recursion has warmed up."""
    f_ = (lambda v: np.log1p(v.clip(lower=0))) if transform == 'log1p' else (lambda v: v)
    api = lambda r, tau: r.ewm(alpha=1 - np.exp(-1 / tau), adjust=False).mean() * tau
    means = {name: hourly.reindex(columns=stations).mean(axis=1) for name, stations in clusters.items()}
    out = {}
    if per_cluster:
        for name, m in means.items():
            for tau in taus:
                out[f'rain_api_{name}_{tau}'] = f_(api(m.fillna(0), tau).reindex(index))
    basin = sum(m.fillna(0) for m in means.values()) / len(means)
    for tau in taus:
        a = api(basin, tau).reindex(index)
        out[f'rain_api_basin_{tau}'] = f_(a)
        if fc_next is not None:
            out[f'rain_api_basin_{tau}_plusfc'] = f_(a + fc_next.reindex(index).fillna(0))
    none = pd.concat(means.values(), axis=1).isna().all(axis=1).astype(float)
    out['rain_api_missing_frac'] = none.rolling(24, min_periods=1).mean().reindex(index)
    return pd.DataFrame(out, index=index)


def reservoir_features(ops, index, facilities, columns=('qdi',), change_hours=(6, 12)):
    """Reservoir operations known at t (hydrology.facility_operations). A record counts at hour t only once published:
    available_at = max(observed_at, source_updated_at) <= t; among published records the one with the latest
    observed_at is used. Channels per facility name: res_<name>_<col> = log1p(value), res_<name>_<col>_d<k> = that
    minus the value known at t-k, res_<name>_age_h = hours since that record's observed_at, res_<name>_missing_frac.
    ops: DataFrame with facility_id, observed_at, source_updated_at and `columns`; facilities: {name: facility_id}."""
    out = {}
    t64 = index.tz_convert('UTC').tz_localize(None).to_numpy() if index.tz is not None else index.to_numpy()
    for name, fid in facilities.items():
        g = ops[ops.facility_id == fid].copy()
        obs_t = pd.to_datetime(g.observed_at, utc=True)
        avail = pd.concat([obs_t, pd.to_datetime(g.source_updated_at, utc=True)], axis=1).max(axis=1)
        g = g.assign(_obs=obs_t.dt.tz_localize(None).to_numpy(), _av=avail.dt.tz_localize(None).to_numpy()).sort_values(['_av', '_obs'])
        av = g._av.to_numpy()
        ob = g._obs.to_numpy()
        best = np.zeros(len(g), dtype=int)                  # index of the latest observed record among the first i+1
        for i in range(1, len(g)):
            best[i] = i if ob[i] >= ob[best[i - 1]] else best[i - 1]

        def pick(times):
            k = np.searchsorted(av, times, side='right') - 1
            j = np.where(k >= 0, best[np.clip(k, 0, None)], -1)
            return j
        j_now = pick(t64)
        block = {}
        for col in columns:
            v = np.log1p(np.clip(g[col].to_numpy(float), 0, None))
            now = np.where(j_now >= 0, v[np.clip(j_now, 0, None)], np.nan)
            block[f'res_{name}_{col}'] = now
            for k in change_hours:
                j_k = pick(t64 - np.timedelta64(k, 'h'))
                block[f'res_{name}_{col}_d{k}'] = now - np.where(j_k >= 0, v[np.clip(j_k, 0, None)], np.nan)
        block[f'res_{name}_age_h'] = np.where(j_now >= 0, (t64 - ob[np.clip(j_now, 0, None)]) / np.timedelta64(1, 'h'), np.nan)
        block = pd.DataFrame(block, index=index)
        block[f'res_{name}_missing_frac'] = block.isna().any(axis=1).astype(float)
        out.update(block)
    return pd.DataFrame(out, index=index)


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
    folds=0: no cross-fitting; every hour (train included) uses the fit on all train anchors, so the channel is the
    same model on train as at validation/test (used as a GRU baseline: an out-of-fold train baseline that is much
    worse than the deployed one on a rare event teaches the GRU a correction the deployed baseline does not need).
    """
    if not isinstance(folds, int) or isinstance(folds, bool) or folds < 0 or folds == 1:
        raise ValueError('folds must be 0 (in-sample, no cross-fitting) or an integer >= 2')
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
        if folds:
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
