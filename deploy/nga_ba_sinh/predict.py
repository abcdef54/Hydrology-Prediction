"""Water-level forecast (+3 / +6 / +12 h) for one station from its deployment bundle. Self-contained.

Requirements: python >= 3.10, numpy, pandas, torch (CPU is enough), joblib; sqlalchemy + a PostgreSQL driver only
when the level is read from the database.

Examples (run inside the station folder, e.g. deploy/nga_ba_sinh/):
    # latest hour, level from PostgreSQL (env DATABASE_URL), Open-Meteo files from ./data/open_meteo
    python predict.py
    # one issue time
    python predict.py --time 2026-11-10T03:00Z
    # every hour of a period, saved to CSV (one row per issue time and horizon)
    python predict.py --start 2026-09-01T00:00+07:00 --end 2026-10-31T23:00+07:00 --out forecasts.csv
    # level from a file of 10-minute readings instead of the database (columns observed_at, depth[, quality_status])
    python predict.py --time 2026-11-10T03:00Z --level-file readings.csv

From Python:
    from predict import Forecaster
    f = Forecaster('model/nga_ba_sinh_combined_v1.joblib', data_dir='data/open_meteo')
    table = f.forecast(pd.date_range('2026-11-10 00:00', '2026-11-10 06:00', freq='h', tz='UTC'))

The bundle (joblib) holds plain dicts / numpy arrays only: weights, input column order, train-only normalizers,
feature settings and the rule that combines the member models. Every input is built here with the same formulas as in
training (functions below are copies of src/ai/station_dataset/features.py and build.py of the research repository).
"""
import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ORIGIN = pd.Timestamp('2025-01-01', tz='UTC')            # time origin of the harmonic fits (hours)
TIDE_PERIODS_7 = {'M2': 12.4206012, 'S2': 12.0, 'N2': 12.65834751, 'K1': 23.93447213, 'O1': 25.81934166,
                  'M4': 6.210300601, 'MS4': 6.103339275}
TIDE_PERIODS_10 = {**{k: TIDE_PERIODS_7[k] for k in ('M2', 'S2', 'N2', 'K1', 'O1')}, 'Q1': 26.868350,
                   'M4': 6.210300601, 'MS4': 6.103339275, 'MN4': 6.269173, 'M6': 4.140200}
FC_WINDOWS = ((0, 3, '0_3h'), (3, 6, '3_6h'), (6, 12, '6_12h'), (12, 24, '12_24h'))
HISTORY_DAYS = 32          # 30-day tide window + 25-h input window + margin


# ===================================================================================== model
class _Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.recurrent = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)


class GRUForecaster(nn.Module):
    """Inference copy of ResidualGRU (head_mode station): prediction = persistence + learned correction, both in
    normalized target units. persistence_h = level(t) * scale_h + offset_h maps the standardized current level to the
    normalized target of horizon h (i.e. 'level stays as it is')."""

    def __init__(self, input_size, hidden_size, num_layers, output_size, num_stations, embedding_size):
        super().__init__()
        self.encoder = _Encoder(input_size, hidden_size, num_layers)
        self.station_embedding = nn.Embedding(num_stations, embedding_size)
        self.fusion = nn.Sequential(nn.Linear(hidden_size + embedding_size, hidden_size), nn.ReLU())
        self.linear = nn.Module()
        self.linear.weight = nn.Parameter(torch.zeros(num_stations, output_size, hidden_size))
        self.linear.bias = nn.Parameter(torch.zeros(num_stations, output_size))
        self.register_buffer('persistence_scale', torch.ones(num_stations, output_size))
        self.register_buffer('persistence_offset', torch.zeros(num_stations, output_size))

    def forward(self, x, station):                       # x [batch, time, features], station [batch] int64
        base = x[:, -1, 0, None] * self.persistence_scale[station] + self.persistence_offset[station]
        h = self.encoder.recurrent(x)[0][:, -1]
        z = self.fusion(torch.cat([h, self.station_embedding(station)], dim=-1))
        correction = torch.bmm(self.linear.weight[station], z.unsqueeze(-1)).squeeze(-1) + self.linear.bias[station]
        return base + correction


def build_model(member):
    a = member['architecture']
    m = GRUForecaster(a['input_size'], a['hidden_size'], a['num_layers'], a['output_size'], a['num_stations'],
                      a['station_embedding_size'])
    m.load_state_dict({k: torch.from_numpy(np.asarray(v)) for k, v in member['state_dict'].items()}, strict=True)
    return m.eval()


# ===================================================================================== water level
def despike(obs, window_minutes=121, threshold_mm=150, keep=(), **_):
    """Drop isolated spikes: readings deviating > threshold from the centred rolling median of valid readings."""
    ok = obs.quality_status.eq('observed') & np.isfinite(obs.depth)
    s = obs.loc[ok].set_index('observed_at').depth.sort_index()
    med = s.rolling(f'{int(window_minutes)}min', center=True, min_periods=5).median()
    hit = (s - med).abs() > threshold_mm
    for k in keep:
        hit &= ~((s.index >= pd.Timestamp(k['start_utc'])) & (s.index <= pd.Timestamp(k['end_utc'])))
    obs = obs.copy()
    obs.loc[ok & obs.observed_at.isin(s.index[hit]), 'quality_status'] = 'spike'
    return obs


def hourly_level(obs, index, cfg):
    """Level at t = mean of the valid readings in (t-1h, t]; an hour with two readings at the same instant is missing."""
    obs = obs.copy()
    obs['observed_at'] = pd.to_datetime(obs.observed_at, utc=True)
    obs['depth'] = pd.to_numeric(obs.depth, errors='coerce').astype(float)
    if 'quality_status' not in obs:
        obs['quality_status'] = 'observed'
    for r in cfg.get('invalid_ranges') or []:
        inside = (obs.observed_at >= pd.Timestamp(r['start_utc'])) & (obs.observed_at <= pd.Timestamp(r['end_utc']))
        obs.loc[inside, 'quality_status'] = 'invalid'
    if cfg.get('despike'):
        obs = despike(obs, **cfg['despike'])
    obs['hour'] = obs.observed_at.dt.ceil('h')
    dup = set(obs.loc[obs.observed_at.duplicated(keep=False), 'hour'])
    valid = obs[obs.quality_status.eq('observed') & np.isfinite(obs.depth)]
    level = valid.groupby('hour').depth.mean().reindex(index)
    level[level.index.isin(dup)] = np.nan
    return level.astype(float)


def readings_from_db(station_id, start, end, database_url=None):
    import sqlalchemy as sa
    engine = sa.create_engine(database_url or os.environ['DATABASE_URL'])
    try:
        with engine.connect() as c:
            return pd.read_sql(sa.text('select observed_at, depth, quality_status from hydrology.observations '
                                       'where station_id=:s and observed_at > :a and observed_at <= :b'), c,
                               params={'s': station_id, 'a': start, 'b': end})
    finally:
        engine.dispose()


# ===================================================================================== input groups
def _design(periods):
    per = list(periods.values())
    return lambda tt: np.column_stack([fn(2 * np.pi * tt / p) for p in per for fn in (np.cos, np.sin)])


def tide_features(level, fit_mask, window_days=30, min_days=7, min_fraction=0.7, horizons=(3, 6, 12), rows=None):
    """Astronomical tide from a trailing harmonic fit of the station's own level (7 days: 7 constituents, fixed
    window; otherwise 10 constituents over min(window_days, available) days). tide_now = T(t), tide_d<h> = T(t+h)-T(t).
    Hours without a valid window use a global fit of the high-passed level (training: train hours; here: all hours
    of the grid) and get tide_missing_frac = 1. rows: only these positions are fitted (others keep the fallback)."""
    long = window_days != 7
    X = _design(TIDE_PERIODS_10 if long else TIDE_PERIODS_7)
    t = (level.index - ORIGIN).total_seconds().to_numpy() / 3600
    y = level.to_numpy(dtype=float)
    hp = level - level.rolling(25, center=True, min_periods=20).mean()
    m = np.asarray(fit_mask) & hp.notna().to_numpy()
    gcoef, *_ = np.linalg.lstsq(X(t[m]), hp.to_numpy()[m], rcond=None)
    now = X(t) @ gcoef
    inc = {h: X(t + h) @ gcoef - now for h in horizons}
    fallback = np.ones(len(y))
    W = window_days * 24
    first = min_days * 24 if long else W
    todo = range(first, len(y)) if rows is None else sorted(i for i in set(rows) if i >= first)
    for i in todo:
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


def wind_features(previous_runs_json, index, onshore_pos=False):
    """Offshore wind/pressure from the *_previous_day1 variables (forecast ~24 h before each hour). For c in
    onshore (from NE towards SW), along, speed: past24 = mean t-23..t, next6/next12 = mean t+1..t+6/12 (>= half
    present), chg12 = next12 - past24; wind_p_now, wind_p_chg12 = mean(t+1..t+12) - mean(t-11..t)."""
    x = pd.DataFrame(json.loads(Path(previous_runs_json).read_text())['hourly'])
    x['time'] = pd.to_datetime(x.time, utc=True)
    x = x.set_index('time')
    sp, th = x.wind_speed_10m_previous_day1, np.deg2rad(x.wind_direction_10m_previous_day1)
    u, v = -sp * np.sin(th), -sp * np.cos(th)
    h = pd.DataFrame({'onshore': -(u + v) / np.sqrt(2), 'along': (u - v) / np.sqrt(2), 'speed': sp,
                      'p': x.pressure_msl_previous_day1})
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


def grid_id(lat, lon):
    return f'ecmwf_ifs:{lat:.6f}:{lon:.6f}'


def load_single_runs(run_dir, first=None, last=None):
    """{initialized_at: {grid_id: hourly precipitation}} from one JSON per ECMWF run (file name = init time),
    reading only the runs initialised in [first, last] when given. Error markers (run not available) are skipped."""
    runs = {}
    for f in sorted(Path(run_dir).glob('*.json')):
        cycle = pd.Timestamp(f.stem, tz='UTC')
        if (first is not None and cycle < first) or (last is not None and cycle > last):
            continue
        payload = json.loads(f.read_text())
        if isinstance(payload, dict):
            continue
        per = {}
        for item in payload:
            t = pd.to_datetime(item['hourly']['time'], unit='s', utc=True)
            per[grid_id(item['latitude'], item['longitude'])] = pd.Series(item['hourly']['precipitation'], index=t, dtype=float)
        runs[cycle] = per
    return runs


def ecmwf_rain_features(runs, index, cfg, rows=None):
    """One ECMWF run per issue time t (never mixed): the newest run with t-24h <= init <= t-6h that has finite,
    non-negative precipitation at every cell for t+1..t+24. Channels: rain sums 0-3/3-6/6-12/12-24 h ahead (log1p)
    for the named cells and the mean of all cells, fc_age_hours (t - init), fc_missing_frac."""
    cells = sorted(cfg['all_cells'])
    lag, age = cfg.get('lag_hours', 6), cfg.get('max_age_hours', 24)
    cycles = pd.DatetimeIndex(sorted(c for c, per in runs.items() if set(cells) <= set(per)))
    recs = []
    anchors = index if rows is None else index[sorted(set(rows))]
    for anchor in anchors:
        eligible = cycles[(cycles <= anchor - pd.Timedelta(hours=lag)) & (cycles >= anchor - pd.Timedelta(hours=age))]
        valid_times = [anchor + pd.Timedelta(hours=h) for h in range(1, 25)]
        for cycle in reversed(eligible):
            vec, ok = {}, True
            for g in cells:
                hourly = runs[cycle][g].reindex(valid_times).to_numpy()
                if not (np.isfinite(hourly).all() and (hourly >= 0).all()):
                    ok = False; break
                for a, b, w in FC_WINDOWS:
                    vec[f'{g}|rain_{w}_mm'] = sum(hourly[a:b].tolist())
            if ok:
                recs.append({'anchor_utc': anchor, 'age_hours': (anchor - cycle) / pd.Timedelta(hours=1), **vec})
                break
    sf = (pd.DataFrame(recs).set_index('anchor_utc') if recs else pd.DataFrame(columns=['age_hours'])).reindex(index)
    tf = (lambda v: np.log1p(v.clip(lower=0))) if cfg.get('transform', 'log1p') == 'log1p' else (lambda v: v)
    col = lambda g, w: sf[f'{g}|rain_{w}_mm'].astype(float) if f'{g}|rain_{w}_mm' in sf else pd.Series(np.nan, index=index)
    out = {}
    for name, g in cfg['named_cells'].items():
        for *_, w in FC_WINDOWS:
            out[f'fc_{name}_rain_{w}'] = tf(col(g, w))
    for *_, w in FC_WINDOWS:
        out[f'fc_basin_rain_{w}'] = tf(pd.concat([col(g, w) for g in cells], axis=1).mean(axis=1))
    frame = pd.DataFrame(out, index=index)
    frame['fc_age_hours'] = sf.age_hours.astype(float)
    frame['fc_missing_frac'] = frame.drop(columns='fc_age_hours').isna().any(axis=1).astype(float)
    if cfg.get('channels'):
        frame = frame[[c for c in frame.columns if c in cfg['channels'] or c == 'fc_missing_frac']]
    return frame


def sea_level_features(marine_json, index, week_hours=168, direct=False):
    """Offshore sea level from sea_level_height_msl_previous_day1 (mm), hours up to t+12 only:
    sea_setup_chg = mean(t+1..t+12) - mean(t-23..t), sea_anom = level(t) - trailing week mean,
    sea_next12_anom = mean(t+1..t+12) - week mean; direct adds sea_now and sea_p<h> = level(t+h) - level(t)."""
    x = pd.DataFrame(json.loads(Path(marine_json).read_text())['hourly'])
    x['time'] = pd.to_datetime(x.time, utc=True)
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


def flood_season(index):
    return pd.DataFrame({'flood_season': index.tz_convert('Asia/Ho_Chi_Minh').month.isin([9, 10, 11, 12]).astype('uint8')},
                        index=index)


# ===================================================================================== forecaster
class Forecaster:
    def __init__(self, bundle_path, data_dir=None, device='cpu'):
        self.bundle = joblib.load(bundle_path)
        if self.bundle.get('format') != 'hydrology_station_bundle_v2':
            raise ValueError('unsupported bundle format')
        self.data_dir = Path(data_dir) if data_dir else Path(bundle_path).resolve().parent.parent / 'data' / 'open_meteo'
        self.device = device
        self.models = {name: build_model(m).to(device) for name, m in self.bundle['members'].items()}

    def source(self, key):
        return self.data_dir / self.bundle['sources'][key]

    def inputs(self, name, index, level, rows=None):
        """Standardized inputs [len(index), features] of one member on an hourly UTC grid (no windowing); with rows,
        tide and ECMWF channels are computed only at those grid positions (the only ones used by the windows)."""
        m = self.bundle['members'][name]
        cfg, norm = m['feature_config'], m['normalizers']
        blocks = [flood_season(index)]
        for g in m['groups']:
            if g == 'wind':
                w = wind_features(self.source('wind'), index, onshore_pos=cfg['wind'].get('onshore_pos', False))
                if cfg['wind'].get('channels'):
                    w = w[[c for c in w.columns if c in cfg['wind']['channels'] or c == 'wind_missing_frac']]
                blocks.append(w)
            elif g == 'tide':
                blocks.append(tide_features(level, np.isfinite(level.to_numpy()), window_days=cfg['tide']['window_days'], rows=rows))
            elif g == 'fc_ecmwf':
                first = index.min() if rows is None else index[min(rows)]
                runs = load_single_runs(self.source('ecmwf'), first - pd.Timedelta(hours=cfg['fc_ecmwf'].get('max_age_hours', 24)),
                                        index.max())
                blocks.append(ecmwf_rain_features(runs, index, cfg['fc_ecmwf'], rows=rows))
            elif g == 'sea':
                blocks.append(sea_level_features(self.source('marine'), index, cfg['sea'].get('week_hours', 168),
                                                 cfg['sea'].get('direct', False)))
            else:
                raise ValueError(f'unknown input group {g}')
        table = pd.concat(blocks, axis=1)
        cols = m['feature_columns']
        x = np.zeros((len(index), len(cols)), dtype=np.float32)
        x[:, 0] = np.nan_to_num((level.to_numpy() - norm['water']['mean']) / norm['water']['scale'], nan=0.)
        x[:, 1] = ~np.isfinite(level.to_numpy())
        for j, c in enumerate(cols[2:], start=2):
            v = table[c].to_numpy(dtype=float)
            if c in norm['shared']:
                v = (v - norm['shared'][c]['mean']) / norm['shared'][c]['scale']
            x[:, j] = np.nan_to_num(v, nan=0.)
        return x

    def forecast(self, times, level=None, readings=None, database_url=None, min_coverage=0.8):
        """Forecasts for issue times `times` (UTC hours). level: hourly Series (mm, UTC) covering at least
        HISTORY_DAYS before the first time; otherwise built from `readings` (10-min DataFrame) or the database.
        Returns a DataFrame: issue_time_utc, horizon_h, valid_time_utc, forecast_mm, level_now_mm, plus one column per
        member (its own forecast) and input_coverage (share of observed hours in the longest input window)."""
        times = pd.DatetimeIndex(pd.to_datetime(times, utc=True)).floor('h')
        index = pd.date_range(times.min() - pd.Timedelta(days=HISTORY_DAYS), times.max(), freq='h')
        if level is None:
            cfg = next(iter(self.bundle['members'].values()))['feature_config']
            if readings is None:
                readings = readings_from_db(self.bundle['station_id'], index.min() - pd.Timedelta(hours=2),
                                            index.max() + pd.Timedelta(hours=1), database_url)
            level = hourly_level(readings, index, cfg)
        level = level.reindex(index).astype(float)
        rows = index.get_indexer(times)
        per, cov = {}, None
        for name, m in self.bundle['members'].items():
            L = m['sequence_length']
            needed = sorted({j for r in rows for j in range(r - L + 1, r + 1)})
            x = self.inputs(name, index, level, needed)
            windows = np.stack([x[r - L + 1:r + 1] for r in rows])
            seen = np.isfinite(level.to_numpy())
            c = np.array([seen[r - L + 1:r + 1].mean() for r in rows])
            cov = c if cov is None else np.minimum(cov, c)
            with torch.inference_mode():
                station = torch.full((len(rows),), m['station_index'], dtype=torch.long, device=self.device)
                out = self.models[name](torch.from_numpy(windows).to(self.device), station).cpu().numpy()
            t = m['normalizers']['targets']
            per[name] = {h: out[:, i] * t[f'h{h}']['scale'] + t[f'h{h}']['mean'] for i, h in enumerate(m['horizons'])}
        recs = []
        for h, names in self.bundle['combination']['horizons'].items():
            h = int(h)
            value = np.mean([per[n][h] for n in names], axis=0)
            frame = pd.DataFrame({'issue_time_utc': times, 'horizon_h': h, 'valid_time_utc': times + pd.Timedelta(hours=h),
                                  'forecast_mm': value, 'level_now_mm': level.to_numpy()[rows], 'input_coverage': cov})
            for n in per:
                frame[f'member_{n}_mm'] = per[n][h] if h in per[n] else np.nan
            recs.append(frame)
        out = pd.concat(recs, ignore_index=True).sort_values(['issue_time_utc', 'horizon_h']).reset_index(drop=True)
        out['reliable'] = out.input_coverage >= min_coverage
        return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bundle', help='default: the only .joblib in ./model')
    p.add_argument('--data-dir', default=str(HERE / 'data' / 'open_meteo'))
    p.add_argument('--time', help='issue time (default: latest full hour, UTC)')
    p.add_argument('--start'); p.add_argument('--end')
    p.add_argument('--level-file', help='CSV/parquet of 10-minute readings: observed_at, depth[, quality_status]')
    p.add_argument('--database-url', help='default: env DATABASE_URL')
    p.add_argument('--out', help='CSV output (default: print)')
    a = p.parse_args()
    bundle = a.bundle or str(next((HERE / 'model').glob('*.joblib')))
    f = Forecaster(bundle, a.data_dir)
    if a.start:
        times = pd.date_range(pd.Timestamp(a.start).tz_convert('UTC') if pd.Timestamp(a.start).tzinfo else pd.Timestamp(a.start, tz='UTC'),
                              pd.Timestamp(a.end).tz_convert('UTC') if pd.Timestamp(a.end).tzinfo else pd.Timestamp(a.end, tz='UTC'), freq='h')
    else:
        t = pd.Timestamp(a.time) if a.time else pd.Timestamp.now(tz='UTC').floor('h')
        times = [t.tz_convert('UTC') if t.tzinfo else t.tz_localize('UTC')]
    readings = None
    if a.level_file:
        readings = pd.read_parquet(a.level_file) if a.level_file.endswith('.parquet') else pd.read_csv(a.level_file)
    out = f.forecast(times, readings=readings, database_url=a.database_url)
    if a.out:
        out.to_csv(a.out, index=False)
        print(a.out, len(out), 'rows')
    else:
        print(out.to_string(index=False))


if __name__ == '__main__':
    main()
