"""Keep the Open-Meteo input files of one station up to date (run every hour, e.g. from cron or a systemd timer).

Requirements: python >= 3.10, pandas, joblib (only to read the request points from the bundle). Open-Meteo is a
public API without key; the free tier allows about 10 000 calls/day (each point of a request counts).

What it updates, inside ./data/open_meteo (names given in the bundle, see README.md):
  * wind     Previous Runs API (ECMWF IFS 0.25 deg): wind speed/direction 10 m and MSL pressure, incl. *_previous_day1.
             Re-downloads [last stored day - 2, today + 2] and merges it into the existing JSON (new non-null values win).
  * ecmwf    Single Runs API (ECMWF IFS): one JSON per run (00/06/12/18 UTC), hourly precipitation 72 h ahead at the
             station's rain points. Downloads every run not stored yet, up to the latest cycle. A run that the API does
             not (yet) have is retried on the next call; after 2 days its error marker is stored and it is skipped.
  * marine   Marine API (only stations whose bundle uses sea level): sea_level_height_msl incl. *_previous_day1, merged
             like the wind file.

    python worker.py              # update everything once
    python worker.py --dry-run    # show what would be requested
"""
import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import joblib
import pandas as pd

HERE = Path(__file__).resolve().parent
PREVIOUS_RUNS_URL = 'https://previous-runs-api.open-meteo.com/v1/forecast'
SINGLE_RUNS_URL = 'https://single-runs-api.open-meteo.com/v1/forecast'
MARINE_URL = 'https://marine-api.open-meteo.com/v1/marine'
WIND_VARIABLES = ['wind_speed_10m', 'wind_direction_10m', 'pressure_msl', 'wind_speed_10m_previous_day1',
                  'wind_direction_10m_previous_day1', 'pressure_msl_previous_day1']
MARINE_VARIABLES = ['sea_level_height_msl', 'sea_level_height_msl_previous_day1', 'invert_barometer_height',
                    'invert_barometer_height_previous_day1']


def get(url, params, retries=5, pause=5.0):
    query = urllib.parse.urlencode(params)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(f'{url}?{query}', timeout=120) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors='replace')
            if e.code == 429 and attempt < retries - 1:            # rate limit: wait and retry
                time.sleep(60 * (attempt + 1)); continue
            if e.code == 400:                                    # e.g. run not available: return the error payload
                try:
                    return json.loads(body)
                except ValueError:
                    pass
            if attempt == retries - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
        time.sleep(pause * (attempt + 1))


def merge_hourly(path, payload):
    """Merge the 'hourly' block of a new response into the stored JSON (same variables, keyed by time)."""
    path = Path(path)
    new = pd.DataFrame(payload['hourly']).set_index('time')
    if path.exists():
        old_payload = json.loads(path.read_text())
        old = pd.DataFrame(old_payload['hourly']).set_index('time')
        merged = new.combine_first(old)                      # new non-null values win, older hours are kept
        out = old_payload
    else:
        merged, out = new, payload
    merged = merged.sort_index()
    out['hourly'] = {'time': merged.index.tolist(), **{c: [None if pd.isna(v) else float(v) for v in merged[c]] for c in merged.columns}}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(out))
    tmp.replace(path)
    return merged.index.min(), merged.index.max()


def update_point_series(path, url, lat, lon, variables, extra, dry_run):
    path = Path(path)
    today = pd.Timestamp.now(tz='UTC').normalize()
    if path.exists():
        last = pd.Timestamp(pd.DataFrame(json.loads(path.read_text())['hourly']).time.max(), tz='UTC').normalize()
        start = min(last, today) - pd.Timedelta(days=2)
    else:
        start = today - pd.Timedelta(days=40)                  # enough history for a first forecast (32 days)
    end = today + pd.Timedelta(days=2)
    params = dict(latitude=lat, longitude=lon, start_date=f'{start:%Y-%m-%d}', end_date=f'{end:%Y-%m-%d}',
                  hourly=','.join(variables), timezone='UTC', **extra)
    if dry_run:
        print('GET', url, params); return
    payload = get(url, params)
    if payload.get('error'):
        raise RuntimeError(f'{url}: {payload.get("reason")}')
    lo, hi = merge_hourly(path, payload)
    print(f'{path.name}: {lo} .. {hi}')


def update_single_runs(run_dir, points, dry_run, give_up_days=2, pause=0.5):
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz='UTC')
    have = sorted(pd.Timestamp(f.stem, tz='UTC') for f in run_dir.glob('*.json'))
    first = (have[-1] + pd.Timedelta(hours=6)) if have else (now - pd.Timedelta(days=3)).floor('6h')
    first = min(first, (now - pd.Timedelta(days=give_up_days)).floor('6h'))     # re-check recent missing runs
    lat = ','.join(f'{p[0]}' for p in points); lon = ','.join(f'{p[1]}' for p in points)
    for cycle in pd.date_range(first, now.floor('6h'), freq='6h'):
        f = run_dir / f'{cycle:%Y%m%dT%H%M}.json'
        if f.exists() and not isinstance(json.loads(f.read_text()), dict):
            continue
        params = dict(latitude=lat, longitude=lon, hourly='precipitation', models='ecmwf_ifs', run=f'{cycle:%Y-%m-%dT%H:%M}',
                      forecast_days=3, cell_selection='land', precipitation_unit='mm', timeformat='unixtime', timezone='UTC')
        if dry_run:
            print('GET', SINGLE_RUNS_URL, params); continue
        payload = get(SINGLE_RUNS_URL, params)
        if isinstance(payload, dict) and payload.get('error'):
            if now - cycle > pd.Timedelta(days=give_up_days):
                f.write_text(json.dumps(payload))             # permanent gap: store the marker, never retried
            print(f'{f.name}: not available ({payload.get("reason")})')
        else:
            f.write_text(json.dumps(payload))
            print(f'{f.name}: stored')
        time.sleep(pause)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bundle', help='default: the only .joblib in ./model')
    p.add_argument('--data-dir', default=str(HERE / 'data' / 'open_meteo'))
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    bundle = joblib.load(a.bundle or next((HERE / 'model').glob('*.joblib')))
    data, src = Path(a.data_dir), bundle['sources']
    cfgs = [m['feature_config'] for m in bundle['members'].values()]
    groups = {g for m in bundle['members'].values() for g in m['groups']}
    if 'wind' in groups:
        lat, lon = next(c for c in cfgs if 'wind' in c)['wind']['request_point']
        update_point_series(data / src['wind'], PREVIOUS_RUNS_URL, lat, lon, WIND_VARIABLES,
                            dict(models='ecmwf_ifs025', wind_speed_unit='ms'), a.dry_run)
    if 'fc_ecmwf' in groups:
        update_single_runs(data / src['ecmwf'], next(c for c in cfgs if 'fc_ecmwf' in c)['fc_ecmwf']['request_points'], a.dry_run)
    if 'sea' in groups:
        lat, lon = next(c for c in cfgs if 'sea' in c)['sea']['request_point']
        update_point_series(data / src['marine'], MARINE_URL, lat, lon, MARINE_VARIABLES, {}, a.dry_run)


if __name__ == '__main__':
    main()
