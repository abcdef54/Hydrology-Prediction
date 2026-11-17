"""Download the ECMWF inputs used by the station datasets from Open-Meteo (public API, no key).

Two sources (https://open-meteo.com/en/docs/previous-runs-api, https://open-meteo.com/en/docs/single-runs-api):

* Previous Runs API: hourly 10 m wind and MSL pressure from ECMWF IFS 0.25 degree (`ecmwf_ifs025`). Only the
  `*_previous_day1` variables are used: the value for hour H was predicted 24 h before H, so features that look
  12 h ahead of an issue time t only use runs initialised before t-12 h.
* Single Runs API: every ECMWF IFS run (`ecmwf_ifs`, 00/06/12/18 UTC) separately, hourly precipitation for 72 h,
  at a list of points with `cell_selection=land`. One JSON file per run is cached, so interrupted downloads resume.

CLI:
    .venv/bin/python -m src.ai.station_dataset.open_meteo previous-runs --lat 16.85 --lon 107.85 \
        --start 2025-11-01 --end 2026-10-21 --out data/external/open_meteo/prev_offshore.json
    .venv/bin/python -m src.ai.station_dataset.open_meteo single-runs --points 16.458944,107.571361 16.554968,107.63527 \
        --start 2025-11-13T00:00 --end 2026-10-15T06:00 --out-dir data/external/open_meteo/ecmwf_single_runs
"""
import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

PREVIOUS_RUNS_URL = 'https://previous-runs-api.open-meteo.com/v1/forecast'
SINGLE_RUNS_URL = 'https://single-runs-api.open-meteo.com/v1/forecast'
MARINE_URL = 'https://marine-api.open-meteo.com/v1/marine'
MARINE_VARIABLES = ['sea_level_height_msl', 'sea_level_height_msl_previous_day1', 'invert_barometer_height',
                    'invert_barometer_height_previous_day1']
WIND_VARIABLES = ['wind_speed_10m', 'wind_direction_10m', 'pressure_msl', 'wind_speed_10m_previous_day1',
                  'wind_direction_10m_previous_day1', 'pressure_msl_previous_day1']


def _get(url, params, retries=3, pause=5.0):
    full = url + '?' + urllib.parse.urlencode(params)
    attempt = 0
    while True:
        try:
            with urllib.request.urlopen(full, timeout=120) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            body = error.read()
            try:
                payload = json.loads(body)
            except ValueError:
                payload = None
            if error.code == 400 and payload and payload.get('error'):
                return payload                                   # e.g. run not available: cached as a marker
            if error.code == 429:                                # rate limit (per minute / hour / day): wait and retry
                if attempt >= retries + 12:
                    raise
                wait = min(900, 60 * (attempt + 1))
                print(f'HTTP 429, waiting {wait} s', flush=True)
                time.sleep(wait)
                attempt += 1
                continue
            if attempt == retries - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
        time.sleep(pause * (attempt + 1))
        attempt += 1


def fetch_previous_runs(lat, lon, start, end, out):
    """Hourly wind/pressure (incl. *_previous_day1) for one point, UTC, wind in m/s, saved as the raw JSON."""
    params = dict(latitude=lat, longitude=lon, start_date=start, end_date=end, hourly=','.join(WIND_VARIABLES),
                  models='ecmwf_ifs025', timezone='UTC', wind_speed_unit='ms')
    payload = _get(PREVIOUS_RUNS_URL, params)
    if payload.get('error'):
        raise RuntimeError(payload.get('reason'))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload))
    return payload


def fetch_single_runs(points, start, end, out_dir, pause=0.5, hourly='precipitation'):
    """Cache one JSON per ECMWF run initialised every 6 h in [start, end] (UTC) for all points in one request.

    points: [(lat, lon), ...] request coordinates (the station locations); Open-Meteo snaps each to its land grid cell.
    A run the API does not have (e.g. 2026-06-10T12:00) is stored as its error payload and skipped by the reader.
    hourly: comma-separated Open-Meteo variables (default precipitation only; e.g. add the ECMWF soil layers
    soil_moisture_0_to_7cm, soil_moisture_7_to_28cm, soil_moisture_28_to_100cm, soil_moisture_100_to_255cm).
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    lat = ','.join(f'{p[0]}' for p in points); lon = ','.join(f'{p[1]}' for p in points)
    for cycle in pd.date_range(pd.Timestamp(start, tz='UTC'), pd.Timestamp(end, tz='UTC'), freq='6h'):
        f = out_dir / f'{cycle:%Y%m%dT%H%M}.json'
        if f.exists():
            continue
        params = dict(latitude=lat, longitude=lon, hourly=hourly, models='ecmwf_ifs', run=f'{cycle:%Y-%m-%dT%H:%M}',
                      forecast_days=3, cell_selection='land', precipitation_unit='mm', timeformat='unixtime', timezone='UTC')
        f.write_text(json.dumps(_get(SINGLE_RUNS_URL, params)))
        time.sleep(pause)


def fetch_marine(lat, lon, start, end, out):
    """Hourly modelled sea level for one sea point (Marine API, UTC, metres), incl. *_previous_day1 = value forecast
    24 h before the valid hour (only these are used as inputs, like the wind)."""
    params = dict(latitude=lat, longitude=lon, start_date=start, end_date=end, hourly=','.join(MARINE_VARIABLES), timezone='UTC')
    payload = _get(MARINE_URL, params)
    if payload.get('error'):
        raise RuntimeError(payload.get('reason'))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload))
    return payload


def grid_id(lat, lon):
    return f'ecmwf_ifs:{lat:.6f}:{lon:.6f}'


def load_single_runs(out_dir):
    """{initialized_at (UTC Timestamp): {grid_id: hourly precipitation Series (UTC valid time)}} from the cache."""
    runs = {}
    for f in sorted(Path(out_dir).glob('*.json')):
        payload = json.loads(f.read_text())
        if isinstance(payload, dict):                            # error marker (run not available)
            continue
        cycle = pd.Timestamp(f.stem, tz='UTC')
        per = {}
        for item in payload:
            t = pd.to_datetime(item['hourly']['time'], unit='s', utc=True)
            per[grid_id(item['latitude'], item['longitude'])] = pd.Series(item['hourly']['precipitation'], index=t, dtype=float)
        runs[cycle] = per
    return runs


def load_single_runs_vars(out_dir, variables):
    """{initialized_at (UTC Timestamp): DataFrame indexed by valid time (UTC), columns '<grid_id>|<variable>'}.
    Error markers (runs the API does not have) are skipped. Missing values stay NaN."""
    runs = {}
    for f in sorted(Path(out_dir).glob('*.json')):
        payload = json.loads(f.read_text())
        if isinstance(payload, dict):
            continue
        cols = {}
        for item in payload:
            t = pd.to_datetime(item['hourly']['time'], unit='s', utc=True)
            g = grid_id(item['latitude'], item['longitude'])
            for v in variables:
                cols[f'{g}|{v}'] = pd.Series(item['hourly'][v], index=t, dtype=float)
        runs[pd.Timestamp(f.stem, tz='UTC')] = pd.DataFrame(cols)
    return runs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('previous-runs'); a.add_argument('--lat', type=float, required=True); a.add_argument('--lon', type=float, required=True)
    a.add_argument('--start', required=True); a.add_argument('--end', required=True); a.add_argument('--out', required=True)
    c = sub.add_parser('marine'); c.add_argument('--lat', type=float, required=True); c.add_argument('--lon', type=float, required=True)
    c.add_argument('--start', required=True); c.add_argument('--end', required=True); c.add_argument('--out', required=True)
    b = sub.add_parser('single-runs'); b.add_argument('--points', nargs='+', required=True, help='lat,lon pairs')
    b.add_argument('--start', required=True); b.add_argument('--end', required=True); b.add_argument('--out-dir', required=True)
    b.add_argument('--hourly', default='precipitation', help='comma-separated hourly variables')
    b.add_argument('--pause', type=float, default=0.5, help='seconds between requests (rate limits)')
    args = p.parse_args()
    if args.cmd == 'marine':
        fetch_marine(args.lat, args.lon, args.start, args.end, args.out)
    elif args.cmd == 'previous-runs':
        fetch_previous_runs(args.lat, args.lon, args.start, args.end, args.out)
    else:
        fetch_single_runs([tuple(map(float, x.split(','))) for x in args.points], args.start, args.end, args.out_dir,
                          hourly=args.hourly, pause=args.pause)


if __name__ == '__main__':
    main()
