"""Resumable Open-Meteo single-run archive collection, without dataset generation."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from uuid import uuid4

import pandas as pd
import requests
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert
from dotenv import load_dotenv

from src.backend.db.models import Station
from src.backend.db.session import build_engine
from src.backend.db.weather import locations, runs, hourly, fetches

UTC = timezone.utc
ENDPOINT = 'https://single-runs-api.open-meteo.com/v1/forecast'
MODEL = 'ecmwf_ifs'


def dataset_times(directory):
    """Read the actual split timestamps, preserving gaps and checking UTC identity."""
    parts = {}
    for split in ('train', 'val', 'test'):
        path = directory / f'{split}.csv'
        if not path.exists() and split == 'val':
            path = directory / 'validation.csv'
        values = pd.read_csv(path, usecols=['timestamp_utc'])['timestamp_utc']
        # utc=True must not silently reinterpret naive local timestamps.
        if any(pd.Timestamp(v).tzinfo is None for v in values):
            raise ValueError(f'{path}: timezone required')
        stamps = pd.DatetimeIndex(pd.to_datetime(values, utc=True))
        if stamps.empty or stamps.has_duplicates or not stamps.is_monotonic_increasing:
            raise ValueError(f'{path}: empty, duplicated or unsorted timestamps')
        if any(t.minute or t.second or t.microsecond for t in stamps):
            raise ValueError(f'{path}: timestamps must be on an hourly grid')
        parts[split] = stamps
    combined = parts['train'].append([parts['val'], parts['test']])
    if combined.has_duplicates or not combined.is_monotonic_increasing:
        raise ValueError('Split timestamps overlap or are out of order')
    return parts


def cycles(start, end, lag_hours=6):
    """Include a preceding cycle to cover the first anchor after assumed release lag."""
    first = (start - timedelta(hours=lag_hours)).floor('6h')
    last = (end - timedelta(hours=lag_hours)).floor('6h')
    return list(pd.date_range(first, last, freq='6h').to_pydatetime())


def request_params(location, cycle):
    return dict(latitude=location['latitude'], longitude=location['longitude'],
                models=MODEL, run=cycle.strftime('%Y-%m-%dT%H:%M'),
                hourly='precipitation', timezone='UTC', timeformat='unixtime',
                precipitation_unit='mm', forecast_days=3, cell_selection='land')


def parse_response(payload, cycle):
    """Reject wrong units, malformed arrays and non-hourly grids; preserve nulls."""
    if payload.get('error'):
        raise ValueError('Provider returned an error payload')
    if payload.get('utc_offset_seconds') != 0:
        raise ValueError('Expected UTC response')
    for key, limit in [('latitude', 90), ('longitude', 180)]:
        val = payload.get(key)
        if not isinstance(val, (float, int)) or not math.isfinite(val) or abs(val) > limit:
            raise ValueError(f'Invalid grid {key}')
    if payload.get('hourly_units', {}).get('time') != 'unixtime':
        raise ValueError('Expected Unix timestamps')
    if payload.get('hourly_units', {}).get('precipitation') != 'mm':
        raise ValueError('Expected precipitation in mm')
    series = payload.get('hourly', {})
    times, values = series.get('time', []), series.get('precipitation', [])
    if not times or len(times) != len(values):
        raise ValueError('Missing or mismatched hourly arrays')
    result = []
    previous = None
    for raw_time, value in zip(times, values):
        stamp = datetime.fromtimestamp(raw_time, UTC)
        if stamp.minute or stamp.second or (previous and stamp - previous != timedelta(hours=1)):
            raise ValueError('Response is not a consecutive hourly grid')
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value) or value < 0):
            raise ValueError('Invalid precipitation value')
        if stamp > cycle:
            result.append(dict(valid_at=stamp, precipitation_mm=value,
                               quality_status='missing' if value is None else 'forecast'))
        previous = stamp
    if not result or result[0]['valid_at'] != cycle + timedelta(hours=1):
        raise ValueError('Response does not start at the requested run')
    if result[-1]['valid_at'] < cycle + timedelta(hours=36):
        raise ValueError('Insufficient forecast horizon for 24h targets plus cycle/release lag')
    return result


def save_response(connection, location, cycle, params, payload, checksum, retrieved,
                  *, available_at=None, availability_basis='unknown_archive', archive_kind=None):
    points = parse_response(payload, cycle)
    # Archive metadata explicitly does not claim historical operational availability.
    transition = datetime(2026, 5, 12, 6, tzinfo=UTC)
    run_id = connection.execute(insert(runs).values(
        id=uuid4(), location_id=location['id'], model=MODEL, initialized_at=cycle,
        retrieved_at=retrieved, available_at=available_at, availability_basis=availability_basis,
        archive_kind=archive_kind or ('hindcast_49r1' if cycle < transition else 'archive_50r1_unverified'),
        grid_latitude=payload['latitude'], grid_longitude=payload['longitude'],
        checksum=checksum, payload=payload, request=params,
    ).on_conflict_do_nothing(constraint='uq_weather_run_vintage').returning(runs.c.id)).scalar()
    if run_id is not None:
        connection.execute(insert(hourly), [dict(run_id=run_id, **point) for point in points])
    return run_id is not None


def coverage(connection, location, parts, lag_hours):
    """Audit 24 consecutive non-null future hours with an explicitly assumed release lag."""
    rows = connection.execute(sa.select(runs.c.id, runs.c.initialized_at, runs.c.retrieved_at)
        .where(runs.c.location_id == location['id'], runs.c.model == MODEL)
        .order_by(runs.c.initialized_at, runs.c.retrieved_at)).all()
    latest = {r.initialized_at: r.id for r in rows}
    points = connection.execute(sa.select(hourly.c.run_id, hourly.c.valid_at)
        .join(runs, hourly.c.run_id == runs.c.id)
        .where(runs.c.location_id == location['id'], runs.c.model == MODEL,
               hourly.c.quality_status == 'forecast')).all()
    valid = {}
    for run_id, stamp in points:
        valid.setdefault(run_id, set()).add(stamp)
    ordered = sorted(latest)
    report = {}
    import bisect
    for split, anchors in parts.items():
        missing = []
        for anchor in anchors.to_pydatetime():
            idx = bisect.bisect_right(ordered, anchor - timedelta(hours=lag_hours)) - 1
            if idx < 0 or anchor - ordered[idx] > timedelta(hours=lag_hours + 6):
                missing.append(anchor.isoformat())
                continue
            stamps = valid.get(latest[ordered[idx]], set())
            if not all(anchor + timedelta(hours=h) in stamps for h in range(1,25)):
                missing.append(anchor.isoformat())
        report[split] = dict(anchors=len(anchors), complete_24h=len(anchors)-len(missing),
                             missing_anchors=missing)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir', type=Path, default=Path('train_data/da_vien_1h_2026-10-15'))
    p.add_argument('--station-ids', nargs='+', default=['750062', '750007'])
    p.add_argument('--report', type=Path, default=Path('reports/weather_collection/report.json'))
    p.add_argument('--max-runs', type=int, help='Limit cycles for an initial smoke test')
    p.add_argument('--request-interval', type=float, default=0.15,
                   help='Pause after each request, seconds (minimum 0.15; at most 400/minute)')
    p.add_argument('--assumed-release-lag-hours', type=int, default=6)
    p.add_argument('--refresh', action='store_true', help='Fetch again, preserving changed vintages')
    p.add_argument('--plan-only', action='store_true', help='Read dataset/catalog only; no API or DB writes')
    p.add_argument('--audit-only', action='store_true', help='Audit stored coverage without fetching')
    args = p.parse_args(argv)
    if not math.isfinite(args.request_interval) or args.request_interval < 0.15 or args.assumed_release_lag_hours < 0 or (args.max_runs is not None and args.max_runs < 1):
        p.error('finite interval >= 0.15, lag >= 0 and max-runs >= 1 required')
    load_dotenv()
    parts = dataset_times(args.dataset_dir)
    start, end = parts['train'][0], parts['test'][-1]
    planned = cycles(start, end, args.assumed_release_lag_hours)
    if args.max_runs:
        planned = planned[:args.max_runs]
    engine = build_engine()
    report = dict(dataset_dir=str(args.dataset_dir.resolve()), start_utc=start.isoformat(),
                  end_utc=end.isoformat(), model=MODEL, endpoint=ENDPOINT,
                  assumed_release_lag_hours=args.assumed_release_lag_hours,
                  availability_warning='Coverage is simulated with assumed lag, NOT verified historical publication.',
                  planned_cycles=len(planned), locations=[], failures=0,
                  dataset_checksums={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in args.dataset_dir.glob('*.csv')
                      if path.stem in {'train', 'val', 'validation', 'test'}})
    args.report.parent.mkdir(parents=True, exist_ok=True)
    def write():
        temporary = args.report.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        temporary.replace(args.report)
    try:
        with engine.connect() as c:
            stations = c.execute(sa.select(Station.id, Station.latitude, Station.longitude)
                .where(Station.id.in_(args.station_ids), Station.station_type == 'rainfall')).mappings().all()
        if set(args.station_ids) != {s['id'] for s in stations}:
            raise ValueError('Rain station IDs not found in catalog')
        for station in stations:
            if station['latitude'] is None or station['longitude'] is None:
                raise ValueError('Station coordinates missing')
            location = dict(id=f"dcu:{station['id']}:{station['latitude']}:{station['longitude']}",
                            station_id=station['id'], latitude=station['latitude'], longitude=station['longitude'])
            item = dict(location=location, fetched=0, skipped=0, failures=[])
            report['locations'].append(item)
            if args.plan_only:
                continue
            if not args.audit_only:
                with engine.begin() as c:
                    c.execute(insert(locations).values(**location).on_conflict_do_nothing())
            if not args.audit_only:
                with requests.Session() as client:
                    for cycle in planned:
                        with engine.connect() as c:
                            existing = c.execute(sa.select(runs.c.id).where(
                                runs.c.location_id == location['id'], runs.c.model == MODEL,
                                runs.c.initialized_at == cycle).limit(1)).scalar()
                        if existing and not args.refresh:
                            item['skipped'] += 1
                            continue
                        params = request_params(location, cycle)
                        status, checksum, error = None, None, None
                        retrieved = datetime.now(UTC)
                        payload = None
                        for attempt in range(3):
                            try:
                                response = client.get(ENDPOINT, params=params, timeout=(10,90))
                                status = response.status_code
                                checksum = hashlib.sha256(response.content).hexdigest()
                                if status == 429 or status >= 500:
                                    if attempt < 2:
                                        time.sleep(min(30, 2 ** (attempt + 1)))
                                        continue
                                response.raise_for_status()
                                payload = response.json()
                                parse_response(payload, cycle)
                                error = None
                                break
                            except (requests.RequestException, ValueError, TypeError, OverflowError) as exc:
                                # No URL/credentials or remote response text in error logs.
                                error = type(exc).__name__
                                if isinstance(exc, (requests.ConnectionError, requests.Timeout)) and attempt < 2:
                                    time.sleep(2 ** (attempt + 1))
                                    continue
                                break
                        retrieved = datetime.now(UTC)
                        with engine.begin() as c:
                            if error is None:
                                save_response(c, location, cycle, params, payload, checksum, retrieved)
                                item['fetched'] += 1
                            else:
                                item['failures'].append(dict(run=cycle.isoformat(), status=status, error=error))
                                report['failures'] += 1
                            c.execute(insert(fetches).values(id=uuid4(), location_id=location['id'],
                                initialized_at=cycle, retrieved_at=retrieved, http_status=status,
                                status='success' if error is None else 'failed', checksum=checksum,
                                request=params, error=error))
                        write()
                        print(f"{station['id']} {cycle.isoformat()} {'OK' if error is None else error}", flush=True)
                        time.sleep(args.request_interval)
            with engine.connect() as c:
                item['coverage'] = coverage(c, location, parts, args.assumed_release_lag_hours)
        write()
    finally:
        engine.dispose()
    print(f'Report: {args.report}')
    return 1 if report['failures'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
