"""Backfill water/rain observation history from the Hue DCU API into PostgreSQL (no SQLite snapshot needed).

The scheduled worker only re-reads the last 24 h after each station's newest stored observation. This command fills
an explicit historical range for selected stations, 7 days per request window, with the same validation, ID-cursor
pagination and idempotent UPSERT as the worker (rerunning a range never duplicates rows).

Credentials come from the environment / .env (see src/backend/ingestion/README.md):
    DATABASE_URL, HYDROLOGY_API_BASE_URL, HYDROLOGY_API_EMAIL, HYDROLOGY_API_PASSWORD
These are the values of `dcu_base_url`, `dcu_email`, `dcu_password` in the Postman environment HueNgapLut.

Run (dry run first, then --apply):
    .venv/bin/python -m src.backend.ingestion.api_backfill --station 0074647402 \
        --start 2025-11-01T00:00:00+00:00 --end 2026-11-12T00:00:00+00:00
    .venv/bin/python -m src.backend.ingestion.api_backfill --station 0074647402 \
        --start 2025-11-01T00:00:00+00:00 --end 2026-11-12T00:00:00+00:00 --apply
The station must exist in hydrology.stations (Alembic migrations + `python -m src.backend.ingestion.station_catalog --apply`).
"""
import argparse
import json
from datetime import datetime, timedelta, timezone
from threading import Event

from dotenv import load_dotenv
import sqlalchemy as sa

from src.backend.cache.latest import LatestObservationCache
from src.backend.db.models import Station
from src.backend.db.session import build_engine
from src.backend.ingestion.history_import import SQLITE_TABLES, write_batch
from src.backend.ingestion.worker import HueSourceClient, SourceError, SyncSettings

WINDOW = timedelta(days=7)


def backfill(station_ids, start, end, *, apply=False):
    """Fetch [start, end] (inclusive, UTC) for each station and optionally UPSERT it.

    Args:
        station_ids: hydrology.stations IDs (source_name hue_dcu, water or rain collection).
        start, end: timezone-aware bounds; end is clipped to now.
        apply: False only counts what the API returns (nothing is written).

    Returns:
        {station_id: {"fetched", "inserted", "updated", "unchanged", "first", "last"}}.
    """
    load_dotenv()
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError('Use timezone-aware ISO times, e.g. 2025-11-01T00:00:00+00:00')
    end = min(end, datetime.now(timezone.utc))
    engine = build_engine()
    cache = LatestObservationCache.from_env()
    client = HueSourceClient(SyncSettings.from_env(), Event())
    report = {}
    try:
        with engine.connect() as connection:
            rows = connection.execute(sa.select(Station.__table__).where(
                Station.id.in_(station_ids), Station.source_name == 'hue_dcu',
                Station.observation_collection.in_(SQLITE_TABLES))).mappings().all()
        stations = {row['id']: dict(row) for row in rows}
        missing = set(station_ids) - set(stations)
        if missing:
            raise SourceError(f'Stations not configured in hydrology.stations: {sorted(missing)}')
        client.login()
        for sid in station_ids:
            station = stations[sid]
            state = report[sid] = dict(fetched=0, inserted=0, updated=0, unchanged=0, first=None, last=None)
            lower = start
            while lower <= end:
                upper = min(end, lower + WINDOW)
                observations = client.observations(station, lower, upper)
                state['fetched'] += len(observations)
                if observations:
                    times = [o['observed_at'] for o in observations]
                    state['first'] = min([t for t in [state['first'], min(times)] if t is not None])
                    state['last'] = max([t for t in [state['last'], max(times)] if t is not None])
                if apply and observations:
                    counts = write_batch(engine, cache, station, observations, authoritative=True)
                    for key in ('inserted', 'updated', 'unchanged'):
                        state[key] += counts.get(key, 0)
                lower = upper + timedelta(microseconds=1)       # inclusive bounds, no omitted instants
            print(json.dumps({sid: {k: (str(v) if isinstance(v, datetime) else v) for k, v in state.items()}},
                             ensure_ascii=False), flush=True)
    finally:
        client.session.close()
        cache.close()
        engine.dispose()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--station', action='append', required=True, help='hydrology.stations ID; repeatable')
    parser.add_argument('--start', required=True, type=datetime.fromisoformat)
    parser.add_argument('--end', required=True, type=datetime.fromisoformat)
    parser.add_argument('--apply', action='store_true', help='write to PostgreSQL (default: dry run)')
    a = parser.parse_args()
    try:
        backfill(a.station, a.start, a.end, apply=a.apply)
    except (SourceError, sa.exc.SQLAlchemyError, ValueError) as error:
        parser.exit(1, f'Backfill stopped: {type(error).__name__}: {error}\n')


if __name__ == '__main__':
    main()
