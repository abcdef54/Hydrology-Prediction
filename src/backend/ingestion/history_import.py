"""Resumable raw water/rain history import; never changes SQLite or model artifacts."""

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
from threading import Event

from dotenv import load_dotenv
import sqlalchemy as sa

from src.backend.cache.latest import LatestObservationCache
from src.backend.db.models import Station
from src.backend.db.session import build_engine
from src.backend.ingestion.worker import (
    HueSourceClient,
    LOCK_KEYS,
    SourceError,
    SyncSettings,
    parse_source_row,
    upsert_observations,
    write_report,
)

SQLITE_TABLES = {
    "water_level_depth": "water_level_observations",
    "rain_water_depth": "rain_observations",
}
START_OF_HISTORY = datetime(1900, 1, 1, tzinfo=timezone.utc)
VALUE_COLUMNS = (
    "station_id",
    "observed_at",
    "depth",
    "source_name",
    "source_collection",
    "source_record_id",
    "quality_status",
)


def sqlite_batches(source, station, cutoff, batch_size=5000, after=None):
    """Read validated rows in bounded batches from an existing read-only snapshot.

    Args:
        source: SQLite connection opened mode=ro and query_only, owned by caller.
        station: PostgreSQL mapping including exact source station ID/collection.
        cutoff: Fixed timezone-aware UTC upper bound; future rows are rejected.
        batch_size: Maximum number of rows per yielded batch.
        after: Optional committed [time_point, key] checkpoint to resume after.

    Yields:
        Tuple of original-unit observation dictionaries and their SQLite checkpoint.

    Raises:
        SourceError: Inconsistent IDs, times, values or duplicate timestamps.
        sqlite3.Error: Source reads fail; no source data is written.
    """
    table = SQLITE_TABLES[station["observation_collection"]]
    parameters = [station["source_station_id"]]
    condition = "station_key = ?"
    if after:
        condition += " AND (time_point > ? OR (time_point = ? AND _key > ?))"
        parameters.extend([after[0], after[0], after[1]])
    cursor = source.execute(
        f"SELECT _key, time_point, station_key, raw_json FROM {table} "
        f"WHERE {condition} ORDER BY time_point, _key",
        parameters,
    )
    previous = datetime.fromisoformat(after[0]) if after else None
    while records := cursor.fetchmany(batch_size):
        observations = []
        for record in records:
            try:
                payload = json.loads(record["raw_json"])
                if str(payload.get("id")) != record["_key"]:
                    raise ValueError()
                observation = parse_source_row(payload, station, START_OF_HISTORY, cutoff)
                if (
                    datetime.fromisoformat(record["time_point"]) != observation["observed_at"]
                    or record["station_key"] != station["source_station_id"]
                ):
                    raise ValueError()
                if previous is not None and observation["observed_at"] <= previous:
                    raise ValueError()
            except (ValueError, TypeError, KeyError, AttributeError):
                raise SourceError("SQLite observation identity/time validation failed.") from None
            previous = observation["observed_at"]
            observations.append(observation)
        yield observations, [records[-1]["time_point"], records[-1]["_key"]]


def append_snapshot_batch(connection, rows):
    """COPY a batch and insert only absent observations, preserving all DB revisions.

    Args:
        connection: PostgreSQL connection inside the caller's transaction.
        rows: Validated raw observations of one station, without derived features.

    Returns:
        Inserted and retained counts. Existing values/received_at remain unchanged.

    Raises:
        SourceError: A source record ID maps to a different station or timestamp.
        sqlalchemy.exc.SQLAlchemyError: SQL failures; caller rolls back the batch.
    """
    if not rows:
        return {"inserted": 0, "retained": 0}
    connection.execute(sa.text("""
        CREATE TEMP TABLE IF NOT EXISTS history_stage (
            station_id text, observed_at timestamptz, depth double precision,
            source_name text, source_collection text, source_record_id bigint,
            quality_status text
        ) ON COMMIT DELETE ROWS
    """))
    with connection.connection.driver_connection.cursor() as cursor:
        with cursor.copy("COPY history_stage (" + ",".join(VALUE_COLUMNS) + ") FROM STDIN") as copy:
            for row in rows:
                copy.write_row(tuple(row[column] for column in VALUE_COLUMNS))
    conflict = connection.scalar(sa.text("""
        SELECT EXISTS (
            SELECT 1 FROM history_stage s JOIN hydrology.observations o
            ON o.source_name=s.source_name AND o.source_collection=s.source_collection
            AND o.source_record_id=s.source_record_id
            WHERE o.station_id<>s.station_id OR o.observed_at<>s.observed_at
        )
    """))
    if conflict:
        raise SourceError("SQLite source record conflicts with existing identity.")
    inserted = connection.execute(sa.text("""
        INSERT INTO hydrology.observations
            (station_id, observed_at, depth, source_name, source_collection,
             source_record_id, quality_status)
        SELECT station_id, observed_at, depth, source_name, source_collection,
               source_record_id, quality_status FROM history_stage
        ON CONFLICT DO NOTHING
    """)).rowcount
    return {"inserted": inserted, "retained": len(rows) - inserted}


def write_batch(engine, cache, station, rows, *, authoritative=False):
    """Commit one bounded batch, then invalidate the station's caches.

    Args:
        engine: PostgreSQL engine owned by caller.
        cache: Existing latest cache/generation invalidator, shared with worker.
        station: Station mapping associated with these observations.
        rows: Validated source rows in original units.
        authoritative: API rows may update revisions; SQLite only inserts missing rows.

    Returns:
        Batch counts and cache invalidation success. No checkpoint precedes commit.

    Raises:
        SourceError, sqlalchemy.exc.SQLAlchemyError: Invalid identities or DB failure.
    """
    with engine.begin() as connection:
        connection.execute(
            sa.text("SELECT pg_advisory_xact_lock(:first,:second)"),
            {"first": LOCK_KEYS[0], "second": LOCK_KEYS[1]},
        )
        result = (
            upsert_observations(connection, rows)
            if authoritative
            else append_snapshot_batch(connection, rows)
        )
    changed = result["inserted"] + result.get("updated", 0)
    result["cache_invalidated"] = not changed or cache.invalidate(station["id"])
    return result


def api_bound(client, station, descending=False):
    """Read earliest/latest source timestamp with bounded authentication renewal.

    Args:
        client: Source client whose token stays in memory.
        station: Exact source station mapping.
        descending: True selects latest; False earliest.

    Returns:
        UTC timestamp or None when source has no observations for this station.

    Raises:
        SourceError: Source request/schema/timezone is invalid.
    """
    params = {
        "filter[station_id][_in]": station["source_station_id"],
        "sort[]": "-time_point" if descending else "time_point",
        "limit": 1,
    }
    try:
        payload = client.request(
            "GET", "/items/" + station["observation_collection"], params=params
        )
    except SourceError as error:
        if error.status_code != 401:
            raise
        client.login()
        payload = client.request(
            "GET", "/items/" + station["observation_collection"], params=params
        )
    rows = payload.get("data")
    if not isinstance(rows, list):
        raise SourceError("Source observation bounds have invalid shape.")
    if not rows:
        return None
    try:
        timestamp = datetime.fromisoformat(rows[0]["time_point"])
        if timestamp.tzinfo is None:
            raise ValueError()
        return timestamp.astimezone(timezone.utc)
    except (KeyError, ValueError, TypeError):
        raise SourceError("Source bound timestamp is invalid.") from None


def api_windows(first, sqlite_first, sqlite_last, cutoff):
    """Select older missing coverage and the tail after a SQLite snapshot.

    Args:
        first: Earliest available API timestamp, or None for an empty station.
        sqlite_first/sqlite_last: Imported SQLite time bounds, or None for new stations.
        cutoff: Fixed upper bound, so a resume keeps the same requested interval.

    Returns:
        Inclusive API windows; a 24-hour overlap refreshes snapshot-tail revisions.
    """
    if first is None or first > cutoff:
        return []
    if sqlite_first is None or sqlite_last is None:
        return [(first, cutoff)]
    windows = []
    if first < sqlite_first:
        windows.append((first, sqlite_first))
    windows.append((max(first, sqlite_last - timedelta(hours=24)), cutoff))
    return [(start, end) for start, end in windows if start <= end]


def run_import(arguments):
    """Run/resume per-station SQLite and API history without enabling new stations.

    Args:
        arguments: CLI settings, including explicit --apply and optional station IDs.

    Returns:
        Report/checkpoint mapping. Partial committed batches survive interruption.

    Raises:
        SourceError: Source/checkpoint mismatch, or unvalidated source data.
        sqlalchemy.exc.SQLAlchemyError, sqlite3.Error: Storage failures.
    """
    load_dotenv()
    engine = build_engine()
    cache = LatestObservationCache.from_env()
    client = HueSourceClient(SyncSettings.from_env(), Event())
    source = sqlite3.connect(arguments.source.resolve().as_uri() + "?mode=ro", uri=True)
    source.row_factory = sqlite3.Row
    source.execute("PRAGMA query_only=ON")
    source.execute("BEGIN")
    identity = {
        "path": str(arguments.source.resolve()),
        "size": arguments.source.stat().st_size,
        "mtime_ns": arguments.source.stat().st_mtime_ns,
    }
    wal = Path(str(arguments.source) + "-wal")
    identity["wal"] = [wal.stat().st_size, wal.stat().st_mtime_ns] if wal.exists() else None
    report = {
        "source": identity,
        "cutoff": datetime.now(timezone.utc).isoformat(),
        "dry_run": not arguments.apply,
        "stations": {},
    }
    try:
        if not arguments.apply and arguments.report.exists():
            raise SourceError(
                "Dry-run requires a fresh report path; checkpoint will not be overwritten."
            )
        if arguments.apply and arguments.report.exists():
            report = json.loads(arguments.report.read_text())
            if report["source"] != identity or report["dry_run"]:
                raise SourceError("Checkpoint source differs; choose a new report path.")
        cutoff = datetime.fromisoformat(report["cutoff"])
        with engine.connect() as connection:
            query = (
                sa.select(Station.__table__)
                .where(
                    Station.source_name == "hue_dcu",
                Station.observation_collection.in_(SQLITE_TABLES),
                )
                .order_by(Station.id)
            )
            if arguments.station:
                query = query.where(Station.id.in_(arguments.station))
            stations = [dict(row) for row in connection.execute(query).mappings()]
        if not stations or (
            arguments.station and set(arguments.station) != {row["id"] for row in stations}
        ):
            raise SourceError("Requested source station is not configured.")
        known = {station["source_station_id"] for station in stations}
        if not arguments.station:
            report["unmapped_sqlite_stations"] = sorted(
                {
                    row[0]
                    for table in SQLITE_TABLES.values()
                    for row in source.execute(f"SELECT DISTINCT station_key FROM {table}")
                }
                - known
            )
        if arguments.stage in ("all", "api"):
            client.login()
        for station in stations:
            state = report["stations"].setdefault(station["id"], {})
            if arguments.refresh_api:
                # Explicit repair for observations published after the first read.
                for key in list(state):
                    if key in {"api_complete", "api_windows"} or key.startswith(
                        ("api_next_", "api_window_complete_")
                    ):
                        del state[key]
            table = SQLITE_TABLES[station["observation_collection"]]
            first, last, count = source.execute(
                f"SELECT MIN(time_point),MAX(time_point),COUNT(*) FROM {table} WHERE station_key=?",
                (station["source_station_id"],),
            ).fetchone()
            state["sqlite_available"] = count
            if arguments.stage in ("all", "sqlite") and not state.get("sqlite_complete"):
                for rows, after in sqlite_batches(
                    source, station, cutoff, arguments.batch_size, state.get("sqlite_after")
                ):
                    counts = write_batch(engine, cache, station, rows) if arguments.apply else {}
                    state["sqlite_processed"] = state.get("sqlite_processed", 0) + len(rows)
                    state["sqlite_inserted"] = state.get("sqlite_inserted", 0) + counts.get(
                        "inserted", 0
                    )
                    state["sqlite_retained"] = state.get("sqlite_retained", 0) + counts.get(
                        "retained", 0
                    )
                    state["sqlite_after"] = after
                    if not counts.get("cache_invalidated", True):
                        state["cache_invalidation_failed"] = True
                    write_report(arguments.report, report)
                state["sqlite_complete"] = True
                write_report(arguments.report, report)
                print(
                    json.dumps(
                        {
                            "station": station["id"],
                            "sqlite_processed": state.get("sqlite_processed", 0),
                            "sqlite_inserted": state.get("sqlite_inserted", 0),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            if arguments.stage in ("all", "api") and not state.get("api_complete"):
                if count and not state.get("sqlite_complete"):
                    raise SourceError("Import the SQLite station before API gap filling.")
                if "api_windows" not in state:
                    earliest = api_bound(client, station)
                    if arguments.refresh_api:
                        windows = api_windows(
                            max(earliest, cutoff - timedelta(hours=24)) if earliest else None,
                            None,
                            None,
                            cutoff,
                        )
                    else:
                        windows = api_windows(
                            earliest,
                            datetime.fromisoformat(first) if first else None,
                            datetime.fromisoformat(last) if last else None,
                            cutoff,
                        )
                    state["api_windows"] = [
                        [start.isoformat(), end.isoformat()] for start, end in windows
                    ]
                for index, (lower, upper) in enumerate(state["api_windows"]):
                    start = datetime.fromisoformat(state.get(f"api_next_{index}", lower))
                    end = datetime.fromisoformat(upper)
                    while start <= end:
                        stop = min(end, start + timedelta(days=7))
                        rows = client.observations(station, start, stop)
                        for offset in range(0, len(rows), arguments.batch_size):
                            batch = rows[offset : offset + arguments.batch_size]
                            counts = (
                                write_batch(engine, cache, station, batch, authoritative=True)
                                if arguments.apply
                                else {}
                            )
                            for key in ("inserted", "updated", "unchanged"):
                                state[f"api_{key}"] = state.get(f"api_{key}", 0) + counts.get(
                                    key, 0
                                )
                            if not counts.get("cache_invalidated", True):
                                state["cache_invalidation_failed"] = True
                        # Inclusive bounds: advance one microsecond, no omitted source instants.
                        start = stop + timedelta(microseconds=1)
                        state[f"api_next_{index}"] = start.isoformat()
                        write_report(arguments.report, report)
                    state[f"api_window_complete_{index}"] = True
                state["api_complete"] = True
                write_report(arguments.report, report)
                print(
                    json.dumps(
                        {
                            "station": station["id"],
                            "api_inserted": state.get("api_inserted", 0),
                            "api_updated": state.get("api_updated", 0),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        print(
            json.dumps(
                {
                    "stations": len(stations),
                    "stage": arguments.stage,
                    "dry_run": not arguments.apply,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return report
    finally:
        source.close()
        client.session.close()
        cache.close()
        engine.dispose()


def main():
    """Validate by default; --apply explicitly commits resumable history batches."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/hue_hydrology_all.sqlite"))
    parser.add_argument("--report", type=Path, default=Path("reports/full_history_import.json"))
    parser.add_argument("--stage", choices=("all", "sqlite", "api"), default="sqlite")
    parser.add_argument("--station", action="append")
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--refresh-api",
        action="store_true",
        help="Recheck selected completed stations for late source arrivals",
    )
    arguments = parser.parse_args()
    if arguments.refresh_api and not (
        arguments.apply and arguments.stage == "api" and arguments.station
    ):
        parser.error("refresh-api requires --apply --stage api and explicit --station")
    if not 1 <= arguments.batch_size <= 10000:
        parser.error("batch-size must be between 1 and 10000")
    try:
        run_import(arguments)
    except (SourceError, sa.exc.SQLAlchemyError, sqlite3.Error, ValueError) as error:
        parser.exit(
            1,
            f"History import stopped; error_type={type(error).__name__}; resume with same report\n",
        )


if __name__ == "__main__":
    main()
