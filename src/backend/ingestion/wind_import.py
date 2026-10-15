"""Import wind metadata/raw history explicitly; credentials remain server-only."""

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
from threading import Event

from dotenv import load_dotenv
import sqlalchemy as sa

from src.backend.db.models import Station
from src.backend.db.session import build_engine
from src.backend.ingestion.station_catalog import import_station_catalog
from src.backend.ingestion.wind import (
    normalize_wind_station,
    parse_wind_row,
    upsert_wind_observations,
)
from src.backend.ingestion.worker import (
    HueSourceClient,
    LOCK_KEYS,
    SourceError,
    SyncSettings,
    write_report,
)


def save_rows(engine, rows, *, snapshot):
    """Commit a wind batch using the existing writer lock, preserving snapshot collisions.

    Args:
        engine: Caller-owned application engine.
        rows: Validated observations of one station.
        snapshot: Older SQLite rows only insert; direct API may update revisions.

    Returns:
        Inserted/updated/unchanged counts; original received_at is retained.

    Raises:
        sqlalchemy.exc.SQLAlchemyError: Batch rolls back on storage/identity failure.
    """
    with engine.begin() as connection:
        connection.execute(
            sa.text("SELECT pg_advisory_xact_lock(:first,:second)"),
            {"first": LOCK_KEYS[0], "second": LOCK_KEYS[1]},
        )
        return upsert_wind_observations(connection, rows, preserve_existing=snapshot)


def import_wind_history(engine, client, source, stations, cutoff, report_path):
    """Import all validated wind sources, then fill history outside the SQLite snapshot.

    Args:
        engine: PostgreSQL engine, owned by caller.
        client: Authenticated source client with ID cursor paging and renewal.
        source: Read-only SQLite connection, owned by caller.
        stations: Exact UUID/internal wind station mappings.
        cutoff: Fixed UTC upper bound excluding future observations.
        report_path: Local ignored path for counts/coverage, without source payloads.

    Returns:
        Per-station report. Reruns are idempotent; prior committed batches survive errors.

    Raises:
        SourceError: Bad source metadata/readings or ambiguous timestamps.
        sqlalchemy.exc.SQLAlchemyError, sqlite3.Error: Storage/source read failure.
    """
    report = {"cutoff_utc": cutoff.isoformat(), "stations": {}}
    start_of_history = datetime(1900, 1, 1, tzinfo=timezone.utc)
    for station in stations:
        summary = {"sqlite_processed": 0, "sqlite_inserted": 0, "api_inserted": 0, "api_updated": 0}
        report["stations"][station["id"]] = summary
        first, last, count = source.execute(
            "SELECT MIN(time_point),MAX(time_point),COUNT(*) FROM wind_observations WHERE station_key=?",
            (station["source_station_id"],),
        ).fetchone()
        summary["sqlite_count"] = count
        cursor = source.execute(
            "SELECT _key,time_point,raw_json FROM wind_observations WHERE station_key=? ORDER BY time_point,_key",
            (station["source_station_id"],),
        )
        previous = None
        while batch := cursor.fetchmany(2000):
            rows = []
            for record in batch:
                payload = json.loads(record["raw_json"])
                row = parse_wind_row(payload, station, start_of_history, cutoff)
                if (
                    str(payload["id"]) != record["_key"]
                    or datetime.fromisoformat(record["time_point"]) != row["observed_at"]
                ):
                    raise SourceError("SQLite wind row identity/time mismatch.")
                if previous is not None and row["observed_at"] <= previous:
                    raise SourceError("Duplicate or unordered SQLite wind timestamp.")
                previous = row["observed_at"]
                rows.append(row)
            counts = save_rows(engine, rows, snapshot=True)
            summary["sqlite_processed"] += len(rows)
            summary["sqlite_inserted"] += counts["inserted"]
            write_report(report_path, report)
        params = {
            "filter[sid][_in]": station["source_station_id"],
            "sort[]": "time_point",
            "limit": 1,
        }
        earliest = client.request("GET", "/items/iot_wind_speed", params=params).get("data")
        if not isinstance(earliest, list):
            raise SourceError("Wind source bounds have invalid shape.")
        windows = []
        if earliest:
            begin = datetime.fromisoformat(earliest[0]["time_point"])
            if begin.tzinfo is None:
                raise SourceError("Wind source bound needs a timezone.")
            if first and last:
                first, last = datetime.fromisoformat(first), datetime.fromisoformat(last)
                if begin < first:
                    windows.append((begin, first))
                windows.append((max(begin, last - timedelta(hours=24)), cutoff))
            else:
                windows.append((begin, cutoff))
        for start, end in windows:
            while start <= end:
                stop = min(start + timedelta(days=7), end)
                rows = client.observations(station, start, stop)
                counts = save_rows(engine, rows, snapshot=False)
                summary["api_inserted"] += counts["inserted"]
                summary["api_updated"] += counts["updated"]
                write_report(report_path, report)
                start = stop + timedelta(microseconds=1)
        summary["complete"] = True
        write_report(report_path, report)
        print(json.dumps({"station": station["id"], **summary}), flush=True)
    return report


def main():
    """Import only with explicit --apply; preview validates the live metadata catalog."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--source", type=Path, default=Path("data/hue_hydrology_all.sqlite"))
    parser.add_argument("--report", type=Path, default=Path("reports/wind_import.json"))
    args = parser.parse_args()
    load_dotenv()
    engine = build_engine()
    client = HueSourceClient(SyncSettings.from_env(), Event())
    try:
        client.login()
        stations = []
        for page in range(1, 11):
            rows = client.request(
                "GET", "/items/iot_wind_station", params={"page": page, "limit": 100, "sort": "id"}
            ).get("data")
            if not isinstance(rows, list):
                raise SourceError("Wind catalog is not an array.")
            if not rows:
                break
            stations.extend(normalize_wind_station(row) for row in rows)
        else:
            raise SourceError("Wind metadata pagination exceeded bound.")
        if (
            not stations
            or len({s["id"] for s in stations}) != len(stations)
            or len({s["source_station_id"] for s in stations}) != len(stations)
        ):
            raise SourceError("Wind catalog is empty or contains duplicate identities.")
        metadata = import_station_catalog(engine, stations, dry_run=not args.apply)
        print(
            json.dumps(
                {"wind_stations": len(stations), "dry_run": not args.apply, **metadata["counts"]}
            ),
            flush=True,
        )
        if args.apply:
            with engine.connect() as connection:
                stations = [
                    dict(row)
                    for row in connection.execute(
                        sa.select(Station.__table__).where(
                            Station.id.in_([s["id"] for s in stations])
                        )
                    ).mappings()
                ]
            source = sqlite3.connect(args.source.resolve().as_uri() + "?mode=ro", uri=True)
            source.row_factory = sqlite3.Row
            try:
                source.execute("PRAGMA query_only=ON")
                source.execute("BEGIN")
                import_wind_history(
                    engine, client, source, stations, datetime.now(timezone.utc), args.report
                )
            finally:
                source.close()
    except (SourceError, sa.exc.SQLAlchemyError, sqlite3.Error, ValueError) as error:
        parser.exit(
            1, f"Wind import stopped; error_type={type(error).__name__}; rerun is idempotent\n"
        )
    finally:
        client.session.close()
        engine.dispose()


if __name__ == "__main__":
    main()
