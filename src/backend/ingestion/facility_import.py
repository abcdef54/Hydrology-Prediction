"""Explicit SOHOA catalog/history import; SQLite is read only, reports exclude private payloads."""

import argparse
from collections import Counter
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3

from dotenv import load_dotenv
import sqlalchemy as sa
from src.backend.db.models import Facility
from src.backend.db.session import build_engine
from src.backend.ingestion.facilities import (
    normalize_facility,
    parse_operation,
    upsert_facilities,
    upsert_operations,
    VIETNAM,
)
from src.backend.ingestion.facility_source import FacilitySourceClient, FacilitySourceSettings
from src.backend.ingestion.worker import LOCK_KEYS, SourceError, write_report


def write_batches(engine, rows, snapshot=False):
    """Commit validated report batches under the common worker lock.

    Args:
        engine: Caller-owned PostgreSQL engine.
        rows: Parsed operational records with no user identifiers or attachments.
        snapshot: True preserves existing rows against older SQLite snapshots.
    Returns:
        Counts; committed earlier batches survive interruption and reruns are idempotent.
    Raises:
        SourceError, sqlalchemy.exc.SQLAlchemyError: Bad provenance or storage failure.
    """
    counts = {"inserted": 0, "updated": 0, "unchanged": 0}
    for offset in range(0, len(rows), 2000):
        with engine.begin() as connection:
            connection.execute(
                sa.text("SELECT pg_advisory_xact_lock(:first,:second)"),
                {"first": LOCK_KEYS[0], "second": LOCK_KEYS[1]},
            )
            batch = upsert_operations(connection, rows[offset : offset + 2000], snapshot=snapshot)
            for key in counts:
                counts[key] += batch[key]
    return counts


def sync_facilities(engine, stop, *, dry_run=False, client=None):
    """Sync independent SOHOA source, including a daily full scan for older deletions/revisions.

    Args:
        engine: Application DB engine; absence of configured facilities skips source calls.
        stop: Worker cancellation event shared with the DCU sync.
        dry_run: Validate source data without writing or advancing full-scan time.
        client: Optional injected source client for tests/operator runs.
    Returns:
        Status/count report. Daily full scan detects old changed/deleted records;
        other cycles reread at least the last 24 hours by observation date.
    Raises:
        SourceError, sqlalchemy.exc.SQLAlchemyError: Failed source/storage; sensor sync
            is handled separately by the worker so this cannot undo DCU commits.
    """
    with engine.connect() as connection:
        facilities = (
            connection.execute(sa.select(Facility.__table__).where(Facility.sync_enabled))
            .mappings()
            .all()
        )
    if not facilities:
        return {"status": "not_configured"}
    owned = client is None
    if owned:
        client = FacilitySourceClient(FacilitySourceSettings.from_env(), stop)
    now = datetime.now(timezone.utc)
    full = any(
        row["last_full_sync_at"] is None or now - row["last_full_sync_at"] >= timedelta(hours=24)
        for row in facilities
    )
    start = date(1900, 1, 1) if full else (now - timedelta(hours=24)).astimezone(VIETNAM).date()
    try:
        metadata = [normalize_facility(row) for row in client.catalog()]
        reports = [
            parse_operation(row) for row in client.operations(start, now.astimezone(VIETNAM).date())
        ]
        allowed = {row["id"] for row in facilities}
        known = {row["id"] for row in metadata}
        if any(row["facility_id"] not in known for row in reports):
            raise SourceError("Operational report has unconfigured facility.")
        reports = [row for row in reports if row["facility_id"] in allowed]
        if dry_run:
            return {"status": "validated", "fetched": len(reports), "full_scan": full}
        with engine.begin() as connection:
            connection.execute(
                sa.text("SELECT pg_advisory_xact_lock(:first,:second)"),
                {"first": LOCK_KEYS[0], "second": LOCK_KEYS[1]},
            )
            upsert_facilities(connection, metadata)
            counts = upsert_operations(connection, reports)
            if full:
                connection.execute(
                    sa.update(Facility)
                    .where(Facility.id.in_(list(allowed)))
                    .values(last_full_sync_at=now)
                )
        return {
            "status": "committed",
            "facilities": len(facilities),
            "fetched": len(reports),
            "full_scan": full,
            **counts,
        }
    finally:
        if owned:
            client.session.close()


def main():
    """Validate by default; --apply imports source metadata and reports explicitly."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--source", type=Path, default=Path("data/hue_hydrology_all.sqlite"))
    parser.add_argument("--report", type=Path, default=Path("reports/facility_import.json"))
    args = parser.parse_args()
    load_dotenv()
    engine = build_engine()
    client = FacilitySourceClient(FacilitySourceSettings.from_env())
    try:
        metadata = [normalize_facility(row) for row in client.catalog()]
        if len({row["id"] for row in metadata}) != len(metadata):
            raise SourceError("Duplicate facility catalog identity.")
        source = sqlite3.connect(args.source.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            source.execute("PRAGMA query_only=ON")
            source.execute("BEGIN")
            sqlite_rows = []
            for key, raw in source.execute(
                "SELECT _key,raw_json FROM reservoir_observations ORDER BY _key"
            ):
                row = parse_operation(json.loads(raw))
                if str(row["source_record_id"]) != key:
                    raise SourceError("SQLite operation source ID mismatch.")
                sqlite_rows.append(row)
        finally:
            source.close()
        cutoff = datetime.now(timezone.utc)
        live_rows = [
            parse_operation(row)
            for row in client.operations(date(1900, 1, 1), cutoff.astimezone(VIETNAM).date())
        ]
        allowed = {row["id"] for row in metadata}
        if any(row["facility_id"] not in allowed for row in sqlite_rows + live_rows):
            raise SourceError("Unknown operational facility mapping.")
        report = {
            "dry_run": not args.apply,
            "cutoff_utc": cutoff.isoformat(),
            "facilities": len(metadata),
            "linked_rivers": sum(row["station_id"] is not None for row in metadata),
            "sqlite_records": len(sqlite_rows),
            "api_records": len(live_rows),
            "api_states": dict(Counter(row["source_status"] or "unknown" for row in live_rows)),
            "api_invalid_time": sum(row["time_status"] == "invalid" for row in live_rows),
        }
        if args.apply:
            with engine.begin() as connection:
                upsert_facilities(connection, metadata)
            report["sqlite_write"] = write_batches(engine, sqlite_rows, snapshot=True)
            report["api_write"] = write_batches(engine, live_rows)
            with engine.begin() as connection:
                connection.execute(
                    sa.update(Facility)
                    .where(Facility.id.in_(list(allowed)))
                    .values(last_full_sync_at=cutoff)
                )
        write_report(args.report, report)
        print(json.dumps(report), flush=True)
    except (SourceError, sa.exc.SQLAlchemyError, sqlite3.Error, ValueError) as error:
        parser.exit(
            1, f"Facility import stopped; error_type={type(error).__name__}; rerun is idempotent\n"
        )
    finally:
        client.session.close()
        engine.dispose()


if __name__ == "__main__":
    main()
