"""Sync selected Hue observations to PostgreSQL every 11 minutes.

Run --once to sync immediately, --once --dry-run to inspect source data without
writing observations, or no flags for the scheduler. Secrets come only from env.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import signal
from threading import Event, Thread
import time

from dotenv import load_dotenv
import requests
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert

from src.backend.db.models import Observation, Station, WindObservation
from src.backend.db.session import build_engine
from src.backend.cache.latest import LatestObservationCache
from src.backend.ingestion.weather_live import sync_weather_forecasts

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SOURCE_COLLECTIONS = {"water_level_depth", "rain_water_depth", "iot_wind_speed"}
VALUE_COLUMNS = ("depth", "source_name", "source_collection", "source_record_id", "quality_status")
# Match the SQLite importer's lock so neither observation writer runs over the other.
LOCK_KEYS = (20261012, 2)


class SourceError(RuntimeError):
    """Sanitized source failure, optionally carrying an HTTP status for recovery.

    Args:
        message: Public explanation containing no credentials or response body.
        status_code: HTTP status when the failure comes from a source response.
    """

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class SyncCancelled(Exception):
    """Stop requested; committed previous cycles remain intact."""


@dataclass(frozen=True)
class SyncSettings:
    """Worker settings, with secrets excluded from repr and report output.

    Args:
        base_url: Root URL of the Hue Directus API from local/server environment.
        email: Source account, used only for authentication.
        password: Source password, never logged or saved to a report.
        interval_minutes: Start-to-start cycle spacing; default 11 minutes.
        overlap_hours: How far before each station's latest timestamp to reread.
        page_size: Requested page size, not an assumed upstream maximum.
    """

    base_url: str
    email: str = field(repr=False)
    password: str = field(repr=False)
    interval_minutes: float = 11
    overlap_hours: float = 24
    page_size: int = 500

    @classmethod
    def from_env(cls) -> SyncSettings:
        """Load credentials and positive scheduling settings from environment.

        Returns:
            Validated settings. No network or database access is performed.

        Raises:
            ValueError: If required credentials, URL or numeric settings are invalid.
        """
        required = ["HYDROLOGY_API_BASE_URL", "HYDROLOGY_API_EMAIL", "HYDROLOGY_API_PASSWORD"]
        if any(not os.environ.get(key) for key in required):
            raise ValueError("Configure HYDROLOGY_API_BASE_URL, EMAIL and PASSWORD in .env.")
        settings = cls(
            os.environ[required[0]].rstrip("/"),
            os.environ[required[1]],
            os.environ[required[2]],
            float(os.environ.get("HYDROLOGY_SYNC_INTERVAL_MINUTES", "11")),
            float(os.environ.get("HYDROLOGY_SYNC_OVERLAP_HOURS", "24")),
        )
        if not settings.base_url.startswith(("https://", "http://")):
            raise ValueError("Source URL must use HTTP or HTTPS.")
        for value in (settings.interval_minutes, settings.overlap_hours):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Interval and overlap must be positive finite numbers.")
        return settings


def parse_source_row(record: dict, station: dict, start: datetime, end: datetime) -> dict:
    """Validate one source observation without changing units or filling missing data.

    Args:
        record: JSON object containing id, station_id, time_point and depth.
        station: PostgreSQL metadata mapping internal ID to external source identity.
        start: Inclusive UTC query bound.
        end: Inclusive UTC query bound, fixed at the cycle's start.

    Returns:
        Observation values ready for SQLAlchemy; depth=None is marked missing.

    Raises:
        SourceError: If schema, source identity, time bounds or numeric values fail.
    """
    if station["observation_collection"] == "iot_wind_speed":
        from src.backend.ingestion.wind import parse_wind_row

        return parse_wind_row(record, station, start, end)
    try:
        if not {"id", "station_id", "time_point", "depth"}.issubset(record):
            raise ValueError()
        if type(record["id"]) is not int or record["id"] <= 0:
            raise ValueError()
        if str(record["station_id"]) != station["source_station_id"]:
            raise ValueError()
        observed_at = datetime.fromisoformat(record["time_point"])
        if observed_at.tzinfo is None:
            raise ValueError()
        observed_at = observed_at.astimezone(timezone.utc)
        if not start <= observed_at <= end:
            raise ValueError()
        depth = record["depth"]
        if depth is not None:
            if (
                isinstance(depth, bool)
                or not isinstance(depth, (int, float))
                or not math.isfinite(depth)
            ):
                raise ValueError()
            depth = float(depth)
    except (ValueError, TypeError, KeyError, OverflowError):
        raise SourceError(
            "Source observation failed schema, station, timestamp or value validation."
        ) from None
    return {
        "station_id": station["id"],
        "observed_at": observed_at,
        "depth": depth,
        "source_name": station["source_name"],
        "source_collection": station["observation_collection"],
        "source_record_id": record["id"],
        "quality_status": "missing" if depth is None else "observed",
    }


class HueSourceClient:
    """Authenticate and page observations with bounded retries and cancellation.

    Args:
        settings: Source credentials and page size.
        stop: Event set by SIGTERM/SIGINT to stop requests and retry waits.
        session: Optional HTTP session for focused tests; normal calls use requests.

    Tokens stay in memory. A fresh login is made each cycle, with one renewal on
    HTTP 401. Transport errors, 429 and selected 5xx statuses get at most 3 attempts.
    """

    def __init__(self, settings: SyncSettings, stop: Event, session=None):
        self.settings = settings
        self.stop = stop
        self.session = session if session is not None else requests.Session()

    def request(self, method: str, path: str, **kwargs) -> dict:
        """Request JSON with connect/read timeouts; never include source bodies in errors.

        Args:
            method: HTTP method, limited by callers to authentication POST and GET.
            path: Fixed API path beneath the configured source root.
            kwargs: Query or JSON body, possibly containing authentication secrets.

        Returns:
            A JSON object from a successful HTTP 200 response.

        Raises:
            SourceError: After bounded retries, invalid JSON or non-success HTTP.
            SyncCancelled: If a stop request arrives before a request/retry.
        """
        for attempt in range(3):
            if self.stop.is_set():
                raise SyncCancelled()
            try:
                response = self.session.request(
                    method,
                    self.settings.base_url + path,
                    timeout=(10, 30),
                    allow_redirects=False,
                    **kwargs,
                )
            except requests.RequestException:
                if attempt == 2:
                    raise SourceError("Source transport failed after 3 attempts.") from None
                if self.stop.wait(2**attempt):
                    raise SyncCancelled()
                continue
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                delay = 2**attempt
                try:
                    delay = min(30, max(delay, float(response.headers.get("Retry-After", delay))))
                except ValueError:
                    pass
                if self.stop.wait(delay):
                    raise SyncCancelled()
                continue
            if response.status_code != 200:
                raise SourceError(
                    f"Source returned HTTP {response.status_code}.", response.status_code
                )
            try:
                payload = response.json()
            except ValueError:
                raise SourceError("Source returned invalid JSON.") from None
            if not isinstance(payload, dict):
                raise SourceError("Source returned an unexpected JSON shape.")
            return payload
        raise SourceError("Source request failed.")

    def login(self) -> None:
        """Authenticate this cycle, replacing the in-memory access token.

        Returns:
            None; the session Authorization header is updated on success.

        Raises:
            SourceError: If authentication fails or no access token is returned.
        """
        self.session.headers.pop("Authorization", None)
        payload = self.request(
            "POST",
            "/auth/login",
            json={"email": self.settings.email, "password": self.settings.password},
        )
        data = payload.get("data")
        if (
            not isinstance(data, dict)
            or not isinstance(data.get("access_token"), str)
            or not data["access_token"]
        ):
            raise SourceError("Source login did not return an access token.")
        self.session.headers["Authorization"] = "Bearer " + data["access_token"]

    def observations(self, station: dict, start: datetime, end: datetime) -> list[dict]:
        """Fetch a station's bounded range using ascending integer ID pagination.

        Args:
            station: Validated database mapping for a supported source collection.
            start: Inclusive UTC lower bound; includes the configured overlap.
            end: Fixed inclusive UTC upper bound for this cycle.

        Returns:
            Validated observations, with no duplicate station/timestamp values.
            ID cursors avoid skipping records sharing a timestamp at page boundaries.
            Pagination continues to an empty page, even if the source caps page size.

        Raises:
            SourceError: For invalid rows, ambiguous duplicate timestamps, stalled
                cursors or more than 1000 pages. No observations are written here.
        """
        if station["observation_collection"] not in SOURCE_COLLECTIONS:
            raise SourceError("Unsupported observation source collection.")
        cursor = 0
        records = []
        timestamps = set()
        source_station_field = (
            "sid" if station["observation_collection"] == "iot_wind_speed" else "station_id"
        )
        for _ in range(1000):
            params = {
                f"filter[{source_station_field}][_in]": station["source_station_id"],
                "filter[time_point][_gte]": start.isoformat(),
                "filter[time_point][_lte]": end.isoformat(),
                "filter[id][_gt]": cursor,
                "sort[]": "id",
                "limit": self.settings.page_size,
            }
            try:
                payload = self.request(
                    "GET", "/items/" + station["observation_collection"], params=params
                )
            except SourceError as error:
                if error.status_code != 401:
                    raise
                self.login()
                payload = self.request(
                    "GET", "/items/" + station["observation_collection"], params=params
                )
            rows = payload.get("data")
            if not isinstance(rows, list):
                raise SourceError("Source observation data is not a list.")
            if not rows:
                return records
            for row in rows:
                if not isinstance(row, dict):
                    raise SourceError("Source observation is not an object.")
                observation = parse_source_row(row, station, start, end)
                source_id = observation["source_record_id"]
                if source_id <= cursor:
                    raise SourceError("Source ID pagination did not advance in ascending order.")
                cursor = source_id
                if observation["observed_at"] in timestamps:
                    raise SourceError(
                        "Multiple source records share a station/timestamp; refusing ambiguous overwrite."
                    )
                timestamps.add(observation["observed_at"])
                records.append(observation)
        raise SourceError("Source pagination exceeded 1000 pages.")


def upsert_observations(connection, rows: list[dict]) -> dict:
    """Save one station in batches, updating only changed source values.

    Args:
        connection: Connection inside the caller's observation-write transaction.
        rows: Validated original-unit observations for one station.

    Returns:
        Inserted, updated and unchanged counts. received_at remains the original
        insertion time; unchanged rows do not fire the updated_at trigger.

    Raises:
        sqlalchemy.exc.SQLAlchemyError: On write errors; caller must roll back.
    """
    counts = {"inserted": 0, "updated": 0, "unchanged": 0}
    for offset in range(0, len(rows), 500):
        batch = rows[offset : offset + 500]
        existing = {
            row["observed_at"]: row
            for row in connection.execute(
                sa.select(Observation.__table__).where(
                    Observation.station_id == batch[0]["station_id"],
                    Observation.observed_at.in_([row["observed_at"] for row in batch]),
                )
            ).mappings()
        }
        statement = insert(Observation).values(batch)
        statement = statement.on_conflict_do_update(
            index_elements=[Observation.station_id, Observation.observed_at],
            set_={column: statement.excluded[column] for column in VALUE_COLUMNS},
            where=sa.or_(
                *(
                    Observation.__table__.c[column].is_distinct_from(statement.excluded[column])
                    for column in VALUE_COLUMNS
                )
            ),
        ).returning(Observation.observed_at)
        changed = set(connection.scalars(statement))
        counts["inserted"] += sum(timestamp not in existing for timestamp in changed)
        counts["updated"] += sum(timestamp in existing for timestamp in changed)
        counts["unchanged"] += len(batch) - len(changed)
    return counts


def write_report(path: Path, report: dict) -> None:
    """Atomically replace the local status file; report must contain no secrets.

    Args:
        path: Writable JSON status path outside the source package.
        report: Public counters, timestamps and sanitized status information.

    Returns:
        None. Creates the report parent if needed, then replaces the file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def sync_once(
    engine,
    client: HueSourceClient,
    dry_run: bool = False,
    latest_cache: LatestObservationCache | None = None,
) -> dict:
    """Download all enabled mapped sources, then commit all station changes together.

    Args:
        engine: PostgreSQL engine; the same session holds the shared importer lock.
        client: Source client with credentials, cancellation event and overlap policy.
        dry_run: If True, validate/download only without writing observations.
        latest_cache: Optional Redis cache to invalidate for changed stations after
            the whole transaction commits. Cache failure never rolls back observations.

    Returns:
        Public per-station counts and watermarks. A locked writer returns skipped.
        A new station starts at now-overlap; use a separate backfill for older history.

    Raises:
        SourceError: If any source station fetch or validation fails, before writes.
        sqlalchemy.exc.SQLAlchemyError: On DB failures; the station group rolls back.
        SyncCancelled: If cancellation arrives while reading source data.
    """
    now = datetime.now(timezone.utc)
    report = {
        "started_at_utc": now.isoformat(),
        "status": "running",
        "dry_run": dry_run,
        "stations": [],
    }
    with engine.connect() as connection:
        locked = connection.scalar(
            sa.text("SELECT pg_try_advisory_lock(:first, :second)"),
            {"first": LOCK_KEYS[0], "second": LOCK_KEYS[1]},
        )
        connection.commit()
        if not locked:
            return {**report, "status": "skipped_writer_busy"}
        try:
            with connection.begin():
                stations = (
                    connection.execute(
                        sa.select(Station.__table__)
                        .where(
                            sa.or_(Station.enabled.is_(True), Station.sync_enabled.is_(True)),
                            Station.source_name == "hue_dcu",
                            Station.observation_collection.in_(SOURCE_COLLECTIONS),
                        )
                        .order_by(Station.id)
                    )
                    .mappings()
                    .all()
                )
                watermarks = dict(
                    connection.execute(
                        sa.select(Observation.station_id, sa.func.max(Observation.observed_at))
                        .where(Observation.station_id.in_([s["id"] for s in stations]))
                        .group_by(Observation.station_id)
                    ).all()
                )
                if any(
                    station["observation_collection"] == "iot_wind_speed" for station in stations
                ):
                    watermarks.update(
                        connection.execute(
                            sa.select(
                                WindObservation.station_id, sa.func.max(WindObservation.observed_at)
                            )
                            .where(WindObservation.station_id.in_([s["id"] for s in stations]))
                            .group_by(WindObservation.station_id)
                        ).all()
                    )
            if not stations:
                raise SourceError("No enabled source stations are configured in PostgreSQL.")
            client.login()
            downloaded = []
            for station in stations:
                latest = watermarks.get(station["id"])
                start = (latest or now) - timedelta(hours=client.settings.overlap_hours)
                rows = client.observations(station, start, now)
                downloaded.append(rows)
                report["stations"].append(
                    {
                        "station_id": station["id"],
                        "fetched": len(rows),
                        "from_utc": start.isoformat(),
                        "to_utc": now.isoformat(),
                        "latest_source_utc": (
                            max((row["observed_at"] for row in rows), default=None).isoformat()
                            if rows
                            else None
                        ),
                        "missing": sum(row["quality_status"] == "missing" for row in rows),
                    }
                )
            if client.stop.is_set():
                raise SyncCancelled()
            if not dry_run:
                with connection.begin():
                    for station, rows, summary in zip(
                        stations, downloaded, report["stations"], strict=True
                    ):
                        if station["observation_collection"] == "iot_wind_speed":
                            from src.backend.ingestion.wind import upsert_wind_observations

                            summary.update(upsert_wind_observations(connection, rows))
                        else:
                            summary.update(upsert_observations(connection, rows))
                # Invalidate only after commit; older readers cannot publish across
                # this cache version change. Redis remains optional for successful sync.
                if latest_cache is not None:
                    for summary in report["stations"]:
                        if summary["inserted"] + summary["updated"] == 0:
                            continue
                        invalidated = latest_cache.invalidate(summary["station_id"])
                        summary["latest_cache_invalidation"] = (
                            "disabled"
                            if not latest_cache.enabled
                            else "ok" if invalidated else "failed"
                        )
            report["status"] = "validated" if dry_run else "committed"
            report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            return report
        finally:
            if connection.in_transaction():
                connection.rollback()
            if not connection.invalidated:
                connection.execute(
                    sa.text("SELECT pg_advisory_unlock(:first, :second)"),
                    {"first": LOCK_KEYS[0], "second": LOCK_KEYS[1]},
                )
                connection.commit()


def wait_for_next_cycle(stop: Event, wake: Event, next_run_at: datetime) -> None:
    """Wait until the wall-clock deadline or an API/resume wake request.

    Args:
        stop: Shutdown event; signal handlers also set wake to interrupt this wait.
        wake: SIGCONT/API request for a new cycle, consumed once.
        next_run_at: Aware UTC deadline for the next start-to-start cycle.

    Returns:
        None when due, woken or stopped. Wall time is rechecked at least every five
        seconds so machine/WSL suspension does not extend an expired deadline.
        A wake during an active sync queues one subsequent cycle, never overlaps it.
    """
    while not stop.is_set():
        remaining = (next_run_at - datetime.now(timezone.utc)).total_seconds()
        if remaining <= 0:
            wake.clear()
            return
        if wake.wait(min(5.0, remaining)):
            wake.clear()
            return


def watch_api_parent(parent_pid: int, stop: Event, wake: Event) -> None:
    """Stop an API-owned worker when its spawning process disappears.

    Args:
        parent_pid: API PID passed through environment by the local process manager.
        stop: Cancellation event used by the sync/source client.
        wake: Interrupt scheduler waiting so shutdown can proceed.

    Returns:
        None after normal shutdown or parent reparenting. Checked each second;
        standalone/systemd workers do not start this watcher.
    """
    while not stop.is_set():
        if os.getppid() != parent_pid:
            stop.set()
            wake.set()
            return
        stop.wait(1.0)


def main() -> int:
    """Run once or schedule start-to-start cycles without overlapping workers.

    Returns:
        Zero after successful one-shot/normal shutdown; one for one-shot failures.
        Long-running failures are logged safely and retried at the next 11-minute slot.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--report", type=Path, help="Status JSON path; overrides HYDROLOGY_SYNC_REPORT_PATH."
    )
    arguments = parser.parse_args()
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    report_path = arguments.report or Path(
        os.environ.get(
            "HYDROLOGY_SYNC_REPORT_PATH", PROJECT_ROOT / "reports/hydrology_sync_status.json"
        )
    )
    settings = SyncSettings.from_env()
    latest_cache = LatestObservationCache.from_env()
    stop = Event()
    wake = Event()

    def request_stop(*_):
        stop.set()
        wake.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, request_stop)
    signal.signal(signal.SIGCONT, lambda *_: wake.set())
    parent_pid = os.environ.get("HYDROLOGY_SYNC_PARENT_PID")
    if parent_pid:
        Thread(target=watch_api_parent, args=(int(parent_pid), stop, wake), daemon=True).start()
    engine = build_engine()
    client = HueSourceClient(settings, stop)
    last_failed = False
    try:
        while not stop.is_set():
            began = time.monotonic()
            started_at = datetime.now(timezone.utc)
            try:
                report = sync_once(engine, client, arguments.dry_run, latest_cache)
                last_failed = False
                if report["status"] in ("committed", "validated"):
                    from src.backend.ingestion.facility_import import sync_facilities

                    try:
                        report["facilities"] = sync_facilities(
                            engine, stop, dry_run=arguments.dry_run
                        )
                    except (SourceError, sa.exc.SQLAlchemyError) as error:
                        last_failed = True
                        report["facilities"] = {
                            "status": "failed",
                            "error_type": type(error).__name__,
                        }
            except SyncCancelled:
                break
            except (SourceError, sa.exc.SQLAlchemyError) as error:
                last_failed = True
                report = {
                    "status": "failed",
                    "started_at_utc": started_at.isoformat(),
                    "error_type": type(error).__name__,
                    "message": (
                        str(error)
                        if isinstance(error, SourceError)
                        else "Database sync failed; no credentials logged."
                    ),
                }
            if stop.is_set():
                break
            try:
                report["weather"] = sync_weather_forecasts(
                    engine, stop, dry_run=arguments.dry_run
                )
                if report["weather"]["status"] in {"partial", "failed"}:
                    last_failed = True
            except (ValueError, OSError, sa.exc.SQLAlchemyError) as error:
                last_failed = True
                report["weather"] = {"status": "failed", "error_type": type(error).__name__}
            report["interval_minutes"] = settings.interval_minutes
            report["overlap_hours"] = settings.overlap_hours
            report["elapsed_seconds"] = round(time.monotonic() - began, 2)
            next_run_at = started_at + timedelta(minutes=settings.interval_minutes)
            report["next_run_at_utc"] = next_run_at.isoformat() if not arguments.once else None
            write_report(report_path, report)
            print(json.dumps(report, ensure_ascii=False), flush=True)
            if arguments.once:
                break
            wait_for_next_cycle(stop, wake, next_run_at)
    finally:
        client.session.close()
        engine.dispose()
        latest_cache.close()
    return 1 if last_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
