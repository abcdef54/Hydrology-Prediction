"""Validate and persist unconverted DCU wind readings with explicit source identity."""

from datetime import datetime, timezone
import math
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert

from src.backend.db.models import WindObservation

FIELDS = ("ws", "wd", "wsg", "wdg")
VALUE_COLUMNS = (*FIELDS, "source_name", "source_collection", "source_record_id", "quality_status")


def normalize_wind_station(record):
    """Map a wind station code to a namespaced ID and preserve its upstream UUID.

    Args:
        record: Source iot_wind_station metadata with code/id/GeoJSON coordinates.

    Returns:
        Station values: sync enabled, map display disabled pending wind frontend.
        Units remain null until the API/logger encoding has been verified.

    Raises:
        SourceError: Invalid metadata or source UUID.
    """
    from src.backend.ingestion.station_catalog import normalize_station
    from src.backend.ingestion.worker import SourceError

    station = normalize_station(record, "water_level_station")
    try:
        source_id = str(UUID(record["id"]))
    except (KeyError, ValueError, TypeError, AttributeError):
        raise SourceError("Wind station UUID is invalid.") from None
    station.update(
        id="wind:" + record["code"],
        station_type="wind",
        source_station_id=source_id,
        observation_collection="iot_wind_speed",
        measurement_unit=None,
        measurement_units={field: None for field in FIELDS},
        sync_enabled=True,
    )
    return station


def parse_wind_row(record, station, start, end):
    """Validate finite raw readings without inventing a unit, scale or angle.

    Args:
        record: Source ID, sid UUID, aware time_point, and ws/wd/wsg/wdg fields.
        station: Internal station ID with exact upstream UUID mapping.
        start/end: Inclusive UTC bounds of the fixed source query.

    Returns:
        WindObservation values preserving null, zero and the original numeric scale.

    Raises:
        SourceError: Bad identity, timezone, bounds, field types or nonfinite numbers.
    """
    from src.backend.ingestion.worker import SourceError

    try:
        if type(record["id"]) is not int or record["id"] <= 0:
            raise ValueError()
        if record["sid"] != station["source_station_id"]:
            raise ValueError()
        observed = datetime.fromisoformat(record["time_point"])
        if observed.tzinfo is None or not start <= observed <= end:
            raise ValueError()
        values = {}
        for field in FIELDS:
            value = record[field]
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError()
            values[field] = float(value) if value is not None else None
    except (KeyError, TypeError, ValueError):
        raise SourceError("Wind source observation failed validation.") from None
    return {
        "station_id": station["id"],
        "observed_at": observed.astimezone(timezone.utc),
        **values,
        "source_name": station["source_name"],
        "source_collection": "iot_wind_speed",
        "source_record_id": record["id"],
        "quality_status": "observed" if any(v is not None for v in values.values()) else "missing",
    }


def upsert_wind_observations(connection, rows, *, preserve_existing=False):
    """Save raw wind batches; SQLite preserves existing DB values, API updates revisions.

    Args:
        connection: Connection in caller-owned transaction.
        rows: Validated rows of one wind station.
        preserve_existing: True for older SQLite snapshots; False for live API.

    Returns:
        Inserted/updated/unchanged counts. received_at is never overwritten.

    Raises:
        sqlalchemy.exc.SQLAlchemyError: Constraint/storage failure; caller rolls back.
    """
    counts = {"inserted": 0, "updated": 0, "unchanged": 0}
    for offset in range(0, len(rows), 500):
        batch = rows[offset : offset + 500]
        existing = {
            row["observed_at"]: row
            for row in connection.execute(
                sa.select(WindObservation.__table__).where(
                    WindObservation.station_id == batch[0]["station_id"],
                    WindObservation.observed_at.in_([row["observed_at"] for row in batch]),
                )
            ).mappings()
        }
        statement = insert(WindObservation).values(batch)
        if preserve_existing:
            statement = statement.on_conflict_do_nothing(
                index_elements=[WindObservation.station_id, WindObservation.observed_at]
            )
        else:
            statement = statement.on_conflict_do_update(
                index_elements=[WindObservation.station_id, WindObservation.observed_at],
                set_={column: statement.excluded[column] for column in VALUE_COLUMNS},
                where=sa.or_(
                    *(
                        WindObservation.__table__.c[column].is_distinct_from(
                            statement.excluded[column]
                        )
                        for column in VALUE_COLUMNS
                    )
                ),
            )
        changed = set(connection.scalars(statement.returning(WindObservation.observed_at)))
        counts["inserted"] += sum(at not in existing for at in changed)
        counts["updated"] += sum(at in existing for at in changed)
        counts["unchanged"] += len(batch) - len(changed)
    return counts
