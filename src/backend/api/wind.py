"""Public raw wind payloads; never infer units, compass headings or Beaufort levels."""

from datetime import timezone
from typing import Literal

from fastapi import HTTPException
from pydantic import BaseModel, Field
import sqlalchemy as sa

from src.backend.api.schemas import (
    HistoricalObservation,
    LatestObservationResponse,
    ObservationHistoryResponse,
)
from src.backend.db.models import WindObservation
from src.backend.ingestion.wind import FIELDS


class WindMeasurement(BaseModel):
    value: float | None = Field(allow_inf_nan=False)
    unit: str | None
    unit_confirmed: bool


class WindLatestResponse(LatestObservationResponse):
    station_type: Literal["wind"] = "wind"
    value: float | None = Field(
        allow_inf_nan=False, description="Primary raw ws; see measurements for all four fields."
    )
    measurements: dict[str, WindMeasurement]


class WindHistoricalObservation(HistoricalObservation):
    measurements: dict[str, WindMeasurement]


class WindHistoryResponse(ObservationHistoryResponse):
    station_type: Literal["wind"] = "wind"
    observations: list[WindHistoricalObservation]


def measurements(row, station):
    """Return original field values and explicit per-field unit confirmation."""
    units = station.get("measurement_units") or {}
    return {
        field: WindMeasurement(
            value=row[field], unit=units.get(field), unit_confirmed=units.get(field) is not None
        )
        for field in FIELDS
    }


def latest_wind(connection, station_id, station, now, stale_after):
    """Read latest nonmissing wind record, without caching or converting source fields.

    Args:
        connection: Caller-owned read connection.
        station_id: Namespaced internal ID, distinct from rain/water codes.
        station: Metadata supplying nullable per-field units.
        now: Current aware request time, excluding future observations.
        stale_after: Freshness threshold in minutes.

    Returns:
        Primary ws plus all four readings, units and recalculated freshness.

    Raises:
        HTTPException: 404 when no usable observation exists.
        sqlalchemy.exc.SQLAlchemyError: Database read failure.
    """
    row = (
        connection.execute(
            sa.select(WindObservation.__table__)
            .where(
                WindObservation.station_id == station_id,
                WindObservation.observed_at <= now,
                WindObservation.quality_status == "observed",
            )
            .order_by(WindObservation.observed_at.desc())
            .limit(1)
        )
        .mappings()
        .first()
    )
    if row is None:
        raise HTTPException(
            404,
            {"code": "OBSERVATION_NOT_FOUND", "message": "Station has no valid wind observation."},
        )
    age = max(0, (now - row["observed_at"]).total_seconds() / 60)
    return WindLatestResponse(
        station_id=station_id,
        value=row["ws"],
        unit=(station.get("measurement_units") or {}).get("ws"),
        observed_at=row["observed_at"],
        received_at=row["received_at"],
        quality_status=row["quality_status"],
        data_age_minutes=age,
        is_stale=age > stale_after,
        stale_after_minutes=stale_after,
        measurements=measurements(row, station),
    )


def wind_history(connection, station_id, station, query, start, end):
    """Page original wind readings in [start,end), preserving partial/null fields.

    Args:
        connection: Caller-owned read connection.
        station_id: Internal wind ID including namespace.
        station: Nullable unit metadata, never inferred from numeric ranges.
        query: Validated limit and exclusive before cursor.
        start/end: Aware UTC bounds validated by the shared observation route.

    Returns:
        Newest-first page with all four raw fields and a nonrepeating cursor.

    Raises:
        sqlalchemy.exc.SQLAlchemyError: Database read failure.
    """
    statement = sa.select(WindObservation.__table__).where(
        WindObservation.station_id == station_id,
        WindObservation.observed_at >= start,
        WindObservation.observed_at < end,
    )
    if query.before is not None:
        statement = statement.where(WindObservation.observed_at < query.before)
    rows = (
        connection.execute(
            statement.order_by(WindObservation.observed_at.desc()).limit(query.limit + 1)
        )
        .mappings()
        .all()
    )
    has_more = len(rows) > query.limit
    page = [
        WindHistoricalObservation(
            value=row["ws"],
            observed_at=row["observed_at"].astimezone(timezone.utc),
            received_at=row["received_at"],
            quality_status=row["quality_status"],
            measurements=measurements(row, station),
        )
        for row in rows[: query.limit]
    ]
    return WindHistoryResponse(
        station_id=station_id,
        unit=(station.get("measurement_units") or {}).get("ws"),
        start=start,
        end=end,
        observations=page,
        has_more=has_more,
        next_before=page[-1].observed_at if has_more else None,
    )
