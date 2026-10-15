"""Read SOHOA infrastructure and reports without mixing them with model observations."""

from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field
import sqlalchemy as sa
from src.backend.db.models import Facility, FacilityOperation
from src.backend.api.schemas import ErrorResponse
from src.backend.ingestion.facilities import METRICS

router = APIRouter(prefix="/api/v1", tags=["Facilities"])


class FacilitySummary(BaseModel):
    facility_id: str
    name: str
    kind: Literal["hydropower", "reservoir", "dam", "river"]
    kind_label: str
    station_id: str | None
    latitude: float | None
    longitude: float | None
    specifications: dict[str, str | None]
    measurement_units: dict[str, str | None]


class FacilityList(BaseModel):
    facilities: list[FacilitySummary]


class RawMetric(BaseModel):
    value: float | None = Field(allow_inf_nan=False)
    unit: str | None
    unit_confirmed: bool


class OperationalReport(BaseModel):
    source_record_id: int
    facility_id: str
    function_code: str
    observed_at: datetime | None
    time_status: Literal["valid", "invalid"]
    observation_date_text: str
    observation_hour_text: str
    source_updated_at: datetime | None
    source_updated_text: str
    source_status: str | None
    source_status_label: str | None
    is_deleted: bool
    is_visible: bool
    measurements: dict[str, RawMetric]
    gate_opening_text: str | None
    operation_start_text: str | None
    operation_end_text: str | None
    received_at: datetime


class OperationalHistory(BaseModel):
    facility_id: str
    operations: list[OperationalReport]
    has_more: bool
    next_before_id: int | None


class LatestOperation(BaseModel):
    facility_id: str
    operation: OperationalReport
    data_age_minutes: float
    stale_after_minutes: float
    is_stale: bool


def summary(row):
    return FacilitySummary(
        facility_id=row["id"],
        **{key: row[key] for key in FacilitySummary.model_fields if key != "facility_id"},
    )


def public_report(row, facility):
    """Expose only operational fields; unit confirmation comes from curated metadata."""
    values = {
        key: row[key]
        for key in OperationalReport.model_fields
        if key not in ("measurements", "is_deleted", "is_visible")
    }
    units = facility["measurement_units"]
    return OperationalReport(
        **values,
        is_deleted=row["source_status"] == "2",
        is_visible=row["source_status"] == "1",
        measurements={
            field: RawMetric(
                value=row[field], unit=units.get(field), unit_confirmed=units.get(field) is not None
            )
            for field in METRICS
        },
    )


def find_facility(connection, facility_id):
    row = (
        connection.execute(sa.select(Facility.__table__).where(Facility.id == facility_id))
        .mappings()
        .first()
    )
    if row is None:
        raise HTTPException(
            404, {"code": "FACILITY_NOT_FOUND", "message": "Facility does not exist."}
        )
    return row


@router.get("/facilities", response_model=FacilityList, responses={503: {"model": ErrorResponse}})
def list_facilities(request: Request, include_linked: bool = False, station_id: str | None = None):
    """List infrastructure; linked rivers are omitted by default to avoid duplicate markers.

    Args:
        request: App DB engine; no upstream calls or model loads.
        include_linked: Include river facilities linked to existing DCU stations.
        station_id: Restrict to links of one exact station, including leading zeros.
    Returns:
        Metadata with original specification text and nullable per-field units.
    Raises:
        sqlalchemy.exc.SQLAlchemyError: Database failure handled as 503 by the app.
    """
    statement = sa.select(Facility.__table__)
    if station_id is not None:
        statement = statement.where(Facility.station_id == station_id)
    elif not include_linked:
        statement = statement.where(Facility.station_id.is_(None))
    with request.app.state.database_engine.connect() as connection:
        rows = connection.execute(statement.order_by(Facility.source_id)).mappings().all()
    return FacilityList(facilities=[summary(row) for row in rows])


@router.get(
    "/facilities/{facility_id}",
    response_model=FacilitySummary,
    responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def get_facility(request: Request, facility_id: str):
    with request.app.state.database_engine.connect() as connection:
        return summary(find_facility(connection, facility_id))


@router.get(
    "/facilities/{facility_id}/latest",
    response_model=LatestOperation,
    responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def get_latest_operation(request: Request, facility_id: str):
    """Use visible reports with valid observation time; skip deleted/unknown/future rows.

    Args:
        request: Database and shared freshness threshold from app state.
        facility_id: Exact namespaced infrastructure ID.
    Returns:
        Latest report and freshness from observed_at, never source_updated_at.
    Raises:
        HTTPException: 404 for unknown facility or no usable report.
        sqlalchemy.exc.SQLAlchemyError: Database failure handled as 503.
    """
    now = datetime.now(timezone.utc)
    with request.app.state.database_engine.connect() as connection:
        facility = find_facility(connection, facility_id)
        row = (
            connection.execute(
                sa.select(FacilityOperation.__table__)
                .where(
                    FacilityOperation.facility_id == facility_id,
                    FacilityOperation.source_status == "1",
                    FacilityOperation.time_status == "valid",
                    FacilityOperation.observed_at <= now,
                    sa.or_(
                        *(getattr(FacilityOperation, field).is_not(None) for field in METRICS),
                        FacilityOperation.gate_opening_text.is_not(None),
                    ),
                )
                .order_by(
                    FacilityOperation.observed_at.desc(),
                    FacilityOperation.source_updated_at.desc().nulls_last(),
                    FacilityOperation.source_record_id.desc(),
                )
                .limit(1)
            )
            .mappings()
            .first()
        )
    if row is None:
        raise HTTPException(
            404,
            {
                "code": "OPERATION_NOT_FOUND",
                "message": "Facility has no usable operational report.",
            },
        )
    age = max(0, (now - row["observed_at"]).total_seconds() / 60)
    threshold = request.app.state.stale_after_minutes
    return LatestOperation(
        facility_id=facility_id,
        operation=public_report(row, facility),
        data_age_minutes=age,
        stale_after_minutes=threshold,
        is_stale=age > threshold,
    )


@router.get(
    "/facilities/{facility_id}/operations",
    response_model=OperationalHistory,
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def get_operations(
    request: Request,
    facility_id: str,
    start: datetime | None = None,
    end: datetime | None = None,
    before_id: int | None = Query(default=None, gt=0),
    limit: int = Query(default=100, ge=1, le=1000),
    include_inactive: bool = False,
    include_invalid_time: bool = False,
):
    """Page by unique source ID, retaining same-time reports and optional audit records.

    Args:
        request: Caller app database, no model or upstream access.
        facility_id: Exact SOHOA facility identity.
        start/end: Optional aware [start,end) observation bounds, end not future.
        before_id: Exclusive source-ID cursor; results ordered by descending source ID.
        limit: Maximum 1000 reports, using one extra row to determine next cursor.
        include_inactive: Include deleted and unknown-status records for audit.
        include_invalid_time: Include NULL observation times when no bounds are used.
    Returns:
        Raw metrics/text/status. Default excludes deleted, unknown and invalid time.
    Raises:
        HTTPException: 404 unknown facility, 422 invalid/unaware/future bounds.
        sqlalchemy.exc.SQLAlchemyError: Database failure handled as 503.
    """
    now = datetime.now(timezone.utc)
    if (
        any(bound is not None and bound.tzinfo is None for bound in (start, end))
        or (start and end and start >= end)
        or (end and end > now)
    ):
        raise HTTPException(
            422,
            {
                "code": "INVALID_REQUEST",
                "message": "Require aware ordered bounds; end must not be future.",
            },
        )
    statement = sa.select(FacilityOperation.__table__).where(
        FacilityOperation.facility_id == facility_id
    )
    if not include_inactive:
        statement = statement.where(FacilityOperation.source_status == "1")
    if include_invalid_time and start is None and end is None:
        statement = statement.where(
            sa.or_(FacilityOperation.observed_at.is_(None), FacilityOperation.observed_at <= now)
        )
    else:
        statement = statement.where(
            FacilityOperation.time_status == "valid", FacilityOperation.observed_at <= now
        )
    if start:
        statement = statement.where(FacilityOperation.observed_at >= start)
    if end:
        statement = statement.where(FacilityOperation.observed_at < end)
    if before_id:
        statement = statement.where(FacilityOperation.source_record_id < before_id)
    with request.app.state.database_engine.connect() as connection:
        facility = find_facility(connection, facility_id)
        rows = (
            connection.execute(
                statement.order_by(FacilityOperation.source_record_id.desc()).limit(limit + 1)
            )
            .mappings()
            .all()
        )
    page = rows[:limit]
    more = len(rows) > limit
    return OperationalHistory(
        facility_id=facility_id,
        operations=[public_report(row, facility) for row in page],
        has_more=more,
        next_before_id=page[-1]["source_record_id"] if more else None,
    )
