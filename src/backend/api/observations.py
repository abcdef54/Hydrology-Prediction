"""Read station observations for display, independently of prediction inputs."""

from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request
import sqlalchemy as sa

from src.backend.api.schemas import (
    ErrorResponse,
    HistoricalObservation,
    LatestObservationResponse,
    ObservationHistoryQuery,
    ObservationHistoryResponse,
)
from src.backend.db.models import Observation, Station

router = APIRouter(prefix="/api/v1", tags=["Observations"])


@router.get(
    "/stations/{station_id}/latest",
    response_model=LatestObservationResponse,
    responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def get_latest_observation(request: Request, station_id: str) -> LatestObservationResponse:
    """Return the latest nonmissing observation at or before the current UTC time.

    Values retain source units and sampling intervals. This is not the hourly
    value prepared for a model. Disabled stations remain readable. Data older
    than the configured threshold is returned with is_stale=true. Unknown
    stations and stations without valid measurements have distinct 404 codes.

    Args:
        request: HTTP request providing the database engine, optional latest cache
            and freshness threshold. Station metadata is always read from PostgreSQL.
        station_id: Exact ID of the water or rain station to inspect.

    Returns:
        Latest valid raw value, source unit, observation/receipt times and data age.
        Supported water/rain stations use mm; unknown units are None. Freshness is based on
        observed_at, not the time the record was imported into PostgreSQL.

    Raises:
        HTTPException: 404 with STATION_NOT_FOUND for an unknown station, or
            OBSERVATION_NOT_FOUND if no valid observation exists at or before now.
        sqlalchemy.exc.SQLAlchemyError: Database failures handled as 503 by the app.
    """
    now = datetime.now(timezone.utc)
    station_query = sa.select(Station.station_type, Station.measurement_unit).where(
        Station.id == station_id
    )
    observation_query = (
        sa.select(
            Observation.depth,
            Observation.observed_at,
            Observation.received_at,
            Observation.quality_status,
        )
        .where(
            Observation.station_id == station_id,
            Observation.observed_at <= now,
            Observation.quality_status == "observed",
            Observation.depth.is_not(None),
        )
        .order_by(Observation.observed_at.desc())
        .limit(1)
    )
    with request.app.state.database_engine.connect() as connection:
        station = connection.execute(station_query).mappings().first()
        if station is None:
            raise HTTPException(
                status_code=404,
                detail={"code": "STATION_NOT_FOUND", "message": "Station does not exist."},
            )
        cache = request.app.state.latest_observation_cache
        cache_version, cached = cache.read(station_id, now)
        if cached is not None:
            observation = cached.model_dump(exclude={"station_id"})
        else:
            observation = connection.execute(observation_query).mappings().first()
            if observation is not None:
                cache.store(station_id, cache_version, observation)
    if observation is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "OBSERVATION_NOT_FOUND",
                "message": "Station has no valid observation at or before the current time.",
            },
        )

    data_age_minutes = (now - observation["observed_at"]).total_seconds() / 60
    stale_after_minutes = request.app.state.stale_after_minutes
    return LatestObservationResponse(
        station_id=station_id,
        station_type=station["station_type"],
        value=observation["depth"],
        unit=station["measurement_unit"],
        observed_at=observation["observed_at"],
        received_at=observation["received_at"],
        quality_status=observation["quality_status"],
        is_stale=data_age_minutes > stale_after_minutes,
        data_age_minutes=data_age_minutes,
        stale_after_minutes=stale_after_minutes,
    )


@router.get(
    "/stations/{station_id}/observations",
    response_model=ObservationHistoryResponse,
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def get_observation_history(
    request: Request,
    station_id: str,
    query: Annotated[ObservationHistoryQuery, Query()],
) -> ObservationHistoryResponse:
    """Page raw observations newest first within [start, end), including missing rows.

    All input timestamps require a timezone; response timestamps are UTC. The
    exclusive before cursor prevents repeating a page's last observation.
    An existing station with no rows in the range returns an empty list. This
    endpoint does not aggregate, fill gaps, or check prediction availability.

    Args:
        request: HTTP request providing the database engine.
        station_id: Exact water/rain station ID; disabled stations remain readable.
        query: Timezone-aware start/end bounds, page limit (1-1000), and optional
            exclusive before cursor. Bounds are [start, end); end cannot be future.

    Returns:
        Page of original-unit readings ordered newest first, including null/missing
        rows, with source unit and next_before when another page exists. Reuse that
        cursor with the same bounds to load older readings without repeating a row.

    Raises:
        HTTPException: 404 for an unknown station or 422 for a future end time.
        sqlalchemy.exc.SQLAlchemyError: Database failures handled as 503 by the app.

    Invalid query formats/ranges are rejected by FastAPI/Pydantic before this handler.
    """
    start = query.start.astimezone(timezone.utc)
    end = query.end.astimezone(timezone.utc)
    now = datetime.now(timezone.utc)
    if end > now:
        raise HTTPException(
            status_code=422,
            detail={"code": "INVALID_REQUEST", "message": "end must not be in the future."},
        )

    station_query = sa.select(Station.station_type, Station.measurement_unit).where(
        Station.id == station_id
    )
    history_query = sa.select(
        Observation.depth,
        Observation.observed_at,
        Observation.received_at,
        Observation.quality_status,
    ).where(
        Observation.station_id == station_id,
        Observation.observed_at >= start,
        Observation.observed_at < end,
    )
    if query.before is not None:
        history_query = history_query.where(Observation.observed_at < query.before)
    # One extra row tells the client whether another page exists, without COUNT(*).
    history_query = history_query.order_by(Observation.observed_at.desc()).limit(query.limit + 1)
    with request.app.state.database_engine.connect() as connection:
        station = connection.execute(station_query).mappings().first()
        if station is None:
            raise HTTPException(
                status_code=404,
                detail={"code": "STATION_NOT_FOUND", "message": "Station does not exist."},
            )
        history_rows = connection.execute(history_query).mappings().all()

    has_more = len(history_rows) > query.limit
    page_rows = history_rows[: query.limit]
    observations = [
        HistoricalObservation(
            observed_at=row["observed_at"].astimezone(timezone.utc),
            received_at=row["received_at"].astimezone(timezone.utc),
            value=row["depth"],
            quality_status=row["quality_status"],
        )
        for row in page_rows
    ]
    return ObservationHistoryResponse(
        station_id=station_id,
        station_type=station["station_type"],
        unit=station["measurement_unit"],
        start=start,
        end=end,
        observations=observations,
        has_more=has_more,
        next_before=observations[-1].observed_at if has_more else None,
    )
