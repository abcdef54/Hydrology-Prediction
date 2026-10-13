"""Read station observations for display, independently of prediction inputs."""

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request
import sqlalchemy as sa

from src.backend.api.schemas import ErrorResponse, LatestObservationResponse
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
        observation = connection.execute(observation_query).mappings().first()
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
