"""Station metadata for map markers and station selection."""

from collections.abc import Mapping
from typing import Any

from fastapi import APIRouter, HTTPException, Request
import sqlalchemy as sa

from src.backend.api.schemas import (
    AlarmLevels,
    ErrorResponse,
    StationDetail,
    StationLocation,
    StationSummary,
    StationsResponse,
)
from src.backend.db.models import Station
from src.backend.prediction.config import ModelNotConfiguredError, StationModelRegistry

router = APIRouter(prefix="/api/v1", tags=["Stations"])


def build_station_summary(
    station: Mapping[str, Any], registry: StationModelRegistry
) -> StationSummary:
    """Describe metadata and effective permission without loading model artifacts.

    Prediction requires an enabled water station's database flag and its own
    registry entry. History sufficiency and runtime availability are checked
    separately by the prediction endpoints.
    """
    try:
        registry.get(station["id"])
        model_configured = True
    except ModelNotConfiguredError:
        model_configured = False

    prediction_enabled = (
        station["enabled"]
        and station["station_type"] == "water_level"
        and station["prediction_enabled"]
        and model_configured
    )
    return StationSummary(
        station_id=station["id"],
        name=station["name"],
        station_type=station["station_type"],
        river_name=station["river_name"],
        location=StationLocation(
            latitude=station["latitude"],
            longitude=station["longitude"],
            address=station["address"],
        ),
        measurement_unit=station["measurement_unit"],
        model_configured=model_configured,
        prediction_enabled=prediction_enabled,
    )


@router.get(
    "/stations",
    response_model=StationsResponse,
    responses={503: {"model": ErrorResponse}},
)
def list_stations(request: Request) -> StationsResponse:
    """List enabled stations ordered by ID; unknown metadata remains null."""
    statement = sa.select(Station.__table__).where(Station.enabled.is_(True)).order_by(Station.id)
    with request.app.state.database_engine.connect() as connection:
        station_rows = connection.execute(statement).mappings().all()
    return StationsResponse(
        stations=[
            build_station_summary(station, request.app.state.model_registry)
            for station in station_rows
        ]
    )


@router.get(
    "/stations/{station_id}",
    response_model=StationDetail,
    responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def get_station(request: Request, station_id: str) -> StationDetail:
    """Read one station, including disabled stations, by its exact ID.

    Coordinates are decimal degrees and alarm levels are metres as stored in
    PostgreSQL. Unknown thresholds remain null, never inferred from the model.
    Disabled stations remain inspectable but cannot permit predictions.
    """
    statement = sa.select(Station.__table__).where(Station.id == station_id)
    with request.app.state.database_engine.connect() as connection:
        station = connection.execute(statement).mappings().first()
    if station is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "STATION_NOT_FOUND", "message": "Station does not exist."},
        )
    summary = build_station_summary(station, request.app.state.model_registry)
    return StationDetail(
        **summary.model_dump(),
        enabled=station["enabled"],
        alarm_levels_m=AlarmLevels(
            level_1=station["alarm_level_1_m"],
            level_2=station["alarm_level_2_m"],
            level_3=station["alarm_level_3_m"],
        ),
    )
