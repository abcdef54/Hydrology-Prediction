"""Read model input for forms without loading weights or running inference."""

from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
import numpy as np
import sqlalchemy as sa

from src.backend.api.schemas import (
    ErrorResponse,
    InputFeature,
    PredictionInputsQuery,
    PredictionInputsResponse,
)
from src.backend.db.models import Station
from src.backend.prediction.config import ModelNotConfiguredError, StationModelConfig
from src.backend.prediction.inputs import InsufficientHistoryError, PredictionInputService
from src.backend.prediction.preprocessing import InputPreprocessor

router = APIRouter(prefix="/api/v1", tags=["Prediction inputs"])


def get_station_model_config(request: Request, station_id: str) -> StationModelConfig:
    """Resolve a supported water station; unknown IDs return 404, unavailable models 503."""
    engine = request.app.state.database_engine
    with engine.connect() as connection:
        station = (
            connection.execute(
                sa.select(
                    Station.id, Station.station_type, Station.enabled, Station.prediction_enabled
                ).where(Station.id == station_id)
            )
            .mappings()
            .first()
        )
        if station is None:
            raise HTTPException(404, {"code": "STATION_NOT_FOUND", "message": "Station not found."})
        if not station["enabled"] or not station["prediction_enabled"]:
            raise HTTPException(
                503,
                {
                    "code": "MODEL_UNAVAILABLE",
                    "message": "Prediction is not enabled for this station.",
                },
            )
        try:
            config = request.app.state.model_registry.get(station_id)
        except ModelNotConfiguredError:
            raise HTTPException(
                503,
                {
                    "code": "MODEL_UNAVAILABLE",
                    "message": "No model is configured for this station.",
                },
            ) from None
    return config


def get_input_service(request: Request, station_id: str) -> tuple[PredictionInputService, dict]:
    """Check station availability and resolve its own scaler and source units.

    Reads station metadata only. Unknown stations return 404; disabled stations
    or stations without a configured/usable model return 503. No model weights
    are loaded. The caller receives a service plus units keyed by source station.
    """
    config = get_station_model_config(request, station_id)
    engine = request.app.state.database_engine
    with engine.connect() as connection:
        source_ids = (config.station_id, *config.rain_station_ids)
        units = dict(
            connection.execute(
                sa.select(Station.id, Station.measurement_unit).where(Station.id.in_(source_ids))
            ).all()
        )
    try:
        preprocessor = InputPreprocessor.from_file(config)
    except (OSError, ValueError, KeyError, EOFError):
        raise HTTPException(
            503, {"code": "MODEL_UNAVAILABLE", "message": "Model preprocessing is unavailable."}
        ) from None
    return PredictionInputService(engine, preprocessor), units


def describe_feature(
    name: str, value: float, config: StationModelConfig, units: dict
) -> InputFeature:
    """Preserve raw missingness and use confirmed station units for the form."""
    source_station_id = None
    if name.startswith("water_level"):
        source_station_id = config.station_id
    elif name.startswith("rain_"):
        source_station_id = name.split("_", 2)[1]
    is_flag = name.endswith("_missing") or name == "flood_season"
    return InputFeature(
        name=name,
        value=float(value) if np.isfinite(value) else None,
        unit="dimensionless" if is_flag else units.get(source_station_id),
        source_station_id=source_station_id,
    )


@router.get(
    "/stations/{station_id}/prediction-inputs",
    response_model=PredictionInputsResponse,
    responses={status: {"model": ErrorResponse} for status in (400, 404, 422, 503)},
)
def prediction_inputs(
    request: Request,
    query: Annotated[PredictionInputsQuery, Query()],
    service_and_units: Annotated[tuple[PredictionInputService, dict], Depends(get_input_service)],
) -> PredictionInputsResponse:
    """Return the observed last timestep for auto-fill; no overrides or inference.

    Historical inputs use only observations at or before the rounded reference.
    Missing rain/lag values remain null in the form but can still be model-ready
    because the saved training preprocessing handles them.
    """
    current_time = datetime.now(timezone.utc)
    if query.reference_time is not None and query.reference_time > current_time:
        raise HTTPException(
            422,
            {
                "code": "INVALID_REFERENCE_TIME",
                "message": "Reference time cannot be in the future.",
            },
        )
    service, source_units = service_and_units
    try:
        prepared = service.prepare(query.reference_time)
    except InsufficientHistoryError as error:
        raise HTTPException(
            400,
            {
                "code": "INSUFFICIENT_HISTORY",
                "message": "Not enough valid historical observations to form model input.",
                "details": {
                    "required": error.required,
                    "available": error.available,
                    "input_ready": False,
                },
            },
        ) from None
    except ValueError:
        raise HTTPException(
            503,
            {
                "code": "INPUT_UNAVAILABLE",
                "message": "Observed input could not be prepared for this model.",
            },
        ) from None
    config = service.config
    features = [
        describe_feature(name, prepared.raw_features.iloc[-1][name], config, source_units)
        for name in config.feature_columns
    ]
    age_reference = current_time if query.reference_time is None else prepared.hourly_reference_time
    input_age_minutes = max(0.0, (age_reference - prepared.anchor_observed_at).total_seconds() / 60)
    stale_after_minutes = request.app.state.stale_after_minutes
    is_stale = input_age_minutes > stale_after_minutes
    return PredictionInputsResponse(
        station_id=prepared.station_id,
        model_version=prepared.model_version,
        feature_schema_version=prepared.feature_schema_version,
        requested_reference_time=prepared.requested_reference_time,
        hourly_reference_time=prepared.hourly_reference_time,
        anchor_observed_at=prepared.anchor_observed_at,
        sequence_length=config.sequence_length,
        feature_count=len(config.feature_columns),
        forecast_horizons_minutes=list(config.forecast_horizons_minutes),
        features=features,
        editable_features=[feature.name for feature in features],
        missing_features=[feature.name for feature in features if feature.value is None],
        input_ready=True,
        is_stale=is_stale,
        input_age_minutes=input_age_minutes,
        stale_after_minutes=stale_after_minutes,
    )
