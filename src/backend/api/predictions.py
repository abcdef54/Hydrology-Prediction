"""Run the configured station model on observed history plus optional last-step edits."""

from dataclasses import asdict
from datetime import datetime, timezone
import pickle

from fastapi import APIRouter, HTTPException, Request

from src.backend.api.prediction_inputs import get_station_model_config
from src.backend.api.schemas import ErrorResponse, PredictionRequest, PredictionResponse
from src.backend.prediction.config import StationModelConfig
from src.backend.prediction.inference import PredictionService, ResidualLSTMRuntime
from src.backend.prediction.inputs import InsufficientHistoryError
from src.backend.prediction.preprocessing import InvalidOverrideError

router = APIRouter(prefix="/api/v1", tags=["Predictions"])


def get_prediction_service(request: Request, config: StationModelConfig) -> PredictionService:
    """Reuse a runtime per station in this worker; never share it across stations.

    The lock covers first loading only. Failed loads are not retained, so a later
    request may retry once the artifact is available. Inference results and user
    overrides are not stored in this dictionary.

    Args:
        request: HTTP request whose app state owns the engine and runtime registry.
        config: Station-specific model and preprocessing configuration to resolve.

    Returns:
        Prediction service for this station and current artifact snapshot.

    Raises:
        HTTPException: With status 503 when model artifacts cannot be loaded.
    """
    with request.app.state.prediction_service_lock:
        service = request.app.state.prediction_services.get(config.station_id)
        try:
            if (
                service is not None
                and service.runtime.config == config
                and service.runtime.artifacts_match_config()
            ):
                return service
            runtime = ResidualLSTMRuntime(config, device=request.app.state.prediction_device)
        except (OSError, ValueError, KeyError, EOFError, RuntimeError, pickle.UnpicklingError):
            raise HTTPException(
                503, {"code": "MODEL_UNAVAILABLE", "message": "Station model could not be loaded."}
            ) from None
        service = PredictionService(request.app.state.database_engine, runtime)
        request.app.state.prediction_services[config.station_id] = service
        return service


@router.post(
    "/predictions",
    response_model=PredictionResponse,
    responses={status: {"model": ErrorResponse} for status in (400, 404, 422, 503)},
)
def predict_water_level(
    request: Request, prediction_request: PredictionRequest
) -> PredictionResponse:
    """Predict supported horizons in mm from the actual last input timestamp.

    Original-input forecasts are reused or saved in PostgreSQL. Override scenarios
    are never stored. No training or future observations are used. Invalid user
    edits return 422; observation/preprocessing or inference failures return 503.
    Missing rain/lag inputs follow the saved training preprocessing.

    Args:
        request: HTTP request providing app resources and the freshness threshold.
        prediction_request: Station ID, optional timezone-aware reference time and
            optional last-timestep edits in original feature units.

    Returns:
        Horizon predictions in mm, storage metadata, applied overrides and freshness
        calculated for the current request.

    Raises:
        HTTPException: For invalid requests/edits (422), unknown stations (404),
            insufficient history (400) or unavailable model/input execution (503).
        sqlalchemy.exc.SQLAlchemyError: Database failures handled as 503 by the app.
    """
    current_time = datetime.now(timezone.utc)
    reference_time = prediction_request.reference_time
    if reference_time is not None and reference_time > current_time:
        raise HTTPException(
            422,
            {
                "code": "INVALID_REFERENCE_TIME",
                "message": "Reference time cannot be in the future.",
            },
        )
    config = get_station_model_config(request, prediction_request.station_id)
    service = get_prediction_service(request, config)
    try:
        # Reject bad edits before fetching the potentially large historical window.
        service.runtime.preprocessor.validate_overrides(prediction_request.overrides)
        forecast = service.predict(reference_time, prediction_request.overrides)
    except InvalidOverrideError as error:
        raise HTTPException(422, {"code": "INVALID_OVERRIDE", "message": str(error)}) from None
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
    except (ValueError, RuntimeError):
        raise HTTPException(
            503,
            {
                "code": "PREDICTION_UNAVAILABLE",
                "message": "Prediction could not be produced for this input.",
            },
        ) from None
    age_reference = current_time if reference_time is None else forecast.hourly_reference_time
    input_age_minutes = max(0.0, (age_reference - forecast.anchor_observed_at).total_seconds() / 60)
    stale_after_minutes = request.app.state.stale_after_minutes
    return PredictionResponse(
        **asdict(forecast),
        overrides_applied=prediction_request.overrides,
        is_stale=input_age_minutes > stale_after_minutes,
        input_age_minutes=input_age_minutes,
        stale_after_minutes=stale_after_minutes,
    )
