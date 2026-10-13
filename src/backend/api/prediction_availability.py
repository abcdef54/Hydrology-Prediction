"""Check selectable historical hours using the prediction input service."""

from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from src.backend.api.prediction_inputs import get_input_service
from src.backend.api.schemas import (
    ErrorResponse,
    PredictionAvailabilityPoint,
    PredictionAvailabilityQuery,
    PredictionAvailabilityResponse,
)
from src.backend.prediction.inputs import InsufficientHistoryError, PredictionInputService

router = APIRouter(prefix="/api/v1", tags=["Prediction inputs"])


@router.get(
    "/stations/{station_id}/prediction-availability",
    response_model=PredictionAvailabilityResponse,
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def get_prediction_availability(
    station_id: str,
    query: Annotated[PredictionAvailabilityQuery, Query()],
    resources: Annotated[tuple[PredictionInputService, dict], Depends(get_input_service)],
) -> PredictionAvailabilityResponse:
    """Check hourly reference times in [start, end), ascending, for up to 7 days.

    Uses each station's saved preprocessing contract and actual prepare() path,
    including allowed missing rain and gaps. No weights or inference are used.
    A ready hour may use an older anchor; readiness does not imply freshness.
    This is advisory: prediction must revalidate if observations change later.
    """
    start = query.start.astimezone(timezone.utc)
    end = query.end.astimezone(timezone.utc)
    if end > datetime.now(timezone.utc):
        raise HTTPException(
            422, {"code": "INVALID_REQUEST", "message": "end must not be in the future."}
        )
    service, _ = resources
    reference = start.replace(minute=0, second=0, microsecond=0)
    if reference < start:
        reference += timedelta(hours=1)
    availability = []
    while reference < end:
        try:
            prepared = service.prepare(reference)
            point = PredictionAvailabilityPoint(
                reference_time=reference,
                input_ready=True,
                available_timesteps=len(prepared.timestamps),
                anchor_observed_at=prepared.anchor_observed_at,
                reason=None,
            )
        except InsufficientHistoryError as error:
            point = PredictionAvailabilityPoint(
                reference_time=reference,
                input_ready=False,
                available_timesteps=error.available,
                anchor_observed_at=None,
                reason="INSUFFICIENT_HISTORY",
            )
        except (ValueError, RuntimeError):
            raise HTTPException(
                503,
                {
                    "code": "PREDICTION_INPUTS_UNAVAILABLE",
                    "message": "Prediction inputs could not be prepared.",
                },
            ) from None
        availability.append(point)
        reference += timedelta(hours=1)
    return PredictionAvailabilityResponse(
        station_id=station_id,
        model_version=service.config.model_version,
        feature_schema_version=service.config.feature_schema_version,
        sequence_length=service.config.sequence_length,
        start=start,
        end=end,
        availability=availability,
    )
