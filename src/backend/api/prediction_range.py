"""Prepare bounded range chunks using exact anchors and normal committed forecast reuse."""

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Request
import sqlalchemy as sa

from src.backend.api.forecast_detail import hourly_actuals_query
from src.backend.api.forecast_history import forecast_comparison_query, resolve_forecast_identity
from src.backend.api.prediction_inputs import get_station_model_config
from src.backend.api.predictions import get_prediction_service
from src.backend.api.schemas import (
    ErrorResponse,
    PredictionOptionsResponse,
    RangePredictionPoint,
    RangePredictionRequest,
    RangePredictionResponse,
    ForecastHistoryQuery,
)
from src.backend.db.models import Station
from src.backend.prediction.cached_service import AnchorUnavailableError
from src.backend.prediction.inputs import InsufficientHistoryError
from src.backend.security.inference import prediction_slot

router = APIRouter(prefix="/api/v1", tags=["Predictions"])


def require_mm(request, station_id):
    with request.app.state.database_engine.connect() as connection:
        unit = connection.scalar(
            sa.select(Station.measurement_unit).where(Station.id == station_id)
        )
    if unit != "mm":
        raise HTTPException(
            503, {"code": "UNIT_UNAVAILABLE", "message": "Water-level unit must be mm."}
        )


@router.get("/stations/{station_id}/prediction-options", response_model=PredictionOptionsResponse)
def get_prediction_options(request: Request, station_id: str) -> PredictionOptionsResponse:
    """Read allowed horizons without preparing inputs or loading model artifacts.

    Args:
        request: App registry/database resources.
        station_id: Enabled water station with its own configured model.

    Returns:
        Model identifier and supported horizon list, for the range-mode selector.

    Raises:
        HTTPException: Unknown/disabled/unconfigured station or unconfirmed mm units.
    """
    config = get_station_model_config(request, station_id)
    require_mm(request, station_id)
    return PredictionOptionsResponse(
        station_id=station_id,
        model_version=config.model_version,
        forecast_horizons_minutes=list(config.forecast_horizons_minutes),
    )


@router.post(
    "/predictions/range",
    response_model=RangePredictionResponse,
    responses={status: {"model": ErrorResponse} for status in (400, 404, 409, 422, 429, 503)},
)
def predict_range(request: Request, body: RangePredictionRequest) -> RangePredictionResponse:
    """Reuse or create original-input forecasts for at most 24 target hours.

    Args:
        request: App resources including the shared prediction semaphore and Redis.
        body: Station, target window [start,end), supported horizon and optional
            pinned model/pipeline identity from the preceding chunk. No overrides.

    Returns:
        Each target hour's forecast and observed hourly mean in mm, plus reuse/create/
        skip counts. For target T the exact input anchor must be T - horizon. Missing
        history or a shifted anchor is skipped, never relabeled as another target.
        Committed earlier rows remain reusable if a later row fails or caller stops.

    Raises:
        HTTPException: Invalid horizon, changed pipeline (409), busy prediction slot,
            model/input failure (503). Database exceptions use the app's safe 503.

    One semaphore covers the chunk, including input/cache/DB/inference. Redis is
    checked after hashing current inputs; DB commit precedes publishing any cache.
    """
    now = datetime.now(timezone.utc)
    start, end = body.start.astimezone(timezone.utc), body.end.astimezone(timezone.utc)
    points = []
    with prediction_slot(request):
        config = get_station_model_config(request, body.station_id)
        require_mm(request, body.station_id)
        if body.horizon_minutes not in config.forecast_horizons_minutes:
            raise HTTPException(
                422, {"code": "INVALID_HORIZON", "message": "Unsupported forecast horizon."}
            )
        history_query = ForecastHistoryQuery(
            start=start, end=end, horizon_minutes=body.horizon_minutes, limit=1000
        )
        _, pipeline = resolve_forecast_identity(request, body.station_id, history_query)
        if body.model_version is not None and (
            body.model_version != config.model_version or body.pipeline_version != pipeline
        ):
            raise HTTPException(
                409,
                {
                    "code": "PIPELINE_CHANGED",
                    "message": "Model/pipeline changed; restart the range.",
                },
            )
        with request.app.state.database_engine.connect() as connection:
            stored_rows = (
                connection.execute(
                    forecast_comparison_query(
                        body.station_id, config.model_version, pipeline, history_query, now
                    )
                )
                .mappings()
                .all()
            )
        stored_by_time = {row["forecast_at"]: row for row in stored_rows}
        service = None
        target = start
        while target < end:
            anchor = target - timedelta(minutes=body.horizon_minutes)
            point = {"forecast_at": target, "status": "future_anchor"}
            stored = stored_by_time.get(target)
            if stored is not None:
                if stored["anchor_observed_at"] != anchor:
                    raise HTTPException(
                        503,
                        {
                            "code": "STORED_FORECAST_INVALID",
                            "message": "Stored anchor does not match target.",
                        },
                    )
                point.update(
                    status="ready",
                    forecast_id=stored["forecast_id"],
                    anchor_observed_at=anchor,
                    predicted_water_level_mm=stored["predicted_water_level_mm"],
                    reused=True,
                )
            elif anchor <= now:
                try:
                    if service is None:
                        service = get_prediction_service(request, config)
                        if service.runtime.pipeline_version != pipeline:
                            raise HTTPException(
                                409,
                                {
                                    "code": "PIPELINE_CHANGED",
                                    "message": "Pipeline changed; restart range.",
                                },
                            )
                    forecast = service.predict(anchor, required_anchor=anchor)
                    selected = next(
                        p for p in forecast.predictions if p.horizon_minutes == body.horizon_minutes
                    )
                    if selected.forecast_at != target:
                        raise ValueError("Forecast target does not match requested hour.")
                    point.update(
                        status="ready",
                        forecast_id=forecast.forecast_id,
                        anchor_observed_at=forecast.anchor_observed_at,
                        predicted_water_level_mm=selected.water_level_mm,
                        reused=forecast.reused,
                    )
                except InsufficientHistoryError:
                    point["status"] = "insufficient_history"
                except AnchorUnavailableError:
                    point["status"] = "anchor_unavailable"
                except (ValueError, RuntimeError, StopIteration):
                    raise HTTPException(
                        503,
                        {
                            "code": "PREDICTION_UNAVAILABLE",
                            "message": "Range prediction failed; completed forecasts remain stored.",
                        },
                    ) from None
            points.append(point)
            target += timedelta(hours=1)
        with request.app.state.database_engine.connect() as connection:
            actuals = (
                connection.execute(
                    hourly_actuals_query(
                        body.station_id, start - timedelta(hours=1), end - timedelta(hours=1), now
                    )
                )
                .mappings()
                .all()
            )
    by_time = {point["timestamp"]: point for point in actuals}
    result = [
        RangePredictionPoint(
            **point,
            actual_water_level_mm=by_time[point["forecast_at"]]["actual_water_level_mm"],
            actual_sample_count=by_time[point["forecast_at"]]["actual_sample_count"]
        )
        for point in points
    ]
    return RangePredictionResponse(
        station_id=body.station_id,
        start=start,
        end=end,
        horizon_minutes=body.horizon_minutes,
        model_version=config.model_version,
        pipeline_version=pipeline,
        points=result,
        created_count=sum(p.status == "ready" and not p.reused for p in result),
        reused_count=sum(p.status == "ready" and p.reused for p in result),
        skipped_count=sum(p.status != "ready" for p in result),
    )
