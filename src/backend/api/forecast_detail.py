"""Visualize every saved horizon from one anchor against hourly observed truth."""

from datetime import datetime, timedelta, timezone
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
import sqlalchemy as sa

from src.backend.api.forecast_history import hourly_truth_subquery
from src.backend.api.schemas import (
    ActualWaterLevelPoint,
    ErrorResponse,
    ForecastDetailResponse,
    PredictedWaterLevel,
)
from src.backend.db.models import Forecast, ForecastPoint, Station

router = APIRouter(prefix="/api/v1", tags=["Saved forecasts"])


def hourly_actuals_query(
    station_id: str, anchor: datetime, window_end: datetime, now: datetime
) -> sa.Select:
    """Build an hourly grid after an anchor, retaining hours without observations.

    Args:
        station_id: Water station attached to the saved forecast.
        anchor: Aware forecast anchor; the first actual timestamp is anchor + 1h.
        window_end: Anchor plus the largest saved horizon, inclusive.
        now: Current aware UTC instant; no future actual values are exposed.

    Returns:
        SQLAlchemy SELECT ordered oldest first, with a row for each whole hour
        after anchor up to window_end. Each row has timestamp, hourly mean in mm
        and valid sample count. Missing/future values are None with count zero.
        Non-hourly horizon endpoints do not create extra hourly observations.
    """
    hours = (
        sa.func.generate_series(
            sa.literal(anchor + timedelta(hours=1), type_=sa.DateTime(timezone=True)),
            sa.literal(window_end, type_=sa.DateTime(timezone=True)),
            sa.literal(timedelta(hours=1), type_=sa.Interval()),
        )
        .table_valued("timestamp")
        .render_derived(name="hours")
    )
    truth = hourly_truth_subquery(station_id, hours.c.timestamp, now)
    return (
        sa.select(hours.c.timestamp, truth.c.actual_water_level_mm, truth.c.actual_sample_count)
        .select_from(hours.outerjoin(truth, sa.true()))
        .order_by(hours.c.timestamp)
    )


@router.get(
    "/forecasts/{forecast_id}",
    response_model=ForecastDetailResponse,
    responses={status: {"model": ErrorResponse} for status in (404, 422, 503)},
)
def get_forecast_detail(request: Request, forecast_id: UUID) -> ForecastDetailResponse:
    """Read one saved forecast and hourly truth for a single-anchor chart.

    Args:
        request: HTTP request providing the PostgreSQL engine.
        forecast_id: UUID returned by POST predictions or the saved-series endpoint.
            It selects the exact original revision, regardless of current model/device.

    Returns:
        Saved horizons ordered by lead time and hourly actuals ordered by timestamp,
        both in mm. Window length comes from the largest saved horizon; the current
        Da Vien model yields five predictions and 24 hourly actual slots. Missing
        or not-yet-observed hours stay None. No artifacts are loaded or rows written.

    Raises:
        HTTPException: 404 if the saved forecast does not exist; 503 if station units
            are not confirmed as mm or stored horizons are empty/inconsistent.
        sqlalchemy.exc.SQLAlchemyError: Database failures handled as 503 by the app.

    Stored model/pipeline and generated_at remain unchanged. Observed truth is read
    from the current database and may reflect corrections made since generation.
    """
    now = datetime.now(timezone.utc)
    with request.app.state.database_engine.connect() as connection:
        parent = (
            connection.execute(
                sa.select(Forecast.__table__, Station.station_type, Station.measurement_unit)
                .join(Station, Station.id == Forecast.station_id)
                .where(Forecast.id == forecast_id)
            )
            .mappings()
            .first()
        )
        if parent is None:
            raise HTTPException(
                404, {"code": "FORECAST_NOT_FOUND", "message": "Saved forecast does not exist."}
            )
        if parent["station_type"] != "water_level" or parent["measurement_unit"] != "mm":
            raise HTTPException(
                503,
                {
                    "code": "UNIT_UNAVAILABLE",
                    "message": "Observed water-level unit must be confirmed as mm.",
                },
            )
        points = (
            connection.execute(
                sa.select(
                    ForecastPoint.horizon_minutes,
                    ForecastPoint.forecast_at,
                    ForecastPoint.water_level_mm,
                )
                .where(ForecastPoint.forecast_id == forecast_id)
                .order_by(ForecastPoint.horizon_minutes)
            )
            .mappings()
            .all()
        )
        anchor = parent["anchor_observed_at"].astimezone(timezone.utc)
        if not points or any(
            point["forecast_at"] != anchor + timedelta(minutes=point["horizon_minutes"])
            for point in points
        ):
            raise HTTPException(
                503,
                {
                    "code": "STORED_FORECAST_INVALID",
                    "message": "Saved forecast horizons are unavailable or inconsistent.",
                },
            )
        window_end = anchor + timedelta(minutes=points[-1]["horizon_minutes"])
        actuals = (
            connection.execute(hourly_actuals_query(parent["station_id"], anchor, window_end, now))
            .mappings()
            .all()
        )
    return ForecastDetailResponse(
        forecast_id=forecast_id,
        station_id=parent["station_id"],
        model_version=parent["model_version"],
        feature_schema_version=parent["feature_schema_version"],
        pipeline_version=parent["pipeline_version"],
        anchor_observed_at=anchor,
        generated_at=parent["generated_at"],
        window_end=window_end,
        predictions=[PredictedWaterLevel(**point) for point in points],
        actuals=[ActualWaterLevelPoint(**actual) for actual in actuals],
    )
