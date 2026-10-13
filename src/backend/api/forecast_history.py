"""Read a saved forecast series and align observed truth without running a model."""

from datetime import datetime, timedelta, timezone
import hashlib
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request
import sqlalchemy as sa

from src.backend.api.schemas import (
    ErrorResponse,
    ForecastComparisonPoint,
    ForecastHistoryQuery,
    ForecastHistoryResponse,
)
from src.backend.db.models import Forecast, ForecastPoint, Observation, Station
from src.backend.prediction.config import ModelNotConfiguredError
from src.backend.prediction.persistence import build_pipeline_version

router = APIRouter(prefix="/api/v1", tags=["Saved forecasts"])


def resolve_forecast_identity(
    request: Request, station_id: str, query: ForecastHistoryQuery
) -> tuple[str, str]:
    """Resolve exactly one series without loading weights or executing inference.

    Args:
        request: Request containing the model registry and backend device setting.
        station_id: Water station whose saved predictions should be read.
        query: Optional model/pipeline identifiers; omitted values select the
            currently configured model and its artifact/code/device identity.

    Returns:
        Model version and pipeline version to apply together as database filters.
        Explicit archived identities remain readable without their artifact files.

    Raises:
        HTTPException: 503 if defaults cannot be resolved, or 422 if an archived
            model is requested without an explicit pipeline version.
    """
    if query.model_version is not None and query.pipeline_version is not None:
        return query.model_version, query.pipeline_version
    try:
        config = request.app.state.model_registry.get(station_id)
    except ModelNotConfiguredError:
        raise HTTPException(
            503,
            {
                "code": "MODEL_UNAVAILABLE",
                "message": "Specify stored model and pipeline versions for this station.",
            },
        ) from None
    model_version = query.model_version or config.model_version
    if query.pipeline_version is not None:
        return model_version, query.pipeline_version
    if model_version != config.model_version:
        raise HTTPException(
            422,
            {
                "code": "INVALID_REQUEST",
                "message": "Archived model versions require an explicit pipeline_version.",
            },
        )
    try:
        artifact_hashes = tuple(
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (config.model_path, config.preprocessing_path)
        )
        pipeline_version = build_pipeline_version(
            config, artifact_hashes, request.app.state.prediction_device
        )
    except OSError:
        raise HTTPException(
            503,
            {
                "code": "MODEL_UNAVAILABLE",
                "message": "Current pipeline identity is unavailable; specify stored versions.",
            },
        ) from None
    return model_version, pipeline_version


def forecast_comparison_query(
    station_id: str,
    model_version: str,
    pipeline_version: str,
    query: ForecastHistoryQuery,
    now: datetime,
) -> sa.Select:
    """Build a bounded PostgreSQL query for one prediction line and hourly truth.

    Args:
        station_id: Exact water-station ID used by both forecasts and observations.
        model_version: Saved model version to read.
        pipeline_version: Exact pipeline identity; CPU/GPU revisions are not mixed.
        query: Target-time bounds [start, end), one horizon and pagination controls.
        now: UTC current instant; future target hours receive no observed truth.

    Returns:
        SQLAlchemy SELECT yielding at most limit + 1 rows, newest target first.
        Only the latest generated revision per target time is selected. A lateral
        aggregate reads observed values in (target - 1h, target], matching training.
        Missing rows are ignored in the mean and counted as zero valid samples;
        timestamps without records are not interpolated or filled.
    """
    latest_revision = (
        sa.select(
            Forecast.id.label("forecast_id"),
            Forecast.feature_schema_version,
            Forecast.anchor_observed_at,
            Forecast.generated_at,
            ForecastPoint.forecast_at,
            ForecastPoint.water_level_mm.label("predicted_water_level_mm"),
            sa.func.row_number()
            .over(
                partition_by=ForecastPoint.forecast_at,
                order_by=(Forecast.generated_at.desc(), Forecast.id.desc()),
            )
            .label("revision_rank"),
        )
        .join(ForecastPoint, ForecastPoint.forecast_id == Forecast.id)
        .where(
            Forecast.station_id == station_id,
            Forecast.model_version == model_version,
            Forecast.pipeline_version == pipeline_version,
            ForecastPoint.horizon_minutes == query.horizon_minutes,
            ForecastPoint.forecast_at >= query.start,
            ForecastPoint.forecast_at < query.end,
        )
        .subquery("latest_revision")
    )
    page_query = sa.select(
        *(column for column in latest_revision.c if column.name != "revision_rank")
    ).where(latest_revision.c.revision_rank == 1)
    if query.before is not None:
        page_query = page_query.where(latest_revision.c.forecast_at < query.before)
    page = (
        page_query.order_by(latest_revision.c.forecast_at.desc())
        .limit(query.limit + 1)
        .subquery("page")
    )
    truth = (
        sa.select(
            sa.func.avg(Observation.depth).label("actual_water_level_mm"),
            sa.func.count(Observation.depth).label("actual_sample_count"),
        )
        .where(
            Observation.station_id == station_id,
            Observation.quality_status == "observed",
            Observation.depth.is_not(None),
            Observation.observed_at > page.c.forecast_at - timedelta(hours=1),
            Observation.observed_at <= page.c.forecast_at,
            page.c.forecast_at <= now,
        )
        .lateral("truth")
    )
    return (
        sa.select(page, truth.c.actual_water_level_mm, truth.c.actual_sample_count)
        .select_from(page.outerjoin(truth, sa.true()))
        .order_by(page.c.forecast_at.desc())
    )


@router.get(
    "/stations/{station_id}/forecasts",
    response_model=ForecastHistoryResponse,
    responses={status: {"model": ErrorResponse} for status in (404, 422, 503)},
)
def get_forecast_history(
    request: Request,
    station_id: str,
    query: Annotated[ForecastHistoryQuery, Query()],
) -> ForecastHistoryResponse:
    """Read saved predictions and actual hourly water levels for a comparison chart.

    Args:
        request: HTTP request providing the database and default pipeline settings.
        station_id: Exact water-station ID; disabled stations' history is readable.
        query: One horizon, optional model/pipeline identity, timezone-aware target
            time bounds, limit and exclusive before cursor. Future bounds are
            allowed so saved forecasts beyond the latest observations can appear.

    Returns:
        Saved points in mm, newest forecast_at first, with observed hourly mean
        and valid sample count. Unobserved/future hours have actual=None and count=0.
        Empty ranges return 200 with no points. No forecast is created or changed.

    Raises:
        HTTPException: 404 for unknown station, 422 for a non-water station or an
            archived model without pipeline identity, 503 if defaults/units cannot
            be resolved safely. Invalid query shapes are rejected by validation.
        sqlalchemy.exc.SQLAlchemyError: Database failures handled as 503 by the app.
    """
    with request.app.state.database_engine.connect() as connection:
        station = (
            connection.execute(
                sa.select(Station.station_type, Station.measurement_unit).where(
                    Station.id == station_id
                )
            )
            .mappings()
            .first()
        )
        if station is None:
            raise HTTPException(
                404, {"code": "STATION_NOT_FOUND", "message": "Station does not exist."}
            )
        if station["station_type"] != "water_level":
            raise HTTPException(
                422,
                {
                    "code": "INVALID_STATION_TYPE",
                    "message": "Saved water-level forecasts require a water station.",
                },
            )
        if station["measurement_unit"] != "mm":
            raise HTTPException(
                503,
                {
                    "code": "UNIT_UNAVAILABLE",
                    "message": "Observed water-level unit must be confirmed as mm.",
                },
            )
        model_version, pipeline_version = resolve_forecast_identity(request, station_id, query)
        rows = (
            connection.execute(
                forecast_comparison_query(
                    station_id, model_version, pipeline_version, query, datetime.now(timezone.utc)
                )
            )
            .mappings()
            .all()
        )
    has_more = len(rows) > query.limit
    points = [ForecastComparisonPoint(**row) for row in rows[: query.limit]]
    return ForecastHistoryResponse(
        station_id=station_id,
        model_version=model_version,
        pipeline_version=pipeline_version,
        horizon_minutes=query.horizon_minutes,
        start=query.start.astimezone(timezone.utc),
        end=query.end.astimezone(timezone.utc),
        points=points,
        has_more=has_more,
        next_before=points[-1].forecast_at if has_more else None,
    )
