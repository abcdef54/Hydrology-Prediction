"""FastAPI entry point; imports never connect to PostgreSQL or load artifacts."""

from contextlib import asynccontextmanager
import logging
import math
import os
from pathlib import Path
from threading import BoundedSemaphore, Lock

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from starlette.exceptions import HTTPException
from starlette.middleware.cors import CORSMiddleware

from src.backend.api.prediction_inputs import router as prediction_inputs_router
from src.backend.api.observations import router as observations_router
from src.backend.api.hourly_history import router as hourly_history_router
from src.backend.api.prediction_availability import router as prediction_availability_router
from src.backend.api.predictions import router as predictions_router
from src.backend.api.forecast_history import router as forecast_history_router
from src.backend.api.forecast_detail import router as forecast_detail_router
from src.backend.api.schemas import ErrorDetail, ErrorResponse
from src.backend.api.stations import router as stations_router
from src.backend.cache.latest import LatestObservationCache
from src.backend.db.session import build_engine
from src.backend.prediction.config import StationModelRegistry
from src.backend.security.config import SecuritySettings
from src.backend.security.middleware import PublicApiMiddleware
from src.backend.security.rate_limit import RequestRateLimiter

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def error_response(
    request: Request, status: int, code: str, message: str, details: dict | None = None
) -> JSONResponse:
    """Return consistent errors without database credentials or artifact paths.

    Args:
        request: Request whose middleware has assigned state.request_id.
        status: HTTP response status, such as 400, 404, 422 or 503.
        code: Stable application error code for frontend handling.
        message: Public error explanation; callers must exclude secrets/internal paths.
        details: Optional structured public context, such as required/available counts.

    Returns:
        JSONResponse with the standard error envelope and the current request ID.
    """
    payload = ErrorResponse(
        error=ErrorDetail(code=code, message=message, details=details or {}),
        request_id=request.state.request_id,
    )
    return JSONResponse(status_code=status, content=payload.model_dump(mode="json"))


def create_app(
    *,
    engine: Engine | None = None,
    registry: StationModelRegistry | None = None,
    stale_after_minutes: float | None = None,
    latest_cache: LatestObservationCache | None = None,
    security_settings: SecuritySettings | None = None,
    rate_limiter: RequestRateLimiter | None = None,
) -> FastAPI:
    """Create the API with optional existing resources for tests or embedding.

    Production reads DATABASE_URL, optional MODEL_ARTIFACT_ROOT/MODEL_CONFIG_PATH,
    and INPUT_STALE_AFTER_MINUTES (default 120). Only an engine owned by this app
    is disposed at shutdown. Prediction runtimes load lazily per station; form
    requests never load weights. PREDICTION_DEVICE defaults to cpu.

    Args:
        engine: Existing SQLAlchemy engine for tests/embedding; None creates an
            application-owned engine from DATABASE_URL during startup.
        registry: Existing station model registry; None loads the configured JSON
            during startup. No model checkpoint is loaded by this factory.
        stale_after_minutes: Positive finite freshness threshold; None reads
            INPUT_STALE_AFTER_MINUTES during startup, defaulting to 120.
        latest_cache: Existing cache owned by the caller; None configures an
            application-owned Redis cache from REDIS_URL during startup. Without
            REDIS_URL caching is disabled. No Redis connection is required to start.
        security_settings: Explicit origins/admission limits; None reads environment.
        rate_limiter: Caller-owned limiter for tests/embedding; None configures an
            app-owned limiter using Redis when configured, with local fallback.

    Returns:
        FastAPI application with routes, request IDs, error handlers and resource
        lifecycle. Creating the app alone does not open a database connection.

    Raises:
        ValueError: During startup if freshness/configuration settings are invalid.
        RuntimeError: During startup if a required DATABASE_URL is absent.
        OSError: During startup if the model configuration file cannot be read.
    """

    configured_security = (
        security_settings if security_settings is not None else SecuritySettings.from_env()
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        artifact_root = Path(os.environ.get("MODEL_ARTIFACT_ROOT", PROJECT_ROOT)).resolve()
        config_path = Path(
            os.environ.get(
                "MODEL_CONFIG_PATH", PROJECT_ROOT / "src/backend/prediction/model_configs.json"
            )
        )
        configured_threshold = stale_after_minutes
        if configured_threshold is None:
            configured_threshold = float(os.environ.get("INPUT_STALE_AFTER_MINUTES", "120"))
        if not math.isfinite(configured_threshold) or configured_threshold <= 0:
            raise ValueError("INPUT_STALE_AFTER_MINUTES must be a positive finite number.")
        application.state.model_registry = (
            registry
            if registry is not None
            else StationModelRegistry.from_json(config_path, artifact_root)
        )
        application.state.database_engine = engine if engine is not None else build_engine()
        application.state.latest_observation_cache = (
            latest_cache if latest_cache is not None else LatestObservationCache.from_env()
        )
        application.state.stale_after_minutes = configured_threshold
        application.state.prediction_device = os.environ.get("PREDICTION_DEVICE", "cpu")
        application.state.prediction_services = {}
        application.state.prediction_service_lock = Lock()
        application.state.security_settings = configured_security
        application.state.prediction_slots = BoundedSemaphore(
            configured_security.max_concurrent_predictions
        )
        application.state.request_rate_limiter = (
            rate_limiter
            if rate_limiter is not None
            else RequestRateLimiter.from_env(configured_security)
        )
        try:
            yield
        finally:
            application.state.prediction_services.clear()
            if rate_limiter is None:
                application.state.request_rate_limiter.close()
            if latest_cache is None:
                application.state.latest_observation_cache.close()
            if engine is None:
                application.state.database_engine.dispose()

    application = FastAPI(
        title="Hydrology Prediction API",
        version="0.1.0",
        lifespan=lifespan,
        responses={status: {"model": ErrorResponse} for status in (403, 408, 413, 415, 429)},
    )

    @application.exception_handler(HTTPException)
    async def handle_http_error(request: Request, error: HTTPException):
        detail = error.detail
        if isinstance(detail, dict) and "code" in detail:
            response = error_response(
                request, error.status_code, detail["code"], detail["message"], detail.get("details")
            )
        else:
            response = error_response(request, error.status_code, "HTTP_ERROR", str(detail))
        response.headers.update(error.headers or {})
        return response

    @application.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, error: RequestValidationError):
        issues = [
            {"location": list(issue["loc"]), "message": issue["msg"], "type": issue["type"]}
            for issue in error.errors()
        ]
        return error_response(
            request, 422, "INVALID_REQUEST", "Request validation failed.", {"issues": issues}
        )

    @application.exception_handler(SQLAlchemyError)
    async def handle_database_error(request: Request, error: SQLAlchemyError):
        logger.error("Database request failed; request_id=%s", request.state.request_id)
        return error_response(
            request, 503, "DATA_SERVICE_UNAVAILABLE", "Observation database is unavailable."
        )

    @application.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, error: Exception):
        logger.error("Unexpected request failure; request_id=%s", request.state.request_id)
        response = error_response(request, 500, "INTERNAL_ERROR", "An internal error occurred.")
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    application.include_router(stations_router)
    application.include_router(observations_router)
    application.include_router(hourly_history_router)
    application.include_router(prediction_availability_router)
    application.include_router(prediction_inputs_router)
    application.include_router(predictions_router)
    application.include_router(forecast_history_router)
    application.include_router(forecast_detail_router)
    application.add_middleware(PublicApiMiddleware)
    # Last added is outermost: CORS also wraps admission errors and safe route 500s.
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(configured_security.cors_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["Accept", "Content-Type"],
        allow_credentials=False,
        expose_headers=["X-Request-ID", "Retry-After"],
        max_age=600,
    )
    return application


app = create_app()
