"""FastAPI entry point; imports never connect to PostgreSQL or load artifacts."""

from contextlib import asynccontextmanager
import logging
import math
import os
from pathlib import Path
from uuid import uuid4
from threading import Lock

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from starlette.exceptions import HTTPException

from src.backend.api.prediction_inputs import router as prediction_inputs_router
from src.backend.api.predictions import router as predictions_router
from src.backend.api.schemas import ErrorDetail, ErrorResponse
from src.backend.api.stations import router as stations_router
from src.backend.db.session import build_engine
from src.backend.prediction.config import StationModelRegistry

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def error_response(
    request: Request, status: int, code: str, message: str, details: dict | None = None
) -> JSONResponse:
    """Return consistent errors without database credentials or artifact paths."""
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
) -> FastAPI:
    """Create the API with optional existing resources for tests or embedding.

    Production reads DATABASE_URL, optional MODEL_ARTIFACT_ROOT/MODEL_CONFIG_PATH,
    and INPUT_STALE_AFTER_MINUTES (default 120). Only an engine owned by this app
    is disposed at shutdown. Prediction runtimes load lazily per station; form
    requests never load weights. PREDICTION_DEVICE defaults to cpu.
    """

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
        application.state.stale_after_minutes = configured_threshold
        application.state.prediction_device = os.environ.get("PREDICTION_DEVICE", "cpu")
        application.state.prediction_services = {}
        application.state.prediction_service_lock = Lock()
        try:
            yield
        finally:
            application.state.prediction_services.clear()
            if engine is None:
                application.state.database_engine.dispose()

    application = FastAPI(title="Hydrology Prediction API", version="0.1.0", lifespan=lifespan)

    @application.middleware("http")
    async def assign_request_id(request: Request, call_next):
        request.state.request_id = str(uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @application.exception_handler(HTTPException)
    async def handle_http_error(request: Request, error: HTTPException):
        detail = error.detail
        if isinstance(detail, dict) and "code" in detail:
            return error_response(
                request, error.status_code, detail["code"], detail["message"], detail.get("details")
            )
        return error_response(request, error.status_code, "HTTP_ERROR", str(detail))

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
    application.include_router(prediction_inputs_router)
    application.include_router(predictions_router)
    return application


app = create_app()
