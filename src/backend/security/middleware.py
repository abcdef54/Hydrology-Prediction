"""Enforce request admission before parsing JSON or entering synchronous API work."""

import asyncio
import logging
from uuid import uuid4

from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from src.backend.api.schemas import ErrorDetail, ErrorResponse

logger = logging.getLogger(__name__)


def rejection(
    scope: Scope, status: int, code: str, message: str, retry_after: int | None = None
) -> JSONResponse:
    """Build the standard public error envelope for a middleware rejection.

    Args:
        scope: HTTP scope with an assigned state.request_id.
        status: HTTP status, such as 413 or 429.
        code: Stable public error code.
        message: Public explanation without secrets or internal paths.
        retry_after: Optional positive seconds before the client should retry.

    Returns:
        JSON response; the surrounding middleware adds X-Request-ID and CORS.
    """
    payload = ErrorResponse(
        error=ErrorDetail(code=code, message=message),
        request_id=scope["state"]["request_id"],
    )
    headers = {"Retry-After": str(retry_after)} if retry_after is not None else None
    return JSONResponse(payload.model_dump(mode="json"), status_code=status, headers=headers)


class PublicApiMiddleware:
    """Limit origins, per-IP requests and complete request bodies before routing.

    Args:
        app: Inner ASGI application. FastAPI lifespan supplies validated settings
            and the owned/injected rate limiter through scope.app.state.

    Bodies are buffered only up to the configured size. Actual bytes are counted,
    even without Content-Length; slow bodies receive a total deadline. No request
    timeout releases an inference slot while a model thread is still executing.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        scope.setdefault("state", {})["request_id"] = str(uuid4())
        request_id = scope["state"]["request_id"]
        response_started = False

        async def send_with_request_id(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
                MutableHeaders(scope=message)["X-Request-ID"] = request_id
            await send(message)

        settings = scope["app"].state.security_settings
        headers = Headers(scope=scope)
        origin = headers.get("origin")
        if origin is not None and origin not in settings.cors_origins:
            await rejection(scope, 403, "ORIGIN_NOT_ALLOWED", "Browser origin is not allowed.")(
                scope, receive, send_with_request_id
            )
            return
        client = scope.get("client")
        client_ip = client[0] if client else "unknown"
        is_prediction = (
            scope["method"] == "POST" and scope["path"].rstrip("/") == "/api/v1/predictions"
        )
        retry_after = await run_in_threadpool(
            scope["app"].state.request_rate_limiter.check, client_ip, is_prediction
        )
        if retry_after:
            await rejection(
                scope, 429, "RATE_LIMITED", "Request limit exceeded. Try again later.", retry_after
            )(scope, receive, send_with_request_id)
            return
        lengths = headers.getlist("content-length")
        try:
            if len(lengths) > 1:
                raise ValueError()
            declared_size = int(lengths[0]) if lengths else 0
            if declared_size < 0:
                raise ValueError()
        except ValueError:
            await rejection(scope, 400, "INVALID_REQUEST", "Invalid Content-Length header.")(
                scope, receive, send_with_request_id
            )
            return
        if declared_size > settings.max_body_bytes:
            await rejection(
                scope, 413, "REQUEST_TOO_LARGE", "Request body exceeds the size limit."
            )(scope, receive, send_with_request_id)
            return
        if headers.get("content-encoding", "identity").lower() != "identity":
            await rejection(
                scope,
                415,
                "UNSUPPORTED_CONTENT_ENCODING",
                "Compressed request bodies are not supported.",
            )(scope, receive, send_with_request_id)
            return

        body = bytearray()
        received_bytes = 0
        too_large = False
        try:
            async with asyncio.timeout(settings.body_timeout_seconds):
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return
                    chunk = message.get("body", b"")
                    received_bytes += len(chunk)
                    if received_bytes > settings.max_body_bytes:
                        too_large = True
                        break
                    body.extend(chunk)
                    if not message.get("more_body", False):
                        break
        except TimeoutError:
            await rejection(
                scope, 408, "REQUEST_BODY_TIMEOUT", "Request body was not received in time."
            )(scope, receive, send_with_request_id)
            return
        if too_large:
            await rejection(
                scope, 413, "REQUEST_TOO_LARGE", "Request body exceeds the size limit."
            )(scope, receive, send_with_request_id)
            return
        complete_body = bytes(body)
        delivered = False

        async def replay_body() -> Message:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": complete_body, "more_body": False}
            return await receive()

        try:
            await self.app(scope, replay_body, send_with_request_id)
        except Exception:
            if response_started:
                raise
            # Handle route bugs inside CORSMiddleware too, so browser clients can
            # read the standard 500 response instead of a misleading CORS error.
            logger.error("Unexpected request failure; request_id=%s", request_id)
            await rejection(scope, 500, "INTERNAL_ERROR", "An internal error occurred.")(
                scope, receive, send_with_request_id
            )
