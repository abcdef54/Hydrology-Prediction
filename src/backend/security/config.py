"""Explicit API limits and browser origins, configurable independently of models."""

from dataclasses import dataclass
import math
import os
from urllib.parse import urlsplit


@dataclass(frozen=True)
class SecuritySettings:
    """Limits applied by each API process; rate counters can also be shared in Redis.

    Args:
        cors_origins: Exact browser origins; defaults allow the local Vite frontend.
        requests_per_minute: Maximum admitted requests per client IP in a 60s window.
        predictions_per_minute: Additional per-IP limit for POST /predictions.
        max_body_bytes: Maximum actual request-body size, including chunked bodies.
        body_timeout_seconds: Maximum total time to receive a complete request body.
        max_concurrent_predictions: Nonqueued prediction slots per API process.
        max_tracked_clients: Bound on IP entries in the local fallback rate counter.

    Positive limits are initial safeguards, not measured production capacity.
    """

    cors_origins: tuple[str, ...] = (
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    )
    requests_per_minute: int = 120
    predictions_per_minute: int = 20
    max_body_bytes: int = 16384
    body_timeout_seconds: float = 10.0
    max_concurrent_predictions: int = 1
    max_tracked_clients: int = 10000

    def __post_init__(self):
        for name in (
            "requests_per_minute",
            "predictions_per_minute",
            "max_body_bytes",
            "max_concurrent_predictions",
            "max_tracked_clients",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if not math.isfinite(self.body_timeout_seconds) or self.body_timeout_seconds <= 0:
            raise ValueError("body_timeout_seconds must be positive and finite.")
        for origin in self.cors_origins:
            try:
                parsed = urlsplit(origin)
                port = parsed.port
                valid = (
                    parsed.scheme in {"http", "https"}
                    and parsed.hostname
                    and not parsed.username
                    and not parsed.password
                    and not parsed.path
                    and not parsed.query
                    and not parsed.fragment
                    and (port is None or port > 0)
                    and "*" not in origin
                )
            except ValueError:
                valid = False
            if not valid:
                raise ValueError("CORS_ORIGINS must contain exact HTTP(S) origins without paths.")

    @classmethod
    def from_env(cls) -> "SecuritySettings":
        """Read limits and comma-separated CORS_ORIGINS without opening connections.

        Returns:
            Validated settings. Empty CORS_ORIGINS permits no cross-origin browser UI.

        Raises:
            ValueError: If an origin or numeric limit is invalid.
        """
        defaults = cls()
        configured = {}
        for field_name, env_name in (
            ("requests_per_minute", "API_RATE_LIMIT_PER_MINUTE"),
            ("predictions_per_minute", "PREDICTION_RATE_LIMIT_PER_MINUTE"),
            ("max_body_bytes", "MAX_REQUEST_BODY_BYTES"),
            ("max_concurrent_predictions", "MAX_CONCURRENT_PREDICTIONS"),
            ("max_tracked_clients", "RATE_LIMIT_MAX_CLIENTS"),
        ):
            try:
                configured[field_name] = int(
                    os.environ.get(env_name, getattr(defaults, field_name))
                )
            except ValueError:
                raise ValueError(f"{env_name} must be a positive integer.") from None
        try:
            configured["body_timeout_seconds"] = float(
                os.environ.get("REQUEST_BODY_TIMEOUT_SECONDS", "10")
            )
        except ValueError:
            raise ValueError("REQUEST_BODY_TIMEOUT_SECONDS must be positive and finite.") from None
        origins = os.environ.get("CORS_ORIGINS", ",".join(defaults.cors_origins))
        configured["cors_origins"] = tuple(
            origin.strip() for origin in origins.split(",") if origin.strip()
        )
        return cls(**configured)
