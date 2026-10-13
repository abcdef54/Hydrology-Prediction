"""Bound per-IP request rates with shared Redis counters and a bounded local fallback."""

from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
import logging
import math
import os
from threading import Lock
import time
from typing import Callable

from redis import Redis
from redis.backoff import NoBackoff
from redis.exceptions import RedisError
from redis.retry import Retry

from .config import SecuritySettings

logger = logging.getLogger(__name__)
WINDOW_SECONDS = 60
RATE_LIMIT_SCRIPT = """
local requests = tonumber(redis.call('GET', KEYS[1]) or '0')
local predictions = tonumber(redis.call('GET', KEYS[2]) or '0')
local retry_after = 0
if requests >= tonumber(ARGV[1]) then
    retry_after = math.max(retry_after, redis.call('TTL', KEYS[1]), 1)
end
if ARGV[3] == '1' and predictions >= tonumber(ARGV[2]) then
    retry_after = math.max(retry_after, redis.call('TTL', KEYS[2]), 1)
end
if retry_after > 0 then return retry_after end
if redis.call('INCR', KEYS[1]) == 1 then
    redis.call('EXPIRE', KEYS[1], ARGV[4])
end
if ARGV[3] == '1' then
    if redis.call('INCR', KEYS[2]) == 1 then
        redis.call('EXPIRE', KEYS[2], math.max(redis.call('TTL', KEYS[1]), 1))
    end
end
return 0
"""


@dataclass
class ClientWindow:
    expires_at: float
    requests: int = 0
    predictions: int = 0


class RequestRateLimiter:
    """Admit requests under both local and optional shared per-IP quotas.

    Args:
        settings: Per-minute quotas and maximum local client entries.
        client: Dedicated Redis client owned by this limiter, or None for local limits.
        clock: Monotonic seconds, injectable to test expiry without sleeping.
        namespace: Redis key prefix; tests use isolated namespaces.

    Local counters run even while Redis is healthy, so an outage cannot reset an
    IP's local quota. State is bounded and protected by a thread lock. A full local
    table rejects new IPs until the oldest window expires; it never evicts active IPs.
    """

    def __init__(
        self,
        settings: SecuritySettings,
        client: Redis | None = None,
        clock: Callable[[], float] = time.monotonic,
        namespace: str = "hydrology:rate:v1",
    ):
        self.settings = settings
        self.client = client
        self.clock = clock
        self.namespace = namespace
        self.windows: OrderedDict[str, ClientWindow] = OrderedDict()
        self.lock = Lock()

    @classmethod
    def from_env(cls, settings: SecuritySettings) -> "RequestRateLimiter":
        """Configure an independent Redis pool, or local-only limits without REDIS_URL.

        Args:
            settings: Validated API security settings.

        Returns:
            Limiter with 0.5s Redis timeouts and no retries. No connection is opened.

        Raises:
            ValueError: If REDIS_URL is malformed; credentials are never included.
        """
        url = os.environ.get("REDIS_URL")
        client = None
        if url:
            try:
                client = Redis.from_url(
                    url,
                    decode_responses=True,
                    socket_connect_timeout=0.5,
                    socket_timeout=0.5,
                    retry=Retry(NoBackoff(), 0),
                )
            except ValueError:
                raise ValueError("REDIS_URL must be a valid Redis connection URL.") from None
        return cls(settings, client)

    def keys(self, client_ip: str) -> tuple[str, str]:
        identity = sha256(client_ip.encode()).hexdigest()
        return f"{self.namespace}:{identity}:all", f"{self.namespace}:{identity}:predict"

    def check(self, client_ip: str, is_prediction: bool) -> int:
        """Consume a local quota and, when available, a shared Redis quota.

        Args:
            client_ip: Client address from the ASGI scope, not an untrusted header.
            is_prediction: Whether this is POST /api/v1/predictions.

        Returns:
            Zero when admitted; positive Retry-After seconds when blocked. Redis
            errors retain the local limit. Invalid requests also consume quota.
        """
        retry_after = self._check_local(client_ip, is_prediction)
        if retry_after or self.client is None:
            return retry_after
        try:
            return int(
                self.client.eval(
                    RATE_LIMIT_SCRIPT,
                    2,
                    *self.keys(client_ip),
                    self.settings.requests_per_minute,
                    self.settings.predictions_per_minute,
                    int(is_prediction),
                    WINDOW_SECONDS,
                )
            )
        except RedisError as error:
            logger.warning(
                "Rate limiter Redis unavailable; using local limits; error_type=%s",
                type(error).__name__,
            )
            return 0

    def _check_local(self, client_ip: str, is_prediction: bool) -> int:
        now = self.clock()
        with self.lock:
            while self.windows and next(iter(self.windows.values())).expires_at <= now:
                self.windows.popitem(last=False)
            window = self.windows.get(client_ip)
            if window is None:
                if len(self.windows) >= self.settings.max_tracked_clients:
                    return max(1, math.ceil(next(iter(self.windows.values())).expires_at - now))
                window = ClientWindow(now + WINDOW_SECONDS)
                self.windows[client_ip] = window
            if window.requests >= self.settings.requests_per_minute or (
                is_prediction and window.predictions >= self.settings.predictions_per_minute
            ):
                return max(1, math.ceil(window.expires_at - now))
            window.requests += 1
            window.predictions += int(is_prediction)
        return 0

    def close(self) -> None:
        """Release the owned Redis pool when the API process stops."""
        if self.client is not None:
            self.client.close()
