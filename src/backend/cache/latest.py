"""Cache raw latest observations without freezing request-time freshness."""

from collections.abc import Mapping
from datetime import datetime
import logging
import os
from typing import Literal
from uuid import uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError
from redis import Redis
from redis.backoff import NoBackoff
from redis.exceptions import RedisError
from redis.retry import Retry

logger = logging.getLogger(__name__)

# The cache version changes on invalidation. An older DB reader must not repopulate
# the cache after a worker has committed new observations and invalidated it.
STORE_IF_CURRENT = """
if (redis.call('GET', KEYS[2]) or '') ~= ARGV[1] then
    return 0
end
redis.call('SET', KEYS[1], ARGV[2], 'EX', ARGV[3])
return 1
"""
INVALIDATE = """
redis.call('SET', KEYS[2], ARGV[1])
redis.call('DEL', KEYS[1])
return 1
"""


class CachedObservation(BaseModel):
    """Stable measurement fields in source units; excludes age and stale flags."""

    model_config = ConfigDict(extra="forbid")

    station_id: str
    depth: float = Field(strict=True, allow_inf_nan=False)
    observed_at: AwareDatetime
    received_at: AwareDatetime
    quality_status: Literal["observed"]


class LatestObservationCache:
    """Read, publish and invalidate latest measurements using optional Redis.

    Args:
        client: Dedicated Redis client owned by this cache, or None to disable it.
        ttl_seconds: Positive integer lifetime for measurement entries (default 600).

    Cache version keys have no TTL so a delayed DB reader cannot reuse an old version
    after invalidation. There is one small cache version key per changed station.
    Redis failures return misses/False and log only the exception type, never URLs.
    Call close() when the owning API/worker stops. No connection is opened here.
    """

    def __init__(self, client: Redis | None = None, ttl_seconds: int = 600):
        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int) or ttl_seconds <= 0:
            raise ValueError("CACHE_TTL_HYDROLOGY_SECONDS must be a positive integer.")
        self.client = client
        self.ttl_seconds = ttl_seconds

    @classmethod
    def from_env(cls) -> "LatestObservationCache":
        """Configure a client from REDIS_URL without contacting Redis.

        Returns:
            Cache with CACHE_TTL_HYDROLOGY_SECONDS (default 600). Missing REDIS_URL
            disables caching. Commands/connects time out after 0.5 seconds each,
            with no automatic retries, so an unavailable cache does not stall API work.

        Raises:
            ValueError: For an invalid TTL or Redis connection URL.
        """
        try:
            ttl_seconds = int(os.environ.get("CACHE_TTL_HYDROLOGY_SECONDS", "600"))
        except ValueError:
            raise ValueError("CACHE_TTL_HYDROLOGY_SECONDS must be a positive integer.") from None
        cache = cls(ttl_seconds=ttl_seconds)
        url = os.environ.get("REDIS_URL")
        if url:
            try:
                cache.client = Redis.from_url(
                    url,
                    decode_responses=True,
                    socket_connect_timeout=0.5,
                    socket_timeout=0.5,
                    retry=Retry(NoBackoff(), 0),
                )
            except ValueError:
                raise ValueError("REDIS_URL must be a valid Redis connection URL.") from None
        return cache

    @property
    def enabled(self) -> bool:
        return self.client is not None

    @staticmethod
    def keys(station_id: str) -> tuple[str, str]:
        key = f"hydrology:v1:latest:{station_id}"
        # Preserve the existing Redis key name for deployed API/worker compatibility.
        return key, f"{key}:generation"

    def read(self, station_id: str, now: datetime) -> tuple[str | None, CachedObservation | None]:
        """Read a cache version and measurement atomically before querying observations.

        Args:
            station_id: Exact internal station ID, including leading zeros.
            now: Timezone-aware request time; a cached future measurement is rejected.

        Returns:
            Cache version token and valid measurement, or (token, None) for a miss.
            A None token means disabled/unavailable Redis; callers must not publish.
            Corrupt entries, wrong station IDs and future timestamps are cache misses.
        """
        if self.client is None:
            return None, None
        try:
            payload, cache_version = self.client.mget(self.keys(station_id))
        except (RedisError, UnicodeError) as error:
            self._warn("read", station_id, error)
            return None, None
        if payload is None:
            logger.debug("Latest cache miss; station_id=%s", station_id)
            return cache_version or "", None
        try:
            observation = CachedObservation.model_validate_json(payload)
            if observation.station_id != station_id or observation.observed_at > now:
                raise ValueError("Cached station/time does not match this request.")
        except (ValidationError, ValueError) as error:
            self._warn("decode", station_id, error)
            return cache_version or "", None
        logger.debug("Latest cache hit; station_id=%s", station_id)
        return cache_version or "", observation

    def store(self, station_id: str, cache_version: str | None, observation: Mapping) -> bool:
        """Publish a DB measurement only if no invalidation happened during its read.

        Args:
            station_id: Station whose PostgreSQL measurement was read.
            cache_version: Token from read(), captured before the observation query.
            observation: DB fields depth, observed_at, received_at and quality_status.

        Returns:
            True if stored with TTL; False if disabled, invalidated, malformed or
            unavailable. Does not cache station metadata or request-time freshness.
        """
        if self.client is None or cache_version is None:
            return False
        try:
            payload = CachedObservation(station_id=station_id, **dict(observation))
            stored = self.client.eval(
                STORE_IF_CURRENT,
                2,
                *self.keys(station_id),
                cache_version,
                payload.model_dump_json(),
                self.ttl_seconds,
            )
        except (RedisError, ValidationError) as error:
            self._warn("store", station_id, error)
            return False
        return bool(stored)

    def invalidate(self, station_id: str) -> bool:
        """Remove one station's entry and advance its cache version after DB commit.

        Args:
            station_id: Station with inserted/corrected observations. Do not call
                before commit or for a rolled-back/dry-run transaction.

        Returns:
            True if invalidation succeeded or cache is disabled; False on Redis
            failure. Failed invalidation leaves existing entries bounded by their TTL.
        """
        if self.client is None:
            return True
        try:
            self.client.eval(INVALIDATE, 2, *self.keys(station_id), uuid4().hex)
        except RedisError as error:
            self._warn("invalidate", station_id, error)
            return False
        logger.debug("Latest cache invalidated; station_id=%s", station_id)
        return True

    def close(self) -> None:
        """Release this cache's connection pool; shutdown needs no live Redis server."""
        if self.client is not None:
            self.client.close()

    @staticmethod
    def _warn(operation: str, station_id: str, error: Exception) -> None:
        logger.warning(
            "Latest cache %s failed; station_id=%s; error_type=%s",
            operation,
            station_id,
            type(error).__name__,
        )
