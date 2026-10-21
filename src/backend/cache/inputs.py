"""Bounded process cache invalidated by source observation generations in Redis."""

from collections import OrderedDict
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from threading import Lock
from time import monotonic

from redis.exceptions import RedisError

from src.backend.cache.latest import LatestObservationCache


class ObservationDependentCache:
    """Reuse derived data while all source generations remain unchanged.

    Args:
        client: Borrowed Redis client; None disables caching.
        ttl_seconds: Maximum lifetime even if invalidation fails.
        max_entries: Bound on process memory usage, shared by all stations.

    Redis failures bypass the cache. Values are copied so callers cannot edit
    cached tensors or tables. Only successful loads are retained; no user edits.
    """

    def __init__(self, client=None, ttl_seconds=600, max_entries=256):
        self.client = client
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.entries = OrderedDict()
        self.lock = Lock()

    def _generation(self, source_ids):
        if self.client is None:
            return None
        try:
            return tuple(
                self.client.mget(
                    [LatestObservationCache.keys(station_id)[1] for station_id in source_ids]
                )
            )
        except RedisError:
            return None

    def load(self, key, source_ids, loader, retain=lambda value: True):
        """Read or load derived data, guarding against concurrent ingestion.

        Args:
            key: Hashable identity including model/config and selected time.
            source_ids: All stations on which the value depends.
            loader: Called on miss; errors propagate without being cached.
            retain: Predicate excluding incomplete results from storage.

        Returns:
            Independent value loaded from source or copied from cache.

        Raises:
            Exception: Any loader error, unchanged for normal API handling.
        """
        generation = self._generation(source_ids)
        if generation is None:
            return loader()
        with self.lock:
            entry = self.entries.get(key)
            if entry is not None:
                expires, version, value = entry
                if expires > monotonic() and version == generation:
                    self.entries.move_to_end(key)
                    return deepcopy(value)
                del self.entries[key]
        value = loader()
        if retain(value) and self._generation(source_ids) == generation:
            with self.lock:
                self.entries[key] = (monotonic() + self.ttl_seconds, generation, deepcopy(value))
                self.entries.move_to_end(key)
                while len(self.entries) > self.max_entries:
                    self.entries.popitem(last=False)
        return value

    def prepare(self, service, reference_time=None, overrides=None):
        """Reuse original model inputs for forms/inference; edits bypass storage.

        Args:
            service: Station-specific input service and loaded preprocessor.
            reference_time: Historical timestamp or None for latest.
            overrides: User edits; never retained in this cache.

        Returns:
            PreparedInput with current request context and independent feature data.

        Raises:
            Exception: Input preparation or artifact access errors from the service.
        """
        if overrides:
            return service.prepare(reference_time, overrides)
        if self.client is None or service.config.weather_location_id is not None:
            # Observation generations do not track new weather vintages.
            return service.prepare(reference_time)
        config = service.config
        artifact = config.preprocessing_path.stat()
        selected = reference_time.astimezone(timezone.utc) if reference_time else None
        hour = (selected or datetime.now(timezone.utc)).replace(minute=0, second=0, microsecond=0)
        key = (
            "inputs",
            repr(config),
            artifact.st_mtime_ns,
            artifact.st_size,
            "latest" if selected is None else "historical",
            hour,
        )
        prepared = self.load(
            key,
            (config.station_id, *config.rain_station_ids),
            lambda: service.prepare(reference_time),
        )
        return replace(prepared, requested_reference_time=selected)
