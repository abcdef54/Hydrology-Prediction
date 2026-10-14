"""Optional Redis cache of committed original-input forecasts, keyed by full identity."""

from dataclasses import asdict
from datetime import timedelta
import logging
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError
from redis.exceptions import RedisError

from src.backend.prediction.results import ForecastPoint, ForecastResult

logger = logging.getLogger(__name__)


class CachedPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    horizon_minutes: int = Field(gt=0)
    forecast_at: AwareDatetime
    water_level_mm: float = Field(allow_inf_nan=False)


class CachedForecast(BaseModel):
    model_config = ConfigDict(extra="forbid")
    station_id: str
    model_version: str
    feature_schema_version: str
    anchor_observed_at: AwareDatetime
    forecast_id: UUID
    generated_at: AwareDatetime
    pipeline_version: str
    predictions: list[CachedPoint]


class ForecastCache:
    """Cache committed output only; never store tensors, overrides or freshness.

    Args:
        client: Borrowed Redis client from latest cache, or None to disable.
        ttl_seconds: Positive entry lifetime; default 600 seconds.

    The caller prepares current inputs before reading. Full input/pipeline hashes
    prevent hits after data/scaler/model changes, without ingestion invalidation.
    The owner of the borrowed client closes it. Redis failures fall back to DB.
    """

    def __init__(self, client=None, ttl_seconds=600):
        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int) or ttl_seconds <= 0:
            raise ValueError("Forecast cache TTL must be a positive integer.")
        self.client = client
        self.ttl_seconds = ttl_seconds

    @staticmethod
    def key(station_id, pipeline_version, input_hash):
        return f"hydrology:v1:forecast:{station_id}:{pipeline_version}:{input_hash}"

    def read(self, config, prepared, pipeline_version, input_hash):
        """Return validated output with current request times, or None on a miss.

        Args:
            config: Station-specific model and horizon contract.
            prepared: Current complete chain, used to validate anchor and context.
            pipeline_version: Identity of the loaded numerical pipeline.
            input_hash: Hash of every input value, mask and timestamp.

        Returns:
            ForecastResult with reused=True, or None for absent/corrupt/unavailable
            Redis. No query or request freshness is taken from cached data.
        """
        if self.client is None:
            return None
        try:
            payload = self.client.get(self.key(config.station_id, pipeline_version, input_hash))
            if payload is None:
                return None
            cached = CachedForecast.model_validate_json(payload)
            if (
                cached.station_id != config.station_id
                or cached.model_version != config.model_version
                or cached.feature_schema_version != config.feature_schema_version
                or cached.pipeline_version != pipeline_version
                or cached.anchor_observed_at != prepared.anchor_observed_at
                or tuple(point.horizon_minutes for point in cached.predictions)
                != config.forecast_horizons_minutes
                or any(
                    point.forecast_at
                    != prepared.anchor_observed_at + timedelta(minutes=point.horizon_minutes)
                    for point in cached.predictions
                )
            ):
                return None
            return ForecastResult(
                config.station_id,
                config.model_version,
                config.feature_schema_version,
                prepared.requested_reference_time,
                prepared.hourly_reference_time,
                prepared.anchor_observed_at,
                tuple(ForecastPoint(**point.model_dump()) for point in cached.predictions),
                forecast_id=cached.forecast_id,
                generated_at=cached.generated_at,
                pipeline_version=pipeline_version,
                reused=True,
            )
        except (RedisError, UnicodeError, ValidationError, ValueError) as error:
            logger.warning("Forecast cache read failed; error_type=%s", type(error).__name__)
            return None

    def store(self, input_hash, forecast):
        """Publish output only after PostgreSQL commit; never publish a scenario.

        Args:
            input_hash: Full current-chain hash used for the committed result.
            forecast: Result with non-null ID, generation time and pipeline identity.

        Returns:
            True when cached, False for scenarios/disabled/failed Redis.
        """
        if self.client is None or forecast.forecast_id is None:
            return False
        try:
            fields = asdict(forecast)
            cached = CachedForecast.model_validate(
                {name: fields[name] for name in CachedForecast.model_fields}
            )
            return bool(
                self.client.set(
                    self.key(forecast.station_id, forecast.pipeline_version, input_hash),
                    cached.model_dump_json(),
                    ex=self.ttl_seconds,
                )
            )
        except (RedisError, ValidationError, ValueError) as error:
            logger.warning("Forecast cache store failed; error_type=%s", type(error).__name__)
            return False
