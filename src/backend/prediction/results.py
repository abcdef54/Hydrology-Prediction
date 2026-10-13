"""Forecast values and request context shared by inference and persistence."""

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID


@dataclass(frozen=True)
class ForecastPoint:
    horizon_minutes: int
    forecast_at: datetime
    water_level_mm: float


@dataclass(frozen=True)
class ForecastResult:
    station_id: str
    model_version: str
    feature_schema_version: str
    requested_reference_time: datetime | None
    hourly_reference_time: datetime
    anchor_observed_at: datetime
    predictions: tuple[ForecastPoint, ...]
    unit: str = "mm"
    forecast_id: UUID | None = None
    generated_at: datetime | None = None
    pipeline_version: str | None = None
    reused: bool = False
