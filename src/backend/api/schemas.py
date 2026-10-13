"""Public input-form values use original units, never standardized tensor values."""

from datetime import datetime
from typing import Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field


class PredictionInputsQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reference_time: AwareDatetime | None = Field(
        default=None,
        description="ISO 8601 with timezone; rounded down to the hour. Omit for latest data.",
    )


class InputFeature(BaseModel):
    name: str
    value: float | None = Field(description="Original value; null means missing, not zero.")
    unit: str | None = Field(description="null means the source unit is not confirmed.")
    source_station_id: str | None
    editable: bool = True


class PredictionInputsResponse(BaseModel):
    station_id: str
    model_version: str
    feature_schema_version: str
    requested_reference_time: datetime | None
    hourly_reference_time: datetime
    anchor_observed_at: datetime
    sequence_length: int
    feature_count: int
    forecast_horizons_minutes: list[int]
    features: list[InputFeature] = Field(description="Last timestep, in model feature order.")
    editable_features: list[str]
    missing_features: list[str]
    input_ready: bool
    is_stale: bool = Field(description="True when input age exceeds the configured threshold.")
    input_age_minutes: float = Field(
        description="Age of anchor versus now (latest) or the rounded historical reference."
    )
    stale_after_minutes: float


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorDetail
    request_id: str
