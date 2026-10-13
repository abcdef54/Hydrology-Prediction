"""Public input-form values use original units, never standardized tensor values."""

from datetime import datetime, timedelta
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator


class StationLocation(BaseModel):
    latitude: float | None
    longitude: float | None
    address: str | None


class StationSummary(BaseModel):
    station_id: str
    name: str
    station_type: Literal["water_level", "rainfall"]
    river_name: str | None
    location: StationLocation
    measurement_unit: str | None
    model_configured: bool = Field(description="A model configuration exists for this station.")
    prediction_enabled: bool = Field(
        description="Station permits prediction and has its own model configuration; "
        "does not guarantee artifact availability or sufficient observations."
    )


class StationsResponse(BaseModel):
    stations: list[StationSummary]


class AlarmLevels(BaseModel):
    """Stored alarm thresholds in metres; null means not configured."""

    level_1: float | None
    level_2: float | None
    level_3: float | None


class StationDetail(StationSummary):
    enabled: bool
    alarm_levels_m: AlarmLevels


class LatestObservationResponse(BaseModel):
    """Latest valid raw measurement; no hourly aggregation or model scaling."""

    station_id: str
    station_type: Literal["water_level", "rainfall"]
    value: float = Field(allow_inf_nan=False)
    unit: str | None = Field(description="null means the source unit is not confirmed.")
    observed_at: AwareDatetime
    received_at: AwareDatetime
    quality_status: Literal["observed"]
    is_stale: bool
    data_age_minutes: float = Field(ge=0)
    stale_after_minutes: float


class ObservationHistoryQuery(BaseModel):
    """Read raw history in [start, end); before is an exclusive paging cursor."""

    model_config = ConfigDict(extra="forbid")
    start: AwareDatetime
    end: AwareDatetime
    limit: int = Field(default=100, ge=1, le=1000)
    before: AwareDatetime | None = None

    @model_validator(mode="after")
    def validate_time_range(self):
        if self.start >= self.end:
            raise ValueError("start must be earlier than end.")
        return self


class HistoricalObservation(BaseModel):
    observed_at: AwareDatetime
    received_at: AwareDatetime
    value: float | None = Field(allow_inf_nan=False)
    quality_status: Literal["observed", "missing"]


class ObservationHistoryResponse(BaseModel):
    station_id: str
    station_type: Literal["water_level", "rainfall"]
    unit: str | None
    start: AwareDatetime
    end: AwareDatetime
    observations: list[HistoricalObservation]
    has_more: bool
    next_before: AwareDatetime | None = Field(
        description="Pass as before with the same start/end for the next page."
    )


class PredictionInputsQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reference_time: AwareDatetime | None = Field(
        default=None,
        description="ISO 8601 with timezone; rounded down to the hour. Omit for latest data.",
    )


class ForecastHistoryQuery(BaseModel):
    """Select one forecast series by target time; before pages toward older points."""

    model_config = ConfigDict(extra="forbid")
    start: AwareDatetime
    end: AwareDatetime
    horizon_minutes: int = Field(gt=0)
    model_version: str | None = Field(default=None, min_length=1)
    pipeline_version: str | None = Field(default=None, min_length=1)
    limit: int = Field(default=100, ge=1, le=1000)
    before: AwareDatetime | None = None

    @model_validator(mode="after")
    def validate_time_range(self):
        if self.start >= self.end:
            raise ValueError("start must be earlier than end.")
        return self


class ForecastComparisonPoint(BaseModel):
    """A saved prediction and its current hourly observed truth, both in mm."""

    forecast_id: UUID
    feature_schema_version: str
    anchor_observed_at: AwareDatetime
    forecast_at: AwareDatetime
    generated_at: AwareDatetime
    predicted_water_level_mm: float = Field(allow_inf_nan=False)
    actual_water_level_mm: float | None = Field(allow_inf_nan=False)
    actual_sample_count: int = Field(ge=0)


class ForecastHistoryResponse(BaseModel):
    station_id: str
    model_version: str
    pipeline_version: str
    horizon_minutes: int
    unit: Literal["mm"] = "mm"
    start: AwareDatetime
    end: AwareDatetime
    points: list[ForecastComparisonPoint]
    has_more: bool
    next_before: AwareDatetime | None


class PredictionAvailabilityQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: AwareDatetime
    end: AwareDatetime

    @model_validator(mode="after")
    def validate_time_range(self):
        if self.start >= self.end:
            raise ValueError("start must be earlier than end.")
        if self.end - self.start > timedelta(days=7):
            raise ValueError("Availability range must not exceed 7 days.")
        return self


class PredictionAvailabilityPoint(BaseModel):
    reference_time: AwareDatetime
    input_ready: bool
    available_timesteps: int = Field(description="Capped at the required sequence length.")
    anchor_observed_at: AwareDatetime | None
    reason: Literal["INSUFFICIENT_HISTORY"] | None


class PredictionAvailabilityResponse(BaseModel):
    station_id: str
    model_version: str
    feature_schema_version: str
    sequence_length: int
    start: AwareDatetime
    end: AwareDatetime
    availability: list[PredictionAvailabilityPoint]


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


class PredictionRequest(BaseModel):
    """Edits use original feature names and units, and apply to the last timestep."""

    model_config = ConfigDict(extra="forbid")
    station_id: str = Field(strict=True, min_length=1)
    reference_time: AwareDatetime | None = None
    overrides: dict[str, Annotated[float, Field(strict=True, allow_inf_nan=False)] | None] = Field(
        default_factory=dict,
        description="Partial edits in original units; null is allowed only for missing rain/lag values.",
    )


class PredictedWaterLevel(BaseModel):
    horizon_minutes: int
    forecast_at: datetime
    water_level_mm: float = Field(allow_inf_nan=False)


class PredictionResponse(BaseModel):
    station_id: str
    model_version: str
    feature_schema_version: str
    requested_reference_time: datetime | None
    hourly_reference_time: datetime
    anchor_observed_at: datetime
    unit: Literal["mm"]
    predictions: list[PredictedWaterLevel]
    overrides_applied: dict[str, float | None]
    is_stale: bool
    input_age_minutes: float
    stale_after_minutes: float
    forecast_id: UUID | None = None
    generated_at: AwareDatetime | None = None
    pipeline_version: str | None = None
    reused: bool = False
