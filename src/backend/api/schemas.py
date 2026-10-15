"""Public input-form values use original units, never standardized tensor values."""

from datetime import date, datetime, timedelta
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
    station_type: Literal["water_level", "rainfall", "wind"]
    measurement_units: dict[str, str | None] | None = None
    station_subtype: str | None = None
    station_subtype_label: str | None = None
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


class HourlyHistoryQuery(BaseModel):
    """Calendar dates are inclusive in Vietnam time; omitted dates select today."""

    model_config = ConfigDict(extra="forbid")
    start_date: date | None = Field(default=None, ge=date(1900, 1, 1))
    end_date: date | None = Field(default=None, ge=date(1900, 1, 1))
    station_type: Literal["water_level", "rainfall", "wind"] | None = None
    station_id: str | None = Field(default=None, min_length=1, max_length=100)
    station_name: str | None = Field(default=None, min_length=1, max_length=100)
    limit: int = Field(default=50, ge=1, le=100)
    cursor_date: date | None = Field(default=None, ge=date(1900, 1, 1))
    cursor_station_id: str | None = Field(default=None, min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_cursor_and_dates(self):
        if (self.cursor_date is None) != (self.cursor_station_id is None):
            raise ValueError("cursor_date and cursor_station_id must be supplied together.")
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date must not be after end_date.")
        return self


class HourlyHistoryCursor(BaseModel):
    date: date
    station_id: str


class HourlyHistoryCell(BaseModel):
    """Scheduled hour in UTC; value is an exact raw reading, never an hourly total."""

    hour: int = Field(ge=0, le=23)
    at: AwareDatetime
    value: float | None = Field(allow_inf_nan=False)
    status: Literal["observed", "missing", "future"]


class HourlyHistoryRow(BaseModel):
    date: date
    station_id: str
    name: str
    station_type: Literal["water_level", "rainfall", "wind"]
    unit: str | None
    cells: list[HourlyHistoryCell] = Field(min_length=24, max_length=24)


class HourlyHistoryResponse(BaseModel):
    timezone: Literal["Asia/Ho_Chi_Minh"] = "Asia/Ho_Chi_Minh"
    start_date: date
    end_date: date
    as_of: AwareDatetime
    rows: list[HourlyHistoryRow]
    has_more: bool
    next_cursor: HourlyHistoryCursor | None


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


class ActualWaterLevelPoint(BaseModel):
    """Observed mean for the hour (timestamp - 1h, timestamp], not a raw reading."""

    timestamp: AwareDatetime
    actual_water_level_mm: float | None = Field(allow_inf_nan=False)
    actual_sample_count: int = Field(ge=0)


class ForecastActualsQuery(BaseModel):
    """Bounded observed comparison window; no model input or forecast persistence."""

    model_config = ConfigDict(extra="forbid")
    anchor_observed_at: AwareDatetime
    horizon_minutes: int = Field(ge=1, le=10080, description="At most seven days.")

    @model_validator(mode="after")
    def validate_anchor(self):
        if self.anchor_observed_at.year < 1900:
            raise ValueError("Anchor must be in or after 1900.")
        return self


class ForecastActualsResponse(BaseModel):
    """Hourly observed means for a transient scenario, without storing its inputs."""

    station_id: str
    anchor_observed_at: AwareDatetime
    window_end: AwareDatetime
    as_of: AwareDatetime
    unit: Literal["mm"] = "mm"
    actuals: list[ActualWaterLevelPoint]


class ForecastDetailResponse(BaseModel):
    """One immutable saved prediction and its current hourly observed comparison."""

    forecast_id: UUID
    station_id: str
    model_version: str
    feature_schema_version: str
    pipeline_version: str
    anchor_observed_at: AwareDatetime
    generated_at: AwareDatetime
    unit: Literal["mm"] = "mm"
    window_end: AwareDatetime
    predictions: list[PredictedWaterLevel]
    actuals: list[ActualWaterLevelPoint]


class PredictionOptionsResponse(BaseModel):
    station_id: str
    model_version: str
    forecast_horizons_minutes: list[int]


class RangePredictionRequest(BaseModel):
    """One bounded chunk of target hours [start, end), without user overrides."""

    model_config = ConfigDict(extra="forbid")
    station_id: str = Field(min_length=1)
    start: AwareDatetime
    end: AwareDatetime
    horizon_minutes: int = Field(gt=0)
    model_version: str | None = None
    pipeline_version: str | None = None

    @model_validator(mode="after")
    def validate_window(self):
        from datetime import timezone

        if self.start >= self.end or self.end - self.start > timedelta(hours=24):
            raise ValueError("Chunk must span at most 24 hours with start < end.")
        for instant in (self.start, self.end):
            utc = instant.astimezone(timezone.utc)
            if utc.year < 1900 or utc.minute or utc.second or utc.microsecond:
                raise ValueError("Bounds must be whole UTC hours in or after 1900.")
        if (self.model_version is None) != (self.pipeline_version is None):
            raise ValueError("Model and pipeline versions must be provided together.")
        return self


class RangePredictionPoint(BaseModel):
    forecast_at: AwareDatetime
    status: Literal["ready", "insufficient_history", "anchor_unavailable", "future_anchor"]
    forecast_id: UUID | None = None
    anchor_observed_at: AwareDatetime | None = None
    predicted_water_level_mm: float | None = Field(default=None, allow_inf_nan=False)
    actual_water_level_mm: float | None = Field(allow_inf_nan=False)
    actual_sample_count: int = Field(ge=0)
    reused: bool | None = None


class RangePredictionResponse(BaseModel):
    station_id: str
    start: AwareDatetime
    end: AwareDatetime
    horizon_minutes: int
    model_version: str
    pipeline_version: str
    unit: Literal["mm"] = "mm"
    points: list[RangePredictionPoint]
    created_count: int
    reused_count: int
    skipped_count: int
