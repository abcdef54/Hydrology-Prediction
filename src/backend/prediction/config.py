"""Each water station owns a separately trained model and preprocessing contract."""

from dataclasses import dataclass, field
import json
from pathlib import Path
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class StationModelConfig:
    """Describe one water station's trained model and preprocessing contract.

    Args:
        station_id: Exact water-station ID; keep leading zeros.
        model_type: Runtime architecture identifier, currently "residual_lstm".
        model_version: Identifier of the trained checkpoint/run.
        model_path: Resolved path to this model's trusted checkpoint.
        preprocessing_path: Resolved path to its fitted scalers and metadata.
        sequence_length: Number of hourly timesteps under sequence_policy;
            observed_steps skips missing water, hourly_grid retains hourly gaps.
        rain_station_ids: Rain sources in the order expected by the saved model.
        forecast_horizons_minutes: Output horizons, in saved target-column order.
        feature_schema_version: Identifier for the input feature contract.
        rain_windows_hours: Backward rainfall accumulation windows in hours.
        water_lags_hours: Water-level lags in physical hours, not retained-row offsets.
        flood_season_months: Calendar months whose season flag is 1.
        calendar_timezone: IANA timezone for assigning calendar-based season flags.
        model_parameters: Architecture parameters needed to rebuild the checkpoint.

    Raises:
        ValueError: If identifiers, sequence length, sources, windows or months
            fail the configuration checks.
        zoneinfo.ZoneInfoNotFoundError: If calendar_timezone cannot be resolved.

    Artifact compatibility is checked when loading preprocessing/runtime, not
    by constructing this configuration. Each station needs its own model entry.
    """
    station_id: str
    model_type: str
    model_version: str
    model_path: Path
    preprocessing_path: Path
    sequence_length: int
    rain_station_ids: tuple[str, ...]
    forecast_horizons_minutes: tuple[int, ...]
    feature_schema_version: str
    rain_windows_hours: tuple[int, ...] = (3, 6, 12, 24)
    water_lags_hours: tuple[int, ...] = (1, 3)
    flood_season_months: tuple[int, ...] = (9, 10, 11, 12)
    calendar_timezone: str = "Asia/Ho_Chi_Minh"
    model_parameters: dict[str, int | float] = field(default_factory=dict)
    model_id: str = "legacy"
    name: str = "Production cũ"
    is_default: bool = False
    sequence_policy: str = "observed_steps"
    weather_location_id: str | None = None
    weather_forecast_model: str = "ecmwf_ifs"
    weather_release_lag_hours: int = 6
    weather_max_age_hours: int = 24
    input_feature_columns: tuple[str, ...] | None = None

    def __post_init__(self):
        if (
            not self.station_id
            or not self.model_type
            or not self.model_version
            or not self.feature_schema_version
        ):
            raise ValueError("Station and model/schema versions are required.")
        if self.sequence_length < 1:
            raise ValueError("Sequence length must be positive.")
        if len(set(self.rain_station_ids)) != len(self.rain_station_ids):
            raise ValueError("Rain station IDs must be unique.")
        if self.station_id in self.rain_station_ids:
            raise ValueError("Water station cannot also be a rain station.")
        for windows in (
            self.rain_windows_hours,
            self.water_lags_hours,
            self.forecast_horizons_minutes,
        ):
            if len(set(windows)) != len(windows) or any(value <= 0 for value in windows):
                raise ValueError("Windows, lags and horizons must be positive and unique.")
        if any(month not in range(1, 13) for month in self.flood_season_months):
            raise ValueError("Season months must be between 1 and 12.")
        ZoneInfo(self.calendar_timezone)
        if not self.model_id or not self.name:
            raise ValueError("Model ID and display name are required.")
        if self.sequence_policy not in {"observed_steps", "hourly_grid"}:
            raise ValueError("Unsupported input sequence policy.")
        if self.weather_release_lag_hours < 0 or self.weather_max_age_hours < self.weather_release_lag_hours:
            raise ValueError("Invalid weather forecast age limits.")
        if self.weather_location_id is not None and not self.weather_location_id:
            raise ValueError("Weather location must be a nonempty configured ID.")
        # Validate an explicit order while allowing the training pool's appended
        # rain blocks; the feature set itself must still match this contract.
        self.feature_columns

    @property
    def feature_columns(self) -> tuple[str, ...]:
        """Build the ordered input schema required by this model.

        Returns:
            Feature names ordered as water depth, water lags, water missing flag,
            each configured rain source's depth/windows/missing flag, then season.
            This order must match the saved scaler; it is not a sorted column list.
        """
        columns = ["water_level_depth"]
        for hours in self.water_lags_hours:
            columns.append(f"water_level_lag_{hours * 60}m")
        columns.append("water_level_missing")
        for station in self.rain_station_ids:
            columns.append(f"rain_{station}_depth")
            for hours in self.rain_windows_hours:
                columns.append(f"rain_{station}_sum_{hours}h")
            columns.append(f"rain_{station}_missing")
        columns.append("flood_season")
        if self.weather_location_id is not None:
            windows = [(0, 3), (3, 6), (6, 12), (12, 24)]
            weather = [f"weather_forecast_rain_{start}_{end}h" for start, end in windows]
            columns.extend(weather)
            columns.extend(f"{name}_missing" for name in weather)
            columns.append("weather_forecast_age_hours")
        if self.input_feature_columns is not None:
            if len(self.input_feature_columns) != len(columns) or set(self.input_feature_columns) != set(columns):
                raise ValueError("Explicit feature order does not match configured inputs.")
            return tuple(self.input_feature_columns)
        return tuple(columns)

    @property
    def warmup_hours(self) -> int:
        return max((1, *self.rain_windows_hours, *self.water_lags_hours))


class ModelNotConfiguredError(KeyError):
    pass


class StationModelRegistry:
    """Map each water station to its own trained model configuration.

    Args:
        configurations: Station configurations to register; station IDs must be unique.

    Raises:
        ValueError: If two configurations use the same station ID.

    Unknown stations never fall back to another station's model. Constructing
    this registry does not connect to PostgreSQL or load weights/scalers.
    """
    def __init__(self, configurations: list[StationModelConfig]):
        self._configurations = {}
        self._defaults = {}
        for configuration in configurations:
            station = self._configurations.setdefault(configuration.station_id, {})
            if configuration.model_id in station:
                raise ValueError(f"Duplicate model configuration for {configuration.station_id}/{configuration.model_id}")
            if any(other.model_version == configuration.model_version for other in station.values()):
                raise ValueError("Model versions must be unique within a station.")
            station[configuration.model_id] = configuration
            if configuration.is_default:
                if configuration.station_id in self._defaults:
                    raise ValueError("Only one default model is allowed per station.")
                self._defaults[configuration.station_id] = configuration.model_id
        for station_id, models in self._configurations.items():
            self._defaults.setdefault(station_id, next(iter(models)))

    def get(self, station_id: str, model_id: str | None = None) -> StationModelConfig:
        """Find the model assigned to an exact station ID.

        Args:
            station_id: Water-station ID, including any leading zeros.
            model_id: Explicit choice, or None for this station's default.

        Returns:
            The registered configuration for that station, without loading artifacts.

        Raises:
            ModelNotConfiguredError: If no model is registered for the requested ID.
        """
        try:
            selected = self._defaults[station_id] if model_id is None else model_id
            return self._configurations[station_id][selected]
        except KeyError:
            raise ModelNotConfiguredError(
                f"No trained model configured for water station {station_id} and model {model_id}"
            ) from None

    def models_for(self, station_id: str) -> tuple[StationModelConfig, ...]:
        """List configured choices without loading checkpoints or scalers."""
        self.get(station_id)
        return tuple(self._configurations[station_id].values())

    @classmethod
    def from_json(cls, path: Path, artifact_root: Path) -> "StationModelRegistry":
        """Load a station registry from a JSON configuration list.

        Args:
            path: JSON list with one configuration per station/model choice.
            artifact_root: Deployment root used to resolve relative checkpoint and
                preprocessing paths. Absolute artifact paths remain absolute.

        Returns:
            Registry of validated station configurations; artifacts are not loaded.

        Raises:
            OSError: If the configuration file cannot be read.
            json.JSONDecodeError: If the file is not valid JSON.
            KeyError: If required artifact path fields are absent.
            TypeError: If entries contain missing or unexpected constructor fields.
            ValueError: If configuration, model IDs, versions or defaults conflict.
        """
        entries = json.loads(path.read_text())
        configurations = []
        tuple_fields = (
            "rain_station_ids",
            "forecast_horizons_minutes",
            "rain_windows_hours",
            "water_lags_hours",
            "flood_season_months",
            "input_feature_columns",
        )
        for entry in entries:
            fields = dict(entry)
            for name in tuple_fields:
                if name in fields:
                    fields[name] = tuple(fields[name])
            for name in ("model_path", "preprocessing_path"):
                fields[name] = artifact_root / fields[name]
            configurations.append(StationModelConfig(**fields))
        return cls(configurations)
