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
        sequence_length: Required number of valid hourly timesteps, not raw rows;
            gaps between selected timesteps are allowed.
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
        for configuration in configurations:
            if configuration.station_id in self._configurations:
                raise ValueError(f"Duplicate model configuration for {configuration.station_id}")
            self._configurations[configuration.station_id] = configuration

    def get(self, station_id: str) -> StationModelConfig:
        """Find the model assigned to an exact station ID.

        Args:
            station_id: Water-station ID, including any leading zeros.

        Returns:
            The registered configuration for that station, without loading artifacts.

        Raises:
            ModelNotConfiguredError: If no model is registered for the requested ID.
        """
        try:
            return self._configurations[station_id]
        except KeyError:
            raise ModelNotConfiguredError(
                f"No trained model configured for water station {station_id}"
            ) from None

    @classmethod
    def from_json(cls, path: Path, artifact_root: Path) -> "StationModelRegistry":
        """Load a station registry from a JSON configuration list.

        Args:
            path: JSON file containing one configuration object per station.
            artifact_root: Deployment root used to resolve relative checkpoint and
                preprocessing paths. Absolute artifact paths remain absolute.

        Returns:
            Registry of validated station configurations; artifacts are not loaded.

        Raises:
            OSError: If the configuration file cannot be read.
            json.JSONDecodeError: If the file is not valid JSON.
            KeyError: If required artifact path fields are absent.
            TypeError: If entries contain missing or unexpected constructor fields.
            ValueError: If a station configuration is invalid or station IDs repeat.
        """
        entries = json.loads(path.read_text())
        configurations = []
        tuple_fields = (
            "rain_station_ids",
            "forecast_horizons_minutes",
            "rain_windows_hours",
            "water_lags_hours",
            "flood_season_months",
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
