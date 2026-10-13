"""Each water station owns a separately trained model and preprocessing contract."""

from dataclasses import dataclass, field
import json
from pathlib import Path
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class StationModelConfig:
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
    def __init__(self, configurations: list[StationModelConfig]):
        self._configurations = {}
        for configuration in configurations:
            if configuration.station_id in self._configurations:
                raise ValueError(f"Duplicate model configuration for {configuration.station_id}")
            self._configurations[configuration.station_id] = configuration

    def get(self, station_id: str) -> StationModelConfig:
        try:
            return self._configurations[station_id]
        except KeyError:
            raise ModelNotConfiguredError(
                f"No trained model configured for water station {station_id}"
            ) from None

    @classmethod
    def from_json(cls, path: Path, artifact_root: Path) -> "StationModelRegistry":
        """Resolve artifact paths relative to an explicitly supplied deployment root."""
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
