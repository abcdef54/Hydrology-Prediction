"""Shared configuration for the Hue hydrology data audit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = PROJECT_ROOT / "data"
REPORTS_DIRECTORY = PROJECT_ROOT / "reports"
FIGURES_DIRECTORY = PROJECT_ROOT / "figures"

TARGET_STATION_NAMES = {
    "rain": "Hồ Hòa Mỹ (Phong Điền)",
    "water_level": "Trạm Dã Viên (Sông Hương)",
    "wind": "Cảng Thuận An",
}

OBSERVATION_FILES = {
    "rain": "rain_observations.csv",
    "water_level": "water_level_observations.csv",
    "wind": "wind_observations.csv",
    "reservoir": "reservoir_observations.csv",
}

METADATA_FILES = {
    "rain": "rain_stations.csv",
    "water_level": "water_level_stations.csv",
    "wind": "wind_stations.csv",
    "reservoir": "reservoirs.csv",
}

FORECAST_HORIZONS_MINUTES = (60, 180, 360)
RAIN_LAGS_HOURS = (0, 1, 2, 3, 6, 12, 24, 48)
PERCENTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.995, 0.999)
HIGH_WATER_PERCENTILES = (0.90, 0.95, 0.99)
RANDOM_SEED = 42


@dataclass(frozen=True)
class AnalysisPaths:
    """Filesystem locations used by all analysis scripts."""

    data: Path = DATA_DIRECTORY
    reports: Path = REPORTS_DIRECTORY
    figures: Path = FIGURES_DIRECTORY

    def create_output_directories(self) -> None:
        self.reports.mkdir(parents=True, exist_ok=True)
        self.figures.mkdir(parents=True, exist_ok=True)

