"""Hourly features matching the existing top-four-rain training dataset."""

from datetime import datetime

import numpy as np
import pandas as pd

from src.backend.prediction.config import StationModelConfig


def utc_timestamp(value: datetime) -> pd.Timestamp:
    """Convert an aware timestamp to UTC without rounding it.

    Args:
        value: Datetime-like value representing an absolute instant with a timezone.

    Returns:
        Pandas Timestamp for the same instant in UTC.

    Raises:
        ValueError: If the value is missing, invalid or has no timezone.
    """
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError("A valid timestamp with a timezone is required.")
    return timestamp.tz_convert("UTC")


def build_hourly_features(
    observations: pd.DataFrame, reference_time: datetime, config: StationModelConfig
) -> pd.DataFrame:
    """Build original-unit hourly features using only observations up to the cutoff.

    Args:
        observations: Raw table with station_id, observed_at and depth. Each
            station/timestamp pair must be unique; missing depth may be NaN.
        reference_time: Timezone-aware cutoff, rounded down to the hour.
        config: Water/rain sources, feature order, windows, lags and season calendar
            belonging to the model that will consume these features.

    Returns:
        DataFrame indexed by UTC hour in model feature order. Label t represents
        (t-1h, t]: water is averaged and rain summed per station. Rain windows need
        every hour in the window; water lags are shifted on the hourly grid.
        Missing values stay missing, flags are 0/1, and no scaling/interpolation
        is performed. Water-missing hours remain for the caller to filter.

    Raises:
        ValueError: If required columns are absent, station/timestamps repeat,
            numeric depths are invalid/infinite, or the cutoff is invalid.

    The input table is copied, not modified; future observations are excluded.
    """
    reference = utc_timestamp(reference_time).floor("h")
    required_columns = {"station_id", "observed_at", "depth"}
    if not required_columns.issubset(observations.columns):
        raise ValueError("Observations require station_id, observed_at and depth.")
    history = observations.loc[:, ["station_id", "observed_at", "depth"]].copy()
    history["observed_at"] = pd.to_datetime(history["observed_at"], utc=True)
    history = history.loc[history["observed_at"] <= reference].sort_values("observed_at")
    if history.duplicated(["station_id", "observed_at"]).any():
        raise ValueError("Duplicate station/timestamp observations.")
    history["depth"] = pd.to_numeric(history["depth"], errors="raise")
    if np.isinf(history["depth"].to_numpy(dtype=float)).any():
        raise ValueError("Observation depth cannot be infinite.")
    hourly_by_station = {}
    for station in (config.station_id, *config.rain_station_ids):
        station_history = history.loc[history["station_id"] == station]
        series = station_history.set_index("observed_at")["depth"]
        grouped = series.resample("1h", closed="right", label="right")
        hourly = grouped.mean() if station == config.station_id else grouped.sum(min_count=1)
        hourly_by_station[station] = hourly.loc[hourly.index <= reference]

    water = hourly_by_station[config.station_id]
    features = water.rename("water_level_depth").to_frame()
    features.index.name = "timestamp_utc"
    for hours in config.water_lags_hours:
        features[f"water_level_lag_{hours * 60}m"] = water.shift(hours)
    features["water_level_missing"] = water.isna().astype("int8")
    for station in config.rain_station_ids:
        hourly_rain = hourly_by_station[station]
        features[f"rain_{station}_depth"] = hourly_rain.reindex(features.index)
        for hours in config.rain_windows_hours:
            features[f"rain_{station}_sum_{hours}h"] = (
                hourly_rain.rolling(hours, min_periods=hours).sum().reindex(features.index)
            )
        features[f"rain_{station}_missing"] = (
            features[f"rain_{station}_depth"].isna().astype("int8")
        )
    features["flood_season"] = (
        features.index.tz_convert(config.calendar_timezone)
        .month.isin(config.flood_season_months)
        .astype("int8")
    )
    return features.loc[:, list(config.feature_columns)]
