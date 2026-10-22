"""Forecast-rain inputs from one available vintage per historical input hour."""

import numpy as np
import pandas as pd
import sqlalchemy as sa

from src.backend.db.weather import hourly, runs


WINDOWS = ((0, 3), (3, 6), (6, 12), (12, 24))


def forecast_features(rows: pd.DataFrame, anchors: pd.DatetimeIndex, config) -> pd.DataFrame:
    """Match the mask dataset's windows, six-hour release lag and older-run fallback.

    Known publication timestamps must also be at or before the input hour.
    Archived runs without publication timestamps use the configured assumed lag,
    matching training. No future observations or mixed forecast vintages are used.
    """
    columns = [f"weather_forecast_rain_{a}_{b}h" for a, b in WINDOWS]
    result = pd.DataFrame(np.nan, index=anchors, columns=columns)
    result["weather_forecast_age_hours"] = np.nan
    if not rows.empty:
        frame = rows.copy()
        for column in ("initialized_at", "retrieved_at", "available_at", "valid_at"):
            frame[column] = pd.to_datetime(frame[column], utc=True)
        if frame.duplicated(["run_id", "valid_at"]).any():
            raise ValueError("Duplicate forecast hours in a weather vintage.")
        values = pd.to_numeric(frame.precipitation_mm, errors="raise")
        if np.isinf(values).any() or values.lt(0).any():
            raise ValueError("Weather rainfall must be nonnegative and finite or missing.")
        frame["precipitation_mm"] = values
        metadata = frame[["run_id", "initialized_at", "retrieved_at", "available_at"]].drop_duplicates()
        metadata = metadata.sort_values(["initialized_at", "retrieved_at"], ascending=False)
        series = {key: group.set_index("valid_at").precipitation_mm for key, group in frame.groupby("run_id")}
        for anchor in anchors:
            eligible = metadata[
                (metadata.initialized_at <= anchor - pd.Timedelta(hours=config.weather_release_lag_hours))
                & (metadata.initialized_at >= anchor - pd.Timedelta(hours=config.weather_max_age_hours))
                & (metadata.available_at.isna() | metadata.available_at.le(anchor))
            ].drop_duplicates("initialized_at", keep="first")
            primary = None
            selected = None
            future_times = pd.date_range(anchor + pd.Timedelta(hours=1), periods=24, freq="h")
            for run in eligible.itertuples(index=False):
                age = (anchor - run.initialized_at).total_seconds() / 3600
                future = series[run.run_id].reindex(future_times)
                windows = [future.iloc[a:b].sum(min_count=b - a) for a, b in WINDOWS]
                if primary is None and age <= 12:
                    primary = (windows, age)
                if future.notna().all():
                    selected = (windows, age)
                    break
            selected = selected or primary
            if selected is not None:
                result.loc[anchor, columns] = selected[0]
                result.loc[anchor, "weather_forecast_age_hours"] = selected[1]
    for column in columns:
        result[f"{column}_missing"] = result[column].isna().astype("int8")
    return result.loc[:, columns + [f"{c}_missing" for c in columns] + ["weather_forecast_age_hours"]]


def add_weather_features(connection, features, config):
    """Read the fixed training grid inside the observation snapshot; no HTTP calls."""
    first, last = features.index[0], features.index[-1]
    query = (
        sa.select(
            runs.c.id.label("run_id"), runs.c.initialized_at, runs.c.retrieved_at,
            runs.c.available_at, hourly.c.valid_at, hourly.c.precipitation_mm,
        )
        .select_from(runs.join(hourly, hourly.c.run_id == runs.c.id))
        .where(
            runs.c.location_id == config.weather_location_id,
            runs.c.model == config.weather_forecast_model,
            runs.c.initialized_at >= (first - pd.Timedelta(hours=config.weather_max_age_hours)).to_pydatetime(),
            runs.c.initialized_at <= (last - pd.Timedelta(hours=config.weather_release_lag_hours)).to_pydatetime(),
            hourly.c.valid_at > first.to_pydatetime(),
            hourly.c.valid_at <= (last + pd.Timedelta(hours=24)).to_pydatetime(),
        )
    )
    weather = forecast_features(pd.read_sql(query, connection), features.index, config)
    result = features.copy()
    result.loc[:, weather.columns] = weather
    return result.loc[:, list(config.feature_columns)]
