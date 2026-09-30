"""Build a leakage-safe 10-minute training table from immutable raw observations."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analysis_config import AnalysisPaths, FORECAST_HORIZONS_MINUTES
from analysis_utils import parse_utc, read_csv


def indexed_numeric_series(
    path: Path,
    timestamp_column: str,
    value_column: str,
    identifier_column: str,
    identifier: object,
) -> pd.Series:
    frame = read_csv(path)
    frame = frame.loc[frame[identifier_column].astype(str) == str(identifier)]
    timestamps = parse_utc(frame[timestamp_column])
    values = pd.to_numeric(frame[value_column], errors="coerce")
    series = pd.Series(values.to_numpy(), index=timestamps, name=value_column).dropna(axis="index", how="all")
    return series.loc[~series.index.isna()].sort_index().groupby(level=0).mean()


def station_identifier(
    data_directory: Path,
    metadata_filename: str,
    station_name: str,
    identifier_column: str,
    name_column: str = "name",
) -> object:
    stations = read_csv(data_directory / metadata_filename)
    matches = stations.loc[stations[name_column].eq(station_name), identifier_column]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {station_name!r} in {metadata_filename}, found {len(matches)}"
        )
    return matches.iloc[0]


def build_training_table(data_directory: Path) -> pd.DataFrame:
    rain_station_code = station_identifier(
        data_directory, "rain_stations.csv", "Hồ Hòa Mỹ (Phong Điền)", "code"
    )
    water_station_code = station_identifier(
        data_directory, "water_level_stations.csv", "Trạm Dã Viên (Sông Hương)", "code"
    )
    wind_station_id = station_identifier(
        data_directory, "wind_stations.csv", "Cảng Thuận An", "id"
    )
    rain = indexed_numeric_series(
        data_directory / "rain_observations.csv",
        "time_point",
        "depth",
        "station_id",
        rain_station_code,
    ).resample("10min").sum(min_count=1)
    water_level = indexed_numeric_series(
        data_directory / "water_level_observations.csv",
        "time_point",
        "depth",
        "station_id",
        water_station_code,
    ).resample("10min").mean()
    wind_frame = read_csv(data_directory / "wind_observations.csv")
    wind_frame = wind_frame.loc[wind_frame["sid"].astype(str) == str(wind_station_id)]
    wind_frame["timestamp"] = parse_utc(wind_frame["time_point"])
    wind = (
        wind_frame.set_index("timestamp")[["ws", "wsg", "wd", "wdg"]]
        .apply(pd.to_numeric, errors="coerce")
        .resample("10min")
        .mean()
    )
    time_index = pd.date_range(water_level.index.min(), water_level.index.max(), freq="10min", tz="UTC")
    training = pd.DataFrame(index=time_index)
    training.index.name = "timestamp_utc"
    training["rain_depth"] = rain.reindex(time_index)
    training["water_level_depth"] = water_level.reindex(time_index)
    training[["wind_speed", "wind_gust", "wind_direction", "wind_gust_direction"]] = wind.reindex(time_index).to_numpy()
    for minutes in (10, 30, 60, 180):
        steps = minutes // 10
        training[f"rain_lag_{minutes}m"] = training["rain_depth"].shift(steps)
        training[f"water_level_lag_{minutes}m"] = training["water_level_depth"].shift(steps)
    for hours in (1, 3, 6, 24):
        window_steps = hours * 6
        training[f"rain_sum_{hours}h"] = training["rain_depth"].shift(1).rolling(window_steps, min_periods=window_steps).sum()
    for minutes in FORECAST_HORIZONS_MINUTES:
        training[f"target_water_level_plus_{minutes}m"] = training["water_level_depth"].shift(-(minutes // 10))
    training["rain_missing"] = training["rain_depth"].isna().astype(int)
    training["water_level_missing"] = training["water_level_depth"].isna().astype(int)
    training["wind_missing"] = training["wind_speed"].isna().astype(int)
    return training.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=AnalysisPaths().data)
    parser.add_argument("--output", type=Path, default=AnalysisPaths().reports / "hydrology_training_10min_rebuilt.csv")
    parser.add_argument("--force", action="store_true", help="Overwrite the requested output, never a raw CSV.")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Output exists: {args.output}. Pass --force to replace this derived file.")
    if args.output.resolve().parent == args.data_dir.resolve():
        raise ValueError("Refusing to write derived data into the raw data directory.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    build_training_table(args.data_dir).to_csv(args.output, index=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
