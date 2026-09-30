"""Analyze persistence, lagged signals, high-water events, wind, and reservoirs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_config import AnalysisPaths, HIGH_WATER_PERCENTILES, RAIN_LAGS_HOURS
from analysis_utils import (
    contiguous_events,
    dataframe_records,
    find_training_table,
    parse_utc,
    read_csv,
    write_json,
)


def load_training_table(paths: AnalysisPaths) -> pd.DataFrame:
    frame = read_csv(find_training_table(paths.data, paths.reports))
    frame["timestamp_utc"] = parse_utc(frame["timestamp_utc"])
    return frame.sort_values("timestamp_utc").set_index("timestamp_utc")


def safe_correlation(left: pd.Series, right: pd.Series) -> tuple[float | None, int]:
    aligned = pd.concat([left, right], axis=1).dropna()
    if len(aligned) < 3 or aligned.iloc[:, 0].nunique() < 2 or aligned.iloc[:, 1].nunique() < 2:
        return None, int(len(aligned))
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1])), int(len(aligned))


def persistence_analysis(training: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon_minutes in (60, 180, 360):
        target = training[f"target_water_level_plus_{horizon_minutes}m"]
        correlation, count = safe_correlation(training["water_level_depth"], target)
        rows.append({"horizon_minutes": horizon_minutes, "correlation": correlation, "paired_observations": count})
    return pd.DataFrame(rows)


def lagged_signal_analysis(training: pd.DataFrame) -> pd.DataFrame:
    high_water_threshold = training["water_level_depth"].quantile(0.90)
    water_change = training["water_level_depth"].diff(6)
    rising_threshold = water_change.quantile(0.90)
    masks = {
        "all": pd.Series(True, index=training.index),
        "high_water_p90": training["water_level_depth"].gt(high_water_threshold),
        "rapid_rise_p90": water_change.gt(rising_threshold),
    }
    rows = []
    for lag_hours in RAIN_LAGS_HOURS:
        lag_steps = lag_hours * 6
        lagged_rain = training["rain_depth"].shift(lag_steps)
        for subset, mask in masks.items():
            level_correlation, level_count = safe_correlation(lagged_rain[mask], training.loc[mask, "water_level_depth"])
            rise_correlation, rise_count = safe_correlation(lagged_rain[mask], water_change[mask])
            rows.append(
                {
                    "input": "rain_depth",
                    "lag_hours": lag_hours,
                    "subset": subset,
                    "water_level_correlation": level_correlation,
                    "water_level_pairs": level_count,
                    "six_hour_rise_correlation": rise_correlation,
                    "rise_pairs": rise_count,
                }
            )
    for wind_column in ("wind_speed", "wind_gust"):
        for lag_hours in RAIN_LAGS_HOURS:
            lagged_wind = training[wind_column].shift(lag_hours * 6)
            for subset, mask in masks.items():
                correlation, count = safe_correlation(lagged_wind[mask], training.loc[mask, "water_level_depth"])
                rows.append(
                    {
                        "input": wind_column,
                        "lag_hours": lag_hours,
                        "subset": subset,
                        "water_level_correlation": correlation,
                        "water_level_pairs": count,
                    }
                )
    return pd.DataFrame(rows)


def rainfall_before(training: pd.DataFrame, event_start: pd.Timestamp, hours: int) -> float | None:
    values = training.loc[(training.index >= event_start - pd.Timedelta(hours=hours)) & (training.index < event_start), "rain_depth"]
    return float(values.sum(min_count=1)) if not values.empty and values.notna().any() else None


def event_analysis(training: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_rows = []
    summary_rows = []
    for percentile in HIGH_WATER_PERCENTILES:
        threshold = float(training["water_level_depth"].quantile(percentile))
        events = contiguous_events(
            training.index.to_series(), training["water_level_depth"], threshold, pd.Timedelta(hours=6)
        )
        summary_rows.append({"percentile": percentile, "threshold_raw_units": threshold, "independent_events": len(events)})
        for event_number, event in enumerate(events, start=1):
            row = {"percentile": percentile, "threshold_raw_units": threshold, "event_number": event_number, **event}
            for hours in (1, 3, 6, 12, 24):
                row[f"rain_before_{hours}h"] = rainfall_before(training, event["start"], hours)
            event_rows.append(row)
    return pd.DataFrame(event_rows), pd.DataFrame(summary_rows)


def reservoir_analysis(paths: AnalysisPaths, training: pd.DataFrame) -> pd.DataFrame:
    observations = read_csv(paths.data / "reservoir_observations.csv")
    observations["timestamp"] = pd.to_datetime(
        observations["ngaylaysolieu"].astype(str) + " " + observations["gio"].astype(str),
        format="%d/%m/%Y %H:%M", errors="coerce", utc=True,
    )
    numeric_columns = ("htl", "hhl", "qden", "qdi", "socuamo", "mucnuocsong", "luuluongdieutiet")
    rows = []
    target_hourly = training["water_level_depth"].resample("1h").mean()
    for reservoir_id, group in observations.groupby("tencongtrinh"):
        group = group.set_index("timestamp").sort_index()
        nominal_interval = group.index.to_series().sort_values().diff().dropna().dt.total_seconds().div(60).median()
        for column in numeric_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            if values.notna().sum() == 0:
                continue
            hourly = values.resample("1h").mean()
            correlation, pairs = safe_correlation(hourly, target_hourly)
            value_change_events = int(values.dropna().diff().abs().gt(0).sum())
            rows.append(
                {
                    "reservoir": reservoir_id,
                    "variable": column,
                    "records": int(values.notna().sum()),
                    "missing_ratio": float(values.isna().mean()),
                    "median_interval_minutes": float(nominal_interval) if pd.notna(nominal_interval) else None,
                    "same_time_correlation_with_da_vien": correlation,
                    "paired_hours": pairs,
                    "value_change_events": value_change_events,
                    "hydrologic_relationship": "UNKNOWN; correlation is exploratory and is not merge authorization.",
                }
            )
    return pd.DataFrame(rows)


def station_metadata_names(
    paths: AnalysisPaths,
    filename: str,
    identifier_column: str,
) -> dict[str, str]:
    stations = read_csv(paths.data / filename)
    return {
        str(identifier): str(name)
        for identifier, name in zip(stations[identifier_column], stations["name"])
    }


def all_station_lagged_signals(
    paths: AnalysisPaths,
    training: pd.DataFrame,
) -> pd.DataFrame:
    """Compare every rain and wind station with the Dã Viên target."""
    target_hourly = training["water_level_depth"].resample("1h").mean()
    target_change_six_hours = target_hourly.diff(6)
    high_water_threshold = target_hourly.quantile(0.90)
    subset_masks = {
        "all": pd.Series(True, index=target_hourly.index),
        "high_water_p90": target_hourly.gt(high_water_threshold),
    }
    source_definitions = (
        {
            "station_type": "rain",
            "observations": "rain_observations.csv",
            "metadata": "rain_stations.csv",
            "observation_id": "station_id",
            "metadata_id": "code",
            "value": "depth",
            "aggregation": "sum",
        },
        {
            "station_type": "wind",
            "observations": "wind_observations.csv",
            "metadata": "wind_stations.csv",
            "observation_id": "sid",
            "metadata_id": "id",
            "value": "ws",
            "aggregation": "mean",
        },
    )
    rows: list[dict[str, object]] = []
    for definition in source_definitions:
        observations = read_csv(paths.data / definition["observations"])
        observations["timestamp"] = parse_utc(observations["time_point"])
        station_names = station_metadata_names(
            paths, definition["metadata"], definition["metadata_id"]
        )
        for station_id, station_observations in observations.groupby(
            definition["observation_id"], dropna=False
        ):
            station_series = pd.Series(
                pd.to_numeric(station_observations[definition["value"]], errors="coerce").to_numpy(),
                index=station_observations["timestamp"],
            ).sort_index()
            station_series = station_series.loc[~station_series.index.isna()]
            station_series = station_series.groupby(level=0).mean()
            if definition["aggregation"] == "sum":
                hourly = station_series.resample("1h").sum(min_count=1)
            else:
                hourly = station_series.resample("1h").mean()
            for lag_hours in RAIN_LAGS_HOURS:
                lagged_input = hourly.shift(lag_hours)
                for subset_name, subset_mask in subset_masks.items():
                    level_correlation, level_pairs = safe_correlation(
                        lagged_input, target_hourly.where(subset_mask)
                    )
                    change_correlation, change_pairs = safe_correlation(
                        lagged_input, target_change_six_hours.where(subset_mask)
                    )
                    rows.append(
                        {
                            "station_type": definition["station_type"],
                            "station_id": station_id,
                            "station_name": station_names.get(
                                str(station_id), "Không có trong station metadata"
                            ),
                            "lag_hours": lag_hours,
                            "subset": subset_name,
                            "water_level_correlation": level_correlation,
                            "water_level_pairs": level_pairs,
                            "six_hour_rise_correlation": change_correlation,
                            "rise_pairs": change_pairs,
                        }
                    )
    return pd.DataFrame(rows)


def plot_rain_station_signal_heatmap(
    paths: AnalysisPaths,
    station_signals: pd.DataFrame,
) -> None:
    rain_signals = station_signals.loc[
        (station_signals["station_type"] == "rain")
        & (station_signals["subset"] == "all")
    ]
    if rain_signals.empty:
        return
    matrix = rain_signals.pivot(
        index="station_name",
        columns="lag_hours",
        values="six_hour_rise_correlation",
    )
    station_order = matrix.abs().max(axis=1).sort_values(ascending=False).index
    matrix = matrix.loc[station_order]
    figure, axis = plt.subplots(figsize=(11, max(7, 0.28 * len(matrix))))
    image = axis.imshow(matrix.fillna(0), aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_xticks(range(len(matrix.columns)), matrix.columns)
    axis.set_yticks(range(len(matrix.index)), matrix.index, fontsize=7)
    axis.set_xlabel("Rain lag (hours)")
    axis.set_ylabel("Rain station")
    axis.set_title("Rain station correlation with Dã Viên 6-hour water-level change")
    figure.colorbar(image, ax=axis, label="Pearson correlation")
    figure.tight_layout()
    figure.savefig(paths.figures / "rain_station_signal_heatmap.png", dpi=170)
    plt.close(figure)


def plot_core_timeseries(paths: AnalysisPaths, training: pd.DataFrame) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    axes[0].plot(training.index, training["water_level_depth"], linewidth=0.65, color="#c82423")
    axes[0].set_ylabel("Water level\n(raw unit)")
    axes[1].plot(training.index, training["rain_depth"], linewidth=0.55, color="#2878b5")
    axes[1].set_ylabel("Rain\n(raw unit)")
    axes[2].plot(training.index, training["wind_speed"], linewidth=0.5, color="#56a64b")
    axes[2].set_ylabel("Wind speed\n(raw unit)")
    axes[2].set_xlabel("UTC time")
    figure.suptitle("Observed series; units are not inferred")
    figure.tight_layout()
    figure.savefig(paths.figures / "core_timeseries.png", dpi=150)
    plt.close(figure)


def run(paths: AnalysisPaths) -> None:
    paths.create_output_directories()
    training = load_training_table(paths)
    persistence = persistence_analysis(training)
    lagged_signals = lagged_signal_analysis(training)
    events, event_summary = event_analysis(training)
    reservoirs = reservoir_analysis(paths, training)
    station_signals = all_station_lagged_signals(paths, training)
    persistence.to_csv(paths.reports / "persistence_correlations.csv", index=False)
    lagged_signals.to_csv(paths.reports / "lagged_signal_correlations.csv", index=False)
    events.to_csv(paths.reports / "high_water_events.csv", index=False)
    event_summary.to_csv(paths.reports / "high_water_event_summary.csv", index=False)
    reservoirs.to_csv(paths.reports / "reservoir_signal_summary.csv", index=False)
    station_signals.to_csv(paths.reports / "all_station_lagged_signals.csv", index=False)
    write_json(
        {
            "persistence": dataframe_records(persistence),
            "event_summary": dataframe_records(event_summary),
            "note": "Threshold exceedances are high-water events, not confirmed floods.",
        },
        paths.reports / "hydrologic_signal_summary.json",
    )
    plot_core_timeseries(paths, training)
    plot_rain_station_signal_heatmap(paths, station_signals)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=AnalysisPaths().data)
    parser.add_argument("--reports-dir", type=Path, default=AnalysisPaths().reports)
    parser.add_argument("--figures-dir", type=Path, default=AnalysisPaths().figures)
    args = parser.parse_args()
    run(AnalysisPaths(args.data_dir, args.reports_dir, args.figures_dir))


if __name__ == "__main__":
    main()
