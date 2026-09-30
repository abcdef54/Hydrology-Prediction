"""Measure per-series temporal coverage, sampling intervals, gaps, and availability."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_config import AnalysisPaths
from analysis_utils import infer_nominal_interval_minutes, parse_utc, read_csv, write_json


def load_timestamped_series(path: Path) -> pd.DataFrame:
    frame = read_csv(path)
    if path.name == "reservoir_observations.csv":
        frame["timestamp"] = pd.to_datetime(
            frame["ngaylaysolieu"].astype(str) + " " + frame["gio"].astype(str),
            format="%d/%m/%Y %H:%M",
            errors="coerce",
            utc=True,
        )
        frame["series_id"] = frame["tencongtrinh"].astype("string")
    elif path.name == "wind_observations.csv":
        frame["timestamp"] = parse_utc(frame["time_point"])
        frame["series_id"] = frame["sid"].astype("string")
    elif "station_id" in frame:
        frame["timestamp"] = parse_utc(frame["time_point"])
        frame["series_id"] = frame["station_id"].astype("string")
    else:
        frame["timestamp"] = parse_utc(frame["timestamp_utc"])
        frame["series_id"] = path.stem
    return frame


def summarize_series(source: str, series_id: str, timestamps: pd.Series) -> dict[str, object]:
    ordered = parse_utc(timestamps).dropna().sort_values()
    unique = ordered.drop_duplicates()
    if unique.empty:
        return {"source": source, "series_id": series_id, "actual_records": 0}
    interval_minutes = infer_nominal_interval_minutes(unique)
    intervals = unique.diff().dropna().dt.total_seconds().div(60)
    duration_minutes = (unique.iloc[-1] - unique.iloc[0]).total_seconds() / 60
    expected_records = (
        int(np.floor(duration_minutes / interval_minutes)) + 1
        if interval_minutes and interval_minutes > 0
        else None
    )
    interval_distribution = intervals.quantile([0, 0.25, 0.5, 0.75, 0.9, 0.99, 1]).to_dict()
    return {
        "source": source,
        "series_id": series_id,
        "start": unique.iloc[0],
        "end": unique.iloc[-1],
        "duration_days": float(duration_minutes / 1_440),
        "actual_records": int(len(ordered)),
        "unique_timestamps": int(len(unique)),
        "duplicate_timestamps": int(len(ordered) - len(unique)),
        "nominal_interval_minutes": interval_minutes,
        "expected_records": expected_records,
        "availability_ratio": float(len(unique) / expected_records) if expected_records else None,
        "longest_gap_minutes": float(intervals.max()) if not intervals.empty else None,
        "gaps_over_10m": int(intervals.gt(10).sum()),
        "gaps_over_30m": int(intervals.gt(30).sum()),
        "gaps_over_1h": int(intervals.gt(60).sum()),
        "gaps_over_6h": int(intervals.gt(360).sum()),
        "gaps_over_24h": int(intervals.gt(1_440).sum()),
        "interval_percentiles_minutes": {str(key): float(value) for key, value in interval_distribution.items()},
    }


def availability_records(
    source: str,
    series_id: str,
    timestamps: pd.Series,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    monthly_records: list[dict[str, object]] = []
    daily_records: list[dict[str, object]] = []
    timestamps = timestamps.dropna().drop_duplicates().sort_values()
    interval = infer_nominal_interval_minutes(timestamps)
    if not interval:
        return monthly_records, daily_records
    utc_without_timezone = timestamps.dt.tz_localize(None)
    monthly_actual = utc_without_timezone.dt.to_period("M").value_counts().sort_index()
    daily_actual = timestamps.dt.floor("D").value_counts().sort_index()
    expected_per_day = 1_440 / interval
    for day, actual in daily_actual.items():
        daily_records.append(
            {
                "series": f"{source}:{series_id}",
                "date_utc": day,
                "actual_records": int(actual),
                "expected_records": expected_per_day,
                "availability": min(float(actual / expected_per_day), 1.0),
            }
        )
    for month, actual in monthly_actual.items():
        month_start = month.start_time.tz_localize("UTC")
        month_end = (month + 1).start_time.tz_localize("UTC")
        expected = (month_end - month_start).total_seconds() / 60 / interval
        monthly_records.append(
            {
                "series": f"{source}:{series_id}",
                "month": str(month),
                "availability": min(float(actual / expected), 1.0),
            }
        )
    return monthly_records, daily_records


def plot_monthly_availability(
    paths: AnalysisPaths,
    monthly_records: list[dict[str, object]],
    daily_records: list[dict[str, object]],
) -> None:
    availability = pd.DataFrame(monthly_records)
    availability.to_csv(paths.reports / "monthly_availability.csv", index=False)
    pd.DataFrame(daily_records).to_csv(paths.reports / "daily_availability.csv", index=False)
    if availability.empty:
        return
    selected = availability.groupby("series")["availability"].mean().nsmallest(25).index
    matrix = availability.loc[availability["series"].isin(selected)].pivot(
        index="series", columns="month", values="availability"
    )
    figure, axis = plt.subplots(figsize=(13, max(4, 0.35 * len(matrix))))
    image = axis.imshow(matrix.fillna(0), aspect="auto", vmin=0, vmax=1, cmap="RdYlGn")
    axis.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(matrix.index)), matrix.index)
    axis.set_title("Monthly observation availability (lowest 25 series)")
    figure.colorbar(image, ax=axis, label="Availability")
    figure.tight_layout()
    figure.savefig(paths.figures / "monthly_availability_heatmap.png", dpi=150)
    plt.close(figure)


def run(paths: AnalysisPaths) -> None:
    paths.create_output_directories()
    summaries = []
    monthly_records: list[dict[str, object]] = []
    daily_records: list[dict[str, object]] = []
    filenames = (
        "rain_observations.csv",
        "water_level_observations.csv",
        "wind_observations.csv",
        "reservoir_observations.csv",
    )
    for filename in filenames:
        observations = load_timestamped_series(paths.data / filename)
        observations["source"] = filename
        for series_id, group in observations.groupby("series_id", dropna=False):
            summaries.append(summarize_series(filename, str(series_id), group["timestamp"]))
            series_monthly, series_daily = availability_records(
                filename, str(series_id), group["timestamp"]
            )
            monthly_records.extend(series_monthly)
            daily_records.extend(series_daily)
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(paths.reports / "temporal_quality.csv", index=False)
    write_json(summaries, paths.reports / "temporal_quality.json")
    plot_monthly_availability(paths, monthly_records, daily_records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=AnalysisPaths().data)
    parser.add_argument("--reports-dir", type=Path, default=AnalysisPaths().reports)
    parser.add_argument("--figures-dir", type=Path, default=AnalysisPaths().figures)
    args = parser.parse_args()
    run(AnalysisPaths(args.data_dir, args.reports_dir, args.figures_dir))


if __name__ == "__main__":
    main()
