"""Inventory schemas, metadata, missingness, duplicates, and distributions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_config import AnalysisPaths, PERCENTILES
from analysis_utils import (
    find_training_table,
    infer_nominal_interval_minutes,
    logical_csv_name,
    parse_utc,
    read_csv,
    write_json,
)


TIMESTAMP_CANDIDATES = ("timestamp_utc", "time_point", "ngaylaysolieu", "date_created")
IDENTIFIER_CANDIDATES = ("station_id", "sid", "code", "id", "tencongtrinh")
OBSERVATION_COLUMNS = {
    "rain_observations.csv": ("depth",),
    "water_level_observations.csv": ("depth",),
    "wind_observations.csv": ("ws", "wsg", "wd", "wdg"),
    "reservoir_observations.csv": ("htl", "hhl", "qden", "qdi", "socuamo", "mucnuocsong", "luuluongdieutiet"),
    "hydrology_training_10min.csv": (
        "rain_depth", "water_level_depth", "wind_speed", "wind_gust",
        "wind_direction", "wind_gust_direction",
    ),
}

SERIES_IDENTIFIER_COLUMNS = {
    "rain_observations.csv": "station_id",
    "water_level_observations.csv": "station_id",
    "wind_observations.csv": "sid",
    "reservoir_observations.csv": "tencongtrinh",
}


def detect_timestamp_column(frame: pd.DataFrame) -> str | None:
    return next((column for column in TIMESTAMP_CANDIDATES if column in frame.columns), None)


def summarize_numeric_column(series: pd.Series) -> dict[str, float | int | None]:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    summary: dict[str, float | int | None] = {
        "count": int(valid.size),
        "missing_ratio": float(numeric.isna().mean()),
        "mean": float(valid.mean()) if not valid.empty else None,
        "std": float(valid.std()) if len(valid) > 1 else None,
        "min": float(valid.min()) if not valid.empty else None,
        "max": float(valid.max()) if not valid.empty else None,
        "zero_ratio_of_valid": float(valid.eq(0).mean()) if not valid.empty else None,
    }
    for percentile, value in valid.quantile(PERCENTILES).items():
        summary[f"p{percentile * 100:g}"] = float(value)
    return summary


def summarize_dataset(path: Path) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    frame = read_csv(path)
    source_name = logical_csv_name(path)
    timestamp_column = detect_timestamp_column(frame)
    timestamp_summary: dict[str, object] = {}
    if timestamp_column == "ngaylaysolieu" and "gio" in frame.columns:
        timestamps = pd.to_datetime(
            frame[timestamp_column].astype(str) + " " + frame["gio"].astype(str),
            format="%d/%m/%Y %H:%M",
            errors="coerce",
            utc=True,
        )
    elif timestamp_column:
        timestamps = parse_utc(frame[timestamp_column])
    else:
        timestamps = pd.Series(dtype="datetime64[ns, UTC]")
    if not timestamps.empty:
        timestamp_summary = {
            "column": timestamp_column,
            "invalid_count": int(timestamps.isna().sum()),
            "start": timestamps.min(),
            "end": timestamps.max(),
            "duration_days": float((timestamps.max() - timestamps.min()).total_seconds() / 86_400),
            "nominal_interval_minutes": infer_nominal_interval_minutes(timestamps),
        }
    identifiers = [column for column in IDENTIFIER_CANDIDATES if column in frame.columns]
    categorical = {}
    for column in frame.select_dtypes(include=["object", "string"]).columns:
        if column == timestamp_column or frame[column].nunique(dropna=True) > 30:
            continue
        categorical[column] = sorted(map(str, frame[column].dropna().unique().tolist()))
    dataset_summary = {
        "file": source_name,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "timestamp": timestamp_summary,
        "identifier_columns": identifiers,
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_ids": {
            column: int(frame[column].duplicated().sum()) for column in identifiers
        },
        "missing": {
            column: {"count": int(frame[column].isna().sum()), "ratio": float(frame[column].isna().mean())}
            for column in frame.columns
        },
        "categories": categorical,
        "units": "Không có metadata đơn vị đáng tin cậy trong file; không suy đoán.",
    }
    numeric_summaries = {
        column: summarize_numeric_column(frame[column])
        for column in OBSERVATION_COLUMNS.get(source_name, ())
        if column in frame.columns
    }
    return dataset_summary, numeric_summaries


def plot_numeric_distributions(paths: AnalysisPaths, source_path: Path) -> None:
    frame = read_csv(source_path)
    source_name = logical_csv_name(source_path)
    columns = [column for column in OBSERVATION_COLUMNS.get(source_name, ()) if column in frame]
    if not columns:
        return
    figure, axes = plt.subplots(len(columns), 1, figsize=(10, max(3, 2.8 * len(columns))))
    axes_array = np.atleast_1d(axes)
    for axis, column in zip(axes_array, columns):
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        axis.hist(values, bins=80, color="#2673a2", alpha=0.85)
        axis.set_title(f"{source_name}: {column} (raw units, unmodified)")
        axis.set_ylabel("Count")
    axes_array[-1].set_xlabel("Value")
    figure.tight_layout()
    figure.savefig(paths.figures / f"distribution_{Path(source_name).stem}.png", dpi=150)
    plt.close(figure)


def run(paths: AnalysisPaths) -> None:
    paths.create_output_directories()
    inventory: list[dict[str, object]] = []
    distributions: dict[str, dict[str, dict[str, object]]] = {}
    extreme_rows: list[dict[str, object]] = []
    station_distribution_rows: list[dict[str, object]] = []
    csv_paths = sorted(paths.data.glob("*.csv")) + sorted(paths.data.glob("*.csv.gz"))
    try:
        training_path = find_training_table(paths.data, paths.reports)
        if training_path not in csv_paths:
            csv_paths.append(training_path)
    except FileNotFoundError:
        pass
    for csv_path in csv_paths:
        dataset_summary, numeric_summary = summarize_dataset(csv_path)
        inventory.append(dataset_summary)
        source_name = logical_csv_name(csv_path)
        distributions[source_name] = numeric_summary
        plot_numeric_distributions(paths, csv_path)
        frame = read_csv(csv_path)
        identifier_column = SERIES_IDENTIFIER_COLUMNS.get(source_name)
        for column in OBSERVATION_COLUMNS.get(source_name, ()):
            if column not in frame:
                continue
            numeric_values = pd.to_numeric(frame[column], errors="coerce").dropna()
            selected_indices = numeric_values.nsmallest(5).index.union(
                numeric_values.nlargest(5).index
            )
            for row_index in selected_indices:
                extreme_rows.append(
                    {
                        "file": source_name,
                        "column": column,
                        "row_index": int(row_index),
                        "value_raw_units": float(numeric_values.loc[row_index]),
                        "classification": "unknown requiring domain verification",
                        "reason": "No authoritative units, physical bounds, or sensor QA flags are supplied.",
                    }
                )
            if identifier_column:
                for series_id, group in frame.groupby(identifier_column, dropna=False):
                    station_distribution_rows.append(
                        {
                            "file": source_name,
                            "series_id": series_id,
                            "variable": column,
                            **summarize_numeric_column(group[column]),
                        }
                    )
    write_json(inventory, paths.reports / "dataset_inventory.json")
    write_json(distributions, paths.reports / "distribution_summary.json")
    inventory_table = pd.DataFrame(
        [
            {
                "file": entry["file"],
                "rows": entry["rows"],
                "columns": entry["columns"],
                "duplicate_rows": entry["duplicate_rows"],
                "timestamp_column": entry["timestamp"].get("column") if entry["timestamp"] else None,
                "start": entry["timestamp"].get("start") if entry["timestamp"] else None,
                "end": entry["timestamp"].get("end") if entry["timestamp"] else None,
            }
            for entry in inventory
        ]
    )
    inventory_table.to_csv(paths.reports / "dataset_inventory.csv", index=False)
    pd.DataFrame(extreme_rows).to_csv(paths.reports / "extreme_values.csv", index=False)
    pd.DataFrame(station_distribution_rows).to_csv(
        paths.reports / "distribution_by_station.csv", index=False
    )
    manifest_path = paths.data / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        declared_counts = {
            **manifest.get("row_counts", {}),
            **manifest.get("row_counts_with_features", {}),
        }
        for dataset in manifest.get("datasets", []):
            declared_counts[f"{dataset['name']}.csv"] = dataset.get("downloaded_total")
        actual_counts = {entry["file"]: entry["rows"] for entry in inventory}
        manifest_checks = [
            {
                "file": filename,
                "declared_rows": declared_rows,
                "actual_rows": actual_counts.get(filename),
                "matches": actual_counts.get(filename) == declared_rows,
            }
            for filename, declared_rows in declared_counts.items()
        ]
        write_json(manifest_checks, paths.reports / "manifest_validation.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=AnalysisPaths().data)
    parser.add_argument("--reports-dir", type=Path, default=AnalysisPaths().reports)
    parser.add_argument("--figures-dir", type=Path, default=AnalysisPaths().figures)
    arguments = parser.parse_args()
    run(AnalysisPaths(arguments.data_dir, arguments.reports_dir, arguments.figures_dir))


if __name__ == "__main__":
    main()
