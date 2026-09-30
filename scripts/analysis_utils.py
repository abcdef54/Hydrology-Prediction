"""Reusable, side-effect-light utilities for hydrology analysis."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def read_csv(path: Path) -> pd.DataFrame:
    """Read a UTF-8 CSV while accepting an optional BOM."""
    return pd.read_csv(resolve_csv_path(path), encoding="utf-8-sig", low_memory=False)


def resolve_csv_path(path: Path) -> Path:
    """Resolve a logical .csv path to either plain CSV or gzip-compressed CSV."""
    if path.exists():
        return path
    compressed_path = Path(f"{path}.gz")
    if compressed_path.exists():
        return compressed_path
    raise FileNotFoundError(f"Neither {path} nor {compressed_path} exists")


def logical_csv_name(path: Path) -> str:
    """Return a stable CSV name for plain and .csv.gz inputs."""
    return path.name.removesuffix(".gz")


def find_training_table(data_directory: Path, reports_directory: Path) -> Path:
    """Locate an existing raw or derived training table."""
    candidates = (
        data_directory / "hydrology_training_10min.csv",
        reports_directory / "hydrology_training_10min.csv",
    )
    for candidate in candidates:
        try:
            return resolve_csv_path(candidate)
        except FileNotFoundError:
            continue
    raise FileNotFoundError("hydrology_training_10min.csv has not been built")


def parse_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, pd.Timedelta, Path)):
        return str(value)
    if pd.isna(value):
        return None
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    clean_frame = frame.astype(object).where(pd.notna(frame), None)
    return clean_frame.to_dict(orient="records")


def infer_nominal_interval_minutes(timestamps: pd.Series) -> float | None:
    ordered = parse_utc(timestamps).dropna().drop_duplicates().sort_values()
    intervals = ordered.diff().dropna().dt.total_seconds().div(60)
    positive_intervals = intervals[intervals > 0]
    if positive_intervals.empty:
        return None
    modes = positive_intervals.mode()
    return float(modes.iloc[0] if not modes.empty else positive_intervals.median())


def haversine_km(
    longitude_a: float,
    latitude_a: float,
    longitude_b: float,
    latitude_b: float,
) -> float:
    earth_radius_km = 6_371.0088
    lon_a, lat_a, lon_b, lat_b = map(
        math.radians, (longitude_a, latitude_a, longitude_b, latitude_b)
    )
    delta_lon = lon_b - lon_a
    delta_lat = lat_b - lat_a
    central_angle = 2 * math.asin(
        math.sqrt(
            math.sin(delta_lat / 2) ** 2
            + math.cos(lat_a) * math.cos(lat_b) * math.sin(delta_lon / 2) ** 2
        )
    )
    return earth_radius_km * central_angle


def parse_geojson_point(value: object) -> tuple[float, float] | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        geometry = json.loads(value)
        coordinates = geometry.get("coordinates")
        if geometry.get("type") != "Point" or len(coordinates) < 2:
            return None
        return float(coordinates[0]), float(coordinates[1])
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def metric_bundle(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    actual = observed[valid]
    forecast = predicted[valid]
    if actual.size == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "r2": np.nan, "nse": np.nan}
    residual = actual - forecast
    squared_error = float(np.sum(residual**2))
    variance_sum = float(np.sum((actual - actual.mean()) ** 2))
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    r2 = 1 - squared_error / variance_sum if variance_sum > 0 else np.nan
    return {"n": int(actual.size), "mae": mae, "rmse": rmse, "r2": r2, "nse": r2}


def markdown_table(frame: pd.DataFrame, columns: Iterable[str] | None = None) -> str:
    selected = frame.loc[:, list(columns)] if columns is not None else frame
    if selected.empty:
        return "_Không có dữ liệu._"
    display = selected.copy()
    for column in display.columns:
        display[column] = display[column].map(format_markdown_value)
    header = "| " + " | ".join(map(str, display.columns)) + " |"
    separator = "| " + " | ".join("---" for _ in display.columns) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in display.to_numpy()]
    return "\n".join([header, separator, *rows])


def format_markdown_value(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def contiguous_events(
    timestamps: pd.Series,
    values: pd.Series,
    threshold: float,
    maximum_break: pd.Timedelta,
) -> list[dict[str, Any]]:
    observations = pd.DataFrame(
        {"timestamp": parse_utc(timestamps), "value": pd.to_numeric(values, errors="coerce")}
    ).dropna()
    observations = observations.loc[observations["value"] > threshold].sort_values("timestamp")
    if observations.empty:
        return []
    group_ids = observations["timestamp"].diff().gt(maximum_break).cumsum()
    events: list[dict[str, Any]] = []
    for _, event in observations.groupby(group_ids):
        peak_index = event["value"].idxmax()
        start = event["timestamp"].iloc[0]
        end = event["timestamp"].iloc[-1]
        events.append(
            {
                "start": start,
                "end": end,
                "peak_time": event.loc[peak_index, "timestamp"],
                "peak_value": float(event.loc[peak_index, "value"]),
                "duration_hours": float((end - start).total_seconds() / 3600),
                "observations_above_threshold": int(len(event)),
            }
        )
    return events
