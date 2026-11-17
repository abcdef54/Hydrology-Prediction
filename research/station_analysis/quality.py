"""Non-destructive screening of sensor series, cadence and hourly aggregation.

Flags indicate records needing review, not verified errors. Negative readings
are permitted, and none of these screens alters observations or window counts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.station_analysis.statistics import hourly_grid


FLAG_COLUMNS = [
    "station_id", "kind", "start_utc", "end_utc", "hours", "value_mm",
    "previous_mm", "next_mm", "magnitude_mm", "threshold_mm", "status", "interpretation",
]


def _runs(mask: pd.Series) -> list[pd.DatetimeIndex]:
    mask = mask.fillna(False)
    groups = mask.ne(mask.shift(fill_value=False)).cumsum()
    return [pd.DatetimeIndex(part.index) for _, part in mask[mask].groupby(groups[mask])]


def _flag(station_id, kind, start, end, **values):
    return {
        "station_id": station_id, "kind": kind, "start_utc": start, "end_utc": end,
        "hours": int((end - start) / pd.Timedelta(hours=1)) + 1,
        "status": "needs_review_not_verified_error",
        "interpretation": "screening only; preserve raw values and investigate sensor/reference context",
        **values,
    }


def _hourly_flags(series, station_id, config):
    jump = float(config.get("quality_jump_mm", 1000))
    return_tolerance = float(config.get("quality_spike_return_mm", 100))
    constant_hours = int(config.get("quality_plateau_hours", 24))
    range_hours = int(config.get("quality_range_plateau_hours", 72))
    range_tolerance = float(config.get("quality_plateau_tolerance_mm", 5))
    if min(jump, constant_hours, range_hours) <= 0 or min(return_tolerance, range_tolerance) < 0:
        raise ValueError("Quality-screen thresholds and durations must be nonnegative/positive")
    rows = []
    difference = series.diff()
    for stamp, value in difference[difference.abs() > jump].items():
        rows.append(_flag(
            station_id, "hourly_jump", stamp, stamp, value_mm=series.loc[stamp],
            previous_mm=series.shift(1).loc[stamp], magnitude_mm=value, threshold_mm=jump,
        ))
    before, after = series.shift(1), series.shift(-1)
    spike = (
        (series - before).abs().gt(jump) & (series - after).abs().gt(jump)
        & (before - after).abs().le(return_tolerance)
    )
    for stamp in series.index[spike]:
        rows.append(_flag(
            station_id, "isolated_reverting_spike", stamp, stamp,
            value_mm=series.loc[stamp], previous_mm=before.loc[stamp], next_mm=after.loc[stamp],
            magnitude_mm=float(series.loc[stamp] - (before.loc[stamp] + after.loc[stamp]) / 2),
            threshold_mm=jump,
        ))
    valid = series.notna()
    labels = (series.ne(series.shift()) | ~valid).cumsum()
    for _, block in series[valid].groupby(labels[valid]):
        if len(block) >= constant_hours:
            rows.append(_flag(
                station_id, "exact_zero_plateau" if block.iloc[0] == 0 else "exact_constant_plateau",
                block.index[0], block.index[-1], value_mm=block.iloc[0], magnitude_mm=0,
                threshold_mm=0,
            ))
    rolling = series.rolling(range_hours, min_periods=range_hours)
    stable = (rolling.max() - rolling.min()).le(range_tolerance)
    for times in _runs(stable):
        start, end = times[0] - pd.Timedelta(hours=range_hours - 1), times[-1]
        block = series.loc[start:end]
        # Exact plateaus already have their own simpler diagnostic. A nonzero
        # rolling-range flag can span gradual drift; report the full span range.
        if block.nunique() > 1:
            rows.append(_flag(
                station_id, "low_variability_rolling_plateau", start, end,
                value_mm=float(block.median()), magnitude_mm=float(block.max() - block.min()),
                threshold_mm=range_tolerance,
            ))
    return rows


def audit_quality(hourly: pd.DataFrame, raw: pd.DataFrame, stations: pd.DataFrame,
                  config: dict) -> dict[str, pd.DataFrame]:
    """Screen original hourly means and raw cadence without cleaning either.

    Raw data requires station_id, observed_at, depth_mm (or depth) and optionally
    quality_status. Non-mm/unknown-unit stations still receive cadence summaries,
    but are excluded from mm-threshold value screens. All timestamps are UTC.
    """
    metadata = stations.copy()
    if "station_id" not in metadata:
        metadata["station_id"] = metadata["id"] if "id" in metadata else metadata.index
    metadata["station_id"] = metadata.station_id.astype(str)
    if metadata.station_id.duplicated().any():
        raise ValueError("Station metadata must have unique station_id")
    ids = metadata.station_id.tolist()
    frame = hourly_grid(hourly, ids)
    observations = raw.copy()
    required = {"station_id", "observed_at"}
    if not required <= set(observations):
        raise ValueError("Raw observations require station_id and observed_at")
    depth_column = "depth_mm" if "depth_mm" in observations else "depth"
    if depth_column not in observations:
        raise ValueError("Raw observations require depth_mm or depth")
    observations["station_id"] = observations.station_id.astype(str)
    observations["observed_at"] = pd.to_datetime(observations.observed_at, utc=True, errors="raise")
    if observations.observed_at.isna().any():
        raise ValueError("Raw observation timestamps must not be missing")
    observations["screen_value"] = pd.to_numeric(observations[depth_column], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    flags, cadence_rows, summary_rows = [], [], []
    for station_id in ids:
        station = metadata.loc[metadata.station_id == station_id].iloc[0]
        unit_known = str(station.get("measurement_unit", "")).strip().lower() == "mm"
        series = frame[station_id]
        selected = observations.loc[observations.station_id == station_id].sort_values("observed_at")
        times = selected.observed_at
        intervals = times.diff().dt.total_seconds().iloc[1:]
        cadence = {
            "station_id": station_id, "raw_records": len(selected),
            "first_raw_utc": times.min(), "last_raw_utc": times.max(),
            "interval_count": len(intervals),
            "median_interval_minutes": float(intervals.median() / 60) if len(intervals) else np.nan,
            "p95_interval_minutes": float(intervals.quantile(.95) / 60) if len(intervals) else np.nan,
            "max_interval_minutes": float(intervals.max() / 60) if len(intervals) else np.nan,
            "non_10min_intervals": int((~np.isclose(intervals, 600, atol=1, rtol=0)).sum()),
            "intervals_over_60min": int(intervals.gt(3600).sum()),
            "duplicate_timestamp_extra_rows": int(times.duplicated().sum()),
            "raw_missing_or_nonfinite_values": int(selected.screen_value.isna().sum()),
            "source_quality_missing_rows": int(selected.quality_status.eq("missing").sum())
            if "quality_status" in selected else np.nan,
            "source_quality_other_rows": int((~selected.quality_status.isin(["observed", "missing"])).sum())
            if "quality_status" in selected else np.nan,
        }
        cadence_rows.append(cadence)
        for stamp, duplicates in selected.loc[times.duplicated(keep=False)].groupby("observed_at"):
            flags.append(_flag(
                station_id, "duplicate_raw_timestamp", stamp, stamp,
                magnitude_mm=np.nan, threshold_mm=np.nan, value_mm=np.nan,
            ))
        station_flags = _hourly_flags(series, station_id, config) if unit_known else []
        flags.extend(station_flags)
        if unit_known and len(selected):
            hourly_ranges = selected.groupby(selected.observed_at.dt.floor("h")).screen_value.agg(
                ["min", "max", "count"]
            )
            ranges = hourly_ranges["max"] - hourly_ranges["min"]
            for stamp in ranges.index[(hourly_ranges["count"] >= 2) & ranges.gt(config.get("quality_jump_mm", 1000))]:
                row = _flag(
                    station_id, "raw_within_hour_large_range", stamp, stamp,
                    value_mm=hourly_ranges.loc[stamp, "max"],
                    previous_mm=hourly_ranges.loc[stamp, "min"],
                    magnitude_mm=ranges.loc[stamp], threshold_mm=config.get("quality_jump_mm", 1000),
                )
                flags.append(row)
                station_flags.append(row)
        counts = pd.Series([row["kind"] for row in station_flags]).value_counts()
        summary_rows.append({
            "station_id": station_id, "value_screens_applied": unit_known,
            "screen_status": "descriptive_flags_only_no_cleaning",
            "hourly_jump_flags": int(counts.get("hourly_jump", 0)),
            "isolated_spike_flags": int(counts.get("isolated_reverting_spike", 0)),
            "exact_zero_plateaus": int(counts.get("exact_zero_plateau", 0)),
            "exact_constant_plateaus": int(counts.get("exact_constant_plateau", 0)),
            "low_variability_plateaus": int(counts.get("low_variability_rolling_plateau", 0)),
            "raw_within_hour_large_ranges": int(counts.get("raw_within_hour_large_range", 0)),
            "negative_hourly_values": int(series.lt(0).sum()),
            "negative_values_are_not_automatically_errors": True,
            "quality_jump_threshold_mm": config.get("quality_jump_mm", 1000),
        })
    return {
        "quality_flags": pd.DataFrame(flags, columns=FLAG_COLUMNS),
        "raw_cadence": pd.DataFrame(cadence_rows),
        "quality_summary": pd.DataFrame(summary_rows),
    }
