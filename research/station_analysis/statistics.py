"""Exact-time descriptive water-level analysis; no target interpolation or fitting.

Values are in the project's observation unit (mm), not DEM elevation or a shared
vertical datum. Window counts inspect water history/targets only; rainfall and
weather availability must still be checked by a future dataset builder.
"""

from __future__ import annotations

from itertools import combinations
import math

import numpy as np
import pandas as pd
from scipy.signal import periodogram


def hourly_grid(hourly: pd.DataFrame, station_ids: list[str]) -> pd.DataFrame:
    """Insert absent hours without filling values; reject ambiguous timestamps."""
    if not isinstance(hourly.index, pd.DatetimeIndex) or hourly.index.tz is None:
        raise ValueError("hourly must have a timezone-aware DatetimeIndex")
    if hourly.index.has_duplicates:
        raise ValueError("Duplicate hourly timestamps are ambiguous")
    result = hourly.copy()
    result.index = result.index.tz_convert("UTC")
    if not result.index.equals(result.index.floor("h")):
        raise ValueError("Hourly data must use exact hour boundaries")
    if result.columns.duplicated().any():
        raise ValueError("Duplicate station columns are ambiguous")
    result.columns = result.columns.astype(str)
    result = result.reindex(columns=station_ids).apply(pd.to_numeric, errors="coerce")
    if len(result):
        result = result.sort_index().reindex(
            pd.date_range(result.index.min(), result.index.max(), freq="h")
        )
    return result.replace([np.inf, -np.inf], np.nan)


def exact_change(series: pd.Series, hours: int) -> pd.Series:
    """H(t+h)-H(t), by timestamp even if the caller's index has holes."""
    if hours <= 0:
        raise ValueError("Horizon must be positive")
    future = series.reindex(series.index + pd.Timedelta(hours=hours))
    return pd.Series(future.to_numpy() - series.to_numpy(), index=series.index)


def _describe(series: pd.Series) -> dict:
    values = series.dropna()
    result = {"valid_hours": int(len(values))}
    for name in ("min", "max", "mean", "median", "std"):
        result[f"{name}_mm"] = float(getattr(values, name)()) if len(values) else np.nan
    for percentile in (1, 5, 10, 25, 75, 90, 95, 99):
        result[f"p{percentile:02d}_mm"] = (
            float(values.quantile(percentile / 100)) if len(values) else np.nan
        )
    result["zero_fraction"] = float(values.eq(0).mean()) if len(values) else np.nan
    result["negative_fraction"] = float(values.lt(0).mean()) if len(values) else np.nan
    return result


def missing_runs(series: pd.Series) -> pd.DataFrame:
    """Return contiguous missing intervals, including leading/trailing gaps."""
    columns = ["start_utc", "end_utc", "missing_hours"]
    if series.empty:
        return pd.DataFrame(columns=columns)
    missing = series.isna()
    labels = missing.ne(missing.shift(fill_value=False)).cumsum()
    rows = []
    for _, block in missing[missing].groupby(labels[missing]):
        rows.append(
            {"start_utc": block.index[0], "end_utc": block.index[-1],
             "missing_hours": len(block)}
        )
    return pd.DataFrame(rows, columns=columns)


def window_counts(series: pd.Series, sequence_length: int, horizons: list[int],
                  observed_fraction: float = 0.8) -> dict:
    """Count feasible water-only windows; never treat missing labels as valid."""
    finite = series.notna()
    history_count = finite.rolling(sequence_length, min_periods=sequence_length).sum()
    target_valid = pd.Series(True, index=series.index)
    for horizon in horizons:
        future = series.reindex(series.index + pd.Timedelta(hours=horizon))
        target_valid &= pd.Series(future.notna().to_numpy(), index=series.index)
    strict = history_count.eq(sequence_length) & target_valid
    partial = (
        history_count.ge(math.ceil(sequence_length * observed_fraction))
        & target_valid & finite
    )
    return {
        "strict_water_windows": int(strict.sum()),
        "history80_water_windows": int(partial.sum()),
        "possible_full_span_windows": max(0, len(series) - sequence_length - max(horizons) + 1),
    }


def _corr(left: pd.Series, right: pd.Series, min_pairs: int) -> tuple[int, float]:
    pair = pd.concat([left, right], axis=1).dropna()
    count = len(pair)
    if count < min_pairs or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return count, np.nan
    return count, float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))


def _future_values(series: pd.Series, hours: int) -> pd.Series:
    return pd.Series(
        series.reindex(series.index + pd.Timedelta(hours=hours)).to_numpy(),
        index=series.index,
    )


def _longest_finite(series: pd.Series) -> pd.Series:
    valid = series.notna()
    if not valid.any():
        return series.iloc[:0]
    labels = valid.ne(valid.shift(fill_value=False)).cumsum()
    sizes = valid[valid].groupby(labels[valid]).size()
    return series.loc[(labels == sizes.idxmax()) & valid]


def _dynamics_features(series: pd.Series, min_pairs: int) -> dict:
    """Offset/scale-resistant descriptors; spectral fractions are tide proxies."""
    values = series.dropna()
    iqr = float(values.quantile(0.75) - values.quantile(0.25)) if len(values) else np.nan
    scale_ok = bool(np.isfinite(iqr) and iqr > 0)
    result = {
        "observed_hours": len(values),
        "iqr_mm": iqr,
        "scale_available": scale_ok,
        "zero_fraction": float(values.eq(0).mean()) if len(values) else np.nan,
    }
    for horizon in (1, 6, 12, 24):
        count, value = _corr(series, _future_values(series, horizon), min_pairs)
        result[f"autocorr_{horizon}h"] = value
        result[f"autocorr_{horizon}h_pairs"] = count
        if horizon in (6, 12):
            changes = exact_change(series, horizon).dropna()
            result[f"change_{horizon}h_iqr_ratio"] = (
                float((changes.quantile(0.75) - changes.quantile(0.25)) / iqr)
                if scale_ok and len(changes) else np.nan
            )
            result[f"change_{horizon}h_abs_p95_iqr_ratio"] = (
                float(changes.abs().quantile(0.95) / iqr)
                if scale_ok and len(changes) else np.nan
            )
            result[f"change_{horizon}h_positive_fraction"] = (
                float(changes.gt(0).mean()) if len(changes) else np.nan
            )
    block = _longest_finite(series)
    result["spectral_contiguous_hours"] = len(block)
    result["diurnal_power_fraction_proxy"] = np.nan
    result["semidiurnal_power_fraction_proxy"] = np.nan
    if len(block) >= 14 * 24 and block.nunique() > 1:
        frequency, power = periodogram(block.to_numpy(), fs=1, detrend="linear")
        total = power[(frequency >= 1 / 72) & (frequency <= 1 / 4)].sum()
        if total > 0:
            result["diurnal_power_fraction_proxy"] = float(
                power[(frequency >= 1 / 28) & (frequency <= 1 / 20)].sum() / total
            )
            result["semidiurnal_power_fraction_proxy"] = float(
                power[(frequency >= 1 / 14) & (frequency <= 1 / 10)].sum() / total
            )
    return result


def _event_rows(station_id: str, kind: str, flagged: pd.Series,
                magnitude: pd.Series, gap_hours: int, horizon: int) -> list[dict]:
    timestamps = flagged.index[flagged.fillna(False)]
    if not len(timestamps):
        return []
    groups = (timestamps.to_series().diff() > pd.Timedelta(hours=gap_hours)).cumsum()
    rows = []
    for _, times in timestamps.to_series().groupby(groups):
        selected = magnitude.reindex(pd.DatetimeIndex(times)).dropna()
        peak = selected.abs().idxmax() if len(selected) else pd.NaT
        rows.append({
            "station_id": station_id, "kind": kind, "horizon_hours": horizon,
            "first_anchor_utc": times.iloc[0], "last_anchor_utc": times.iloc[-1],
            "flagged_anchors": len(times), "peak_anchor_utc": peak,
            "peak_value_mm": float(selected.loc[peak]) if len(selected) else np.nan,
            "gap_merge_hours": gap_hours,
        })
    return rows


def analyze(hourly: pd.DataFrame, stations: pd.DataFrame, config: dict) -> dict[str, pd.DataFrame]:
    """Analyze every metadata station, retaining empty/constant stations.

    `grouping_end`, when supplied, is an exclusive timezone-aware cutoff used
    only for grouping_features and grouping_correlations. All other tables are
    descriptive over the complete requested range and must not select sources
    for a future held-out evaluation. Missingness is measured on hourly means;
    source record counts and cadence are the database access stage's concern.
    """
    if "station_id" in stations:
        station_ids = stations.station_id.astype(str).tolist()
    elif "id" in stations:
        station_ids = stations.id.astype(str).tolist()
    else:
        station_ids = stations.index.astype(str).tolist()
    if len(set(station_ids)) != len(station_ids):
        raise ValueError("Station metadata IDs must be unique")
    frame = hourly_grid(hourly, station_ids)
    sequence_length = int(config.get("sequence_length", 96))
    horizons = [int(h) for h in config.get("horizons", [6, 12])]
    threshold = float(config.get("rapid_threshold_mm", 200))
    event_gap = int(config.get("event_gap_hours", 12))
    min_pairs = int(config.get("correlation_min_pairs", 48))
    months = set(config.get("flood_months", [9, 10, 11, 12]))
    timezone = config.get("timezone", "Asia/Ho_Chi_Minh")
    if sequence_length < 1 or not horizons or min(horizons) < 1:
        raise ValueError("Sequence length and horizons must be positive")
    if threshold <= 0 or event_gap < 1:
        raise ValueError("Rapid threshold and event gap must be positive")
    local_index = frame.index.tz_convert(timezone)
    grouping_end = config.get("grouping_end")
    grouping = frame
    if grouping_end:
        cutoff = pd.Timestamp(grouping_end)
        if cutoff.tzinfo is None:
            raise ValueError("grouping_end must include a timezone")
        grouping = frame.loc[frame.index < cutoff]
    summaries, gaps, monthly, seasonal = [], [], [], []
    changes_rows, change_summaries, events, features = [], [], [], []
    for station_id in station_ids:
        series = frame[station_id]
        values = series.dropna()
        station_gaps = missing_runs(series)
        station_gaps.insert(0, "station_id", station_id)
        gaps.append(station_gaps)
        active = series.loc[values.index.min():values.index.max()] if len(values) else series.iloc[:0]
        summaries.append({
            "station_id": station_id,
            "requested_start_utc": series.index.min(),
            "requested_end_utc": series.index.max(),
            "requested_hours": len(series),
            "first_valid_utc": values.index.min(), "last_valid_utc": values.index.max(),
            "missing_hours": int(series.isna().sum()),
            "missing_rate": float(series.isna().mean()) if len(series) else np.nan,
            "active_span_hours": len(active),
            "active_span_missing_rate": float(active.isna().mean()) if len(active) else np.nan,
            "longest_missing_hours": int(station_gaps.missing_hours.max()) if len(station_gaps) else 0,
            **_describe(series), **window_counts(series, sequence_length, horizons),
        })
        month_labels = local_index.strftime("%Y-%m")
        for label in pd.unique(month_labels):
            part = series.loc[month_labels == label]
            monthly.append({"station_id": station_id, "month": label,
                            "expected_hours": len(part),
                            "missing_rate": float(part.isna().mean()), **_describe(part)})
        for label, mask in (("flood_months", local_index.month.isin(months)),
                            ("other_months", ~local_index.month.isin(months))):
            part = series.loc[mask]
            seasonal.append({"station_id": station_id, "season": label,
                             "expected_hours": len(part),
                             "missing_rate": float(part.isna().mean()) if len(part) else np.nan,
                             **_describe(part)})
        for horizon in horizons:
            change = exact_change(series, horizon)
            finite_change = change.dropna()
            changes_rows.append(pd.DataFrame({
                "station_id": station_id, "anchor_utc": series.index,
                "target_utc": series.index + pd.Timedelta(hours=horizon),
                "horizon_hours": horizon, "change_mm": change.to_numpy(),
            }))
            change_summaries.append({
                "station_id": station_id, "horizon_hours": horizon,
                **_describe(change),
                "mean_abs_change_mm": float(finite_change.abs().mean()) if len(finite_change) else np.nan,
                "abs_change_p95_mm": float(finite_change.abs().quantile(0.95)) if len(finite_change) else np.nan,
                "rise_above_threshold_hours": int(finite_change.gt(threshold).sum()),
                "fall_below_threshold_hours": int(finite_change.lt(-threshold).sum()),
                "rapid_threshold_mm": threshold,
            })
            events.extend(_event_rows(station_id, "rapid_rise", change.gt(threshold),
                                      change, event_gap, horizon))
            events.extend(_event_rows(station_id, "rapid_fall", change.lt(-threshold),
                                      change, event_gap, horizon))
        if len(values) and values.nunique() > 1:
            # A sparse flood spike can leave p95 equal to the dry baseline.
            # Excluding the median prevents calling the whole baseline an event.
            high = series.ge(values.quantile(0.95)) & series.gt(values.median())
            events.extend(_event_rows(station_id, "high_level_p95", high, series, event_gap, 0))
        features.append({
            "station_id": station_id,
            "scope": "before_grouping_end" if grouping_end else "full_period_descriptive_only",
            "grouping_end_exclusive": grouping_end,
            **_dynamics_features(grouping[station_id], min_pairs),
        })
    tables = {
        "station_statistics": pd.DataFrame(summaries),
        "missing_gaps": pd.concat(gaps, ignore_index=True) if gaps else pd.DataFrame(),
        "monthly_statistics": pd.DataFrame(monthly),
        "seasonal_statistics": pd.DataFrame(seasonal),
        "changes": pd.concat(changes_rows, ignore_index=True) if changes_rows else pd.DataFrame(),
        "change_statistics": pd.DataFrame(change_summaries),
        "events": pd.DataFrame(events, columns=[
            "station_id", "kind", "horizon_hours", "first_anchor_utc", "last_anchor_utc",
            "flagged_anchors", "peak_anchor_utc", "peak_value_mm", "gap_merge_hours",
        ]),
        "grouping_features": pd.DataFrame(features),
        "correlations": correlations(frame, min_pairs),
        "grouping_correlations": correlations(grouping, min_pairs),
    }
    return tables


def correlations(frame: pd.DataFrame, min_pairs: int = 48) -> pd.DataFrame:
    """Pairwise exact-time correlations and lag scan with overlap counts.

    A positive lag compares station_a(t) with station_b(t+lag). No flow direction
    or causal connection follows from a maximal Pearson coefficient.
    """
    rows = []
    transforms = {"level_centered": frame - frame.median()}
    for hours in (1, 6, 12):
        transforms[f"change_{hours}h"] = pd.DataFrame(
            {column: exact_change(frame[column], hours) for column in frame}, index=frame.index
        )
    for kind, transformed in transforms.items():
        for left, right in combinations(frame.columns, 2):
            lags = (-24, -12, -6, 0, 6, 12, 24) if kind == "change_1h" else (0,)
            for lag in lags:
                count, value = _corr(transformed[left], _future_values(transformed[right], lag), min_pairs)
                rows.append({
                    "station_a": left, "station_b": right, "kind": kind,
                    "lag_hours": lag, "paired_hours": count, "pearson_r": value,
                })
    return pd.DataFrame(rows, columns=[
        "station_a", "station_b", "kind", "lag_hours", "paired_hours", "pearson_r"
    ])
