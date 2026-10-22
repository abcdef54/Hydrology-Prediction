"""Shared hourly dataset paths, physical targets and timestamp validation."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POOL = ROOT / "train_data/da_vien_1h_2026-10-15_rain_pool_o24"
COMMON = [f"target_water_level_plus_{hours * 60}m" for hours in (1, 3, 6, 12, 24)]
TARGETS = [f"target_water_level_plus_{hours * 60}m" for hours in range(1, 25)]


def read_frame(path):
    """Read an ordered hourly UTC grid, retaining missing feature/label values."""
    frame = pd.read_csv(path)
    timestamps = pd.DatetimeIndex(
        pd.to_datetime(frame.pop("timestamp_utc"), utc=True), name="timestamp_utc"
    )
    if (
        not timestamps.is_unique
        or not timestamps.is_monotonic_increasing
        or not timestamps.to_series().diff().dropna().eq(pd.Timedelta(hours=1)).all()
    ):
        raise ValueError(f"Expected a complete ordered hourly grid: {path}")
    frame.index = timestamps
    return frame


def add_targets(source):
    """Use observed hourly means at exact future timestamps, never interpolation."""
    water = source.water_level_depth
    labels = pd.DataFrame(
        {
            column: water.reindex(source.index + pd.Timedelta(hours=hours)).to_numpy()
            for hours, column in enumerate(TARGETS, 1)
        },
        index=source.index,
    )
    for column in COMMON:
        if column not in source or not np.allclose(
            source[column], labels[column], rtol=0, atol=1e-9, equal_nan=True
        ):
            raise ValueError(f"Existing target does not agree with hourly observation: {column}")
    return labels


def assert_matched_population(labels, water, sequence_length=96):
    """Require O5/O24 to use identical label-complete anchors."""
    all_complete = labels[TARGETS].notna().all(axis=1)
    common_complete = labels[COMMON].notna().all(axis=1)
    if not all_complete.equals(common_complete):
        raise ValueError(
            "O5/O24 have different complete-target rows; resolve paired eligibility "
            "explicitly before building"
        )
    eligible = all_complete & water.notna()
    return int((eligible & (np.arange(len(labels)) >= sequence_length - 1)).sum())
