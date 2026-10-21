"""Repartition the hourly pool: former train+validation -> train, test -> validation."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from research.build_output24_dataset import COMMON, TARGETS, add_targets, read_frame
from research.build_rain_station_dataset import DEFAULT_POOL
from src.ai.comparison import comparison_observations_source


DEFAULT_OUTPUT = DEFAULT_POOL.with_name("da_vien_1h_2026-10-15_expanded_train_o24")


def build(source_dir=DEFAULT_POOL, output_dir=DEFAULT_OUTPUT):
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refuse to overwrite {output_dir}")
    spec = json.loads((source_dir / "manifest.json").read_text())
    if spec.get("resolution_minutes") != 60 or spec.get("target_columns") != TARGETS:
        raise ValueError("Source must be the hourly O24 pool")
    hourly = read_frame(source_dir / "hourly.csv")
    original = {name: read_frame(source_dir / f"{name}.csv") for name in ("train", "val", "test")}
    # Recompute labels only to verify provenance; preserve all source values.
    expected = add_targets(hourly)
    np.testing.assert_allclose(hourly[TARGETS], expected[TARGETS], atol=1e-8, rtol=0, equal_nan=True)
    for frame in original.values():
        pd.testing.assert_frame_equal(frame, hourly.loc[frame.index, frame.columns])
    train = hourly.loc[original["train"].index.min():original["val"].index.max()].copy()
    validation = hourly.loc[original["test"].index].copy()
    if train.index.max() + pd.Timedelta(hours=24) >= validation.index.min():
        raise ValueError("A 24h target purge is required before new validation")
    reference_csv, _ = comparison_observations_source(source_dir / "test.csv")
    reference_csv = reference_csv or source_dir / "test.csv"
    if not reference_csv.is_file():
        raise FileNotFoundError("Original production observation snapshot is required")
    # Copy the original production inputs so comparisons stay portable and exact.
    reference = read_frame(reference_csv)
    pd.testing.assert_index_equal(reference.index, validation.index)
    for column in COMMON:
        np.testing.assert_allclose(reference[column], validation[column], rtol=0, atol=1e-8, equal_nan=True)
    splits = {"train": train, "val": validation}
    coverage = {}
    for name, frame in splits.items():
        eligible = frame[COMMON].notna().all(axis=1) & frame.water_level_depth.notna()
        coverage[name] = {
            "rows": len(frame), "start_utc": str(frame.index.min()), "end_utc": str(frame.index.max()),
            "sequences_96": int((eligible & (np.arange(len(frame)) >= 95)).sum()),
        }
    result = {
        **spec, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "expanded_train", "parent_dataset": str(source_dir),
        "splits": coverage, "has_independent_test": False,
        "validation_origin": "former test; already inspected; used for checkpoint selection",
        "repartition_policy": "continuous hourly train through former val end; former test unchanged",
        "restored_internal_purge_rows": len(train) - len(original["train"]) - len(original["val"]),
        "reference_validation_csv": "reference_validation.csv", "purge_hours": 24,
        "caveats": spec.get("caveats", []) + [
            "No independent test remains; all reported held-out metrics are validation metrics.",
            "No synthetic observations or label-based predictor selection.",
        ],
    }
    for key in ("comparison_observations_dir", "source_split_sha256"):
        result.pop(key, None)
    output_dir.mkdir(parents=True)
    hourly.to_csv(output_dir / "hourly.csv", index_label="timestamp_utc")
    for name, frame in splits.items():
        frame.to_csv(output_dir / f"{name}.csv", index_label="timestamp_utc")
    reference.to_csv(output_dir / "reference_validation.csv", index_label="timestamp_utc")
    provenance = source_dir / "weather_provenance.csv.gz"
    if provenance.is_file():
        shutil.copyfile(provenance, output_dir / provenance.name)
    (output_dir / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output_dir": str(output_dir), "splits": coverage,
                      "restored_internal_purge_rows": result["restored_internal_purge_rows"],
                      "has_independent_test": False}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.source_dir, args.output_dir)
