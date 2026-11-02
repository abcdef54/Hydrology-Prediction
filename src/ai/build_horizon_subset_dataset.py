"""Project O24 to +6/+12h without changing inputs, splits or eligible anchors."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd

from src.ai.dataset import HydrologyDataset
from src.ai.hourly_dataset import ROOT, TARGETS, add_targets, read_frame


DEFAULT_SOURCE = ROOT / "train_data/da_vien_1h_2026-10-15_forecast_mask_o24"
DEFAULT_OUTPUT = ROOT / "train_data/da_vien_1h_2026-10-15_forecast_mask_o2_h6_h12"
SELECTED_TARGETS = ["target_water_level_plus_360m", "target_water_level_plus_720m"]


def project_csv(source, destination, columns, complete, additional_columns=None):
    """Preserve input/target text exactly; mask labels outside the O24 population."""
    with source.open(newline="") as incoming, destination.open("w", newline="") as outgoing:
        reader = csv.DictReader(incoming)
        writer = csv.DictWriter(outgoing, fieldnames=["timestamp_utc", *columns])
        writer.writeheader()
        count = 0
        for count, row in enumerate(reader, start=1):
            projected = {
                column: row[column] if column in row else additional_columns[column][count - 1]
                for column in writer.fieldnames
            }
            if not complete[count - 1]:
                for target in SELECTED_TARGETS:
                    projected[target] = ""
            writer.writerow(projected)
        if count != len(complete):
            raise ValueError("Source changed while projecting CSV rows")


def make_dataset(frame, features, targets, training=None):
    return HydrologyDataset(
        frame.reset_index(), feature_cols=features, target_cols=targets,
        seq_len=96, require_current_level=True, fit_scalers=training is None,
        feature_scaler=None if training is None else training.feature_scaler,
        target_scaler=None if training is None else training.target_scaler,
    )


def build(source_dir=DEFAULT_SOURCE, output_dir=DEFAULT_OUTPUT):
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refuse to overwrite: {output_dir}")
    spec = json.loads((source_dir / "manifest.json").read_text())
    features = spec["feature_columns"]
    if (spec["target_columns"] != TARGETS or len(features) != 26
            or len(set(features)) != 26 or any(c.startswith("target_") for c in features)
            or spec.get("sequence_length") != 96 or spec.get("purge_hours") != 24
            or spec.get("resolution_minutes") != 60 or spec.get("variant") != "mask"):
        raise ValueError("Require the 26-feature hourly mask O24 dataset, sequence 96, purge 24")

    frames = {name: read_frame(source_dir / f"{name}.csv")
              for name in ("hourly", "train", "val", "test")}
    expected = add_targets(frames["hourly"])
    for column in TARGETS:
        available = np.isfinite(frames["hourly"][column])
        np.testing.assert_allclose(frames["hourly"].loc[available, column],
                                   expected.loc[available, column], rtol=0, atol=1e-8)
    for name in ("train", "val", "test"):
        pd.testing.assert_frame_equal(frames[name][features + TARGETS],
                                      frames["hourly"].loc[frames[name].index, features + TARGETS])
    for left, right in (("train", "val"), ("val", "test")):
        if frames[left].index.max() + pd.Timedelta(hours=24) >= frames[right].index.min():
            raise ValueError("Source does not preserve the 24-hour target purge")

    complete = {name: np.isfinite(frame[TARGETS].to_numpy()).all(axis=1)
                for name, frame in frames.items()}
    projected = {}
    for name, frame in frames.items():
        projected[name] = frame[features + SELECTED_TARGETS].copy()
        projected[name].loc[~complete[name], SELECTED_TARGETS] = np.nan

    parent_train = make_dataset(frames["train"], features, TARGETS)
    train = make_dataset(projected["train"], features, SELECTED_TARGETS)
    np.testing.assert_array_equal(train.feature_scaler.mean_, parent_train.feature_scaler.mean_)
    np.testing.assert_array_equal(train.feature_scaler.scale_, parent_train.feature_scaler.scale_)
    selected_indices = [TARGETS.index(column) for column in SELECTED_TARGETS]
    np.testing.assert_array_equal(train.target_scaler.mean_, parent_train.target_scaler.mean_[selected_indices])
    np.testing.assert_array_equal(train.target_scaler.scale_, parent_train.target_scaler.scale_[selected_indices])
    coverage = {}
    for name, frame in frames.items():
        original = make_dataset(frame, features, TARGETS, parent_train)
        subset = make_dataset(projected[name], features, SELECTED_TARGETS, train)
        np.testing.assert_array_equal(subset.end_indices, original.end_indices)
        np.testing.assert_array_equal(subset.timestamps, original.timestamps)
        np.testing.assert_array_equal(subset.X, original.X)
        eligible = subset.end_indices
        coverage[name] = dict(
            rows=len(frame), complete_24h=int(complete[name].sum()),
            masked_target_rows=int((~complete[name]).sum()), sequences_96=len(subset),
            start_utc=str(frame.index.min()), end_utc=str(frame.index.max()),
            first_anchor_utc=str(frame.index[eligible[0]]) if len(eligible) else None,
            last_anchor_utc=str(frame.index[eligible[-1]]) if len(eligible) else None,
        )

    level_index = features.index("water_level_depth")
    persistence_scale = train.feature_scaler.scale_[level_index] / train.target_scaler.scale_
    persistence_offset = ((train.feature_scaler.mean_[level_index] - train.target_scaler.mean_)
                          / train.target_scaler.scale_)
    result = {
        **spec, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "O24_O2_h6_h12", "parent_dataset": str(source_dir),
        "target_columns": SELECTED_TARGETS, "common_target_columns": SELECTED_TARGETS,
        "output_size": 2, "input_size": 26, "splits": coverage,
        "target_definition": "Observed H(t+6h), H(t+12h); exact timestamp lookup; no interpolation",
        "matching_policy": "Require all 24 parent labels; mask both selected labels otherwise; retain every input row",
        "preprocessing_file": "preprocessing.joblib", "scaler_fit_split": "train",
        "persistence_scale": persistence_scale.tolist(),
        "persistence_offset": persistence_offset.tolist(),
    }
    preprocessing = dict(
        feature_scaler=train.feature_scaler, target_scaler=train.target_scaler,
        feature_columns=features, target_columns=SELECTED_TARGETS,
        sequence_length=96, resolution_minutes=60, missing_scaled_value=0.0,
        raw_missing_value=None, rain_station_ids=spec["rain_station_ids"],
        forecast_history_policy=spec.get("forecast_history_policy", "all_timesteps"),
        persistence_scale=persistence_scale, persistence_offset=persistence_offset,
    )

    output_dir.mkdir(parents=True)
    for name in frames:
        project_csv(source_dir / f"{name}.csv", output_dir / f"{name}.csv",
                    features + SELECTED_TARGETS, complete[name])
        restored = read_frame(output_dir / f"{name}.csv")
        pd.testing.assert_frame_equal(restored[features], frames[name][features])
        np.testing.assert_allclose(restored[SELECTED_TARGETS], projected[name][SELECTED_TARGETS],
                                   rtol=0, atol=0, equal_nan=True)
    provenance = source_dir / "weather_provenance.csv.gz"
    if provenance.exists():
        shutil.copyfile(provenance, output_dir / provenance.name)
    joblib.dump(preprocessing, output_dir / "preprocessing.joblib")
    (output_dir / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output_dir": str(output_dir), "splits": coverage}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.source_dir, args.output_dir)
