"""Remove only forecast rain at +12..24h from the paired +6/+12h dataset."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.ai.build_horizon_subset_dataset import (
    DEFAULT_OUTPUT as DEFAULT_SOURCE, SELECTED_TARGETS, make_dataset, project_csv,
)
from src.ai.hourly_dataset import ROOT, read_frame
from src.ai.rain_stations import available_rain_ids, combination_name, resolve_rain_stations, select_rain_columns


DEFAULT_OUTPUT = ROOT / "train_data/da_vien_1h_2026-10-15_forecast_mask_o2_h6_h12_weather12"
REMOVED_FEATURES = ("weather_forecast_rain_12_24h", "weather_forecast_rain_12_24h_missing")
DEFAULT_RAIN_POOL = ROOT / "train_data/da_vien_1h_2026-10-15_expanded_train_o24"


def select_scaler_columns(scaler, indices):
    """Project an already-fitted scaler, including per-column missing counts."""
    result = deepcopy(scaler)
    for attribute in ("mean_", "var_", "scale_", "feature_names_in_"):
        values = getattr(result, attribute, None)
        if values is not None:
            setattr(result, attribute, values[indices].copy())
    if np.ndim(result.n_samples_seen_) == 1:
        result.n_samples_seen_ = result.n_samples_seen_[indices].copy()
    result.n_features_in_ = len(indices)
    return result


def build(source_dir=DEFAULT_SOURCE, output_dir=DEFAULT_OUTPUT, rain_stations=None,
          rain_pool_dir=DEFAULT_RAIN_POOL, forecast_hours=12):
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refuse to overwrite: {output_dir}")
    if forecast_hours not in (12, 24):
        raise ValueError("Forecast feature horizon must be 12 or 24 hours")
    spec = json.loads((source_dir / "manifest.json").read_text())
    original_features = spec["feature_columns"]
    if (len(original_features) != 26 or len(set(original_features)) != 26
            or not set(REMOVED_FEATURES).issubset(original_features)
            or spec["target_columns"] != SELECTED_TARGETS
            or spec.get("sequence_length") != 96 or spec.get("purge_hours") != 24
            or spec.get("resolution_minutes") != 60 or spec.get("variant") != "mask"
            or spec.get("sample_stride", 1) != 1):
        raise ValueError("Require the paired 26-input, two-output hourly mask dataset")
    saved = joblib.load(source_dir / spec.get("preprocessing_file", "preprocessing.joblib"))
    if (saved["feature_columns"] != original_features
            or saved["target_columns"] != SELECTED_TARGETS
            or saved["sequence_length"] != 96
            or saved["feature_scaler"].n_features_in_ != 26
            or saved["target_scaler"].n_features_in_ != 2):
        raise ValueError("Source preprocessing does not match its dataset contract")

    original_frames = {name: read_frame(source_dir / f"{name}.csv")
                       for name in ("hourly", "train", "val", "test")}
    frames = {name: frame.copy() for name, frame in original_frames.items()}
    selected_features = original_features.copy()
    station_ids = spec["rain_station_ids"]
    added_columns = []
    extra_text = None
    if rain_stations is not None:
        rain_pool_dir = Path(rain_pool_dir).resolve()
        pool_spec = json.loads((rain_pool_dir / "manifest.json").read_text())
        pool = read_frame(rain_pool_dir / "hourly.csv")
        station_ids = resolve_rain_stations(
            rain_stations, available_rain_ids(pool.columns), pool_spec.get("rain_station_aliases"),
        )
        # Use only matching raw hourly predictors, never the pool's expanded splits.
        pd.testing.assert_frame_equal(pool.loc[frames["hourly"].index, original_features],
                                      frames["hourly"][original_features])
        for target in SELECTED_TARGETS:
            observed = frames["hourly"][target].notna()
            np.testing.assert_allclose(frames["hourly"].loc[observed, target],
                                       pool.loc[frames["hourly"].index[observed], target],
                                       rtol=0, atol=1e-8)
        pool_features = select_rain_columns(pool_spec["feature_columns"], station_ids)
        selected_features = select_rain_columns(original_features + [
            c for c in pool_features if c.startswith("rain_") and c not in original_features
        ], station_ids)
        added_columns = [c for c in selected_features if c not in original_features]
        for name, frame in frames.items():
            frames[name] = frame.join(pool.loc[frame.index, added_columns])
        if added_columns:
            extra_text = pd.read_csv(rain_pool_dir / "hourly.csv", dtype=str,
                                     keep_default_na=False, index_col="timestamp_utc")
            extra_text.index = pd.to_datetime(extra_text.index, utc=True)
    removed = REMOVED_FEATURES if forecast_hours == 12 else ()
    features = [column for column in selected_features if column not in removed]

    preprocessing = deepcopy(saved)
    if added_columns:
        fs = StandardScaler().fit(frames["train"][features].to_numpy(dtype=np.float32))
        fs.hydrology_raw_missing_value_ = saved["feature_scaler"].hydrology_raw_missing_value_
        fs.hydrology_normalized_missing_value_ = saved["feature_scaler"].hydrology_normalized_missing_value_
        preprocessing["feature_scaler"] = fs
    else:
        indices = [original_features.index(column) for column in features]
        preprocessing["feature_scaler"] = select_scaler_columns(saved["feature_scaler"], indices)
    preprocessing["feature_columns"] = features
    preprocessing["input_size"] = len(features)
    preprocessing["rain_station_ids"] = station_ids
    preprocessing["rain_station_combination"] = combination_name(station_ids)
    preprocessing["output_size"] = 2
    preprocessing["sample_stride"] = 1
    preprocessing["feature_indices"] = {column: index for index, column in enumerate(features)}
    weather = [i for i, column in enumerate(features) if column.startswith("weather_forecast_")]
    masks = [i for i in weather if features[i].endswith("_missing")]
    fs, ts = preprocessing["feature_scaler"], preprocessing["target_scaler"]
    level_index = features.index("water_level_depth")
    preprocessing.update(
        level_feature_index=level_index,
        weather_feature_indices=weather, weather_mask_indices=masks,
        weather_mask_values=((1. - fs.mean_[masks]) / fs.scale_[masks]).tolist(),
        persistence_scale=fs.scale_[level_index] / ts.scale_,
        persistence_offset=(fs.mean_[level_index] - ts.mean_) / ts.scale_,
    )
    parent_scalers = SimpleNamespace(feature_scaler=saved["feature_scaler"],
                                     target_scaler=saved["target_scaler"])
    scalers = SimpleNamespace(feature_scaler=fs, target_scaler=ts)
    for left, right in (("train", "val"), ("val", "test")):
        if frames[left].index.max() + pd.Timedelta(hours=24) >= frames[right].index.min():
            raise ValueError("Source does not preserve the 24-hour target purge")
    coverage = {}
    for name, frame in frames.items():
        if list(original_frames[name].columns) != original_features + SELECTED_TARGETS:
            raise ValueError(f"Unexpected input or target columns in {name}")
        if name != "hourly":
            pd.testing.assert_frame_equal(frame, frames["hourly"].loc[frame.index])
        old = make_dataset(original_frames[name], original_features, SELECTED_TARGETS, parent_scalers)
        new = make_dataset(frame[features + SELECTED_TARGETS], features, SELECTED_TARGETS, scalers)
        np.testing.assert_array_equal(new.end_indices, old.end_indices)
        np.testing.assert_array_equal(new.timestamps, old.timestamps)
        retained = [c for c in features if c in original_features]
        np.testing.assert_allclose(new.X[:, [features.index(c) for c in retained]],
                                   old.X[:, [original_features.index(c) for c in retained]],
                                   rtol=0, atol=1e-6)
        np.testing.assert_array_equal(new.y, old.y)
        coverage[name] = {
            **spec.get("splits", {}).get(name, {}),
            "rows": len(frame), "sequences_96": len(new),
            "start_utc": str(frame.index.min()), "end_utc": str(frame.index.max()),
            "first_anchor_utc": str(frame.index[new.end_indices[0]]) if len(new) else None,
            "last_anchor_utc": str(frame.index[new.end_indices[-1]]) if len(new) else None,
        }
    result = {
        **spec, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": f"O2_h6_h12_weather{forecast_hours}_ablation", "parent_dataset": str(source_dir),
        "feature_columns": features, "input_size": len(features), "output_size": 2,
        "rain_station_ids": station_ids,
        "prediction_horizons": [6, 12], "sample_stride": 1,
        "removed_feature_columns": list(removed), "splits": coverage,
        "feature_indices": preprocessing["feature_indices"],
        "weather_feature_indices": weather, "weather_mask_indices": masks,
        "weather_mask_values": preprocessing["weather_mask_values"],
        "weather_windows_hours": [[0, 3], [3, 6], [6, 12]] + ([[12, 24]] if forecast_hours == 24 else []),
        "forecast_feature_horizon_hours": forecast_hours,
        "scaler_policy": ("Fit added rain inputs using original train split only; reuse target scaler"
                          if added_columns else "Project parent train-fitted input scaler; reuse target scaler; no refit"),
        "forecast_history_policy": saved.get("forecast_history_policy", "all_timesteps"),
        "persistence_scale": preprocessing["persistence_scale"].tolist(),
        "persistence_offset": preprocessing["persistence_offset"].tolist(),
    }
    if rain_stations is not None:
        result["experiment"] = f"O2_h6_h12_weather{forecast_hours}_rain_combinations"
        result["rain_pool_source"] = str(rain_pool_dir)
        result["rain_station_aliases"] = pool_spec.get("rain_station_aliases", {})
        result["rain_station_combination"] = combination_name(station_ids)
    output_dir.mkdir(parents=True)
    for name, frame in frames.items():
        # Keep all existing target masks; do not recompute weather availability.
        additional = (None if extra_text is None else {
            c: extra_text.loc[frame.index, c].to_numpy() for c in added_columns
        })
        project_csv(source_dir / f"{name}.csv", output_dir / f"{name}.csv",
                    features + SELECTED_TARGETS, np.ones(len(frame), dtype=bool), additional)
        pd.testing.assert_frame_equal(read_frame(output_dir / f"{name}.csv"),
                                      frame[features + SELECTED_TARGETS])
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
    parser.add_argument("--rain-stations", help="Named combination or all; omitted preserves the original two stations")
    parser.add_argument("--rain-pool-dir", type=Path, default=DEFAULT_RAIN_POOL)
    parser.add_argument("--forecast-hours", type=int, choices=(12, 24), default=12)
    args = parser.parse_args()
    build(args.source_dir, args.output_dir, args.rain_stations, args.rain_pool_dir, args.forecast_hours)
