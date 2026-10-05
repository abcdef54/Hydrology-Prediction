"""Evaluate saved experiment-3 models on the same test rows for the report.

Reads a local MLflow snapshot and saved artifacts; never retrains or writes to MLflow.
Run from the repository root with .venv/bin/python scripts/export_model_training_results.py.
"""

import json
import hashlib
import sys
from pathlib import Path

import joblib
import mlflow.lightgbm
import mlflow.pytorch
import mlflow.xgboost
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import HydrologyDataset
from src.metrics import evaluate_all
from src.utils import select_feature_columns

OUTPUT_DIR = PROJECT_ROOT / "reports/model_training"
FEATURE_SETS = ("water_level", "water_level+rain", "all")
METHODS = ("lgbm", "xgb", "lstm", "residual_lstm", "lstm_adapter")


def unpack_run(run: dict) -> dict:
    return {
        "run_id": run["info"]["run_id"],
        "status": run["info"]["status"],
        **{
            field: {entry["key"]: entry["value"] for entry in run["data"].get(field, [])}
            for field in ("params", "metrics", "tags")
        },
    }


def model_directory(uri: str, run_id: str) -> Path:
    directory = PROJECT_ROOT / "mlartifacts/3/models" / uri.split("/")[-1] / "artifacts"
    metadata = yaml.safe_load((directory / "MLmodel").read_text())
    if metadata["run_id"] != run_id:
        raise ValueError(f"Model artifact does not belong to run {run_id}")
    return directory


def sequence_predictions(run: dict, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    directory = model_directory(run["tags"]["model_uri/best_model"], run["run_id"])
    preprocessing = joblib.load(directory / "extra_files/preprocessing.joblib")
    feature_columns = preprocessing["feature_columns"]
    dataset = HydrologyDataset(
        test, feature_cols=feature_columns, target_cols=preprocessing["target_columns"],
        feature_scaler=preprocessing["feature_scaler"],
        target_scaler=preprocessing["target_scaler"],
        seq_len=preprocessing["sequence_length"], require_current_level=True,
    )
    model = mlflow.pytorch.load_model(str(directory), map_location="cpu")
    model.eval()
    batches = []
    with torch.inference_mode():
        for features, _ in DataLoader(dataset, batch_size=128, shuffle=False):
            batches.append(model(features).numpy())
    predicted_scaled = np.concatenate(batches)
    predictions = dataset.target_scaler.inverse_transform(predicted_scaled)
    # Check the reconstructed evaluation against the metric recorded during training.
    reproduced_loss = float(np.mean((predicted_scaled - dataset.y[dataset.end_indices]) ** 2))
    np.testing.assert_allclose(reproduced_loss, run["metrics"]["test_loss"], rtol=1e-3, atol=1e-6)
    print(f"Verified {run['params']['train_method']} / {run['params']['feature_set']}: {reproduced_loss:.6f}")
    return dataset.end_indices, predictions


def tree_predictions(run: dict, test: pd.DataFrame, targets: list[str]) -> tuple[np.ndarray, np.ndarray]:
    feature_set = run["params"]["feature_set"]
    feature_columns = select_feature_columns(test.columns, feature_set)
    if feature_columns is None:
        feature_columns = [name for name in test if not name.startswith(("target_", "timestamp"))]
    load_model = mlflow.xgboost.load_model if run["params"]["train_method"] == "xgb" else mlflow.lightgbm.load_model
    predictions = []
    for target in targets:
        directory = model_directory(run["tags"][f"model_uri/{target}"], run["run_id"])
        model = load_model(str(directory))
        predictions.append(model.predict(test[feature_columns]))
    predictions = np.column_stack(predictions)
    reproduced_loss = float(np.mean((predictions - test[targets].to_numpy()) ** 2))
    np.testing.assert_allclose(reproduced_loss, run["metrics"]["test_loss"], rtol=1e-3, atol=1e-6)
    print(f"Verified {run['params']['train_method']} / {feature_set}: {reproduced_loss:.6f}")
    return np.arange(len(test)), predictions


def main() -> None:
    torch.set_num_threads(2)
    snapshot = json.loads((OUTPUT_DIR / "mlflow_experiment_3.json").read_text())
    runs = [unpack_run(run) for run in snapshot["runs"] if run["info"]["status"] == "FINISHED"]
    selected = []
    for feature_set in FEATURE_SETS:
        for method in METHODS:
            candidates = [run for run in runs if run["params"].get("train_method") == method
                          and run["params"].get("feature_set") == feature_set]
            if not candidates:
                raise ValueError(f"Missing run: {method} / {feature_set}")
            # Neural runs have validation loss; tree targets use their own early stopping.
            selected.append(min(candidates, key=lambda run: run["metrics"].get("best_val_loss", float("inf"))))
    (OUTPUT_DIR / "selected_runs.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2))

    test = pd.read_csv(PROJECT_ROOT / "train_data/1h/test.csv")
    targets = [name for name in test if name.startswith("target_")]
    evaluations = []
    common_indices = np.arange(len(test))
    for run in selected:
        if run["params"]["train_method"] in ("xgb", "lgbm"):
            indices, predictions = tree_predictions(run, test, targets)
        else:
            indices, predictions = sequence_predictions(run, test)
        common_indices = np.intersect1d(common_indices, indices)
        evaluations.append((run, indices, predictions))

    # Persistence uses the exact same forecast origins as every learned model.
    observed = test.iloc[common_indices][targets].to_numpy()
    timestamps = test.iloc[common_indices]["timestamp_utc"].reset_index(drop=True)
    persistence = np.repeat(test.iloc[common_indices][["water_level_depth"]].to_numpy(), len(targets), axis=1)
    rows = []
    prediction_rows = []
    for run, indices, predictions in evaluations:
        positions = np.searchsorted(indices, common_indices)
        predictions = predictions[positions]
        feature_set = run["params"]["feature_set"]
        method = run["params"]["train_method"]
        for target_index, target in enumerate(targets):
            metrics = evaluate_all(observed[:, target_index], predictions[:, target_index], timestamps)
            metrics["mse"] = float(np.mean((observed[:, target_index] - predictions[:, target_index]) ** 2))
            rows.append({"feature_set": feature_set, "model": method, "target": target,
                         "run_id": run["run_id"], "n_test": len(common_indices), **metrics})
            prediction_rows.append(pd.DataFrame({
                "feature_set": feature_set, "model": method, "target": target,
                "timestamp_utc": timestamps, "observed": observed[:, target_index],
                "predicted": predictions[:, target_index],
            }))
    for feature_set in FEATURE_SETS:
        for target_index, target in enumerate(targets):
            metrics = evaluate_all(observed[:, target_index], persistence[:, target_index], timestamps)
            metrics["mse"] = float(np.mean((observed[:, target_index] - persistence[:, target_index]) ** 2))
            rows.append({"feature_set": feature_set, "model": "persistence", "target": target,
                         "run_id": "recomputed_baseline", "n_test": len(common_indices), **metrics})
    detailed = pd.DataFrame(rows)
    detailed.to_csv(OUTPUT_DIR / "test_metrics_common_rows.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(OUTPUT_DIR / "test_predictions_common_rows.csv", index=False)
    summary = detailed.groupby(["feature_set", "model"], sort=False)[["mse", "mae", "nse", "kge", "flood_f1", "flood_precision", "flood_recall"]].mean().reset_index()
    summary["rmse"] = np.sqrt(summary["mse"])
    summary.to_csv(OUTPUT_DIR / "test_metrics_summary.csv", index=False)
    (OUTPUT_DIR / "evaluation_notes.json").write_text(json.dumps({
        "n_test": len(common_indices), "first_origin_utc": timestamps.iloc[0],
        "last_origin_utc": timestamps.iloc[-1], "targets": targets,
        "test_row_indices": common_indices.tolist(),
        "selection": "lowest logged validation loss within each method/feature set; one tree run per pair",
        "regression_units": "original dataset units, physical unit and station datum unconfirmed",
        "flood_metrics": "90th percentile of observed test levels for each horizon; not official flood alarms",
        "aggregation": "mean MSE/MAE/NSE/KGE across five horizons; overall RMSE = sqrt(mean MSE)",
        "input_sha256": {
            str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [OUTPUT_DIR / "mlflow_experiment_3.json"] + [
                PROJECT_ROOT / f"train_data/1h/{split}.csv" for split in ("train", "val", "test")
            ]
        },
    }, ensure_ascii=False, indent=2))
    print(summary[["feature_set", "model", "rmse", "mse", "nse", "kge"]].to_string(index=False))
    print(f"Evaluated all models on {len(common_indices)} common test origins.")


if __name__ == "__main__":
    main()
