"""Configurable experiments on expanded train; defaults reproduce B's settings.

Keep the same hyperparameters across arms to isolate the training sampler.
Former test is validation, never independent test. Production is not modified.
"""

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:2")

import joblib
import mlflow
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from research.build_expanded_train_dataset import DEFAULT_OUTPUT
from research.build_output24_dataset import COMMON, read_frame
from research.expanded_train_report import write_comparison_report
from src.ai.comparison import evaluation_dataset, load_candidate, predict
from src.ai.dataset import HydrologyDataset
from src.ai.losses import HorizonWeightedTailLoss
from src.ai.lstm import ResidualLSTM
from src.ai.rain_stations import (
    available_rain_ids, combination_name, resolve_rain_stations, select_rain_columns,
)
from src.ai.sampling import EventBalancedSampler
from src.ai.trainers import ResidualLSTMTrainer
from src.ai.utils import seed_worker, set_seed
from src.backend.prediction.config import StationModelRegistry
from src.backend.prediction.inference import ResidualLSTMRuntime


ROOT = Path(__file__).resolve().parents[1]


def prepare_data(directory, rain_stations="hong_thai+nham", sequence_length=96):
    if sequence_length < 1:
        raise ValueError("Sequence length must be positive")
    directory = Path(directory)
    spec = json.loads((directory / "manifest.json").read_text())
    if spec.get("experiment") != "expanded_train" or spec.get("has_independent_test") is not False:
        raise ValueError("Build the expanded-train dataset first")
    if (directory / "test.csv").exists():
        raise ValueError("Expanded-train experiments must not contain a pretend test split")
    frames = {name: read_frame(directory / f"{name}.csv").reset_index() for name in ("train", "val")}
    last_train = pd.Timestamp(frames["train"].timestamp_utc.iloc[-1])
    first_val = pd.Timestamp(frames["val"].timestamp_utc.iloc[0])
    if last_train + pd.Timedelta(hours=24) >= first_val:
        raise ValueError("Training targets overlap new validation anchors")
    ids = resolve_rain_stations(
        rain_stations, available_rain_ids(spec["feature_columns"]), spec.get("rain_station_aliases"),
    )
    features = select_rain_columns(spec["feature_columns"], ids)
    if any(column.startswith(("target_", "timestamp")) for column in features):
        raise ValueError("Targets and timestamps cannot be model predictors")
    train = HydrologyDataset(frames["train"], feature_cols=features, target_cols=COMMON,
                             fit_scalers=True, seq_len=sequence_length, require_current_level=True)
    validation = HydrologyDataset(
        frames["val"], feature_cols=features, target_cols=COMMON, seq_len=sequence_length,
        feature_scaler=train.feature_scaler, target_scaler=train.target_scaler,
        require_current_level=True,
    )
    for dataset in (train, validation):
        if not len(dataset):
            raise ValueError("Train and validation must both have eligible sequences")
        dataset.feature_set = "water_level+rain+flood_season+weather_forecast"
        dataset.rain_station_ids = ids
        dataset.rain_station_combination = combination_name(ids)
        dataset.resolution_minutes = 60
    return spec, frames, train, validation


def build_trainer(args, prepared):
    _, frames, train, validation = prepared
    if train.seq_len != args.sequence_length:
        raise ValueError("Dataset sequence length does not match CLI settings")
    generator = torch.Generator().manual_seed(args.seed)
    sampler = None
    if args.variant == "event-balanced":
        sampler = EventBalancedSampler(
            frames["train"], train, seed=args.seed,
            rise_fraction=args.rise_fraction, fall_fraction=args.fall_fraction,
            threshold_mm=args.change_threshold_mm, event_gap_hours=args.event_gap_hours,
        )
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=sampler is None,
                              sampler=sampler, generator=generator, num_workers=args.num_worker,
                              worker_init_fn=seed_worker)
    val_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False,
                           generator=torch.Generator().manual_seed(args.seed + 1),
                           num_workers=args.num_worker, worker_init_fn=seed_worker)
    fs, ts = train.feature_scaler, train.target_scaler
    index = train.feature_cols.index("water_level_depth")
    model = ResidualLSTM(
        input_size=train.num_features, output_size=5, hidden_size=args.hidden_size,
        num_layers=args.num_layers, dropout=args.dropout, input_dropout=args.input_dropout,
        head_dropout=args.head_dropout, level_feature_index=index,
        persistence_scale=(fs.scale_[index] / ts.scale_).tolist(),
        persistence_offset=((fs.mean_[index] - ts.mean_) / ts.scale_).tolist(),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = (torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma,
    ) if args.scheduler == "step" else None)
    loss = HorizonWeightedTailLoss(args.horizon_weights, ts.scale_, strength=args.tail_penalty,
                                   threshold_mm=args.tail_threshold_mm, transition_mm=args.tail_transition_mm)
    trainer = ResidualLSTMTrainer(
        model=model, correction_penalty=args.correction_penalty, horizon="1h", num_epochs=args.num_epochs,
        train_dataloader=train_loader, eval_dataloader=val_loader, test_dataloader=None,
        loss_func=loss, optimizer=optimizer, scheduler=scheduler,
        early_stopping=args.early_stopping, device=args.device,
        gradient_accumulation=args.gradient_accumulation, gradient_clip=args.gradient_clip,
        checkpoint_dir=args.output_dir or Path("/tmp/unused-expanded-train-check"),
        checkpoint_metric=args.checkpoint_metric, validation_threshold_mm=200.,
    )
    return trainer, sampler


def prediction_metrics(frame, indices, values, prefix):
    metrics = {}
    current = frame.water_level_depth.to_numpy()[indices]
    for j, hours in enumerate((1, 3, 6, 12, 24)):
        actual = frame[COMMON[j]].to_numpy()[indices]
        error = values[:, j] - actual
        delta = values[:, j] - current
        true_delta = actual - current
        metrics.update({
            f"{prefix}/h{hours}/rmse_mm": float(np.sqrt(np.mean(error ** 2))),
            f"{prefix}/h{hours}/n200": int((np.abs(error) > 200).sum()),
            f"{prefix}/h{hours}/mean_abs_change_mm": float(np.abs(delta).mean()),
        })
        fast = np.abs(true_delta) > 200
        metrics[f"{prefix}/h{hours}/fast_n"] = int(fast.sum())
        if fast.any():
            metrics[f"{prefix}/h{hours}/fast_pred_abs_mm"] = float(np.abs(delta[fast]).mean())
            metrics[f"{prefix}/h{hours}/fast_actual_abs_mm"] = float(np.abs(true_delta[fast]).mean())
    errors = values - frame[COMMON].to_numpy()[indices]
    metrics[f"{prefix}/rmse_mm"] = float(np.sqrt(np.mean(errors ** 2)))
    metrics[f"{prefix}/n200"] = int((np.abs(errors) > 200).sum())
    metrics[f"{prefix}/anchors"] = len(indices)
    return metrics


def compare_references(args, prepared, trainer, report_dir=None, metadata=None):
    spec, frames, _, _ = prepared
    validation = trainer.eval_dataloader.dataset
    registry = StationModelRegistry.from_json(
        ROOT / "src/backend/prediction/model_configs.json", artifact_root=ROOT,
    )
    reference_frame = pd.read_csv(args.dataset_dir / spec["reference_validation_csv"])
    np.testing.assert_array_equal(pd.to_datetime(reference_frame.timestamp_utc, utc=True),
                                  pd.to_datetime(frames["val"].timestamp_utc, utc=True))
    references = {}
    identities = {
        "candidate": {"checkpoint": str(trainer.checkpoint_path),
                      "preprocessing": str(trainer.checkpoint_path.with_suffix(".preprocessing.joblib")),
                      "feature_columns": validation.feature_cols},
    }
    for model_id in ("legacy", "model_b"):
        runtime = ResidualLSTMRuntime(registry.get("0074647403", model_id), device=args.device)
        saved = joblib.load(runtime.config.preprocessing_path)
        raw = reference_frame if model_id == "legacy" else frames["val"]
        np.testing.assert_allclose(raw[COMMON], frames["val"][COMMON], atol=1e-8, rtol=0, equal_nan=True)
        dataset = evaluation_dataset(raw, saved)
        references[model_id] = (runtime.model, dataset)
        identities[model_id] = {
            "model_version": runtime.config.model_version,
            "checkpoint": str(runtime.config.model_path),
            "preprocessing": str(runtime.config.preprocessing_path),
            "feature_columns": saved["feature_columns"],
        }
    common = np.asarray(sorted(set(validation.end_indices).intersection(
        *(set(dataset.end_indices) for _, dataset in references.values()),
    )))
    if not len(common):
        raise ValueError("No common validation anchors for frozen references")
    metrics = {}
    points = []
    for name, (model, dataset) in {"candidate": (trainer.model, validation), **references}.items():
        values = predict(model, dataset, common, args.device, args.batch_size)
        metrics.update(prediction_metrics(frames["val"], common, values, f"validation_comparison/{name}"))
        if report_dir is not None:
            anchors = pd.to_datetime(frames["val"].timestamp_utc.iloc[common], utc=True).reset_index(drop=True)
            current = frames["val"].water_level_depth.to_numpy()[common]
            for j, hours in enumerate((1, 3, 6, 12, 24)):
                points.append(pd.DataFrame({
                    "model": name, "anchor_utc": anchors,
                    "target_utc": anchors + pd.Timedelta(hours=hours), "horizon_hours": hours,
                    "current_mm": current, "actual_mm": frames["val"][COMMON[j]].to_numpy()[common],
                    "predicted_mm": values[:, j],
                }))
    if report_dir is not None:
        details = {
            **(metadata or {}), "variant": args.variant, "seed": args.seed,
            "run_name": getattr(args, "run_name", None),
            "dataset_dir": str(args.dataset_dir.resolve()), "checkpoint_path": str(trainer.checkpoint_path.resolve()),
            "device": args.device, "models": identities,
        }
        report = write_comparison_report(report_dir, pd.concat(points, ignore_index=True), details)
        print(f"Comparison report: {report}", flush=True)
    return metrics


def run_training(args, prepared, trainer, sampler, log_metrics):
    """Use the shared epoch/objective/checkpoint methods without its test entrypoint."""
    trainer.checkpoint_path.parent.mkdir(parents=True)
    trainer.prepare_checkpoint()
    for epoch in range(1, args.num_epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch - 1)
        training_loss = trainer.train_epoch(epoch)
        val_loss = trainer.evaluate_loss(trainer.eval_dataloader)
        metrics = {"train_loss": training_loss, "val_loss": val_loss, **trainer.last_validation_metrics}
        log_metrics(metrics, step=epoch)
        trainer.record_validation(epoch, val_loss)
        print(f"Epoch {epoch}: {json.dumps(metrics)}", flush=True)
        if args.early_stopping and trainer.epochs_without_improvement >= args.early_stopping:
            break
    trainer.restore_best_checkpoint(purpose="validation comparison, not independent testing")
    trainer.save_preprocessing()
    metrics = compare_references(args, prepared, trainer, report_dir=getattr(args, "report_dir", None), metadata={
        "best_checkpoint_epoch": trainer.best_checkpoint_epoch, "epochs_completed": epoch,
        "training_settings": getattr(args, "run_settings", {}),
    })
    metrics.update({"epochs_completed": epoch, "best_checkpoint_epoch": trainer.best_checkpoint_epoch})
    log_metrics(metrics)
    print(json.dumps(metrics, indent=2))
    return metrics


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset-dir", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--variant", choices=("uniform", "event-balanced"), required=True)
    result.add_argument("--rain-stations", default="hong_thai+nham")
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    result.add_argument("--num-epochs", type=int, default=50)
    result.add_argument("--early-stopping", type=int, default=25)
    result.add_argument("--checkpoint-metric", choices=("large_error_rate", "rmse", "val_loss"),
                        default="large_error_rate", help="Checkpoint and patience criterion; RMSE uses pooled physical mm")
    result.add_argument("--batch-size", type=int, default=128)
    result.add_argument("--num-worker", type=int, default=0)
    result.add_argument("--sequence-length", "--lstm-seq-len", type=int, default=96)
    result.add_argument("--hidden-size", "--lstm-hidden-size", type=int, default=32)
    result.add_argument("--num-layers", "--lstm-num-layers", type=int, default=3)
    result.add_argument("--dropout", "--lstm-drop-out", type=float, default=.7)
    result.add_argument("--input-dropout", "--lstm-input-dropout", type=float, default=.2)
    result.add_argument("--head-dropout", "--lstm-head-dropout", type=float, default=.2)
    result.add_argument("--learning-rate", "--lstm-lr", type=float, default=.0009)
    result.add_argument("--weight-decay", "--lstm-weight-decay", type=float, default=.001)
    result.add_argument("--correction-penalty", "--lstm-correction-penalty", type=float, default=.1)
    result.add_argument("--horizon-weights", "--lstm-horizon-weights", type=float, nargs=5, default=[1.] * 5)
    result.add_argument("--tail-penalty", "--lstm-tail-penalty", type=float, default=.5)
    result.add_argument("--tail-threshold-mm", "--lstm-tail-threshold-mm", type=float, default=200.)
    result.add_argument("--tail-transition-mm", "--lstm-tail-transition-mm", type=float, default=20.)
    result.add_argument("--gradient-accumulation", type=int, default=1)
    result.add_argument("--gradient-clip", type=float, default=1.)
    result.add_argument("--scheduler", choices=("step", "none"), default="step")
    result.add_argument("--scheduler-step-size", type=int, default=10)
    result.add_argument("--scheduler-gamma", type=float, default=.9)
    result.add_argument("--rise-fraction", type=float, default=.1)
    result.add_argument("--fall-fraction", type=float, default=.1)
    result.add_argument("--change-threshold-mm", type=float, default=200.)
    result.add_argument("--event-gap-hours", type=float, default=12.)
    result.add_argument("--output-dir", type=Path, help="Checkpoint base folder; each invocation gets a unique run subfolder")
    result.add_argument("--check-only", action="store_true")
    result.add_argument("--report-only", action="store_true", help="Compare an existing checkpoint without training")
    result.add_argument("--checkpoint", type=Path, help="Existing best checkpoint for --report-only")
    result.add_argument("--best-epoch", type=int, help="Recorded best epoch for an existing run; never inferred from scores")
    result.add_argument("--report-dir", type=Path, help="Report base folder; each invocation gets a unique run subfolder")
    result.add_argument("--no-mlflow", action="store_true")
    result.add_argument("--mlflow-ex-name", default="da_vien_expanded_train")
    result.add_argument("--mlflow-tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    return result


def validate_args(args):
    for name in ("num_epochs", "batch_size", "sequence_length", "hidden_size", "num_layers",
                 "gradient_accumulation", "scheduler_step_size"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.early_stopping < 0 or args.num_worker < 0:
        raise ValueError("Patience and worker count must be nonnegative")
    for name in ("learning_rate", "tail_threshold_mm", "tail_transition_mm", "gradient_clip", "scheduler_gamma"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    for name in ("weight_decay", "tail_penalty"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and nonnegative")
    for name in ("dropout", "input_dropout", "head_dropout"):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1)")
    if not math.isfinite(args.correction_penalty):
        raise ValueError("Correction penalty must be finite")
    if len(args.horizon_weights) != 5 or not all(math.isfinite(x) and x > 0 for x in args.horizon_weights):
        raise ValueError("Provide five finite positive horizon weights")


def main():
    arguments = parser()
    args = arguments.parse_args()
    try:
        validate_args(args)
    except ValueError as error:
        arguments.error(str(error))
    if args.report_only and (args.check_only or args.checkpoint is None):
        arguments.error("--report-only requires --checkpoint and cannot be combined with --check-only")
    if args.best_epoch is not None and args.best_epoch < 0:
        arguments.error("Best epoch must be nonnegative")
    if not args.check_only and args.device == "cuda" and not torch.cuda.is_available():
        arguments.error("CUDA unavailable; use --device cpu explicitly for a smoke test")
    set_seed(args.seed)
    torch.set_num_threads(2)
    prepared = prepare_data(args.dataset_dir, args.rain_stations, args.sequence_length)
    train, validation = prepared[2:]
    configure_run_directories(args, train.rain_station_combination)
    if args.report_only:
        saved = joblib.load(args.checkpoint.with_suffix(".preprocessing.joblib"))
        if saved["feature_columns"] != train.feature_cols:
            arguments.error("Selected rain stations do not match checkpoint preprocessing")
        model = load_candidate(args.checkpoint.read_bytes(), saved, args.device)
        candidate_data = evaluation_dataset(prepared[1]["val"], saved)
        trainer = SimpleNamespace(model=model, checkpoint_path=args.checkpoint,
                                  eval_dataloader=SimpleNamespace(dataset=candidate_data))
        metrics = compare_references(args, prepared, trainer, report_dir=args.report_dir,
                                     metadata={"best_checkpoint_epoch": args.best_epoch,
                                               "report_origin": "existing checkpoint; no training"})
        print(json.dumps(metrics, indent=2))
        return
    # Check-only always stays on CPU and creates no output directory or MLflow run.
    if args.check_only:
        args.device = "cpu"
    trainer, sampler = build_trainer(args, prepared)
    settings = {
        "run_name": args.run_name, "output_dir": str(args.output_dir), "report_dir": str(args.report_dir),
        "variant": args.variant, "seed": args.seed, "dataset_dir": str(args.dataset_dir.resolve()),
        "train_sequences": len(train), "validation_sequences": len(validation),
        "independent_test": False, "validation_origin": "former test; exploratory",
        "rain_station_ids": train.rain_station_ids, "input_size": train.num_features,
        "target_columns": COMMON, "sequence_length": args.sequence_length, "batch_size": args.batch_size,
        "num_worker": args.num_worker,
        "num_epochs": args.num_epochs, "early_stopping": args.early_stopping,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "hidden_size": args.hidden_size, "num_layers": args.num_layers,
        "dropout": args.dropout, "input_dropout": args.input_dropout,
        "head_dropout": args.head_dropout, "correction_penalty": args.correction_penalty,
        "tail_penalty": args.tail_penalty, "tail_threshold_mm": args.tail_threshold_mm,
        "tail_transition_mm": args.tail_transition_mm,
        "horizon_weights": args.horizon_weights, "checkpoint_metric": args.checkpoint_metric,
        "checkpoint_tiebreaker": "rmse" if args.checkpoint_metric == "large_error_rate" else "none",
        "device": args.device, "gradient_accumulation": args.gradient_accumulation,
        "gradient_clip": args.gradient_clip, "scheduler": args.scheduler,
        "scheduler_step_size": args.scheduler_step_size, "scheduler_gamma": args.scheduler_gamma,
        "initialization": "scratch, persistence-initialized head",
        "sampling": sampler.summary if sampler is not None else "uniform without replacement",
    }
    print(json.dumps(settings, ensure_ascii=False, indent=2))
    if args.check_only:
        return
    args.run_settings = settings
    if not args.no_mlflow:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(args.mlflow_ex_name)
    context = nullcontext() if args.no_mlflow else mlflow.start_run(
        run_name=f"expanded_{args.variant}_{train.rain_station_combination}_seed{args.seed}_{args.run_name}",
    )
    with context:
        if not args.no_mlflow:
            mlflow.log_params({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                               for key, value in settings.items()})
            mlflow.set_tags({"evaluation_scope": "validation_only", "test_available": "false"})
        logger = (lambda *a, **kw: None) if args.no_mlflow else mlflow.log_metrics
        run_training(args, prepared, trainer, sampler, logger)


def configure_run_directories(args, combination):
    """Existing base folders are reusable; checkpoint/report runs never overwrite."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    prefix = ("report" if args.report_only else
              f"seq{args.sequence_length}_h{args.hidden_size}_{args.checkpoint_metric}")
    args.run_name = f"{prefix}_{timestamp}_{uuid4().hex[:8]}"
    relative = Path(combination) / args.variant / f"seed_{args.seed}" / args.run_name
    checkpoint_root = args.output_dir or ROOT / "model/experiments/expanded_train"
    report_root = args.report_dir or ROOT / "reports/model_comparisons/expanded_train"
    args.output_dir = (checkpoint_root / relative).resolve()
    args.report_dir = (report_root / relative).resolve()
    for path in (args.output_dir, args.report_dir):
        if path.exists():
            raise FileExistsError(f"Refuse to overwrite generated run {path}")
    # Preview/check-only computes paths without reserving folders or touching data.
    checkpoint_display = args.checkpoint if args.report_only else args.output_dir
    print(f"Run: {args.run_name}\nCheckpoints: {checkpoint_display}\nReport: {args.report_dir}", flush=True)


if __name__ == "__main__":
    main()
