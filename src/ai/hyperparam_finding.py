"""Grid/TPE search using train/validation only, with iterative TPE pruning.

Run as python -m src.ai.hyperparam_finding.
"""

import argparse
from copy import copy
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import random

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:2")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/hydrology-matplotlib")

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.ai.dataset import HydrologyDataset
from src.ai.losses import CommonHorizonAuxiliaryLoss
from src.ai.tree_sequence import (
    TreeSequenceForecast,
    sequence_table,
    train_horizon,
)
from research.build_output24_dataset import (
    COMMON,
    TARGETS,
    assert_matched_population,
    read_frame,
)
from research.finetune_direction_experiment import SOURCE
from research.output_horizon_experiment import (
    evaluate,
    make_model,
    training_objective,
)

ROOT = Path(__file__).resolve().parents[2]
TREES = {"xgb", "lgbm"}
NEURALS = {"residual_lstm", "lstm"}
DEFAULT_TPE_TRIALS = 100
TREE_REFERENCE_LEARNING_RATE = 0.05
MIN_SEQUENCE_LENGTH = 12
MAX_SEQUENCE_LENGTH = 168

TREE_KEYS = {
    "sequence_length",
    "learning_rate",
    "max_depth",
    "num_leaves",
    "min_data_in_leaf",
    "min_child_weight",
    "reg_lambda",
    "reg_alpha",
    "subsample",
    "colsample_bytree",
}

NEURAL_KEYS = {
    "sequence_length",
    "lstm_hidden_size",
    "lstm_num_layers",
    "lstm_drop_out",
    "lstm_input_dropout",
    "lstm_head_dropout",
    "lstm_lr",
    "lstm_weight_decay",
    "batch_size",
    "lstm_correction_penalty",
    "lstm_tail_penalty",
    "lstm_tail_threshold_mm",
    "lstm_tail_transition_mm",
    "aux_weight",
}

INTEGER_KEYS = {
    "sequence_length",
    "max_depth",
    "num_leaves",
    "min_data_in_leaf",
    "lstm_hidden_size",
    "lstm_num_layers",
    "batch_size",
}


def objective_value(count, rmse, metric="large_error_rate"):
    """A whole error count always dominates the bounded RMSE tie breaker."""
    if (
        not np.isfinite(count)
        or count < 0
        or count != int(count)
        or not np.isfinite(rmse)
        or rmse < 0
    ):
        raise ValueError("Invalid physical metrics")

    if metric == "rmse":
        return float(rmse)

    return float(count) + 0.49 * (2 / math.pi) * math.atan(float(rmse) / 200)


def focus_metrics(metrics, horizons):
    count = sum(
        metrics[f"horizon/target_water_level_plus_{hour * 60}m/above_200"]
        for hour in horizons
    )
    rmse = math.sqrt(np.mean([
        metrics[f"horizon/target_water_level_plus_{hour * 60}m/rmse_mm"] ** 2
        for hour in horizons
    ]))

    return {"above_200": int(count), "rmse_mm": float(rmse)}


def model_search_space(method, search):
    """Broad initial search; custom JSON can override any default space."""
    if method in TREES:
        if search == "grid":
            return {
                "max_depth": [2, 4, 6, 8, 12, 16],
                "learning_rate": [0.001, 0.005, 0.01, 0.03, 0.1, 0.3],
                "reg_lambda": [0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            }

        space = {
            "max_depth": {"type": "int", "low": 2, "high": 16},
            "learning_rate": {
                "type": "float", "low": 0.001, "high": 0.3, "log": True,
            },
            "reg_lambda": {
                "type": "float", "low": 0.0001, "high": 1000.0, "log": True,
            },
            "reg_alpha": {
                "type": "float", "low": 1e-8, "high": 100.0, "log": True,
            },
            "subsample": {"type": "float", "low": 0.3, "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.3, "high": 1.0},
        }

        if method == "xgb":
            space["min_child_weight"] = {
                "type": "float", "low": 0.1, "high": 100.0, "log": True,
            }
        else:
            space["num_leaves"] = {
                "type": "int", "low": 8, "high": 256, "log": True,
            }
            space["min_data_in_leaf"] = {
                "type": "int", "low": 3, "high": 256, "log": True,
            }

        return space

    if search == "grid":
        return {
            "lstm_hidden_size": [8, 16, 32, 64, 128, 256],
            "lstm_lr": [1e-5, 1e-4, 0.0003, 0.001, 0.003, 0.01],
            "lstm_drop_out": [0.0, 0.2, 0.5, 0.7, 0.85],
        }

    return {
        "lstm_hidden_size": {"type": "int", "low": 8, "high": 256, "step": 8},
        "lstm_num_layers": {"type": "int", "low": 1, "high": 5},
        "lstm_lr": {"type": "float", "low": 1e-5, "high": 0.01, "log": True},
        "lstm_drop_out": {"type": "float", "low": 0.0, "high": 0.85},
        "lstm_input_dropout": {"type": "float", "low": 0.0, "high": 0.6},
        "lstm_head_dropout": {"type": "float", "low": 0.0, "high": 0.6},
        "lstm_weight_decay": {
            "type": "float", "low": 1e-8, "high": 0.1, "log": True,
        },
        "batch_size": [32, 64, 128, 256],
    }


def default_space(method, search):
    space = model_search_space(method, search)

    if search == "grid":
        space["sequence_length"] = [12, 24, 48, 72, 96, 120, 144, 168]
    else:
        space["sequence_length"] = {
            "type": "int",
            "low": MIN_SEQUENCE_LENGTH,
            "high": MAX_SEQUENCE_LENGTH,
        }

    return space


def resolve_search_space(args):
    """Explicit fixed length overrides defaults; a custom space is used as supplied."""
    space = (
        json.loads(args.search_space.read_text()) if args.search_space
        else default_space(args.train_method, args.search_method)
    )
    choices = args.sequence_lengths

    if choices is not None:
        if args.search_space and "sequence_length" in space:
            raise ValueError("Specify sequence lengths in the CLI or custom space, not both")
        space["sequence_length"] = choices
    elif args.sequence_length is not None:
        if args.search_space and "sequence_length" in space:
            raise ValueError("A fixed --sequence-length conflicts with the custom search space")
        space.pop("sequence_length", None)

    validate_space(space, args.train_method, args.search_method)
    sequence_spec = space.get("sequence_length")

    if isinstance(sequence_spec, list):
        cohort_length = max(sequence_spec)
    elif isinstance(sequence_spec, dict):
        cohort_length = sequence_spec["high"]
    else:
        cohort_length = args.sequence_length if args.sequence_length is not None else 96

    return space, cohort_length


def check_parameter(name, value, method):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number")

    if name in INTEGER_KEYS and (not isinstance(value, int) or value < 1):
        raise ValueError(f"{name} must be a positive integer")

    if name == "sequence_length" and not MIN_SEQUENCE_LENGTH <= value <= MAX_SEQUENCE_LENGTH:
        raise ValueError("Searchable sequence_length must be between 12 and 168 hours")

    if name == "num_leaves" and value < 2:
        raise ValueError("num_leaves must be >=2")

    positive = {
        "learning_rate", "lstm_lr", "lstm_tail_threshold_mm", "lstm_tail_transition_mm",
    }
    nonnegative = {
        "reg_lambda", "reg_alpha", "min_child_weight", "lstm_weight_decay",
        "lstm_tail_penalty", "aux_weight",
    }

    if name in positive and value <= 0:
        raise ValueError(f"{name} must be positive")

    if name in nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")

    if name in {"subsample", "colsample_bytree"} and not 0 < value <= 1:
        raise ValueError(f"{name} outside (0,1]")

    if name in {"lstm_drop_out", "lstm_input_dropout", "lstm_head_dropout"}:
        if not 0 <= value < 1:
            raise ValueError(f"{name} outside [0,1)")

    if name == "lstm_correction_penalty":
        if value <= -1:
            raise ValueError("Correction penalty must be >-1")
        if method == "lstm" and value != 0:
            raise ValueError("Direct model requires correction penalty 0")


def validate_space(space, method, search):
    if not isinstance(space, dict) or not space:
        raise ValueError("Search space must be a nonempty JSON object")

    if method in TREES:
        inactive = {"num_leaves", "min_data_in_leaf"} if method == "xgb" else {"min_child_weight"}
        allowed = TREE_KEYS - inactive
    else:
        allowed = NEURAL_KEYS

    for name, spec in space.items():
        if name not in allowed:
            raise ValueError(f"Unsupported parameter {name}")

        if isinstance(spec, list):
            if not spec:
                raise ValueError(f"{name}: nonempty unique list required")
            for value in spec:
                check_parameter(name, value, method)
            if len(set(spec)) != len(spec):
                raise ValueError(f"{name}: nonempty unique list required")
            continue

        if search != "tpe" or not isinstance(spec, dict):
            raise ValueError(f"{name}: grid requires a list; TPE accepts a list or range")

        if (
            spec.get("type") not in {"int", "float"}
            or not {"low", "high"}.issubset(spec)
            or set(spec) - {"type", "low", "high", "log", "step"}
        ):
            raise ValueError(f"{name}: expected type/int|float, low, high, optional log/step")

        low, high = spec["low"], spec["high"]
        check_parameter(name, low, method)
        check_parameter(name, high, method)

        if not isinstance(spec.get("log", False), bool):
            raise ValueError(f"{name}: log must be boolean")
        if low > high or (spec.get("log") and low <= 0):
            raise ValueError(f"{name}: invalid range")
        if name in INTEGER_KEYS and spec["type"] != "int":
            raise ValueError(f"{name} needs int distribution")
        if spec["type"] == "int" and (not isinstance(low, int) or not isinstance(high, int)):
            raise ValueError("Integer endpoints required")

        if "step" in spec:
            step = spec["step"]
            if (
                isinstance(step, bool)
                or not isinstance(step, (int, float))
                or not math.isfinite(step)
                or step <= 0
                or (spec["type"] == "int" and not isinstance(step, int))
            ):
                raise ValueError("Invalid step")
            if spec.get("log") and (spec["type"] == "float" or step != 1):
                raise ValueError("log distribution cannot use this step")


def suggest(trial, space):
    params = {}

    for name, spec in space.items():
        if isinstance(spec, list):
            params[name] = trial.suggest_categorical(name, spec)
        else:
            distribution = {key: value for key, value in spec.items() if key != "type"}
            sampler = trial.suggest_int if spec["type"] == "int" else trial.suggest_float
            params[name] = sampler(name, **distribution)

    return params


def boundary_messages(study, space):
    """Describe a best value/top-trial cluster near a bound; never expand silently."""
    completed = sorted(
        (trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE),
        key=lambda trial: trial.value,
    )
    if not completed:
        return []

    top = completed[:5]
    messages = []

    for name, spec in space.items():
        if not isinstance(spec, dict) or spec["low"] == spec["high"]:
            continue

        low, high = spec["low"], spec["high"]
        transform = math.log if spec.get("log") else float
        lower, upper = transform(low), transform(high)
        values = [trial.params[name] for trial in top]

        for side in ["lower", "upper"]:
            if spec["type"] == "int":
                near = [value == (low if side == "lower" else high) for value in values]
            else:
                positions = [(transform(value) - lower) / (upper - lower) for value in values]
                near = [position <= 0.05 if side == "lower" else position >= 0.95 for position in positions]

            clustered = len(top) >= 3 and sum(near) >= math.ceil(len(top) / 2)
            if near[0] or clustered:
                messages.append(
                    f"Search boundary: {name}, {sum(near)}/{len(top)} top trials near "
                    f"{side} bound {low if side == 'lower' else high}. "
                    "Consider a wider range in a NEW study; bounds are not proof of optimality."
                )

    return messages


def load_search_data(args):
    """Read only train/val. Test is deliberately absent during optimization."""
    spec = json.loads((args.dataset_dir / "manifest.json").read_text())
    if spec.get("variant", "mask") != "mask":
        raise ValueError("Use the mask forecast dataset")

    features = spec["feature_columns"]
    targets = COMMON if args.outputs == 5 else TARGETS
    frames = {}

    for split in ["train", "val"]:
        frame = read_frame(args.dataset_dir / f"{split}.csv")
        if not set(targets).issubset(frame):
            raise ValueError("Requested targets absent")
        if set(TARGETS).issubset(frame):
            assert_matched_population(frame[TARGETS], frame.water_level_depth, args.sequence_length)
        frames[split] = frame.reset_index()

    train_end = pd.to_datetime(frames["train"].timestamp_utc, utc=True).max()
    val_start = pd.to_datetime(frames["val"].timestamp_utc, utc=True).min()
    if train_end + pd.Timedelta(hours=24) >= val_start:
        raise ValueError("24h purge required")

    if args.train_method in TREES:
        data = {
            split: sequence_table(frame, features, targets, args.sequence_length)
            for split, frame in frames.items()
        }
        if any(not len(item["indices"]) for item in data.values()):
            raise ValueError("Empty data")
        saved = {
            "feature_columns": features,
            "target_columns": targets,
            "sequence_length": args.sequence_length,
        }
        return frames, data, saved

    train = HydrologyDataset(
        frames["train"],
        feature_cols=features,
        target_cols=targets,
        fit_scalers=True,
        seq_len=args.sequence_length,
        require_current_level=True,
    )
    saved = {
        "feature_columns": features,
        "target_columns": targets,
        "sequence_length": args.sequence_length,
        "feature_scaler": train.feature_scaler,
        "target_scaler": train.target_scaler,
        "train_method": args.train_method,
        "forecast_history_policy": "all_timesteps",
        "rain_station_ids": spec.get("rain_station_ids", []),
    }

    from src.ai.comparison import evaluation_dataset

    data = {"train": train, "val": evaluation_dataset(frames["val"], saved)}
    if any(not len(item) for item in data.values()):
        raise ValueError("Empty data")

    return frames, data, saved


def maybe_prune(trial, value, step):
    trial.report(value, step)
    if trial.should_prune():
        raise optuna.TrialPruned(f"Validation pruning at step {step}")


def data_at_sequence(reference_data, reference_saved, sequence_length, method):
    """Change history length while retaining identical anchors, labels and scalers."""
    cohort_length = reference_saved["sequence_length"]
    if not 1 <= sequence_length <= cohort_length:
        raise ValueError("Sequence exceeds the common cohort warmup")

    saved = dict(
        reference_saved,
        sequence_length=sequence_length,
        cohort_sequence_length=cohort_length,
    )
    data = {}

    for split, reference in reference_data.items():
        if method in TREES:
            # sequence_table orders newest lags first, so a prefix is the exact
            # shorter history. Views avoid caching a full matrix per candidate.
            width = sequence_length * len(saved["feature_columns"])
            data[split] = dict(
                reference,
                X=reference["X"][:, :width],
                names=reference["names"][:width],
            )
        else:
            # Keep end_indices/timestamps and scaled arrays from the longest
            # history. __getitem__ uses seq_len to select only the input window.
            dataset = copy(reference)
            dataset.seq_len = sequence_length
            data[split] = dataset

    return data, saved


def tree_progress_scale(learning_rate):
    """Approximate comparable progress without penalizing small learning rates early."""
    return min(1.0, learning_rate / TREE_REFERENCE_LEARNING_RATE)


def train_neural(trial, args, params, data, saved, folder):
    from research.output_horizon_experiment import parser as neural_parser

    config = neural_parser().parse_args([
        "--outputs", str(args.outputs), "--train-method", args.train_method,
    ])
    config.device = args.device
    config.seed = args.seed
    config.lstm_seq_len = args.sequence_length
    config.num_epochs = args.num_epochs
    config.early_stopping = args.early_stopping
    config.lstm_correction_penalty = args.correction_penalty if args.train_method != "lstm" else 0.0
    config.lstm_tail_penalty = args.tail_penalty
    config.aux_weight = args.aux_weight

    for name, value in params.items():
        setattr(config, name, value)
    if args.outputs == 24 and config.aux_weight <= 0:
        raise ValueError("O24 requires positive aux_weight")

    model = make_model(saved, config, args.device)
    joblib.dump(saved, folder / "model_best.preprocessing.joblib")

    loss = CommonHorizonAuxiliaryLoss(
        saved["target_columns"],
        saved["target_scaler"].scale_,
        auxiliary_weight=config.aux_weight if args.outputs == 24 else 0.0,
        tail_strength=config.lstm_tail_penalty,
        threshold_mm=config.lstm_tail_threshold_mm,
        transition_mm=config.lstm_tail_transition_mm,
    ).to(args.device)
    train = DataLoader(
        data["train"],
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),
        pin_memory=args.device == "cuda",
    )
    val = DataLoader(
        data["val"],
        batch_size=128,
        shuffle=False,
        generator=torch.Generator().manual_seed(args.seed + 1),
    )
    if not len(train):
        raise ValueError("Batch size exceeds train population")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lstm_lr, weight_decay=config.lstm_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    baseline = evaluate(model, val, loss, saved, args.device)
    best_metrics = focus_metrics(baseline, args.focus_horizons)
    best = objective_value(best_metrics["above_200"], best_metrics["rmse_mm"], args.objective)
    best_epoch = 0
    wait = 0
    history = [{"epoch": 0, **best_metrics, "objective": best}]

    torch.save(model.state_dict(), folder / "model_best.pt")
    (folder / "effective_config.json").write_text(json.dumps(vars(config), default=str, indent=2))

    try:
        for epoch in range(1, args.num_epochs + 1):
            model.train()

            for inputs, targets in train:
                inputs, targets = inputs.to(args.device), targets.to(args.device)
                optimizer.zero_grad(set_to_none=True)
                predictions = model(inputs)
                value = training_objective(
                    model, inputs, predictions, loss(predictions, targets), loss,
                    config.lstm_correction_penalty,
                )
                if not torch.isfinite(value):
                    raise ValueError("Nonfinite training loss")

                value.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()

            scheduler.step()
            metrics = evaluate(model, val, loss, saved, args.device)
            current = focus_metrics(metrics, args.focus_horizons)
            score = objective_value(current["above_200"], current["rmse_mm"], args.objective)

            if score < best:
                best = score
                best_metrics = current
                best_epoch = epoch
                wait = 0
                torch.save(model.state_dict(), folder / "model_best.pt")
            else:
                wait += 1

            history.append({"epoch": epoch, **current, "objective": score, "best_epoch": best_epoch})
            pd.DataFrame(history).to_csv(folder / "validation.csv", index=False)
            trial.set_user_attr("validation", dict(**best_metrics, selected_epoch=best_epoch))
            print(
                f"trial{trial.number} epoch{epoch}: N200={current['above_200']}, "
                f"RMSE={current['rmse_mm']:.2f}, best={best_epoch}",
                flush=True,
            )
            maybe_prune(trial, best, epoch)

            if wait >= args.early_stopping:
                break
    finally:
        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return best, dict(**best_metrics, selected_epoch=best_epoch)


def train_tree(trial, args, params, data, saved, folder):
    options = {
        "target_mode": args.target_mode,
        "checkpoint_metric": args.objective,
        "n_estimators": args.n_estimators,
        "early_stopping": args.early_stopping,
        "learning_rate": 0.05,
        "max_depth": 4,
        "num_leaves": 15,
        "n_jobs": args.n_jobs,
        "seed": args.seed,
        "device": args.device,
    }
    options.update(params)

    # A low learning rate needs more raw rounds for the same approximate progress.
    # Keep the requested n_estimators as a hard maximum, never silently extend it.
    progress_scale = tree_progress_scale(options["learning_rate"])
    options["early_stopping"] = min(
        args.n_estimators, math.ceil(args.early_stopping / progress_scale),
    )
    print(
        f"trial{trial.number}: lr={options['learning_rate']:.6g}, "
        f"round limit={args.n_estimators}, patience={options['early_stopping']}, "
        f"pruning step=raw round * {progress_scale:.6g}",
        flush=True,
    )

    targets = saved["target_columns"]
    focus = [targets.index(f"target_water_level_plus_{hour * 60}m") for hour in args.focus_horizons]
    order = focus + [index for index in range(len(targets)) if index not in focus]
    scores = {}

    for index in focus:
        error = data["val"]["level"] - data["val"]["y"][:, index]
        scores[index] = {
            "above_200": int((abs(error) > 200).sum()),
            "rmse_mm": float(np.sqrt(np.mean(error ** 2))),
        }

    heads = {}
    artifact = dict(
        **saved, train_method=args.train_method, target_mode=args.target_mode,
        status="training", horizons=[],
    )
    effective = dict(
        options,
        sequence_length=saved["sequence_length"],
        cohort_sequence_length=saved.get("cohort_sequence_length", saved["sequence_length"]),
        pruning_progress_scale=progress_scale,
    )
    (folder / "effective_config.json").write_text(json.dumps(effective, indent=2))

    def aggregate():
        return {
            "above_200": sum(item["above_200"] for item in scores.values()),
            "rmse_mm": math.sqrt(np.mean([item["rmse_mm"] ** 2 for item in scores.values()])),
        }

    last_reported_step = 0

    for index in order:
        hour = int(targets[index].rsplit("_", 1)[-1][:-1]) // 60

        def progress(raw_step, selector):
            nonlocal last_reported_step

            # Train focus heads first; prune during the last focus head only.
            if index != focus[-1]:
                return

            scores[index] = selector.best_metrics.copy()
            current = aggregate()
            trial.set_user_attr("validation", current)
            step = math.floor(raw_step * progress_scale)
            if step <= last_reported_step:
                return

            last_reported_step = step
            value = objective_value(current["above_200"], current["rmse_mm"], args.objective)
            maybe_prune(trial, value, step)

        extension = "json" if args.train_method == "xgb" else "txt"
        path = folder / f"horizon_{hour:02d}h.{extension}"
        result = train_horizon(
            args.train_method, data["train"], data["val"], index, options, path,
            on_iteration=progress,
        )
        pd.DataFrame(result.pop("history")).to_csv(folder / f"validation_h{hour:02d}.csv", index=False)

        if index in scores:
            scores[index] = result["validation_metrics"]
        result["target"] = targets[index]
        heads[index] = result
        artifact["horizons"] = [heads[key] for key in sorted(heads)]
        (folder / "tree_model.json").write_text(json.dumps(artifact, indent=2))
        print(
            f"trial{trial.number} +{hour}h: {result['kind']}, rounds={result['best_round']}",
            flush=True,
        )

    artifact["status"] = "complete"
    (folder / "tree_model.json").write_text(json.dumps(artifact, indent=2))
    metrics = aggregate()
    value = objective_value(metrics["above_200"], metrics["rmse_mm"], args.objective)

    return value, metrics


def evaluate_best(args, folder, out, saved):
    """Invoked only after the study has finished selecting a COMPLETE trial."""
    from src.ai.comparison import DEFAULT_REGISTRY, evaluation_dataset, load_candidate

    frame = read_frame(args.dataset_dir / "test.csv").reset_index()
    cohort_length = saved.get("cohort_sequence_length", saved["sequence_length"])
    # Preserve all historical predictors; exclude early labels from evaluation
    # so a shorter winning window uses the same cohort as the study's longest.
    frame.loc[frame.index < cohort_length - 1, saved["target_columns"]] = np.nan

    if args.train_method in NEURALS:
        from research.output_horizon_experiment import write_comparisons

        pre = joblib.load(folder / "model_best.preprocessing.joblib")
        model = load_candidate((folder / "model_best.pt").read_bytes(), pre, args.device)
        config = argparse.Namespace(device=args.device, reference_checkpoint=args.reference_checkpoint)
        summaries, errors = write_comparisons(
            out, model, pre, {"test": frame}, {"test": evaluation_dataset(frame, pre)}, config,
        )
        if errors:
            raise RuntimeError(f"Best model comparison incomplete: {errors}")
        return summaries

    from research.tree_horizon_experiment import compare, prediction_points, summarize
    from src.ai.metrics import markdown_table

    model = TreeSequenceForecast(folder)
    data, values = model.predict_frame(frame)
    points = prediction_points(data, values, saved["target_columns"])
    points.to_csv(out / "all_predictions.csv.gz", index=False)
    pd.DataFrame([
        dict(horizon_hours=hour // 60, **summarize(group))
        for hour, group in points.groupby("horizon_minutes")
    ]).to_csv(out / "all_horizon_metrics.csv", index=False)

    config = next(
        item for item in json.loads(DEFAULT_REGISTRY.read_text())
        if item["station_id"] == "0074647403"
    )
    refs = [
        ("B", args.reference_checkpoint, args.reference_checkpoint.with_suffix(".preprocessing.joblib"), frame),
        (
            "production", ROOT / config["model_path"], ROOT / config["preprocessing_path"],
            pd.read_csv(ROOT / "train_data/da_vien_1h_2026-10-15/test.csv"),
        ),
    ]
    result = {}
    text = "# Cấu hình tree được chọn trên validation\n\n"

    for name, path, pre, reference_frame in refs:
        pre = joblib.load(pre)
        reference = load_candidate(path.read_bytes(), pre, "cpu")
        metrics, paired, coverage = compare(
            data, values, frame, saved["target_columns"], reference_frame, pre, reference,
        )
        metrics.to_csv(out / f"comparison_{name}_metrics.csv", index=False)
        paired.to_csv(out / f"comparison_{name}_predictions.csv.gz", index=False)
        result[name] = coverage
        text += f"## So với {name}\n\n" + markdown_table(metrics, list(metrics.columns)) + "\n\n"

    text += (
        "Test được đọc sau khi chọn trial, không chọn lại bằng test. Dataset test đã được "
        "xem trong các thử nghiệm trước nên vẫn là đánh giá khám phá.\n"
    )
    (out / "REPORT.md").write_text(text)

    return result


@contextmanager
def study_storage(path):
    storage = optuna.storages.RDBStorage("sqlite:///" + str(path))
    try:
        yield storage
    finally:
        storage.remove_session()
        storage.engine.dispose()


@contextmanager
def study_lock(out):
    import fcntl

    with (out / "search.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another search process owns this output directory") from None

        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-method", choices=["grid", "tpe"], required=True)
    parser.add_argument("--train-method", choices=sorted(TREES | NEURALS), required=True)
    parser.add_argument(
        "--dataset-dir", type=Path,
        default=ROOT / "train_data/da_vien_1h_2026-10-15_forecast_mask_o24",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--search-space", type=Path, help="JSON lists (grid), or lists/int/float ranges (TPE)")
    parser.add_argument("--outputs", type=int, choices=[5, 24], default=5)
    sequence = parser.add_mutually_exclusive_group()
    sequence.add_argument(
        "--sequence-length", type=int,
        help="Fix history length; otherwise search every integer hour from 12 to 168",
    )
    sequence.add_argument(
        "--sequence-lengths", type=int, nargs="+",
        help="Search these history lengths (12..168 hours) instead of the default range",
    )
    parser.add_argument("--focus-horizons", type=int, nargs="+", default=[12, 24])
    parser.add_argument("--objective", choices=["large_error_rate", "rmse"], default="large_error_rate")
    parser.add_argument(
        "--n-trials", type=int,
        help=f"Total study budget including previous attempts; default full grid / TPE {DEFAULT_TPE_TRIALS}",
    )
    parser.add_argument("--tpe-startup-trials", type=int, default=20, help="Random exploration before TPE-guided suggestions")
    parser.add_argument("--timeout", type=float, help="Seconds per invocation; not a hard kill of an active trial")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--n-estimators", type=int, default=5000, help="Hard maximum boosting rounds per tree head")
    parser.add_argument("--target-mode", choices=["residual", "direct"], default="residual", help="Tree targets only")
    parser.add_argument(
        "--early-stopping", type=int,
        help="Default neural25 / tree50; tree patience scales up for learning_rate <0.05",
    )
    parser.add_argument("--n-jobs", type=int, default=4, help="Threads within one tree trial; trials run sequentially")
    parser.add_argument("--correction-penalty", type=float, default=0.1)
    parser.add_argument("--tail-penalty", type=float, default=0.5)
    parser.add_argument("--aux-weight", type=float, default=0.25)
    parser.add_argument("--pruning-startup-trials", type=int, default=5)
    parser.add_argument(
        "--pruning-warmup", type=int,
        help="Default neural15 epochs / tree100 LR-adjusted steps of last focus head",
    )
    parser.add_argument("--pruning-interval", type=int, default=5)
    parser.add_argument("--no-pruning", action="store_true")
    parser.add_argument(
        "--evaluate-best", action="store_true",
        help="After search only, score selected trial on test against B/production",
    )
    parser.add_argument("--reference-checkpoint", type=Path, default=SOURCE)
    parser.add_argument(
        "--check-only", action="store_true",
        help="Validate space/train/val; no training, database or output directory",
    )

    return parser


def run(args):
    if args.early_stopping is None:
        args.early_stopping = 50 if args.train_method in TREES else 25
    if args.pruning_warmup is None:
        args.pruning_warmup = 100 if args.train_method in TREES else 15

    positive = [
        "num_epochs", "n_estimators", "early_stopping",
        "n_jobs", "pruning_interval", "tpe_startup_trials",
    ]
    for name in positive:
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")

    if args.sequence_length is not None and args.sequence_length < 1:
        raise ValueError("sequence_length must be positive")

    if min(args.pruning_startup_trials, args.pruning_warmup) < 0:
        raise ValueError("Pruning startup/warmup must be nonnegative")
    if args.n_trials is not None and args.n_trials < 1:
        raise ValueError("n_trials must be positive")
    if args.timeout is not None and (not math.isfinite(args.timeout) or args.timeout <= 0):
        raise ValueError("Invalid timeout")
    if not 0 <= args.seed < 2 ** 31:
        raise ValueError("Invalid seed")
    if not args.focus_horizons or len(set(args.focus_horizons)) != len(args.focus_horizons):
        raise ValueError("Unique focus horizons required")

    available = [1, 3, 6, 12, 24] if args.outputs == 5 else range(1, 25)
    if not set(args.focus_horizons).issubset(available):
        raise ValueError("Focus horizon absent")

    for name, value in [
        ("lstm_correction_penalty", args.correction_penalty),
        ("lstm_tail_penalty", args.tail_penalty),
        ("aux_weight", args.aux_weight),
    ]:
        check_parameter(name, value, "residual_lstm")

    if args.outputs == 24 and args.aux_weight <= 0:
        raise ValueError("Positive aux_weight required")
    if args.train_method == "lgbm" and args.device != "cpu":
        raise ValueError("LightGBM uses CPU")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent CPU fallback")

    space, cohort_length = resolve_search_space(args)
    reference_args = copy(args)
    reference_args.sequence_length = cohort_length
    frames, data, saved = load_search_data(reference_args)
    population = {
        split: len(item["indices"]) if args.train_method in TREES else len(item)
        for split, item in data.items()
    }
    print(json.dumps({
        "search_space": space,
        "focus": args.focus_horizons,
        "cohort_sequence_length": cohort_length,
        "population": population,
    }, indent=2), flush=True)

    if args.check_only:
        return

    out = args.output_dir.resolve()
    if args.resume:
        if not (out / "study.sqlite3").exists():
            raise FileNotFoundError("No existing study to resume")
    else:
        out.mkdir(parents=True, exist_ok=False)

    variable_flags = {"n_trials", "timeout", "resume", "check_only", "evaluate_best", "search_space"}
    config = {
        name: str(value.resolve()) if isinstance(value, Path) else value
        for name, value in vars(args).items() if name not in variable_flags
    }
    config.update(
        space=space,
        population=population,
        feature_columns=saved["feature_columns"],
        target_columns=saved["target_columns"],
        cohort_sequence_length=cohort_length,
        validation_first=str(frames["val"].timestamp_utc.iloc[0]),
        validation_last=str(frames["val"].timestamp_utc.iloc[-1]),
    )

    with study_lock(out):
        config_file = out / "search_config.json"
        if args.resume:
            if json.loads(config_file.read_text()) != config:
                raise ValueError("Resume configuration differs; use a new output-dir")
        else:
            config_file.write_text(json.dumps(config, indent=2))

        with study_storage(out / "study.sqlite3") as storage:
            if args.search_method == "grid":
                sampler = optuna.samplers.GridSampler(space, seed=args.seed)
            else:
                sampler = optuna.samplers.TPESampler(
                    seed=args.seed, n_startup_trials=args.tpe_startup_trials,
                )

            if args.search_method == "grid" or args.no_pruning:
                pruner = optuna.pruners.NopPruner()
            else:
                pruner = optuna.pruners.MedianPruner(
                    n_startup_trials=args.pruning_startup_trials,
                    n_warmup_steps=args.pruning_warmup,
                    interval_steps=args.pruning_interval,
                )

            study = optuna.create_study(
                study_name="hydrology", direction="minimize", sampler=sampler,
                pruner=pruner, storage=storage, load_if_exists=args.resume,
            )
            # flock gives sequential ownership; a stale RUNNING trial was interrupted.
            for trial in study.trials:
                if trial.state == optuna.trial.TrialState.RUNNING:
                    study.tell(trial.number, state=optuna.trial.TrialState.FAIL)

            grid_size = math.prod(len(values) for values in space.values()) if args.search_method == "grid" else None
            total = args.n_trials if args.n_trials is not None else (
                grid_size if args.search_method == "grid" else DEFAULT_TPE_TRIALS
            )
            if grid_size is not None:
                total = min(total, grid_size)
            remaining = max(total - len(study.trials), 0)

            def objective(trial):
                params = suggest(trial, space)
                sequence_length = params.get("sequence_length", cohort_length)
                folder = out / f"trial_{trial.number:04d}"
                folder.mkdir()
                (folder / "params.json").write_text(json.dumps(params, indent=2))
                trial.set_user_attr("artifact_dir", str(folder))
                trial.set_user_attr("sequence_length", sequence_length)

                random.seed(args.seed)
                np.random.seed(args.seed)
                torch.manual_seed(args.seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                torch.use_deterministic_algorithms(True)
                torch.set_num_threads(2)

                try:
                    trial_args = copy(args)
                    trial_args.sequence_length = sequence_length
                    trial_data, trial_saved = data_at_sequence(
                        data, saved, sequence_length, args.train_method,
                    )
                    model_params = {
                        name: value for name, value in params.items()
                        if name != "sequence_length"
                    }
                    trainer = train_tree if args.train_method in TREES else train_neural
                    value, metrics = trainer(
                        trial, trial_args, model_params, trial_data, trial_saved, folder,
                    )
                    trial.set_user_attr("validation", metrics)
                    result = {"status": "COMPLETE", "value": value, "validation": metrics}
                    (folder / "result.json").write_text(json.dumps(result, indent=2))
                    return value
                except optuna.TrialPruned:
                    result = {"status": "PRUNED", "validation": trial.user_attrs.get("validation")}
                    (folder / "result.json").write_text(json.dumps(result, indent=2))
                    raise
                except Exception as exc:
                    result = {"status": "FAIL", "exception": type(exc).__name__}
                    (folder / "result.json").write_text(json.dumps(result, indent=2))
                    raise

            def export_results(study, trial=None):
                study.trials_dataframe().to_csv(out / "trials.csv", index=False)
                completed = [
                    item for item in study.trials
                    if item.state == optuna.trial.TrialState.COMPLETE
                ]
                if completed:
                    best = study.best_trial
                    result = {
                        "number": best.number,
                        "value": best.value,
                        "params": best.params,
                        "sequence_length": best.user_attrs["sequence_length"],
                        "validation": best.user_attrs["validation"],
                        "artifact_dir": best.user_attrs["artifact_dir"],
                    }
                    (out / "best_trial.json").write_text(json.dumps(result, indent=2))

            try:
                if remaining:
                    study.optimize(
                        objective, n_trials=remaining, timeout=args.timeout,
                        callbacks=[export_results], gc_after_trial=True,
                    )
            finally:
                export_results(study)

            if args.evaluate_best and any(
                trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
            ):
                best = study.best_trial
                evaluation = out / f"evaluation_trial_{best.number:04d}"
                if not (evaluation / "summary.json").exists():
                    evaluation.mkdir(exist_ok=True)
                    result = evaluate_best(args, Path(best.user_attrs["artifact_dir"]), evaluation, saved)
                    (evaluation / "summary.json").write_text(json.dumps(result, indent=2))

            for message in boundary_messages(study, space):
                print(message, flush=True)

            print("Results:", out / "trials.csv", flush=True)
            if (out / "best_trial.json").exists():
                print((out / "best_trial.json").read_text(), flush=True)


def main():
    run(parser().parse_args())


if __name__ == "__main__":
    main()
