"""Command-line entry point for hydrology model training."""

import os
import math
import sys
from pathlib import Path
from dataclasses import asdict

# Set this before importing PyTorch so CUDA LSTM kernels can be deterministic.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:2")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import mlflow
import mlflow.xgboost
import mlflow.lightgbm
import pandas as pd
import torch
from torch import nn
from torch import optim

from src.metrics import evaluate_all
from src.models import (
    LightGBM, LightGBMSettings, LSTM, MeanEmbeddingForecastLSTMWithAdapter,
    ResidualLSTM, XGBoost, XGBoostSettings,
)
from src.trainers import LSTMTrainer, MEFLSTMAdapterTrainer, ResidualLSTMTrainer, VanillaLSTMTrainer
from src.utils import FEATURE_SETS, get_dataloader, load_dataset, load_mef_backbone, set_seed


MLFLOW_TRACKING_URI = "http://localhost:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--mlflow-ex-name", type=str.strip, required=True,
    help="MLflow experiment name, for example 30min or 1h",
)
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible training")
parser.add_argument("--horizon", required=True, choices=["10min", "30min", "1h"])
parser.add_argument("--train-method", required=True, choices=["xgb", "lgbm", "lstm", "residual_lstm", "lstm_adapter"])
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--num-worker", type=int, default=0)
parser.add_argument("--num-epochs", type=int, default=100)
parser.add_argument("--early-stopping", type=int, default=10)
parser.add_argument("--gradient-accumulation", type=int, default=1)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--pretrained-model", help="MEF checkpoint state_dict path for lstm_adapter")
parser.add_argument("--pretrained-config", help="MEF YAML configuration path for lstm_adapter")
parser.add_argument(
    "--mef-unfreeze-backbone", action="store_true",
    help="Train the pretrained MEF backbone as well as the adapter and output head",
)
parser.add_argument("--mef-two-stage", action="store_true",
                    help="Train frozen MEF first, then fine-tune from its best checkpoint")
parser.add_argument("--mef-finetune-epochs", type=int, default=30,
                    help="Additional stage-two epochs; --num-epochs controls stage one")
parser.add_argument("--mef-finetune-backbone", choices=["forecast", "all"], default="forecast",
                    help="Recurrent MEF modules to unfreeze in stage two")
parser.add_argument("--mef-backbone-lr", type=float, default=1e-5,
                    help="Stage-two learning rate for unfrozen recurrent weights")
parser.add_argument("--mef-adapter-lr", type=float, default=1e-4,
                    help="Stage-two learning rate for the adapter and output head")
parser.add_argument("--lstm-output-size", type=int, default=3)
parser.add_argument("--lstm-hidden-size", type=int, default=128)
parser.add_argument("--lstm-num-layers", type=int, default=2)
parser.add_argument("--lstm-drop-out", type=float, default=0.2)
parser.add_argument("--lstm-input-dropout", type=float, default=0.0)
parser.add_argument("--lstm-head-dropout", type=float, default=0.0)
parser.add_argument("--lstm-correction-penalty", type=float, default=0.0)
parser.add_argument(
    "--feature-set", "--lstm-feature-set", dest="feature_set",
    choices=FEATURE_SETS, default="all",
    help="Input groups for tree and LSTM models",
)
parser.add_argument("--lstm-lr", type=float, default=1e-4)
parser.add_argument("--lstm-weight-decay", type=float, default=0.01)
parser.add_argument("--lstm-seq-len", type=int)

xgb_defaults = XGBoostSettings()
xgb_args = parser.add_argument_group("XGBoost settings")
xgb_args.add_argument("--xgb-n-estimators", type=int, default=xgb_defaults.n_estimators)
xgb_args.add_argument("--xgb-learning-rate", type=float, default=xgb_defaults.learning_rate)
xgb_args.add_argument("--xgb-max-depth", type=int, default=xgb_defaults.max_depth)
xgb_args.add_argument("--xgb-min-child-weight", type=float, default=xgb_defaults.min_child_weight)
xgb_args.add_argument("--xgb-subsample", type=float, default=xgb_defaults.subsample)
xgb_args.add_argument("--xgb-colsample-bytree", type=float, default=xgb_defaults.colsample_bytree)
xgb_args.add_argument("--xgb-reg-alpha", type=float, default=xgb_defaults.reg_alpha)
xgb_args.add_argument("--xgb-reg-lambda", type=float, default=xgb_defaults.reg_lambda)
xgb_args.add_argument("--xgb-early-stopping-rounds", type=int, default=xgb_defaults.early_stopping_rounds,
                      help="Set to 0 to disable early stopping")
xgb_args.add_argument("--xgb-n-jobs", type=int, default=xgb_defaults.n_jobs)

lgbm_defaults = LightGBMSettings()
lgbm_args = parser.add_argument_group("LightGBM settings")
lgbm_args.add_argument("--lgbm-n-estimators", type=int, default=lgbm_defaults.n_estimators)
lgbm_args.add_argument("--lgbm-learning-rate", type=float, default=lgbm_defaults.learning_rate)
lgbm_args.add_argument("--lgbm-num-leaves", type=int, default=lgbm_defaults.num_leaves)
lgbm_args.add_argument("--lgbm-max-depth", type=int, default=lgbm_defaults.max_depth)
lgbm_args.add_argument("--lgbm-subsample", type=float, default=lgbm_defaults.subsample)
lgbm_args.add_argument("--lgbm-subsample-freq", type=int, default=lgbm_defaults.subsample_freq)
lgbm_args.add_argument("--lgbm-colsample-bytree", type=float, default=lgbm_defaults.colsample_bytree)
lgbm_args.add_argument("--lgbm-reg-alpha", type=float, default=lgbm_defaults.reg_alpha)
lgbm_args.add_argument("--lgbm-reg-lambda", type=float, default=lgbm_defaults.reg_lambda)
lgbm_args.add_argument("--lgbm-early-stopping-rounds", type=int, default=lgbm_defaults.early_stopping_rounds,
                       help="Set to 0 to disable early stopping")
lgbm_args.add_argument("--lgbm-n-jobs", type=int, default=lgbm_defaults.n_jobs)
lgbm_args.add_argument("--lgbm-verbose", type=int, default=lgbm_defaults.verbose)


def validate_args(args: argparse.Namespace) -> None:
    if not args.mlflow_ex_name:
        parser.error("--mlflow-ex-name must not be empty")
    if args.mef_unfreeze_backbone and args.train_method != "lstm_adapter":
        parser.error("--mef-unfreeze-backbone requires --train-method lstm_adapter")
    if args.mef_two_stage:
        if args.train_method != "lstm_adapter":
            parser.error("--mef-two-stage requires --train-method lstm_adapter")
        if args.mef_unfreeze_backbone:
            parser.error("--mef-two-stage starts frozen; omit --mef-unfreeze-backbone")
        if args.num_epochs < 1 or args.mef_finetune_epochs < 1:
            parser.error("Both training stages require at least one epoch")
        if not (0 < args.mef_backbone_lr < float("inf") and 0 < args.mef_adapter_lr < float("inf")):
            parser.error("MEF fine-tuning learning rates must be finite and positive")
    if args.lstm_correction_penalty < 0:
        parser.error("--lstm-correction-penalty must be non-negative")
    if args.lstm_correction_penalty and args.train_method != "residual_lstm":
        parser.error("--lstm-correction-penalty requires --train-method residual_lstm")
    if args.train_method == "residual_lstm":
        groups = args.feature_set.split("+")
        if args.feature_set != "all" and "water_level" not in groups:
            parser.error("residual_lstm requires water_level in --feature-set")
    if args.train_method == "lstm_adapter":
        if not args.pretrained_model or not args.pretrained_config:
            parser.error("lstm_adapter requires --pretrained-model and --pretrained-config")
        for path in (args.pretrained_model, args.pretrained_config):
            if not Path(path).is_file():
                parser.error(f"MEF file not found: {path}")

    if args.train_method == "xgb":
        if args.xgb_n_estimators < 1 or args.xgb_early_stopping_rounds < 0:
            parser.error("XGBoost estimators must be positive and early stopping must be non-negative")
        if args.xgb_learning_rate <= 0 or args.xgb_max_depth < 0 or args.xgb_min_child_weight < 0:
            parser.error("XGBoost learning rate must be positive; depth and child weight must be non-negative")
        proportions = (args.xgb_subsample, args.xgb_colsample_bytree)
        regularizers = (args.xgb_reg_alpha, args.xgb_reg_lambda)
    elif args.train_method == "lgbm":
        if args.lgbm_n_estimators < 1 or args.lgbm_early_stopping_rounds < 0:
            parser.error("LightGBM estimators must be positive and early stopping must be non-negative")
        if args.lgbm_learning_rate <= 0 or args.lgbm_num_leaves < 2:
            parser.error("LightGBM learning rate must be positive and num leaves must be at least 2")
        if args.lgbm_max_depth != -1 and args.lgbm_max_depth < 1:
            parser.error("LightGBM max depth must be -1 or positive")
        if args.lgbm_subsample_freq < 0:
            parser.error("LightGBM subsample frequency must be non-negative")
        proportions = (args.lgbm_subsample, args.lgbm_colsample_bytree)
        regularizers = (args.lgbm_reg_alpha, args.lgbm_reg_lambda)
    else:
        return

    if any(not 0 < value <= 1 for value in proportions):
        parser.error("Tree sampling fractions must be in (0, 1]")
    if any(value < 0 for value in regularizers):
        parser.error("Tree regularization values must be non-negative")


def evaluate_tabular(model: XGBoost | LightGBM, test_df: pd.DataFrame) -> pd.DataFrame:
    predictions = model.predict(test_df)
    observed = test_df[model.target_columns].to_numpy(dtype=float)
    predicted = predictions[model.target_columns].to_numpy(dtype=float)
    test_loss = float(((predicted - observed) ** 2).mean())
    mlflow.log_metric("test_loss", test_loss)
    mlflow.set_tag("test_loss_scale", "original_target_units")
    print(f"Test Loss (MSE, original target units): {test_loss:.4f}")
    timestamps = test_df["timestamp_utc"] if "timestamp_utc" in test_df.columns else None
    summaries = []
    for target_name in model.target_columns:
        metrics = evaluate_all(test_df[target_name], predictions[target_name], timestamps=timestamps)
        metrics["target"] = target_name
        summaries.append(metrics)
    summary = pd.DataFrame(summaries).set_index("target")
    displayed_metrics = (
        "rmse", "mae", "nse", "kge", "flood_f1", "flood_precision",
        "flood_recall", "flood_accuracy", "peak_absolute_error",
        "peak_timing_error_minutes",
    )
    print("\n" + "=" * 45 + " TEST EVALUATION " + "=" * 45)
    print(summary[[name for name in displayed_metrics if name in summary]].to_string())
    print("=" * 107 + "\n")
    return summary


def train_tabular(args: argparse.Namespace) -> None:
    train_df, eval_df, test_df = load_dataset(args.horizon)
    if args.train_method == "xgb":
        settings = XGBoostSettings(
            n_estimators=args.xgb_n_estimators,
            learning_rate=args.xgb_learning_rate,
            max_depth=args.xgb_max_depth,
            min_child_weight=args.xgb_min_child_weight,
            subsample=args.xgb_subsample,
            colsample_bytree=args.xgb_colsample_bytree,
            reg_alpha=args.xgb_reg_alpha,
            reg_lambda=args.xgb_reg_lambda,
            early_stopping_rounds=args.xgb_early_stopping_rounds,
            n_jobs=args.xgb_n_jobs,
        )
        model = XGBoost(settings=settings, feature_set=args.feature_set)
    else:
        settings = LightGBMSettings(
            n_estimators=args.lgbm_n_estimators,
            learning_rate=args.lgbm_learning_rate,
            num_leaves=args.lgbm_num_leaves,
            max_depth=args.lgbm_max_depth,
            subsample=args.lgbm_subsample,
            subsample_freq=args.lgbm_subsample_freq,
            colsample_bytree=args.lgbm_colsample_bytree,
            reg_alpha=args.lgbm_reg_alpha,
            reg_lambda=args.lgbm_reg_lambda,
            early_stopping_rounds=args.lgbm_early_stopping_rounds,
            n_jobs=args.lgbm_n_jobs,
            verbose=args.lgbm_verbose,
        )
        model = LightGBM(settings=settings, feature_set=args.feature_set)

    run_name = f"{args.train_method}_{args.horizon}_{args.feature_set}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "train_method": args.train_method,
                "horizon": args.horizon,
                "feature_set": args.feature_set,
                "seed": args.seed,
                "train_size": len(train_df),
                "eval_size": len(eval_df),
                "test_size": len(test_df),
                **asdict(model.settings)
            }
        )
        model.train(train_df, eval_df, random_state=args.seed)

        mlflow.log_param("input_size", len(model.feature_columns))
        mlflow.log_dict({"feature_columns": model.feature_columns}, "features.json")

        summary = evaluate_tabular(model, test_df)

        metrics = {}
        for target_name, target_metrics in summary.iterrows():
            for metric_name, value in target_metrics.items():
                numeric_value = float(value)
                if math.isfinite(numeric_value):
                    metrics[f"test/{target_name}/{metric_name}"] = numeric_value

        mlflow.log_metrics(metrics)
        mlflow.log_table(summary.reset_index(), f"{run_name}_test_results.json")

        for target_name, estimator in model.models.items():
            model_name = f"{args.train_method}_{target_name}"

            if args.train_method == "xgb":
                model_info = mlflow.xgboost.log_model(estimator, name=model_name)
            else:
                model_info = mlflow.lightgbm.log_model(estimator, name=model_name)

            mlflow.set_tag(f"model_uri/{target_name}", model_info.model_uri)


def build_sequence_trainer(args: argparse.Namespace) -> LSTMTrainer:
    default_sequence_lengths = {"10min": 144, "30min": 48, "1h": 24}
    sequence_length = (
        default_sequence_lengths[args.horizon]
        if args.lstm_seq_len is None
        else args.lstm_seq_len
    )
    includes_current_level = (
        args.feature_set == "all"
        or "water_level" in args.feature_set.split("+")
    )
    train_loader, eval_loader, test_loader = get_dataloader(
        args.horizon,
        args.batch_size,
        args.num_worker,
        pin_memory=True,
        drop_last=True,
        train_shuffle=True,
        eval_shuffle=False,
        test_shuffle=False,
        persistent_workers=False,
        seq_len=sequence_length,
        seed=args.seed,
        require_current_level=includes_current_level,
        feature_set=args.feature_set,
    )
    input_size = train_loader.dataset.num_features
    output_size = train_loader.dataset.num_targets

    if args.train_method == "lstm_adapter":
        model = MeanEmbeddingForecastLSTMWithAdapter(
            load_mef_backbone(args.pretrained_config, args.pretrained_model),
            input_size,
            output_size,
            freeze_backbone=not args.mef_unfreeze_backbone,
            input_dropout=args.lstm_input_dropout,
            head_dropout=args.lstm_head_dropout,
        )
        parameters = (parameter for parameter in model.parameters() if parameter.requires_grad)
        trainer_class = MEFLSTMAdapterTrainer
    else:
        model_options = dict(
            input_size=input_size,
            hidden_size=args.lstm_hidden_size,
            num_layers=args.lstm_num_layers,
            dropout=args.lstm_drop_out,
            input_dropout=args.lstm_input_dropout,
            head_dropout=args.lstm_head_dropout,
            output_size=output_size,
        )
        if args.train_method == "residual_lstm":
            dataset = train_loader.dataset
            level_index = dataset.feature_cols.index("water_level_depth")
            level_mean = dataset.feature_scaler.mean_[level_index]
            level_scale = dataset.feature_scaler.scale_[level_index]
            target_mean = dataset.target_scaler.mean_
            target_scale = dataset.target_scaler.scale_
            model = ResidualLSTM(
                **model_options,
                level_feature_index=level_index,
                persistence_scale=(level_scale / target_scale).tolist(),
                persistence_offset=((level_mean - target_mean) / target_scale).tolist(),
            )
            trainer_class = ResidualLSTMTrainer
        else:
            model = LSTM(**model_options)
            trainer_class = VanillaLSTMTrainer
        parameters = model.parameters()

    optimizer = optim.AdamW(parameters, lr=args.lstm_lr, weight_decay=args.lstm_weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    trainer_options = dict(
        model=model,
        horizon=args.horizon,
        num_epochs=args.num_epochs,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        test_dataloader=test_loader,
        loss_func=nn.MSELoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping=args.early_stopping,
        gradient_accumulation=args.gradient_accumulation,
        device=args.device,
    )
    if trainer_class is ResidualLSTMTrainer:
        return ResidualLSTMTrainer(**trainer_options, correction_penalty=args.lstm_correction_penalty)
    if trainer_class is MEFLSTMAdapterTrainer:
        return MEFLSTMAdapterTrainer(
            **trainer_options, two_stage=args.mef_two_stage,
            finetune_epochs=args.mef_finetune_epochs,
            backbone_lr=args.mef_backbone_lr, adapter_lr=args.mef_adapter_lr,
            finetune_backbone=args.mef_finetune_backbone,
        )
    return trainer_class(**trainer_options)


def main() -> None:
    args = parser.parse_args()
    validate_args(args)
    set_seed(args.seed)
    print(f"Seed: {args.seed}")
    mlflow.set_experiment(args.mlflow_ex_name)
    print(f"MLflow experiment: {args.mlflow_ex_name}")
    if args.train_method in ("xgb", "lgbm"):
        train_tabular(args)
    else:
        trainer = build_sequence_trainer(args)
        dataset = trainer.train_dataloader.dataset
        run_name = f"{args.train_method}_{args.horizon}_{args.feature_set}"
        if args.train_method == "lstm_adapter":
            backbone_mode = "two_stage" if args.mef_two_stage else (
                "unfrozen" if args.mef_unfreeze_backbone else "frozen"
            )
            run_name += f"_{backbone_mode}"

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(
                {
                    "train_method": args.train_method,
                    "horizon": args.horizon,
                    "feature_set": args.feature_set,
                    "seed": args.seed,
                    "batch_size": args.batch_size,
                    "num_epochs": args.num_epochs,
                    "early_stopping": args.early_stopping,
                    "gradient_accumulation": args.gradient_accumulation,
                    "sequence_length": dataset.seq_len,
                    "input_size": dataset.num_features,
                    "output_size": dataset.num_targets,
                    "learning_rate": trainer.optimizer.defaults["lr"],
                    "weight_decay": trainer.optimizer.defaults["weight_decay"],
                    "device": str(trainer.device),
                    **trainer.model_settings()
                }
            )
            mlflow.log_dict({"feature_columns": dataset.feature_cols}, "features.json")
            trainer.train()


if __name__ == "__main__":
    main()
