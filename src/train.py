"""Command-line entry point for hydrology model training."""

import os
import sys
from pathlib import Path

# Set this before importing PyTorch so CUDA LSTM kernels can be deterministic.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:2")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

import pandas as pd
import torch
from torch import nn
from torch import optim

from src.metrics import evaluate_all
from src.models import LightGBM, LSTM, MeanEmbeddingForecastLSTMWithAdapter, ResidualLSTM, XGBoost
from src.trainers import LSTMTrainer, MEFLSTMAdapterTrainer, ResidualLSTMTrainer, VanillaLSTMTrainer
from src.utils import FEATURE_SETS, get_dataloader, load_dataset, load_mef_backbone, set_seed


parser = argparse.ArgumentParser()
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
parser.add_argument("--lstm-output-size", type=int, default=3)
parser.add_argument("--lstm-hidden-size", type=int, default=128)
parser.add_argument("--lstm-num-layers", type=int, default=2)
parser.add_argument("--lstm-drop-out", type=float, default=0.2)
parser.add_argument("--lstm-input-dropout", type=float, default=0.0)
parser.add_argument("--lstm-head-dropout", type=float, default=0.0)
parser.add_argument("--lstm-correction-penalty", type=float, default=0.0)
parser.add_argument("--lstm-feature-set", choices=FEATURE_SETS, default="all")
parser.add_argument("--lstm-lr", type=float, default=1e-4)
parser.add_argument("--lstm-weight-decay", type=float, default=0.01)
parser.add_argument("--lstm-seq-len", type=int)


def validate_args(args: argparse.Namespace) -> None:
    if args.lstm_correction_penalty < 0:
        parser.error("--lstm-correction-penalty must be non-negative")
    if args.lstm_correction_penalty and args.train_method != "residual_lstm":
        parser.error("--lstm-correction-penalty requires --train-method residual_lstm")
    if args.train_method == "residual_lstm":
        groups = args.lstm_feature_set.split("+")
        if args.lstm_feature_set != "all" and "water_level" not in groups:
            parser.error("residual_lstm requires water_level in --lstm-feature-set")
    if args.train_method == "lstm_adapter":
        if not args.pretrained_model or not args.pretrained_config:
            parser.error("lstm_adapter requires --pretrained-model and --pretrained-config")
        for path in (args.pretrained_model, args.pretrained_config):
            if not Path(path).is_file():
                parser.error(f"MEF file not found: {path}")


def evaluate_tabular(model: XGBoost | LightGBM, test_df: pd.DataFrame) -> pd.DataFrame:
    predictions = model.predict(test_df)
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
    model = XGBoost() if args.train_method == "xgb" else LightGBM()
    model.train(train_df, eval_df, random_state=args.seed)
    evaluate_tabular(model, test_df)


def build_sequence_trainer(args: argparse.Namespace) -> LSTMTrainer:
    default_sequence_lengths = {"10min": 144, "30min": 48, "1h": 24}
    sequence_length = (
        default_sequence_lengths[args.horizon]
        if args.lstm_seq_len is None
        else args.lstm_seq_len
    )
    includes_current_level = (
        args.lstm_feature_set == "all"
        or "water_level" in args.lstm_feature_set.split("+")
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
        feature_set=args.lstm_feature_set,
    )
    input_size = train_loader.dataset.num_features
    output_size = train_loader.dataset.num_targets

    if args.train_method == "lstm_adapter":
        model = MeanEmbeddingForecastLSTMWithAdapter(
            load_mef_backbone(args.pretrained_config, args.pretrained_model),
            input_size,
            output_size,
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
    return trainer_class(**trainer_options)


def main() -> None:
    args = parser.parse_args()
    validate_args(args)
    set_seed(args.seed)
    print(f"Seed: {args.seed}")
    if args.train_method in ("xgb", "lgbm"):
        train_tabular(args)
    else:
        build_sequence_trainer(args).train()


if __name__ == "__main__":
    main()
