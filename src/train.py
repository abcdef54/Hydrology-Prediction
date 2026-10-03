import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import pandas as pd
import xgboost as xgb
import lightgbm as gbm
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from googlehydrology.modelzoo.mean_embedding_forecast_lstm import MeanEmbeddingForecastLSTM
from src.models import XGBoost, LightGBM, LSTM, MeanEmbeddingForecastLSTMWithAdapter
from src.metrics import evaluate_all
from src.dataset import HydrologyDataset

DATASET_PATH_10MIN = "./train_data/10min/"
DATASET_PATH_30MIN = "./train_data/30min/"
DATASET_PATH_1H = "./train_data/1h/"


parser = argparse.ArgumentParser()
parser.add_argument("--horizon", type=str, required=True, choices=["10min", "30min", "1h"])
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--num-worker", type=int, default=0)
parser.add_argument("--num-epochs", type=int, default=100)
parser.add_argument("--early-stopping", type=int, default=10)
parser.add_argument("--gradient-accumulation", type=int, default=1)
parser.add_argument("--train-method", type=str, required=True, choices=["xgb", "lgbm", "lstm", "lstm_adapter"])
parser.add_argument("--pretrained-model", type=str, required=False)
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--local-hindcast-size", type=int, default=178)
parser.add_argument("--pretrained-hindcast-size", type=int, default=64)
parser.add_argument("--local-forecast-size", type=int, default=178)
parser.add_argument("--pretrained-forecast-size", type=int, default=64)
parser.add_argument("--lstm-output-size", type=int, default=3)
parser.add_argument("--lstm-hidden-size", type=int, default=128, required=False, help="LSTM Hidden Size")
parser.add_argument("--lstm-num-layers", type=int, default=2, required=False)
parser.add_argument("--lstm-drop-out", type=float, default=0.2, required=False)
parser.add_argument("--lstm-lr", type=float, required=False, default=1e-4)
parser.add_argument("--lstm-weight-decay", type=float, default=0.01, required=False)
parser.add_argument("--lstm-seq-len", type=int, default=None, required=False)


def load_dataset(horizon: str):
    if horizon == "10min":
        base_path = DATASET_PATH_10MIN
    elif horizon == "30min":
        base_path = DATASET_PATH_30MIN
    elif horizon == "1h":
        base_path = DATASET_PATH_1H
    
    train = pd.read_csv(os.path.join(base_path, "train.csv"))
    eval = pd.read_csv(os.path.join(base_path, "val.csv"))
    test = pd.read_csv(os.path.join(base_path, "test.csv"))
    return train, eval, test


def get_dataloader(
    horizon: str,
    batch_size: int,
    num_worker: int = 1,
    pin_memory: bool = True,
    drop_last: bool = True,
    train_shuffle: bool = True,
    eval_shuffle: bool = False,
    test_shuffle: bool = False,
    persistent_workers: bool = False,
    seq_len: int = 24,
):
    train_df, eval_df, test_df = load_dataset(horizon)
    train_dataset = HydrologyDataset(train_df, fit_scalers=True, seq_len=seq_len)
    eval_dataset = HydrologyDataset(
        eval_df,
        feature_cols=train_dataset.feature_cols,
        target_cols=train_dataset.target_cols,
        feature_scaler=train_dataset.feature_scaler,
        target_scaler=train_dataset.target_scaler,
        fit_scalers=False,
        seq_len=seq_len,
    )
    test_dataset = HydrologyDataset(
        test_df,
        feature_cols=train_dataset.feature_cols,
        target_cols=train_dataset.target_cols,
        feature_scaler=train_dataset.feature_scaler,
        target_scaler=train_dataset.target_scaler,
        fit_scalers=False,
        seq_len=seq_len,
    )

    use_persistent = persistent_workers if num_worker > 0 else False

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        num_workers=num_worker,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=use_persistent,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=eval_shuffle,
        num_workers=num_worker,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=use_persistent,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=test_shuffle,
        num_workers=num_worker,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=use_persistent,
    )
    return train_dataloader, eval_dataloader, test_dataloader

def train_xgb(model: XGBoost, train_df: pd.DataFrame, eval_df: pd.DataFrame) -> xgb.XGBRegressor:
    model.train(train_df, eval_df)
    return model


def train_lgbm(model: LightGBM, train_df: pd.DataFrame, eval_df: pd.DataFrame) -> LightGBM:
    model.train(train_df, eval_df)
    return model


def evaluate_tabular(model: XGBoost | LightGBM, test_df: pd.DataFrame) -> pd.DataFrame:
    preds = model.predict(test_df)
    results = []
    timestamps = test_df["timestamp_utc"] if "timestamp_utc" in test_df.columns else None
    for target in model.target_columns:
        m = evaluate_all(test_df[target], preds[target], timestamps=timestamps)
        m["target"] = target
        results.append(m)
    summary = pd.DataFrame(results).set_index("target")
    cols = [
        "rmse", "mae", "nse", "kge", 
        "flood_f1", "flood_precision", "flood_recall", "flood_accuracy",
        "peak_absolute_error", "peak_timing_error_minutes"
    ]
    print("\n" + "=" * 45 + " TEST EVALUATION " + "=" * 45)
    print(summary[[c for c in cols if c in summary.columns]].to_string())
    print("=" * 107 + "\n")
    return summary
    

def train_lstm(
    model: LSTM | MeanEmbeddingForecastLSTMWithAdapter,
    horizon: str,
    num_epochs: int,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    test_dataloader: DataLoader,
    loss_func: nn.Module,
    optim: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler | None,
    early_stopping: int,
    gradient_accumulation: int = 1,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    model.train().to(device)
    losses = []
    val_losses = []
    best_val_loss = float("inf")
    save_dir = f"./model/{horizon}"
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, f"{model.__class__.__name__}_best.pt")

    for epoch in tqdm(range(num_epochs), desc="Training LSTM"):
        model.train()
        epoch_losses = []
        for batch_num, (xb, yb) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)):
            xb, yb = xb.to(device), yb.to(device)
            if batch_num % gradient_accumulation == 0:
                optim.zero_grad()
            
            y_pred = model(xb)
            loss = loss_func(y_pred, yb)
            loss_val = loss.item()
            epoch_losses.append(loss_val)

            (loss / gradient_accumulation).backward()

            if (batch_num + 1) % gradient_accumulation == 0 or (batch_num + 1) == len(train_dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optim.step()
        
        if scheduler is not None:
            scheduler.step()

        avg_train_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        losses.append(avg_train_loss)

        val_loss = 0.0
        model.eval()
        with torch.inference_mode():
            for xb, yb in eval_dataloader:
                xb, yb = xb.to(device), yb.to(device)
                y_pred = model(xb)
                val_loss += loss_func(y_pred, yb).item()
        val_loss /= max(1, len(eval_dataloader))
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"New best val loss: {val_loss:.4f} (saving to {best_model_path})")
            torch.save(model.state_dict(), best_model_path)

        if early_stopping and len(val_losses) > early_stopping:
            best_past = min(val_losses[:-early_stopping])
            recent_best = min(val_losses[-early_stopping:])
            if recent_best >= best_past:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    if os.path.exists(best_model_path):
        print(f"\nLoading best checkpoint from {best_model_path} for testing...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    model.eval()
    test_losses = []
    all_preds = []
    all_targets = []
    with torch.inference_mode():
        for xb, yb in tqdm(test_dataloader, desc="Testing LSTM"):
            xb, yb = xb.to(device), yb.to(device)
            y_pred = model(xb)
            test_loss = loss_func(y_pred, yb).item()
            test_losses.append(test_loss)
            all_preds.append(y_pred.detach().cpu())
            all_targets.append(yb.detach().cpu())
    avg_test_loss = sum(test_losses) / max(1, len(test_losses))
    print(f"Test Loss (MSE): {avg_test_loss:.4f}")

    if all_preds and hasattr(test_dataloader.dataset, "target_cols"):
        preds_scaled = torch.cat(all_preds, dim=0).numpy()
        targets_scaled = torch.cat(all_targets, dim=0).numpy()

        target_scaler = getattr(test_dataloader.dataset, "target_scaler", None)
        if target_scaler is not None:
            preds_arr = target_scaler.inverse_transform(preds_scaled)
            targets_arr = target_scaler.inverse_transform(targets_scaled)
        else:
            preds_arr = preds_scaled
            targets_arr = targets_scaled

        target_cols = test_dataloader.dataset.target_cols
        timestamps = getattr(test_dataloader.dataset, "timestamps", None)

        results = []
        for i, target_name in enumerate(target_cols):
            y_true = pd.Series(targets_arr[:, i])
            y_pred = pd.Series(preds_arr[:, i])
            ts = timestamps.iloc[:len(y_true)] if timestamps is not None else None
            m = evaluate_all(y_true, y_pred, timestamps=ts)
            m["target"] = target_name
            results.append(m)

        summary = pd.DataFrame(results).set_index("target")
        cols = [
            "rmse", "mae", "nse", "kge", 
            "flood_f1", "flood_precision", "flood_recall", "flood_accuracy",
            "peak_absolute_error", "peak_timing_error_minutes"
        ]
        print("\n" + "=" * 45 + " TEST EVALUATION " + "=" * 45)
        print(summary[[c for c in cols if c in summary.columns]].to_string())
        print("=" * 107 + "\n")

    return model


if __name__ == "__main__":
    args = parser.parse_args()

    if args.train_method == "xgb":
        train_df, eval_df, test_df = load_dataset(args.horizon)
        model = XGBoost()
        train_xgb(model, train_df, eval_df)
        evaluate_tabular(model, test_df)
    elif args.train_method == "lgbm":
        train_df, eval_df, test_df = load_dataset(args.horizon)
        model = LightGBM()
        train_lgbm(model, train_df, eval_df)
        evaluate_tabular(model, test_df)
    else:
        if args.lstm_seq_len is None:
            if args.horizon == "10min":
                seq_len = 144
            elif args.horizon == "30min":
                seq_len = 48
            elif args.horizon == "1h":
                seq_len = 24
        else:
            seq_len = args.lstm_seq_len

        train_dataloader, eval_dataloader, test_dataloader = get_dataloader(
            args.horizon,
            args.batch_size,
            args.num_worker,
            pin_memory=True,
            drop_last=True,
            train_shuffle=True,
            eval_shuffle=False,
            test_shuffle=False,
            persistent_workers=False,
            seq_len=seq_len
        )
        input_size = getattr(train_dataloader.dataset, "num_features", args.local_hindcast_size)
        output_size = getattr(train_dataloader.dataset, "num_targets", args.lstm_output_size)

        if args.train_method == "lstm":
            model = LSTM(
                input_size=input_size,
                hidden_size=args.lstm_hidden_size,
                num_layers=args.lstm_num_layers,
                dropout=args.lstm_drop_out,
                output_size=output_size,
            )
            optimizer = optim.AdamW(model.parameters(), lr=args.lstm_lr, weight_decay=args.lstm_weight_decay)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
            train_lstm(
                model,
                args.horizon,
                args.num_epochs,
                train_dataloader,
                eval_dataloader,
                test_dataloader,
                nn.MSELoss(),
                optimizer,
                scheduler,
                args.early_stopping,
                args.gradient_accumulation,
                args.device,
            )
        elif args.train_method == "lstm_adapter":
            model = MeanEmbeddingForecastLSTMWithAdapter(
                MeanEmbeddingForecastLSTM.from_pretrained(args.pretrained_model),
                input_size,
                args.pretrained_hindcast_size,
                input_size,
                args.pretrained_forecast_size,
                output_size,
            )
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
            train_lstm(
                model,
                args.horizon,
                args.num_epochs,
                train_dataloader,
                eval_dataloader,
                test_dataloader,
                nn.MSELoss(),
                optimizer,
                scheduler,
                args.early_stopping,
                args.gradient_accumulation,
                args.device,
            )

    