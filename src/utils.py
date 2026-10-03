import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from itertools import combinations
from src.dataset import HydrologyDataset
from googlehydrology.modelzoo.mean_embedding_forecast_lstm import MeanEmbeddingForecastLSTM
from googlehydrology.modelzoo.mean_embedding_forecast_lstm import Config


DATASET_PATH_10MIN = "./train_data/10min/"
DATASET_PATH_30MIN = "./train_data/30min/"
DATASET_PATH_1H = "./train_data/1h/"


FEATURE_GROUPS = ("water_level", "rain", "wind", "reservoir")
FEATURE_SETS = ["all"] + [
    "+".join(groups)
    for size in range(1, len(FEATURE_GROUPS) + 1)
    for groups in combinations(FEATURE_GROUPS, size)
]


def set_seed(seed: int) -> None:
    if not 0 <= seed < 2**32:
        raise ValueError("--seed must be between 0 and 2**32 - 1")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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


def select_feature_columns(columns, feature_set: str) -> list[str] | None:
    if feature_set == "all":
        return None

    groups = feature_set.split("+")
    selected = [
        column for column in columns
        if any(column.startswith(f"{group}_") for group in groups)
    ]
    for group in groups:
        if not any(column.startswith(f"{group}_") for column in selected):
            raise ValueError(f"No {group} features found for {feature_set}")
    return selected


def load_mef_backbone(config_path: str, checkpoint_path: str) -> MeanEmbeddingForecastLSTM:
    config_file = Path(config_path)
    checkpoint_file = Path(checkpoint_path)
    if not config_file.is_file() or not checkpoint_file.is_file():
        raise FileNotFoundError("MEF config and checkpoint must both be existing files")
    backbone = MeanEmbeddingForecastLSTM(Config(config_file))
    state_dict = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    backbone.load_state_dict(state_dict)
    return backbone


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
    seed: int = 42,
    require_current_level: bool = False,
    feature_set: str = "all",
):
    train_df, eval_df, test_df = load_dataset(horizon)
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown LSTM feature set: {feature_set}")
    feature_cols = select_feature_columns(train_df.columns, feature_set)
    train_dataset = HydrologyDataset(
        train_df, feature_cols=feature_cols, fit_scalers=True, seq_len=seq_len,
        require_current_level=require_current_level,
    )
    train_dataset.feature_set = feature_set
    eval_dataset = HydrologyDataset(
        eval_df,
        feature_cols=train_dataset.feature_cols,
        target_cols=train_dataset.target_cols,
        feature_scaler=train_dataset.feature_scaler,
        target_scaler=train_dataset.target_scaler,
        fit_scalers=False,
        seq_len=seq_len,
        require_current_level=require_current_level,
    )
    test_dataset = HydrologyDataset(
        test_df,
        feature_cols=train_dataset.feature_cols,
        target_cols=train_dataset.target_cols,
        feature_scaler=train_dataset.feature_scaler,
        target_scaler=train_dataset.target_scaler,
        fit_scalers=False,
        seq_len=seq_len,
        require_current_level=require_current_level,
    )

    use_persistent = persistent_workers if num_worker > 0 else False
    train_generator = torch.Generator().manual_seed(seed)
    eval_generator = torch.Generator().manual_seed(seed + 1)
    test_generator = torch.Generator().manual_seed(seed + 2)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        num_workers=num_worker,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=use_persistent,
        worker_init_fn=seed_worker,
        generator=train_generator,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=eval_shuffle,
        num_workers=num_worker,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=use_persistent,
        worker_init_fn=seed_worker,
        generator=eval_generator,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=test_shuffle,
        num_workers=num_worker,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=use_persistent,
        worker_init_fn=seed_worker,
        generator=test_generator,
    )
    return train_dataloader, eval_dataloader, test_dataloader