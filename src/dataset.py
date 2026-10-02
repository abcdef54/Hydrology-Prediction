from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class HydrologyDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str] | None = None,
        target_cols: list[str] | None = None,
        feature_scaler: StandardScaler | None = None,
        target_scaler: StandardScaler | None = None,
        fit_scalers: bool = False,
        seq_len: int = 24,
    ) -> None:
        super().__init__()

        self.seq_len = seq_len

        self.target_cols = target_cols or [
            column
            for column in df.columns
            if column.startswith("target_")
        ]

        self.feature_cols = feature_cols or [
            column
            for column in df.columns
            if column not in self.target_cols
            and not column.startswith("timestamp")
        ]

        X_raw = df[self.feature_cols].to_numpy(dtype=np.float32)
        y_raw = df[self.target_cols].to_numpy(dtype=np.float32)

        if fit_scalers:
            self.feature_scaler = StandardScaler()
            self.feature_scaler.fit(X_raw)

            valid_targets = np.isfinite(y_raw).all(axis=1)

            self.target_scaler = StandardScaler()
            self.target_scaler.fit(y_raw[valid_targets])

        else:
            if feature_scaler is None or target_scaler is None:
                raise ValueError(
                    "feature_scaler and target_scaler are required "
                    "when fit_scalers=False."
                )

            self.feature_scaler = feature_scaler
            self.target_scaler = target_scaler

        X_scaled = self.feature_scaler.transform(X_raw)
        y_scaled = self.target_scaler.transform(y_raw)

        # StandardScaler preserves NaN.
        # LSTM cannot consume NaN, so replace missing normalized values
        # with 0. Missing flags in the dataset tell the model they were absent.
        self.X = np.nan_to_num(
            X_scaled,
            nan=0.0,
        ).astype(np.float32)

        self.y = y_scaled.astype(np.float32)

        valid_target_rows = np.isfinite(self.y).all(axis=1)

        possible_end_indices = np.arange(len(df))
        valid_sequence_rows = possible_end_indices >= self.seq_len - 1

        self.end_indices = possible_end_indices[
            valid_target_rows & valid_sequence_rows
        ]

        if "timestamp_utc" in df.columns:
            self.timestamps = (
                df["timestamp_utc"]
                .iloc[self.end_indices]
                .reset_index(drop=True)
            )
        else:
            self.timestamps = None

    @property
    def num_features(self) -> int:
        return self.X.shape[1]

    @property
    def num_targets(self) -> int:
        return self.y.shape[1]

    def __len__(self) -> int:
        return len(self.end_indices)

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        end_idx = self.end_indices[idx]
        start_idx = end_idx - self.seq_len + 1

        features = self.X[start_idx:end_idx + 1]
        targets = self.y[end_idx]

        return (
            torch.from_numpy(features),
            torch.from_numpy(targets),
        )