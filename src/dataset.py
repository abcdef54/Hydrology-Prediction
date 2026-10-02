from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class HydrologyDataset(Dataset):
    """PyTorch Dataset for Hydrological time series with feature scaling and NaN handling."""

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str] | None = None,
        target_cols: list[str] | None = None,
        scaler: StandardScaler | None = None,
        fit_scaler: bool = False,
        seq_len: int = 1,
    ) -> None:
        super().__init__()
        self.seq_len = max(1, seq_len)

        if target_cols is None:
            self.target_cols = [c for c in df.columns if c.startswith("target_")]
        else:
            self.target_cols = target_cols

        if feature_cols is None:
            self.feature_cols = [
                c for c in df.columns
                if c not in self.target_cols and not c.startswith("timestamp")
            ]
        else:
            self.feature_cols = feature_cols

        features = df[self.feature_cols].copy()
        targets = df[self.target_cols].copy()

        # Handle missing values (forward fill, back fill, and fill remainder with 0)
        features = features.ffill().bfill().fillna(0.0)
        targets = targets.ffill().bfill().fillna(0.0)

        X_raw = features.to_numpy(dtype=np.float32).copy()
        y_raw = targets.to_numpy(dtype=np.float32).copy()

        if fit_scaler or scaler is None:
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(X_raw).astype(np.float32).copy()
        else:
            self.scaler = scaler
            self.X = self.scaler.transform(X_raw).astype(np.float32).copy()

        self.y = y_raw
        self.timestamps = df["timestamp_utc"].reset_index(drop=True) if "timestamp_utc" in df.columns else None

    @property
    def num_features(self) -> int:
        return self.X.shape[1]

    @property
    def num_targets(self) -> int:
        return self.y.shape[1]

    def __len__(self) -> int:
        return len(self.X) - self.seq_len + 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start_idx = idx
        end_idx = idx + self.seq_len
        xb = torch.from_numpy(self.X[start_idx:end_idx])  # Shape: (seq_len, num_features)
        yb = torch.from_numpy(self.y[end_idx - 1])        # Target at sequence end
        return xb, yb
