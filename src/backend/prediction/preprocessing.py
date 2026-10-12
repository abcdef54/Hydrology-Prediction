"""Apply the trusted saved training scaler; never fit a scaler at serving time."""

from collections.abc import Mapping

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.backend.prediction.config import StationModelConfig


class InputPreprocessor:
    def __init__(self, feature_scaler: StandardScaler, config: StationModelConfig):
        self.config = config
        if getattr(feature_scaler, "n_features_in_", None) != len(config.feature_columns):
            raise ValueError("Saved scaler does not match the feature schema.")
        if (
            not np.isfinite(feature_scaler.mean_).all()
            or not np.isfinite(feature_scaler.scale_).all()
        ):
            raise ValueError("Saved scaler contains invalid statistics.")
        if (feature_scaler.scale_ <= 0).any():
            raise ValueError("Saved scaler scales must be positive.")
        self.feature_scaler = feature_scaler
        self.feature_columns = config.feature_columns
        self.sequence_length = config.sequence_length

    @classmethod
    def from_file(cls, config: StationModelConfig) -> "InputPreprocessor":
        """Load the preprocessing.joblib belonging to the verified checkpoint."""
        saved = joblib.load(config.preprocessing_path)
        if saved.get("missing_scaled_value") != 0.0:
            raise ValueError("Expected missing normalized values to be replaced with zero.")
        if tuple(saved["feature_columns"]) != config.feature_columns:
            raise ValueError(
                "Saved feature order does not match this station's model configuration."
            )
        if saved["sequence_length"] != config.sequence_length:
            raise ValueError(
                "Saved sequence length does not match this station's model configuration."
            )
        return cls(saved["feature_scaler"], config)

    def prepare_features(
        self, features: pd.DataFrame, overrides: Mapping[str, float | None] | None = None
    ) -> tuple[pd.DataFrame, torch.Tensor]:
        if tuple(features.columns) != self.feature_columns:
            raise ValueError("Feature names or order do not match the saved scaler.")
        if len(features) != self.sequence_length:
            raise ValueError(f"Expected {self.sequence_length} input timesteps.")
        prepared = features.copy()
        for name, value in (overrides or {}).items():
            if name not in self.feature_columns:
                raise ValueError(f"Unknown feature override: {name}")
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
            ):
                raise ValueError(f"Override must be a finite number or None: {name}")
            if (name.endswith("_missing") or name == "flood_season") and value not in (0, 1):
                raise ValueError(f"Flag override must be 0 or 1: {name}")
            # Integer flag columns stay integer; other columns already use float.
            prepared.loc[prepared.index[-1], name] = np.nan if value is None else value
        raw = prepared.to_numpy(dtype=np.float32)
        if np.isinf(raw).any():
            raise ValueError("Input features cannot contain infinity.")
        if not np.isfinite(raw[-1, 0]):
            raise ValueError("Current water level is required for the persistence branch.")
        scaled = self.feature_scaler.transform(raw)
        if np.isinf(scaled).any():
            raise ValueError("Feature scaling overflowed.")
        scaled = np.nan_to_num(scaled, nan=0.0).astype(np.float32)
        if not np.isfinite(scaled).all():
            raise ValueError("Normalized input contains invalid values.")
        return prepared, torch.from_numpy(scaled).unsqueeze(0)
