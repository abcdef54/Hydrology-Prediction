"""Apply the trusted saved training scaler; never fit a scaler at serving time."""

from collections.abc import Mapping

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.backend.prediction.config import StationModelConfig


class InvalidOverrideError(ValueError):
    """User-supplied feature cannot be represented by this model's input contract."""


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
        """Load the trusted preprocessing artifact belonging to this model.

        Args:
            config: Station/model contract, including preprocessing_path.

        Returns:
            Preprocessor using the saved feature scaler without fitting it again.

        Raises:
            OSError: If the artifact cannot be read.
            KeyError: If required preprocessing metadata is absent.
            ValueError: If metadata or scaler statistics violate the input contract.
        """
        return cls.from_saved(config, joblib.load(config.preprocessing_path))

    @classmethod
    def from_saved(cls, config: StationModelConfig, saved: dict) -> "InputPreprocessor":
        """Validate a loaded artifact snapshot against this station's input contract.

        Args:
            config: Expected station/model feature order and sequence length.
            saved: Trusted joblib payload containing the fitted feature scaler,
                feature columns, sequence length and missing-value policy.

        Returns:
            Preprocessor that uses the supplied saved feature scaler.

        Raises:
            KeyError: If required metadata is absent.
            ValueError: If feature order, sequence length, missing-value policy
                or scaler statistics do not match the serving contract.
        """
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

    def validate_overrides(self, overrides: Mapping[str, float | None] | None) -> None:
        """Validate original-unit edits without changing observations or scaler state.

        Flags require numeric 0/1; missing rain/lag values may be None, but the
        current water level is required by the persistence branch. Raises
        InvalidOverrideError for invalid names, types or unrepresentable values.
        """
        for name, value in (overrides or {}).items():
            if name not in self.feature_columns:
                raise InvalidOverrideError(f"Unknown feature override: {name}")
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
            ):
                raise InvalidOverrideError(f"Override must be a finite number or None: {name}")
            if (name.endswith("_missing") or name == "flood_season") and value not in (0, 1):
                raise InvalidOverrideError(f"Flag override must be 0 or 1: {name}")
            if name == "water_level_depth" and value is None:
                raise InvalidOverrideError("Current water level cannot be missing.")
            if value is not None and abs(value) > float(np.finfo(np.float32).max):
                raise InvalidOverrideError(f"Override exceeds float32 range: {name}")

    def prepare_features(
        self, features: pd.DataFrame, overrides: Mapping[str, float | None] | None = None
    ) -> tuple[pd.DataFrame, torch.Tensor]:
        if tuple(features.columns) != self.feature_columns:
            raise ValueError("Feature names or order do not match the saved scaler.")
        if len(features) != self.sequence_length:
            raise ValueError(f"Expected {self.sequence_length} input timesteps.")
        self.validate_overrides(overrides)
        prepared = features.copy()
        for name, value in (overrides or {}).items():
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
