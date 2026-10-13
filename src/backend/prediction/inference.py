"""Station-specific, evaluation-only Residual LSTM forecasting in millimeters."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta

import joblib
import numpy as np
from sqlalchemy.engine import Engine
import torch

from src.ai.lstm import ResidualLSTM
from src.backend.prediction.config import StationModelConfig
from src.backend.prediction.inputs import PreparedInput, PredictionInputService
from src.backend.prediction.preprocessing import InputPreprocessor


@dataclass(frozen=True)
class ForecastPoint:
    horizon_minutes: int
    forecast_at: datetime
    water_level_mm: float


@dataclass(frozen=True)
class ForecastResult:
    station_id: str
    model_version: str
    feature_schema_version: str
    requested_reference_time: datetime | None
    hourly_reference_time: datetime
    anchor_observed_at: datetime
    predictions: tuple[ForecastPoint, ...]
    unit: str = "mm"


class ResidualLSTMRuntime:
    """Load once per configured station; reuse without an optimizer or training loss."""

    def __init__(self, config: StationModelConfig, device: str = "cpu"):
        if config.model_type != "residual_lstm":
            raise ValueError(f"Unsupported inference model type: {config.model_type}")
        self.config = config
        self.device = torch.device(device)
        self.preprocessor = InputPreprocessor.from_file(config)
        saved = joblib.load(config.preprocessing_path)
        expected_targets = [
            f"target_water_level_plus_{minutes}m" for minutes in config.forecast_horizons_minutes
        ]
        if not expected_targets or list(saved["target_columns"]) != expected_targets:
            raise ValueError("Saved target order does not match configured forecast horizons.")
        self.target_scaler = saved["target_scaler"]
        output_size = len(expected_targets)
        if (
            self.target_scaler.n_features_in_ != output_size
            or not np.isfinite(self.target_scaler.mean_).all()
            or not np.isfinite(self.target_scaler.scale_).all()
            or (self.target_scaler.scale_ <= 0).any()
        ):
            raise ValueError("Saved target scaler is invalid for this model.")
        required_parameters = {
            "hidden_size", "num_layers", "dropout", "input_dropout", "head_dropout"
        }
        if set(config.model_parameters) != required_parameters:
            raise ValueError("Residual LSTM configuration must specify all architecture parameters.")
        level_index = config.feature_columns.index("water_level_depth")
        feature_scaler = self.preprocessor.feature_scaler
        persistence_scale = feature_scaler.scale_[level_index] / self.target_scaler.scale_
        persistence_offset = (
            feature_scaler.mean_[level_index] - self.target_scaler.mean_
        ) / self.target_scaler.scale_
        model = ResidualLSTM(
            input_size=len(config.feature_columns),
            output_size=output_size,
            level_feature_index=level_index,
            persistence_scale=persistence_scale.tolist(),
            persistence_offset=persistence_offset.tolist(),
            **config.model_parameters,
        )
        state = torch.load(config.model_path, map_location="cpu", weights_only=True)
        for name in ("persistence_scale", "persistence_offset"):
            expected = getattr(model, name)
            actual = state[name]
            if actual.shape != expected.shape or not torch.allclose(actual, expected):
                raise ValueError("Checkpoint persistence coefficients do not match the saved scalers.")
        model.load_state_dict(state, strict=True)
        if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
            raise ValueError("Checkpoint contains non-finite weights.")
        self.model = model.to(self.device)
        self.model.requires_grad_(False)
        self.model.eval()

    def predict(self, prepared: PreparedInput) -> ForecastResult:
        if (
            prepared.station_id != self.config.station_id
            or prepared.model_version != self.config.model_version
            or prepared.feature_schema_version != self.config.feature_schema_version
        ):
            raise ValueError("Prepared input belongs to a different station/model configuration.")
        expected_shape = (1, self.config.sequence_length, len(self.config.feature_columns))
        if prepared.tensor.shape != expected_shape or not torch.isfinite(prepared.tensor).all():
            raise ValueError(f"Expected finite input with shape {expected_shape}.")
        if (
            len(prepared.timestamps) != self.config.sequence_length
            or prepared.timestamps[-1].to_pydatetime() != prepared.anchor_observed_at
            or prepared.anchor_observed_at.tzinfo is None
        ):
            raise ValueError("Input timestamps must match the timezone-aware forecast anchor.")
        # eval disables input/head/recurrent dropout. Correction penalty exists
        # only in the training loss and is never part of this forward pass.
        self.model.eval()
        with torch.inference_mode():
            scaled = self.model(prepared.tensor.to(device=self.device, dtype=torch.float32))
        expected_output = (1, len(self.config.forecast_horizons_minutes))
        if scaled.shape != expected_output or not torch.isfinite(scaled).all():
            raise ValueError("Model returned invalid predictions.")
        predictions_mm = self.target_scaler.inverse_transform(scaled.cpu().numpy())[0]
        if not np.isfinite(predictions_mm).all():
            raise ValueError("Inverse scaling returned invalid water levels.")
        predictions = tuple(
            ForecastPoint(
                horizon_minutes=minutes,
                forecast_at=prepared.anchor_observed_at + timedelta(minutes=minutes),
                water_level_mm=float(value),
            )
            for minutes, value in zip(self.config.forecast_horizons_minutes, predictions_mm)
        )
        return ForecastResult(
            station_id=self.config.station_id,
            model_version=self.config.model_version,
            feature_schema_version=self.config.feature_schema_version,
            requested_reference_time=prepared.requested_reference_time,
            hourly_reference_time=prepared.hourly_reference_time,
            anchor_observed_at=prepared.anchor_observed_at,
            predictions=predictions,
        )


class PredictionService:
    def __init__(self, engine: Engine, runtime: ResidualLSTMRuntime):
        self.runtime = runtime
        self.inputs = PredictionInputService(engine, runtime.preprocessor)

    def predict(
        self,
        reference_time: datetime | None = None,
        overrides: Mapping[str, float | None] | None = None,
    ) -> ForecastResult:
        prepared = self.inputs.prepare(reference_time, overrides)
        return self.runtime.predict(prepared)
