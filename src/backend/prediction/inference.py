"""Station-specific, evaluation-only Residual LSTM forecasting in millimeters."""

from collections.abc import Mapping
from datetime import datetime, timedelta
from io import BytesIO
import hashlib

import joblib
import numpy as np
from sqlalchemy.engine import Engine
import torch

from src.ai.lstm import ResidualLSTM
from src.backend.prediction.config import StationModelConfig
from src.backend.prediction.inputs import PreparedInput, PredictionInputService
from src.backend.prediction.preprocessing import InputPreprocessor
from src.backend.prediction.results import ForecastPoint, ForecastResult
from src.backend.prediction.persistence import (
    build_pipeline_version,
    build_input_hash,
    load_forecast,
    save_forecast,
)


class ResidualLSTMRuntime:
    """Load once per configured station without an optimizer or training loss.

    Args:
        config: This station's checkpoint, scaler, architecture and feature contract.
        device: PyTorch device for inference; defaults to CPU.

    Raises:
        OSError: If an artifact or pipeline source file cannot be read.
        ValueError: If architecture, metadata or scalers violate the model contract.
        RuntimeError: If weights cannot be loaded or the device cannot be used.
    """

    def __init__(self, config: StationModelConfig, device: str = "cpu"):
        if config.model_type != "residual_lstm":
            raise ValueError(f"Unsupported inference model type: {config.model_type}")
        self.config = config
        self.device = torch.device(device)
        # Hash and load the same bytes, so saved results identify the loaded artifacts.
        model_bytes = config.model_path.read_bytes()
        preprocessing_bytes = config.preprocessing_path.read_bytes()
        self.artifact_hashes = (
            hashlib.sha256(model_bytes).hexdigest(),
            hashlib.sha256(preprocessing_bytes).hexdigest(),
        )
        saved = joblib.load(BytesIO(preprocessing_bytes))
        self.preprocessor = InputPreprocessor.from_saved(config, saved)
        self.pipeline_version = build_pipeline_version(
            config, self.artifact_hashes, str(self.device)
        )
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
            "hidden_size",
            "num_layers",
            "dropout",
            "input_dropout",
            "head_dropout",
        }
        if set(config.model_parameters) != required_parameters:
            raise ValueError(
                "Residual LSTM configuration must specify all architecture parameters."
            )
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
        state = torch.load(BytesIO(model_bytes), map_location="cpu", weights_only=True)
        for name in ("persistence_scale", "persistence_offset"):
            expected = getattr(model, name)
            actual = state[name]
            if actual.shape != expected.shape or not torch.allclose(actual, expected):
                raise ValueError(
                    "Checkpoint persistence coefficients do not match the saved scalers."
                )
        model.load_state_dict(state, strict=True)
        if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
            raise ValueError("Checkpoint contains non-finite weights.")
        self.model = model.to(self.device)
        self.model.requires_grad_(False)
        self.model.eval()

    def artifacts_match_config(self) -> bool:
        """Check whether configured files still match the loaded artifact snapshot.

        Returns:
            True when both checkpoint and preprocessing SHA-256 digests match
            the bytes loaded by this runtime; False when either has changed.

        Raises:
            OSError: If either configured artifact cannot be read.
        """
        current_hashes = tuple(
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (self.config.model_path, self.config.preprocessing_path)
        )
        return current_hashes == self.artifact_hashes

    def predict(self, prepared: PreparedInput) -> ForecastResult:
        """Run evaluation-only inference and inverse-transform the output to mm.

        Args:
            prepared: Validated input chain belonging to this station/model, with
                a tensor shaped (1, sequence_length, feature_count) and timestamps.

        Returns:
            Forecast for each configured horizon, dated from the actual input
            anchor. No storage metadata is attached by this runtime method.

        Raises:
            ValueError: If input identity, shape, timestamps or output are invalid.
            RuntimeError: If PyTorch inference or device operations fail.
        """
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
    """Prepare station inputs and reuse or persist original-observation forecasts.

    Args:
        engine: PostgreSQL engine for reading inputs and storing forecast results.
        runtime: Loaded station-specific model and its saved preprocessing contract.
    """

    def __init__(self, engine: Engine, runtime: ResidualLSTMRuntime):
        self.engine = engine
        self.runtime = runtime
        self.inputs = PredictionInputService(engine, runtime.preprocessor)

    def predict(
        self,
        reference_time: datetime | None = None,
        overrides: Mapping[str, float | None] | None = None,
    ) -> ForecastResult:
        """Predict from original observations or an edited last-timestep scenario.

        Args:
            reference_time: Timezone-aware historical instant, rounded down to the
                hour by input preparation; None selects the latest usable hour.
            overrides: Optional feature-name/value edits in original units applied
                only to the last timestep. Any nonempty mapping bypasses storage.

        Returns:
            Forecast values in millimetres and current request context. Original
            inputs receive storage metadata; edited scenarios are not persisted.

        Raises:
            InsufficientHistoryError: If fewer valid timesteps than required exist.
            InvalidOverrideError: If an edit violates the preprocessing contract.
            ValueError: If inputs, model output or stored results are invalid.
            RuntimeError: If model execution fails.
            sqlalchemy.exc.SQLAlchemyError: If input reads or forecast storage fail.
        """
        prepared = self.inputs.prepare(reference_time, overrides)
        # Any nonempty override request is a scenario, even if values equal observations.
        if overrides:
            return self.runtime.predict(prepared)
        input_hash = build_input_hash(prepared, self.runtime.config)
        with self.engine.connect() as connection:
            stored = load_forecast(
                connection, self.runtime.config, prepared, self.runtime.pipeline_version, input_hash
            )
        if stored is not None:
            return stored
        forecast = self.runtime.predict(prepared)
        return save_forecast(
            self.engine,
            self.runtime.config,
            prepared,
            self.runtime.pipeline_version,
            input_hash,
            forecast,
        )
