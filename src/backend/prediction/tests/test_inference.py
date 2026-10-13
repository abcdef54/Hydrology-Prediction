"""Inference contract checks using small synthetic artifacts, without a database."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

from src.ai.lstm import ResidualLSTM
from src.backend.prediction.config import StationModelConfig
from src.backend.prediction.inference import ResidualLSTMRuntime
from src.backend.prediction.inputs import PreparedInput


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config = StationModelConfig(
            station_id="water-a", model_type="residual_lstm", model_version="model-a",
            model_path=Path(self.directory.name) / "model.pt",
            preprocessing_path=Path(self.directory.name) / "preprocessing.joblib",
            sequence_length=4, rain_station_ids=("rain-a",),
            forecast_horizons_minutes=(60, 180), feature_schema_version="test-v1",
            model_parameters={"hidden_size": 3, "num_layers": 2, "dropout": 0.5,
                              "input_dropout": 0.2, "head_dropout": 0.2},
        )
        features = len(self.config.feature_columns)
        feature_scaler = StandardScaler().fit([np.zeros(features), np.full(features, 2.0)])
        target_scaler = StandardScaler().fit([[10.0, 20.0], [14.0, 26.0]])
        self.saved = {
            "feature_scaler": feature_scaler, "target_scaler": target_scaler,
            "feature_columns": list(self.config.feature_columns),
            "target_columns": ["target_water_level_plus_60m", "target_water_level_plus_180m"],
            "sequence_length": 4, "missing_scaled_value": 0.0,
        }
        joblib.dump(self.saved, self.config.preprocessing_path)
        model = ResidualLSTM(
            input_size=features, output_size=2, level_feature_index=0,
            persistence_scale=(feature_scaler.scale_[0] / target_scaler.scale_).tolist(),
            persistence_offset=((feature_scaler.mean_[0] - target_scaler.mean_) /
                                target_scaler.scale_).tolist(),
            **self.config.model_parameters,
        )
        self.state = model.state_dict()
        torch.save(self.state, self.config.model_path)

    def prepared_input(self, runtime):
        timestamps = pd.date_range("2026-09-01T00:00Z", periods=4, freq="h")
        features = pd.DataFrame(1.0, index=timestamps, columns=self.config.feature_columns)
        features["water_level_depth"] = 123.0
        raw, tensor = runtime.preprocessor.prepare_features(features)
        return PreparedInput(
            station_id=self.config.station_id, model_version=self.config.model_version,
            feature_schema_version=self.config.feature_schema_version, tensor=tensor,
            raw_features=raw, timestamps=timestamps,
            requested_reference_time=(timestamps[-1] + pd.Timedelta(hours=5)).to_pydatetime(),
            hourly_reference_time=(timestamps[-1] + pd.Timedelta(hours=5)).to_pydatetime(),
            anchor_observed_at=timestamps[-1].to_pydatetime(),
        )

    def test_persistence_is_inverse_scaled_to_mm_and_uses_actual_anchor(self):
        runtime = ResidualLSTMRuntime(self.config)
        prepared = self.prepared_input(runtime)
        result = runtime.predict(prepared)
        self.assertEqual(result.unit, "mm")
        self.assertEqual(result.station_id, "water-a")
        for point in result.predictions:
            self.assertAlmostEqual(point.water_level_mm, 123.0, places=4)
            self.assertEqual(point.forecast_at, prepared.anchor_observed_at +
                             pd.Timedelta(minutes=point.horizon_minutes))

    def test_dropout_and_autograd_are_disabled_and_prediction_is_repeatable(self):
        # A nonzero head ensures this test exercises the recurrent correction too.
        self.state["linear.weight"].fill_(0.1)
        torch.save(self.state, self.config.model_path)
        runtime = ResidualLSTMRuntime(self.config)
        prepared = self.prepared_input(runtime)
        observed = []
        handle = runtime.model.register_forward_hook(
            lambda module, inputs, output: observed.append(
                (torch.is_grad_enabled(), torch.is_inference_mode_enabled(), output.requires_grad)
            )
        )
        self.addCleanup(handle.remove)
        first = runtime.predict(prepared)
        second = runtime.predict(prepared)
        self.assertEqual(first, second)
        self.assertEqual(observed, [(False, True, False)] * 2)
        self.assertTrue(all(not module.training for module in runtime.model.modules()))
        self.assertTrue(all(not parameter.requires_grad for parameter in runtime.model.parameters()))

    def test_wrong_station_model_schema_or_tensor_is_rejected(self):
        runtime = ResidualLSTMRuntime(self.config)
        prepared = self.prepared_input(runtime)
        for change in (
            {"station_id": "water-b"}, {"model_version": "model-b"},
            {"feature_schema_version": "other"}, {"tensor": prepared.tensor[:, :2]},
            {"tensor": torch.full_like(prepared.tensor, float("nan"))},
            {"anchor_observed_at": prepared.requested_reference_time},
        ):
            with self.subTest(change=list(change)), self.assertRaises(ValueError):
                runtime.predict(replace(prepared, **change))

    def test_wrong_target_order_is_rejected(self):
        self.saved["target_columns"].reverse()
        joblib.dump(self.saved, self.config.preprocessing_path)
        with self.assertRaisesRegex(ValueError, "target order"):
            ResidualLSTMRuntime(self.config)

    def test_checkpoint_from_different_scalers_is_rejected(self):
        self.state["persistence_scale"] *= 2
        torch.save(self.state, self.config.model_path)
        with self.assertRaisesRegex(ValueError, "coefficients"):
            ResidualLSTMRuntime(self.config)

    def test_unsupported_model_type_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            ResidualLSTMRuntime(replace(self.config, model_type="xgb"))


if __name__ == "__main__":
    unittest.main()
