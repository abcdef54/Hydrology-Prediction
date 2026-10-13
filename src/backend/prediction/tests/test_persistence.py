"""Forecast identities and persistence decisions without database access."""

from dataclasses import replace
from datetime import timedelta
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch

from src.backend.prediction.inference import PredictionService
from src.backend.prediction.inputs import PreparedInput
from src.backend.prediction.persistence import build_input_hash, build_pipeline_version
from src.backend.prediction.results import ForecastPoint, ForecastResult
from src.backend.prediction.tests.test_inputs import TEST_CONFIG


def persistence_example():
    config = replace(TEST_CONFIG, sequence_length=4)
    timestamps = pd.date_range("2026-09-13T00:00:00Z", periods=4, freq="h")
    raw = pd.DataFrame(1.0, index=timestamps, columns=config.feature_columns)
    prepared = PreparedInput(
        config.station_id,
        config.model_version,
        config.feature_schema_version,
        torch.ones(1, 4, len(config.feature_columns)),
        raw,
        timestamps,
        (timestamps[-1] + pd.Timedelta(minutes=30)).to_pydatetime(),
        timestamps[-1].to_pydatetime(),
        timestamps[-1].to_pydatetime(),
    )
    forecast = ForecastResult(
        config.station_id,
        config.model_version,
        config.feature_schema_version,
        prepared.requested_reference_time,
        prepared.hourly_reference_time,
        prepared.anchor_observed_at,
        tuple(
            ForecastPoint(horizon, prepared.anchor_observed_at + timedelta(minutes=horizon), 123.5)
            for horizon in config.forecast_horizons_minutes
        ),
    )
    return config, prepared, forecast


class ForecastIdentityTests(TestCase):
    def setUp(self):
        self.config, self.prepared, self.forecast = persistence_example()

    def test_different_request_times_with_same_chain_have_same_identity(self):
        changed = replace(
            self.prepared,
            requested_reference_time=None,
            hourly_reference_time=self.prepared.hourly_reference_time + timedelta(hours=1),
        )
        self.assertEqual(
            build_input_hash(self.prepared, self.config), build_input_hash(changed, self.config)
        )

    def test_changes_anywhere_in_raw_scaled_or_timestamp_chain_change_hash(self):
        original_hash = build_input_hash(self.prepared, self.config)
        raw = self.prepared.raw_features.copy()
        raw.iloc[0, 5] += 0.000000001
        scaled = self.prepared.tensor.clone()
        scaled[0, 0, 5] += 1
        timestamps = self.prepared.timestamps.copy()
        timestamps = pd.DatetimeIndex([timestamps[0] - pd.Timedelta(hours=1), *timestamps[1:]])
        shifted_raw = self.prepared.raw_features.copy()
        shifted_raw.index = timestamps
        for prepared in (
            replace(self.prepared, raw_features=raw),
            replace(self.prepared, tensor=scaled),
            replace(self.prepared, raw_features=shifted_raw, timestamps=timestamps),
        ):
            with self.subTest(prepared=prepared.timestamps[0]):
                self.assertNotEqual(original_hash, build_input_hash(prepared, self.config))

    def test_missing_is_distinct_from_zero_and_timezone_representation_is_equal(self):
        raw = self.prepared.raw_features.copy()
        raw.iloc[0, 5] = np.nan
        missing_hash = build_input_hash(replace(self.prepared, raw_features=raw), self.config)
        raw.iloc[0, 5] = 0
        self.assertNotEqual(
            missing_hash, build_input_hash(replace(self.prepared, raw_features=raw), self.config)
        )
        local = self.prepared.timestamps.tz_convert("Asia/Ho_Chi_Minh")
        local_raw = self.prepared.raw_features.copy()
        local_raw.index = local
        self.assertEqual(
            build_input_hash(self.prepared, self.config),
            build_input_hash(
                replace(self.prepared, raw_features=local_raw, timestamps=local), self.config
            ),
        )

    def test_pipeline_changes_with_scaler_weights_configuration_and_source(self):
        original = build_pipeline_version(self.config, ("weights", "scaler"), "cpu")
        for config, hashes in (
            (self.config, ("new-weights", "scaler")),
            (self.config, ("weights", "new-scaler")),
            (replace(self.config, calendar_timezone="UTC"), ("weights", "scaler")),
        ):
            self.assertNotEqual(original, build_pipeline_version(config, hashes, "cpu"))
        with patch(
            "src.backend.prediction.persistence.Path.read_bytes", return_value=b"changed code"
        ):
            self.assertNotEqual(
                original, build_pipeline_version(self.config, ("weights", "scaler"), "cpu")
            )

    def service(self):
        runtime = MagicMock(config=self.config, pipeline_version="pipeline-test")
        runtime.predict.return_value = self.forecast
        service = PredictionService(MagicMock(), runtime)
        service.inputs = MagicMock()
        service.inputs.prepare.return_value = self.prepared
        return service

    def test_stored_result_skips_model_execution(self):
        service = self.service()
        stored = replace(self.forecast, reused=True)
        with patch("src.backend.prediction.inference.load_forecast", return_value=stored), patch(
            "src.backend.prediction.inference.save_forecast"
        ) as save:
            self.assertEqual(service.predict(), stored)
        service.runtime.predict.assert_not_called()
        save.assert_not_called()

    def test_miss_runs_model_then_saves(self):
        service = self.service()
        with patch("src.backend.prediction.inference.load_forecast", return_value=None), patch(
            "src.backend.prediction.inference.save_forecast", return_value=self.forecast
        ) as save:
            service.predict()
        service.runtime.predict.assert_called_once_with(self.prepared)
        save.assert_called_once()

    def test_override_scenario_never_reads_or_writes_forecast_tables(self):
        service = self.service()
        with patch("src.backend.prediction.inference.load_forecast") as load, patch(
            "src.backend.prediction.inference.save_forecast"
        ) as save:
            service.predict(overrides={"water_level_depth": 123})
        load.assert_not_called()
        save.assert_not_called()
        service.runtime.predict.assert_called_once_with(self.prepared)


if __name__ == "__main__":
    main()
