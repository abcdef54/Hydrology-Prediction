"""Model isolation, forecast vintage selection, and hourly-grid serving contracts."""

from dataclasses import replace
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch

from src.backend.api.predictions import get_prediction_service
from src.backend.cache.inputs import ObservationDependentCache
from src.backend.prediction.config import ModelNotConfiguredError, StationModelRegistry
from src.backend.prediction.inference import ResidualLSTMRuntime
from src.backend.prediction.inputs import PreparedInput
from src.backend.prediction.tests.test_inputs import (
    TEST_CONFIG, example_observations, example_preprocessor,
)
from src.backend.prediction.weather import forecast_features


ROOT = Path(__file__).resolve().parents[4]


class ModelChoiceTests(TestCase):
    def test_three_models_share_station_but_have_separate_defaults_and_ids(self):
        b = replace(TEST_CONFIG, model_id="model_b", model_version="b")
        three = replace(TEST_CONFIG, model_id="three_station", model_version="three")
        registry = StationModelRegistry([TEST_CONFIG, b, three])
        self.assertEqual(registry.get(TEST_CONFIG.station_id), TEST_CONFIG)
        self.assertEqual(registry.get(TEST_CONFIG.station_id, "model_b"), b)
        self.assertEqual(len(registry.models_for(TEST_CONFIG.station_id)), 3)
        with self.assertRaises(ModelNotConfiguredError):
            registry.get(TEST_CONFIG.station_id, "unknown")
        with self.assertRaises(ValueError):
            StationModelRegistry([replace(TEST_CONFIG, is_default=True), replace(b, is_default=True)])
        with self.assertRaises(ValueError):
            StationModelRegistry([TEST_CONFIG, replace(b, model_version=TEST_CONFIG.model_version)])

    def test_runtime_cache_does_not_reuse_another_model_for_same_station(self):
        b = replace(TEST_CONFIG, model_id="model_b", model_version="b")
        state = SimpleNamespace(
            prediction_service_lock=Lock(), prediction_services={}, prediction_device="cpu",
            database_engine=MagicMock(), forecast_cache=MagicMock(), prediction_input_cache=MagicMock(),
        )
        request = SimpleNamespace(app=SimpleNamespace(state=state))
        runtimes = [MagicMock(config=configuration) for configuration in (TEST_CONFIG, b)]
        for runtime in runtimes:
            runtime.artifacts_match_config.return_value = True
        with patch("src.backend.api.predictions.ResidualLSTMRuntime", side_effect=runtimes) as loader, \
                patch("src.backend.api.predictions.CachedPredictionService",
                      side_effect=lambda engine, runtime, *args: MagicMock(runtime=runtime)):
            legacy = get_prediction_service(request, TEST_CONFIG)
            selected = get_prediction_service(request, b)
            self.assertIsNot(legacy, selected)
            self.assertIs(get_prediction_service(request, TEST_CONFIG), legacy)
            self.assertEqual(loader.call_count, 2)

    def test_registered_artifacts_load_and_produce_finite_outputs_for_all_choices(self):
        registry = StationModelRegistry.from_json(
            ROOT / "src/backend/prediction/model_configs.json", ROOT)
        self.assertEqual(registry.get("0074647403").model_id, "legacy")
        for model_id, count in [("legacy", 29), ("model_b", 26), ("three_station", 32)]:
            with self.subTest(model_id=model_id):
                config = registry.get("0074647403", model_id)
                runtime = ResidualLSTMRuntime(config)
                self.assertEqual(len(config.feature_columns), count)
                raw = pd.DataFrame(np.zeros((96, count)), columns=config.feature_columns,
                                   index=pd.date_range("2026-10-01", periods=96, freq="h", tz="UTC"))
                raw["water_level_depth"] = 500.
                prepared_raw, tensor = runtime.preprocessor.prepare_features(raw)
                anchor = raw.index[-1].to_pydatetime()
                prepared = PreparedInput(config.station_id, config.model_version,
                    config.feature_schema_version, tensor, prepared_raw, raw.index, anchor, anchor, anchor)
                result = runtime.predict(prepared)
                self.assertEqual(result.model_version, config.model_version)
                self.assertEqual(len(result.predictions), 5)
                self.assertTrue(all(np.isfinite(p.water_level_mm) for p in result.predictions))
                self.assertFalse(runtime.model.training)

    def test_weather_input_cache_reads_new_vintages_without_observation_change(self):
        cache = ObservationDependentCache(MagicMock())
        config = replace(TEST_CONFIG, weather_location_id="weather-location")
        service = MagicMock(config=config)
        cache.prepare(service)
        cache.prepare(service)
        self.assertEqual(service.prepare.call_count, 2)


class WeatherVintageTests(TestCase):
    def setUp(self):
        self.anchor = pd.Timestamp("2026-10-07T12:00:00Z")
        self.config = replace(TEST_CONFIG, weather_location_id="weather-location")
        self.anchors = pd.DatetimeIndex([self.anchor])

    def run_rows(self, run_id, cycle, value, available=None):
        initialized = self.anchor - pd.Timedelta(hours=cycle)
        return pd.DataFrame({
            "run_id": run_id, "initialized_at": initialized,
            "retrieved_at": initialized + pd.Timedelta(days=20),
            "available_at": available,
            "valid_at": pd.date_range(initialized + pd.Timedelta(hours=1), periods=72, freq="h"),
            "precipitation_mm": value,
        })

    def test_windows_and_age_use_same_eligible_run_and_ignore_newer_forecasts(self):
        frame = pd.concat([self.run_rows("eligible", 6, 2.), self.run_rows("future", 0, 999.)])
        result = forecast_features(frame, self.anchors, self.config).iloc[0]
        self.assertEqual(result.weather_forecast_rain_0_3h, 6.)
        self.assertEqual(result.weather_forecast_rain_3_6h, 6.)
        self.assertEqual(result.weather_forecast_rain_6_12h, 12.)
        self.assertEqual(result.weather_forecast_rain_12_24h, 24.)
        self.assertEqual(result.weather_forecast_age_hours, 6.)

    def test_missing_window_falls_back_to_one_older_complete_run(self):
        newer = self.run_rows("newer", 6, 100.)
        newer.loc[newer.valid_at.eq(self.anchor + pd.Timedelta(hours=2)), "precipitation_mm"] = np.nan
        frame = pd.concat([newer, self.run_rows("older", 12, 3.)])
        result = forecast_features(frame, self.anchors, self.config).iloc[0]
        self.assertEqual(result.weather_forecast_rain_0_3h, 9.)
        self.assertEqual(result.weather_forecast_rain_12_24h, 36.)
        self.assertEqual(result.weather_forecast_age_hours, 12.)

    def test_known_publication_after_anchor_is_excluded_and_expired_run_not_used(self):
        rows = pd.concat([
            self.run_rows("unpublished", 6, 999., self.anchor + pd.Timedelta(hours=1)),
            self.run_rows("old", 30, 999.),
            self.run_rows("available", 12, 1., self.anchor - pd.Timedelta(hours=1)),
        ])
        result = forecast_features(rows, self.anchors, self.config).iloc[0]
        self.assertEqual(result.weather_forecast_rain_0_3h, 3.)
        self.assertEqual(result.weather_forecast_age_hours, 12.)

    def test_missing_is_not_zero_and_primary_partial_forecast_preserves_missing_flags(self):
        missing = forecast_features(pd.DataFrame(), self.anchors, self.config).iloc[0]
        self.assertTrue(np.isnan(missing.weather_forecast_rain_0_3h))
        self.assertEqual(missing.weather_forecast_rain_0_3h_missing, 1)
        frame = self.run_rows("partial", 6, 0.)
        frame.loc[frame.valid_at.eq(self.anchor + pd.Timedelta(hours=2)), "precipitation_mm"] = np.nan
        result = forecast_features(frame, self.anchors, self.config).iloc[0]
        self.assertTrue(np.isnan(result.weather_forecast_rain_0_3h))
        self.assertEqual(result.weather_forecast_rain_0_3h_missing, 1)
        self.assertEqual(result.weather_forecast_rain_3_6h, 0.)
        self.assertEqual(result.weather_forecast_rain_3_6h_missing, 0)

    def test_negative_forecast_and_duplicate_hour_are_rejected(self):
        frame = self.run_rows("invalid", 6, -1.)
        with self.assertRaises(ValueError):
            forecast_features(frame, self.anchors, self.config)
        frame = self.run_rows("duplicate", 6, 1.)
        with self.assertRaises(ValueError):
            forecast_features(pd.concat([frame, frame.iloc[:1]]), self.anchors, self.config)


class HourlyGridTests(TestCase):
    def test_hourly_grid_retains_missing_historical_water_and_requires_observed_anchor(self):
        observations = example_observations(220)
        reference = observations.observed_at.max()
        observations = observations.loc[~(
            observations.station_id.eq(TEST_CONFIG.station_id)
            & observations.observed_at.dt.hour.mod(2).eq(0))]
        config = replace(TEST_CONFIG, sequence_policy="hourly_grid")
        from src.backend.prediction.inputs import PredictionInputService
        service = PredictionInputService(None, example_preprocessor(config))
        connection = MagicMock()
        connection.scalar.return_value = observations.observed_at.min().to_pydatetime()
        with patch("src.backend.prediction.inputs.pd.read_sql", return_value=observations):
            selected = service._select_history(connection, reference)
        self.assertEqual(len(selected), 96)
        self.assertTrue(selected.index.to_series().diff().dropna().eq(pd.Timedelta(hours=1)).all())
        self.assertEqual(int(selected.water_level_missing.sum()), 48)
        self.assertTrue(np.isfinite(selected.water_level_depth.iloc[-1]))
        _, tensor = service.preprocessor.prepare_features(selected)
        self.assertTrue(torch.isfinite(tensor).all())
