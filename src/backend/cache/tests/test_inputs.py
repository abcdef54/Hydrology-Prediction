"""Input cache isolation and invalidation without database, Redis or inference."""

from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import MagicMock, patch

from redis.exceptions import ConnectionError

from src.backend.cache.inputs import ObservationDependentCache
from src.backend.prediction.tests.test_persistence import persistence_example


class InputCacheTests(TestCase):
    def setUp(self):
        self.client = MagicMock()
        self.client.mget.return_value = [b"generation"] * 5
        self.cache = ObservationDependentCache(self.client, max_entries=2)
        self.loader = MagicMock(return_value={"value": [1]})

    def test_hit_skips_loader_and_callers_cannot_mutate_cached_values(self):
        first = self.cache.load("key", ["water"], self.loader)
        first["value"][0] = 999
        self.assertEqual(self.cache.load("key", ["water"], self.loader), {"value": [1]})
        self.loader.assert_called_once()

    def test_source_changes_expiry_and_eviction_each_force_reload(self):
        self.cache.load("key", ["water"], self.loader)
        self.client.mget.return_value = [b"new"] * 5
        self.cache.load("key", ["water"], self.loader)
        with patch("src.backend.cache.inputs.monotonic", return_value=10**12):
            self.cache.load("key", ["water"], self.loader)
        self.cache.load("other", ["water"], self.loader)
        self.cache.load("third", ["water"], self.loader)
        self.assertEqual(len(self.cache.entries), 2)
        self.assertNotIn("key", self.cache.entries)
        self.assertEqual(self.loader.call_count, 5)

    def test_outage_disabled_errors_and_invalidation_race_do_not_publish(self):
        self.client.mget.side_effect = ConnectionError("private")
        self.cache.load("key", [], self.loader)
        self.assertFalse(self.cache.entries)
        self.client.mget.side_effect = [[b"old"], [b"new"]]
        self.cache.load("key", [], self.loader)
        self.assertFalse(self.cache.entries)
        disabled = ObservationDependentCache()
        disabled.load("key", [], self.loader)
        self.assertFalse(disabled.entries)
        self.client.mget.side_effect = None
        with self.assertRaises(ValueError):
            self.cache.load("error", [], MagicMock(side_effect=ValueError("failed")))
        self.assertFalse(self.cache.entries)

    def test_incomplete_ranges_are_not_retained(self):
        for _ in range(2):
            self.cache.load("key", [], self.loader, retain=lambda rows: False)
        self.assertEqual(self.loader.call_count, 2)

    def test_auto_fill_and_prediction_share_original_inputs_with_current_context(self):
        config, prepared, _ = persistence_example()
        with TemporaryDirectory() as folder:
            artifact = Path(folder) / "scaler"
            artifact.write_bytes(b"test")
            config = replace(config, preprocessing_path=artifact)
            service = MagicMock(config=config)
            service.prepare.return_value = prepared
            self.cache.prepare(service, prepared.requested_reference_time)
            requested = prepared.requested_reference_time + timedelta(minutes=1)
            hit = self.cache.prepare(service, requested)
            self.assertEqual(hit.requested_reference_time, requested)
            service.prepare.assert_called_once()
            hit.raw_features.iloc[-1, 0] = 999
            hit.tensor.fill_(999)
            clean = self.cache.prepare(service, requested)
            self.assertEqual(float(clean.tensor[0, 0, 0]), 1)
            self.assertEqual(clean.raw_features.iloc[-1, 0], 1)
            artifact.write_bytes(b"changed artifact")
            self.cache.prepare(service, requested)
            self.assertEqual(service.prepare.call_count, 2)
            overrides = {"water_level_depth": 99}
            self.cache.prepare(service, requested, overrides)
            service.prepare.assert_called_with(requested, overrides)

    def test_auto_fill_then_repeated_prediction_skips_history_and_forecast_queries(self):
        from src.backend.prediction.cached_service import CachedPredictionService

        config, prepared, forecast = persistence_example()
        with TemporaryDirectory() as folder:
            artifact = Path(folder) / "scaler"
            artifact.write_bytes(b"test")
            config = replace(config, preprocessing_path=artifact)
            inputs = MagicMock(config=config)
            inputs.prepare.return_value = prepared
            self.cache.prepare(inputs, prepared.requested_reference_time)
            engine, runtime, results = MagicMock(), MagicMock(config=config), MagicMock()
            runtime.pipeline_version = "pipeline"
            results.read.side_effect = [None, replace(forecast, reused=True)]
            service = CachedPredictionService(engine, runtime, results, self.cache)
            service.inputs = inputs
            with patch(
                "src.backend.prediction.cached_service.load_forecast", return_value=forecast
            ):
                service.predict(prepared.requested_reference_time)
                service.predict(prepared.requested_reference_time)
            inputs.prepare.assert_called_once()
            engine.connect.assert_called_once()
            runtime.predict.assert_not_called()
            results.store.assert_called_once()
