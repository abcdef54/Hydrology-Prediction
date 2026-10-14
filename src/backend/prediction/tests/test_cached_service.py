"""Cache/DB/inference ordering without GPU, Redis or database writes."""

from dataclasses import replace
from datetime import timedelta, timezone, datetime
from unittest import TestCase
from unittest.mock import MagicMock, patch
from uuid import uuid4

from redis.exceptions import ConnectionError
from src.backend.cache.forecasts import ForecastCache
from src.backend.prediction.cached_service import CachedPredictionService, AnchorUnavailableError
from src.backend.prediction.tests.test_persistence import persistence_example


class CachedServiceTests(TestCase):
    def setUp(self):
        self.config, self.prepared, forecast = persistence_example()
        self.forecast = replace(
            forecast,
            forecast_id=uuid4(),
            generated_at=datetime.now(timezone.utc),
            pipeline_version="pipeline",
        )
        self.runtime = MagicMock(config=self.config, pipeline_version="pipeline")
        self.engine = MagicMock()
        self.cache = MagicMock()
        self.cache.read.return_value = None
        self.service = CachedPredictionService(self.engine, self.runtime, self.cache)
        self.service.inputs = MagicMock()
        self.service.inputs.prepare.return_value = self.prepared

    def test_cache_hit_skips_stored_lookup_and_inference(self):
        self.cache.read.return_value = replace(self.forecast, reused=True)
        result = self.service.predict()
        self.assertTrue(result.reused)
        self.engine.connect.assert_not_called()
        self.runtime.predict.assert_not_called()

    def test_database_hit_warms_cache_and_does_not_infer(self):
        with patch(
            "src.backend.prediction.cached_service.load_forecast",
            return_value=replace(self.forecast, reused=True),
        ):
            self.service.predict()
        self.cache.store.assert_called_once()
        self.runtime.predict.assert_not_called()

    def test_miss_infers_saves_then_publishes_and_failed_save_does_not_publish(self):
        events = []
        self.runtime.predict.side_effect = lambda prepared: events.append("infer") or self.forecast
        self.cache.store.side_effect = lambda *args: events.append("cache")
        with patch("src.backend.prediction.cached_service.load_forecast", return_value=None), patch(
            "src.backend.prediction.cached_service.save_forecast",
            side_effect=lambda *args: events.append("commit") or self.forecast,
        ):
            self.service.predict()
        self.assertEqual(events, ["infer", "commit", "cache"])
        self.cache.store.reset_mock()
        with patch("src.backend.prediction.cached_service.load_forecast", return_value=None), patch(
            "src.backend.prediction.cached_service.save_forecast", side_effect=ValueError("failed")
        ):
            with self.assertRaises(ValueError):
                self.service.predict()
        self.cache.store.assert_not_called()

    def test_overrides_bypass_cache_and_storage_and_wrong_anchor_is_rejected(self):
        self.service.predict(overrides={"water_level_depth": 1})
        self.cache.read.assert_not_called()
        self.cache.store.assert_not_called()
        self.engine.connect.assert_not_called()
        with self.assertRaises(AnchorUnavailableError):
            self.service.predict(
                required_anchor=self.prepared.anchor_observed_at + timedelta(hours=1)
            )
        self.cache.read.assert_not_called()

    def test_cache_roundtrip_rebuilds_context_and_rejects_corrupt_output(self):
        client = MagicMock()
        cache = ForecastCache(client, 600)
        cache.store("input", self.forecast)
        payload = client.set.call_args.args[1]
        client.get.return_value = payload
        changed = replace(self.prepared, requested_reference_time=None)
        result = cache.read(self.config, changed, "pipeline", "input")
        self.assertEqual(result.forecast_id, self.forecast.forecast_id)
        self.assertIsNone(result.requested_reference_time)
        self.assertTrue(result.reused)
        self.assertEqual(client.set.call_args.kwargs["ex"], 600)
        self.assertIsNone(cache.read(self.config, self.prepared, "other", "input"))
        client.get.return_value = "invalid"
        self.assertIsNone(cache.read(self.config, self.prepared, "pipeline", "input"))
        client.get.side_effect = ConnectionError("private")
        self.assertIsNone(cache.read(self.config, self.prepared, "pipeline", "input"))
