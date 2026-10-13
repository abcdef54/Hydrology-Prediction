"""Prediction HTTP contract, override validation and per-station runtime reuse."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from unittest import TestCase, main
from unittest.mock import MagicMock, patch
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from src.backend.main import create_app
from src.backend.prediction.config import StationModelRegistry
from src.backend.prediction.inference import ForecastPoint, ForecastResult
from src.backend.prediction.inputs import InsufficientHistoryError
from src.backend.prediction.tests.test_inputs import TEST_CONFIG, example_preprocessor


class PredictionsApiTests(TestCase):
    def setUp(self):
        self.application = create_app(
            engine=MagicMock(),
            registry=StationModelRegistry([TEST_CONFIG]),
            stale_after_minutes=120,
        )
        self.client = self.enterContext(TestClient(self.application, raise_server_exceptions=False))
        self.service = MagicMock()
        self.service.runtime.preprocessor = example_preprocessor()
        anchor = datetime(2026, 9, 20, 7, tzinfo=timezone.utc)
        self.forecast = ForecastResult(
            station_id=TEST_CONFIG.station_id,
            model_version=TEST_CONFIG.model_version,
            feature_schema_version=TEST_CONFIG.feature_schema_version,
            requested_reference_time=anchor + timedelta(minutes=30),
            hourly_reference_time=anchor,
            anchor_observed_at=anchor,
            predictions=tuple(
                ForecastPoint(minutes, anchor + timedelta(minutes=minutes), 200.5)
                for minutes in TEST_CONFIG.forecast_horizons_minutes
            ),
        )
        self.service.predict.return_value = self.forecast
        self.station_patch = patch(
            "src.backend.api.predictions.get_station_model_config", return_value=TEST_CONFIG
        )
        self.station_lookup = self.station_patch.start()
        self.addCleanup(self.station_patch.stop)
        self.service_patch = patch(
            "src.backend.api.predictions.get_prediction_service", return_value=self.service
        )
        self.service_lookup = self.service_patch.start()
        self.addCleanup(self.service_patch.stop)
        self.body = {
            "station_id": TEST_CONFIG.station_id,
            "reference_time": self.forecast.requested_reference_time.isoformat(),
        }
        self.url = "/api/v1/predictions"

    def assert_error(self, response, status, code):
        self.assertEqual(response.status_code, status, response.text)
        payload = response.json()
        self.assertEqual(payload["error"]["code"], code)
        self.assertEqual(payload["request_id"], response.headers["X-Request-ID"])
        return payload

    def test_historical_predictions_are_mm_with_anchor_based_horizons(self):
        response = self.client.post(self.url, json=self.body)
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["unit"], "mm")
        self.assertEqual(payload["model_version"], TEST_CONFIG.model_version)
        self.assertEqual(payload["overrides_applied"], {})
        self.assertFalse(payload["is_stale"])
        self.assertEqual(
            [point["horizon_minutes"] for point in payload["predictions"]],
            list(TEST_CONFIG.forecast_horizons_minutes),
        )
        self.assertEqual(payload["predictions"][0]["water_level_mm"], 200.5)
        self.assertEqual(
            datetime.fromisoformat(payload["predictions"][0]["forecast_at"]),
            self.forecast.anchor_observed_at + timedelta(hours=1),
        )
        self.service.predict.assert_called_once_with(self.forecast.requested_reference_time, {})

    def test_reused_forecast_metadata_and_current_freshness_are_returned(self):
        forecast_id = uuid4()
        self.service.predict.return_value = replace(
            self.forecast,
            forecast_id=forecast_id,
            generated_at=self.forecast.anchor_observed_at,
            pipeline_version="test-pipeline",
            reused=True,
        )
        response = self.client.post(self.url, json=self.body)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["forecast_id"], str(forecast_id))
        self.assertTrue(response.json()["reused"])
        self.assertEqual(response.json()["pipeline_version"], "test-pipeline")
        self.assertFalse(response.json()["is_stale"])

    def test_latest_and_explicit_null_reference_use_latest_data(self):
        self.service.predict.return_value = replace(self.forecast, requested_reference_time=None)
        for body in (
            {"station_id": TEST_CONFIG.station_id},
            {"station_id": TEST_CONFIG.station_id, "reference_time": None},
        ):
            with self.subTest(body=body):
                response = self.client.post(self.url, json=body)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertIsNone(response.json()["requested_reference_time"])
                self.assertTrue(response.json()["is_stale"])
                self.assertEqual(self.service.predict.call_args.args, (None, {}))

    def test_partial_override_values_and_null_rain_are_forwarded_in_original_units(self):
        overrides = {"water_level_depth": -10, "rain_750062_depth": None, "flood_season": 1}
        response = self.client.post(self.url, json={**self.body, "overrides": overrides})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["overrides_applied"], overrides)
        self.assertEqual(self.service.predict.call_args.args[1], overrides)

    def test_unknown_feature_invalid_flags_and_missing_current_water_are_422(self):
        for overrides in (
            {"unknown_feature": 3},
            {"flood_season": 2},
            {"water_level_missing": None},
            {"water_level_depth": None},
            {"water_level_depth": 1e100},
        ):
            with self.subTest(overrides=overrides):
                response = self.client.post(self.url, json={**self.body, "overrides": overrides})
                self.assert_error(response, 422, "INVALID_OVERRIDE")
        self.service.predict.assert_not_called()

    def test_invalid_request_types_dates_and_extra_fields_are_422(self):
        for body in (
            {},
            {"station_id": 123},
            {**self.body, "unexpected": True},
            {**self.body, "reference_time": "2026-09-01T00:00:00"},
            {**self.body, "overrides": {"water_level_depth": "500"}},
            {**self.body, "overrides": {"flood_season": True}},
            {**self.body, "overrides": None},
        ):
            with self.subTest(body=body):
                self.assert_error(self.client.post(self.url, json=body), 422, "INVALID_REQUEST")
        self.service.predict.assert_not_called()

    def test_non_finite_json_numbers_are_rejected(self):
        for value in ("NaN", "Infinity", "-Infinity"):
            body = '{"station_id":"0074647403","overrides":{"water_level_depth":' + value + "}}"
            response = self.client.post(
                self.url, content=body, headers={"Content-Type": "application/json"}
            )
            self.assert_error(response, 422, "INVALID_REQUEST")
        self.service.predict.assert_not_called()

    def test_future_reference_is_rejected_before_station_or_model_lookup(self):
        response = self.client.post(
            self.url, json={**self.body, "reference_time": "2999-01-01T00:00:00Z"}
        )
        self.assert_error(response, 422, "INVALID_REFERENCE_TIME")
        self.station_lookup.assert_not_called()
        self.service_lookup.assert_not_called()

    def test_insufficient_history_returns_required_and_available(self):
        self.service.predict.side_effect = InsufficientHistoryError(71, 96)
        payload = self.assert_error(
            self.client.post(self.url, json=self.body), 400, "INSUFFICIENT_HISTORY"
        )
        self.assertEqual(
            payload["error"]["details"], {"required": 96, "available": 71, "input_ready": False}
        )

    def test_runtime_error_is_safe_503(self):
        self.service.predict.side_effect = RuntimeError("private artifact path")
        response = self.client.post(self.url, json=self.body)
        self.assert_error(response, 503, "PREDICTION_UNAVAILABLE")
        self.assertNotIn("private artifact", response.text)

    def test_database_error_is_safe_503(self):
        self.service.predict.side_effect = OperationalError(
            "SELECT", {}, Exception("password=secret")
        )
        response = self.client.post(self.url, json=self.body)
        self.assert_error(response, 503, "DATA_SERVICE_UNAVAILABLE")
        self.assertNotIn("secret", response.text)

    def test_runtime_is_loaded_once_per_station_and_version(self):
        self.service_patch.stop()
        with patch("src.backend.api.predictions.ResidualLSTMRuntime") as runtime_loader:
            runtime_loader.side_effect = lambda config, device: MagicMock(config=config)
            # Avoid SQL; these calls exercise the loader directly within app lifespan.
            from src.backend.api.predictions import get_prediction_service

            request = MagicMock(app=self.application)
            first = get_prediction_service(request, TEST_CONFIG)
            second = get_prediction_service(request, TEST_CONFIG)
            station_b = replace(TEST_CONFIG, station_id="water-b", model_version="model-b")
            third = get_prediction_service(request, station_b)
            upgraded = replace(TEST_CONFIG, model_version="model-a-v2")
            fourth = get_prediction_service(request, upgraded)
            self.assertIs(first, second)
            self.assertIsNot(first, third)
            self.assertIsNot(first, fourth)
            self.assertEqual(runtime_loader.call_count, 3)

    def test_missing_checkpoint_returns_503_and_does_not_keep_failed_runtime(self):
        self.service_patch.stop()
        with patch(
            "src.backend.api.predictions.ResidualLSTMRuntime",
            side_effect=FileNotFoundError("private/path"),
        ):
            response = self.client.post(self.url, json=self.body)
        self.assert_error(response, 503, "MODEL_UNAVAILABLE")
        self.assertNotIn("private/path", response.text)
        self.assertEqual(self.application.state.prediction_services, {})

    def test_unknown_station_and_unconfigured_model_use_shared_availability_check(self):
        self.station_patch.stop()
        connection = (
            self.application.state.database_engine.connect.return_value.__enter__.return_value
        )
        connection.execute.return_value.mappings.return_value.first.return_value = None
        self.assert_error(self.client.post(self.url, json=self.body), 404, "STATION_NOT_FOUND")
        connection.execute.return_value.mappings.return_value.first.return_value = {
            "enabled": True,
            "prediction_enabled": True,
        }
        self.application.state.model_registry = StationModelRegistry([])
        self.assert_error(self.client.post(self.url, json=self.body), 503, "MODEL_UNAVAILABLE")

    def test_openapi_includes_prediction_request_and_response(self):
        operation = self.client.get("/openapi.json").json()["paths"][self.url]["post"]
        self.assertIn("requestBody", operation)
        self.assertIn("503", operation["responses"])


if __name__ == "__main__":
    main()
