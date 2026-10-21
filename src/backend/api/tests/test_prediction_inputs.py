"""HTTP contract tests without a running database or a model checkpoint."""

from dataclasses import replace
from datetime import timedelta
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import numpy as np
from sqlalchemy.exc import OperationalError

from src.backend.api.prediction_inputs import get_input_service
from src.backend.main import create_app
from src.backend.prediction.config import StationModelRegistry
from src.backend.prediction.inputs import InsufficientHistoryError, PreparedInput
from src.backend.prediction.tests.test_inputs import (
    TEST_CONFIG,
    example_observations,
    example_preprocessor,
)
from src.backend.prediction.features import build_hourly_features


class PredictionInputsApiTests(TestCase):
    def setUp(self):
        observations = example_observations()
        anchor = observations.observed_at.max().to_pydatetime()
        raw = build_hourly_features(observations, anchor, TEST_CONFIG).tail(96)
        self.raw, tensor = example_preprocessor().prepare_features(raw)
        self.prepared = PreparedInput(
            station_id=TEST_CONFIG.station_id,
            model_version=TEST_CONFIG.model_version,
            feature_schema_version=TEST_CONFIG.feature_schema_version,
            tensor=tensor,
            raw_features=self.raw,
            timestamps=self.raw.index,
            requested_reference_time=anchor + timedelta(minutes=30),
            hourly_reference_time=anchor,
            anchor_observed_at=anchor,
        )
        self.engine = MagicMock()
        self.service = MagicMock()
        self.service.config = TEST_CONFIG
        self.service.prepare.return_value = self.prepared
        self.application = create_app(
            engine=self.engine,
            registry=StationModelRegistry([TEST_CONFIG]),
            stale_after_minutes=120,
        )
        self.units = {TEST_CONFIG.station_id: "mm"}
        self.application.dependency_overrides[get_input_service] = lambda: (
            self.service,
            self.units,
        )
        self.client = self.enterContext(TestClient(self.application, raise_server_exceptions=False))
        self.url = f"/api/v1/stations/{TEST_CONFIG.station_id}/prediction-inputs"

    def historical_request(self):
        return self.client.get(
            self.url, params={"reference_time": self.prepared.requested_reference_time.isoformat()}
        )

    def assert_error(self, response, status, code):
        self.assertEqual(response.status_code, status, response.text)
        payload = response.json()
        self.assertEqual(payload["error"]["code"], code)
        self.assertEqual(payload["request_id"], response.headers["X-Request-ID"])
        return payload

    def test_form_returns_raw_values_in_model_order_not_tensor_or_predictions(self):
        response = self.historical_request()
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(
            [feature["name"] for feature in payload["features"]], list(TEST_CONFIG.feature_columns)
        )
        self.assertEqual(payload["feature_count"], 29)
        self.assertEqual(payload["sequence_length"], 96)
        self.assertEqual(payload["features"][0]["value"], 149.0)
        self.assertEqual(payload["features"][0]["unit"], "mm")
        self.assertIsNone(payload["features"][4]["unit"])
        self.assertEqual(payload["features"][-1]["unit"], "dimensionless")
        self.assertEqual(payload["editable_features"], list(TEST_CONFIG.feature_columns))
        self.assertTrue(payload["input_ready"])
        self.assertFalse(payload["is_stale"])
        self.assertNotIn("tensor", payload)
        self.assertNotIn("predictions", payload)
        self.assertEqual(
            payload["forecast_horizons_minutes"], list(TEST_CONFIG.forecast_horizons_minutes)
        )
        self.assertEqual(payload["model_version"], TEST_CONFIG.model_version)

    def test_latest_passes_no_reference_and_reports_stale_data(self):
        self.service.prepare.return_value = replace(self.prepared, requested_reference_time=None)
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200, response.text)
        self.service.prepare.assert_called_once_with(None)
        self.assertIsNone(response.json()["requested_reference_time"])
        self.assertTrue(response.json()["is_stale"])

    def test_missing_rain_stays_null_and_does_not_make_input_unready(self):
        raw = self.raw.copy()
        raw.iloc[-1, 4] = np.nan
        self.service.prepare.return_value = replace(self.prepared, raw_features=raw)
        response = self.historical_request()
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertIsNone(payload["features"][4]["value"])
        self.assertEqual(payload["missing_features"], [TEST_CONFIG.feature_columns[4]])
        self.assertTrue(payload["input_ready"])

    def test_historical_age_uses_reference_and_two_hour_boundary(self):
        for lag_hours, expected_stale in ((2, False), (3, True)):
            with self.subTest(lag_hours=lag_hours):
                self.service.prepare.return_value = replace(
                    self.prepared,
                    anchor_observed_at=self.prepared.hourly_reference_time
                    - timedelta(hours=lag_hours),
                )
                payload = self.historical_request().json()
                self.assertEqual(payload["is_stale"], expected_stale)
                self.assertEqual(payload["input_age_minutes"], lag_hours * 60)

    def test_bad_naive_future_time_and_extra_overrides_are_rejected(self):
        for parameters in (
            {"reference_time": "invalid"},
            {"reference_time": "2026-09-01T14:30:00"},
            {"reference_time": "2999-09-01T14:30:00Z"},
            {"overrides": "water_level_depth=5"},
        ):
            with self.subTest(parameters=parameters):
                response = self.client.get(self.url, params=parameters)
                self.assertEqual(response.status_code, 422, response.text)
                self.assertIn("request_id", response.json())
        self.service.prepare.assert_not_called()

    def test_insufficient_history_returns_counts_and_not_ready(self):
        self.service.prepare.side_effect = InsufficientHistoryError(71, 96)
        payload = self.assert_error(self.historical_request(), 400, "INSUFFICIENT_HISTORY")
        self.assertEqual(
            payload["error"]["details"], {"required": 96, "available": 71, "input_ready": False}
        )

    def test_database_failure_does_not_expose_connection_details(self):
        self.service.prepare.side_effect = OperationalError(
            "SELECT", {}, Exception("password=secret")
        )
        response = self.historical_request()
        self.assert_error(response, 503, "DATA_SERVICE_UNAVAILABLE")
        self.assertNotIn("secret", response.text)

    def test_unknown_station_returns_404(self):
        self.application.dependency_overrides.clear()
        connection = self.engine.connect.return_value.__enter__.return_value
        connection.execute.return_value.mappings.return_value.first.return_value = None
        self.assert_error(self.client.get(self.url), 404, "STATION_NOT_FOUND")

    def test_disabled_prediction_returns_503(self):
        self.application.dependency_overrides.clear()
        connection = self.engine.connect.return_value.__enter__.return_value
        connection.execute.return_value.mappings.return_value.first.return_value = {
            "id": TEST_CONFIG.station_id,
            "enabled": True,
            "prediction_enabled": False,
        }
        self.assert_error(self.client.get(self.url), 503, "MODEL_UNAVAILABLE")

    def test_existing_station_without_configured_model_returns_503(self):
        self.application.dependency_overrides.clear()
        self.application.state.model_registry = StationModelRegistry([])
        connection = self.engine.connect.return_value.__enter__.return_value
        connection.execute.return_value.mappings.return_value.first.return_value = {
            "id": TEST_CONFIG.station_id,
            "enabled": True,
            "prediction_enabled": True,
        }
        self.assert_error(self.client.get(self.url), 503, "MODEL_UNAVAILABLE")

    def test_missing_artifact_returns_safe_503(self):
        self.application.dependency_overrides.clear()
        connection = self.engine.connect.return_value.__enter__.return_value
        connection.execute.return_value.mappings.return_value.first.return_value = {
            "id": TEST_CONFIG.station_id,
            "enabled": True,
            "prediction_enabled": True,
        }
        connection.execute.return_value.all.return_value = [(TEST_CONFIG.station_id, "mm")]
        with patch(
            "src.backend.api.prediction_inputs.InputPreprocessor.from_file",
            side_effect=FileNotFoundError("private/path"),
        ):
            response = self.client.get(self.url)
        self.assert_error(response, 503, "MODEL_UNAVAILABLE")
        self.assertNotIn("private/path", response.text)

    def test_openapi_describes_endpoint_and_timezone_parameter(self):
        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)
        operation = response.json()["paths"]["/api/v1/stations/{station_id}/prediction-inputs"][
            "get"
        ]
        self.assertEqual(
            {parameter["name"] for parameter in operation["parameters"]},
            {"station_id", "reference_time", "model_id"},
        )
        self.assertIn("503", operation["responses"])


if __name__ == "__main__":
    main()
