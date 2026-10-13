"""Saved-series API contracts without a database or model artifacts."""

from datetime import datetime, timedelta, timezone
from unittest import TestCase, main
from unittest.mock import MagicMock, patch
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from src.backend.main import create_app
from src.backend.prediction.config import StationModelRegistry
from src.backend.prediction.tests.test_inputs import TEST_CONFIG


class ForecastHistoryTests(TestCase):
    def setUp(self):
        self.engine = MagicMock()
        self.connection = self.engine.connect.return_value.__enter__.return_value
        self.station = MagicMock()
        self.station.mappings.return_value.first.return_value = {
            "station_type": "water_level",
            "measurement_unit": "mm",
        }
        self.saved = MagicMock()
        self.saved.mappings.return_value.all.return_value = []
        self.connection.execute.side_effect = [self.station, self.saved]
        self.application = create_app(
            engine=self.engine, registry=StationModelRegistry([TEST_CONFIG])
        )
        self.client = self.enterContext(TestClient(self.application, raise_server_exceptions=False))
        self.url = "/api/v1/stations/0074647403/forecasts"
        self.parameters = {
            "start": "2026-09-13T09:00:00+07:00",
            "end": "2026-09-14T09:00:00+07:00",
            "horizon_minutes": 60,
            "model_version": "archived-model",
            "pipeline_version": "gpu-pipeline",
        }

    def point(self, hours_before=0, actual=20):
        target = datetime(2026, 9, 14, 1, tzinfo=timezone.utc) - timedelta(hours=hours_before)
        return {
            "forecast_id": uuid4(),
            "feature_schema_version": "test-schema",
            "anchor_observed_at": target - timedelta(hours=1),
            "forecast_at": target,
            "generated_at": datetime(2026, 10, 13, tzinfo=timezone.utc),
            "predicted_water_level_mm": 30.0,
            "actual_water_level_mm": actual,
            "actual_sample_count": 3 if actual is not None else 0,
        }

    def test_page_preserves_prediction_truth_and_missing_truth_without_loading_model(self):
        self.saved.mappings.return_value.all.return_value = [
            self.point(),
            self.point(1, None),
            self.point(2),
        ]
        with patch("torch.load", side_effect=AssertionError("No weight loading")):
            response = self.client.get(self.url, params={**self.parameters, "limit": 2})
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["start"], "2026-09-13T02:00:00Z")
        self.assertEqual(payload["model_version"], "archived-model")
        self.assertEqual(payload["pipeline_version"], "gpu-pipeline")
        self.assertEqual(payload["unit"], "mm")
        self.assertEqual(len(payload["points"]), 2)
        self.assertEqual(payload["points"][0]["actual_sample_count"], 3)
        self.assertIsNone(payload["points"][1]["actual_water_level_mm"])
        self.assertTrue(payload["has_more"])
        self.assertEqual(payload["next_before"], "2026-09-14T00:00:00Z")
        self.assertEqual(self.application.state.prediction_services, {})

    def test_empty_series_is_200(self):
        response = self.client.get(self.url, params=self.parameters)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["points"], [])
        self.assertIsNone(response.json()["next_before"])
        self.assertFalse(response.json()["has_more"])

    def test_unknown_station_and_rain_station_are_distinct_errors(self):
        for station, status, code in (
            (None, 404, "STATION_NOT_FOUND"),
            ({"station_type": "rainfall", "measurement_unit": None}, 422, "INVALID_STATION_TYPE"),
        ):
            self.connection.execute.side_effect = [self.station]
            self.station.mappings.return_value.first.return_value = station
            response = self.client.get(self.url, params=self.parameters)
            self.assertEqual(response.status_code, status, response.text)
            self.assertEqual(response.json()["error"]["code"], code)

    def test_unknown_observation_unit_is_not_silently_compared_with_mm(self):
        self.station.mappings.return_value.first.return_value = {
            "station_type": "water_level",
            "measurement_unit": None,
        }
        response = self.client.get(self.url, params=self.parameters)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "UNIT_UNAVAILABLE")

    def test_archived_model_requires_pipeline_if_not_provided(self):
        parameters = dict(self.parameters)
        parameters.pop("pipeline_version")
        response = self.client.get(self.url, params=parameters)
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "INVALID_REQUEST")

    def test_default_identity_uses_backend_configuration_without_loading_weights(self):
        parameters = {
            key: value
            for key, value in self.parameters.items()
            if key not in ("model_version", "pipeline_version")
        }
        with patch("pathlib.Path.read_bytes", return_value=b"artifact"):
            with patch(
                "src.backend.api.forecast_history.build_pipeline_version",
                return_value="configured-pipeline",
            ):
                with patch("torch.load", side_effect=AssertionError("No weight loading")):
                    response = self.client.get(self.url, params=parameters)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["model_version"], TEST_CONFIG.model_version)
        self.assertEqual(response.json()["pipeline_version"], "configured-pipeline")

    def test_invalid_parameters_are_422_before_database_access(self):
        for parameters in (
            {},
            {**self.parameters, "horizon_minutes": 0},
            {**self.parameters, "start": "2026-09-13T00:00:00"},
            {**self.parameters, "start": self.parameters["end"]},
            {**self.parameters, "before": "invalid"},
            {**self.parameters, "limit": 1001},
            {**self.parameters, "device": "cpu"},
        ):
            response = self.client.get(self.url, params=parameters)
            self.assertEqual(response.status_code, 422, response.text)
        self.connection.execute.assert_not_called()

    def test_database_failure_is_safe_503(self):
        self.connection.execute.side_effect = OperationalError(
            "SELECT", {}, Exception("password=secret")
        )
        response = self.client.get(self.url, params=self.parameters)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "DATA_SERVICE_UNAVAILABLE")
        self.assertNotIn("secret", response.text)


if __name__ == "__main__":
    main()
