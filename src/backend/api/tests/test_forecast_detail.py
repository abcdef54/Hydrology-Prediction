"""Single-anchor chart contracts without model loading or a running database."""

from datetime import datetime, timedelta, timezone
from unittest import TestCase, main
from unittest.mock import MagicMock, patch
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from src.backend.main import create_app
from src.backend.prediction.config import StationModelRegistry


class ForecastDetailTests(TestCase):
    def setUp(self):
        self.engine = MagicMock()
        self.connection = self.engine.connect.return_value.__enter__.return_value
        self.forecast_id = uuid4()
        self.anchor = datetime(2026, 9, 13, 2, tzinfo=timezone.utc)
        self.parent = {
            "id": self.forecast_id,
            "station_id": "0074647403",
            "station_type": "water_level",
            "measurement_unit": "mm",
            "model_version": "archived-model",
            "pipeline_version": "archived-gpu-pipeline",
            "feature_schema_version": "archived-schema",
            "anchor_observed_at": self.anchor,
            "generated_at": self.anchor + timedelta(days=20),
        }
        self.points = [
            {
                "horizon_minutes": horizon,
                "forecast_at": self.anchor + timedelta(minutes=horizon),
                "water_level_mm": 100.0,
            }
            for horizon in (60, 180, 360, 720, 1440)
        ]
        self.actuals = [
            {
                "timestamp": self.anchor + timedelta(hours=hour),
                "actual_water_level_mm": 0.0 if hour == 1 else None,
                "actual_sample_count": 6 if hour == 1 else 0,
            }
            for hour in range(1, 25)
        ]
        self.application = create_app(engine=self.engine, registry=StationModelRegistry([]))
        self.client = self.enterContext(TestClient(self.application, raise_server_exceptions=False))
        self.url = f"/api/v1/forecasts/{self.forecast_id}"

    def request_detail(self):
        parent_result = MagicMock()
        parent_result.mappings.return_value.first.return_value = self.parent
        points_result = MagicMock()
        points_result.mappings.return_value.all.return_value = self.points
        actuals_result = MagicMock()
        actuals_result.mappings.return_value.all.return_value = self.actuals
        self.connection.execute.side_effect = [parent_result, points_result, actuals_result]
        return self.client.get(self.url)

    def test_exact_archived_forecast_five_predictions_and_24_actuals_without_loading(self):
        with patch("torch.load", side_effect=AssertionError("No weights")):
            with patch("joblib.load", side_effect=AssertionError("No scalers")):
                response = self.request_detail()
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["forecast_id"], str(self.forecast_id))
        self.assertEqual(payload["model_version"], "archived-model")
        self.assertEqual(payload["pipeline_version"], "archived-gpu-pipeline")
        self.assertEqual(payload["unit"], "mm")
        self.assertEqual(
            [point["horizon_minutes"] for point in payload["predictions"]],
            [60, 180, 360, 720, 1440],
        )
        self.assertEqual(len(payload["actuals"]), 24)
        self.assertEqual(payload["actuals"][0]["actual_water_level_mm"], 0)
        self.assertIsNone(payload["actuals"][1]["actual_water_level_mm"])
        self.assertEqual(payload["window_end"], "2026-09-14T02:00:00Z")
        self.assertEqual(self.application.state.prediction_services, {})

    def test_different_model_window_uses_saved_horizons(self):
        self.points = self.points[:3]  # Max 6h, not 24h.
        self.actuals = self.actuals[:6]
        response = self.request_detail()
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["window_end"], "2026-09-13T08:00:00Z")
        self.assertEqual(len(response.json()["actuals"]), 6)

    def test_unknown_id_returns_404(self):
        self.parent = None
        response = self.request_detail()
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "FORECAST_NOT_FOUND")

    def test_malformed_id_is_422_before_database_access(self):
        response = self.client.get("/api/v1/forecasts/not-a-uuid")
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "INVALID_REQUEST")
        self.connection.execute.assert_not_called()

    def test_empty_or_inconsistent_stored_horizons_are_not_plotted(self):
        for points in ([], [{**self.points[0], "forecast_at": self.anchor}]):
            self.points = points
            response = self.request_detail()
            self.assertEqual(response.status_code, 503, response.text)
            self.assertEqual(response.json()["error"]["code"], "STORED_FORECAST_INVALID")

    def test_unconfirmed_unit_is_not_compared_as_mm(self):
        self.parent["measurement_unit"] = None
        response = self.request_detail()
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "UNIT_UNAVAILABLE")

    def test_database_failure_returns_safe_503(self):
        self.connection.execute.side_effect = OperationalError(
            "SELECT", {}, Exception("password=secret")
        )
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "DATA_SERVICE_UNAVAILABLE")
        self.assertNotIn("secret", response.text)
        self.assertEqual(response.json()["request_id"], response.headers["X-Request-ID"])

    def test_openapi_describes_detail_endpoint(self):
        operation = self.client.get("/openapi.json").json()["paths"][
            "/api/v1/forecasts/{forecast_id}"
        ]["get"]
        self.assertIn("404", operation["responses"])
        self.assertEqual(operation["parameters"][0]["schema"]["format"], "uuid")


if __name__ == "__main__":
    main()
