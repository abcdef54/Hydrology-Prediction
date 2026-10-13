"""HTTP contracts for paged raw observation history."""

from datetime import datetime, timedelta, timezone
from unittest import TestCase, main
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from src.backend.main import create_app
from src.backend.prediction.config import StationModelRegistry


class ObservationHistoryTests(TestCase):
    def setUp(self):
        self.engine = MagicMock()
        self.connection = self.engine.connect.return_value.__enter__.return_value
        self.station_result = MagicMock()
        self.station_result.mappings.return_value.first.return_value = {
            "station_type": "water_level",
            "measurement_unit": "mm",
        }
        self.history_result = MagicMock()
        self.history_result.mappings.return_value.all.return_value = []
        self.connection.execute.side_effect = [self.station_result, self.history_result]
        self.application = create_app(engine=self.engine, registry=StationModelRegistry([]))
        self.client = self.enterContext(TestClient(self.application, raise_server_exceptions=False))
        self.url = "/api/v1/stations/0074647403/observations"
        self.parameters = {"start": "2026-09-13T09:00:00+07:00", "end": "2026-09-14T09:00:00+07:00"}
        self.anchor = datetime(2026, 9, 14, 1, tzinfo=timezone.utc)

    def observation(self, hours_before=0, value=0):
        timestamp = self.anchor - timedelta(hours=hours_before)
        return {
            "depth": value,
            "observed_at": timestamp,
            "received_at": timestamp + timedelta(minutes=1),
            "quality_status": "missing" if value is None else "observed",
        }

    def test_paging_keeps_missing_values_and_returns_cursor(self):
        self.history_result.mappings.return_value.all.return_value = [
            self.observation(value=-25.5),
            self.observation(1, None),
            self.observation(2),
        ]
        response = self.client.get(self.url, params={**self.parameters, "limit": 2})
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["start"], "2026-09-13T02:00:00Z")
        self.assertEqual(payload["end"], "2026-09-14T02:00:00Z")
        self.assertEqual([row["value"] for row in payload["observations"]], [-25.5, None])
        self.assertEqual(payload["observations"][1]["quality_status"], "missing")
        self.assertTrue(payload["has_more"])
        self.assertEqual(payload["next_before"], "2026-09-14T00:00:00Z")
        self.assertEqual(self.application.state.prediction_services, {})

    def test_last_page_has_no_cursor_and_preserves_zero_rain(self):
        self.station_result.mappings.return_value.first.return_value = {
            "station_type": "rainfall",
            "measurement_unit": None,
        }
        self.history_result.mappings.return_value.all.return_value = [self.observation()]
        response = self.client.get(
            self.url, params={**self.parameters, "limit": 1, "before": "2026-09-14T01:30:00Z"}
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(response.json()["has_more"])
        self.assertIsNone(response.json()["next_before"])
        self.assertIsNone(response.json()["unit"])
        self.assertEqual(response.json()["observations"][0]["value"], 0)

    def test_empty_range_is_successful(self):
        response = self.client.get(self.url, params=self.parameters)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["observations"], [])
        self.assertFalse(response.json()["has_more"])
        self.assertIsNone(response.json()["next_before"])

    def test_invalid_ranges_timezone_limits_and_unknown_parameters_are_rejected(self):
        invalid_parameters = [
            {},
            {**self.parameters, "start": self.parameters["end"]},
            {**self.parameters, "start": "2026-09-15T00:00:00Z"},
            {**self.parameters, "start": "2026-09-13T00:00:00"},
            {**self.parameters, "end": "invalid"},
            {**self.parameters, "before": "2026-09-14T00:00:00"},
            {**self.parameters, "limit": 0},
            {**self.parameters, "limit": 1001},
            {**self.parameters, "end": "2999-01-01T00:00:00Z"},
            {**self.parameters, "extra": "invalid"},
        ]
        for parameters in invalid_parameters:
            with self.subTest(parameters=parameters):
                response = self.client.get(self.url, params=parameters)
                self.assertEqual(response.status_code, 422, response.text)
                self.assertEqual(response.json()["error"]["code"], "INVALID_REQUEST")
        self.connection.execute.assert_not_called()

    def test_unknown_station_returns_404(self):
        self.station_result.mappings.return_value.first.return_value = None
        response = self.client.get(self.url, params=self.parameters)
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "STATION_NOT_FOUND")
        self.connection.execute.assert_called_once()

    def test_database_error_does_not_expose_credentials(self):
        self.connection.execute.side_effect = OperationalError(
            "SELECT", {}, Exception("password=secret")
        )
        response = self.client.get(self.url, params=self.parameters)
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("secret", response.text)
        self.assertEqual(response.json()["request_id"], response.headers["X-Request-ID"])

    def test_openapi_describes_time_range_and_paging(self):
        operation = self.client.get("/openapi.json").json()["paths"][
            "/api/v1/stations/{station_id}/observations"
        ]["get"]
        self.assertEqual(
            {parameter["name"] for parameter in operation["parameters"]},
            {"station_id", "start", "end", "limit", "before"},
        )
        self.assertIn("404", operation["responses"])
        self.assertIn("503", operation["responses"])


if __name__ == "__main__":
    main()
