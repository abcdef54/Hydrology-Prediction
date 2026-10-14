"""Read-only scenario comparison contracts, independent of forecast persistence."""

from datetime import datetime, timedelta, timezone
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from src.backend.main import create_app
from src.backend.prediction.config import StationModelRegistry


class ForecastActualsTests(TestCase):
    def setUp(self):
        self.engine = MagicMock()
        self.connection = self.engine.connect.return_value.__enter__.return_value
        self.app = create_app(engine=self.engine, registry=StationModelRegistry([]))
        self.client = self.enterContext(TestClient(self.app, raise_server_exceptions=False))
        self.url = "/api/v1/stations/0074647403/forecast-actuals"
        self.anchor = datetime(2026, 9, 13, 2, tzinfo=timezone.utc)
        self.params = {"anchor_observed_at": self.anchor.isoformat(), "horizon_minutes": 1440}
        self.station = {"station_type": "water_level", "measurement_unit": "mm"}

    def request_actuals(self, **changes):
        params = {**self.params, **changes}
        station_result = MagicMock()
        station_result.mappings.return_value.first.return_value = self.station
        actual_result = MagicMock()
        actual_result.mappings.return_value.all.return_value = [
            {
                "timestamp": self.anchor + timedelta(hours=hour),
                "actual_water_level_mm": 0 if hour == 1 else -20 if hour == 2 else None,
                "actual_sample_count": 6 if hour <= 2 else 0,
            }
            for hour in range(1, int(params["horizon_minutes"]) // 60 + 1)
        ]
        self.connection.execute.side_effect = [station_result, actual_result]
        return self.client.get(self.url, params=params)

    def test_reads_24_hourly_means_without_models_or_database_writes(self):
        with patch("torch.load", side_effect=AssertionError("No weights")), patch(
            "joblib.load", side_effect=AssertionError("No scaler")
        ):
            response = self.request_actuals()
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(data["unit"], "mm")
        self.assertEqual(data["anchor_observed_at"], "2026-09-13T02:00:00Z")
        self.assertEqual(data["window_end"], "2026-09-14T02:00:00Z")
        self.assertEqual(len(data["actuals"]), 24)
        self.assertEqual(
            [point["actual_water_level_mm"] for point in data["actuals"][:3]], [0, -20, None]
        )
        self.assertEqual(self.app.state.prediction_services, {})
        for call in self.connection.execute.call_args_list:
            self.assertTrue(call.args[0].is_select)
        self.connection.commit.assert_not_called()
        self.engine.begin.assert_not_called()

    def test_timezone_and_non_hourly_window(self):
        response = self.request_actuals(
            anchor_observed_at="2026-09-13T09:00:00+07:00", horizon_minutes=90
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["anchor_observed_at"], "2026-09-13T02:00:00Z")
        self.assertEqual(len(response.json()["actuals"]), 1)
        self.assertEqual(response.json()["window_end"], "2026-09-13T03:30:00Z")

    def test_invalid_query_is_rejected_before_database_access(self):
        for changes in (
            {"anchor_observed_at": "2026-09-13T02:00:00"},
            {"anchor_observed_at": "1899-01-01T00:00:00Z"},
            {"anchor_observed_at": "2999-01-01T00:00:00Z"},
            {"horizon_minutes": 0},
            {"horizon_minutes": 10081},
            {"horizon_minutes": "1.5"},
            {"overrides": "not accepted"},
        ):
            with self.subTest(changes=changes):
                response = self.client.get(self.url, params={**self.params, **changes})
                self.assertEqual(response.status_code, 422, response.text)
        self.connection.execute.assert_not_called()

    def test_station_missing_wrong_type_and_unknown_units(self):
        for station, status, code in (
            (None, 404, "STATION_NOT_FOUND"),
            ({"station_type": "rainfall", "measurement_unit": "mm"}, 400, "INVALID_STATION_TYPE"),
            ({"station_type": "water_level", "measurement_unit": None}, 503, "UNIT_UNAVAILABLE"),
        ):
            self.station = station
            response = self.request_actuals()
            self.assertEqual(response.status_code, status, response.text)
            self.assertEqual(response.json()["error"]["code"], code)

    def test_database_error_is_safe(self):
        self.connection.execute.side_effect = OperationalError("SELECT", {}, Exception("private"))
        response = self.client.get(self.url, params=self.params)
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private", response.text)

    def test_openapi_exposes_bounded_window(self):
        operation = self.client.get("/openapi.json").json()["paths"][
            "/api/v1/stations/{station_id}/forecast-actuals"
        ]["get"]
        parameters = {parameter["name"]: parameter for parameter in operation["parameters"]}
        self.assertEqual(parameters["horizon_minutes"]["schema"]["maximum"], 10080)
        self.assertEqual(parameters["anchor_observed_at"]["schema"]["format"], "date-time")


if __name__ == "__main__":
    main()
