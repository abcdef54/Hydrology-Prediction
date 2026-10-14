"""Station-list HTTP contracts without database access or model loading."""

from unittest import TestCase, main
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from src.backend.main import create_app
from src.backend.prediction.config import StationModelRegistry
from src.backend.prediction.tests.test_inputs import TEST_CONFIG


def station_metadata(station_id, *, prediction_enabled=True, station_type="water_level"):
    return {
        "id": station_id,
        "name": "Dã Viên" if station_type == "water_level" else "Hồng Thái",
        "station_type": station_type,
        "river_name": None,
        "latitude": None,
        "longitude": None,
        "address": None,
        "measurement_unit": "mm",
        "prediction_enabled": prediction_enabled,
        "enabled": True,
        "alarm_level_1_m": None,
        "alarm_level_2_m": None,
        "alarm_level_3_m": None,
    }


class StationsApiTests(TestCase):
    def setUp(self):
        self.engine = MagicMock()
        self.connection = self.engine.connect.return_value.__enter__.return_value
        self.rows = self.connection.execute.return_value.mappings.return_value
        self.rows.all.return_value = []
        self.application = create_app(
            engine=self.engine,
            registry=StationModelRegistry([TEST_CONFIG]),
        )
        self.client = self.enterContext(TestClient(self.application, raise_server_exceptions=False))

    def test_water_and_rain_metadata_and_permission(self):
        water = station_metadata(TEST_CONFIG.station_id)
        water.update(latitude=16.458944, longitude=107.571361, address="Cầu Dã Viên")
        rain = station_metadata("750062", prediction_enabled=False, station_type="rainfall")
        self.rows.all.return_value = [water, rain]
        response = self.client.get("/api/v1/stations")
        self.assertEqual(response.status_code, 200, response.text)
        water_payload, rain_payload = response.json()["stations"]
        self.assertEqual(water_payload["station_id"], TEST_CONFIG.station_id)
        self.assertEqual(water_payload["location"]["latitude"], 16.458944)
        self.assertTrue(water_payload["model_configured"])
        self.assertTrue(water_payload["prediction_enabled"])
        self.assertFalse(rain_payload["model_configured"])
        self.assertFalse(rain_payload["prediction_enabled"])
        self.assertEqual(rain_payload["measurement_unit"], "mm")
        self.assertIsNone(rain_payload["location"]["latitude"])
        self.assertEqual(self.application.state.prediction_services, {})
        self.assertNotIn("model_path", response.text)
        self.assertNotIn("source_name", response.text)

    def test_other_water_station_does_not_inherit_da_vien_model(self):
        self.rows.all.return_value = [station_metadata("other-water")]
        response = self.client.get("/api/v1/stations")
        self.assertEqual(response.status_code, 200)
        station = response.json()["stations"][0]
        self.assertFalse(station["model_configured"])
        self.assertFalse(station["prediction_enabled"])

    def test_configured_model_does_not_override_database_permission(self):
        self.rows.all.return_value = [
            station_metadata(TEST_CONFIG.station_id, prediction_enabled=False)
        ]
        station = self.client.get("/api/v1/stations").json()["stations"][0]
        self.assertTrue(station["model_configured"])
        self.assertFalse(station["prediction_enabled"])

    def test_empty_list_is_successful(self):
        response = self.client.get("/api/v1/stations")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"stations": []})

    def test_database_failure_uses_safe_error_envelope(self):
        self.connection.execute.side_effect = OperationalError(
            "SELECT", {}, Exception("password=secret")
        )
        response = self.client.get("/api/v1/stations")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "DATA_SERVICE_UNAVAILABLE")
        self.assertEqual(response.json()["request_id"], response.headers["X-Request-ID"])
        self.assertNotIn("secret", response.text)

    def test_openapi_includes_station_list(self):
        operation = self.client.get("/openapi.json").json()["paths"]["/api/v1/stations"]["get"]
        self.assertIn("503", operation["responses"])
        self.assertEqual(
            operation["responses"]["200"]["content"]["application/json"]["schema"],
            {"$ref": "#/components/schemas/StationsResponse"},
        )

    def test_detail_returns_stored_thresholds_in_metres(self):
        station = station_metadata(TEST_CONFIG.station_id)
        station.update(alarm_level_1_m=2.0, alarm_level_2_m=2.5, alarm_level_3_m=3.5)
        self.rows.first.return_value = station
        response = self.client.get(f"/api/v1/stations/{TEST_CONFIG.station_id}")
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["station_id"], TEST_CONFIG.station_id)
        self.assertEqual(
            payload["alarm_levels_m"], {"level_1": 2.0, "level_2": 2.5, "level_3": 3.5}
        )
        self.assertTrue(payload["enabled"])
        self.assertTrue(payload["prediction_enabled"])
        self.assertNotIn("model_path", payload)
        self.assertEqual(self.application.state.prediction_services, {})

    def test_rain_detail_preserves_unknown_thresholds_and_units(self):
        station = station_metadata("future-rain", station_type="rainfall", prediction_enabled=False)
        station["measurement_unit"] = None
        self.rows.first.return_value = station
        response = self.client.get("/api/v1/stations/future-rain")
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertIsNone(payload["measurement_unit"])
        self.assertEqual(
            payload["alarm_levels_m"], {"level_1": None, "level_2": None, "level_3": None}
        )
        self.assertFalse(payload["prediction_enabled"])

    def test_disabled_station_is_inspectable_but_prediction_is_disabled(self):
        station = station_metadata(TEST_CONFIG.station_id)
        station["enabled"] = False
        self.rows.first.return_value = station
        response = self.client.get(f"/api/v1/stations/{TEST_CONFIG.station_id}")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(response.json()["enabled"])
        self.assertFalse(response.json()["prediction_enabled"])
        self.assertTrue(response.json()["model_configured"])

    def test_unknown_station_detail_returns_404(self):
        self.rows.first.return_value = None
        response = self.client.get("/api/v1/stations/unknown")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "STATION_NOT_FOUND")
        self.assertEqual(response.json()["request_id"], response.headers["X-Request-ID"])

    def test_station_detail_database_error_is_safe(self):
        self.connection.execute.side_effect = OperationalError(
            "SELECT", {}, Exception("password=secret")
        )
        response = self.client.get("/api/v1/stations/unknown")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "DATA_SERVICE_UNAVAILABLE")
        self.assertNotIn("secret", response.text)

    def test_openapi_includes_station_detail(self):
        operation = self.client.get("/openapi.json").json()["paths"][
            "/api/v1/stations/{station_id}"
        ]["get"]
        self.assertIn("404", operation["responses"])
        self.assertIn("503", operation["responses"])


if __name__ == "__main__":
    main()
