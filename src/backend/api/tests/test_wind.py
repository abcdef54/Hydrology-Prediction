"""Wind latest/history HTTP responses expose raw values and unknown units honestly."""

from datetime import datetime, timedelta, timezone
from unittest import TestCase
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from src.backend.main import create_app
from src.backend.prediction.config import StationModelRegistry


class WindApiTests(TestCase):
    def setUp(self):
        self.engine = MagicMock()
        self.connection = self.engine.connect.return_value.__enter__.return_value
        self.station_result = MagicMock()
        self.station_result.mappings.return_value.first.return_value = {
            "station_type": "wind",
            "measurement_unit": None,
            "measurement_units": None,
        }
        self.readings_result = MagicMock()
        self.connection.execute.side_effect = [self.station_result, self.readings_result]
        self.at = datetime(2026, 10, 1, tzinfo=timezone.utc)
        self.row = {
            "observed_at": self.at,
            "received_at": self.at,
            "ws": None,
            "wd": 961,
            "wsg": 0,
            "wdg": 1292,
            "quality_status": "observed",
        }
        self.client = self.enterContext(
            TestClient(
                create_app(engine=self.engine, registry=StationModelRegistry([])),
                raise_server_exceptions=False,
            )
        )
        self.url = "/api/v1/stations/wind:228005"

    def test_latest_preserves_raw_angles_partial_values_and_unknown_units(self):
        self.readings_result.mappings.return_value.first.return_value = self.row
        response = self.client.get(self.url + "/latest")
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["station_type"], "wind")
        self.assertIsNone(payload["value"])
        self.assertEqual(
            payload["measurements"]["wd"],
            {
                "value": 961,
                "unit": None,
                "unit_confirmed": False,
            },
        )
        self.assertEqual(payload["measurements"]["wsg"]["value"], 0)

    def test_history_pages_all_fields_with_exclusive_cursor(self):
        self.readings_result.mappings.return_value.all.return_value = [
            self.row,
            {**self.row, "observed_at": self.at - timedelta(minutes=10)},
        ]
        response = self.client.get(
            self.url + "/observations",
            params={
                "start": "2026-09-30T00:00:00Z",
                "end": "2026-10-02T00:00:00Z",
                "limit": 1,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertTrue(payload["has_more"])
        self.assertEqual(payload["next_before"], "2026-10-01T00:00:00Z")
        self.assertEqual(payload["observations"][0]["measurements"]["wdg"]["value"], 1292)

    def test_latest_without_usable_reading_has_standard_404(self):
        self.readings_result.mappings.return_value.first.return_value = None
        response = self.client.get(self.url + "/latest")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "OBSERVATION_NOT_FOUND")
