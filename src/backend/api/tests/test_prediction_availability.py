"""Availability respects per-model lengths and the shared preparation service."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from src.backend.api.prediction_inputs import get_input_service
from src.backend.main import create_app
from src.backend.prediction.config import StationModelRegistry
from src.backend.prediction.inputs import InsufficientHistoryError
from src.backend.prediction.tests.test_inputs import TEST_CONFIG


class PredictionAvailabilityTests(TestCase):
    def setUp(self):
        self.application = create_app(engine=MagicMock(), registry=StationModelRegistry([]))
        self.service = MagicMock()
        self.service.config = replace(TEST_CONFIG, sequence_length=72)
        self.application.dependency_overrides[get_input_service] = lambda: (self.service, {})
        self.client = self.enterContext(TestClient(self.application, raise_server_exceptions=False))
        self.url = f"/api/v1/stations/{TEST_CONFIG.station_id}/prediction-availability"
        self.start = datetime(2026, 9, 13, 2, tzinfo=timezone.utc)
        self.parameters = {
            "start": self.start.isoformat(),
            "end": (self.start + timedelta(hours=2)).isoformat(),
        }
        self.prepared = SimpleNamespace(timestamps=list(range(72)), anchor_observed_at=self.start)
        self.service.prepare.return_value = self.prepared

    def test_configured_length_and_older_anchor_are_used(self):
        self.prepared.anchor_observed_at = self.start - timedelta(hours=4)
        response = self.client.get(self.url, params=self.parameters)
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["sequence_length"], 72)
        self.assertEqual(payload["model_version"], TEST_CONFIG.model_version)
        self.assertEqual(len(payload["availability"]), 2)
        for point in payload["availability"]:
            self.assertTrue(point["input_ready"])
            self.assertEqual(point["available_timesteps"], 72)
            self.assertEqual(point["anchor_observed_at"], "2026-09-12T22:00:00Z")
            self.assertIsNone(point["reason"])
        self.assertEqual(self.service.prepare.call_args_list[0].args, (self.start,))
        self.assertEqual(
            self.service.prepare.call_args_list[1].args, (self.start + timedelta(hours=1),)
        )
        self.assertEqual(self.application.state.prediction_services, {})

    def test_insufficient_hour_does_not_hide_next_ready_hour(self):
        self.service.prepare.side_effect = [InsufficientHistoryError(71, 72), self.prepared]
        response = self.client.get(self.url, params=self.parameters)
        self.assertEqual(response.status_code, 200, response.text)
        first, second = response.json()["availability"]
        self.assertFalse(first["input_ready"])
        self.assertEqual(first["available_timesteps"], 71)
        self.assertEqual(first["reason"], "INSUFFICIENT_HISTORY")
        self.assertIsNone(first["anchor_observed_at"])
        self.assertTrue(second["input_ready"])

    def test_only_hour_boundaries_inside_range_are_checked(self):
        response = self.client.get(
            self.url,
            params={"start": "2026-09-13T09:30:00+07:00", "end": "2026-09-13T11:00:00+07:00"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            [point["reference_time"] for point in response.json()["availability"]],
            ["2026-09-13T03:00:00Z"],
        )
        self.service.prepare.assert_called_once_with(self.start + timedelta(hours=1))

    def test_subhour_range_can_have_no_hourly_points(self):
        response = self.client.get(
            self.url, params={"start": "2026-09-13T02:10:00Z", "end": "2026-09-13T02:40:00Z"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["availability"], [])
        self.service.prepare.assert_not_called()

    def test_invalid_ranges_and_naive_times_are_422(self):
        invalid_parameters = [
            {},
            {**self.parameters, "end": self.parameters["start"]},
            {**self.parameters, "end": "2026-09-21T02:00:00Z"},
            {**self.parameters, "start": "2026-09-13T02:00:00"},
            {**self.parameters, "end": "2999-01-01T00:00:00Z"},
            {**self.parameters, "extra": "invalid"},
        ]
        for parameters in invalid_parameters:
            with self.subTest(parameters=parameters):
                response = self.client.get(self.url, params=parameters)
                self.assertEqual(response.status_code, 422, response.text)
        self.service.prepare.assert_not_called()

    def test_input_failure_is_not_reported_as_insufficient_history(self):
        self.service.prepare.side_effect = ValueError("private/path")
        response = self.client.get(self.url, params=self.parameters)
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private/path", response.text)

    def test_database_failure_uses_safe_error(self):
        self.service.prepare.side_effect = OperationalError(
            "SELECT", {}, Exception("password=secret")
        )
        response = self.client.get(self.url, params=self.parameters)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "DATA_SERVICE_UNAVAILABLE")
        self.assertNotIn("secret", response.text)


if __name__ == "__main__":
    main()
