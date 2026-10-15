"""Facility HTTP contracts: linked identities, raw operations, audit and safe failures."""

from datetime import datetime, timezone
from unittest import TestCase
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError
from src.backend.main import create_app
from src.backend.prediction.config import StationModelRegistry
from src.backend.ingestion.facilities import normalize_facility, parse_operation
from src.backend.ingestion.tests.test_facilities import report


class FacilityApiTests(TestCase):
    def setUp(self):
        self.engine = MagicMock()
        self.connection = self.engine.connect.return_value.__enter__.return_value
        self.facility = normalize_facility(
            {"id": 1, "loaicongtrinhid": "1", "tencongtrinh": "Hương Điền", "mndbt": "+58 m"}
        )
        self.row = {
            **parse_operation(report()),
            "received_at": datetime(2026, 10, 14, tzinfo=timezone.utc),
        }
        self.facility_result = MagicMock()
        self.facility_result.mappings.return_value.first.return_value = self.facility
        self.rows = MagicMock()
        self.client = self.enterContext(
            TestClient(
                create_app(engine=self.engine, registry=StationModelRegistry([])),
                raise_server_exceptions=False,
            )
        )
        self.url = "/api/v1/facilities/sohoa:1"

    def test_default_list_excludes_linked_rivers_and_preserves_specs(self):
        self.facility_result.mappings.return_value.all.return_value = [self.facility]
        self.connection.execute.return_value = self.facility_result
        response = self.client.get("/api/v1/facilities")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["facilities"][0]["specifications"]["mndbt"], "+58 m")
        self.assertIn("station_id IS NULL", str(self.connection.execute.call_args.args[0]))

    def test_station_filter_reuses_existing_station_id(self):
        self.facility_result.mappings.return_value.all.return_value = []
        self.connection.execute.return_value = self.facility_result
        response = self.client.get("/api/v1/facilities", params={"station_id": "0074647403"})
        self.assertEqual(response.status_code, 200)
        self.assertIn(
            "0074647403", self.connection.execute.call_args.args[0].compile().params.values()
        )

    def test_latest_skips_inactive_invalid_future_and_preserves_zero(self):
        self.rows.mappings.return_value.first.return_value = self.row
        self.connection.execute.side_effect = [self.facility_result, self.rows]
        response = self.client.get(self.url + "/latest")
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()["operation"]
        self.assertEqual(
            payload["measurements"]["htl"], {"value": 0, "unit": None, "unit_confirmed": False}
        )
        self.assertNotIn("username", payload)
        statement = self.connection.execute.call_args.args[0]
        self.assertIn("source_status", str(statement))
        self.assertIn("1", statement.compile().params.values())
        self.assertIn("valid", statement.compile().params.values())
        self.assertIn("observed_at <=", str(statement))

    def test_history_keeps_same_time_reports_and_uses_unique_id_cursor(self):
        self.rows.mappings.return_value.all.return_value = [
            {**self.row, "source_record_id": 2},
            self.row,
        ]
        self.connection.execute.side_effect = [self.facility_result, self.rows]
        response = self.client.get(self.url + "/operations", params={"limit": 1})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["has_more"])
        self.assertEqual(response.json()["next_before_id"], 2)
        self.assertEqual(response.json()["operations"][0]["source_record_id"], 2)

    def test_audit_can_read_deleted_invalid_time_without_inventing_timestamp(self):
        self.rows.mappings.return_value.all.return_value = [
            {
                **self.row,
                "source_status": "2",
                "observed_at": None,
                "time_status": "invalid",
                "observation_hour_text": "",
            }
        ]
        self.connection.execute.side_effect = [self.facility_result, self.rows]
        response = self.client.get(
            self.url + "/operations",
            params={"include_inactive": "true", "include_invalid_time": "true"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        row = response.json()["operations"][0]
        self.assertTrue(row["is_deleted"])
        self.assertFalse(row["is_visible"])
        self.assertIsNone(row["observed_at"])
        self.assertNotIn("source_status =", str(self.connection.execute.call_args.args[0]))

    def test_unaware_or_invalid_bounds_are_rejected_before_database(self):
        for params in (
            {"start": "2026-10-01T00:00:00"},
            {"start": "2026-10-02T00:00:00Z", "end": "2026-10-01T00:00:00Z"},
            {"end": "2999-01-01T00:00:00Z"},
            {"limit": 1001},
            {"before_id": 0},
        ):
            with self.subTest(params=params):
                response = self.client.get(self.url + "/operations", params=params)
                self.assertEqual(response.status_code, 422)
        self.connection.execute.assert_not_called()

    def test_missing_report_returns_distinct_error(self):
        self.rows.mappings.return_value.first.return_value = None
        self.connection.execute.side_effect = [self.facility_result, self.rows]
        response = self.client.get(self.url + "/latest")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "OPERATION_NOT_FOUND")

    def test_database_error_does_not_expose_credentials(self):
        self.connection.execute.side_effect = OperationalError(
            "SELECT", {}, Exception("private-token")
        )
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private-token", response.text)
