"""SOHOA identity, deleted states, timezone semantics and bounded paging."""

from datetime import date, datetime, timedelta, timezone
from unittest import TestCase
from unittest.mock import MagicMock, patch
from src.backend.ingestion.facilities import (
    normalize_facility,
    parse_operation,
    parse_observation_time,
)
from src.backend.ingestion.facility_source import FacilitySourceClient, FacilitySourceSettings
from src.backend.ingestion.worker import SourceError
from src.backend.ingestion.facility_import import sync_facilities


def report(**changes):
    return {
        "id": 1,
        "tencongtrinh": "1",
        "chucnang": "1",
        "ngaylaysolieu": "14/10/2026",
        "gio": "7:00",
        "ngaycapnhat": "10/14/2026 8:12:30 AM",
        "trangthai": "1",
        "htl": "0",
        "hhl": "-0.03",
        "qden": "12.5",
        "qdi": "",
        "socuamo": "0/15",
        "username": "private-user",
        "hovaten": "private-name",
        "ghichu": "private-note",
        **changes,
    }


class FacilityParserTests(TestCase):
    def test_river_links_existing_station_and_keeps_original_specification(self):
        row = normalize_facility(
            {
                "id": 16,
                "loaicongtrinhid": "4",
                "tencongtrinh": "Trạm Dã Viên (Sông Hương)",
                "baodongmuc1": "1",
            }
        )
        self.assertEqual(row["station_id"], "0074647403")
        self.assertIsNone(row["latitude"])
        self.assertEqual(row["specifications"]["baodongmuc1"], "1")
        self.assertTrue(all(unit is None for unit in row["measurement_units"].values()))
        with self.assertRaises(SourceError):
            normalize_facility({"id": 99, "loaicongtrinhid": "4", "tencongtrinh": "Trạm chưa biết"})

    def test_source_times_are_separate_and_vietnam_wall_time_becomes_utc(self):
        row = parse_operation(report())
        self.assertEqual(row["observed_at"], datetime(2026, 10, 14, tzinfo=timezone.utc))
        self.assertEqual(
            row["source_updated_at"], datetime(2026, 10, 14, 1, 12, 30, tzinfo=timezone.utc)
        )
        self.assertEqual(row["htl"], 0)
        self.assertEqual(row["hhl"], -0.03)
        self.assertIsNone(row["qdi"])
        self.assertEqual(row["gate_opening_text"], "0/15")
        self.assertFalse(set(row) & {"username", "hovaten", "ghichu", "fildinhkem"})

    def test_invalid_time_is_retained_without_using_update_time_as_observation(self):
        for hour in ("", "7h:00", "3,:00", "23:", "11", "24:00"):
            with self.subTest(hour=hour):
                row = parse_operation(report(gio=hour))
                self.assertIsNone(row["observed_at"])
                self.assertEqual(row["time_status"], "invalid")
                self.assertEqual(row["observation_hour_text"], hour)
        self.assertIsNone(parse_observation_time("31/02/2026", "01:00"))

    def test_deleted_and_unknown_status_are_not_lost_and_bad_numbers_fail(self):
        self.assertEqual(parse_operation(report(trangthai="2"))["source_status"], "2")
        self.assertIsNone(parse_operation(report(trangthai=""))["source_status"])
        for value in (True, "nan", "Infinity", "12 m", "1/15"):
            with self.subTest(value=value), self.assertRaises(SourceError):
                parse_operation(report(htl=value))


class FacilitySourceTests(TestCase):
    def client(self):
        return FacilitySourceClient(
            FacilitySourceSettings("https://source.invalid", "private-token"), session=MagicMock()
        )

    def test_short_pages_continue_until_total_and_include_deleted(self):
        client = self.client()
        client.request_page = MagicMock(
            side_effect=[([report()], 2), ([report(id=2, trangthai="2")], 2)]
        )
        rows = client.read_all("public-service", {})
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["trangthai"], "2")
        self.assertNotIn("private-token", repr(client.settings))

    def test_changed_total_duplicate_or_early_empty_page_fails(self):
        for pages in (
            [([report()], 2), ([], 2)],
            [([report()], 2), ([report()], 2)],
            [([report()], 2), ([report(id=2)], 3)],
        ):
            client = self.client()
            client.request_page = MagicMock(side_effect=pages)
            with self.subTest(pages=pages), self.assertRaises(SourceError):
                client.read_all("service", {})

    def test_provider_business_error_does_not_become_empty_data(self):
        client = self.client()
        response = MagicMock(status_code=200)
        response.json.return_value = {
            "code": 1,
            "message": "private-token",
            "data": [],
            "totalrow": 0,
        }
        client.session.post.return_value = response
        with self.assertRaises(SourceError) as raised:
            client.request_page("service", {}, 1)
        self.assertNotIn("private-token", str(raised.exception))


class FacilitySyncTests(TestCase):
    def test_unconfigured_database_does_not_read_private_token_or_call_source(self):
        engine = MagicMock()
        engine.connect.return_value.__enter__.return_value.execute.return_value.mappings.return_value.all.return_value = (
            []
        )
        with patch(
            "src.backend.ingestion.facility_import.FacilitySourceSettings.from_env"
        ) as settings:
            self.assertEqual(sync_facilities(engine, MagicMock())["status"], "not_configured")
            settings.assert_not_called()

    def test_daily_full_scan_and_recent_window_do_not_advance_on_dry_run(self):
        for previous, full in (
            (None, True),
            (datetime.now(timezone.utc) - timedelta(days=2), True),
            (datetime.now(timezone.utc), False),
        ):
            engine = MagicMock()
            engine.connect.return_value.__enter__.return_value.execute.return_value.mappings.return_value.all.return_value = [
                {"id": "sohoa:1", "last_full_sync_at": previous}
            ]
            client = MagicMock()
            client.catalog.return_value = [
                {"id": 1, "loaicongtrinhid": "1", "tencongtrinh": "Hương Điền"}
            ]
            client.operations.return_value = [report(trangthai="2")]
            with self.subTest(full=full):
                result = sync_facilities(engine, MagicMock(), dry_run=True, client=client)
                self.assertEqual(result["full_scan"], full)
                self.assertEqual(result["fetched"], 1)
                self.assertEqual(client.operations.call_args.args[0] == date(1900, 1, 1), full)
                engine.begin.assert_not_called()
