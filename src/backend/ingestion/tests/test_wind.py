"""Raw wind identity, numeric preservation and SID-based source pagination."""

from datetime import datetime, timedelta, timezone
from unittest import TestCase
from unittest.mock import MagicMock
from threading import Event

from src.backend.ingestion.wind import normalize_wind_station, parse_wind_row
from src.backend.ingestion.worker import HueSourceClient, SourceError, SyncSettings

START = datetime(2026, 10, 1, tzinfo=timezone.utc)
END = START + timedelta(hours=1)
SOURCE_ID = "63586eaa-f1cc-11f0-8f75-0692ec44dcd5"
STATION = {
    "id": "wind:228005",
    "source_station_id": SOURCE_ID,
    "source_name": "hue_dcu",
    "observation_collection": "iot_wind_speed",
}


def record(**changes):
    payload = {
        "id": 17,
        "sid": SOURCE_ID,
        "time_point": START.isoformat(),
        "ws": 0,
        "wd": 961,
        "wsg": 139.7,
        "wdg": 1292,
    }
    return {**payload, **changes}


class WindSourceTests(TestCase):
    def test_namespaced_code_does_not_collide_with_rain_and_preserves_uuid(self):
        station = normalize_wind_station(
            {
                "id": SOURCE_ID,
                "code": "228005",
                "name": "Cảng Chân Mây",
                "geom": {"type": "Point", "coordinates": [108.020278, 16.3325]},
            }
        )
        self.assertEqual(station["id"], "wind:228005")
        self.assertEqual(station["source_station_id"], SOURCE_ID)
        self.assertTrue(station["sync_enabled"])
        self.assertFalse(station["enabled"])
        self.assertFalse(station["prediction_enabled"])
        self.assertEqual(station["measurement_units"], dict.fromkeys(("ws", "wd", "wsg", "wdg")))

    def test_raw_scale_zero_and_partial_missing_are_preserved(self):
        row = parse_wind_row(record(ws=None), STATION, START, END)
        self.assertIsNone(row["ws"])
        self.assertEqual(row["wd"], 961)
        self.assertEqual(row["wdg"], 1292)
        self.assertEqual(row["quality_status"], "observed")
        self.assertEqual(parse_wind_row(record(), STATION, START, END)["ws"], 0)
        missing = parse_wind_row(record(ws=None, wd=None, wsg=None, wdg=None), STATION, START, END)
        self.assertEqual(missing["quality_status"], "missing")

    def test_invalid_fields_identity_and_bounds_fail_instead_of_guessing(self):
        for change in (
            {"ws": True},
            {"wd": float("nan")},
            {"wdg": "961"},
            {"wsg": float("inf")},
            {"sid": "228005"},
            {"id": True},
            {"time_point": "2026-10-01T00:00:00"},
            {"time_point": (END + timedelta(seconds=1)).isoformat()},
        ):
            with self.subTest(change=change), self.assertRaises(SourceError):
                parse_wind_row(record(**change), STATION, START, END)

    def test_source_client_filters_by_uuid_sid_and_continues_short_pages(self):
        session = MagicMock()
        session.headers = {}
        responses = []
        for rows in ([record()], []):
            response = MagicMock(status_code=200, headers={})
            response.json.return_value = {"data": rows}
            responses.append(response)
        session.request.side_effect = responses
        client = HueSourceClient(
            SyncSettings("https://source.invalid", "test", "test"), Event(), session
        )
        self.assertEqual(len(client.observations(STATION, START, END)), 1)
        params = session.request.call_args_list[0].kwargs["params"]
        self.assertEqual(params["filter[sid][_in]"], SOURCE_ID)
        self.assertNotIn("filter[station_id][_in]", params)
        self.assertEqual(session.request.call_args_list[1].kwargs["params"]["filter[id][_gt]"], 17)
