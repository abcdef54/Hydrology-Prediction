"""Station/day paging, hourly mean water semantics and safe API failures."""

from datetime import datetime, timedelta, timezone
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from src.backend.main import create_app
from src.backend.prediction.config import StationModelRegistry


class HourlyHistoryTests(TestCase):
    def setUp(self):
        self.engine = MagicMock()
        self.connection = self.engine.connect.return_value.__enter__.return_value
        self.stations = [
            {
                "id": "0074647403",
                "name": "Dã Viên",
                "station_type": "water_level",
                "measurement_unit": "mm",
            },
            {
                "id": "750062",
                "name": "Hồng Thái",
                "station_type": "rainfall",
                "measurement_unit": "mm",
            },
        ]
        # Vietnam is already 14 October, while the UTC date is still 13 October.
        self.now = datetime(2026, 10, 13, 18, 30, tzinfo=timezone.utc)
        self.records = []
        self.application = create_app(engine=self.engine, registry=StationModelRegistry([]))
        self.client = self.enterContext(TestClient(self.application, raise_server_exceptions=False))
        self.clock = self.enterContext(patch("src.backend.api.hourly_history.datetime"))
        self.clock.now.return_value = self.now
        # Keep calendar construction real while controlling only the current clock.
        self.clock.combine.side_effect = datetime.combine
        self.url = "/api/v1/observations/hourly"

    def get_history(self, parameters=None):
        station_result = MagicMock()
        station_result.mappings.return_value.all.return_value = self.stations
        record_result = MagicMock()
        record_result.mappings.return_value.all.return_value = self.records
        means = {}
        for record in self.records:
            at = record["observed_at"]
            hour = at.replace(minute=0, second=0, microsecond=0)
            if at != hour:
                hour += timedelta(hours=1)
            if record["depth"] is not None:
                means.setdefault((record["station_id"], hour), []).append(record["depth"])
        mean_result = MagicMock()
        mean_result.mappings.return_value.all.return_value = [
            {"station_id": sid, "at": at, "depth": sum(values) if sid == "750062" else sum(values) / len(values)}
            for (sid, at), values in means.items()
        ]
        self.connection.execute.side_effect = [station_result, mean_result, record_result]
        return self.client.get(self.url, params=parameters or {})

    def record(self, at, value, station_id="0074647403", quality_status="observed"):
        return {
            "station_id": station_id,
            "observed_at": at,
            "depth": value,
            "quality_status": quality_status,
        }

    def test_today_uses_vietnam_date_and_preserves_zero_negative_missing_and_future(self):
        midnight = datetime(2026, 10, 13, 17, tzinfo=timezone.utc)
        self.records = [
            self.record(midnight, -25.5),
            self.record(midnight + timedelta(hours=1), 0),
            self.record(midnight + timedelta(minutes=10), 999),
            self.record(midnight + timedelta(hours=2), 999),
        ]
        response = self.get_history()
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["start_date"], "2026-10-14")
        self.assertEqual(payload["end_date"], "2026-10-14")
        self.assertEqual(payload["timezone"], "Asia/Ho_Chi_Minh")
        self.assertEqual(payload["as_of"], "2026-10-13T18:30:00Z")
        water, rain = payload["rows"]
        self.assertEqual(len(water["cells"]), 24)
        self.assertEqual(
            water["cells"][0],
            {"hour": 0, "at": "2026-10-13T17:00:00Z", "value": -25.5, "status": "observed"},
        )
        self.assertEqual(water["cells"][1]["value"], 499.5)
        self.assertEqual(water["cells"][2]["status"], "future")
        self.assertIsNone(water["cells"][2]["value"])
        self.assertEqual(rain["cells"][0]["status"], "missing")
        self.assertIsNone(rain["cells"][0]["value"])
        self.assertEqual(rain["unit"], "mm")
        self.assertEqual(self.application.state.prediction_services, {})
        mean_query = str(self.connection.execute.call_args_list[1].args[0])
        self.assertIn("avg(", mean_query)
        self.assertIn("observed_at >", mean_query)
        self.assertIn("observed_at <=", mean_query)
        # Raw rain/wind queries retain exact keys.
        observation_query = self.connection.execute.call_args_list[-1].args[0]
        keys = list(observation_query.compile().params.values())[0]
        self.assertEqual(len(keys), 4)
        self.assertTrue(all(at.minute == at.second == at.microsecond == 0 for _, at in keys))

    def test_mean_uses_non_clock_readings_without_filling_empty_bins(self):
        midnight = datetime(2026, 10, 12, 17, tzinfo=timezone.utc)
        self.records = [
            self.record(midnight, None, quality_status="missing"),
            self.record(midnight + timedelta(minutes=10), 123),
            self.record(midnight + timedelta(hours=1, microseconds=1), 456),
        ]
        response = self.get_history({"start_date": "2026-10-13"})
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["end_date"], "2026-10-13")
        cells = payload["rows"][0]["cells"]
        self.assertEqual(cells[0]["status"], "missing")
        self.assertEqual(cells[1]["value"], 123)
        self.assertEqual(cells[2]["value"], 456)
        self.assertTrue(all(cell["status"] == "missing" for cell in cells[3:]))
        self.assertEqual(payload["rows"][0]["cells"][-1]["at"], "2026-10-13T16:00:00Z")

    def test_rain_sums_preceding_hour_preserves_zero_and_missing(self):
        midnight = datetime(2026, 10, 12, 17, tzinfo=timezone.utc)
        self.records = [
            self.record(midnight - timedelta(hours=1), 99, "750062"),
            self.record(midnight - timedelta(minutes=50), 0.2, "750062"),
            self.record(midnight, 0.3, "750062"),
            self.record(midnight + timedelta(minutes=10), 0, "750062"),
        ]
        response = self.get_history({"start_date": "2026-10-13"})
        self.assertEqual(response.status_code, 200, response.text)
        cells = response.json()["rows"][1]["cells"]
        self.assertAlmostEqual(cells[0]["value"], 0.5)
        self.assertEqual(cells[1]["value"], 0)
        self.assertEqual(cells[1]["status"], "observed")
        self.assertEqual(cells[2]["status"], "missing")
        self.assertIsNone(cells[2]["value"])
        self.assertIn("sum(", str(self.connection.execute.call_args_list[1].args[0]))

    def test_cursor_pages_all_station_days_once_including_empty_days(self):
        parameters = {"start_date": "2026-10-13", "end_date": "2026-10-14", "limit": 1}
        seen = []
        for _ in range(4):
            response = self.get_history(parameters)
            self.assertEqual(response.status_code, 200, response.text)
            payload = response.json()
            self.assertEqual(len(payload["rows"]), 1)
            row = payload["rows"][0]
            seen.append((row["date"], row["station_id"]))
            if payload["has_more"]:
                cursor = payload["next_cursor"]
                parameters.update(
                    cursor_date=cursor["date"], cursor_station_id=cursor["station_id"]
                )
            else:
                self.assertIsNone(payload["next_cursor"])
        self.assertEqual(
            seen,
            [
                ("2026-10-14", "0074647403"),
                ("2026-10-14", "750062"),
                ("2026-10-13", "0074647403"),
                ("2026-10-13", "750062"),
            ],
        )

    def test_multi_year_range_builds_only_requested_rows(self):
        response = self.get_history(
            {"start_date": "1900-01-01", "end_date": "2026-10-13", "limit": 2}
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(response.json()["rows"]), 2)
        self.assertTrue(response.json()["has_more"])

    def test_unmatched_filters_return_empty_without_observation_query(self):
        self.stations = []
        response = self.get_history({"station_name": "không tồn tại"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["rows"], [])
        self.assertFalse(response.json()["has_more"])
        self.connection.execute.assert_called_once()

    def test_invalid_dates_filters_cursors_and_limits_fail_before_database(self):
        invalid = [
            {"start_date": "2026-10-14", "end_date": "2026-10-13"},
            {"start_date": "2026-10-15"},
            {"end_date": "2999-01-01"},
            {"start_date": "invalid"},
            {"start_date": "0001-01-01"},
            {"station_type": "unsupported_type"},
            {"station_name": ""},
            {"station_id": ""},
            {"limit": 0},
            {"limit": 101},
            {"extra": "bad"},
            {"cursor_date": "2026-10-14"},
            {"cursor_station_id": "750062"},
            {"cursor_date": "2026-10-12", "cursor_station_id": "750062"},
        ]
        for parameters in invalid:
            with self.subTest(parameters=parameters):
                response = self.get_history(parameters)
                self.assertEqual(response.status_code, 422, response.text)
                self.assertEqual(response.json()["error"]["code"], "INVALID_REQUEST")
        self.connection.execute.assert_not_called()

    def test_database_errors_are_safe(self):
        self.connection.execute.side_effect = OperationalError(
            "SELECT", {}, Exception("password=secret")
        )
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("secret", response.text)
        self.assertEqual(response.json()["request_id"], response.headers["X-Request-ID"])

    def test_wind_filter_uses_raw_ws_at_exact_hour_without_assuming_unit(self):
        self.stations = [
            {
                "id": "wind:228005",
                "name": "Cảng Chân Mây",
                "station_type": "wind",
                "measurement_unit": None,
            }
        ]
        stations = MagicMock()
        stations.mappings.return_value.all.return_value = self.stations
        depth = MagicMock()
        depth.mappings.return_value.all.return_value = []
        wind = MagicMock()
        wind.mappings.return_value.all.return_value = [
            self.record(
                datetime(2026, 10, 13, 17, tzinfo=timezone.utc),
                0,
                "wind:228005",
            )
        ]
        self.connection.execute.side_effect = [stations, depth, wind]
        response = self.client.get(self.url, params={"station_type": "wind"})
        self.assertEqual(response.status_code, 200, response.text)
        row = response.json()["rows"][0]
        self.assertEqual(row["station_type"], "wind")
        self.assertIsNone(row["unit"])
        self.assertEqual(row["cells"][0]["value"], 0)
        self.assertIn("wind_observations", str(self.connection.execute.call_args.args[0]))

    def test_openapi_describes_filters_dates_and_row_cursor(self):
        operation = self.client.get("/openapi.json").json()["paths"][self.url]["get"]
        self.assertEqual(
            {parameter["name"] for parameter in operation["parameters"]},
            {
                "start_date",
                "end_date",
                "station_type",
                "station_id",
                "station_name",
                "limit",
                "cursor_date",
                "cursor_station_id",
            },
        )
        self.assertIn("422", operation["responses"])
        self.assertIn("503", operation["responses"])


if __name__ == "__main__":
    main()
