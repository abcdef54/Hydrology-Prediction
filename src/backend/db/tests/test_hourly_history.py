"""PostgreSQL query checks using only transaction-scoped temporary tables."""

from datetime import datetime, timedelta, timezone
from unittest import TestCase, main
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
import sqlalchemy as sa

from src.backend.cache.latest import LatestObservationCache
from src.backend.db.session import build_engine
from src.backend.main import create_app
from src.backend.prediction.config import StationModelRegistry
from src.backend.security.config import SecuritySettings
from src.backend.security.rate_limit import RequestRateLimiter


class HourlyHistoryPostgresTests(TestCase):
    def setUp(self):
        engine = build_engine()
        self.addCleanup(engine.dispose)
        connection = engine.connect().execution_options(
            schema_translate_map={"hydrology": "pg_temp"}
        )
        self.addCleanup(connection.close)
        transaction = connection.begin()
        self.addCleanup(transaction.rollback)
        connection.execute(
            sa.text(
                "CREATE TEMP TABLE stations (id text PRIMARY KEY, name text, "
                "station_type text, measurement_unit text, enabled boolean) ON COMMIT DROP"
            )
        )
        connection.execute(
            sa.text(
                "CREATE TEMP TABLE observations (station_id text, observed_at timestamptz, "
                "depth double precision, quality_status text, "
                "PRIMARY KEY (station_id, observed_at)) ON COMMIT DROP"
            )
        )
        connection.execute(
            sa.text(
                "INSERT INTO pg_temp.stations VALUES "
                "('0074647403', 'Dã Viên', 'water_level', 'mm', true), "
                "('750062', 'Hồng Thái 50%_test', 'rainfall', NULL, true), "
                "('999', 'Disabled', 'water_level', 'mm', false)"
            )
        )
        midnight = datetime(2020, 1, 1, 17, tzinfo=timezone.utc)
        values = [
            ("0074647403", midnight - timedelta(hours=1), 43, "observed"),
            ("0074647403", midnight, -25.5, "observed"),
            ("0074647403", midnight + timedelta(minutes=10), 999, "observed"),
            ("0074647403", midnight + timedelta(hours=1, microseconds=1), 998, "observed"),
            ("0074647403", midnight + timedelta(hours=2), 0, "observed"),
            ("0074647403", midnight + timedelta(hours=3), None, "missing"),
            ("750062", midnight, 1.4, "observed"),
        ]
        connection.execute(
            sa.text("INSERT INTO pg_temp.observations VALUES (:station, :at, :value, :quality)"),
            [
                {"station": station, "at": at, "value": value, "quality": quality}
                for station, at, value, quality in values
            ],
        )

        # Requests share this one transaction; schema translation prevents any
        # query from reaching the application's real hydrology tables.
        engine_proxy = MagicMock()
        engine_proxy.connect.return_value.__enter__.return_value = connection
        settings = SecuritySettings()
        application = create_app(
            engine=engine_proxy,
            registry=StationModelRegistry([]),
            latest_cache=LatestObservationCache(),
            security_settings=settings,
            rate_limiter=RequestRateLimiter(settings),
        )
        self.client = self.enterContext(TestClient(application, raise_server_exceptions=False))
        self.url = "/api/v1/observations/hourly"

    def get_history(self, **parameters):
        response = self.client.get(self.url, params={"start_date": "2020-01-02", **parameters})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_exact_sql_lookup_dates_units_and_station_filters(self):
        water, rain = self.get_history()["rows"]
        self.assertEqual(water["cells"][0]["at"], "2020-01-01T17:00:00Z")
        self.assertEqual(water["cells"][0]["value"], -25.5)
        self.assertIsNone(water["cells"][1]["value"])
        self.assertEqual(water["cells"][2]["value"], 0)
        self.assertEqual(water["cells"][3]["status"], "missing")
        self.assertEqual(rain["cells"][0]["value"], 1.4)
        self.assertIsNone(rain["unit"])
        for filters in (
            {"station_type": "water_level"},
            {"station_id": "0074647403"},
            {"station_name": "viên"},
        ):
            with self.subTest(filters=filters):
                self.assertEqual(
                    [row["station_id"] for row in self.get_history(**filters)["rows"]],
                    ["0074647403"],
                )
        # Search metacharacters are literal, never SQL LIKE wildcards.
        for name in ("%", "_"):
            self.assertEqual(
                [row["station_id"] for row in self.get_history(station_name=name)["rows"]],
                ["750062"],
            )
        self.assertEqual(self.get_history(station_id="999")["rows"], [])
        self.assertEqual(self.get_history(station_name="missing")["rows"], [])
        self.assertEqual(self.get_history(station_type="water_level", station_name="%")["rows"], [])

    def test_cursor_covers_station_days_and_local_midnight_boundary(self):
        parameters = {"start_date": "2020-01-01", "end_date": "2020-01-02", "limit": 1}
        seen = []
        while True:
            payload = self.get_history(**parameters)
            self.assertEqual(len(payload["rows"]), 1)
            row = payload["rows"][0]
            key = (row["date"], row["station_id"])
            self.assertNotIn(key, seen)
            seen.append(key)
            if key == ("2020-01-01", "0074647403"):
                self.assertEqual(row["cells"][23]["value"], 43)
            if not payload["has_more"]:
                self.assertIsNone(payload["next_cursor"])
                break
            cursor = payload["next_cursor"]
            parameters.update(cursor_date=cursor["date"], cursor_station_id=cursor["station_id"])
            self.assertLess(len(seen), 5)
        self.assertEqual(
            seen,
            [
                ("2020-01-02", "0074647403"),
                ("2020-01-02", "750062"),
                ("2020-01-01", "0074647403"),
                ("2020-01-01", "750062"),
            ],
        )


if __name__ == "__main__":
    main()
