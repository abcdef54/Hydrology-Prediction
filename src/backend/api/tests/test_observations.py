"""Latest-observation contract and freshness boundaries."""

from datetime import datetime, timedelta, timezone
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError
from redis.exceptions import ConnectionError

from src.backend.main import create_app
from src.backend.cache.latest import CachedObservation, LatestObservationCache
from src.backend.prediction.config import StationModelRegistry


class LatestObservationTests(TestCase):
    def setUp(self):
        self.now = datetime(2026, 10, 13, 7, tzinfo=timezone.utc)
        self.engine = MagicMock()
        self.connection = self.engine.connect.return_value.__enter__.return_value
        self.station = {"station_type": "water_level", "measurement_unit": "mm"}
        self.observation = {
            "depth": -25.5,
            "observed_at": self.now - timedelta(minutes=30),
            "received_at": self.now - timedelta(minutes=25),
            "quality_status": "observed",
        }
        self.cache = LatestObservationCache()
        self.application = create_app(
            engine=self.engine, registry=StationModelRegistry([]), latest_cache=self.cache
        )
        self.client = self.enterContext(TestClient(self.application, raise_server_exceptions=False))
        self.clock = self.enterContext(patch("src.backend.api.observations.datetime"))
        self.clock.now.return_value = self.now
        self.url = "/api/v1/stations/0074647403/latest"

    def request_latest(self, station=None, observation=None):
        station_result = MagicMock()
        station_result.mappings.return_value.first.return_value = station or self.station
        observation_result = MagicMock()
        observation_result.mappings.return_value.first.return_value = (
            observation or self.observation
        )
        self.connection.execute.side_effect = [station_result, observation_result]
        return self.client.get(self.url)

    def test_returns_raw_negative_water_and_utc_timestamps_without_model(self):
        response = self.request_latest()
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["value"], -25.5)
        self.assertEqual(payload["unit"], "mm")
        self.assertEqual(payload["observed_at"], "2026-10-13T06:30:00Z")
        self.assertEqual(payload["received_at"], "2026-10-13T06:35:00Z")
        self.assertEqual(payload["data_age_minutes"], 30)
        self.assertFalse(payload["is_stale"])
        self.assertEqual(self.application.state.prediction_services, {})

    def test_exactly_two_hours_is_fresh_but_one_second_later_is_stale(self):
        for seconds, expected in ((7200, False), (7201, True)):
            with self.subTest(seconds=seconds):
                self.observation["observed_at"] = self.now - timedelta(seconds=seconds)
                response = self.request_latest()
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["is_stale"], expected)
                self.assertEqual(response.json()["stale_after_minutes"], 120)

    def test_zero_rain_and_unknown_unit_are_preserved(self):
        self.observation["depth"] = 0
        response = self.request_latest(
            station={"station_type": "rainfall", "measurement_unit": None}
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["value"], 0)
        self.assertIsNone(response.json()["unit"])

    def test_unknown_station_does_not_query_observations(self):
        self.connection.execute.return_value.mappings.return_value.first.return_value = None
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "STATION_NOT_FOUND")
        self.connection.execute.assert_called_once()

    def test_station_without_valid_observations_has_distinct_error(self):
        station_result = MagicMock()
        station_result.mappings.return_value.first.return_value = self.station
        empty_result = MagicMock()
        empty_result.mappings.return_value.first.return_value = None
        self.connection.execute.side_effect = [station_result, empty_result]
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "OBSERVATION_NOT_FOUND")
        self.assertEqual(response.json()["request_id"], response.headers["X-Request-ID"])

    def test_database_errors_do_not_expose_credentials(self):
        self.connection.execute.side_effect = OperationalError(
            "SELECT", {}, Exception("password=secret")
        )
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "DATA_SERVICE_UNAVAILABLE")
        self.assertNotIn("secret", response.text)

    def test_openapi_documents_latest(self):
        operation = self.client.get("/openapi.json").json()["paths"][
            "/api/v1/stations/{station_id}/latest"
        ]["get"]
        self.assertIn("404", operation["responses"])
        self.assertIn("503", operation["responses"])

    def test_cache_hit_skips_observation_query_but_refreshes_age_and_stale(self):
        self.observation["observed_at"] = self.now - timedelta(hours=2)
        redis_client = MagicMock()
        redis_client.mget.return_value = [
            CachedObservation(station_id="0074647403", **self.observation).model_dump_json(),
            "g",
        ]
        self.cache.client = redis_client
        station_result = MagicMock()
        station_result.mappings.return_value.first.return_value = self.station
        self.connection.execute.return_value = station_result
        first = self.client.get(self.url)
        self.assertEqual(first.status_code, 200, first.text)
        self.assertFalse(first.json()["is_stale"])
        self.clock.now.return_value = self.now + timedelta(seconds=1)
        second = self.client.get(self.url)
        self.assertTrue(second.json()["is_stale"])
        self.assertGreater(second.json()["data_age_minutes"], first.json()["data_age_minutes"])
        self.assertEqual(self.connection.execute.call_count, 2)  # Station metadata only.
        redis_client.eval.assert_not_called()

    def test_cache_miss_publishes_database_value(self):
        redis_client = MagicMock()
        redis_client.mget.return_value = [None, "cache_version"]
        self.cache.client = redis_client
        response = self.request_latest()
        self.assertEqual(response.status_code, 200, response.text)
        redis_client.eval.assert_called_once()
        self.assertEqual(self.connection.execute.call_count, 2)

    def test_redis_outage_falls_back_to_database(self):
        redis_client = MagicMock()
        redis_client.mget.side_effect = ConnectionError("private-password")
        self.cache.client = redis_client
        with self.assertLogs("src.backend.cache.latest", "WARNING"):
            response = self.request_latest()
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["value"], self.observation["depth"])
        redis_client.eval.assert_not_called()

    def test_cache_write_outage_does_not_fail_database_response(self):
        redis_client = MagicMock()
        redis_client.mget.return_value = [None, ""]
        redis_client.eval.side_effect = ConnectionError("private-password")
        self.cache.client = redis_client
        with self.assertLogs("src.backend.cache.latest", "WARNING"):
            response = self.request_latest()
        self.assertEqual(response.status_code, 200, response.text)

    def test_cached_measurement_does_not_hide_database_failure(self):
        redis_client = MagicMock()
        self.cache.client = redis_client
        self.connection.execute.side_effect = OperationalError("SELECT", {}, Exception("secret"))
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 503)
        redis_client.mget.assert_not_called()

    def test_injected_cache_is_not_closed_by_application(self):
        cache = MagicMock(spec=LatestObservationCache)
        app = create_app(engine=self.engine, registry=StationModelRegistry([]), latest_cache=cache)
        with TestClient(app):
            pass
        cache.close.assert_not_called()


if __name__ == "__main__":
    main()
