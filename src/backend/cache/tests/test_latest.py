"""Latest cache unit tests and opt-in real Redis checks using isolated keys."""

from datetime import datetime, timedelta, timezone
import json
import os
import time
from unittest import TestCase, main, skipUnless
from unittest.mock import MagicMock, patch
from uuid import uuid4

from redis import Redis
from redis.exceptions import ConnectionError, TimeoutError

from src.backend.cache.latest import CachedObservation, LatestObservationCache

NOW = datetime(2026, 10, 13, 7, tzinfo=timezone.utc)


def observation():
    return {
        "depth": -25.5,
        "observed_at": NOW - timedelta(minutes=30),
        "received_at": NOW - timedelta(minutes=25),
        "quality_status": "observed",
    }


class LatestCacheTests(TestCase):
    def setUp(self):
        self.client = MagicMock()
        self.cache = LatestObservationCache(self.client)
        self.station = "0074647403"

    def test_valid_entry_preserves_units_and_does_not_store_freshness(self):
        self.client.eval.return_value = 1
        self.assertTrue(self.cache.store(self.station, "cache_version", observation()))
        arguments = self.client.eval.call_args.args
        payload = arguments[-2]
        self.assertEqual(arguments[-1], 600)
        self.assertNotIn("is_stale", payload)
        self.assertNotIn("data_age_minutes", payload)
        self.client.mget.return_value = [payload, "cache_version"]
        cache_version, cached = self.cache.read(self.station, NOW)
        self.assertEqual(cache_version, "cache_version")
        self.assertEqual(cached.depth, -25.5)
        self.assertEqual(cached.observed_at, observation()["observed_at"])

    def test_invalid_payload_station_or_future_timestamp_is_a_miss(self):
        valid = CachedObservation(station_id=self.station, **observation())
        payloads = [
            "not JSON",
            "{}",
            json.dumps({**valid.model_dump(mode="json"), "depth": True}),
            valid.model_copy(update={"station_id": "wrong"}).model_dump_json(),
            valid.model_copy(update={"observed_at": NOW + timedelta(seconds=1)}).model_dump_json(),
        ]
        for payload in payloads:
            with self.subTest(payload=payload), self.assertLogs(
                "src.backend.cache.latest", "WARNING"
            ):
                self.client.mget.return_value = [payload, "cache_version"]
                self.assertEqual(self.cache.read(self.station, NOW), ("cache_version", None))

    def test_unavailable_read_prevents_another_write_attempt_and_hides_secrets(self):
        self.client.mget.side_effect = ConnectionError("redis://user:secret@host")
        with self.assertLogs("src.backend.cache.latest", "WARNING") as logs:
            cache_version, cached = self.cache.read(self.station, NOW)
        self.assertIsNone(cached)
        self.assertFalse(self.cache.store(self.station, cache_version, observation()))
        self.client.eval.assert_not_called()
        self.assertNotIn("secret", " ".join(logs.output))

    def test_store_and_invalidation_failures_do_not_raise(self):
        self.client.eval.side_effect = TimeoutError("private details")
        with self.assertLogs("src.backend.cache.latest", "WARNING") as logs:
            self.assertFalse(self.cache.store(self.station, "", observation()))
            self.assertFalse(self.cache.invalidate(self.station))
        self.assertNotIn("private", " ".join(logs.output))

    def test_disabled_cache_never_needs_a_connection(self):
        cache = LatestObservationCache()
        self.assertFalse(cache.enabled)
        self.assertEqual(cache.read(self.station, NOW), (None, None))
        self.assertFalse(cache.store(self.station, None, observation()))
        self.assertTrue(cache.invalidate(self.station))
        cache.close()

    def test_config_validation_and_connection_is_lazy(self):
        for ttl in (0, -1, True, 1.5):
            with self.subTest(ttl=ttl), self.assertRaises(ValueError):
                LatestObservationCache(ttl_seconds=ttl)
        with patch.dict(os.environ, {"REDIS_URL": "redis://127.0.0.1:1/0"}, clear=True):
            cache = LatestObservationCache.from_env()
            self.assertTrue(cache.enabled)
            self.assertEqual(cache.client.connection_pool.connection_kwargs["socket_timeout"], 0.5)
            self.assertEqual(
                cache.client.connection_pool.connection_kwargs["retry"].get_retries(), 0
            )
            cache.close()
        with patch.dict(os.environ, {"REDIS_URL": "http://secret@host"}, clear=True):
            with self.assertRaisesRegex(ValueError, "REDIS_URL") as raised:
                LatestObservationCache.from_env()
            self.assertNotIn("secret", str(raised.exception))


@skipUnless(os.environ.get("REDIS_TEST_URL"), "Set REDIS_TEST_URL to run real Redis checks.")
class RealLatestCacheTests(TestCase):
    def setUp(self):
        self.client = Redis.from_url(os.environ["REDIS_TEST_URL"], decode_responses=True)
        self.cache = LatestObservationCache(self.client, ttl_seconds=1)
        self.station = f"cache-test-{uuid4().hex}"

    def tearDown(self):
        self.client.delete(*self.cache.keys(self.station))
        self.cache.close()

    def test_round_trip_and_real_ttl_expiry(self):
        cache_version, cached = self.cache.read(self.station, NOW)
        self.assertEqual((cache_version, cached), ("", None))
        self.assertTrue(self.cache.store(self.station, cache_version, observation()))
        self.assertEqual(self.cache.read(self.station, NOW)[1].depth, -25.5)
        self.assertGreater(self.client.pttl(self.cache.keys(self.station)[0]), 0)
        time.sleep(1.1)
        self.assertIsNone(self.cache.read(self.station, NOW)[1])

    def test_invalidation_rejects_in_flight_old_reader(self):
        old_cache_version, _ = self.cache.read(self.station, NOW)
        self.assertTrue(self.cache.store(self.station, old_cache_version, observation()))
        self.assertTrue(self.cache.invalidate(self.station))
        self.assertFalse(self.cache.store(self.station, old_cache_version, observation()))
        cache_version, cached = self.cache.read(self.station, NOW)
        self.assertIsNone(cached)
        self.assertNotEqual(cache_version, old_cache_version)
        corrected = {**observation(), "depth": 100}
        self.assertTrue(self.cache.store(self.station, cache_version, corrected))
        self.assertEqual(self.cache.read(self.station, NOW)[1].depth, 100)


if __name__ == "__main__":
    main()
