"""Focused sync tests: source recovery/pagination and rollback-only PostgreSQL checks."""

from datetime import datetime, timedelta, timezone
from threading import Event
from unittest import TestCase, main
from unittest.mock import MagicMock, patch
from pathlib import Path

from dotenv import load_dotenv
import requests
import sqlalchemy as sa

from src.backend.ingestion.worker import (
    HueSourceClient,
    SourceError,
    SyncCancelled,
    SyncSettings,
    parse_source_row,
    sync_once,
    upsert_observations,
    LOCK_KEYS,
)
from src.backend.db.models import Observation
from src.backend.db.session import build_engine

PROJECT_ROOT = Path(__file__).resolve().parents[4]

STATION = {
    "id": "0074647403",
    "source_name": "hue_dcu",
    "source_station_id": "0074647403",
    "observation_collection": "water_level_depth",
}
START = datetime(2026, 10, 1, tzinfo=timezone.utc)
END = START + timedelta(hours=2)


def response(payload, status=200, headers=None):
    value = MagicMock(status_code=status, headers=headers or {})
    value.json.return_value = payload
    return value


def record(identifier=1, minutes=10, **overrides):
    row = {
        "id": identifier,
        "station_id": STATION["source_station_id"],
        "time_point": (START + timedelta(minutes=minutes)).isoformat(),
        "depth": 0.0,
    }
    row.update(overrides)
    return row


class SourceClientTests(TestCase):
    def setUp(self):
        self.settings = SyncSettings("https://source.invalid", "test-account", "private-password")
        self.stop = MagicMock(spec=Event)
        self.stop.is_set.return_value = False
        self.stop.wait.return_value = False
        self.session = MagicMock()
        self.session.headers = {}
        self.client = HueSourceClient(self.settings, self.stop, self.session)

    def test_id_pagination_continues_after_short_pages(self):
        self.session.request.side_effect = [
            response({"data": [record()]}),
            response({"data": [record(2, 20)]}),
            response({"data": []}),
        ]
        rows = self.client.observations(STATION, START, END)
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            [
                call.kwargs["params"]["filter[id][_gt]"]
                for call in self.session.request.call_args_list
            ],
            [0, 1, 2],
        )
        self.assertEqual(rows[0]["station_id"], "0074647403")

    def test_expired_token_renews_once_then_retries_page(self):
        self.session.request.side_effect = [
            response({}, 401),
            response({"data": {"access_token": "private-token"}}),
            response({"data": [record()]}),
            response({"data": []}),
        ]
        self.assertEqual(len(self.client.observations(STATION, START, END)), 1)
        self.assertEqual(self.session.headers["Authorization"], "Bearer private-token")
        self.assertEqual(self.session.request.call_count, 4)

    def test_retry_after_is_bounded(self):
        self.session.request.side_effect = [
            response({}, 429, {"Retry-After": "900"}),
            response({"data": []}),
        ]
        self.client.request("GET", "/items/water_level_depth")
        self.stop.wait.assert_called_once_with(30)

    def test_transport_error_does_not_expose_secrets(self):
        self.session.request.side_effect = requests.Timeout("private-password private-token")
        with self.assertRaises(SourceError) as raised:
            self.client.request("GET", "/items/water_level_depth")
        self.assertNotIn("private-", str(raised.exception))
        self.assertEqual(self.session.request.call_count, 3)

    def test_invalid_source_fields_are_rejected_but_zero_and_missing_survive(self):
        for incoming in (
            record(depth=True),
            record(depth=float("nan")),
            record(depth=float("inf")),
            record(time_point="2026-10-01T00:10:00"),
            record(station_id="007464740"),
            record(id=True),
            record(minutes=180),
        ):
            with self.subTest(incoming=incoming), self.assertRaises(SourceError):
                parse_source_row(incoming, STATION, START, END)
        self.assertEqual(parse_source_row(record(), STATION, START, END)["depth"], 0)
        self.assertEqual(
            parse_source_row(record(depth=None), STATION, START, END)["quality_status"], "missing"
        )

    def test_duplicate_timestamp_or_stalled_id_is_not_silently_discarded(self):
        for second in (record(2), record(1, 20)):
            self.session.request.side_effect = [
                response({"data": [record()]}),
                response({"data": [second]}),
            ]
            with self.assertRaises(SourceError):
                self.client.observations(STATION, START, END)

    def test_cancellation_prevents_new_requests(self):
        self.stop.is_set.return_value = True
        with self.assertRaises(SyncCancelled):
            self.client.request("GET", "/items/water_level_depth")
        self.session.request.assert_not_called()

    def test_setting_repr_does_not_contain_credentials(self):
        self.assertNotIn("private-password", repr(self.settings))
        self.assertNotIn("test-account", repr(self.settings))


class SyncDatabaseTests(TestCase):
    @classmethod
    def setUpClass(cls):
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        cls.engine = build_engine()

    @classmethod
    def tearDownClass(cls):
        cls.engine.dispose()

    def row(self, station=STATION, depth=100):
        return {
            "station_id": station["id"],
            "observed_at": datetime(2020, 2, 1, tzinfo=timezone.utc),
            "depth": depth,
            "source_name": station["source_name"],
            "source_collection": station["observation_collection"],
            "source_record_id": -990001,
            "quality_status": "observed",
        }

    def test_upsert_is_idempotent_and_correction_preserves_received_at(self):
        with self.engine.connect() as connection, connection.begin() as transaction:
            incoming = self.row()
            self.assertEqual(
                upsert_observations(connection, [incoming]),
                {"inserted": 1, "updated": 0, "unchanged": 0},
            )
            original = (
                connection.execute(
                    sa.select(Observation.__table__).where(
                        Observation.station_id == incoming["station_id"],
                        Observation.observed_at == incoming["observed_at"],
                    )
                )
                .mappings()
                .one()
            )
            self.assertEqual(
                upsert_observations(connection, [incoming]),
                {"inserted": 0, "updated": 0, "unchanged": 1},
            )
            unchanged = connection.execute(
                sa.select(Observation.updated_at).where(
                    Observation.station_id == incoming["station_id"],
                    Observation.observed_at == incoming["observed_at"],
                )
            ).scalar_one()
            self.assertEqual(original["updated_at"], unchanged)
            self.assertEqual(
                upsert_observations(connection, [{**incoming, "depth": 101}]),
                {"inserted": 0, "updated": 1, "unchanged": 0},
            )
            corrected = (
                connection.execute(
                    sa.select(Observation.__table__).where(
                        Observation.station_id == incoming["station_id"],
                        Observation.observed_at == incoming["observed_at"],
                    )
                )
                .mappings()
                .one()
            )
            self.assertEqual(corrected["received_at"], original["received_at"])
            self.assertGreater(corrected["updated_at"], unchanged)
            transaction.rollback()

    def fake_client(self, invalid_second=False):
        client = MagicMock()
        client.stop.is_set.return_value = False
        client.settings.overlap_hours = 24

        def fetch(station, start, end):
            incoming = self.row(station)
            if invalid_second and station["id"] == "750003":
                incoming["source_collection"] = "water_level_depth"
            return [incoming]

        client.observations.side_effect = fetch
        return client

    def test_group_rolls_back_when_one_station_write_fails(self):
        cache = MagicMock()
        with self.assertRaises(sa.exc.IntegrityError):
            sync_once(self.engine, self.fake_client(invalid_second=True), latest_cache=cache)
        cache.invalidate.assert_not_called()
        with self.engine.connect() as connection:
            count = connection.scalar(
                sa.select(sa.func.count())
                .select_from(Observation)
                .where(Observation.source_record_id == -990001)
            )
        self.assertEqual(count, 0)

    def test_dry_run_writes_no_observations(self):
        cache = MagicMock()
        result = sync_once(self.engine, self.fake_client(), dry_run=True, latest_cache=cache)
        cache.invalidate.assert_not_called()
        self.assertEqual(result["status"], "validated")
        self.assertEqual(len(result["stations"]), 5)
        with self.engine.connect() as connection:
            self.assertEqual(
                connection.scalar(
                    sa.select(sa.func.count())
                    .select_from(Observation)
                    .where(Observation.source_record_id == -990001)
                ),
                0,
            )

    def test_second_writer_skips_without_calling_source(self):
        client = self.fake_client()
        with self.engine.connect() as connection:
            connection.execute(
                sa.text("SELECT pg_advisory_lock(:first, :second)"),
                {"first": LOCK_KEYS[0], "second": LOCK_KEYS[1]},
            )
            connection.commit()
            try:
                result = sync_once(self.engine, client)
                self.assertEqual(result["status"], "skipped_writer_busy")
                client.login.assert_not_called()
            finally:
                connection.execute(
                    sa.text("SELECT pg_advisory_unlock(:first, :second)"),
                    {"first": LOCK_KEYS[0], "second": LOCK_KEYS[1]},
                )
                connection.commit()


class SyncCacheTests(TestCase):
    def setUp(self):
        self.engine = MagicMock()
        self.connection = self.engine.connect.return_value.__enter__.return_value
        self.connection.scalar.return_value = True
        self.connection.in_transaction.return_value = False
        self.connection.invalidated = False
        stations = [STATION, {**STATION, "id": "750003"}]
        station_result = MagicMock()
        station_result.mappings.return_value.all.return_value = stations
        watermark_result = MagicMock()
        watermark_result.all.return_value = []
        self.connection.execute.side_effect = [station_result, watermark_result, MagicMock()]
        self.client = MagicMock()
        self.client.settings.overlap_hours = 24
        self.client.stop.is_set.return_value = False
        self.client.observations.return_value = [{"observed_at": START, "depth": 1}]
        self.cache = MagicMock()
        self.cache.enabled = True
        self.cache.invalidate.return_value = True

    def test_only_changed_station_is_invalidated_after_transaction_commit(self):
        transaction = self.connection.begin.return_value

        def invalidate(station_id):
            self.assertEqual(transaction.__exit__.call_count, 2)
            self.assertEqual(station_id, STATION["id"])
            return True

        self.cache.invalidate.side_effect = invalidate
        with patch("src.backend.ingestion.worker.upsert_observations") as upsert:
            upsert.side_effect = [
                {"inserted": 1, "updated": 0, "unchanged": 0},
                {"inserted": 0, "updated": 0, "unchanged": 1},
            ]
            result = sync_once(self.engine, self.client, latest_cache=self.cache)
        self.assertEqual(result["status"], "committed")
        self.cache.invalidate.assert_called_once_with(STATION["id"])
        self.assertEqual(result["stations"][0]["latest_cache_invalidation"], "ok")
        self.assertNotIn("latest_cache_invalidation", result["stations"][1])

    def test_failed_invalidation_keeps_successful_sync_and_reports_failure(self):
        self.cache.invalidate.return_value = False
        with patch("src.backend.ingestion.worker.upsert_observations") as upsert:
            upsert.return_value = {"inserted": 0, "updated": 1, "unchanged": 0}
            result = sync_once(self.engine, self.client, latest_cache=self.cache)
        self.assertEqual(result["status"], "committed")
        self.assertTrue(
            all(summary["latest_cache_invalidation"] == "failed" for summary in result["stations"])
        )


if __name__ == "__main__":
    main()
