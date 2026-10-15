"""Read-only snapshots, API coverage and commit/invalidation ordering."""

from datetime import datetime, timedelta, timezone
import json
import sqlite3
from unittest import TestCase
from unittest.mock import MagicMock, patch

from src.backend.ingestion.history_import import api_windows, sqlite_batches, write_batch
from src.backend.ingestion.worker import SourceError

STATION = {
    "id": "0074647403",
    "source_station_id": "0074647403",
    "source_name": "hue_dcu",
    "observation_collection": "water_level_depth",
}


class HistoryImportTests(TestCase):
    def setUp(self):
        self.source = sqlite3.connect(":memory:")
        self.addCleanup(self.source.close)
        self.source.row_factory = sqlite3.Row
        self.source.execute(
            "CREATE TABLE water_level_observations (_key TEXT, time_point TEXT, station_key TEXT, raw_json TEXT)"
        )
        self.cutoff = datetime(2026, 10, 14, tzinfo=timezone.utc)

    def add_row(self, source_id, timestamp, depth):
        payload = {
            "id": source_id,
            "station_id": STATION["id"],
            "time_point": timestamp,
            "depth": depth,
        }
        self.source.execute(
            "INSERT INTO water_level_observations VALUES (?,?,?,?)",
            (str(source_id), timestamp, STATION["id"], json.dumps(payload)),
        )

    def test_batches_keep_missing_zero_negative_and_resume_after_commit(self):
        for index, value in enumerate([None, 0, -20]):
            self.add_row(index + 1, f"2026-10-13T0{index}:00:00Z", value)
        batches = list(sqlite_batches(self.source, STATION, self.cutoff, batch_size=2))
        self.assertEqual([row["depth"] for batch, _ in batches for row in batch], [None, 0, -20])
        self.assertEqual(batches[0][0][0]["quality_status"], "missing")
        resumed = list(sqlite_batches(self.source, STATION, self.cutoff, after=batches[0][1]))
        self.assertEqual(len(resumed[0][0]), 1)
        self.assertEqual(resumed[0][0][0]["depth"], -20)

    def test_duplicate_and_future_times_fail_validation(self):
        self.add_row(1, "2026-10-13T00:00:00Z", 0)
        self.add_row(2, "2026-10-13T00:00:00Z", 1)
        with self.assertRaises(SourceError):
            list(sqlite_batches(self.source, STATION, self.cutoff))
        self.source.execute("DELETE FROM water_level_observations")
        self.add_row(3, "2099-01-01T00:00:00Z", 1)
        with self.assertRaises(SourceError):
            list(sqlite_batches(self.source, STATION, self.cutoff))

    def test_api_coverage_includes_old_missing_records_and_tail_overlap(self):
        first = self.cutoff - timedelta(days=365)
        sqlite_first = first + timedelta(days=2)
        sqlite_last = self.cutoff - timedelta(days=14)
        windows = api_windows(first, sqlite_first, sqlite_last, self.cutoff)
        self.assertEqual(
            windows, [(first, sqlite_first), (sqlite_last - timedelta(hours=24), self.cutoff)]
        )
        self.assertEqual(api_windows(first, None, None, self.cutoff), [(first, self.cutoff)])
        self.assertEqual(api_windows(None, None, None, self.cutoff), [])

    def test_sqlite_insert_and_api_update_use_distinct_policies(self):
        engine, cache = MagicMock(), MagicMock()
        with patch(
            "src.backend.ingestion.history_import.append_snapshot_batch",
            return_value={"inserted": 1, "retained": 2},
        ) as append, patch(
            "src.backend.ingestion.history_import.upsert_observations",
            return_value={"inserted": 0, "updated": 1, "unchanged": 2},
        ) as upsert:
            write_batch(engine, cache, STATION, [{}])
            append.assert_called_once()
            upsert.assert_not_called()
            write_batch(engine, cache, STATION, [{}], authoritative=True)
            upsert.assert_called_once()
        self.assertEqual(cache.invalidate.call_count, 2)

    def test_cache_is_invalidated_after_commit_and_not_after_failed_batch(self):
        engine, cache = MagicMock(), MagicMock()
        events = []
        engine.begin.return_value.__exit__.side_effect = lambda *args: events.append("commit")
        cache.invalidate.side_effect = lambda *args: events.append("invalidate") or True
        with patch(
            "src.backend.ingestion.history_import.append_snapshot_batch",
            return_value={"inserted": 1},
        ):
            write_batch(engine, cache, STATION, [{}])
        self.assertEqual(events, ["commit", "invalidate"])
        cache.reset_mock()
        with patch(
            "src.backend.ingestion.history_import.append_snapshot_batch",
            side_effect=SourceError("bad"),
        ):
            with self.assertRaises(SourceError):
                write_batch(engine, cache, STATION, [{}])
        cache.invalidate.assert_not_called()
