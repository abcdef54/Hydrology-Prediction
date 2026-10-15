"""Rollback-only PostgreSQL checks for wind provenance and revised readings."""

from datetime import datetime, timedelta, timezone
from unittest import TestCase

from dotenv import load_dotenv
import sqlalchemy as sa

from src.backend.db.models import Station, WindObservation
from src.backend.db.session import build_engine
from src.backend.ingestion.wind import upsert_wind_observations


class WindDatabaseTests(TestCase):
    @classmethod
    def setUpClass(cls):
        load_dotenv()
        cls.engine = build_engine()

    @classmethod
    def tearDownClass(cls):
        cls.engine.dispose()

    def setUp(self):
        self.connection = self.engine.connect()
        self.transaction = self.connection.begin()
        self.addCleanup(self.connection.close)
        self.addCleanup(self.transaction.rollback)
        self.connection.execute(
            sa.insert(Station).values(
                id="wind:test-rollback",
                name="Test rollback",
                station_type="wind",
                source_name="test-wind",
                source_station_id="test-source",
                observation_collection="iot_wind_speed",
                sync_enabled=True,
            )
        )
        self.row = {
            "station_id": "wind:test-rollback",
            "observed_at": datetime(2000, 1, 1, tzinfo=timezone.utc),
            "ws": 0,
            "wd": 961,
            "wsg": None,
            "wdg": 1292,
            "source_name": "test-wind",
            "source_collection": "iot_wind_speed",
            "source_record_id": 1,
            "quality_status": "observed",
        }

    def stored(self):
        return (
            self.connection.execute(
                sa.select(WindObservation.__table__).where(
                    WindObservation.station_id == self.row["station_id"],
                )
            )
            .mappings()
            .one()
        )

    def test_idempotence_snapshot_preservation_and_api_revision(self):
        self.assertEqual(upsert_wind_observations(self.connection, [self.row])["inserted"], 1)
        original = self.stored()
        self.assertEqual(upsert_wind_observations(self.connection, [self.row])["unchanged"], 1)
        self.assertEqual(self.stored()["updated_at"], original["updated_at"])
        changed = {**self.row, "ws": 12.5}
        self.assertEqual(
            upsert_wind_observations(
                self.connection,
                [changed],
                preserve_existing=True,
            )["unchanged"],
            1,
        )
        self.assertEqual(self.stored()["ws"], 0)
        self.assertEqual(upsert_wind_observations(self.connection, [changed])["updated"], 1)
        self.assertEqual(self.stored()["ws"], 12.5)
        self.assertEqual(self.stored()["wd"], 961)
        self.assertEqual(self.stored()["received_at"], original["received_at"])

    def test_snapshot_does_not_silently_discard_wrong_source_id_mapping(self):
        upsert_wind_observations(self.connection, [self.row])
        wrong_mapping = {**self.row, "observed_at": self.row["observed_at"] + timedelta(hours=1)}
        with self.assertRaises(sa.exc.IntegrityError):
            with self.connection.begin_nested():
                upsert_wind_observations(self.connection, [wrong_mapping], preserve_existing=True)

    def test_database_rejects_nonfinite_values_and_inconsistent_quality(self):
        for change in ({"wd": float("nan")}, {"ws": float("inf")}, {"quality_status": "missing"}):
            with self.subTest(change=change), self.assertRaises(sa.exc.IntegrityError):
                with self.connection.begin_nested():
                    upsert_wind_observations(self.connection, [{**self.row, **change}])
