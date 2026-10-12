"""PostgreSQL integration checks; each test rolls back all its observations."""

from datetime import datetime, timedelta, timezone
import unittest

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.backend.db.models import Observation, Station
from src.backend.db.session import build_engine


class SchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = build_engine()

    @classmethod
    def tearDownClass(cls):
        cls.engine.dispose()

    def setUp(self):
        self.connection = self.engine.connect()
        self.transaction = self.connection.begin()
        self.session = Session(bind=self.connection, join_transaction_mode="create_savepoint")
        self.observed_at = datetime(2099, 1, 1, 7, tzinfo=timezone.utc)

    def tearDown(self):
        self.session.close()
        self.transaction.rollback()
        self.connection.close()

    def observation_values(self, **overrides):
        values = {
            "station_id": "0074647403",
            "observed_at": self.observed_at,
            "depth": 500.0,
            "source_name": "hue_dcu",
            "source_collection": "water_level_depth",
            "source_record_id": -9001,
        }
        values.update(overrides)
        return values

    def assert_rejected(self, expected_sqlstate, **overrides):
        with self.assertRaises(IntegrityError) as raised:
            with self.session.begin_nested():
                self.session.execute(
                    sa.insert(Observation).values(**self.observation_values(**overrides))
                )
        self.assertEqual(raised.exception.orig.sqlstate, expected_sqlstate)

    def test_station_metadata(self):
        stations = self.session.scalars(sa.select(Station)).all()
        self.assertEqual(len(stations), 5)
        water_station = self.session.get(Station, "0074647403")
        self.assertEqual(water_station.measurement_unit, "mm")
        self.assertTrue(water_station.prediction_enabled)
        rain_stations = [station for station in stations if station.station_type == "rainfall"]
        self.assertEqual(len(rain_stations), 4)
        self.assertTrue(all(station.measurement_unit is None for station in rain_stations))
        self.assertTrue(all(not station.prediction_enabled for station in rain_stations))

    def test_duplicate_instant_and_source_id(self):
        self.session.add(Observation(**self.observation_values()))
        self.session.flush()
        vietnam_time = self.observed_at.astimezone(timezone(timedelta(hours=7)))
        self.assert_rejected("23505", observed_at=vietnam_time, source_record_id=-9002)
        self.assert_rejected("23505", observed_at=self.observed_at + timedelta(hours=1))

    def test_station_and_collection_foreign_keys(self):
        self.assert_rejected("23503", station_id="unknown")
        self.assert_rejected("23503", station_id="750007")

    def test_missing_and_nonfinite_values(self):
        for depth in (None, float("nan"), float("inf"), float("-inf")):
            with self.subTest(depth=depth):
                self.assert_rejected("23514", depth=depth)
        self.assert_rejected("23514", quality_status="missing", depth=0.0)
        self.session.add(
            Observation(**self.observation_values(depth=None, quality_status="missing"))
        )
        self.session.flush()
        stored = self.session.get(Observation, ("0074647403", self.observed_at))
        self.assertIsNone(stored.depth)

    def test_source_correction_preserves_first_received_time(self):
        original = Observation(**self.observation_values())
        self.session.add(original)
        self.session.flush()
        first_received = original.received_at
        previous_update = original.updated_at
        statement = insert(Observation).values(**self.observation_values(depth=550.0))
        statement = statement.on_conflict_do_update(
            index_elements=[Observation.station_id, Observation.observed_at],
            set_={"depth": statement.excluded.depth},
        )
        self.session.execute(statement)
        self.session.refresh(original)
        self.assertEqual(original.depth, 550.0)
        self.assertEqual(original.received_at, first_received)
        self.assertGreater(original.updated_at, previous_update)

    def test_history_latest_and_source_ids_across_collections(self):
        self.session.add_all(
            [
                Observation(**self.observation_values()),
                Observation(
                    **self.observation_values(
                        observed_at=self.observed_at + timedelta(hours=1),
                        depth=-10.0,
                        source_record_id=-9002,
                    )
                ),
                Observation(
                    **self.observation_values(
                        station_id="750062", source_collection="rain_water_depth", depth=0.0
                    )
                ),
            ]
        )
        self.session.flush()
        latest = self.session.scalars(
            sa.select(Observation)
            .where(Observation.station_id == "0074647403")
            .order_by(Observation.observed_at.desc())
            .limit(1)
        ).one()
        self.assertEqual(latest.depth, -10.0)
        history = self.session.scalars(
            sa.select(Observation).where(
                Observation.station_id == "0074647403",
                Observation.observed_at >= self.observed_at,
                Observation.observed_at < self.observed_at + timedelta(hours=1),
            )
        ).all()
        self.assertEqual(len(history), 1)
        self.assertEqual(self.session.scalar(sa.text("SHOW TIME ZONE")), "UTC")


if __name__ == "__main__":
    unittest.main()
