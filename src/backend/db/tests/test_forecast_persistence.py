"""Real PostgreSQL save/reuse checks, enclosed in rollback-only test transactions."""

from contextlib import contextmanager
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace
from unittest import TestCase, main

import sqlalchemy as sa

from src.backend.db.models import ForecastPoint
from src.backend.db.session import build_engine
from src.backend.prediction.persistence import build_input_hash, load_forecast, save_forecast
from src.backend.prediction.tests.test_persistence import persistence_example


class ForecastPersistenceTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = build_engine()

    @classmethod
    def tearDownClass(cls):
        cls.engine.dispose()

    def setUp(self):
        self.connection = self.engine.connect()
        self.transaction = self.connection.begin()
        self.addCleanup(self.connection.close)
        self.addCleanup(self.transaction.rollback)
        self.config, self.prepared, self.forecast = persistence_example()
        self.input_hash = build_input_hash(self.prepared, self.config)
        self.pipeline = "persistence-test"
        self.test_engine = SimpleNamespace(begin=self.savepoint_connection)

    @contextmanager
    def savepoint_connection(self):
        with self.connection.begin_nested():
            yield self.connection

    def save(self, forecast=None):
        return save_forecast(
            self.test_engine,
            self.config,
            self.prepared,
            self.pipeline,
            self.input_hash,
            forecast or self.forecast,
        )

    def load(self, prepared=None, pipeline=None, input_hash=None):
        return load_forecast(
            self.connection,
            self.config,
            prepared or self.prepared,
            pipeline or self.pipeline,
            input_hash or self.input_hash,
        )

    def test_saved_result_is_complete_and_reuses_current_request_context(self):
        saved = self.save()
        self.assertFalse(saved.reused)
        self.assertIsNotNone(saved.forecast_id)
        self.assertIsNotNone(saved.generated_at)
        new_request = replace(
            self.prepared,
            requested_reference_time=None,
            hourly_reference_time=self.prepared.hourly_reference_time + timedelta(hours=1),
        )
        loaded = self.load(prepared=new_request)
        self.assertTrue(loaded.reused)
        self.assertEqual(loaded.predictions, saved.predictions)
        self.assertEqual(loaded.forecast_id, saved.forecast_id)
        self.assertEqual(loaded.generated_at, saved.generated_at)
        self.assertIsNone(loaded.requested_reference_time)
        self.assertEqual(loaded.hourly_reference_time, new_request.hourly_reference_time)
        self.assertIsNone(self.load(pipeline="other-pipeline"))
        self.assertIsNone(self.load(input_hash="b" * 64))

    def test_duplicate_save_returns_winning_result_without_overwrite(self):
        first = self.save()
        different = replace(
            self.forecast,
            predictions=tuple(
                replace(point, water_level_mm=999) for point in self.forecast.predictions
            ),
        )
        second = self.save(different)
        self.assertTrue(second.reused)
        self.assertEqual(first.forecast_id, second.forecast_id)
        self.assertEqual(first.predictions, second.predictions)

    def test_partial_or_wrong_time_stored_forecasts_are_not_returned(self):
        saved = self.save()
        first_point = sa.and_(
            ForecastPoint.forecast_id == saved.forecast_id,
            ForecastPoint.horizon_minutes == self.config.forecast_horizons_minutes[0],
        )
        with self.connection.begin_nested() as transaction:
            self.connection.execute(
                sa.update(ForecastPoint)
                .where(first_point)
                .values(forecast_at=self.prepared.anchor_observed_at)
            )
            with self.assertRaises(ValueError):
                self.load()
            transaction.rollback()
        self.connection.execute(sa.delete(ForecastPoint).where(first_point))
        with self.assertRaises(ValueError):
            self.load()

    def test_failed_point_insert_does_not_leave_parent(self):
        # Fail after the real parent insert, without installing a DB trigger.
        def fail_points(connection, cursor, statement, parameters, context, executemany):
            if statement.startswith("INSERT INTO hydrology.forecast_points"):
                raise RuntimeError("simulated point write failure")

        sa.event.listen(self.connection, "before_cursor_execute", fail_points)
        try:
            with self.assertRaises(RuntimeError):
                self.save()
        finally:
            sa.event.remove(self.connection, "before_cursor_execute", fail_points)
        self.assertIsNone(self.load())


if __name__ == "__main__":
    main()
