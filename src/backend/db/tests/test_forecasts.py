"""PostgreSQL constraints for forecasts; every test rolls back its inserted rows."""

from datetime import datetime, timedelta, timezone
import unittest
from uuid import uuid4

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.backend.db.models import Forecast, ForecastPoint
from src.backend.db.session import build_engine


class ForecastSchemaTests(unittest.TestCase):
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
        self.anchor = datetime(2026, 9, 13, 2, tzinfo=timezone.utc)

    def tearDown(self):
        self.session.close()
        self.transaction.rollback()
        self.connection.close()

    def forecast_values(self, **overrides):
        values = dict(
            station_id="0074647403",
            model_version="test-model",
            feature_schema_version="test-schema",
            pipeline_version="test-scaler-pipeline",
            input_hash="a" * 64,
            anchor_observed_at=self.anchor,
        )
        values.update(overrides)
        return values

    def add_forecast(self, **overrides):
        forecast = Forecast(**self.forecast_values(**overrides))
        self.session.add(forecast)
        self.session.flush()
        return forecast

    def point_values(self, forecast_id, **overrides):
        values = dict(
            forecast_id=forecast_id,
            horizon_minutes=60,
            forecast_at=self.anchor + timedelta(hours=1),
            water_level_mm=0.0,
        )
        values.update(overrides)
        return values

    def assert_rejected(self, model, values, sqlstate):
        with self.assertRaises(IntegrityError) as raised:
            with self.session.begin_nested():
                self.session.add(model(**values))
                self.session.flush()
        self.assertEqual(raised.exception.orig.sqlstate, sqlstate)

    def test_forecast_and_multiple_horizons_round_trip(self):
        forecast = self.add_forecast()
        self.session.add_all(
            [
                ForecastPoint(**self.point_values(forecast.id, water_level_mm=-25.5)),
                ForecastPoint(
                    **self.point_values(
                        forecast.id,
                        horizon_minutes=180,
                        forecast_at=self.anchor + timedelta(hours=3),
                    )
                ),
            ]
        )
        self.session.flush()
        points = self.session.scalars(
            sa.select(ForecastPoint)
            .where(ForecastPoint.forecast_id == forecast.id)
            .order_by(ForecastPoint.horizon_minutes)
        ).all()
        self.assertEqual([point.water_level_mm for point in points], [-25.5, 0.0])
        self.assertIsNotNone(forecast.generated_at.tzinfo)
        self.assertEqual(forecast.anchor_observed_at, self.anchor)

    def test_duplicate_identity_is_rejected_even_with_different_generated_time(self):
        self.add_forecast()
        self.assert_rejected(Forecast, self.forecast_values(generated_at=self.anchor), "23505")

    def test_changed_input_model_schema_or_pipeline_can_be_saved(self):
        self.add_forecast()
        for overrides in (
            {"input_hash": "b" * 64},
            {"model_version": "other-model"},
            {"feature_schema_version": "other-schema"},
            {"pipeline_version": "other-pipeline"},
        ):
            with self.subTest(overrides=overrides):
                self.add_forecast(**overrides)

    def test_unknown_station_or_parent_is_rejected(self):
        self.assert_rejected(Forecast, self.forecast_values(station_id="unknown"), "23503")
        self.assert_rejected(ForecastPoint, self.point_values(uuid4()), "23503")

    def test_duplicate_horizon_and_parent_deletion_are_rejected(self):
        forecast = self.add_forecast()
        self.session.add(ForecastPoint(**self.point_values(forecast.id)))
        self.session.flush()
        self.assert_rejected(ForecastPoint, self.point_values(forecast.id), "23505")
        with self.assertRaises(IntegrityError):
            with self.session.begin_nested():
                self.session.execute(sa.delete(Forecast).where(Forecast.id == forecast.id))

    def test_invalid_hash_and_blank_versions_are_rejected(self):
        for overrides in (
            {"input_hash": "bad"},
            {"input_hash": "A" * 64},
            {"model_version": " "},
            {"feature_schema_version": ""},
            {"pipeline_version": " "},
        ):
            with self.subTest(overrides=overrides):
                self.assert_rejected(Forecast, self.forecast_values(**overrides), "23514")

    def test_invalid_horizons_and_nonfinite_water_are_rejected(self):
        forecast = self.add_forecast()
        for overrides in (
            {"horizon_minutes": 0},
            {"horizon_minutes": -60},
            {"water_level_mm": float("nan")},
            {"water_level_mm": float("inf")},
            {"water_level_mm": float("-inf")},
        ):
            with self.subTest(overrides=overrides):
                self.assert_rejected(
                    ForecastPoint, self.point_values(forecast.id, **overrides), "23514"
                )

    def test_failed_point_write_rolls_back_parent_in_same_transaction(self):
        forecast_id = uuid4()
        with self.assertRaises(IntegrityError):
            with self.session.begin_nested():
                self.add_forecast(id=forecast_id)
                self.session.add(ForecastPoint(**self.point_values(forecast_id, horizon_minutes=0)))
                self.session.flush()
        self.assertIsNone(self.session.get(Forecast, forecast_id))


if __name__ == "__main__":
    unittest.main()
