"""Real PostgreSQL range, revision and hourly-truth checks; all fixtures roll back."""

from datetime import datetime, timedelta, timezone
from unittest import TestCase, main
from uuid import uuid4

import sqlalchemy as sa

from src.backend.api.forecast_history import forecast_comparison_query
from src.backend.api.schemas import ForecastHistoryQuery
from src.backend.db.models import Forecast, ForecastPoint, Observation
from src.backend.db.session import build_engine


class ForecastComparisonTests(TestCase):
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
        self.target = datetime(2020, 1, 2, 2, tzinfo=timezone.utc)
        self.station = "0074647403"
        self.model = "comparison-fixture"
        self.pipeline = "comparison-pipeline"
        self.now = datetime(2026, 10, 13, tzinfo=timezone.utc)

    def insert_forecast(self, target=None, predicted=100, generated=None, pipeline=None):
        forecast_id = uuid4()
        target = target or self.target
        self.connection.execute(
            sa.insert(Forecast).values(
                id=forecast_id,
                station_id=self.station,
                model_version=self.model,
                feature_schema_version="comparison-schema",
                pipeline_version=pipeline or self.pipeline,
                input_hash=uuid4().hex * 2,
                anchor_observed_at=target - timedelta(hours=1),
                generated_at=generated or self.target,
            )
        )
        self.connection.execute(
            sa.insert(ForecastPoint).values(
                forecast_id=forecast_id,
                horizon_minutes=60,
                forecast_at=target,
                water_level_mm=predicted,
            )
        )
        return forecast_id

    def insert_observation(self, timestamp, depth):
        self.connection.execute(
            sa.insert(Observation).values(
                station_id=self.station,
                observed_at=timestamp,
                depth=depth,
                source_name="hue_dcu",
                source_collection="water_level_depth",
                quality_status="missing" if depth is None else "observed",
            )
        )

    def read(self, **overrides):
        parameters = {
            "start": self.target,
            "end": self.target + timedelta(hours=3),
            "horizon_minutes": 60,
        }
        parameters.update(overrides)
        query = ForecastHistoryQuery(**parameters)
        return (
            self.connection.execute(
                forecast_comparison_query(self.station, self.model, self.pipeline, query, self.now)
            )
            .mappings()
            .all()
        )

    def test_right_closed_hourly_mean_missing_and_zero_are_correct(self):
        self.insert_forecast()
        self.insert_observation(self.target - timedelta(hours=1), 999)  # Excluded left edge.
        self.insert_observation(self.target - timedelta(minutes=30), 0)
        self.insert_observation(self.target - timedelta(minutes=20), None)
        self.insert_observation(self.target - timedelta(minutes=10), 20)
        self.insert_observation(self.target, 40)  # Included right edge.
        self.insert_observation(self.target + timedelta(minutes=1), 999)  # Excluded future row.
        row = self.read()[0]
        self.assertEqual(row["actual_water_level_mm"], 20)
        self.assertEqual(row["actual_sample_count"], 3)

    def test_latest_revision_one_pipeline_and_exclusive_cursor(self):
        self.insert_forecast(predicted=100)
        latest_id = self.insert_forecast(
            predicted=200, generated=self.target + timedelta(minutes=1)
        )
        self.insert_forecast(predicted=999, pipeline="other-device")
        newer_target = self.target + timedelta(hours=1)
        self.insert_forecast(target=newer_target, predicted=300)
        self.insert_forecast(
            target=self.target + timedelta(hours=3), predicted=400
        )  # Excluded end.
        first = self.read(limit=1)
        self.assertEqual(len(first), 2)  # Page + one lookahead.
        self.assertEqual(first[0]["predicted_water_level_mm"], 300)
        self.assertEqual(first[1]["forecast_id"], latest_id)
        second = self.read(limit=1, before=newer_target)
        self.assertEqual(len(second), 1)
        self.assertEqual(second[0]["forecast_id"], latest_id)
        self.assertEqual(second[0]["predicted_water_level_mm"], 200)
        self.assertIsNone(second[0]["actual_water_level_mm"])
        self.assertEqual(second[0]["actual_sample_count"], 0)
        self.assertEqual(self.read(horizon_minutes=180), [])

    def test_future_truth_is_not_returned_even_if_future_rows_exist(self):
        future = datetime(2099, 1, 1, 2, tzinfo=timezone.utc)
        self.insert_forecast(target=future)
        self.insert_observation(future, 700)
        row = self.read(start=future, end=future + timedelta(hours=1))[0]
        self.assertIsNone(row["actual_water_level_mm"])
        self.assertEqual(row["actual_sample_count"], 0)


if __name__ == "__main__":
    main()
