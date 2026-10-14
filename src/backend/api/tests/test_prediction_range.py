"""Bounded range contract and original-input-only replay, without real persistence."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from unittest import TestCase
from unittest.mock import MagicMock, patch
from uuid import uuid4
from fastapi.testclient import TestClient
from src.backend.main import create_app
from src.backend.prediction.config import StationModelRegistry
from src.backend.prediction.inputs import InsufficientHistoryError
from src.backend.prediction.cached_service import AnchorUnavailableError
from src.backend.prediction.results import ForecastPoint, ForecastResult
from src.backend.prediction.tests.test_inputs import TEST_CONFIG


class RangeApiTests(TestCase):
    def setUp(self):
        self.engine = MagicMock()
        self.connection = self.engine.connect.return_value.__enter__.return_value
        self.connection.scalar.return_value = "mm"
        self.app = create_app(engine=self.engine, registry=StationModelRegistry([TEST_CONFIG]))
        self.client = self.enterContext(TestClient(self.app, raise_server_exceptions=False))
        self.start = datetime(2026, 9, 13, 4, tzinfo=timezone.utc)
        self.body = {
            "station_id": TEST_CONFIG.station_id,
            "start": self.start.isoformat(),
            "end": (self.start + timedelta(hours=2)).isoformat(),
            "horizon_minutes": 60,
        }
        self.service = MagicMock()
        self.service.runtime.pipeline_version = "pipeline"
        self.service.predict.side_effect = self.forecast
        self.enterContext(
            patch(
                "src.backend.api.prediction_range.get_station_model_config",
                return_value=TEST_CONFIG,
            )
        )
        self.loader = self.enterContext(
            patch(
                "src.backend.api.prediction_range.get_prediction_service", return_value=self.service
            )
        )
        self.enterContext(
            patch(
                "src.backend.api.prediction_range.resolve_forecast_identity",
                return_value=(TEST_CONFIG.model_version, "pipeline"),
            )
        )
        self.actual_rows = [
            {
                "timestamp": self.start + timedelta(hours=i),
                "actual_water_level_mm": 0 if i == 0 else None,
                "actual_sample_count": 6 if i == 0 else 0,
            }
            for i in range(2)
        ]
        self.stored_rows = []

        def execute(statement):
            result = MagicMock()
            result.mappings.return_value.all.return_value = (
                self.stored_rows if "latest_revision" in str(statement) else self.actual_rows
            )
            return result

        self.connection.execute.side_effect = execute

    def forecast(self, reference_time, *, required_anchor):
        self.assertEqual(reference_time, required_anchor)
        return ForecastResult(
            TEST_CONFIG.station_id,
            TEST_CONFIG.model_version,
            TEST_CONFIG.feature_schema_version,
            reference_time,
            reference_time,
            reference_time,
            tuple(
                ForecastPoint(h, reference_time + timedelta(minutes=h), 100)
                for h in TEST_CONFIG.forecast_horizons_minutes
            ),
            forecast_id=uuid4(),
            generated_at=reference_time,
            pipeline_version="pipeline",
            reused=True,
        )

    def test_exact_hours_use_original_inputs_and_return_stats_and_actuals(self):
        response = self.client.post("/api/v1/predictions/range", json=self.body)
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(data["reused_count"], 2)
        self.assertEqual(data["created_count"], 0)
        self.assertEqual(data["points"][0]["actual_water_level_mm"], 0)
        self.assertIsNone(data["points"][1]["actual_water_level_mm"])
        self.assertEqual(
            self.service.predict.call_args_list[0].args[0], self.start - timedelta(hours=1)
        )
        self.assertEqual(len(data["points"]), 2)

    def test_missing_history_and_shifted_anchor_are_skipped_not_mislabeled(self):
        self.service.predict.side_effect = [
            InsufficientHistoryError(96, 10),
            AnchorUnavailableError("gap"),
        ]
        response = self.client.post("/api/v1/predictions/range", json=self.body)
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(
            [p["status"] for p in data["points"]], ["insufficient_history", "anchor_unavailable"]
        )
        self.assertEqual(data["skipped_count"], 2)
        self.assertIsNone(data["points"][0]["forecast_id"])

    def test_future_anchor_does_not_infer(self):
        self.start = datetime(2999, 1, 1, tzinfo=timezone.utc)
        self.actual_rows = [
            {
                "timestamp": self.start + timedelta(hours=i),
                "actual_water_level_mm": None,
                "actual_sample_count": 0,
            }
            for i in range(2)
        ]
        response = self.client.post(
            "/api/v1/predictions/range",
            json={
                **self.body,
                "start": self.start.isoformat(),
                "end": (self.start + timedelta(hours=2)).isoformat(),
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.service.predict.assert_not_called()
        self.assertEqual(response.json()["skipped_count"], 2)

    def test_bounds_overrides_invalid_horizon_and_pipeline_changes(self):
        for changes in [
            {"end": (self.start + timedelta(hours=25)).isoformat()},
            {"start": "2026-09-13T04:30:00Z"},
            {"overrides": {"water_level_depth": 1}},
            {"horizon_minutes": 2},
            {"start": "2026-09-13T04:00:00"},
        ]:
            with self.subTest(changes=changes):
                response = self.client.post(
                    "/api/v1/predictions/range", json={**self.body, **changes}
                )
                self.assertEqual(response.status_code, 422, response.text)
        response = self.client.post(
            "/api/v1/predictions/range",
            json={
                **self.body,
                "model_version": TEST_CONFIG.model_version,
                "pipeline_version": "other",
            },
        )
        self.assertEqual(response.status_code, 409, response.text)
        self.service.predict.assert_not_called()

    def test_options_does_not_load_runtime_or_prepare_inputs(self):
        response = self.client.get(f"/api/v1/stations/{TEST_CONFIG.station_id}/prediction-options")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            response.json()["forecast_horizons_minutes"],
            list(TEST_CONFIG.forecast_horizons_minutes),
        )
        self.service.predict.assert_not_called()

    def test_all_stored_points_skip_runtime_and_input_preparation(self):
        self.stored_rows = [
            {
                "forecast_at": self.start + timedelta(hours=i),
                "anchor_observed_at": self.start + timedelta(hours=i - 1),
                "forecast_id": uuid4(),
                "predicted_water_level_mm": 123.0,
            }
            for i in range(2)
        ]
        response = self.client.post("/api/v1/predictions/range", json=self.body)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["reused_count"], 2)
        self.loader.assert_not_called()
        self.service.predict.assert_not_called()
