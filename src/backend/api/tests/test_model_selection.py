"""Select one registered model without falling back or mixing response identities."""

from dataclasses import replace
from unittest import TestCase
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.backend.main import create_app
from src.backend.prediction.config import StationModelRegistry
from src.backend.prediction.features import build_hourly_features
from src.backend.prediction.inputs import PreparedInput
from src.backend.prediction.results import ForecastPoint, ForecastResult
from src.backend.prediction.tests.test_inputs import TEST_CONFIG, example_observations, example_preprocessor


class ModelSelectionApiTests(TestCase):
    def setUp(self):
        self.b = replace(TEST_CONFIG, model_id="model_b", model_version="b-version", name="Model B")
        self.engine = MagicMock()
        connection = self.engine.connect.return_value.__enter__.return_value
        connection.execute.return_value.mappings.return_value.first.return_value = {
            "id": TEST_CONFIG.station_id, "station_type": "water_level", "enabled": True,
            "prediction_enabled": True,
        }
        connection.execute.return_value.all.return_value = [(TEST_CONFIG.station_id, "mm")]
        connection.scalar.return_value = "mm"
        self.app = create_app(engine=self.engine, registry=StationModelRegistry([TEST_CONFIG, self.b]))
        self.client = self.enterContext(TestClient(self.app, raise_server_exceptions=False))
        self.url = f"/api/v1/stations/{TEST_CONFIG.station_id}"

    def test_model_catalog_and_options_preserve_default_and_select_requested_model(self):
        with patch("src.backend.api.predictions.ResidualLSTMRuntime") as runtime:
            response = self.client.get(self.url + "/prediction-models")
            self.assertEqual(response.status_code, 200, response.text)
            catalog = response.json()
            self.assertEqual(catalog["default_model_id"], "legacy")
            self.assertEqual([m["model_id"] for m in catalog["models"]], ["legacy", "model_b"])
            self.assertEqual(catalog["models"][1]["name"], "Model B")
            response = self.client.get(self.url + "/prediction-options", params={"model_id": "model_b"})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["model_version"], "b-version")
            self.assertEqual(response.json()["model_id"], "model_b")
            runtime.assert_not_called()

    def test_unknown_model_does_not_fall_back_to_legacy_for_any_prediction_entrypoint(self):
        with patch("src.backend.api.predictions.get_prediction_service") as loader:
            for suffix in ["prediction-options", "prediction-inputs"]:
                response = self.client.get(self.url + "/" + suffix, params={"model_id": "missing"})
                self.assertEqual(response.status_code, 422, response.text)
                self.assertEqual(response.json()["error"]["code"], "MODEL_NOT_CONFIGURED")
            response = self.client.post("/api/v1/predictions", json={
                "station_id": TEST_CONFIG.station_id, "model_id": "missing"})
            self.assertEqual(response.status_code, 422, response.text)
            loader.assert_not_called()

    def test_single_and_scenario_prediction_use_selected_runtime_and_return_identity(self):
        from datetime import datetime, timedelta, timezone
        anchor = datetime(2026, 10, 1, tzinfo=timezone.utc)
        service = MagicMock()
        service.predict.return_value = ForecastResult(
            self.b.station_id, self.b.model_version, self.b.feature_schema_version,
            anchor, anchor, anchor,
            tuple(ForecastPoint(h, anchor + timedelta(minutes=h), 500.) for h in self.b.forecast_horizons_minutes),
        )
        with patch("src.backend.api.predictions.get_prediction_service", return_value=service) as loader:
            for edits in [{}, {"water_level_depth": 501.}]:
                response = self.client.post("/api/v1/predictions", json={
                    "station_id": self.b.station_id, "model_id": "model_b",
                    "reference_time": anchor.isoformat(), "overrides": edits})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["model_id"], "model_b")
                self.assertEqual(response.json()["model_version"], "b-version")
                self.assertEqual(loader.call_args.args[1], self.b)

    def test_input_query_and_dependency_resolve_the_same_selected_scaler(self):
        observations = example_observations()
        anchor = observations.observed_at.max().to_pydatetime()

        def make_service(engine, preprocessor):
            config = preprocessor.config
            raw = build_hourly_features(observations, anchor, config).tail(96)
            raw, tensor = preprocessor.prepare_features(raw)
            prepared = PreparedInput(config.station_id, config.model_version, config.feature_schema_version,
                                     tensor, raw, raw.index, anchor, anchor, anchor)
            service = MagicMock(config=config)
            service.prepare.return_value = prepared
            return service

        with patch("src.backend.api.prediction_inputs.InputPreprocessor.from_file", side_effect=example_preprocessor) as scaler, \
                patch("src.backend.api.prediction_inputs.PredictionInputService", side_effect=make_service):
            response = self.client.get(self.url + "/prediction-inputs", params={
                "model_id": "model_b", "reference_time": anchor.isoformat()})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["model_id"], "model_b")
            self.assertEqual(response.json()["model_version"], "b-version")
            self.assertEqual(scaler.call_args.args[0], self.b)
