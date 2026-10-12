"""Boundary, missingness and leakage checks independent of local artifacts/DB."""

from datetime import datetime, timezone
from pathlib import Path
from dataclasses import replace
import tempfile
import joblib
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.backend.prediction.config import (
    StationModelConfig,
    StationModelRegistry,
    ModelNotConfiguredError,
)
from src.backend.prediction.features import build_hourly_features as build_configured_features
from src.backend.prediction.inputs import InsufficientHistoryError, PredictionInputService
from src.backend.prediction.preprocessing import InputPreprocessor

TEST_CONFIG = StationModelConfig(
    station_id="0074647403",
    model_type="residual_lstm",
    model_version="test-model-a",
    model_path=Path("model-a.pt"),
    preprocessing_path=Path("preprocessing-a.joblib"),
    sequence_length=96,
    rain_station_ids=("750062", "750007", "750003", "750011"),
    forecast_horizons_minutes=(60, 180, 360, 720, 1440),
    feature_schema_version="test-hourly-v1",
)
WATER_STATION = TEST_CONFIG.station_id
RAIN_STATIONS = TEST_CONFIG.rain_station_ids
FEATURE_COLUMNS = TEST_CONFIG.feature_columns


def build_hourly_features(observations, reference):
    return build_configured_features(observations, reference, TEST_CONFIG)


def example_observations(hours=150):
    timestamps = pd.date_range("2026-08-27T00:00:00Z", periods=hours, freq="h")
    rows = []
    for hour, timestamp in enumerate(timestamps):
        rows.append((WATER_STATION, timestamp, float(hour)))
        for station in RAIN_STATIONS:
            rows.append((station, timestamp, 1.0))
    return pd.DataFrame(rows, columns=["station_id", "observed_at", "depth"])


def example_preprocessor(config=TEST_CONFIG):
    # Synthetic fitted fixture only; the production path never fits a scaler.
    scaler = StandardScaler().fit(
        np.stack([np.zeros(len(config.feature_columns)), np.full(len(config.feature_columns), 2.0)])
    )
    return InputPreprocessor(scaler, config)


class FeatureTests(unittest.TestCase):
    def test_right_closed_window_and_future_exclusion(self):
        observations = example_observations(150)
        reference = observations.observed_at.max()
        water_row = pd.DataFrame(
            [
                (WATER_STATION, reference - pd.Timedelta(minutes=30), 300.0),
                (WATER_STATION, reference + pd.Timedelta(minutes=10), 99999.0),
            ],
            columns=observations.columns,
        )
        features = build_hourly_features(
            pd.concat([observations, water_row]), reference.to_pydatetime()
        )
        self.assertEqual(tuple(features.columns), FEATURE_COLUMNS)
        self.assertEqual(features.iloc[-1].water_level_depth, (149.0 + 300.0) / 2)
        self.assertEqual(features.iloc[-1].water_level_lag_60m, 148.0)
        self.assertEqual(features.iloc[-1].water_level_lag_180m, 146.0)
        self.assertEqual(features.iloc[-1].rain_750062_sum_24h, 24.0)
        baseline = build_hourly_features(observations, reference.to_pydatetime())
        future_only = pd.concat([observations, water_row.iloc[1:]])
        pd.testing.assert_frame_equal(
            baseline, build_hourly_features(future_only, reference.to_pydatetime())
        )

    def test_local_calendar_season_boundary(self):
        observations = example_observations(150)
        features = build_hourly_features(
            observations, observations.observed_at.max().to_pydatetime()
        )
        self.assertEqual(features.loc["2026-08-31T16:00:00Z", "flood_season"], 0)
        self.assertEqual(features.loc["2026-08-31T17:00:00Z", "flood_season"], 1)

    def test_missing_rain_is_not_zero_and_missing_windows_stay_nan(self):
        observations = example_observations()
        missing_time = observations.observed_at.max() - pd.Timedelta(hours=1)
        observations = observations.loc[
            ~(
                (observations.station_id == RAIN_STATIONS[0])
                & (observations.observed_at == missing_time)
            )
        ]
        observations = observations.loc[observations.station_id != RAIN_STATIONS[-1]]
        features = build_hourly_features(
            observations, observations.observed_at.max().to_pydatetime()
        )
        self.assertTrue(pd.isna(features.loc[missing_time, "rain_750062_depth"]))
        self.assertEqual(features.loc[missing_time, "rain_750062_missing"], 1)
        self.assertTrue(pd.isna(features.iloc[-1].rain_750062_sum_3h))
        self.assertTrue(features.rain_750011_depth.isna().all())

    def test_preprocessing_and_overrides_only_change_last_timestep(self):
        observations = example_observations()
        features = build_hourly_features(
            observations, observations.observed_at.max().to_pydatetime()
        ).tail(96)
        original = features.copy()
        preprocessor = example_preprocessor()
        changed, tensor = preprocessor.prepare_features(
            features, {"rain_750062_depth": None, "water_level_depth": 200.0}
        )
        self.assertEqual(tuple(tensor.shape), (1, 96, 29))
        self.assertEqual(str(tensor.dtype), "torch.float32")
        self.assertEqual(tensor[0, -1, 4].item(), 0.0)
        self.assertEqual(tensor[0, -1, 0].item(), 199.0)
        pd.testing.assert_frame_equal(features, original)
        pd.testing.assert_frame_equal(changed.iloc[:-1], original.iloc[:-1])
        for overrides in (
            {"water_level_depth": None},
            {"flood_season": 2},
            {"unknown": 1},
            {"rain_750062_depth": float("inf")},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                preprocessor.prepare_features(features, overrides)
        with self.assertRaises(ValueError):
            preprocessor.prepare_features(features.loc[:, list(reversed(FEATURE_COLUMNS))])

    def test_naive_reference_is_rejected(self):
        with self.assertRaises(ValueError):
            build_hourly_features(example_observations(), datetime(2026, 9, 1))


class HistorySelectionTests(unittest.TestCase):
    def select(self, observations, reference):
        service = PredictionInputService(None, example_preprocessor())

        class ReadConnection:
            def scalar(self, query):
                water = observations.loc[observations.station_id == WATER_STATION, "observed_at"]
                return water.min().to_pydatetime()

        def read_sql(query, connection):
            parameters = query.compile().params
            start = parameters["observed_at_1"]
            end = parameters["observed_at_2"]
            return observations.loc[
                (observations.observed_at > start) & (observations.observed_at <= end)
            ].copy()

        with patch("src.backend.prediction.inputs.pd.read_sql", side_effect=read_sql) as reader:
            selected = service._select_history(ReadConnection(), reference)
            return selected, reader.call_count

    def test_expand_history_allow_gaps_and_keep_physical_lags(self):
        observations = example_observations(220)
        reference = observations.observed_at.max()
        # Last 120h contains only 60 water observations, requiring an older query.
        remove = (observations.station_id == WATER_STATION) & (
            observations.observed_at.dt.hour % 2 == 0
        )
        observations = observations.loc[~remove]
        selected, query_count = self.select(observations, reference)
        self.assertEqual(query_count, 2)
        self.assertEqual(len(selected), 96)
        self.assertTrue(selected.water_level_lag_60m.isna().all())
        self.assertTrue(selected.water_level_depth.notna().all())
        self.assertTrue((selected.index.to_series().diff().dropna() == pd.Timedelta(hours=2)).all())
        self.assertTrue(selected.rain_750062_sum_24h.eq(24.0).all())

    def test_gaps_require_extra_rain_warmup_even_with_96_water_rows(self):
        observations = example_observations(200)
        reference = observations.observed_at.max()
        missing_hours = pd.date_range("2026-09-03T12:00:00Z", periods=10, freq="h")
        observations = observations.loc[
            ~(
                (observations.station_id == WATER_STATION)
                & observations.observed_at.isin(missing_hours)
            )
        ]
        selected, query_count = self.select(observations, reference)
        self.assertEqual(query_count, 2)
        self.assertEqual(len(selected), 96)
        self.assertTrue(selected.rain_750062_sum_24h.eq(24.0).all())

    def test_insufficient_history(self):
        observations = example_observations(20)
        with self.assertRaises(InsufficientHistoryError) as raised:
            self.select(observations, observations.observed_at.max())
        self.assertEqual(raised.exception.available, 20)


class StationModelTests(unittest.TestCase):
    def test_registry_rejects_unknown_station_and_duplicate_configuration(self):
        registry = StationModelRegistry([TEST_CONFIG])
        self.assertEqual(registry.get(TEST_CONFIG.station_id), TEST_CONFIG)
        with self.assertRaises(ModelNotConfiguredError):
            registry.get("unconfigured-water-station")
        with self.assertRaises(ValueError):
            StationModelRegistry([TEST_CONFIG, TEST_CONFIG])

    def test_two_stations_use_separate_inputs_and_sequence_shapes(self):
        config_b = replace(
            TEST_CONFIG,
            station_id="water-b",
            model_version="test-model-b",
            model_path=Path("model-b.pt"),
            preprocessing_path=Path("preprocessing-b.joblib"),
            sequence_length=48,
            rain_station_ids=("rain-b1", "rain-b2"),
        )
        observations_a = example_observations()
        timestamps = observations_a.observed_at.drop_duplicates()
        rows_b = []
        for hour, timestamp in enumerate(timestamps):
            rows_b.append((config_b.station_id, timestamp, 1000.0 + hour))
            for rain_station in config_b.rain_station_ids:
                rows_b.append((rain_station, timestamp, 10.0))
        observations = pd.concat(
            [observations_a, pd.DataFrame(rows_b, columns=observations_a.columns)]
        )
        reference = observations.observed_at.max().to_pydatetime()
        registry = StationModelRegistry([TEST_CONFIG, config_b])
        for station_id, expected_level, expected_rain in (
            (TEST_CONFIG.station_id, 149.0, 1.0),
            (config_b.station_id, 1149.0, 10.0),
        ):
            with self.subTest(station=station_id):
                config = registry.get(station_id)
                features = build_configured_features(observations, reference, config).tail(
                    config.sequence_length
                )
                preprocessor = example_preprocessor(config)
                service = PredictionInputService(None, preprocessor)
                self.assertEqual(service.config.station_id, station_id)
                raw, tensor = preprocessor.prepare_features(features)
                self.assertEqual(
                    tuple(tensor.shape), (1, config.sequence_length, len(config.feature_columns))
                )
                self.assertEqual(raw.iloc[-1].water_level_depth, expected_level)
                self.assertEqual(
                    raw.iloc[-1][f"rain_{config.rain_station_ids[0]}_depth"], expected_rain
                )

    def test_saved_preprocessing_must_match_its_station_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preprocessing.joblib"
            joblib.dump(
                {
                    "feature_scaler": example_preprocessor().feature_scaler,
                    "feature_columns": list(TEST_CONFIG.feature_columns),
                    "sequence_length": 96,
                    "missing_scaled_value": 0.0,
                },
                path,
            )
            configured = replace(TEST_CONFIG, preprocessing_path=path)
            loaded = InputPreprocessor.from_file(configured)
            self.assertEqual(loaded.config, configured)
            with self.assertRaises(ValueError):
                InputPreprocessor.from_file(replace(configured, sequence_length=48))
            with self.assertRaises(ValueError):
                InputPreprocessor.from_file(replace(configured, rain_station_ids=("another-rain",)))


if __name__ == "__main__":
    unittest.main()
