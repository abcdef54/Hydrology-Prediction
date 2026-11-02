"""Forecast-feature ablation preserves all samples, labels and fitted scales."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

from src.ai.build_horizon_subset_dataset import make_dataset, SELECTED_TARGETS
from src.ai.build_weather_horizon_ablation import build, DEFAULT_SOURCE, REMOVED_FEATURES
from src.ai.comparison import load_candidate
from src.ai.hourly_dataset import read_frame
from src.ai.lstm import ResidualLSTM
from src.ai.utils import get_dataloader


class WeatherHorizonAblationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.source = Path(self.temporary.name) / "parent"
        self.output = Path(self.temporary.name) / "ablation"
        self.source.mkdir()
        spec = json.loads((DEFAULT_SOURCE / "manifest.json").read_text())
        self.features = spec["feature_columns"]
        index = pd.date_range("2026-01-01", periods=500, freq="h", tz="UTC", name="timestamp_utc")
        frame = pd.DataFrame({column: np.arange(500, dtype=float) + i
                              for i, column in enumerate(self.features)}, index=index)
        for column in self.features:
            if column.endswith("_missing"):
                frame[column] = 0.
        frame["water_level_depth"] = 100 + np.arange(500, dtype=float)
        frame.loc[index[20], "weather_forecast_rain_0_3h"] = np.nan
        frame.loc[index[20], "weather_forecast_rain_0_3h_missing"] = 1.
        for hours, column in zip((6, 12), SELECTED_TARGETS):
            frame[column] = frame.water_level_depth.shift(-hours)
        frame.loc[index[-24:], SELECTED_TARGETS] = np.nan
        frame.loc[index[[100, 290, 440]], SELECTED_TARGETS] = np.nan
        self.frames = dict(hourly=frame, train=frame.iloc[:160],
                           val=frame.iloc[185:310], test=frame.iloc[335:])
        for name, subset in self.frames.items():
            subset.to_csv(self.source / f"{name}.csv")
        (self.source / "manifest.json").write_text(json.dumps(spec))
        train = make_dataset(read_frame(self.source / "train.csv"), self.features, SELECTED_TARGETS)
        self.saved = dict(feature_columns=self.features, target_columns=SELECTED_TARGETS,
                          feature_scaler=train.feature_scaler, target_scaler=train.target_scaler,
                          sequence_length=96, resolution_minutes=60,
                          missing_scaled_value=0., raw_missing_value=None,
                          forecast_history_policy="all_timesteps", rain_station_ids=spec["rain_station_ids"])
        joblib.dump(self.saved, self.source / "preprocessing.joblib")

    def generate(self):
        with redirect_stdout(io.StringIO()):
            return build(self.source, self.output)

    def test_only_two_columns_removed_without_refitting_or_changing_samples(self):
        with patch.object(StandardScaler, "fit", side_effect=AssertionError("unexpected scaler fit")):
            manifest = self.generate()
        features = manifest["feature_columns"]
        self.assertEqual(features, [c for c in self.features if c not in REMOVED_FEATURES])
        self.assertEqual((manifest["input_size"], manifest["output_size"]), (24, 2))
        self.assertEqual(manifest["sample_stride"], 1)
        self.assertEqual(manifest["target_columns"], SELECTED_TARGETS)
        saved = joblib.load(self.output / "preprocessing.joblib")
        indices = [self.features.index(c) for c in features]
        for attribute in ("mean_", "var_", "scale_", "n_samples_seen_"):
            np.testing.assert_array_equal(getattr(saved["feature_scaler"], attribute),
                                          getattr(self.saved["feature_scaler"], attribute)[indices])
        for attribute in ("mean_", "var_", "scale_"):
            np.testing.assert_array_equal(getattr(saved["target_scaler"], attribute),
                                          getattr(self.saved["target_scaler"], attribute))
        for name in self.frames:
            parent = read_frame(self.source / f"{name}.csv")
            child = read_frame(self.output / f"{name}.csv")
            pd.testing.assert_frame_equal(child, parent[features + SELECTED_TARGETS])
            old = make_dataset(parent, self.features, SELECTED_TARGETS, SimpleNamespace(**self.saved))
            new = make_dataset(child, features, SELECTED_TARGETS, SimpleNamespace(**saved))
            np.testing.assert_array_equal(old.end_indices, new.end_indices)
            np.testing.assert_array_equal(old.timestamps, new.timestamps)
            np.testing.assert_array_equal(new.X, old.X[:, indices])

    def test_weather_mapping_model_reload_and_existing_loader(self):
        manifest = self.generate()
        saved = joblib.load(self.output / "preprocessing.joblib")
        features = saved["feature_columns"]
        self.assertEqual([features[i] for i in saved["weather_mask_indices"]], [
            "weather_forecast_rain_0_3h_missing", "weather_forecast_rain_3_6h_missing",
            "weather_forecast_rain_6_12h_missing",
        ])
        self.assertEqual(manifest["weather_mask_indices"], [20, 21, 22])
        loaders = get_dataloader("1h", 2, num_worker=0, pin_memory=False,
                                 dataset_dir=self.output, seq_len=96, require_current_level=True,
                                 feature_set="water_level+rain+flood_season+weather_forecast",
                                 target_horizons=[6, 12])
        for policy in ("all_timesteps", "anchor_only"):
            model = ResidualLSTM(
                input_size=24, hidden_size=3, num_layers=1, output_size=2, dropout=0.,
                level_feature_index=saved["level_feature_index"],
                persistence_scale=saved["persistence_scale"].tolist(),
                persistence_offset=saved["persistence_offset"].tolist(),
                forecast_history_policy=policy,
                weather_feature_indices=saved["weather_feature_indices"],
                weather_mask_indices=saved["weather_mask_indices"],
                weather_mask_values=saved["weather_mask_values"],
            ).eval()
            torch.nn.init.normal_(model.linear.weight)
            stream = io.BytesIO()
            torch.save(model.state_dict(), stream)
            reload_metadata = {**saved, "forecast_history_policy": policy}
            reloaded = load_candidate(stream.getvalue(), reload_metadata, "cpu")
            for loader in loaders:
                x, y = next(iter(loader))
                self.assertEqual(tuple(x.shape), (2, 96, 24))
                self.assertEqual(tuple(y.shape), (2, 2))
                with torch.inference_mode():
                    prediction = model(x)
                    torch.testing.assert_close(prediction, reloaded(x))
                self.assertEqual(tuple(prediction.shape), (2, 2))
                if policy == "anchor_only":
                    prepared = model.prepare_inputs(x)
                    torch.testing.assert_close(prepared[:, -1], x[:, -1])
                    masks = torch.tensor(saved["weather_mask_values"])
                    torch.testing.assert_close(prepared[0, 0, saved["weather_mask_indices"]], masks)
                else:
                    torch.testing.assert_close(model.prepare_inputs(x), x)

    def test_existing_output_is_preserved(self):
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("keep")
        with self.assertRaises(FileExistsError):
            self.generate()
        self.assertEqual(marker.read_text(), "keep")

    def test_24_hour_control_keeps_all_forecast_inputs(self):
        with redirect_stdout(io.StringIO()):
            manifest = build(self.source, self.output, forecast_hours=24)
        self.assertEqual(manifest["input_size"], 26)
        self.assertEqual(manifest["removed_feature_columns"], [])
        for name in self.frames:
            pd.testing.assert_frame_equal(read_frame(self.source / f"{name}.csv"),
                                          read_frame(self.output / f"{name}.csv"))

    def test_rain_pool_supports_new_combinations_without_expanding_training_period(self):
        pool_dir = Path(self.temporary.name) / "pool"
        pool_dir.mkdir()
        pool = self.frames["hourly"].copy()
        added = []
        for station in ("750041", "750014"):
            for suffix in ("depth", "sum_3h", "sum_6h", "sum_12h", "sum_24h", "missing"):
                column = f"rain_{station}_{suffix}"
                added.append(column)
                pool[column] = 0. if suffix == "missing" else np.arange(len(pool), dtype=float)
                if suffix != "missing":
                    pool.loc[pool.index[185:], column] += 1000000.
        pool.to_csv(pool_dir / "hourly.csv")
        (pool_dir / "manifest.json").write_text(json.dumps({
            "feature_columns": self.features + added,
            "rain_station_aliases": {"bach_ma": "750041", "phu_bai": "750014"},
        }))
        with redirect_stdout(io.StringIO()):
            manifest = build(self.source, self.output, "all", pool_dir)
        self.assertEqual(manifest["input_size"], 36)  # Four rain gauges in this fixture.
        saved = joblib.load(self.output / "preprocessing.joblib")
        index = saved["feature_columns"].index("rain_750041_depth")
        self.assertEqual(saved["feature_scaler"].mean_[index], 79.5)
        loaders = get_dataloader("1h", 2, num_worker=0, pin_memory=False,
                                 dataset_dir=self.output, seq_len=96, require_current_level=True,
                                 feature_set="water_level+rain+flood_season+weather_forecast",
                                 rain_stations="hong_thai+bach_ma+phu_bai", target_horizons=[6, 12])
        for name, loader in zip(("train", "val", "test"), loaders):
            original = make_dataset(read_frame(self.source / f"{name}.csv"),
                                    self.features, SELECTED_TARGETS, SimpleNamespace(**self.saved))
            np.testing.assert_array_equal(pd.to_datetime(loader.dataset.timestamps, utc=True),
                                          pd.to_datetime(original.timestamps, utc=True))
            x, y = next(iter(loader))
            self.assertEqual(tuple(x.shape), (2, 96, 30))
            self.assertEqual(tuple(y.shape), (2, 2))

    def test_source_purge_is_required(self):
        self.frames["hourly"].iloc[170:310].to_csv(self.source / "val.csv")
        with self.assertRaisesRegex(ValueError, "purge"):
            self.generate()
        self.assertFalse(self.output.exists())
