"""Keep O24 eligibility, input preprocessing and train-only scaling for O2."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
import torch

from src.ai.build_horizon_subset_dataset import build, make_dataset, SELECTED_TARGETS
from src.ai.hourly_dataset import TARGETS, read_frame
from src.ai.lstm import ResidualLSTM
from src.ai.utils import get_dataloader


class HorizonSubsetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.source = Path(self.temporary.name) / "o24"
        self.output = Path(self.temporary.name) / "o2"
        self.source.mkdir()
        spec = json.loads(Path(
            "train_data/da_vien_1h_2026-10-15_forecast_mask_o24/manifest.json"
        ).read_text())
        self.features = spec["feature_columns"]
        index = pd.date_range("2026-01-01", periods=500, freq="h", tz="UTC", name="timestamp_utc")
        frame = pd.DataFrame({column: np.arange(500, dtype=float) + i
                              for i, column in enumerate(self.features)}, index=index)
        frame["water_level_depth"] = 100 + np.arange(500, dtype=float)
        frame.loc[index[20], "weather_forecast_rain_0_3h"] = np.nan
        for hours, column in enumerate(TARGETS, start=1):
            frame[column] = frame.water_level_depth.shift(-hours)
        # Holes outside the two selected horizons must still exclude samples.
        frame.loc[index[[100, 290, 440]], "target_water_level_plus_120m"] = np.nan
        self.frames = dict(hourly=frame, train=frame.iloc[:160],
                           val=frame.iloc[185:310], test=frame.iloc[335:])
        for name, subset in self.frames.items():
            subset.to_csv(self.source / f"{name}.csv")
        (self.source / "manifest.json").write_text(json.dumps(spec))

    def generate(self):
        with redirect_stdout(io.StringIO()):
            return build(self.source, self.output)

    def test_exact_anchors_inputs_targets_and_train_only_scalers(self):
        manifest = self.generate()
        saved = joblib.load(self.output / "preprocessing.joblib")
        self.assertEqual(manifest["target_columns"], SELECTED_TARGETS)
        self.assertEqual(manifest["common_target_columns"], SELECTED_TARGETS)
        self.assertEqual((manifest["input_size"], manifest["output_size"]), (26, 2))
        self.assertEqual(manifest["purge_hours"], 24)
        original_train = make_dataset(read_frame(self.source / "train.csv"), self.features, TARGETS)
        train = make_dataset(read_frame(self.output / "train.csv"), self.features, SELECTED_TARGETS)
        for name in ("train", "val", "test"):
            parent = read_frame(self.source / f"{name}.csv")
            subset = read_frame(self.output / f"{name}.csv")
            pd.testing.assert_frame_equal(subset[self.features], parent[self.features])
            original = make_dataset(parent, self.features, TARGETS, original_train)
            selected = make_dataset(subset, self.features, SELECTED_TARGETS, train)
            np.testing.assert_array_equal(selected.end_indices, original.end_indices)
            np.testing.assert_array_equal(selected.timestamps, original.timestamps)
            np.testing.assert_array_equal(selected.X, original.X)
            np.testing.assert_allclose(subset.iloc[selected.end_indices][SELECTED_TARGETS],
                                       parent.iloc[original.end_indices][SELECTED_TARGETS])
            naive = make_dataset(parent, self.features, SELECTED_TARGETS)
            self.assertGreater(len(naive), len(selected))
        np.testing.assert_array_equal(saved["feature_scaler"].mean_, original_train.feature_scaler.mean_)
        np.testing.assert_array_equal(saved["target_scaler"].mean_, original_train.target_scaler.mean_[[5, 11]])
        full = read_frame(self.source / "hourly.csv")
        self.assertNotEqual(saved["feature_scaler"].mean_[0], full.water_level_depth.mean())

    def test_existing_training_loader_and_two_output_persistence(self):
        self.generate()
        saved = joblib.load(self.output / "preprocessing.joblib")
        loaders = get_dataloader("1h", 2, num_worker=0, pin_memory=False,
                                 dataset_dir=self.output, seq_len=96,
                                 feature_set="all", require_current_level=True,
                                 target_horizons=[6, 12])
        model = ResidualLSTM(input_size=26, hidden_size=32, num_layers=3, output_size=2,
                             level_feature_index=0, persistence_scale=saved["persistence_scale"].tolist(),
                             persistence_offset=saved["persistence_offset"].tolist()).eval()
        for loader in loaders:
            self.assertEqual(loader.dataset.feature_cols, self.features)
            self.assertEqual(loader.dataset.target_cols, SELECTED_TARGETS)
            x, y = next(iter(loader))
            self.assertEqual(tuple(x.shape), (2, 96, 26))
            self.assertEqual(tuple(y.shape), (2, 2))
            with torch.inference_mode():
                prediction = model(x)
            self.assertEqual(tuple(prediction.shape), (2, 2))
            current = (x[:, -1, 0].numpy() * saved["feature_scaler"].scale_[0]
                       + saved["feature_scaler"].mean_[0])
            np.testing.assert_allclose(saved["target_scaler"].inverse_transform(prediction.numpy()),
                                       np.repeat(current[:, None], 2, axis=1), atol=1e-4)

    def test_wrong_timestamp_target_is_rejected_before_writing(self):
        frame = read_frame(self.source / "hourly.csv")
        frame.iloc[100, frame.columns.get_loc(SELECTED_TARGETS[0])] += 1
        frame.to_csv(self.source / "hourly.csv")
        with self.assertRaises((ValueError, AssertionError)):
            self.generate()
        self.assertFalse(self.output.exists())

    def test_existing_output_is_never_overwritten(self):
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("keep")
        with self.assertRaises(FileExistsError):
            self.generate()
        self.assertEqual(marker.read_text(), "keep")

    def test_target_purge_is_required(self):
        frame = read_frame(self.source / "hourly.csv")
        frame.iloc[170:310].to_csv(self.source / "val.csv")
        with self.assertRaisesRegex(ValueError, "purge"):
            self.generate()
        self.assertFalse(self.output.exists())
