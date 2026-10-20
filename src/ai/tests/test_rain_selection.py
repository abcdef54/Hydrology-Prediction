"""Station ablation must preserve targets, samples, scalers and checkpoint identity."""

import hashlib
import tempfile
from pathlib import Path
from unittest import TestCase, main

import numpy as np
import pandas as pd
import torch

from src.ai.utils import get_dataloader, rank_rain_stations, select_feature_columns


IDS = ["750062", "750007", "750003", "750011"]
FEATURE_SET = "water_level+rain+flood_season"


def example_frame():
    values = np.arange(12, dtype=float)
    data = {"timestamp_utc": pd.date_range("2026-01-01", periods=12, freq="h", tz="UTC"),
            "water_level_depth": values + 100,
            "water_level_lag_60m": values + 99,
            "water_level_lag_180m": values + 97,
            "water_level_missing": np.zeros(12), "flood_season": np.zeros(12),
            "wind_test_ws": values}
    for i, sid in enumerate(IDS):
        for suffix in ("depth", "sum_3h", "sum_6h", "sum_12h", "sum_24h"):
            data[f"rain_{sid}_{suffix}"] = values + i
        data[f"rain_{sid}_missing"] = np.zeros(12)
    for h in (1, 3, 6, 12, 24):
        data[f"target_water_level_plus_{h * 60}m"] = values + 100 + h
    data["rain_750007_sum_6h"] = values
    data["rain_750062_sum_6h"] = values + np.sin(values) * 2
    data["rain_750003_sum_6h"] = values + np.cos(values) * 4
    data["rain_750011_sum_6h"] = values + np.sin(values * 2) * 8
    data["target_water_level_plus_720m"] = data["water_level_depth"] + values
    frame = pd.DataFrame(data)
    frame.loc[5, "rain_750007_depth"] = np.nan
    frame.loc[5, "rain_750007_missing"] = 1
    frame.loc[6, "target_water_level_plus_1440m"] = np.nan
    return frame


class RainSelectionTests(TestCase):
    def setUp(self):
        self.frame = example_frame()
        self.temp = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(self.temp)
        for split in ("train", "val", "test"):
            frame = self.frame.copy()
            if split != "train":
                frame.water_level_depth += 1000  # detect accidental scaler refit
            frame.to_csv(self.root / f"{split}.csv", index=False)

    def loaders(self, count):
        return get_dataloader("1h", 4, num_worker=0, pin_memory=False, drop_last=False,
                              train_shuffle=False, seq_len=4, require_current_level=True,
                              feature_set=FEATURE_SET, dataset_dir=self.root, num_rain_station=count)

    def test_selection_removes_entire_station_blocks_and_preserves_order(self):
        default = select_feature_columns(self.frame.columns, FEATURE_SET)
        selected = select_feature_columns(self.frame.columns, FEATURE_SET, ["750007"])
        self.assertEqual(len(default), 29)
        self.assertEqual(len(selected), 11)
        self.assertEqual(selected, [c for c in default if not c.startswith("rain_") or c.startswith("rain_750007_")])
        self.assertIn("rain_750007_missing", selected)
        self.assertNotIn("rain_750062_missing", selected)
        self.assertIsNone(select_feature_columns(self.frame.columns, "all"))
        all_selected = select_feature_columns(self.frame.columns, "all", ["750007"])
        self.assertIn("wind_test_ws", all_selected)
        self.assertFalse(any(c.startswith(("target_", "timestamp")) for c in all_selected))
        self.assertEqual(select_feature_columns(self.frame.columns, FEATURE_SET, IDS[::-1]), default)

    def test_invalid_selection_fails_instead_of_silently_using_all_stations(self):
        for ids in ([], ["999999"], ["750007", "750007"], [750007], ["750007x"]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                select_feature_columns(self.frame.columns, FEATURE_SET, ids)
        with self.assertRaises(ValueError):
            select_feature_columns(self.frame.columns, "water_level", ["750007"])

    def test_dataloaders_keep_same_samples_targets_and_train_fitted_scalers(self):
        original_hash = hashlib.sha256((self.root / "train.csv").read_bytes()).hexdigest()
        full = self.loaders(None)
        for count, ids, expected_size in ((1, ["750007"], 11), (2, ["750007", "750062"], 17), (3, ["750007", "750062", "750003"], 23), (4, IDS, 29)):
            loaders = self.loaders(count)
            for reference, loader in zip(full, loaders):
                ds = loader.dataset
                self.assertEqual(ds.num_features, expected_size)
                np.testing.assert_array_equal(ds.end_indices, reference.dataset.end_indices)
                np.testing.assert_array_equal(ds.y, reference.dataset.y)
                self.assertIs(ds.feature_scaler, loaders[0].dataset.feature_scaler)
                self.assertEqual(set(ds.rain_station_ids), set(ids))
                self.assertEqual(ds[0][0].shape, (4, expected_size))
            column = loaders[0].dataset.feature_cols.index("rain_750007_depth")
            self.assertEqual(loaders[0].dataset.X[5, column], 0.)
        self.assertEqual(hashlib.sha256((self.root / "train.csv").read_bytes()).hexdigest(), original_hash)

    def test_automatic_ranking_uses_train_and_rejects_invalid_counts(self):
        ranked = rank_rain_stations(self.frame)
        self.assertEqual([r["station_id"] for r in ranked], ["750007", "750062", "750003", "750011"])
        self.assertAlmostEqual(ranked[0]["pearson_r"], 1.)
        # Make validation rainfall imply the opposite ranking. Selection must
        # still use train, and every split must receive that same selection.
        validation = self.frame.copy()
        validation["rain_750007_sum_6h"] *= -1
        validation["rain_750011_sum_6h"] = np.arange(len(validation))
        validation.to_csv(self.root / "val.csv", index=False)
        self.assertEqual(rank_rain_stations(validation)[0]["station_id"], "750011")
        for loader in self.loaders(1):
            self.assertEqual(loader.dataset.rain_station_ids, ["750007"])
        for count in (0, 5, -1, True, 1.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.loaders(count)

    def test_real_trainer_forward_backward_and_distinct_station_checkpoints(self):
        from src.ai.train import build_sequence_trainer, parser, validate_args

        checkpoints = []
        for count, ids, expected_size in ((1, ["750007"], 11), (2, ["750007", "750062"], 17), (3, ["750007", "750062", "750003"], 23), (4, IDS, 29)):
            args = parser.parse_args([
                "--mlflow-ex-name", "test-only", "--dataset-dir", str(self.root),
                "--train-method", "residual_lstm", "--horizon", "1h", "--device", "cpu",
                "--feature-set", FEATURE_SET, "--lstm-seq-len", "4", "--batch-size", "4",
                "--lstm-hidden-size", "4", "--lstm-num-layers", "2",
                "--checkpoint-dir", str(self.root / "checkpoints"), "--num-rain-station", str(count),
            ])
            validate_args(args)
            trainer = build_sequence_trainer(args)
            x, y = next(iter(trainer.train_dataloader))
            self.assertEqual(x.shape[-1], expected_size)
            prediction = trainer.model(x)
            self.assertEqual(prediction.shape, y.shape)
            loss = trainer.training_loss(prediction, x, trainer.loss_func(prediction, y))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in trainer.model.parameters()))
            checkpoints.append(trainer.checkpoint_path)
        self.assertEqual(len(set(checkpoints)), 4)
        self.assertFalse((self.root / "checkpoints").exists())  # building does not train/save


if __name__ == "__main__":
    main()
