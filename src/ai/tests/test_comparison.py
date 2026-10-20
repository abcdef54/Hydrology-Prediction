"""Compare physical-unit metrics on common anchors using each model's own scaler."""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.ai.comparison import paired_metrics, run_post_training_comparison, write_report


class LastLevel(torch.nn.Module):
    def __init__(self, saved, offsets):
        super().__init__()
        self.index = saved["feature_columns"].index("water_level_depth")
        fs, ts = saved["feature_scaler"], saved["target_scaler"]
        self.register_buffer("level_scale", torch.tensor(float(fs.scale_[self.index])))
        self.register_buffer("level_mean", torch.tensor(float(fs.mean_[self.index])))
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.float32))
        self.register_buffer("target_scale", torch.tensor(ts.scale_, dtype=torch.float32))
        self.register_buffer("target_mean", torch.tensor(ts.mean_, dtype=torch.float32))

    def forward(self, x):
        physical = x[:, -1, self.index, None] * self.level_scale + self.level_mean
        return (physical + self.offsets - self.target_mean) / self.target_scale


def example():
    t1, t3 = "target_water_level_plus_60m", "target_water_level_plus_180m"
    frame = pd.DataFrame({"timestamp_utc": pd.date_range("2026-01-01", periods=8, freq="h", tz="UTC"),
                          "water_level_depth": np.arange(100., 108.), "rain_750007_depth": 1.,
                          t1: np.arange(100., 108.), t3: np.arange(100., 108.)})
    frame.loc[6, t3] = np.nan
    frame.loc[5, "rain_750007_depth"] = np.nan  # missing predictor must not drop the row
    new = dict(feature_columns=["water_level_depth"], target_columns=[t1, t3], sequence_length=3,
               feature_scaler=StandardScaler().fit([[0.], [2.]]),
               target_scaler=StandardScaler().fit([[0., 0.], [2., 2.]]))
    app = dict(feature_columns=["rain_750007_depth", "water_level_depth"],
               target_columns=[t3, t1], sequence_length=5,
               feature_scaler=StandardScaler().fit([[10., 20.], [14., 28.]]),
               target_scaler=StandardScaler().fit([[10., 10.], [14., 14.]]))
    return frame, new, app


class ComparisonTests(TestCase):
    def test_common_rows_own_scalers_target_order_and_strict_200_threshold(self):
        frame, new, app = example()
        metrics, summary, points, coverage = paired_metrics(
            frame, LastLevel(new, [200., 201.]), new, LastLevel(app, [0., -200.]), app,
            batch_size=2)
        self.assertEqual(coverage["candidate_eligible"], 5)
        self.assertEqual(coverage["serving_eligible"], 3)
        self.assertEqual(coverage["common_anchors"], 3)
        self.assertEqual(set(pd.to_datetime(points.anchor_utc).dt.hour), {4, 5, 7})
        for minutes, new_rmse, app_rmse, new_bad in [(60, 200., 200., 0), (180, 201., 0., 3)]:
            group = metrics[metrics.horizon_minutes.eq(minutes)].set_index("model")
            self.assertEqual(group.loc["candidate", "rmse_mm"], new_rmse)
            self.assertEqual(group.loc["serving", "rmse_mm"], app_rmse)
            self.assertEqual(group.loc["candidate", "above_200"], new_bad)
            self.assertEqual(group.loc["serving", "above_200"], 0)
            self.assertTrue(group.n.eq(3).all())
        self.assertEqual(summary["candidate"]["mean_horizon_rmse_mm"], 200.5)
        self.assertAlmostEqual(summary["candidate"]["pooled_rmse_mm"], np.sqrt((200**2 + 201**2) / 2))
        self.assertEqual(summary["serving"]["mean_horizon_rmse_mm"], 100.)
        self.assertEqual(summary["candidate"]["above_200_pct"], 50.)
        self.assertFalse(summary["automatic_promotion"])
        with tempfile.TemporaryDirectory() as d:
            write_report(Path(d), metrics, summary, coverage)
            text = (Path(d) / "comparison.md").read_text()
            self.assertIn("200.50", text)
            self.assertIn("100.00%", text)

    def test_missing_horizons_are_explicit_and_not_mixed_into_comparison(self):
        frame, new, app = example()
        new["target_columns"] = ["target_water_level_plus_60m"]
        new["target_scaler"] = StandardScaler().fit([[0.], [2.]])
        metrics, _, _, coverage = paired_metrics(
            frame, LastLevel(new, [0.]), new, LastLevel(app, [0., 0.]), app)
        self.assertEqual(metrics.horizon_minutes.unique().tolist(), [60])
        self.assertEqual(coverage["serving_only_targets"], ["target_water_level_plus_180m"])

    def test_invalid_grid_and_no_common_samples_fail_clearly(self):
        frame, new, app = example()
        invalid = frame.copy()
        invalid.loc[1, "timestamp_utc"] = invalid.loc[0, "timestamp_utc"]
        with self.assertRaisesRegex(ValueError, "hourly grid"):
            paired_metrics(invalid, LastLevel(new, [0., 0.]), new, LastLevel(app, [0., 0.]), app)
        frame["target_water_level_plus_60m"] = np.nan
        with self.assertRaisesRegex(ValueError, "No common"):
            paired_metrics(frame, LastLevel(new, [0., 0.]), new, LastLevel(app, [0., 0.]), app)

    def test_comparison_failure_retains_training_and_writes_error_report(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = SimpleNamespace(checkpoint_path=Path(directory) / "best.pt", device="cpu", model=None)
            with patch("src.ai.comparison.compare_checkpoint", side_effect=FileNotFoundError("missing app model")):
                result = run_post_training_comparison(trainer, "test.csv", "run1", output_root=directory)
            self.assertEqual(result["status"], "failed")
            self.assertFalse(json.loads((result["output_dir"] / "comparison_error.json").read_text())["automatic_promotion"])

    def test_training_entrypoint_calls_comparison_after_training(self):
        from src.ai import train

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("train", "val", "test"):
                (root / f"{split}.csv").write_text("timestamp_utc\n")
            args = train.parser.parse_args(["--mlflow-ex-name", "test", "--dataset-dir", str(root),
                                            "--train-method", "residual_lstm", "--horizon", "1h", "--device", "cpu"])
            ds = SimpleNamespace(rain_station_ids=[], rain_station_ranking=[], seq_len=96,
                                 num_features=1, num_targets=1, target_cols=["target_water_level_plus_60m"],
                                 feature_cols=["water_level_depth"])
            trainer = Mock()
            trainer.train_dataloader.dataset = ds
            trainer.optimizer.defaults = {"lr": .01, "weight_decay": .001}
            trainer.device = "cpu"
            trainer.model_settings.return_value = {}
            order = []
            trainer.train.side_effect = lambda: order.append("train")

            def compare(*a, **kw):
                order.append("compare")
                self.assertEqual(a[1], root / "test.csv")
                return dict(status="failed", output_dir=root, metrics=None, summary=None)

            with patch.object(train.parser, "parse_args", return_value=args), \
                 patch.object(train, "build_sequence_trainer", return_value=trainer), \
                 patch.object(train, "set_seed"), patch.object(train, "log_dataset_source"), \
                 patch.object(train, "run_post_training_comparison", side_effect=compare), \
                 patch.object(train.mlflow, "set_experiment"), patch.object(train.mlflow, "start_run"), \
                 patch.object(train.mlflow, "log_params"), patch.object(train.mlflow, "log_dict"), \
                 patch.object(train.mlflow, "set_tags") as tags:
                train.main()
                self.assertEqual(order, ["train", "compare"])
                self.assertEqual(tags.call_args.args[0]["comparison/status"], "failed")


if __name__ == "__main__":
    main()
