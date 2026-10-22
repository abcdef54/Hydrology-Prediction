"""Protect temporal isolation, event sampling and validation-only experiments."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import torch

from src.ai.build_expanded_train_dataset import build
from src.ai.build_expanded_train_dataset import DEFAULT_OUTPUT
from src.ai.hourly_dataset import COMMON, TARGETS, DEFAULT_POOL, read_frame
from src.ai.expanded_train_experiment import (
    build_trainer, configure_run_directories, parser, prepare_data, prediction_metrics, run_training, validate_args,
)
from src.ai.comparison import load_candidate
import joblib
from src.ai.expanded_train_report import write_comparison_report
from src.ai.sampling import EventBalancedSampler


class EventSamplingTests(TestCase):
    def setUp(self):
        size = 80
        times = pd.date_range("2026-01-01", periods=size, freq="h", tz="UTC")
        self.frame = pd.DataFrame({"timestamp_utc": times, "water_level_depth": np.zeros(size)})
        # A long rise and a short rise belong to different events, as does a fall.
        changes = np.zeros(size)
        changes[5:15] = 250
        changes[35] = 250
        changes[60:65] = -250
        for target in COMMON:
            self.frame[target] = changes
        class Anchors(SimpleNamespace):
            def __len__(self):
                return len(self.end_indices)

        self.dataset = Anchors(end_indices=np.arange(5, size), target_cols=COMMON)

    def test_sampler_uses_eligible_sequence_positions_balances_events_and_keeps_epoch_length(self):
        sampler = EventBalancedSampler(self.frame, self.dataset)
        self.assertEqual(sampler.summary["rare_anchors"], 16)
        self.assertEqual(sampler.summary["rise"], {"anchors": 11, "events": 2})
        # The total weight of each event is equal despite tenfold duration.
        weights = sampler.weights["rise"].numpy()
        self.assertAlmostEqual(weights[:10].sum(), weights[10:].sum())
        self.assertEqual(sum(sampler.counts.values()), len(self.dataset.end_indices))
        first = list(sampler)
        self.assertEqual(first, list(sampler))
        self.assertEqual(len(first), len(self.dataset.end_indices))
        self.assertTrue(all(0 <= i < len(first) for i in first))
        self.assertGreater(len(first) - len(set(first)), 0)
        sampler.set_epoch(1)
        self.assertNotEqual(first, list(sampler))

    def test_invalid_or_absent_priority_groups_fail_instead_of_silently_changing_experiment(self):
        for options in [{"rise_fraction": .6, "fall_fraction": .5},
                        {"threshold_mm": float("nan")}, {"event_gap_hours": 0}]:
            with self.assertRaises(ValueError):
                EventBalancedSampler(self.frame, self.dataset, **options)
        with self.assertRaises(ValueError):
            EventBalancedSampler(self.frame, self.dataset, threshold_mm=1000)

    def test_dominant_horizon_handles_mixed_signs_without_using_short_horizons(self):
        self.frame.loc[:, COMMON[0]] = -9999
        self.frame.loc[35, COMMON[-1]] = -300
        sampler = EventBalancedSampler(self.frame, self.dataset)
        self.assertEqual(sampler.summary["rise"]["anchors"], 10)
        self.assertEqual(sampler.summary["fall"]["anchors"], 6)


class ExpandedDatasetTests(TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.source = self.root / "source"
        self.source.mkdir()
        times = pd.date_range("2026-01-01", periods=500, freq="h", tz="UTC", name="timestamp_utc")
        frame = pd.DataFrame({"water_level_depth": np.arange(500, dtype=float)}, index=times)
        for h, target in enumerate(TARGETS, 1):
            frame[target] = frame.water_level_depth.shift(-h)
        frame.to_csv(self.source / "hourly.csv")
        for name, start, end in [("train", 0, 120), ("val", 144, 240), ("test", 264, 500)]:
            frame.iloc[start:end].to_csv(self.source / f"{name}.csv")
        spec = dict(resolution_minutes=60, target_columns=TARGETS, feature_columns=["water_level_depth"])
        (self.source / "manifest.json").write_text(json.dumps(spec))

    def test_restores_internal_purge_preserves_old_test_and_has_no_independent_test(self):
        output = self.root / "expanded"
        with redirect_stdout(io.StringIO()):
            result = build(self.source, output)
        self.assertFalse(result["has_independent_test"])
        self.assertEqual(result["restored_internal_purge_rows"], 24)
        self.assertFalse((output / "test.csv").exists())
        pd.testing.assert_frame_equal(read_frame(output / "val.csv"), read_frame(self.source / "test.csv"))
        train = read_frame(output / "train.csv")
        self.assertEqual(len(train), 240)
        self.assertTrue(train.index.to_series().diff().dropna().eq(pd.Timedelta(hours=1)).all())
        with self.assertRaises(FileExistsError):
            build(self.source, output)

    def test_incorrect_future_labels_are_rejected_before_output_is_created(self):
        hourly = read_frame(self.source / "hourly.csv")
        hourly.iloc[20, hourly.columns.get_loc(TARGETS[0])] = -999
        hourly.to_csv(self.source / "hourly.csv")
        output = self.root / "invalid"
        with self.assertRaises(ValueError):
            build(self.source, output)
        self.assertFalse(output.exists())


class ExpandedTrainingTests(TestCase):
    def test_run_names_are_unique_and_existing_base_folders_are_reusable(self):
        with tempfile.TemporaryDirectory() as root, redirect_stdout(io.StringIO()):
            base = Path(root)
            (base / "old.pt").write_bytes(b"existing checkpoint")
            calls = []
            for _ in range(2):
                args = parser().parse_args([
                    "--variant", "event-balanced", "--output-dir", str(base),
                    "--report-dir", str(base / "reports"),
                ])
                configure_run_directories(args, "hong_thai+nham")
                self.assertEqual(args.output_dir.name, args.report_dir.name)
                self.assertFalse(args.output_dir.exists())
                self.assertFalse(args.report_dir.exists())
                calls.append(args)
            self.assertNotEqual(calls[0].output_dir, calls[1].output_dir)
            self.assertNotEqual(calls[0].report_dir, calls[1].report_dir)
            self.assertEqual((base / "old.pt").read_bytes(), b"existing checkpoint")

    def test_directory_arguments_are_optional_and_default_paths_share_run_name(self):
        args = parser().parse_args(["--variant", "uniform", "--check-only"])
        with redirect_stdout(io.StringIO()):
            configure_run_directories(args, "hong_thai+nham")
        self.assertIn("model/experiments/expanded_train", str(args.output_dir))
        self.assertIn("reports/model_comparisons/expanded_train", str(args.report_dir))
        self.assertEqual(args.output_dir.name, args.report_dir.name)
        self.assertFalse(args.output_dir.exists())
        self.assertFalse(args.report_dir.exists())

    def test_invalid_hyperparameters_are_rejected(self):
        for flags in (["--learning-rate", "0"], ["--weight-decay", "-1"],
                      ["--sequence-length", "0"], ["--dropout", "1"],
                      ["--tail-transition-mm", "nan"], ["--gradient-clip", "0"],
                      ["--horizon-weights", "1", "1", "0", "1", "1"]):
            args = parser().parse_args(["--variant", "uniform", *flags])
            with self.assertRaises(ValueError):
                validate_args(args)

    def test_custom_cli_hyperparameters_reach_training_and_checkpoint_reload(self):
        if not DEFAULT_OUTPUT.exists():
            self.skipTest("Expanded dataset required")
        torch.set_num_threads(2)
        args = parser().parse_args([
            "--variant", "uniform", "--device", "cpu", "--batch-size", "16",
            "--lstm-seq-len", "48", "--lstm-hidden-size", "16", "--lstm-num-layers", "2",
            "--lstm-drop-out", ".35", "--lstm-input-dropout", ".15", "--lstm-head-dropout", ".25",
            "--lstm-lr", ".0013", "--lstm-weight-decay", ".002", "--lstm-correction-penalty", ".03",
            "--lstm-horizon-weights", "1", "2", "3", "4", "5",
            "--lstm-tail-penalty", ".8", "--lstm-tail-threshold-mm", "150",
            "--lstm-tail-transition-mm", "30", "--gradient-accumulation", "2",
            "--gradient-clip", ".6", "--scheduler-step-size", "7", "--scheduler-gamma", ".85",
        ])
        validate_args(args)
        prepared = prepare_data(DEFAULT_OUTPUT, args.rain_stations, args.sequence_length)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            args.output_dir = Path(directory)
            trainer, _ = build_trainer(args, prepared)
            model = trainer.model
            self.assertEqual((prepared[2].seq_len, prepared[3].seq_len), (48, 48))
            self.assertEqual((model.hidden_size, model.num_layers), (16, 2))
            self.assertEqual((model.dropout, model.input_dropout, model.head_dropout), (.35, .15, .25))
            self.assertEqual(trainer.optimizer.defaults["lr"], .0013)
            self.assertEqual(trainer.optimizer.defaults["weight_decay"], .002)
            self.assertEqual(trainer.correction_penalty, .03)
            self.assertEqual((trainer.loss_func.strength, trainer.loss_func.threshold_mm,
                              trainer.loss_func.transition_mm), (.8, 150., 30.))
            np.testing.assert_allclose(trainer.loss_func.weights.numpy(), np.arange(1, 6) / 15)
            self.assertEqual((trainer.scheduler.step_size, trainer.scheduler.gamma), (7, .85))
            self.assertEqual(trainer.gradient_accumulation, 2)
            original_loader = trainer.train_dataloader
            trainer.train_dataloader = torch.utils.data.DataLoader(
                torch.utils.data.Subset(prepared[2], list(range(32))), batch_size=16,
            )
            clip = torch.nn.utils.clip_grad_norm_
            with patch("src.ai.trainers.nn.utils.clip_grad_norm_", wraps=clip) as clipping:
                trainer.train_epoch(1)
            self.assertEqual(clipping.call_count, 1)
            self.assertEqual(clipping.call_args.kwargs["max_norm"], .6)
            self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
            trainer.train_dataloader = original_loader
            trainer.checkpoint_path.parent.mkdir(parents=True)
            trainer.last_validation_metrics = {"val/large_error_count": 1, "val/rmse_mm": 1.}
            trainer.record_validation(1, .1)
            preprocessing_path = trainer.save_preprocessing()
            saved = joblib.load(preprocessing_path)
            self.assertEqual(saved["sequence_length"], 48)
            reloaded = load_candidate(trainer.checkpoint_path.read_bytes(), saved, "cpu")
            x, _ = next(iter(trainer.train_dataloader))
            with torch.inference_mode():
                torch.testing.assert_close(reloaded(x), model.eval()(x))
            args.scheduler = "none"
            without_scheduler, _ = build_trainer(args, prepared)
            self.assertIsNone(without_scheduler.scheduler)

    def test_rmse_checkpoint_and_patience_ignore_improving_loss_or_error_count(self):
        with tempfile.TemporaryDirectory() as directory:
            args = parser().parse_args(["--variant", "uniform", "--checkpoint-metric", "rmse",
                                        "--device", "cpu", "--output-dir", directory])
            from src.ai.trainers import LSTMTrainer
            model = torch.nn.Linear(1, 1)
            loader = SimpleNamespace(dataset=SimpleNamespace(feature_set="all"))
            trainer = LSTMTrainer(
                model, "1h", 4, loader, loader, None, torch.nn.MSELoss(),
                torch.optim.SGD(model.parameters(), lr=.01), None, 2,
                checkpoint_metric=args.checkpoint_metric, checkpoint_dir=directory,
            )
            trainer.checkpoint_path.parent.mkdir(parents=True)
            for epoch, loss, count, rmse, expected_epoch, patience in [
                (1, .5, 51, 80., 1, 0), (2, .4, 45, 82., 1, 1),
                (3, .7, 60, 79., 3, 0), (4, .1, 40, 79., 3, 1),
            ]:
                trainer.last_validation_metrics = {"val/large_error_count": count, "val/rmse_mm": rmse}
                trainer.record_validation(epoch, loss)
                self.assertEqual(trainer.best_checkpoint_epoch, expected_epoch)
                self.assertEqual(trainer.epochs_without_improvement, patience)
            self.assertEqual(trainer.best_checkpoint_score, (79.,))

    def test_validation_only_loop_never_calls_shared_automatic_test_or_uploads(self):
        args = SimpleNamespace(num_epochs=1, early_stopping=25, report_dir=Path("unused-report"))
        trainer = Mock()
        trainer.checkpoint_path = Path(self.enterContext(tempfile.TemporaryDirectory())) / "1h/model.pt"
        trainer.train_epoch.return_value = .1
        trainer.evaluate_loss.return_value = .2
        trainer.last_validation_metrics = {"val/large_error_count": 1}
        trainer.epochs_without_improvement = 0
        trainer.best_checkpoint_epoch = 1
        logger = Mock()
        with patch("src.ai.expanded_train_experiment.compare_references", return_value={}) as compare, \
                redirect_stdout(io.StringIO()):
            run_training(args, None, trainer, None, logger)
        trainer.train.assert_not_called()
        trainer.test.assert_not_called()
        trainer.save_preprocessing.assert_called_once()
        self.assertEqual(compare.call_args.kwargs["report_dir"], args.report_dir)
        self.assertEqual(compare.call_args.kwargs["metadata"]["best_checkpoint_epoch"], 1)
        self.assertFalse(any("test/" in key for call in logger.call_args_list for key in call.args[0]))

    def test_real_pool_scalers_fit_expanded_train_only_and_both_arms_share_initialization(self):
        if not DEFAULT_POOL.exists():
            self.skipTest("Local rain pool required")
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory) / "expanded"
            build(DEFAULT_POOL, output)
            prepared = prepare_data(output)
            _, frames, train, validation = prepared
            expected = np.nanmean(frames["train"][train.feature_cols].to_numpy(dtype=np.float32), axis=0, dtype=np.float64)
            np.testing.assert_allclose(train.feature_scaler.mean_, expected, atol=1e-10)
            self.assertIs(train.feature_scaler, validation.feature_scaler)
            self.assertEqual((len(train), len(validation), train.num_features), (7199, 592, 26))
            states = []
            for variant in ("uniform", "event-balanced"):
                torch.manual_seed(42)
                args = parser().parse_args(["--variant", variant, "--device", "cpu"])
                trainer, sampler = build_trainer(args, prepared)
                states.append({k: v.clone() for k, v in trainer.model.state_dict().items()})
                x, y = next(iter(trainer.train_dataloader))
                pred = trainer.model(x)
                loss = trainer.training_loss(pred, x, trainer.loss_func(pred, y))
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in trainer.model.parameters() if p.grad is not None))
                self.assertEqual(len(trainer.train_dataloader.sampler), len(train))
                self.assertIsNone(trainer.test_dataloader)
                if sampler is not None:
                    self.assertEqual(sum(sampler.counts.values()), 7199)
            for key in states[0]:
                torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)


class ExpandedReportTests(TestCase):
    def example_points(self):
        rows = []
        for model in ("legacy", "model_b", "candidate"):
            for anchor in pd.date_range("2026-10-01", periods=2, freq="h", tz="UTC"):
                for horizon in (1, 3, 6, 12, 24):
                    error = 210 if model == "legacy" else 10
                    rows.append(dict(model=model, anchor_utc=anchor, horizon_hours=horizon,
                                     target_utc=anchor + pd.Timedelta(hours=horizon),
                                     current_mm=100., actual_mm=350., predicted_mm=350. + error))
        return pd.DataFrame(rows)

    def test_report_compares_all_three_models_and_labels_validation_without_uploading(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "report"
            points = self.example_points()
            metadata = dict(variant="uniform", seed=42, best_checkpoint_epoch=43,
                            dataset_dir="local-dataset", checkpoint_path="local-checkpoint")
            report = write_comparison_report(output, points, metadata)
            text = report.read_text()
            for label in ("Production cũ", "Model B", "Model mới", "+1h", "+24h", "validation mới = test cũ"):
                self.assertIn(label, text)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertFalse(manifest["independent_test"])
            self.assertEqual(manifest["coverage"]["common_anchors"], 2)
            self.assertEqual(manifest["summary"]["legacy"]["n200"], 10)
            self.assertEqual(manifest["summary"]["model_b"]["n200"], 0)
            self.assertEqual(len(pd.read_csv(output / "predictions.csv.gz")), 30)
            with self.assertRaises(FileExistsError):
                write_comparison_report(output, points, metadata)

    def test_missing_pairs_or_different_truth_are_rejected_before_creating_report(self):
        points = self.example_points()
        invalid = points.copy()
        invalid.loc[0, "actual_mm"] = -1
        for frame in (points.iloc[1:], invalid):
            with tempfile.TemporaryDirectory() as root:
                path = Path(root) / "invalid"
                with self.assertRaises(ValueError):
                    write_comparison_report(path, frame, {})
                self.assertFalse(path.exists())
