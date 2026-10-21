"""Physical time, target subsets and scaler isolation for the temporal experiments."""

from pathlib import Path
import tempfile
import unittest

import pandas as pd
import torch

from research.build_rain_station_dataset import DEFAULT_POOL
from research.build_output24_dataset import read_frame
from research.build_temporal_experiment_dataset import build
from src.ai.train import build_sequence_trainer, parser, validate_args
from src.ai.utils import get_dataloader


ROOT = Path(__file__).resolve().parents[3]


@unittest.skipUnless(DEFAULT_POOL.exists(), "Local source pool required")
class TemporalExperimentTests(unittest.TestCase):
    def test_subsampling_keeps_exact_predictors_targets_and_split_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            for step, hours in [(12, [12, 24]), (24, [24])]:
                out = Path(directory) / str(step)
                spec = build(DEFAULT_POOL, out, step, hours)
                self.assertEqual(spec['resolution_minutes'], step * 60)
                source = read_frame(DEFAULT_POOL / 'hourly.csv')
                for split in ['train', 'val', 'test']:
                    frame = pd.read_csv(out / f'{split}.csv')
                    times = pd.DatetimeIndex(pd.to_datetime(frame.pop('timestamp_utc'), utc=True), name='timestamp_utc')
                    frame.index = times
                    pd.testing.assert_frame_equal(frame, source.loc[times, frame.columns])
                    self.assertTrue((times.tz_convert('Asia/Ho_Chi_Minh').hour % step == 0).all())
                with self.assertRaises(FileExistsError):
                    build(DEFAULT_POOL, out, step, hours)

    def test_four_real_configurations_forward_backward_without_target_leakage(self):
        cases = [
            ('1h', DEFAULT_POOL, 96, [12]),
            ('1h', DEFAULT_POOL, 96, [24]),
            ('12h', ROOT / 'train_data/da_vien_12h_2026-10-15_rain_pool_h12_h24', 9, [12, 24]),
            ('24h', ROOT / 'train_data/da_vien_24h_2026-10-15_rain_pool_h24', 5, [24]),
        ]
        torch.set_num_threads(2)
        paths = set()
        for resolution, directory, sequence, hours in cases:
            args = parser.parse_args([
                '--mlflow-ex-name', 'test-only', '--train-method', 'residual_lstm',
                '--horizon', resolution, '--dataset-dir', str(directory),
                '--rain-stations', 'hong_thai+nham', '--device', 'cpu',
                '--lstm-seq-len', str(sequence), '--batch-size', '8',
                '--target-horizons', *map(str, hours),
            ])
            validate_args(args)
            trainer = build_sequence_trainer(args)
            dataset = trainer.train_dataloader.dataset
            self.assertEqual(dataset.num_features, 26)
            self.assertEqual(dataset.num_targets, len(hours))
            self.assertFalse(any(c.startswith('target_') for c in dataset.feature_cols))
            x, y = next(iter(trainer.train_dataloader))
            pred = trainer.model(x)
            self.assertEqual(pred.shape, y.shape)
            loss = trainer.training_loss(pred, x, trainer.loss_func(pred, y))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in trainer.model.parameters() if p.grad is not None))
            fs = dataset.feature_scaler
            self.assertIs(fs, trainer.eval_dataloader.dataset.feature_scaler)
            self.assertIs(fs, trainer.test_dataloader.dataset.feature_scaler)
            paths.add(trainer.checkpoint_path)
        self.assertEqual(len(paths), 4)

    def test_invalid_or_conflicting_horizons_are_rejected(self):
        for horizons in [[12, 12], [0], [-1]]:
            with self.assertRaises(ValueError):
                get_dataloader('1h', 8, dataset_dir=DEFAULT_POOL, target_horizons=horizons)
        with self.assertRaises(ValueError):
            get_dataloader('1h', 8, dataset_dir=DEFAULT_POOL, outputs=5, target_horizons=[24])


if __name__ == '__main__':
    unittest.main()
