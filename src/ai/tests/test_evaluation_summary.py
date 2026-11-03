import unittest
import numpy as np
import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from src.ai.evaluation_summary import station_error_table


class StationSummaryTests(unittest.TestCase):
    def test_persistence_and_error_counts_for_all_horizons(self):
        observed = np.array([[0., 300.], [200., 200.]])
        predicted = np.array([[201., 0.], [0., 200.]])
        table = station_error_table(observed, predicted, np.array(['a', 'a']), {'a': {'name': 'A'}},
                                    ['target_water_level_plus_60m', 'target_water_level_plus_120m'],
                                    current_levels=np.array([100., 200.]))
        self.assertEqual(table.loc['a', 'n200_h1'], 1)
        self.assertEqual(table.loc['a', 'n200_h2'], 1)
        self.assertEqual(table.loc['a', 'n200_all'], 2)
        self.assertAlmostEqual(table.loc['a', 'persistence_mae_h1_mm'], 50.)
        self.assertAlmostEqual(table.loc['a', 'persistence_mae_h2_mm'], 100.)

    def test_macro_mean_gives_equal_station_weight_with_unequal_counts(self):
        actual = np.zeros((3, 2))
        predicted = np.array([[3., 4.], [0., 0.], [0., 0.]])
        result = station_error_table(actual, predicted, ['a', 'b', 'b'], {'a': {'name': 'Station A'}},
            ['target_water_level_plus_360m', 'target_water_level_plus_720m'])
        self.assertEqual(result.loc['a', 'name'], 'Station A')
        self.assertEqual(result.loc['a', 'rmse_h6_mm'], 3.)
        self.assertEqual(result.loc['a', 'mae_h12_mm'], 4.)
        self.assertEqual(result.loc['mean_across_stations', 'rmse_h6_mm'], 1.5)
        self.assertEqual(result.loc['mean_across_stations', 'mae_h12_mm'], 2.)
        self.assertAlmostEqual(result.loc['mean_across_stations', 'rmse_all_mm'], np.sqrt(12.5)/2)
        self.assertNotAlmostEqual(result.loc['mean_across_stations', 'rmse_h6_mm'], np.sqrt(3))

    def test_misaligned_or_nonfinite_values_are_rejected(self):
        for predicted in [np.ones((2, 1)), np.array([[np.nan, 1.]])]:
            with self.assertRaises(ValueError):
                station_error_table(np.ones((1, 2)), predicted, ['a'], {}, ['x', 'y'])

    def test_test_output_uses_names_and_saves_macro_metrics(self):
        from src.ai.trainers import LSTMTrainer
        class Samples(Dataset):
            regional_metadata = {'station_mapping': {'UNREADABLE_001': {'name': 'Station A'}, 'UNREADABLE_002': {'name': 'Station B'}}}
            target_cols = ['target_water_level_plus_360m', 'target_water_level_plus_720m']
            timestamps = pd.Series(pd.date_range('2026-09-15', periods=3, freq='h', tz='UTC'))
            view = SimpleNamespace(samples=pd.DataFrame({'station_id': ['UNREADABLE_001', 'UNREADABLE_002', 'UNREADABLE_002']}))
            def __len__(self): return 3
            def __getitem__(self, i):
                return dict(x=torch.tensor([[[3., 4.], [0., 0.], [0., 0.]][i]]), y=torch.zeros(2),
                    station_ids=torch.tensor(0 if i == 0 else 1), target_scale=torch.ones(2), target_mean=torch.zeros(2))
        class Echo(torch.nn.Module):
            num_stations = 2
            def forward(self, x, station_ids): return x[:, -1]
        trainer = object.__new__(LSTMTrainer)
        trainer.model, trainer.device = Echo(), torch.device('cpu')
        trainer.test_dataloader = DataLoader(Samples(), batch_size=2)
        trainer.validation_threshold_mm = 200.
        def simple_metrics(actual, pred, **kwargs):
            error = pred-actual
            return {'rmse': float(np.sqrt(np.mean(error**2))), 'mae': float(np.mean(abs(error)))}
        with tempfile.TemporaryDirectory() as tmp:
            trainer.checkpoint_path = Path(tmp)/'model.pt'
            output = io.StringIO()
            with patch('src.ai.trainers.mlflow.log_metric'), patch('src.ai.trainers.mlflow.set_tag'), \
                    patch('src.ai.trainers.evaluate_all', side_effect=simple_metrics), redirect_stdout(output):
                result = trainer.test()
            self.assertIn('Station A', output.getvalue())
            self.assertIn('Mean across stations', output.getvalue())
            self.assertNotIn('UNREADABLE_001', output.getvalue())
            self.assertEqual(result.loc['mean_across_stations/target_water_level_plus_360m', 'rmse'], 1.5)
            self.assertTrue(trainer.checkpoint_path.with_suffix('.test_station_metrics.csv').exists())
