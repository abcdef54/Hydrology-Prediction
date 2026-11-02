"""Performance changes must preserve samples, shuffle order and model semantics."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, default_collate

from src.ai.regional_data.builder import build_dataset
from src.ai.regional_training_data import RegionalTrainingDataset, RegionalShuffleSampler, regional_loaders, regional_collate
from src.ai.residual_sequence import residual_model
from src.ai.trainers import ResidualLSTMTrainer
from src.ai.tests.test_regional_data import fixture


class PerformanceParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temp.name)/'data'
        level, rain, strict, weather, mapping, config, catalog = fixture()
        build_dataset(cls.path, level=level, rain_standard=rain, rain_strict=strict, weather=weather,
            stations=mapping, clusters=pd.DataFrame({'station_id': list(mapping)}), config=config,
            source_paths=[], rain_catalog=catalog)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def assert_reference(self, dataset, index, actual):
        row = dataset.view[index]
        stats = dataset.view.normalizers['stations'][row['station_id']]['targets']
        expected = dict(x=torch.from_numpy(row['x']), y=torch.from_numpy(row['y']),
            station_ids=torch.tensor(row['station_index']),
            target_scale=torch.tensor([stats[f'h{h}']['scale'] for h in [6, 12]], dtype=torch.float32),
            target_mean=torch.tensor([stats[f'h{h}']['mean'] for h in [6, 12]], dtype=torch.float32))
        for key in expected:
            self.assertTrue(torch.equal(actual[key], expected[key]), (index, key))

    def test_cached_and_batched_inputs_equal_reference_bit_for_bit(self):
        for weather in [False, True]:
            for time in ['none', 'all']:
                data = RegionalTrainingDataset(self.path, split='train', use_weather_forecast=weather,
                    time_features=time, rain_stations='nham+hong_thai')
                positions = list(range(0, len(data), 7))+[len(data)-1]
                for index in positions:
                    self.assert_reference(data, index, data[index])
                for index, sample in zip(positions, data.__getitems__(positions)):
                    self.assert_reference(data, index, sample)

    def test_shuffle_matches_previous_loader_across_epochs(self):
        data = list(range(97))
        previous = DataLoader(data, batch_size=16, shuffle=True, generator=torch.Generator().manual_seed(42))
        current = DataLoader(data, batch_size=16, sampler=RegionalShuffleSampler(data, 42),
                             generator=torch.Generator().manual_seed(42))
        for _ in range(3):
            self.assertEqual(torch.cat(list(previous)).tolist(), torch.cat(list(current)).tolist())

    def test_prebatched_collation_matches_rows_and_subset_with_partial_batch(self):
        data = RegionalTrainingDataset(self.path, split='train', time_features='all')
        positions = [len(data)-1, 3, 1, 3, 0]
        expected = default_collate([data[i] for i in positions])
        actual = regional_collate(data.__getitems__(positions))
        legacy = default_collate(data.__getitems__(positions))
        sliced = default_collate(data.__getitems__(positions)[1:3])
        for key in expected:
            self.assertTrue(torch.equal(actual[key], expected[key]), key)
            self.assertTrue(torch.equal(legacy[key], expected[key]), key)
            self.assertTrue(torch.equal(sliced[key], expected[key][1:3]), key)
        loader = DataLoader(Subset(data, positions), batch_size=3, collate_fn=regional_collate)
        batches = list(loader)
        self.assertEqual([len(b['y']) for b in batches], [3, 2])
        for key in expected:
            self.assertTrue(torch.equal(torch.cat([b[key] for b in batches]), expected[key]), key)

    def test_selection_replacement_refreshes_cached_metadata(self):
        data = RegionalTrainingDataset(self.path, split='train')
        data.view.samples = data.view.samples.iloc[[len(data)-1, 3, 1]].reset_index(drop=True)
        self.assertEqual(len(data), 3)
        self.assert_reference(data, 0, data[0])
        self.assert_reference(data, 1, data.__getitems__([1])[0])

    def test_invalid_station_mapping_fails_on_cpu(self):
        data = RegionalTrainingDataset(self.path, split='train')
        data.view.samples = data.view.samples.copy()
        data.view.samples.loc[0, 'station_index'] = 999
        with self.assertRaises(ValueError):
            data[0]

    def test_persistence_and_forward_trusted_path_equal_public_path(self):
        torch.manual_seed(42)
        model = residual_model('gru', input_size=4, hidden_size=3, num_layers=1, dropout=0.,
            num_stations=2, persistence_scale=[[1., 2.], [3., 4.]], persistence_offset=[[0., 0.], [1., 1.]])
        torch.nn.init.normal_(model.linear.weight)
        x, ids = torch.randn(2, 96, 4), torch.tensor([0, 1])
        torch.testing.assert_close(model(x, ids), model(x, ids, _validate_station_ids=False), rtol=0, atol=0)
        torch.testing.assert_close(model.persistence(x, ids), model.persistence(x, ids, _validate_station_ids=False), rtol=0, atol=0)
        with self.assertRaises(ValueError):
            model(x, torch.tensor([0, 2]))

    def test_zero_correction_penalty_skips_unneeded_persistence(self):
        trainer = object.__new__(ResidualLSTMTrainer)
        trainer.model = SimpleNamespace(persistence=lambda _: self.fail('Persistence must not run for zero penalty'))
        trainer.correction_penalty = 0.
        loss = torch.tensor(1., requires_grad=True)
        self.assertIs(trainer.training_loss(torch.ones(1), torch.ones(1), loss), loss)

    def test_worker_persistence_is_enabled_only_with_workers(self):
        with patch('src.ai.regional_training_data.torch.cuda.is_available', return_value=False):
            zero = regional_loaders(self.path, batch_size=8, num_workers=0, seed=42)
            two = regional_loaders(self.path, batch_size=8, num_workers=2, seed=42)
        self.assertTrue(all(not loader.persistent_workers for loader in zero))
        self.assertTrue(all(loader.persistent_workers for loader in two))
