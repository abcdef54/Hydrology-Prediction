"""SurgeScaleAugmenter on a real Dã Viên dataset (skipped when train_data/da_vien/dv_h36_all is absent). The flow_*
channel checks run only on a dataset that has the flow group."""
import unittest
from pathlib import Path

import numpy as np
import torch

from src.ai.augmentation import SurgeScaleAugmenter, parse_surge_scale
from src.ai.regional_training_data import RegionalTrainingDataset, regional_collate, regional_loaders

DS = Path('train_data/da_vien/dv_h36_all')
GROUPS = 'wind+tide+fc_ecmwf+sea+station_obs'
KW = dict(use_weather_forecast=False, time_features='none', target_horizons=[3, 6], sequence_length=25)


@unittest.skipUnless((DS / 'samples.parquet').is_file(), 'dataset not built')
class SurgeScale(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = RegionalTrainingDataset(DS, split='train', rain_stations=GROUPS, **KW)
        cls.pos = np.arange(0, 5000, 37)
        cls.batch = cls.ds.__getitems__(cls.pos.tolist())
        cls.k = 1.3
        cls.aug = SurgeScaleAugmenter(cls.ds, cls.k, cls.k, 1.0, seed=0)
        cls.out = cls.aug(cls.batch)
        v = cls.ds.view
        cls.sid = next(iter(v.mapping))
        cls.w = v.normalizers['stations'][cls.sid]['water']
        cls.sh = v.normalizers['shared']
        cls.cols = cls.ds.feature_cols

    def mm(self, x, col):
        s = self.sh[col]
        return x[..., self.cols.index(col)].numpy() * s['scale'] + s['mean']

    def test_level_and_target_residual_scaled(self):
        x0, x1 = self.batch.tensors['x'], self.out['x']
        lvl0 = x0[..., 0].numpy() * self.w['scale'] + self.w['mean']
        lvl1 = x1[..., 0].numpy() * self.w['scale'] + self.w['mean']
        present = x0[..., 1].numpy() == 0
        tide = self.mm(x0, 'tide_now')
        r0, r1 = lvl0 - tide, lvl1 - tide
        rbar = np.array([np.nanmean(np.where(p, r, np.nan)) for r, p in zip(r0, present)])
        np.testing.assert_allclose(r1[present], (rbar[:, None] + self.k * (r0 - rbar[:, None]))[present], atol=0.05)
        # targets: residual at t+h (tide_now of the target hour on the grid) scaled the same way
        tide_grid = self.ds.view.rain['tide_now'].to_numpy()
        anchor = self.ds._ends[self.pos] - 1
        for j, h in enumerate([3, 6]):
            sc, mu = self.out['target_scale'][:, j].numpy(), self.out['target_mean'][:, j].numpy()
            y0 = self.batch.tensors['y'][:, j].numpy() * sc + mu
            y1 = self.out['y'][:, j].numpy() * sc + mu
            rt0, rt1 = y0 - tide_grid[anchor + h], y1 - tide_grid[anchor + h]
            np.testing.assert_allclose(rt1, rbar + self.k * (rt0 - rbar), atol=0.05)

    def test_neighbour_and_flow_channels(self):
        x0, x1 = self.batch.tensors['x'], self.out['x']
        np.testing.assert_allclose(self.mm(x1, 'obs_nbs_minus_self'), self.mm(x0, 'obs_nbs_minus_self'), atol=1e-3)
        if 'flow_grad' in self.cols:
            np.testing.assert_allclose(self.mm(x1, 'flow_grad'), self.mm(x0, 'flow_grad'), atol=1e-3)
            lvl_shift = (x1[..., 0] - x0[..., 0]).numpy() * self.w['scale']
            present = (x0[..., 1].numpy() == 0) & np.isfinite(self.ds.view.rain['flow_dvres'].to_numpy()[
                self.ds._starts[self.pos][:, None] + self.ds._history_offsets])
            np.testing.assert_allclose((self.mm(x1, 'flow_dvres') - self.mm(x0, 'flow_dvres'))[present], lvl_shift[present], atol=0.05)
        # neighbour change over 3 h moves by a(t) - a(t-3), with a = level shift of the station
        rows = self.ds._starts[self.pos][:, None] + self.ds._history_offsets
        R = self.aug.R
        a_diff = (self.k - 1) * (R[rows] - R[rows - 3])
        ok = np.isfinite(self.ds.view.rain['obs_nbs_d3'].to_numpy()[rows])
        np.testing.assert_allclose((self.mm(x1, 'obs_nbs_d3') - self.mm(x0, 'obs_nbs_d3'))[ok], a_diff[ok], atol=0.05)
        # exogenous channels untouched
        for c in ('tide_now', 'sea_anom', 'fc_basin_rain_0_3h', 'wind_onshore_next6'):
            i = self.cols.index(c)
            self.assertTrue(torch.equal(x1[..., i], x0[..., i]), c)

    def test_identity_cases(self):
        for k, p in ((1.0, 1.0), (1.4, 0.0)):
            out = SurgeScaleAugmenter(self.ds, k, k, p, seed=0)(self.batch)
            self.assertTrue(torch.allclose(out['x'], self.batch.tensors['x'], atol=1e-6))
            self.assertTrue(torch.allclose(out['y'], self.batch.tensors['y'], atol=1e-6))

    def test_original_batch_not_modified_and_random_per_call(self):
        aug = SurgeScaleAugmenter(self.ds, 0.8, 1.3, 0.5, seed=1)
        before = self.batch.tensors['x'].clone()
        a, b = aug(self.batch), aug(self.batch)
        self.assertTrue(torch.equal(self.batch.tensors['x'], before))
        self.assertFalse(torch.equal(a['y'], b['y']))

    def test_refuses_linear_channels_and_bad_args(self):
        ds = RegionalTrainingDataset(DS, split='train', rain_stations=GROUPS + '+lin_fc', **KW)
        with self.assertRaises(ValueError):
            SurgeScaleAugmenter(ds, 0.8, 1.2, 0.5, seed=0)
        with self.assertRaises(ValueError):
            parse_surge_scale('1.2:0.8:0.5')
        self.assertEqual(parse_surge_scale('0.8:1.3:0.5'), (0.8, 1.3, 0.5))

    def test_only_train_loader_augmented(self):
        tr, va, te = regional_loaders(DS, batch_size=64, num_workers=0, seed=23, rain_stations=GROUPS,
                                      augment_surge_scale=(0.8, 1.3, 0.5), **KW)
        self.assertIsInstance(tr.collate_fn, SurgeScaleAugmenter)
        self.assertIs(va.collate_fn, regional_collate)
        self.assertIs(te.collate_fn, regional_collate)
        batch = next(iter(tr))
        self.assertEqual(tuple(batch['x'].shape[1:]), (25, len(self.cols)))


if __name__ == '__main__':
    unittest.main()
