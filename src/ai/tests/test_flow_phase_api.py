"""flow_features, tide_phase_features, rain_api_features (Dã Viên Task 20)."""
import unittest

import numpy as np
import pandas as pd

from src.ai.station_dataset.features import flow_features, rain_api_features, tide_phase_features

M2 = 12.4206012


class TidePhase(unittest.TestCase):
    def setUp(self):
        self.idx = pd.date_range('2026-01-01', periods=24 * 12, freq='h', tz='UTC')
        t = (self.idx - pd.Timestamp('2025-01-01', tz='UTC')).total_seconds().to_numpy() / 3600
        self.tide = lambda tt: 500 * np.cos(2 * np.pi * tt / M2)
        self.t = t
        self.f = tide_phase_features(pd.Series(1000 + self.tide(t), index=self.idx))

    def test_pure_m2(self):
        i = 24 * 9
        row = self.f.iloc[i]
        self.assertEqual(row.tph_missing_frac, 0.)
        lead = np.arange(15)
        curve = self.tide(self.t[i] + lead)
        self.assertEqual(row.tph_hw_hours, float(np.argmax(curve[1:14]) + 1))
        self.assertAlmostEqual(row.tph_hw_rise, curve[int(row.tph_hw_hours)] - curve[0], places=6)
        self.assertAlmostEqual(row.tph_curv6, curve[7] - 2 * curve[6] + curve[5], places=6)
        self.assertAlmostEqual(row.tph_range12, curve[:13].max() - curve[:13].min(), places=6)

    def test_first_week_missing(self):
        self.assertEqual(self.f.tph_missing_frac.iloc[10], 1.)
        self.assertTrue(np.isnan(self.f.tph_slope6.iloc[10]))


class RainApi(unittest.TestCase):
    def test_pulse_decay(self):
        ext = pd.date_range('2026-01-01', periods=200, freq='h', tz='UTC')
        idx = ext[100:]
        r = pd.DataFrame({'a': 0., 'b': 0.}, index=ext)
        r.loc[ext[120], 'a'] = 10.                                      # cluster x = mean(a, b) = 5 mm at hour 120
        f = rain_api_features(r, {'x': ['a', 'b']}, idx, taus=(12,), transform='none',
                              fc_next=pd.Series(1., index=idx))
        alpha = 1 - np.exp(-1 / 12)
        self.assertAlmostEqual(f.loc[ext[120], 'rain_api_basin_12'], 5 * alpha * 12)
        self.assertAlmostEqual(f.loc[ext[130], 'rain_api_basin_12'], 5 * alpha * 12 * np.exp(-10 / 12))
        self.assertAlmostEqual(f.loc[ext[119], 'rain_api_basin_12'], 0.)  # causal
        self.assertAlmostEqual(f.loc[ext[130], 'rain_api_basin_12_plusfc'], f.loc[ext[130], 'rain_api_basin_12'] + 1)
        self.assertEqual(f.rain_api_missing_frac.max(), 0.)


class Flow(unittest.TestCase):
    def test_same_tide_shifted_level(self):
        idx = pd.date_range('2026-01-01', periods=24 * 10, freq='h', tz='UTC')
        t = np.arange(len(idx))
        dv = pd.Series(800 + 300 * np.cos(2 * np.pi * t / M2), index=idx)
        other = dv - 60                                                 # same tide, 60 mm lower
        f = flow_features(dv, other, np.ones(len(idx), bool))
        late = f.iloc[24 * 8:]
        self.assertTrue(np.allclose(late.flow_grad, 60, atol=1e-6))
        self.assertTrue(np.allclose(late.flow_grad_d6, 0, atol=1e-6))
        self.assertTrue(np.allclose(late.flow_lp_grad, 60))
        self.assertEqual(late.flow_missing_frac.max(), 0.)


if __name__ == '__main__':
    unittest.main()
