"""reservoir_features: only records published by t are used; the latest observed among them wins."""
import unittest

import numpy as np
import pandas as pd

from src.ai.station_dataset.features import reservoir_features, station_obs_features


def ts(s):
    return pd.Timestamp(s, tz='UTC')


class ReservoirFeatures(unittest.TestCase):
    def setUp(self):
        self.ops = pd.DataFrame({
            'facility_id': ['f'] * 4,
            'observed_at': [ts('2026-01-01 00:00'), ts('2026-01-01 06:00'), ts('2026-01-01 03:00'), ts('2026-01-01 12:00')],
            'source_updated_at': [ts('2026-01-01 00:10'), ts('2026-01-01 06:00'), ts('2026-01-01 09:00'), ts('2026-01-02 00:00')],
            'qdi': [100., 200., 999., 50.]})
        self.idx = pd.date_range('2026-01-01 00:00', '2026-01-02 00:00', freq='h', tz='UTC')
        self.f = reservoir_features(self.ops, self.idx, {'r': 'f'}, ['qdi'], change_hours=(6,))

    def v(self, t, col):
        return self.f.loc[ts(t), f'res_r_{col}']

    def test_publication_rule(self):
        self.assertTrue(np.isnan(self.v('2026-01-01 00:00', 'qdi')))                 # published 00:10
        self.assertAlmostEqual(self.v('2026-01-01 01:00', 'qdi'), np.log1p(100))
        self.assertAlmostEqual(self.v('2026-01-01 06:00', 'qdi'), np.log1p(200))     # 06:00 record published at 06:00
        # 03:00 record published at 09:00 is older than the 06:00 one -> never replaces it
        self.assertAlmostEqual(self.v('2026-01-01 10:00', 'qdi'), np.log1p(200))
        self.assertAlmostEqual(self.v('2026-01-01 23:00', 'qdi'), np.log1p(200))     # 12:00 record only published next day
        self.assertAlmostEqual(self.v('2026-01-02 00:00', 'qdi'), np.log1p(50))
        self.assertAlmostEqual(self.v('2026-01-01 10:00', 'age_h'), 4.)
        self.assertAlmostEqual(self.v('2026-01-01 08:00', 'qdi_d6'), np.log1p(200) - np.log1p(100))
        self.assertEqual(self.v('2026-01-01 00:00', 'missing_frac'), 1.)

    def test_station_obs_surge(self):
        idx = pd.date_range('2026-01-01', periods=24 * 20, freq='h', tz='UTC')
        t = np.arange(len(idx))
        other = pd.Series(500 + 300 * np.sin(2 * np.pi * t / 12.42), index=idx)
        own = pd.Series(np.zeros(len(idx)), index=idx)
        f = station_obs_features({'ta': other}, own, 3, surge=['ta'], train_mask=np.ones(len(idx), bool))
        self.assertIn('obs_ta_surge', f)
        self.assertIn('obs_ta_tide_d6', f)
        late = f.iloc[24 * 10:]                                  # after a full 7-day window: pure tide -> surge ~ mean level
        self.assertLess(late.obs_ta_surge.std(), 5.)
        with self.assertRaises(ValueError):
            station_obs_features({'ta': other}, own, 3, surge=['ta'])


if __name__ == '__main__':
    unittest.main()
