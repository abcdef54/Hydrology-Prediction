"""spring_neap_features against known moon phases (UTC)."""
import unittest

import pandas as pd

from src.ai.station_dataset.features import spring_neap_features


class SpringNeap(unittest.TestCase):
    def at(self, *times):
        return spring_neap_features(pd.DatetimeIndex(pd.to_datetime(list(times), utc=True)))

    def test_new_full_and_quarter_moons(self):
        # new moon 2024-01-11 11:57, full moon 2024-01-25 17:54, first quarter 2024-01-18 03:52, last quarter 2024-02-02 23:18
        f = self.at('2024-01-11 11:57', '2024-01-25 17:54', '2024-01-18 03:52', '2024-02-02 23:18')
        self.assertTrue((f.sn_cos.iloc[:2] > 0.9).all(), f.sn_cos.tolist())      # springs
        self.assertTrue((f.sn_cos.iloc[2:] < -0.9).all(), f.sn_cos.tolist())     # neaps

    def test_periods_and_bounds(self):
        idx = pd.date_range('2026-01-01', periods=24 * 60, freq='h', tz='UTC')
        f = spring_neap_features(idx)
        self.assertTrue(((f[['sn_cos', 'sn_sin', 'sn_decl_cos', 'sn_decl_sin']].abs()) <= 1).all().all())
        self.assertEqual(f.sn_missing_frac.max(), 0.)
        # value repeats after one spring-neap period (14.765 d) and one half tropical month (13.661 d)
        a = spring_neap_features(pd.DatetimeIndex([pd.Timestamp('2026-03-01', tz='UTC')]))
        b = spring_neap_features(pd.DatetimeIndex([pd.Timestamp('2026-03-01', tz='UTC') + pd.Timedelta(days=14.765294)]))
        c = spring_neap_features(pd.DatetimeIndex([pd.Timestamp('2026-03-01', tz='UTC') + pd.Timedelta(days=13.660790)]))
        self.assertAlmostEqual(a.sn_cos.iloc[0], b.sn_cos.iloc[0], places=3)
        self.assertAlmostEqual(a.sn_decl_cos.iloc[0], c.sn_decl_cos.iloc[0], places=3)


if __name__ == '__main__':
    unittest.main()
