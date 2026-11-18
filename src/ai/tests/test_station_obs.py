"""station_obs group: neighbour-level features and datum offsets on raw readings."""
import unittest

import numpy as np
import pandas as pd

from src.ai.station_dataset.build import apply_offsets, hourly_level
from src.ai.station_dataset.features import station_obs_features


class StationObs(unittest.TestCase):
    def test_features(self):
        idx = pd.date_range('2026-01-01', periods=30, freq='h', tz='UTC')
        own = pd.Series(np.arange(30.) * 2, index=idx)
        other = pd.Series(100 + np.arange(30.) * 5, index=idx)
        other.iloc[10] = np.nan
        f = station_obs_features({'nbs': other}, own, lags_hours=3)
        self.assertEqual(list(f.columns), ['obs_nbs_minus_self', 'obs_nbs_d1', 'obs_nbs_d2', 'obs_nbs_d3',
                                           'obs_nbs_missing_frac'])
        self.assertEqual(f.obs_nbs_minus_self.iloc[5], 100 + 25 - 10)
        self.assertEqual(f.obs_nbs_d3.iloc[5], 15)
        self.assertTrue(np.isnan(f.obs_nbs_d1.iloc[0]))                   # no history before the grid
        self.assertEqual(f.obs_nbs_missing_frac.iloc[11], 1.)             # t-1 missing
        self.assertEqual(f.obs_nbs_missing_frac.iloc[20], 0.)

    def test_offsets_on_raw_readings(self):
        t = pd.date_range('2026-10-10 14:10', '2026-10-10 17:00', freq='10min', tz='UTC')
        depth = np.where(t >= pd.Timestamp('2026-10-10 15:20Z'), 343., 500.)    # datum drop of 157 mm at 15:20
        obs = pd.DataFrame({'observed_at': t, 'depth': depth, 'quality_status': 'observed'})
        fixed = apply_offsets(obs, [{'from_utc': '2026-10-10T15:20:00+00:00', 'add_mm': 157}])
        idx = pd.date_range('2026-10-10 15:00', '2026-10-10 17:00', freq='h', tz='UTC')
        self.assertTrue(np.allclose(hourly_level(fixed, idx).to_numpy(), 500.))   # incl. the mixed 16:00 hour
        self.assertFalse(np.allclose(hourly_level(obs, idx).to_numpy(), 500.))
        self.assertTrue(np.allclose(obs.depth.to_numpy(), depth))                  # input not modified


if __name__ == '__main__':
    unittest.main()
