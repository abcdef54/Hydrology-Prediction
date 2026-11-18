"""self_forecast_features: channel k at t = forecast issued at t-L for t+k minus level(t)."""
import unittest

import numpy as np
import pandas as pd

from src.ai.station_dataset.features import self_forecast_features


class SelfForecast(unittest.TestCase):
    def test_alignment(self):
        idx = pd.date_range('2026-01-01', periods=10, freq='h', tz='UTC')
        level = pd.Series(np.arange(10) * 10., index=idx)
        fc = pd.DataFrame({f'p{h}': 1000. * h + np.arange(10) for h in (1, 2, 3)}, index=idx)
        f = self_forecast_features(level, fc, issue_lag_hours=1)
        self.assertEqual(list(f.columns), ['selffc_t0', 'selffc_t1', 'selffc_t2', 'selffc_missing_frac'])
        t = 5   # issued at t-1 = 4: p1 = 1004 (for t), p3 = 3004 (for t+2); level(t) = 50
        self.assertEqual(f.selffc_t0.iloc[t], 1004 - 50)
        self.assertEqual(f.selffc_t2.iloc[t], 3004 - 50)
        self.assertEqual(f.selffc_missing_frac.iloc[0], 1.)        # no issue at t-1
        self.assertEqual(f.selffc_missing_frac.iloc[1], 0.)


if __name__ == '__main__':
    unittest.main()
