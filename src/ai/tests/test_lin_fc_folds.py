"""lin_fc folds: 0 = one full-train fit on every hour; >= 2 = out-of-fold on train hours (unchanged)."""
import unittest

import numpy as np
import pandas as pd

from src.ai.station_dataset.features import linear_forecast_features


def data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2026-01-01', periods=n, freq='h', tz='UTC')
    t = np.arange(n)
    level = pd.Series(500 + 200 * np.sin(2 * np.pi * t / 12.42) + rng.normal(0, 10, n), index=idx)
    block = pd.DataFrame({'x': rng.normal(size=n), 'x_missing_frac': 0.}, index=idx)
    return level, block, np.arange(20, 400)


class LinFcFolds(unittest.TestCase):
    def test_folds0_is_one_fit_everywhere(self):
        level, block, train = data()
        full = linear_forecast_features(level, [block], train, horizons=(3,), folds=0)
        oof = linear_forecast_features(level, [block], train, horizons=(3,), folds=5)
        later = slice(400, None)                                  # after the last train anchor: identical
        self.assertTrue(np.allclose(full.lin_d3.iloc[later], oof.lin_d3.iloc[later], equal_nan=True))
        self.assertFalse(np.allclose(full.lin_d3.iloc[train], oof.lin_d3.iloc[train]))   # train hours differ
        # folds=0: train hours come from the same coefficients -> one linear map of the inputs everywhere
        y = level.shift(-3) - level
        e_full = (full.lin_d3 - y).iloc[train].abs().mean()
        e_oof = (oof.lin_d3 - y).iloc[train].abs().mean()
        self.assertLessEqual(e_full, e_oof)                        # in-sample fit is never worse on its own rows

    def test_invalid_folds(self):
        level, block, train = data(200)
        for f in (1, -1, 2.5, True):
            with self.assertRaises(ValueError):
                linear_forecast_features(level, [block], np.arange(20, 150), horizons=(3,), folds=f)


if __name__ == '__main__':
    unittest.main()
