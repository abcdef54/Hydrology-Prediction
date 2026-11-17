"""Regression checks for Stage 2 cutoff, shared anchors and data-driven reports."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from research.station_analysis.phase2.rainfall_preprocessing import preprocess_canonical_water_levels
from research.station_analysis.phase2.station_selection import (
    evaluate_multistation_combinations, evaluate_single_station_candidates, _pairwise_correlation,
)
from research.station_analysis.phase2.water_relationship import compare_policy_sensitivity
from research.station_analysis.phase2.stage2_report import markdown_table


class Stage2FinalizationTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range('2026-07-20', periods=500, freq='h', tz='UTC')
        rng = np.random.default_rng(42)
        self.rain = pd.DataFrame({'R1': rng.random(500), 'R2': rng.random(500)}, index=self.index)
        self.water = pd.DataFrame([{'station_id': 'W1', 'name': 'Water'}])
        self.rmeta = pd.DataFrame([{'station_id': 'R1', 'name': 'Rain 1'},
                                   {'station_id': 'R2', 'name': 'Rain 2'}])
        self.level = pd.Series(rng.normal(size=500).cumsum(), index=self.index)
        self.cut = pd.Timestamp('2026-08-01T00:00:00+07:00')

    def targets(self, level):
        return {'W1': pd.DataFrame({'water_level_mm': level,
                                   'dh_p6h': level.shift(-6)-level,
                                   'dh_p12h': level.shift(-12)-level})}

    def candidates(self, targets):
        return evaluate_single_station_candidates(
            {1: self.rain}, targets, self.water, self.rmeta,
            pd.DataFrame([[1., 2.]], index=['W1'], columns=['R1', 'R2']),
            pd.DataFrame(), train_end_utc=str(self.cut), windows=(1,), lags=(0, 2))

    def test_pairwise_correlations_match_pandas_with_missing_and_constant_data(self):
        frame = self.rain.copy()
        frame.iloc[5:15, 0] = np.nan
        frame['constant'] = 1.
        frame['absent'] = np.nan
        target = self.level.copy()
        target.iloc[15:25] = np.nan
        with np.errstate(divide='ignore', invalid='ignore'):
            expected = frame.corrwith(target)
        pd.testing.assert_series_equal(_pairwise_correlation(frame, target), expected)

    def test_future_water_cannot_change_train_selection(self):
        before = self.candidates(self.targets(self.level))
        changed = self.level.copy()
        changed.loc[changed.index > self.cut] += np.arange((changed.index > self.cut).sum())*10000
        after = self.candidates(self.targets(changed))
        pd.testing.assert_frame_equal(before, after)

    def test_same_anchors_purge_and_train_only_collinearity(self):
        rain = self.rain.copy()
        rain.loc[self.index[[30, 40, 400]], 'R2'] = np.nan
        subsets = {'one': ['R1'], 'two': ['R1', 'R2']}
        def evaluate(frame):
            return evaluate_multistation_combinations(
                {6: frame}, self.targets(self.level), self.water, subsets,
                water_station_id='W1', train_end_utc=str(self.cut))
        result = evaluate(rain).set_index('subset_name')
        self.assertEqual(result.train_n.nunique(), 1)
        self.assertEqual(result.evaluation_n.nunique(), 1)
        self.assertTrue((pd.to_datetime(result.last_train_target_utc, utc=True) <= self.cut).all())
        mask = ((rain.index + pd.Timedelta(hours=12)) <= self.cut) & rain.notna().all(axis=1)
        expected = round(rain.loc[mask].corr().loc['R1', 'R2'], 3)
        self.assertEqual(result.loc['two', 'max_collinearity'], expected)
        rain.loc[rain.index > self.cut, 'R2'] *= -10000
        again = evaluate(rain).set_index('subset_name')
        pd.testing.assert_series_equal(result.max_collinearity, again.max_collinearity)
        pd.testing.assert_series_equal(result.train_pearson_r, again.train_pearson_r)

    def test_missing_policy_separates_value_from_sample_effect(self):
        strict = self.rain.copy()
        strict.iloc[10, 0] = np.nan
        result = compare_policy_sensitivity(
            {6: self.rain}, {6: strict}, self.targets(self.level), [('W1', 'R1', 6, 0)])
        self.assertEqual(result.iloc[0].sample_gain, 1)
        self.assertAlmostEqual(result.iloc[0].common_anchor_r_difference, 0.)

    def test_markdown_preserves_station_id_and_current_values(self):
        result = markdown_table(pd.DataFrame({'station_id': ['0074647403'], 'r': [.123456]}))
        self.assertIn('0074647403', result)
        self.assertIn('0.1235', result)


if __name__ == '__main__':
    unittest.main()
