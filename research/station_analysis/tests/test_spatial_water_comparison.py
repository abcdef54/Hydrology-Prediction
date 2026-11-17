"""Guard exact baseline pairing and signs in the human-readable comparison."""
import unittest
import numpy as np
import pandas as pd

from research.station_analysis.phase2.spatial_weather.water_comparison_reporting import (
    BASELINE, paired_water_comparisons,
)


class SpatialWaterComparisonTests(unittest.TestCase):
    def fixture(self):
        rows = []
        for scope in ['train', 'evaluation']:
            for horizon in [6, 12]:
                for grid in [BASELINE, 'new']:
                    value = (-.2 if horizon == 6 else .4) if grid == BASELINE else .1
                    rows.append(dict(water_station_id='001', scope=scope, horizon_h=horizon,
                                     feature='rain_6_12h_mm', condition='all', grid_id=grid,
                                     n=120, pearson_r=value, spearman_r=-value))
        return pd.DataFrame(rows)

    def test_pairs_same_horizon_scope_and_retains_signed_and_absolute_deltas(self):
        output = paired_water_comparisons(self.fixture())
        new = output[output.grid_id.eq('new') & output.scope.eq('train')].set_index('horizon_h')
        self.assertAlmostEqual(new.loc[6, 'delta_pearson_r'], .3)
        self.assertAlmostEqual(new.loc[6, 'delta_abs_pearson_r'], -.1)
        self.assertAlmostEqual(new.loc[12, 'delta_pearson_r'], -.3)
        self.assertAlmostEqual(new.loc[6, 'delta_spearman_r'], -.3)

    def test_rejects_misaligned_counts_missing_reference_and_duplicates(self):
        source = self.fixture()
        mismatch = source.copy(); mismatch.loc[0, 'n'] = 119
        for invalid in [mismatch, source[source.grid_id.ne(BASELINE)],
                        pd.concat([source, source.iloc[:1]], ignore_index=True)]:
            with self.assertRaises(ValueError):
                paired_water_comparisons(invalid)

    def test_no_training_samples_remain_undefined_not_zero_correlation(self):
        source = self.fixture()
        source.loc[source.scope.eq('train'), ['pearson_r', 'spearman_r']] = np.nan
        source.loc[source.scope.eq('train'), 'n'] = 0
        output = paired_water_comparisons(source)
        self.assertTrue(output.loc[output.scope.eq('train'), 'delta_pearson_r'].isna().all())
        self.assertTrue(output.loc[output.scope.eq('train'), 'n'].eq(0).all())
