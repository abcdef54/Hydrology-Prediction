"""Unit tests for Phase 2 Stage 2: Rainfall EDA & Relationship Analysis."""
import unittest
from datetime import datetime, timezone
import numpy as np
import pandas as pd

from research.station_analysis.phase2.rainfall_preprocessing import (
    preprocess_hourly_rainfall,
    preprocess_canonical_water_levels,
    compute_multi_scale_accumulations,
    calculate_station_distance_matrix,
    WINDOWS
)
from research.station_analysis.phase2.station_selection import (
    evaluate_single_station_candidates,
    build_exclusion_list,
    evaluate_multistation_combinations
)


class TestPhase2Stage2(unittest.TestCase):

    def setUp(self):
        self.anchors = pd.date_range('2026-01-01 01:00:00+00:00', periods=48, freq='1h')
        self.rain_stations = pd.DataFrame([
            {'station_id': 'R1', 'name': 'Rain Station 1', 'latitude': 16.5, 'longitude': 107.5},
            {'station_id': 'R2', 'name': 'Rain Station 2', 'latitude': 16.2, 'longitude': 107.8},
        ])
        self.water_stations = pd.DataFrame([
            {'station_id': 'W1', 'name': 'Water Station 1', 'latitude': 16.45, 'longitude': 107.55},
        ])

    def test_rainfall_semantics_and_missing_policy(self):
        # Construct synthetic 10-minute observations
        obs_records = []
        # Hour 01:00 for R1 has 6/6 observations with 2.0 mm each -> sum = 12.0 mm (complete)
        for m in (10, 20, 30, 40, 50, 60):
            obs_records.append({
                'station_id': 'R1',
                'observed_at': pd.Timestamp('2026-01-01 00:00:00+00:00') + pd.Timedelta(minutes=m),
                'depth': 2.0,
                'quality_status': 'observed'
            })
        # Hour 02:00 for R1 has 5/6 observations with 2.0 mm each -> sum = 10.0 * 1.2 = 12.0 mm (estimated)
        for m in (10, 20, 30, 40, 50):
            obs_records.append({
                'station_id': 'R1',
                'observed_at': pd.Timestamp('2026-01-01 01:00:00+00:00') + pd.Timedelta(minutes=m),
                'depth': 2.0,
                'quality_status': 'observed'
            })
        # Hour 03:00 for R1 has 4/6 observations with 2.0 mm each -> missing (NaN)
        for m in (10, 20, 30, 40):
            obs_records.append({
                'station_id': 'R1',
                'observed_at': pd.Timestamp('2026-01-01 02:00:00+00:00') + pd.Timedelta(minutes=m),
                'depth': 2.0,
                'quality_status': 'observed'
            })

        df_obs = pd.DataFrame(obs_records)
        p_std, p_str, p_q, records = preprocess_hourly_rainfall(df_obs, self.rain_stations, self.anchors[:3])

        # Hour 01:00
        self.assertAlmostEqual(p_std.loc[self.anchors[0], 'R1'], 12.0)
        self.assertAlmostEqual(p_str.loc[self.anchors[0], 'R1'], 12.0)
        self.assertEqual(p_q.loc[self.anchors[0], 'R1'], 'complete')

        # Hour 02:00 (5/6 estimated in standard, NaN in strict)
        self.assertAlmostEqual(p_std.loc[self.anchors[1], 'R1'], 12.0)
        self.assertTrue(np.isnan(p_str.loc[self.anchors[1], 'R1']))
        self.assertEqual(p_q.loc[self.anchors[1], 'R1'], 'estimated')

        # Hour 03:00 (<5 missing in both)
        self.assertTrue(np.isnan(p_std.loc[self.anchors[2], 'R1']))
        self.assertTrue(np.isnan(p_str.loc[self.anchors[2], 'R1']))
        self.assertEqual(p_q.loc[self.anchors[2], 'R1'], 'missing')

    def test_multiscale_accumulations_min_periods(self):
        # Create series with 1 missing hour
        s = pd.Series(5.0, index=self.anchors)
        s.iloc[10] = np.nan
        df_rain = pd.DataFrame({'R1': s})

        acc_std, acc_str = compute_multi_scale_accumulations(df_rain, df_rain, windows=(1, 3, 6))

        # At anchor 9: window 3 has 3 valid hours -> 15.0
        self.assertAlmostEqual(acc_std[3].loc[self.anchors[9], 'R1'], 15.0)

        # At anchor 10: missing hour -> window 3 is NaN
        self.assertTrue(np.isnan(acc_std[3].loc[self.anchors[10], 'R1']))

        # At anchor 11: contains missing hour 10 -> window 3 is NaN due to min_periods=3
        self.assertTrue(np.isnan(acc_std[3].loc[self.anchors[11], 'R1']))

        # At anchor 13: hours 11, 12, 13 are valid -> window 3 is 15.0
        self.assertAlmostEqual(acc_std[3].loc[self.anchors[13], 'R1'], 15.0)

    def test_canonical_water_targets(self):
        avail_records = []
        for i, t in enumerate(self.anchors):
            avail_records.append({
                'station_id': 'W1',
                'timestamp_utc': t,
                'water_mean_mm': float(1000 + i * 10)
            })
        df_avail = pd.DataFrame(avail_records)
        pivot_h, targets = preprocess_canonical_water_levels(df_avail, self.water_stations, self.anchors)

        t0 = self.anchors[0]
        # H(t0) = 1000
        # H(t0 + 12h) = 1000 + 12 * 10 = 1120
        # dh_p12h = 1120 - 1000 = 120
        self.assertAlmostEqual(targets['W1'].loc[t0, 'water_level_mm'], 1000.0)
        self.assertAlmostEqual(targets['W1'].loc[t0, 'dh_p12h'], 120.0)
        self.assertAlmostEqual(targets['W1'].loc[t0, 'dh_p6h'], 60.0)
        self.assertAlmostEqual(targets['W1'].loc[t0, 'dh_p1h'], 10.0)

    def test_planar_distance_matrix(self):
        dist_df = calculate_station_distance_matrix(self.rain_stations, self.water_stations)
        self.assertEqual(dist_df.shape, (1, 2))
        self.assertEqual(dist_df.index[0], 'W1')
        self.assertTrue(dist_df.loc['W1', 'R1'] > 0.0)
        self.assertTrue(dist_df.loc['W1', 'R2'] > dist_df.loc['W1', 'R1'])

    def test_training_period_isolation_and_exclusion(self):
        # Ensure station with high missingness gets flagged in exclusion list
        df_p = pd.DataFrame(index=self.anchors)
        df_p['R1'] = 10.0
        df_p['R2'] = np.nan  # completely missing
        excl = build_exclusion_list(df_p, self.rain_stations, train_end_utc='2026-01-02T00:00:00+00:00')
        self.assertIn('R2', excl['rain_station_id'].tolist())


if __name__ == '__main__':
    unittest.main()
