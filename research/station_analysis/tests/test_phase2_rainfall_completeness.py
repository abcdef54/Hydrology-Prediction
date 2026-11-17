"""Unit tests for Phase 2 rainfall observation completeness module."""
import unittest
from datetime import datetime, timezone
import pandas as pd
import numpy as np

from research.station_analysis.phase2.rainfall_completeness import (
    perform_integrity_checks,
    build_hourly_completeness_grid,
    calculate_station_metrics,
    calculate_system_summary,
    calculate_monthly_metrics,
)


class TestRainfallCompleteness(unittest.TestCase):
    def setUp(self):
        # Create synthetic rainfall stations
        self.stations = pd.DataFrame([
            {
                'station_id': 'R1',
                'name': 'Trạm R1',
                'station_type': 'rainfall',
                'latitude': 16.2,
                'longitude': 107.5,
                'measurement_unit': 'mm'
            },
            {
                'station_id': 'R2',
                'name': 'Trạm R2',
                'station_type': 'rainfall',
                'latitude': 16.4,
                'longitude': 107.6,
                'measurement_unit': 'mm'
            }
        ])
        
        # 3 hourly anchors: 01:00, 02:00, 03:00 UTC
        self.anchors = pd.date_range('2026-01-01 01:00:00+00:00', '2026-01-01 03:00:00+00:00', freq='h', tz='UTC')
        
        # Create synthetic observations:
        # For R1:
        # Hour 01:00 -> exactly 6 observations at 00:10, 00:20, 00:30, 00:40, 00:50, 01:00 (full 6/6)
        # Hour 02:00 -> 5 observations (missing 01:30) (5/6)
        # Hour 03:00 -> 0 observations (0/6)
        r1_times_h1 = [
            '2026-01-01 00:10:00+00:00', '2026-01-01 00:20:00+00:00', '2026-01-01 00:30:00+00:00',
            '2026-01-01 00:40:00+00:00', '2026-01-01 00:50:00+00:00', '2026-01-01 01:00:00+00:00'
        ]
        r1_times_h2 = [
            '2026-01-01 01:10:00+00:00', '2026-01-01 01:20:00+00:00',
            '2026-01-01 01:40:00+00:00', '2026-01-01 01:50:00+00:00', '2026-01-01 02:00:00+00:00'
        ]
        
        # For R2:
        # Starts late: first observation in Hour 02:00 -> 6 observations in hour 02:00
        # Hour 03:00 -> 6 observations in hour 03:00
        r2_times_h2 = [
            '2026-01-01 01:10:00+00:00', '2026-01-01 01:20:00+00:00', '2026-01-01 01:30:00+00:00',
            '2026-01-01 01:40:00+00:00', '2026-01-01 01:50:00+00:00', '2026-01-01 02:00:00+00:00'
        ]
        r2_times_h3 = [
            '2026-01-01 02:10:00+00:00', '2026-01-01 02:20:00+00:00', '2026-01-01 02:30:00+00:00',
            '2026-01-01 02:40:00+00:00', '2026-01-01 02:50:00+00:00', '2026-01-01 03:00:00+00:00'
        ]
        
        obs_rows = []
        for t in r1_times_h1:
            obs_rows.append({'station_id': 'R1', 'observed_at': pd.Timestamp(t), 'depth': 0.0, 'quality_status': 'observed'})
        for t in r1_times_h2:
            obs_rows.append({'station_id': 'R1', 'observed_at': pd.Timestamp(t), 'depth': 1.5, 'quality_status': 'observed'})
        for t in r2_times_h2:
            obs_rows.append({'station_id': 'R2', 'observed_at': pd.Timestamp(t), 'depth': 0.0, 'quality_status': 'observed'})
        for t in r2_times_h3:
            obs_rows.append({'station_id': 'R2', 'observed_at': pd.Timestamp(t), 'depth': 2.0, 'quality_status': 'observed'})
            
        self.obs = pd.DataFrame(obs_rows)

    def test_integrity_checks(self):
        summary = perform_integrity_checks(self.stations, self.obs, self.anchors)
        self.assertEqual(summary['total_stations'], 2)
        self.assertEqual(summary['total_records'], 23)
        self.assertEqual(summary['non_finite_depth_records'], 0)
        self.assertEqual(summary['negative_depth_records'], 0)
        self.assertEqual(summary['zero_depth_records'], 12)
        self.assertEqual(summary['positive_depth_records'], 11)
        self.assertEqual(summary['off_grid_records'], 0)
        self.assertEqual(summary['duplicate_timestamp_records'], 0)
        self.assertEqual(summary['max_records_in_any_hour'], 6)
        self.assertEqual(summary['hours_with_gt_6_records'], 0)

    def test_hourly_completeness_grid(self):
        df_grid, active_ranges = build_hourly_completeness_grid(self.stations, self.obs, self.anchors)
        self.assertEqual(len(df_grid), 6)  # 2 stations x 3 anchors
        
        # Verify R1 counts
        r1_counts = df_grid[df_grid['station_id'] == 'R1'].set_index('timestamp_utc')['valid_obs_count']
        self.assertEqual(r1_counts.loc[pd.Timestamp('2026-01-01 01:00:00+00:00')], 6)
        self.assertEqual(r1_counts.loc[pd.Timestamp('2026-01-01 02:00:00+00:00')], 5)
        self.assertEqual(r1_counts.loc[pd.Timestamp('2026-01-01 03:00:00+00:00')], 0)
        
        # Verify R2 active span
        r2_grid = df_grid[df_grid['station_id'] == 'R2'].set_index('timestamp_utc')
        self.assertFalse(r2_grid.loc[pd.Timestamp('2026-01-01 01:00:00+00:00'), 'in_active_span'])
        self.assertTrue(r2_grid.loc[pd.Timestamp('2026-01-01 02:00:00+00:00'), 'in_active_span'])
        self.assertTrue(r2_grid.loc[pd.Timestamp('2026-01-01 03:00:00+00:00'), 'in_active_span'])

    def test_station_and_system_metrics(self):
        df_grid, active_ranges = build_hourly_completeness_grid(self.stations, self.obs, self.anchors)
        df_full, df_act, df_comb = calculate_station_metrics(df_grid, self.stations, active_ranges)
        
        # R1 in active span: from first valid hour (01:00) to last valid hour (02:00) = 2 hours
        # (hour 03:00 is trailing, after the last valid observation)
        r1_act = df_act[df_act['station_id'] == 'R1'].iloc[0]
        self.assertEqual(r1_act['total_hours'], 2)
        self.assertEqual(r1_act['complete_hours'], 1)
        self.assertEqual(r1_act['incomplete_hours'], 1)
        self.assertEqual(r1_act['hours_6_obs'], 1)
        self.assertEqual(r1_act['hours_5_obs'], 1)
        self.assertEqual(r1_act['hours_0_obs'], 0)
        self.assertEqual(r1_act['gain_if_accept_ge_5'], 1)
        self.assertAlmostEqual(r1_act['mean_obs_per_hour'], 11.0 / 2.0, places=4)
        
        # R1 in full period: 3 hours, 1 with 6/6, 1 with 5/6, 1 with 0/6
        r1_full = df_full[df_full['station_id'] == 'R1'].iloc[0]
        self.assertEqual(r1_full['total_hours'], 3)
        self.assertEqual(r1_full['complete_hours'], 1)
        self.assertEqual(r1_full['incomplete_hours'], 2)
        self.assertEqual(r1_full['hours_0_obs'], 1)
        
        # R2 in active span: 2 hours (02:00 to 03:00), both 6/6
        r2_act = df_act[df_act['station_id'] == 'R2'].iloc[0]
        self.assertEqual(r2_act['total_hours'], 2)
        self.assertEqual(r2_act['complete_hours'], 2)
        self.assertEqual(r2_act['incomplete_hours'], 0)
        self.assertEqual(r2_act['completeness_pct'], 100.0)
        
        # System summary
        df_sys = calculate_system_summary(df_grid)
        self.assertEqual(len(df_sys), 2)
        act_sys = df_sys[df_sys['scope'].str.contains('Active')].iloc[0]
        self.assertEqual(act_sys['total_station_hours'], 4)  # 2 for R1 + 2 for R2
        self.assertEqual(act_sys['complete_hours_6_6'], 3)
        self.assertEqual(act_sys['incomplete_hours'], 1)
        self.assertAlmostEqual(act_sys['completeness_pct'], 75.0, places=2)
        self.assertEqual(act_sys['gain_if_accept_ge_5'], 1)
        self.assertAlmostEqual(act_sys['pct_ge_5_obs'], 100.0, places=2)


if __name__ == '__main__':
    unittest.main()
