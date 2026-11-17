"""Forecast provenance, timestamp interval and no-look-ahead selection tests."""
import unittest

import numpy as np
import pandas as pd

from research.station_analysis.phase2.forecasts import audit_forecasts


class ForecastIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.init = pd.Timestamp('2026-10-01T00:00Z')
        self.locations = pd.DataFrame([{'location_id': 'grid1', 'station_id': '750062',
                                        'latitude': 16.2, 'longitude': 107.2}])
        self.config = {'assumed_release_lag_hours': 6, 'max_forecast_age_hours': 24}

    def run_row(self, run_id='r1', *, init=None, retrieved=None, available=None,
                basis='unknown_archive', kind='archive_50r1_unverified'):
        init = init if init is not None else self.init
        return {'run_id': run_id, 'location_id': 'grid1', 'model': 'ecmwf_ifs',
                'initialized_at': init, 'retrieved_at': retrieved if retrieved is not None else init+pd.Timedelta(days=20),
                'available_at': available, 'availability_basis': basis, 'archive_kind': kind,
                'grid_latitude': 16.27, 'grid_longitude': 107.26, 'checksum': run_id}

    def points(self, run_id='r1', *, init=None, length=48, value=1.0):
        init = init if init is not None else self.init
        return pd.DataFrame({'run_id': run_id,
                             'valid_at': pd.date_range(init+pd.Timedelta(hours=1), periods=length, freq='h'),
                             'precipitation_mm': value, 'quality_status': 'forecast'})

    def audit(self, rows, points=None, hours=6):
        points = self.points() if points is None else points
        anchors = pd.DatetimeIndex([self.init+pd.Timedelta(hours=hours)])
        return audit_forecasts(pd.DataFrame(rows), points, self.locations, anchors, self.config)

    @staticmethod
    def selected(result, policy):
        table = result['forecast_anchor_availability']
        return table.loc[table.policy.eq(policy)].iloc[0]

    def test_unknown_archive_only_enters_assumed_diagnostic_branch(self):
        result = self.audit([self.run_row()])
        legacy = self.selected(result, 'legacy_assumed_release')
        strict = self.selected(result, 'strict_operational')
        self.assertEqual(legacy.selected_run_id, 'r1')
        self.assertTrue(legacy.retrieved_after_anchor)
        self.assertTrue(legacy.complete_24h)
        self.assertTrue(pd.isna(strict.selected_run_id))
        self.assertFalse(strict.complete_6h)

    def test_actual_availability_can_precede_six_hour_assumption(self):
        retrieval = self.init+pd.Timedelta(hours=4)
        row = self.run_row(retrieved=retrieval, available=retrieval,
                           basis='observed_retrieval', kind='operational_single_run')
        result = self.audit([row], hours=4)
        self.assertEqual(self.selected(result, 'strict_operational').selected_run_id, 'r1')
        self.assertTrue(pd.isna(self.selected(result, 'legacy_assumed_release').selected_run_id))

    def test_later_revision_does_not_displace_available_earlier_vintage(self):
        earlier = self.init+pd.Timedelta(hours=5)
        later = self.init+pd.Timedelta(hours=8)
        rows = [self.run_row('old', retrieved=earlier, available=earlier,
                             basis='observed_retrieval', kind='operational_single_run'),
                self.run_row('new', retrieved=later, available=later,
                             basis='observed_retrieval', kind='operational_single_run')]
        points = pd.concat([self.points('old', value=1), self.points('new', value=9)], ignore_index=True)
        result = self.audit(rows, points)
        strict = self.selected(result, 'strict_operational')
        self.assertEqual(strict.selected_run_id, 'old')
        self.assertEqual(strict.rain_0_3h_mm, 3)

    def test_hindcast_cannot_be_promoted_by_a_timestamp(self):
        retrieval = self.init+pd.Timedelta(hours=5)
        row = self.run_row(retrieved=retrieval, available=retrieval,
                           basis='observed_retrieval', kind='hindcast_49r1')
        result = self.audit([row])
        self.assertTrue(self.selected(result, 'legacy_assumed_release').hindcast_selected)
        self.assertTrue(pd.isna(self.selected(result, 'strict_operational').selected_run_id))

    def test_unknown_availability_basis_is_not_evidence(self):
        retrieval = self.init+pd.Timedelta(hours=5)
        row = self.run_row(retrieved=retrieval, available=retrieval,
                           basis='assumed_6h', kind='operational_single_run')
        result = self.audit([row])
        self.assertFalse(result['forecast_run_integrity'].iloc[0].strict_eligible_run)

    def test_interval_windows_use_exact_nonoverlapping_hour_ends(self):
        points = self.points()
        points['precipitation_mm'] = np.arange(1, len(points)+1)
        result = self.audit([self.run_row()], points)
        row = self.selected(result, 'legacy_assumed_release')
        # Anchor is06UTC: (anchor,anchor+3h] selects07,08,09UTC.
        self.assertEqual(row.rain_0_3h_mm, 7+8+9)
        self.assertEqual(row.rain_3_6h_mm, 10+11+12)
        self.assertEqual(row.rain_6_12h_mm, sum(range(13, 19)))
        self.assertEqual(row.rain_12_24h_mm, sum(range(19, 31)))

    def test_missing_hour_does_not_shift_later_values_into_interval(self):
        points = self.points().drop(index=7).reset_index(drop=True)
        row = self.selected(self.audit([self.run_row()], points), 'legacy_assumed_release')
        self.assertTrue(np.isnan(row.rain_0_3h_mm))
        self.assertFalse(row.complete_6h)
        self.assertFalse(row.complete_24h)
        self.assertEqual(row.rain_3_6h_mm, 3)

    def test_duplicate_hours_and_conflicting_metadata_are_quarantined(self):
        points = pd.concat([self.points(), self.points().iloc[[7]]], ignore_index=True)
        result = self.audit([self.run_row()], points)
        self.assertEqual(result['forecast_run_integrity'].iloc[0].duplicated_hour_rows, 2)
        self.assertTrue(pd.isna(self.selected(result, 'legacy_assumed_release').selected_run_id))
        result = self.audit([self.run_row(), self.run_row()])
        self.assertTrue(result['forecast_run_integrity'].iloc[0].duplicate_metadata_id)
        self.assertTrue(pd.isna(self.selected(result, 'legacy_assumed_release').selected_run_id))

    def test_invalid_values_and_missing_quality_are_reported(self):
        points = self.points()
        points.loc[6, 'precipitation_mm'] = -1
        points.loc[7, 'precipitation_mm'] = np.inf
        points.loc[8, 'precipitation_mm'] = np.nan
        result = self.audit([self.run_row()], points)
        audit = result['forecast_run_integrity'].iloc[0]
        self.assertEqual(audit.negative_value_rows, 1)
        self.assertEqual(audit.nonfinite_value_rows, 1)
        self.assertEqual(audit.quality_value_mismatch_rows, 3)
        self.assertFalse(self.selected(result, 'legacy_assumed_release').complete_6h)

    def test_available_at_before_retrieval_is_not_trusted(self):
        result = self.audit([self.run_row(
            retrieved=self.init+pd.Timedelta(hours=8), available=self.init+pd.Timedelta(hours=5),
            basis='observed_retrieval', kind='operational_single_run')])
        audit = result['forecast_run_integrity'].iloc[0]
        self.assertIn('availability_precedes_retrieval', audit.strict_exclusion_reason)
        self.assertTrue(pd.isna(self.selected(result, 'strict_operational').selected_run_id))

    def test_empty_forecast_tables_keep_location_denominator(self):
        runs = pd.DataFrame([self.run_row()]).iloc[:0]
        points = self.points().iloc[:0]
        result = audit_forecasts(runs, points, self.locations,
                                 pd.date_range(self.init, periods=3, freq='h'), self.config)
        self.assertTrue(result['forecast_run_integrity'].empty)
        coverage = result['forecast_coverage']
        self.assertEqual(len(coverage), 2)
        self.assertTrue(coverage.anchor_count.eq(3).all())
        self.assertTrue(coverage.complete_6h_anchors.eq(0).all())

    def test_timezone_and_interval_validation(self):
        bad = self.run_row(init=pd.Timestamp('2026-10-01'))
        with self.assertRaisesRegex(ValueError, 'timezone'):
            self.audit([bad])
        self.config['windows'] = [[0, 6], [3, 12]]
        with self.assertRaisesRegex(ValueError, 'non-overlapping'):
            self.audit([self.run_row()])

    def test_orphan_forecast_points_are_retained_in_audit(self):
        points = pd.concat([self.points(), self.points('orphan').iloc[:2]], ignore_index=True)
        result = self.audit([self.run_row()], points)
        self.assertEqual(len(result['forecast_orphan_points']), 2)

    def test_payload_units_and_utc_offset_are_audited(self):
        retrieval = self.init+pd.Timedelta(hours=5)
        row = self.run_row(retrieved=retrieval, available=retrieval,
                           basis='observed_retrieval', kind='operational_single_run')
        row['hourly_units'] = '{"precipitation":"mm","time":"unixtime"}'
        row['utc_offset_seconds'] = '0'
        result = self.audit([row])
        self.assertTrue(result['forecast_run_integrity'].iloc[0].payload_precipitation_unit_verified_mm)
        self.assertTrue(self.selected(result, 'strict_operational').complete_24h)
        row['hourly_units'] = '{"precipitation":"inch"}'
        result = self.audit([row])
        self.assertFalse(result['forecast_run_integrity'].iloc[0].strict_eligible_run)
        self.assertFalse(self.selected(result, 'legacy_assumed_release').complete_24h)

    def test_grid_collisions_are_not_counted_as_independent_forecasts(self):
        locations = pd.concat([self.locations, self.locations.assign(location_id='grid2', station_id='750007')])
        runs = pd.DataFrame([self.run_row(), {**self.run_row('r2'), 'location_id': 'grid2'}])
        points = pd.concat([self.points(), self.points('r2')], ignore_index=True)
        result = audit_forecasts(runs, points, locations,
                                 pd.DatetimeIndex([self.init+pd.Timedelta(hours=6)]), self.config)
        self.assertTrue(result['forecast_location_grids'].locations_sharing_returned_grid.eq(2).all())

    def test_missing_six_hour_cycle_is_reported(self):
        later = self.init+pd.Timedelta(hours=12)
        rows = [self.run_row(), self.run_row('r2', init=later)]
        points = pd.concat([self.points(), self.points('r2', init=later)], ignore_index=True)
        result = self.audit(rows, points)
        self.assertEqual(result['forecast_cycle_inventory'].iloc[0].missing_6h_cycles_within_span, 1)
        self.assertEqual(result['forecast_cycle_gaps'].iloc[0].elapsed_hours, 12)

    def test_microsecond_datetime_storage_uses_same_absolute_hour_keys(self):
        points = self.points()
        points['valid_at'] = points.valid_at.astype('datetime64[us, UTC]')
        result = self.audit([self.run_row()], points)
        row = self.selected(result, 'legacy_assumed_release')
        self.assertTrue(row.complete_24h)
        self.assertEqual(row.rain_6_12h_mm, 6)

    def test_local_timezone_represents_same_forecast_instants(self):
        points = self.points()
        points['valid_at'] = points.valid_at.dt.tz_convert('Asia/Ho_Chi_Minh')
        result = self.audit([self.run_row()], points)
        row = self.selected(result, 'legacy_assumed_release')
        self.assertTrue(row.complete_24h)
        self.assertEqual(row.rain_0_3h_mm, 3)


if __name__ == '__main__':
    unittest.main()
