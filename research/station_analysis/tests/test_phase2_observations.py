"""Measurement meaning, time boundaries and missingness before feature analysis."""
import unittest

import numpy as np
import pandas as pd

from research.station_analysis.phase2.observations import (
    audit_observations, strict_accumulation, verified_rain_hourly,
)


class Phase2ObservationsTests(unittest.TestCase):
    def setUp(self):
        self.hours = pd.date_range('2026-01-01T00:00Z', periods=4, freq='h')
        self.stations = pd.DataFrame([
            {'station_id': '001', 'name': 'Water', 'station_type': 'water_level', 'measurement_unit': 'mm'},
            {'station_id': '002', 'name': 'Rain', 'station_type': 'rainfall', 'measurement_unit': 'mm'},
            {'station_id': '003', 'name': 'Empty', 'station_type': 'rainfall', 'measurement_unit': 'mm'},
        ])

    def records(self, sid, stamps, values):
        return pd.DataFrame({'station_id': sid, 'observed_at': stamps, 'depth': values,
                             'quality_status': 'observed',
                             'received_at': pd.Timestamp('2026-02-01T00:00Z'),
                             'updated_at': pd.Timestamp('2026-02-01T00:00Z')})

    def contract(self, kind='interval_amount', unit='mm'):
        return {'kind': kind, 'unit': unit, 'interval_minutes': 10,
                'confirmed': True, 'source': 'verified sensor manual'}

    def test_right_closed_label_excludes_future_measurements(self):
        raw = self.records('001', pd.to_datetime(['2026-01-01T00:00Z', '2026-01-01T00:10Z']), [-2, 10])
        tables = audit_observations(raw, self.stations, self.hours, {})
        matrix = tables['observation_availability'].query("station_id=='001'").set_index('timestamp_utc')
        self.assertEqual(matrix.loc[self.hours[0], 'water_mean_mm'], -2)
        self.assertEqual(matrix.loc[self.hours[1], 'water_mean_mm'], 10)
        self.assertEqual(matrix.loc[self.hours[0], 'receipt_by_anchor_count'], 0)
        self.assertEqual(matrix.loc[self.hours[0], 'receipt_after_anchor_count'], 1)
        alignment = tables['hourly_alignment_audit'].set_index('station_id')
        self.assertEqual(alignment.loc['001', 'phase1_left_label_readings_after_label'], 1)

    def test_unknown_semantics_never_produces_rain_totals(self):
        raw = self.records('002', pd.to_datetime(['2026-01-01T00:10Z']), [2])
        tables = audit_observations(raw, self.stations, self.hours, {})
        matrix = tables['observation_availability'].query("station_id=='002'")
        self.assertTrue(matrix.rain_hourly_verified_mm.isna().all())
        self.assertEqual(matrix.legacy_rain_sum_mm_diagnostic.dropna().iloc[0], 2)
        self.assertFalse(matrix.rainfall_semantics_confirmed.any())
        sem = tables['rainfall_semantics_audit'].set_index('station_id')
        self.assertEqual(sem.loc['002', 'semantics_status'], 'unknown_do_not_build_rainfall_features')

    def test_zero_station_and_missing_runs_remain_visible(self):
        raw = self.records('001', pd.to_datetime(['2026-01-01T01:00Z']), [10])
        tables = audit_observations(raw, self.stations, self.hours, {})
        inv = tables['observation_inventory'].set_index('station_id')
        self.assertEqual(inv.loc['003', 'raw_records'], 0)
        self.assertEqual(inv.loc['003', 'missing_rate'], 1)
        gaps = tables['observation_missing_gaps']
        self.assertEqual(set(gaps[gaps.station_id.eq('001')].position), {'leading', 'trailing'})
        self.assertEqual(gaps[gaps.station_id.eq('003')].missing_hours.iloc[0], 4)

    def test_negative_rain_invalid_water_negative_retained(self):
        stamps = pd.to_datetime(['2026-01-01T01:00Z'])
        raw = pd.concat([self.records('001', stamps, [-10]), self.records('002', stamps, [-10])])
        tables = audit_observations(raw, self.stations, self.hours, {})
        inventory = tables['observation_inventory'].set_index('station_id')
        self.assertEqual(inventory.loc['001', 'hours_with_valid_values'], 1)
        self.assertEqual(inventory.loc['002', 'hours_with_valid_values'], 0)
        self.assertEqual(inventory.loc['002', 'negative_values'], 1)

    def test_partial_hours_are_distinct_from_presence(self):
        stamps = pd.date_range('2026-01-01T00:10Z', periods=6, freq='10min')
        raw = self.records('002', stamps, [1]*6)
        raw = pd.concat([raw, self.records('002', pd.to_datetime(['2026-01-01T01:10Z']), [0])])
        tables = audit_observations(raw, self.stations, self.hours, {})
        matrix = tables['observation_availability'].query("station_id=='002'").set_index('timestamp_utc')
        self.assertTrue(matrix.loc[self.hours[1], 'complete_expected_grid'])
        self.assertTrue(matrix.loc[self.hours[2], 'has_valid_value'])
        self.assertFalse(matrix.loc[self.hours[2], 'complete_expected_grid'])

    def test_duplicate_rows_and_orphan_stations_are_flagged(self):
        stamps = pd.to_datetime(['2026-01-01T01:00Z', '2026-01-01T01:00Z'])
        raw = pd.concat([self.records('001', stamps, [1, 2]),
                         self.records('unknown', stamps[:1], [3])])
        tables = audit_observations(raw, self.stations, self.hours, {})
        inv = tables['observation_inventory'].set_index('station_id')
        self.assertEqual(inv.loc['001', 'duplicate_timestamp_extra_rows'], 1)
        self.assertEqual(tables['observation_orphans'].raw_records.sum(), 1)
        self.assertTrue(tables['observation_availability'].query("station_id=='001'").water_mean_mm.isna().all())

    def test_known_interval_amount_and_rate_have_different_totals(self):
        index = pd.date_range('2026-01-01T00:10Z', periods=6, freq='10min')
        series = pd.Series(2., index=index)
        amount = verified_rain_hourly(series, self.hours, self.contract())
        rate = verified_rain_hourly(series, self.hours, self.contract('rate', 'mm/h'))
        self.assertEqual(amount.loc[self.hours[1]], 12)
        self.assertAlmostEqual(rate.loc[self.hours[1]], 2)
        self.assertTrue(np.isnan(amount.loc[self.hours[0]]))

    def test_counter_reset_is_missing_instead_of_guessed_rain(self):
        index = pd.date_range('2026-01-01T00:00Z', periods=13, freq='10min')
        series = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3], index=index)
        result = verified_rain_hourly(series, self.hours, self.contract('cumulative_counter'))
        self.assertEqual(result.loc[self.hours[1]], 6)
        self.assertTrue(np.isnan(result.loc[self.hours[2]]))

    def test_unknown_or_crossing_interval_contract_rejected(self):
        series = pd.Series([1], index=pd.to_datetime(['2026-01-01T00:05Z']))
        with self.assertRaisesRegex(ValueError, 'cross'):
            verified_rain_hourly(series, self.hours, self.contract())
        with self.assertRaisesRegex(ValueError, 'confirmed'):
            verified_rain_hourly(series, self.hours, {'kind': 'interval_amount'})
        contract = self.contract(); contract['interval_minutes'] = 17
        with self.assertRaisesRegex(ValueError, 'dividing'):
            verified_rain_hourly(series, self.hours, contract)

    def test_strict_accumulation_retains_timestamp_holes_and_zero(self):
        series = pd.Series([0., 1., 2.], index=self.hours[[0, 1, 3]])
        result = strict_accumulation(series, [1, 3])
        self.assertEqual(result.loc[self.hours[0], 'rain_sum_1h'], 0)
        self.assertTrue(np.isnan(result.loc[self.hours[2], 'rain_sum_1h']))
        self.assertTrue(np.isnan(result.loc[self.hours[3], 'rain_sum_3h']))

    def test_naive_and_nonhour_anchors_rejected(self):
        raw = self.records('001', ['2026-01-01 01:00'], [1])
        with self.assertRaisesRegex(ValueError, 'timezone'):
            audit_observations(raw, self.stations, self.hours, {})
        series = pd.Series([0], index=pd.date_range('2026-01-01', periods=1, freq='h'))
        with self.assertRaises(ValueError):
            strict_accumulation(series)


if __name__ == '__main__':
    unittest.main()
