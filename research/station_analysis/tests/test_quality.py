"""Screening must not reject negative levels or turn missing gaps into jumps."""

import unittest

import numpy as np
import pandas as pd

from research.station_analysis.quality import audit_quality


class QualityTests(unittest.TestCase):
    def run_audit(self, values, raw=None, **config):
        index = pd.date_range("2026-01-01", periods=len(values), freq="h", tz="UTC")
        hourly = pd.DataFrame({"a": values}, index=index)
        if raw is None:
            raw = pd.DataFrame({"station_id": "a", "observed_at": index,
                                "depth_mm": values, "quality_status": "observed"})
        result = audit_quality(hourly, raw, pd.DataFrame({"station_id": ["a"], "measurement_unit": ["mm"]}), config)
        return result, hourly, raw

    def test_negative_values_are_not_errors_and_inputs_are_preserved(self):
        result, hourly, raw = self.run_audit([-500., -450., -400., -350.])
        self.assertTrue(result["quality_flags"].empty)
        self.assertEqual(result["quality_summary"].iloc[0].negative_hourly_values, 4)
        self.assertEqual(hourly.a.tolist(), [-500., -450., -400., -350.])
        self.assertNotIn("screen_value", raw)

    def test_reverting_spike_differs_from_sustained_trend(self):
        spike, _, _ = self.run_audit([0., 2000., 20., 30.])
        trend, _, _ = self.run_audit([0., 2000., 4000., 6000.])
        self.assertEqual(spike["quality_summary"].iloc[0].isolated_spike_flags, 1)
        self.assertEqual(trend["quality_summary"].iloc[0].isolated_spike_flags, 0)
        self.assertEqual(trend["quality_summary"].iloc[0].hourly_jump_flags, 3)

    def test_missing_gap_is_not_an_hourly_jump(self):
        result, _, _ = self.run_audit([0., np.nan, 5000., 5001.])
        self.assertEqual(result["quality_summary"].iloc[0].hourly_jump_flags, 0)

    def test_absent_hour_timestamp_is_not_bridged(self):
        index = pd.to_datetime(["2026-01-01T00:00Z", "2026-01-01T02:00Z"])
        hourly = pd.DataFrame({"a": [0., 5000.]}, index=index)
        raw = pd.DataFrame({"station_id": "a", "observed_at": index, "depth_mm": [0., 5000.]})
        result = audit_quality(hourly, raw, pd.DataFrame({"station_id": ["a"], "measurement_unit": ["mm"]}), {})
        self.assertEqual(result["quality_summary"].iloc[0].hourly_jump_flags, 0)

    def test_unknown_unit_retains_cadence_without_mm_screen(self):
        index = pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC")
        hourly = pd.DataFrame({"a": [0., 5000.]}, index=index)
        raw = pd.DataFrame({"station_id": "a", "observed_at": index, "depth": [0., 5000.],
                            "quality_status": ["observed", "unverified"]})
        result = audit_quality(hourly, raw, pd.DataFrame({"station_id": ["a"], "measurement_unit": [None]}), {})
        self.assertFalse(result["quality_summary"].iloc[0].value_screens_applied)
        self.assertTrue(result["quality_flags"].empty)
        self.assertEqual(result["raw_cadence"].iloc[0].raw_records, 2)
        self.assertEqual(result["raw_cadence"].iloc[0].source_quality_other_rows, 1)

    def test_raw_within_hour_range_and_cadence(self):
        index = pd.to_datetime(["2026-01-01T00:00Z", "2026-01-01T00:10Z", "2026-01-01T00:20Z", "2026-01-01T02:00Z"])
        raw = pd.DataFrame({"station_id": "a", "observed_at": index,
                            "depth_mm": [0., 3000., 0., np.nan],
                            "quality_status": ["observed", "observed", "observed", "missing"]})
        result, _, _ = self.run_audit([1000., np.nan, np.nan], raw)
        self.assertEqual(result["quality_summary"].iloc[0].raw_within_hour_large_ranges, 1)
        cadence = result["raw_cadence"].iloc[0]
        self.assertEqual(cadence.median_interval_minutes, 10)
        self.assertEqual(cadence.max_interval_minutes, 100)
        self.assertEqual(cadence.non_10min_intervals, 1)
        self.assertEqual(cadence.source_quality_missing_rows, 1)

    def test_long_plateaus_are_diagnostics_including_natural_dry_zeros(self):
        result, _, _ = self.run_audit([0.] * 30 + [1200. + i % 2 for i in range(100)])
        summary = result["quality_summary"].iloc[0]
        self.assertEqual(summary.exact_zero_plateaus, 1)
        self.assertEqual(summary.low_variability_plateaus, 1)
        self.assertTrue(result["quality_flags"].status.str.startswith("needs_review").all())

    def test_duplicate_raw_timestamps_reported(self):
        index = pd.to_datetime(["2026-01-01T00:00Z", "2026-01-01T00:00Z"])
        raw = pd.DataFrame({"station_id": "a", "observed_at": index, "depth_mm": [1., 2.]})
        result, _, _ = self.run_audit([1.5], raw)
        self.assertEqual(result["raw_cadence"].iloc[0].duplicate_timestamp_extra_rows, 1)
        self.assertEqual(result["quality_flags"].iloc[0].kind, "duplicate_raw_timestamp")


if __name__ == "__main__":
    unittest.main()
