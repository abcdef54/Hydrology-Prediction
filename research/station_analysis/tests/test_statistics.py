"""Protect physical-time alignment and missing-data semantics in station EDA."""

import unittest

import numpy as np
import pandas as pd

from research.station_analysis.statistics import analyze, exact_change, hourly_grid, window_counts


class StationStatisticsTests(unittest.TestCase):
    def index(self, periods):
        return pd.date_range("2026-09-01", periods=periods, freq="h", tz="UTC")

    def run_analysis(self, frame, **config):
        return analyze(frame, pd.DataFrame({"station_id": frame.columns}), {
            "sequence_length": 4, "horizons": [6, 12], "correlation_min_pairs": 3,
            **config,
        })

    def test_exact_change_does_not_shift_over_missing_timestamps(self):
        index = self.index(20).delete(6)
        series = pd.Series(np.arange(20, dtype=float)[np.arange(20) != 6], index=index)
        result = exact_change(series, 6)
        self.assertTrue(pd.isna(result.iloc[0]))
        self.assertEqual(result.loc[index[1]], 6)

    def test_grid_inserts_holes_and_retains_negative_zero(self):
        index = self.index(4).delete(1)
        frame = hourly_grid(pd.DataFrame({"a": [-1., 0., np.inf]}, index=index), ["a", "b"])
        self.assertEqual(len(frame), 4)
        self.assertEqual(frame.a.iloc[0], -1)
        self.assertEqual(frame.a.iloc[2], 0)
        self.assertTrue(pd.isna(frame.a.iloc[1]))
        self.assertTrue(pd.isna(frame.a.iloc[3]))
        self.assertTrue(frame.b.isna().all())

    def test_windows_require_history_and_exact_targets(self):
        series = pd.Series(1., index=self.index(20))
        result = window_counts(series, 4, [6, 12])
        self.assertEqual(result["strict_water_windows"], 5)
        series.iloc[6] = np.nan
        # Anchors 3,4,5 remain; 6,7 lose history. Anchor 0's missing target
        # does not matter because no complete four-hour history exists there.
        self.assertEqual(window_counts(series, 4, [6, 12])["strict_water_windows"], 3)
        series.iloc[16] = np.nan
        self.assertEqual(window_counts(series, 4, [6, 12])["strict_water_windows"], 2)

    def test_missing_runs_include_prefix_and_suffix(self):
        frame = pd.DataFrame({"a": [np.nan, np.nan, 0., 1., np.nan, 1., np.nan]}, index=self.index(7))
        tables = self.run_analysis(frame)
        self.assertEqual(tables["missing_gaps"].missing_hours.tolist(), [2, 1, 1])
        row = tables["station_statistics"].iloc[0]
        self.assertEqual(row.missing_hours, 4)
        self.assertEqual(row.active_span_hours, 4)
        self.assertEqual(row.active_span_missing_rate, .25)

    def test_constant_and_empty_stations_remain_visible(self):
        frame = pd.DataFrame({"a": 0., "b": np.nan}, index=self.index(80))
        tables = self.run_analysis(frame)
        self.assertEqual(len(tables["station_statistics"]), 2)
        stats = tables["station_statistics"].set_index("station_id")
        self.assertEqual(stats.loc["a", "zero_fraction"], 1.)
        self.assertEqual(stats.loc["b", "valid_hours"], 0)
        self.assertEqual(stats.loc["b", "strict_water_windows"], 0)
        self.assertTrue(tables["correlations"].pearson_r.isna().all())
        self.assertTrue(tables["events"].empty)

    def test_events_keep_rise_and_fall_separate(self):
        values = np.zeros(50)
        values[18:30] = 500
        tables = self.run_analysis(pd.DataFrame({"a": values}, index=self.index(50)))
        events = tables["events"]
        self.assertTrue((events.loc[events.kind == "rapid_rise", "peak_value_mm"] > 0).all())
        self.assertTrue((events.loc[events.kind == "rapid_fall", "peak_value_mm"] < 0).all())
        self.assertEqual(set(events.kind), {"rapid_rise", "rapid_fall", "high_level_p95"})

    def test_grouping_cutoff_does_not_use_future_changes(self):
        frame = pd.DataFrame({"a": np.r_[np.arange(40), np.arange(40) * 100.]}, index=self.index(80))
        before = frame.index[40]
        tables = self.run_analysis(frame, grouping_end=str(before))
        expected = self.run_analysis(frame.iloc[:40])["grouping_features"].iloc[0]
        result = tables["grouping_features"].iloc[0]
        self.assertEqual(result.observed_hours, 40)
        self.assertAlmostEqual(result.change_12h_iqr_ratio, expected.change_12h_iqr_ratio)
        self.assertEqual(tables["station_statistics"].iloc[0].valid_hours, 80)

    def test_sparse_peak_does_not_turn_dry_baseline_into_high_water_event(self):
        values = np.zeros(100)
        values[50] = 500
        tables = self.run_analysis(pd.DataFrame({"a": values}, index=self.index(100)))
        high = tables["events"].query("kind == 'high_level_p95'")
        self.assertEqual(len(high), 1)
        self.assertEqual(high.iloc[0].flagged_anchors, 1)
        self.assertEqual(high.iloc[0].peak_value_mm, 500)

    def test_correlation_alignment_uses_elapsed_time(self):
        values = np.arange(50.) ** 2
        frame = pd.DataFrame({"a": values, "b": values * 2 + 700}, index=self.index(50))
        frame.iloc[10, 0] = np.nan
        tables = self.run_analysis(frame)
        row = tables["correlations"].query("kind == 'change_6h' and lag_hours == 0").iloc[0]
        self.assertEqual(row.paired_hours, 42)
        self.assertAlmostEqual(row.pearson_r, 1)

    def test_invalid_time_and_duplicate_station_inputs_rejected(self):
        frame = pd.DataFrame({"a": [1., 2.]}, index=pd.date_range("2026-01-01", periods=2, freq="h"))
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            self.run_analysis(frame)
        frame.index = self.index(2)
        with self.assertRaisesRegex(ValueError, "unique"):
            analyze(frame, pd.DataFrame({"station_id": ["a", "a"]}), {})


if __name__ == "__main__":
    unittest.main()
