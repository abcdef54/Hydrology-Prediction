"""Grouping must respect temporal scope, units and finite dynamics."""

import unittest

import numpy as np
import pandas as pd

from research.station_analysis.grouping import DYNAMIC_FEATURES, group_stations


class GroupingTests(unittest.TestCase):
    cutoff = "2026-08-01T00:00:00+00:00"

    def fixtures(self, count=8):
        ids = [f"{i:03d}" for i in range(count)]
        rng = np.random.default_rng(42)
        matrix = np.r_[rng.normal(0, .1, size=(count // 2, len(DYNAMIC_FEATURES))),
                       rng.normal(4, .1, size=(count - count // 2, len(DYNAMIC_FEATURES)))]
        features = pd.DataFrame(matrix, columns=DYNAMIC_FEATURES)
        features["station_id"] = ids
        features["observed_hours"] = 500
        features["scale_available"] = True
        features["iqr_mm"] = 100
        features["scope"] = "before_grouping_end"
        features["grouping_end_exclusive"] = self.cutoff
        features["diurnal_power_fraction_proxy"] = np.linspace(0, 1, count)
        features["semidiurnal_power_fraction_proxy"] = np.linspace(0, 1, count)
        metadata = pd.DataFrame({"station_id": ids, "measurement_unit": "mm",
                                 "measured_quantity": "unknown"})
        terrain = pd.DataFrame({"station_id": ids, "dem_source": "FABDEM",
                                "radius_m": 200, "median_m": np.arange(count)})
        return features, metadata, terrain

    def run_group(self, features, metadata, terrain, **config):
        return group_stations(features, metadata, terrain, {"grouping_end": self.cutoff,
                              "min_pair_hours": 168, "cluster_max_k": 4, **config})

    def test_dynamics_clusters_ignore_elevation_and_absolute_levels(self):
        features, metadata, terrain = self.fixtures()
        features["mean_mm"] = np.arange(8) * 10000
        first = self.run_group(features, metadata, terrain)
        terrain["median_m"] = terrain.median_m.iloc[::-1].to_numpy() * 50
        features["mean_mm"] *= -1
        second = self.run_group(features, metadata, terrain)
        self.assertEqual(first["cluster_assignments"].cluster.tolist(), second["cluster_assignments"].cluster.tolist())
        self.assertEqual(first["cluster_scores"].query("selected").iloc[0].k, 2)
        self.assertNotIn("mean_mm", first["cluster_features"])
        self.assertNotIn("median_m", first["cluster_features"])
        self.assertTrue(first["cluster_sensitivity"].adjusted_rand_index.between(-1, 1).all())

    def test_unknown_unit_sparse_and_constant_stations_excluded(self):
        features, metadata, terrain = self.fixtures()
        metadata.loc[0, "measurement_unit"] = None
        features.loc[1, "observed_hours"] = 10
        features.loc[2, "scale_available"] = False
        rows = self.run_group(features, metadata, terrain)["cluster_assignments"]
        self.assertTrue(rows.iloc[:3].cluster.isna().all())
        self.assertIn("measurement_unit", rows.iloc[0].exclusion_reason)
        self.assertIn("insufficient", rows.iloc[1].exclusion_reason)
        self.assertIn("constant", rows.iloc[2].exclusion_reason)

    def test_no_cutoff_does_not_fit_full_period_features(self):
        features, metadata, terrain = self.fixtures()
        features["scope"] = "full_period_descriptive_only"
        result = group_stations(features, metadata, terrain, {})
        self.assertTrue(result["cluster_assignments"].cluster.isna().all())
        self.assertTrue(result["cluster_scores"].empty)

    def test_cutoff_mismatch_excluded(self):
        features, metadata, terrain = self.fixtures()
        features.loc[0, "grouping_end_exclusive"] = "2026-09-01T00:00Z"
        row = self.run_group(features, metadata, terrain)["cluster_assignments"].iloc[0]
        self.assertEqual(row.status, "excluded")
        self.assertIn("cutoff_mismatch", row.exclusion_reason)

    def test_two_stations_or_constant_features_do_not_force_clusters(self):
        features, metadata, terrain = self.fixtures(2)
        result = self.run_group(features, metadata, terrain)
        self.assertTrue(result["cluster_scores"].empty)
        self.assertTrue(result["cluster_assignments"].cluster.isna().all())
        features, metadata, terrain = self.fixtures()
        features[DYNAMIC_FEATURES] = 1.
        features[["diurnal_power_fraction_proxy", "semidiurnal_power_fraction_proxy"]] = 0.
        self.assertTrue(self.run_group(features, metadata, terrain)["cluster_scores"].empty)

    def test_missing_spectral_features_are_dropped_without_imputation(self):
        features, metadata, terrain = self.fixtures()
        features.loc[0, "diurnal_power_fraction_proxy"] = np.nan
        result = self.run_group(features, metadata, terrain)
        self.assertNotIn("diurnal_power_fraction_proxy", result["cluster_features"])
        self.assertEqual(len(result["cluster_assignments"].query("status == 'clustered_exploratory'")), 8)

    def test_small_terrain_sample_has_no_inferential_coefficient(self):
        features, metadata, terrain = self.fixtures(4)
        result = self.run_group(features, metadata, terrain)["terrain_associations"]
        self.assertTrue(result.spearman_rho.isna().all())
        self.assertTrue(result.descriptive_p_value.isna().all())
        self.assertTrue(result.n.eq(4).all())


if __name__ == "__main__":
    unittest.main()
