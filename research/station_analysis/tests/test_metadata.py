from unittest import TestCase

import pandas as pd

from research.station_analysis.metadata import enrich_stations, load_sources


class StationMetadataTests(TestCase):
    def test_da_vien_and_tu_phu_do_not_inherit_official_gauge_alerts(self):
        frame = pd.DataFrame({
            "id": ["0074647403", "0074647404"],
            "name": ["Da Vien", "Tu Phu"],
            "station_subtype": ["water_level", "water_level"],
            "alarm_level_1_m": [None, None],
            "measurement_unit": ["mm", "mm"],
        })
        out = enrich_stations(frame)
        self.assertTrue(out.alarm_level_1_m.isna().all())
        self.assertEqual(out.alarm_basis.tolist(), ["not_available"] * 2)
        self.assertEqual(out.measured_quantity.tolist(), ["unknown"] * 2)
        self.assertEqual(out.water_level_datum.tolist(), ["unknown"] * 2)
        self.assertTrue(out.station_zero_elevation_m.isna().all())

    def test_flood_subtype_is_hypothesis_not_ground_zero(self):
        frame = pd.DataFrame({"station_id": ["751115"], "station_subtype": ["flood_1m5"]})
        out = enrich_stations(frame)
        self.assertEqual(out.hydrological_type.iloc[0], "inundation_tower_provisional")
        self.assertEqual(out.measured_quantity.iloc[0], "unknown")
        self.assertIn("local_inundation", out.measured_quantity_hypothesis.iloc[0])

    def test_preserve_coordinates_raw_alarm_and_idempotence(self):
        frame = pd.DataFrame({
            "id": ["unknown"], "latitude": [16.5], "longitude": [107.5],
            "alarm_level_1_m": [1.25],
        })
        out = enrich_stations(frame)
        self.assertEqual(out.alarm_level_1_m.iloc[0], 1.25)
        self.assertEqual(out.latitude.iloc[0], 16.5)
        self.assertIn("unverified", out.alarm_basis.iloc[0])
        pd.testing.assert_frame_equal(out, enrich_stations(out))

    def test_all_19_curated_station_ids_and_vietnam_license(self):
        sources = load_sources()
        self.assertEqual(len(sources["stations"]), 19)
        self.assertIn("750048", sources["stations"])
        self.assertIn("750049", sources["stations"])
        self.assertIn("CC BY 4.0", sources["sources"]["fabdem_vietnam"]["supports"])
