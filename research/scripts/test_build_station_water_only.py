"""Audit of the water-only station dataset (run: .venv/bin/python -m unittest research.scripts.test_build_station_water_only -v)."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from research.scripts import build_station_water_only as B
from src.ai.regional_training_data import regional_loaders


class WaterOnly(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.d, cls.s, _ = B.build(output=Path(cls.tmp.name))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_splits_match_regional_pool(self):
        src = pd.read_parquet(B.SOURCE / 'samples.parquet')
        src = src[src.station_id == B.STATION].reset_index(drop=True)
        pd.testing.assert_series_equal(self.s.anchor_utc, src.anchor_utc, check_names=False)
        pd.testing.assert_series_equal(self.s.split, src.split, check_names=False)

    def test_labels_exact(self):
        w = pd.read_parquet(self.d / 'water.parquet').set_index('timestamp_utc')[B.STATION]
        for h in B.HORIZONS:
            np.testing.assert_allclose(self.s[f'target_h{h}_mm'], w.reindex(pd.DatetimeIndex(self.s.anchor_utc) + pd.Timedelta(hours=h)).to_numpy())

    def test_normalisers_train_only(self):
        n = json.loads((self.d / 'normalizers.json').read_text())['fixed']['stations'][B.STATION]
        val_start = pd.Timestamp(json.loads((self.d / 'config.json').read_text())['folds'][0]['validation_start'])
        self.assertLess(pd.Timestamp(n['input_fit_last']), val_start)
        self.assertLess(pd.Timestamp(n['target_fit_last']), val_start)

    def test_reader_water_only(self):
        tr, va, te = regional_loaders(self.d, batch_size=32, num_workers=0, seed=0, rain_stations='none',
                                      use_weather_forecast=False, time_features='none', target_horizons=[3, 6, 12])
        self.assertEqual(tr.dataset.feature_cols, ['water_level_depth', 'water_level_missing', 'flood_season'])
        self.assertEqual((len(tr.dataset), len(va.dataset), len(te.dataset)), (6073, 1080, 723))
        self.assertEqual(tuple(next(iter(va))['x'].shape[1:]), (96, 3))

    def test_refuses_overwrite(self):
        with self.assertRaises(FileExistsError):
            B.build(output=Path(self.tmp.name))


if __name__ == '__main__':
    unittest.main()
