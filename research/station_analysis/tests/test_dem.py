"""Coordinate ordering, projected CRS, metric neighborhoods and nodata tests."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from pyproj import Transformer
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from research.station_analysis.dem import clip_dem, station_terrain


class TerrainTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def raster(self, name, values, transform, crs='EPSG:32648'):
        path = self.folder / name
        with rasterio.open(path, 'w', driver='GTiff', height=values.shape[0],
                           width=values.shape[1], count=1, dtype='float32',
                           crs=crs, transform=transform, nodata=-9999) as out:
            out.write(values.astype('float32'), 1)
        return path

    def test_metric_circle_and_nodata(self):
        x, y = Transformer.from_crs(4326, 32648, always_xy=True).transform(107.57, 16.46)
        array = np.full((31, 31), 10.0)
        array[15, 15] = -9999
        path = self.raster('projected.tif', array, from_origin(x-155, y+155, 10, 10))
        stations = pd.DataFrame([{'station_id': 'x', 'longitude': 107.57, 'latitude': 16.46}])
        result = station_terrain(path, stations, [100]).iloc[0]
        self.assertEqual(result.status, 'ok')
        self.assertAlmostEqual(result.median_m, 10)
        self.assertLess(result.valid_fraction, 1)
        self.assertTrue(np.isnan(result.point_m))
        self.assertGreater(result.n_pixels, 300)
        self.assertLess(result.n_pixels, 320)
        self.assertAlmostEqual(result.slope_median_deg, 0)

    def test_longitude_latitude_pixel_order(self):
        path = self.raster('geo.tif', np.arange(400).reshape(20, 20),
                           from_origin(107.56, 16.47, .001, .001), 'EPSG:4326')
        stations = pd.DataFrame([{'station_id': 'x', 'longitude': 107.5705, 'latitude': 16.4595}])
        result = station_terrain(path, stations, [100]).iloc[0]
        self.assertEqual(result.point_m, 210)
        self.assertGreater(result.n_pixels, 0)

    def test_outside_and_missing_coordinates(self):
        path = self.raster('geo.tif', np.ones((20, 20)), from_origin(107.56, 16.47, .001, .001), 'EPSG:4326')
        stations = pd.DataFrame([
            {'station_id': 'out', 'longitude': 108.5, 'latitude': 17.5},
            {'station_id': 'missing', 'longitude': np.nan, 'latitude': 16.4}])
        rows = station_terrain(path, stations, [100]).set_index('station_id')
        self.assertEqual(rows.loc['out', 'status'], 'no_valid_pixels')
        self.assertEqual(rows.loc['missing', 'status'], 'missing_coordinates')

    def test_clip_keeps_source_grid_and_original(self):
        source = self.raster('raw.tif', np.full((40, 40), 5),
                             from_origin(107.5, 16.6, .01, .01), 'EPSG:4326')
        original = source.read_bytes()
        result = clip_dem(source, self.folder/'clipped.tif', box(107.6, 16.3, 107.7, 16.4), 'test')
        self.assertEqual(source.read_bytes(), original)
        self.assertEqual(result['resolution_x_native'], .01)
        with rasterio.open(self.folder/'clipped.tif') as clipped:
            self.assertEqual(clipped.tags()['vertical_datum'], 'EGM2008')
            self.assertAlmostEqual(clipped.read(1, masked=True).mean(), 5)


if __name__ == '__main__':
    unittest.main()
