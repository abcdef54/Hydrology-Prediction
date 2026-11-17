"""Preserve source rasters, clip to Huế, and sample metric station neighborhoods."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
import rasterio
from rasterio.features import geometry_mask
from rasterio.mask import mask
from rasterio.windows import Window
from shapely.geometry import Point, mapping, shape
from shapely.ops import transform, unary_union

from .config import sha256, write_json

NODATA = -9999.0


def prepare_boundary(config, stations):
    """Use a supplied WGS84 boundary; download the documented boundary if absent.

    A buffered union includes station surroundings even if an old administrative
    coastline omits a harbor. The original administrative polygon is also saved.
    """
    path = Path(config['boundary_file'])
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        import requests
        response = requests.get(config['boundary_api'], timeout=60)
        response.raise_for_status()
        meta = response.json()
        response = requests.get(meta['gjDownloadURL'], timeout=120)
        response.raise_for_status()
        collection = response.json()
        candidates = [feature for feature in collection['features']
                      if feature['properties'].get('shapeName') == config['boundary_name']]
        if len(candidates) != 1:
            raise ValueError('Expected one Huế administrative boundary; supply boundary_file explicitly')
        write_json(path, {'type': 'FeatureCollection', 'features': candidates})
        write_json(path.with_suffix('.source.json'), meta)
    features = json.loads(path.read_text())['features']
    polygon = unary_union([shape(feature['geometry']) for feature in features])
    if not polygon.is_valid or polygon.is_empty:
        raise ValueError('Invalid administrative geometry')
    to_metric = Transformer.from_crs('EPSG:4326', config['metric_crs'], always_xy=True).transform
    to_geo = Transformer.from_crs(config['metric_crs'], 'EPSG:4326', always_xy=True).transform
    points = []
    for row in stations.itertuples():
        if pd.notna(row.longitude) and pd.notna(row.latitude):
            if not (-180 <= row.longitude <= 180 and -90 <= row.latitude <= 90):
                raise ValueError(f'Invalid coordinates: {row.station_id}')
            points.append(transform(to_metric, Point(row.longitude, row.latitude)))
    buffered = transform(to_metric, polygon).buffer(config['boundary_buffer_m'])
    if points:
        buffered = unary_union([buffered, *[point.buffer(config['boundary_buffer_m']) for point in points]])
    region = transform(to_geo, buffered)
    write_json(Path(config['work_dir']) / 'dem/analysis_region.geojson', {
        'type': 'FeatureCollection', 'features': [{'type': 'Feature',
        'properties': {'description': 'administrative boundary plus metric buffer and station neighborhoods',
                       'buffer_m': config['boundary_buffer_m']}, 'geometry': mapping(region)}]})
    return region


def clip_dem(source, destination, region, source_name):
    """Keep the source grid, mask pixels outside the ROI, and report partial coverage."""
    destination = Path(destination)
    if Path(source).resolve() == destination.resolve():
        raise ValueError('Processed DEM must not overwrite the original raster')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(source) as src:
        if src.crs is None or src.count != 1:
            raise ValueError(f'Expected a georeferenced single-band DEM: {source}')
        converter = Transformer.from_crs('EPSG:4326', src.crs, always_xy=True).transform
        raster_region = transform(converter, region)
        try:
            data, affine = mask(src, [mapping(raster_region)], crop=True, filled=False)
        except ValueError as exc:
            raise ValueError(f'DEM does not overlap requested region: {source_name}') from exc
        array = data.filled(NODATA).astype('float32')
        array[~np.isfinite(array)] = NODATA
        profile = src.profile.copy()
        profile.update(driver='GTiff', height=array.shape[1], width=array.shape[2],
                       transform=affine, dtype='float32', nodata=NODATA,
                       compress='deflate', tiled=True, blockxsize=256, blockysize=256)
        temp = destination.with_name(destination.stem + '.tmp.tif')
        with rasterio.open(temp, 'w', **profile) as out:
            out.write(array)
            out.update_tags(source=source_name, vertical_datum='EGM2008', vertical_units='m',
                            usage='terrain context only; not station zero or bathymetry')
        temp.replace(destination)
        # Source bounds coverage is geometric, distinct from valid raster pixels.
        from shapely.geometry import box
        covered = raster_region.intersection(box(*src.bounds)).area / raster_region.area
        valid = array[0] != NODATA
        return {'dem_source': source_name, 'source_path': str(Path(source).resolve()),
                'source_sha256': sha256(source), 'processed_path': str(destination.resolve()),
                'crs': src.crs.to_string(), 'vertical_datum': 'EGM2008 (EPSG:3855)',
                'resolution_x_native': src.res[0], 'resolution_y_native': src.res[1],
                'height': array.shape[1], 'width': array.shape[2],
                'valid_pixels': int(valid.sum()), 'roi_bounds_coverage_fraction': covered,
                'grid_unchanged': True, 'source_kind': 'bare-earth estimate' if source_name == 'FABDEM' else 'DSM',
                'min_m': float(array[0][valid].min()), 'max_m': float(array[0][valid].max())}


def station_terrain(path, stations, radii_m, metric_crs='EPSG:32648', source_name='FABDEM'):
    """Pixel-center inclusion in true metric disks, no degree-based radius approximation.

    The neighborhood contains water/bridge/urban pixels too. Metrics do not claim
    surveyed sensor elevation. Slope is a local DEM gradient on metre coordinates.
    """
    rows = []
    with rasterio.open(path) as src:
        to_raster = Transformer.from_crs('EPSG:4326', src.crs, always_xy=True)
        to_metric = Transformer.from_crs(src.crs, metric_crs, always_xy=True)
        geo_to_metric = Transformer.from_crs('EPSG:4326', metric_crs, always_xy=True)
        metric_to_raster = Transformer.from_crs(metric_crs, src.crs, always_xy=True)
        for station in stations.itertuples():
            lon, lat = station.longitude, station.latitude
            for radius in radii_m:
                base = {'station_id': station.station_id, 'dem_source': source_name,
                        'radius_m': radius, 'vertical_datum': 'EGM2008',
                        'method': 'pixel centers inside UTM metric circle; nodata excluded'}
                if pd.isna(lon) or pd.isna(lat):
                    rows.append({**base, 'status': 'missing_coordinates', 'n_pixels': 0})
                    continue
                mx, my = geo_to_metric.transform(lon, lat)
                bounds = metric_to_raster.transform_bounds(mx-radius, my-radius, mx+radius, my+radius)
                raw_window = rasterio.windows.from_bounds(*bounds, transform=src.transform)
                left = int(np.floor(raw_window.col_off))-1
                top = int(np.floor(raw_window.row_off))-1
                window = Window(left, top, int(np.ceil(raw_window.width))+3, int(np.ceil(raw_window.height))+3)
                data = src.read(1, window=window, boundless=True, masked=True).astype(float).filled(np.nan)
                aff = src.window_transform(window)
                rr, cc = np.indices(data.shape)
                xx = aff.c + aff.a*(cc+.5) + aff.b*(rr+.5)
                yy = aff.f + aff.d*(cc+.5) + aff.e*(rr+.5)
                px, py = to_metric.transform(xx, yy)
                inside = (px-mx)**2 + (py-my)**2 <= radius**2
                finite = np.isfinite(data)
                values = data[inside & finite]
                center_x, center_y = to_raster.transform(lon, lat)
                point = next(src.sample([(center_x, center_y)], masked=True))[0]
                point_value = float(point) if not np.ma.is_masked(point) and np.isfinite(point) else np.nan
                dx = float(np.nanmedian(np.abs(np.diff(px, axis=1))))
                dy = float(np.nanmedian(np.abs(np.diff(py, axis=0))))
                gy, gx = np.gradient(data, max(dy, .001), max(dx, .001))
                slope = np.degrees(np.arctan(np.sqrt(gx**2+gy**2)))
                valid_slopes = slope[inside & np.isfinite(slope)]
                output = {**base, 'status': 'ok' if len(values) else 'no_valid_pixels',
                          'n_pixels': int(len(values)), 'disk_pixels': int(inside.sum()),
                          'valid_fraction': len(values)/max(int(inside.sum()), 1),
                          'point_m': point_value,
                          'slope_median_deg': float(np.median(valid_slopes)) if len(valid_slopes) else np.nan}
                if len(values):
                    output.update(min_m=float(values.min()), max_m=float(values.max()),
                                  mean_m=float(values.mean()), std_m=float(values.std()),
                                  median_m=float(np.median(values)), p10_m=float(np.percentile(values, 10)),
                                  p90_m=float(np.percentile(values, 90)),
                                  relief_p90_p10_m=float(np.percentile(values, 90)-np.percentile(values, 10)))
                rows.append(output)
    return pd.DataFrame(rows)


def process(config, stations):
    region = prepare_boundary(config, stations)
    work = Path(config['work_dir']) / 'dem'
    registry, tables = [], []
    for name, key in [('FABDEM', 'fabdem'), ('Copernicus', 'copernicus')]:
        target = work / f'hue_{name.lower()}.tif'
        registry.append(clip_dem(config[key], target, region, name))
        tables.append(station_terrain(target, stations, config['radii_m'], config['metric_crs'], name))
    return pd.DataFrame(registry), pd.concat(tables, ignore_index=True)
