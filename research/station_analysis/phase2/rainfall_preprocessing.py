"""Preprocess rainfall observations under confirmed semantics and missing data policy.

Data Contract:
- Observation: 10-minute accumulated rainfall in mm for the interval ending at observed_at.
- Hourly window: (t-1h, t], containing readings at t-50m, t-40m, t-30m, t-20m, t-10m, and t.
- Missing data policy:
    * 6/6 observations: sum(6) -> Quality: 'complete'
    * 5/6 observations: sum(5) * 6 / 5 -> Quality: 'estimated'
    * 1-4/6 observations: mark as missing (NaN) -> Quality: 'missing'
    * 0/6 observations: mark as missing (NaN) -> Quality: 'missing'
- Strict 6/6 baseline: only hours with 6/6 are valid, all others NaN.
- Multi-scale rolling accumulation: W in {1, 3, 6, 12, 24, 48} hours with min_periods = W.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pyproj
import rasterio

from research.station_analysis.config import ROOT

LOG = logging.getLogger(__name__)

WINDOWS = (1, 3, 6, 12, 24, 48)


def extract_rain_station_dem(
    rain_stations: pd.DataFrame,
    fabdem_path: Path = ROOT / 'data/processed/station_analysis/dem/hue_fabdem.tif',
    copernicus_path: Path = ROOT / 'data/processed/station_analysis/dem/hue_copernicus.tif'
) -> pd.DataFrame:
    """Extract terrain elevation from local processed DEM GeoTIFF rasters."""
    df = rain_stations[['station_id', 'name', 'latitude', 'longitude']].copy()
    coords = [(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])]

    if fabdem_path.exists():
        with rasterio.open(fabdem_path) as src:
            df['elevation_fabdem_m'] = [float(val[0]) for val in src.sample(coords)]
    else:
        df['elevation_fabdem_m'] = np.nan

    if copernicus_path.exists():
        with rasterio.open(copernicus_path) as src:
            vals = [float(val[0]) for val in src.sample(coords)]
            df['elevation_copernicus_m'] = [v if v > -9000 else np.nan for v in vals]
    else:
        df['elevation_copernicus_m'] = np.nan

    df['elevation_m'] = df['elevation_fabdem_m'].fillna(df['elevation_copernicus_m'])
    return df


def calculate_station_distance_matrix(
    rain_stations: pd.DataFrame,
    water_stations: pd.DataFrame,
    metric_crs: str = 'EPSG:32648'
) -> pd.DataFrame:
    """Calculate planar distance in kilometers between rain stations and water stations."""
    proj = pyproj.Transformer.from_crs('EPSG:4326', metric_crs, always_xy=True)

    rain_x, rain_y = proj.transform(rain_stations['longitude'].values, rain_stations['latitude'].values)
    water_x, water_y = proj.transform(water_stations['longitude'].values, water_stations['latitude'].values)

    dist_matrix = np.zeros((len(water_stations), len(rain_stations)), dtype=float)
    for i, (wx, wy) in enumerate(zip(water_x, water_y)):
        dx = rain_x - wx
        dy = rain_y - wy
        dist_matrix[i, :] = np.sqrt(dx * dx + dy * dy) / 1000.0  # in km

    df_dist = pd.DataFrame(
        dist_matrix,
        index=water_stations['station_id'].astype(str),
        columns=rain_stations['station_id'].astype(str)
    )
    df_dist.index.name = 'water_station_id'
    return df_dist


def preprocess_hourly_rainfall(
    rain_obs: pd.DataFrame,
    rain_stations: pd.DataFrame,
    anchors: pd.DatetimeIndex
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply confirmed rainfall semantics and missing data policy.

    Returns:
    - pivot_standard: hourly rainfall in mm (complete + estimated 5/6) [anchors x 55 stations]
    - pivot_strict: hourly rainfall in mm (only 6/6 complete, else NaN) [anchors x 55 stations]
    - pivot_quality: string quality flag ('complete', 'estimated', 'missing')
    - df_hourly_records: unpivoted detailed records with counts, sums, flags
    """
    station_ids = sorted(rain_stations['station_id'].astype(str).unique().tolist())

    # Filter valid observations: finite, observed quality, depth >= 0
    obs = rain_obs[rain_obs['station_id'].isin(station_ids)].copy()
    obs['observed_at'] = pd.to_datetime(obs['observed_at'], utc=True)
    depth_numeric = pd.to_numeric(obs['depth'], errors='coerce')
    valid_mask = (
        np.isfinite(depth_numeric)
        & obs['quality_status'].eq('observed')
        & (depth_numeric >= 0)
    )
    obs_valid = obs[valid_mask].copy()
    obs_valid['depth_mm'] = depth_numeric[valid_mask].astype(float)
    obs_valid['hour'] = obs_valid['observed_at'].dt.ceil('h')

    # Restrict to requested anchors
    obs_valid = obs_valid[obs_valid['hour'].isin(anchors)]

    # Aggregate by (station_id, hour)
    grouped = obs_valid.groupby(['station_id', 'hour']).agg(
        obs_count=('depth_mm', 'count'),
        obs_sum_mm=('depth_mm', 'sum')
    )

    full_idx = pd.MultiIndex.from_product([station_ids, anchors], names=['station_id', 'timestamp_utc'])
    df_records = pd.DataFrame(index=full_idx)
    df_records = df_records.join(grouped.rename_axis(index={'hour': 'timestamp_utc'}), how='left')

    df_records['obs_count'] = df_records['obs_count'].fillna(0).astype(int)
    df_records['obs_sum_mm'] = df_records['obs_sum_mm'].fillna(0.0).astype(float)

    # Vectorized policy
    c = df_records['obs_count']
    s = df_records['obs_sum_mm']

    is_complete = (c == 6)
    is_estimated = (c == 5)
    is_missing = (c < 5)

    # Standard policy: 6/6 -> s; 5/6 -> s * 1.2; <5 -> NaN
    df_records['hourly_rainfall_mm'] = np.where(
        is_complete, s,
        np.where(is_estimated, s * 1.2, np.nan)
    )

    # Strict policy: only 6/6 -> s; else NaN
    df_records['strict_hourly_rainfall_mm'] = np.where(is_complete, s, np.nan)

    df_records['quality'] = np.where(
        is_complete, 'complete',
        np.where(is_estimated, 'estimated', 'missing')
    )
    df_records['coverage_fraction'] = np.minimum(c / 6.0, 1.0)

    df_records = df_records.reset_index()

    # Pivot tables
    pivot_standard = df_records.pivot(index='timestamp_utc', columns='station_id', values='hourly_rainfall_mm').reindex(anchors)
    pivot_strict = df_records.pivot(index='timestamp_utc', columns='station_id', values='strict_hourly_rainfall_mm').reindex(anchors)
    pivot_quality = df_records.pivot(index='timestamp_utc', columns='station_id', values='quality').reindex(anchors)

    return pivot_standard, pivot_strict, pivot_quality, df_records


def compute_multi_scale_accumulations(
    pivot_standard: pd.DataFrame,
    pivot_strict: pd.DataFrame,
    windows: Tuple[int, ...] = WINDOWS
) -> Tuple[Dict[int, pd.DataFrame], Dict[int, pd.DataFrame]]:
    """Compute rolling past rainfall accumulations for multiple window sizes.

    For each window W in {1, 3, 6, 12, 24, 48} hours, computes:
    sum_{k=0}^{W-1} rainfall(t - k) with min_periods = W.
    If ANY hour in the past W hours is missing, the window is NaN.
    """
    accum_standard = {}
    accum_strict = {}

    for w in windows:
        accum_standard[w] = pivot_standard.rolling(w, min_periods=w).sum()
        accum_strict[w] = pivot_strict.rolling(w, min_periods=w).sum()

    return accum_standard, accum_strict


def preprocess_canonical_water_levels(
    avail_df: pd.DataFrame,
    water_stations: pd.DataFrame,
    anchors: pd.DatetimeIndex
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Extract canonical water levels in (t-1h, t] and compute forecast target changes.

    Target changes for anchor t:
    - Delta H +1h:  H(t + 1h) - H(t)
    - Delta H +3h:  H(t + 3h) - H(t)
    - Delta H +6h:  H(t + 6h) - H(t)
    - Delta H +12h: H(t + 12h) - H(t)
    - Delta H +24h: H(t + 24h) - H(t)

    Historical changes at anchor t:
    - Delta H -1h:  H(t) - H(t - 1h)
    - Delta H -3h:  H(t) - H(t - 3h)
    - Delta H -6h:  H(t) - H(t - 6h)
    """
    water_ids = sorted(water_stations['station_id'].astype(str).unique().tolist())
    part = avail_df[avail_df['station_id'].isin(water_ids)].copy()
    part['timestamp_utc'] = pd.to_datetime(part['timestamp_utc'], utc=True)

    pivot_h = part.pivot(index='timestamp_utc', columns='station_id', values='water_mean_mm').reindex(anchors)

    targets = {}
    for sid in water_ids:
        h = pivot_h[sid]
        df_target = pd.DataFrame(index=anchors)
        df_target['water_level_mm'] = h

        # Future targets
        df_target['dh_p1h'] = h.shift(-1) - h
        df_target['dh_p3h'] = h.shift(-3) - h
        df_target['dh_p6h'] = h.shift(-6) - h
        df_target['dh_p12h'] = h.shift(-12) - h
        df_target['dh_p24h'] = h.shift(-24) - h

        # Historical changes
        df_target['dh_m1h'] = h - h.shift(1)
        df_target['dh_m3h'] = h - h.shift(3)
        df_target['dh_m6h'] = h - h.shift(6)

        # Rapid rise flags
        df_target['rapid_rise_12h_ge200'] = df_target['dh_p12h'].ge(200)
        df_target['rapid_rise_6h_ge100'] = df_target['dh_p6h'].ge(100)

        targets[sid] = df_target

    return pivot_h, targets
