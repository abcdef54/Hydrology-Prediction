"""Stage 2A: Rainfall Exploratory Data Analysis.

Implements exploratory data analysis of 55 rain gauges in Thua Thien Hue:
- Intensity, frequency, percentiles, extremes, and missingness.
- Multi-scale rolling accumulation distributions (W in {1, 3, 6, 12, 24, 48} hours).
- Seasonal analysis (flood season: Sep-Dec vs dry season: Jan-Aug).
- Spatial correlation matrix (55x55) and distance-decay relationship.
- Hierarchical agglomerative clustering into hydrological rainfall zones.
- Independent rainfall event extraction and characterization.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

LOG = logging.getLogger(__name__)

WINDOWS = (1, 3, 6, 12, 24, 48)


def compute_station_rainfall_summary(
    pivot_standard: pd.DataFrame,
    rain_stations: pd.DataFrame,
    dem_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute per-station rainfall statistics over all available anchors."""
    total_hours = len(pivot_standard)
    records = []
    
    # Merge station metadata with DEM elevation
    meta = rain_stations[['station_id', 'name', 'latitude', 'longitude']].copy()
    meta['station_id'] = meta['station_id'].astype(str)
    if 'elevation_m' in dem_df.columns:
        dem_map = dem_df.set_index(dem_df['station_id'].astype(str))['elevation_m'].to_dict()
        meta['elevation_m'] = meta['station_id'].map(dem_map)
    else:
        meta['elevation_m'] = np.nan
        
    meta_dict = meta.set_index('station_id').to_dict('index')

    for sid in pivot_standard.columns:
        s = pivot_standard[sid]
        valid_s = s.dropna()
        valid_count = len(valid_s)
        missing_count = total_hours - valid_count
        completeness_pct = (valid_count / total_hours) * 100.0 if total_hours > 0 else 0.0

        total_rain = float(valid_s.sum())
        mean_all = float(valid_s.mean()) if valid_count > 0 else 0.0
        max_1h = float(valid_s.max()) if valid_count > 0 else 0.0

        rain_hours = valid_s[valid_s > 0]
        rain_count = len(rain_hours)
        rain_pct = (rain_count / valid_count) * 100.0 if valid_count > 0 else 0.0
        zero_count = len(valid_s[valid_s == 0])
        zero_pct = (zero_count / valid_count) * 100.0 if valid_count > 0 else 0.0
        mean_intensity = float(rain_hours.mean()) if rain_count > 0 else 0.0

        # Percentiles during rain hours
        if rain_count > 0:
            p50 = float(np.percentile(rain_hours, 50))
            p75 = float(np.percentile(rain_hours, 75))
            p90 = float(np.percentile(rain_hours, 90))
            p95 = float(np.percentile(rain_hours, 95))
            p99 = float(np.percentile(rain_hours, 99))
        else:
            p50 = p75 = p90 = p95 = p99 = 0.0

        # Heavy rain counts
        ge_10 = int((valid_s >= 10.0).sum())
        ge_25 = int((valid_s >= 25.0).sum())
        ge_50 = int((valid_s >= 50.0).sum())

        s_meta = meta_dict.get(sid, {})
        records.append({
            'station_id': sid,
            'name': s_meta.get('name', sid),
            'elevation_m': round(s_meta.get('elevation_m', np.nan), 1),
            'latitude': s_meta.get('latitude', np.nan),
            'longitude': s_meta.get('longitude', np.nan),
            'total_hours': total_hours,
            'valid_hours': valid_count,
            'missing_hours': missing_count,
            'completeness_pct': round(completeness_pct, 2),
            'total_rainfall_mm': round(total_rain, 1),
            'mean_hourly_mm': round(mean_all, 3),
            'mean_intensity_mm': round(mean_intensity, 2),
            'max_1h_rainfall_mm': round(max_1h, 1),
            'rain_hours': rain_count,
            'rain_hours_pct': round(rain_pct, 2),
            'zero_rain_hours': zero_count,
            'zero_rain_hours_pct': round(zero_pct, 2),
            'p50_mm': round(p50, 2),
            'p75_mm': round(p75, 2),
            'p90_mm': round(p90, 2),
            'p95_mm': round(p95, 2),
            'p99_mm': round(p99, 2),
            'hours_ge_10mm': ge_10,
            'hours_ge_25mm': ge_25,
            'hours_ge_50mm': ge_50,
        })

    df_summary = pd.DataFrame(records).sort_values('total_rainfall_mm', ascending=False).reset_index(drop=True)
    return df_summary


def compute_multi_scale_accumulation_summary(
    accum_standard: Dict[int, pd.DataFrame]
) -> pd.DataFrame:
    """Summarize multi-scale rolling accumulations (mean, max, P95, P99) per station."""
    records = []
    station_ids = accum_standard[1].columns
    
    for sid in station_ids:
        for w in WINDOWS:
            s = accum_standard[w][sid].dropna()
            count = len(s)
            if count == 0:
                continue
            mean_val = float(s.mean())
            max_val = float(s.max())
            p95_val = float(np.percentile(s, 95))
            p99_val = float(np.percentile(s, 99))
            
            records.append({
                'station_id': sid,
                'window_hours': w,
                'valid_count': count,
                'mean_mm': round(mean_val, 2),
                'max_mm': round(max_val, 1),
                'p95_mm': round(p95_val, 2),
                'p99_mm': round(p99_val, 2),
            })
            
    return pd.DataFrame(records)


def compute_seasonal_rainfall_summary(
    pivot_standard: pd.DataFrame,
    rain_stations: pd.DataFrame
) -> pd.DataFrame:
    """Analyze flood season (months 9-12) vs dry season (months 1-8) rainfall characteristics."""
    meta = rain_stations.set_index(rain_stations['station_id'].astype(str))['name'].to_dict()
    
    months = pivot_standard.index.tz_convert("Asia/Ho_Chi_Minh").month
    flood_mask = months.isin([9, 10, 11, 12])
    dry_mask = months.isin([1, 2, 3, 4, 5, 6, 7, 8])
    
    records = []
    for sid in pivot_standard.columns:
        s_flood = pivot_standard.loc[flood_mask, sid].dropna()
        s_dry = pivot_standard.loc[dry_mask, sid].dropna()
        
        flood_tot = float(s_flood.sum())
        flood_mean = float(s_flood.mean()) if len(s_flood) > 0 else 0.0
        flood_rain_pct = float((s_flood > 0).sum() / len(s_flood) * 100.0) if len(s_flood) > 0 else 0.0
        flood_max = float(s_flood.max()) if len(s_flood) > 0 else 0.0
        
        dry_tot = float(s_dry.sum())
        dry_mean = float(s_dry.mean()) if len(s_dry) > 0 else 0.0
        dry_rain_pct = float((s_dry > 0).sum() / len(s_dry) * 100.0) if len(s_dry) > 0 else 0.0
        dry_max = float(s_dry.max()) if len(s_dry) > 0 else 0.0
        
        total = flood_tot + dry_tot
        ratio = (flood_tot / total * 100.0) if total > 0 else 0.0
        
        records.append({
            'station_id': sid,
            'name': meta.get(sid, sid),
            'flood_total_mm': round(flood_tot, 1),
            'flood_mean_hourly_mm': round(flood_mean, 3),
            'flood_rain_hours_pct': round(flood_rain_pct, 1),
            'flood_max_1h_mm': round(flood_max, 1),
            'dry_total_mm': round(dry_tot, 1),
            'dry_mean_hourly_mm': round(dry_mean, 3),
            'dry_rain_hours_pct': round(dry_rain_pct, 1),
            'dry_max_1h_mm': round(dry_max, 1),
            'flood_volume_ratio_pct': round(ratio, 1),
        })
        
    return pd.DataFrame(records).sort_values('flood_total_mm', ascending=False).reset_index(drop=True)


def compute_spatial_correlation_matrix(
    pivot_standard: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute pairwise Pearson and Spearman correlation matrices across all 55 rain stations."""
    # min_periods=100 ensures enough overlapping valid hours
    pearson_corr = pivot_standard.corr(method='pearson', min_periods=100)
    spearman_corr = pivot_standard.corr(method='spearman', min_periods=100)
    return pearson_corr, spearman_corr


def analyze_spatial_distance_decay(
    corr_matrix: pd.DataFrame,
    rain_stations: pd.DataFrame,
    metric_crs: str = 'EPSG:32648'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze rainfall correlation vs planar distance decay between rain stations."""
    import pyproj
    proj = pyproj.Transformer.from_crs('EPSG:4326', metric_crs, always_xy=True)
    
    sub = rain_stations[rain_stations['station_id'].astype(str).isin(corr_matrix.columns)].copy()
    sub['station_id'] = sub['station_id'].astype(str)
    coords = {
        row['station_id']: proj.transform(row['longitude'], row['latitude'])
        for _, row in sub.iterrows()
    }
    
    stations = list(corr_matrix.columns)
    pairs = []
    
    for i in range(len(stations)):
        sid_i = stations[i]
        if sid_i not in coords:
            continue
        xi, yi = coords[sid_i]
        for j in range(i + 1, len(stations)):
            sid_j = stations[j]
            if sid_j not in coords:
                continue
            xj, yj = coords[sid_j]
            dist_km = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2) / 1000.0
            r = corr_matrix.loc[sid_i, sid_j]
            if np.isfinite(r):
                pairs.append({
                    'station_a': sid_i,
                    'station_b': sid_j,
                    'distance_km': round(dist_km, 2),
                    'pearson_r': round(r, 4),
                })
                
    df_pairs = pd.DataFrame(pairs)
    
    # Distance bins
    bins = [0, 10, 25, 50, 75, 100, 200]
    labels = ['0-10 km', '10-25 km', '25-50 km', '50-75 km', '75-100 km', '>100 km']
    df_pairs['distance_bin'] = pd.cut(df_pairs['distance_km'], bins=bins, labels=labels)
    
    decay_summary = df_pairs.groupby('distance_bin', observed=False).agg(
        pair_count=('pearson_r', 'count'),
        mean_r=('pearson_r', 'mean'),
        median_r=('pearson_r', 'median'),
        p25_r=('pearson_r', lambda x: np.percentile(x.dropna(), 25) if len(x.dropna()) > 0 else np.nan),
        p75_r=('pearson_r', lambda x: np.percentile(x.dropna(), 75) if len(x.dropna()) > 0 else np.nan),
        min_r=('pearson_r', 'min'),
        max_r=('pearson_r', 'max'),
    ).reset_index()
    
    return df_pairs, decay_summary


def cluster_rain_stations(
    corr_matrix: pd.DataFrame,
    rain_stations: pd.DataFrame,
    dem_df: pd.DataFrame,
    n_clusters: int = 5
) -> pd.DataFrame:
    """Cluster rain stations into hydrologically cohesive zones using spatial & correlation features."""
    from sklearn.cluster import AgglomerativeClustering
    stations = list(corr_matrix.columns)
    corr_np = np.nan_to_num(corr_matrix.values, nan=0.0)
    
    meta = rain_stations.set_index(rain_stations['station_id'].astype(str))
    elev_map = dem_df.set_index(dem_df['station_id'].astype(str))['elevation_m'].to_dict() if 'elevation_m' in dem_df.columns else {}
    
    features = []
    for sid in stations:
        lat = meta.loc[sid, 'latitude'] if sid in meta.index else 16.5
        lon = meta.loc[sid, 'longitude'] if sid in meta.index else 107.5
        elev = elev_map.get(sid, 0.0)
        elev = elev if np.isfinite(elev) else 0.0
        c_profile = corr_np[corr_matrix.columns.get_loc(sid)]
        features.append(np.hstack([[lat * 10.0, lon * 10.0, np.log1p(elev)], c_profile]))
        
    X = np.array(features)
    clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = clusterer.fit_predict(X)
    
    df_clust = pd.DataFrame({
        'station_id': stations,
        'cluster_id': labels,
        'name': [meta.loc[s, 'name'] if s in meta.index else s for s in stations],
        'latitude': [meta.loc[s, 'latitude'] if s in meta.index else np.nan for s in stations],
        'longitude': [meta.loc[s, 'longitude'] if s in meta.index else np.nan for s in stations],
        'elevation_m': [round(elev_map.get(s, np.nan), 1) if np.isfinite(elev_map.get(s, np.nan)) else np.nan for s in stations],
    })
    
    # Assign clear hydrological zone names
    cluster_stats = df_clust.groupby('cluster_id').agg(
        med_elev=('elevation_m', 'median'),
        mean_lat=('latitude', 'mean'),
        mean_lon=('longitude', 'mean')
    ).to_dict('index')
    
    def name_zone(cid):
        stats = cluster_stats.get(cid, {})
        elev = stats.get('med_elev', 0)
        lat = stats.get('mean_lat', 16.5)
        lon = stats.get('mean_lon', 107.5)
        
        if elev > 200:
            return 'A Lưới Upstream Mountain (Headwaters)'
        elif lon > 107.75 and lat < 16.35:
            if elev > 30:
                return 'Bạch Mã Massif & Southern Foothills (Tả Trạch Basin)'
            else:
                return 'Southeastern Coastal & Lagoon (Phú Lộc / Chân Mây)'
        elif lat > 16.55 and lon < 107.45:
            return 'Northwest Basin (Phong Điền / Sông Ô Lâu)'
        elif lon > 107.7:
            return 'Southeastern Coastal & Lagoon (Phú Lộc / Chân Mây)'
        else:
            return 'Central Plains & Lower Sông Hương (Huế Urban)'
            
    df_clust['cluster_name'] = df_clust['cluster_id'].map(name_zone)
    return df_clust


def extract_independent_rainfall_events(
    pivot_standard: pd.DataFrame,
    basin_rain_threshold_mm: float = 1.0,
    min_separation_hours: int = 12
) -> pd.DataFrame:
    """Extract independent basin-wide rainfall events separated by at least min_separation_hours."""
    # Basin-mean hourly rainfall
    basin_mean = pivot_standard.mean(axis=1)
    
    is_rain = basin_mean >= basin_rain_threshold_mm
    events = []
    
    in_event = False
    event_start = None
    last_rain_time = None
    
    for t, flag in is_rain.items():
        if flag:
            if not in_event:
                in_event = True
                event_start = t
            last_rain_time = t
        else:
            if in_event:
                # Check separation
                gap = (t - last_rain_time).total_seconds() / 3600.0
                if gap >= min_separation_hours:
                    in_event = False
                    event_end = last_rain_time
                    events.append((event_start, event_end))
                    
    if in_event and last_rain_time is not None:
        events.append((event_start, last_rain_time))
        
    records = []
    for i, (start_t, end_t) in enumerate(events, 1):
        # Slice event
        sub = pivot_standard.loc[start_t:end_t]
        duration_h = len(sub)
        basin_sum = float(sub.mean(axis=1).sum())
        basin_max_1h = float(sub.mean(axis=1).max())
        
        # Max single station
        station_sums = sub.sum(axis=0)
        max_sid = station_sums.idxmax()
        max_station_rain = float(station_sums.max())
        
        records.append({
            'event_id': i,
            'start_utc': str(start_t),
            'end_utc': str(end_t),
            'duration_hours': duration_h,
            'basin_total_rain_mm': round(basin_sum, 1),
            'basin_peak_1h_mm': round(basin_max_1h, 2),
            'max_station_id': max_sid,
            'max_station_rain_mm': round(max_station_rain, 1),
        })
        
    df_events = pd.DataFrame(records).sort_values('basin_total_rain_mm', ascending=False).reset_index(drop=True)
    return df_events
