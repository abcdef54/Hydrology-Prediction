"""Stage 2B: Rainfall–Water Level Relationship Analysis.

Implements deep hydrologic relationship analysis:
- Temporal lagged cross-correlations across L in [0, 72] hours and W in {1, 3, 6, 12, 24, 48} hours.
- Target variables: Delta H +6h, Delta H +12h, H(t), Delta H +1h, Delta H +3h, Delta H +24h.
- Pearson and Spearman rank correlations.
- Conditional analysis:
    * Flood season (months 9-12) vs Dry season (months 1-8).
    * Hydrograph limbs: Rising limb (dH_m3h > 0) vs Falling limb (dH_m3h < 0).
    * Rainfall intensity: Unconditional vs Rain hours vs Significant rain (>=5mm, >=10mm).
    * Water dynamics: Rapid rise (|dH_p12h| >= 200mm) vs Normal periods.
- Event Stability / Leave-One-Out (LOO) Analysis across identified flood episodes.
- Sensitivity Analysis: Standard policy (5/6 estimated) vs Strict 6/6 baseline.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

WINDOWS = (1, 3, 6, 12, 24, 48)
DEFAULT_LAGS = tuple(range(0, 73, 2))  # 0 to 72 hours, step 2


def compute_lagged_cross_correlations(
    accum_standard: Dict[int, pd.DataFrame],
    targets: Dict[str, pd.DataFrame],
    water_stations: pd.DataFrame,
    rain_stations: pd.DataFrame,
    windows: Tuple[int, ...] = (1, 3, 6, 12, 24),
    lags: Tuple[int, ...] = DEFAULT_LAGS,
    target_cols: Tuple[str, ...] = ('dh_p6h', 'dh_p12h', 'water_level_mm'),
    mask: Optional[pd.Series] = None
) -> pd.DataFrame:
    """Compute cross-correlations across lags L and accumulation windows W for all pairs.

    Zero-future-leakage guarantee:
    Rainfall at t - L represents rainfall observed at or before prediction anchor t.
    """
    water_sids = sorted(water_stations['station_id'].astype(str).tolist())
    rain_sids = sorted(rain_stations['station_id'].astype(str).tolist())
    
    water_names = water_stations.set_index(water_stations['station_id'].astype(str))['name'].to_dict()
    rain_names = rain_stations.set_index(rain_stations['station_id'].astype(str))['name'].to_dict()

    records = []
    
    for wid in water_sids:
        if wid not in targets:
            continue
        df_target = targets[wid]
        if mask is not None:
            df_target = df_target.loc[mask]

        w_name = water_names.get(wid, wid)

        for col in target_cols:
            if col not in df_target.columns:
                continue
            y = df_target[col]
            valid_y = y.dropna()
            if len(valid_y) < 50:
                continue

            for rsid in rain_sids:
                r_name = rain_names.get(rsid, rsid)
                for w in windows:
                    if w not in accum_standard or rsid not in accum_standard[w].columns:
                        continue
                    rain_series = accum_standard[w][rsid]
                    if mask is not None:
                        rain_series = rain_series.loc[mask]

                    for l in lags:
                        lagged_rain = rain_series.shift(l)
                        
                        # Pairwise finite mask
                        combined = pd.concat([y, lagged_rain], axis=1).dropna()
                        n_obs = len(combined)
                        if n_obs < 50:
                            continue

                        r_pearson = float(combined.iloc[:, 0].corr(combined.iloc[:, 1], method='pearson'))
                        
                        records.append({
                            'water_station_id': wid,
                            'water_station_name': w_name,
                            'target_variable': col,
                            'rain_station_id': rsid,
                            'rain_station_name': r_name,
                            'window_hours': w,
                            'lag_hours': l,
                            'n_obs': n_obs,
                            'pearson_r': round(r_pearson, 4),
                        })

    return pd.DataFrame(records)


def extract_flood_episodes(
    pivot_h: pd.DataFrame,
    water_targets: Dict[str, pd.DataFrame],
    reference_station_id: str = '0074647403',
    height_quantile: float = 0.90,
    dh12_threshold_mm: float = 200.0,
    min_separation_hours: int = 12
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Identify discrete flood episodes from observed water level and rapid rises."""
    if reference_station_id not in water_targets:
        return []
    
    df_ref = water_targets[reference_station_id]
    h = df_ref['water_level_mm']
    dh12 = df_ref['dh_p12h']

    h_thresh = h.quantile(height_quantile)
    is_flood = (h >= h_thresh) | (dh12.abs() >= dh12_threshold_mm)

    episodes = []
    in_ep = False
    ep_start = None
    last_t = None

    for t, flag in is_flood.items():
        if flag:
            if not in_ep:
                in_ep = True
                ep_start = t
            last_t = t
        else:
            if in_ep:
                gap = (t - last_t).total_seconds() / 3600.0
                if gap >= min_separation_hours:
                    in_ep = False
                    episodes.append((ep_start, last_t))

    if in_ep and last_t is not None:
        episodes.append((ep_start, last_t))

    return episodes


def compute_conditional_relationships(
    accum_standard: Dict[int, pd.DataFrame],
    targets: Dict[str, pd.DataFrame],
    water_stations: pd.DataFrame,
    rain_stations: pd.DataFrame,
    key_pairs: List[Tuple[str, str, int, int]],  # (water_id, rain_id, window, lag)
) -> pd.DataFrame:
    """Analyze relationships conditioned on season, limb state, and rainfall intensity."""
    water_names = water_stations.set_index(water_stations['station_id'].astype(str))['name'].to_dict()
    rain_names = rain_stations.set_index(rain_stations['station_id'].astype(str))['name'].to_dict()

    records = []

    for wid, rsid, w, l in key_pairs:
        if wid not in targets or w not in accum_standard or rsid not in accum_standard[w].columns:
            continue

        df_t = targets[wid]
        y12 = df_t['dh_p12h']
        y6 = df_t['dh_p6h']
        h = df_t['water_level_mm']
        dh_m3 = df_t['dh_m3h']

        rain = accum_standard[w][rsid].shift(l)
        anchors = df_t.index

        # Conditions
        conds = {
            'All Hours (Unconditional)': pd.Series(True, index=anchors),
            'Flood Season (Sep-Dec)': anchors.tz_convert("Asia/Ho_Chi_Minh").month.isin([9, 10, 11, 12]),
            'Dry Season (Jan-Aug)': anchors.tz_convert("Asia/Ho_Chi_Minh").month.isin([1, 2, 3, 4, 5, 6, 7, 8]),
            'Rising Limb (dH_-3h > 0)': dh_m3 > 0,
            'Falling Limb (dH_-3h < 0)': dh_m3 < 0,
            'Rain >= 1mm': rain >= 1.0,
            'Rain >= 5mm': rain >= 5.0,
            'Rain >= 10mm': rain >= 10.0,
            'Rapid Rise (|dH_+12h| >= 200mm)': y12.abs() >= 200.0,
            'Normal Periods (|dH_+12h| < 200mm)': y12.abs() < 200.0,
        }

        w_name = water_names.get(wid, wid)
        r_name = rain_names.get(rsid, rsid)

        for cond_name, mask in conds.items():
            sub_y12 = y12[mask]
            sub_y6 = y6[mask]
            sub_rain = rain[mask]

            # Pairwise correlation
            pair12 = pd.concat([sub_y12, sub_rain], axis=1).dropna()
            pair6 = pd.concat([sub_y6, sub_rain], axis=1).dropna()

            r12 = float(pair12.iloc[:, 0].corr(pair12.iloc[:, 1])) if len(pair12) >= 20 else np.nan
            r6 = float(pair6.iloc[:, 0].corr(pair6.iloc[:, 1])) if len(pair6) >= 20 else np.nan

            records.append({
                'water_station_id': wid,
                'water_station_name': w_name,
                'rain_station_id': rsid,
                'rain_station_name': r_name,
                'window_hours': w,
                'lag_hours': l,
                'condition': cond_name,
                'n_obs': len(pair12),
                'corr_dh_p12h': round(r12, 4) if np.isfinite(r12) else np.nan,
                'corr_dh_p6h': round(r6, 4) if np.isfinite(r6) else np.nan,
            })

    return pd.DataFrame(records)


def compute_leave_one_out_stability(
    accum_standard: Dict[int, pd.DataFrame],
    targets: Dict[str, pd.DataFrame],
    water_stations: pd.DataFrame,
    rain_stations: pd.DataFrame,
    episodes: List[Tuple[pd.Timestamp, pd.Timestamp]],
    candidate_pairs: List[Tuple[str, str, int, int]]  # (water_id, rain_id, window, lag)
) -> pd.DataFrame:
    """Evaluate stability of correlations when removing each flood episode in turn."""
    water_names = water_stations.set_index(water_stations['station_id'].astype(str))['name'].to_dict()
    rain_names = rain_stations.set_index(rain_stations['station_id'].astype(str))['name'].to_dict()

    records = []

    for wid, rsid, w, l in candidate_pairs:
        if wid not in targets or w not in accum_standard or rsid not in accum_standard[w].columns:
            continue

        df_t = targets[wid]
        y12 = df_t['dh_p12h']
        rain = accum_standard[w][rsid].shift(l)
        anchors = df_t.index

        base_pair = pd.concat([y12, rain], axis=1).dropna()
        base_r = float(base_pair.iloc[:, 0].corr(base_pair.iloc[:, 1]))

        loo_corrs = []
        for ep_idx, (s_ep, e_ep) in enumerate(episodes, 1):
            mask = (anchors < s_ep) | (anchors > e_ep)
            loo_pair = pd.concat([y12[mask], rain[mask]], axis=1).dropna()
            if len(loo_pair) >= 50:
                c = float(loo_pair.iloc[:, 0].corr(loo_pair.iloc[:, 1]))
                loo_corrs.append(c)

        if not loo_corrs:
            continue

        loo_mean = float(np.mean(loo_corrs))
        loo_std = float(np.std(loo_corrs))
        loo_min = float(np.min(loo_corrs))
        loo_max = float(np.max(loo_corrs))
        # Stability index: high score means robust across events
        stability_score = round(max(0.0, 1.0 - (loo_std / max(abs(loo_mean), 0.05))), 3)

        records.append({
            'water_station_id': wid,
            'water_station_name': water_names.get(wid, wid),
            'rain_station_id': rsid,
            'rain_station_name': rain_names.get(rsid, rsid),
            'window_hours': w,
            'lag_hours': l,
            'baseline_r': round(base_r, 4),
            'loo_mean_r': round(loo_mean, 4),
            'loo_std_r': round(loo_std, 4),
            'loo_min_r': round(loo_min, 4),
            'loo_max_r': round(loo_max, 4),
            'stability_score': stability_score,
            'n_episodes': len(loo_corrs),
        })

    return pd.DataFrame(records).sort_values('stability_score', ascending=False).reset_index(drop=True)


def compare_policy_sensitivity(
    accum_standard: Dict[int, pd.DataFrame],
    accum_strict: Dict[int, pd.DataFrame],
    targets: Dict[str, pd.DataFrame],
    candidate_pairs: List[Tuple[str, str, int, int]]  # (water_id, rain_id, window, lag)
) -> pd.DataFrame:
    """Compare cross-correlations and sample sizes between Standard and Strict 6/6 missing data policies."""
    records = []

    for wid, rsid, w, l in candidate_pairs:
        if wid not in targets or w not in accum_standard:
            continue

        y12 = targets[wid]['dh_p12h']
        rain_std = accum_standard[w][rsid].shift(l)
        rain_str = accum_strict[w][rsid].shift(l)

        pair_std = pd.concat([y12, rain_std], axis=1).dropna()
        pair_str = pd.concat([y12, rain_str], axis=1).dropna()

        r_std = float(pair_std.iloc[:, 0].corr(pair_std.iloc[:, 1])) if len(pair_std) >= 50 else np.nan
        r_str = float(pair_str.iloc[:, 0].corr(pair_str.iloc[:, 1])) if len(pair_str) >= 50 else np.nan

        delta_r = r_std - r_str if np.isfinite(r_std) and np.isfinite(r_str) else np.nan
        sample_diff = len(pair_std) - len(pair_str)

        records.append({
            'water_station_id': wid,
            'rain_station_id': rsid,
            'window_hours': w,
            'lag_hours': l,
            'n_obs_standard': len(pair_std),
            'n_obs_strict': len(pair_str),
            'sample_gain': sample_diff,
            'r_standard': round(r_std, 4) if np.isfinite(r_std) else np.nan,
            'r_strict': round(r_str, 4) if np.isfinite(r_str) else np.nan,
            'r_difference': round(delta_r, 5) if np.isfinite(delta_r) else np.nan,
            'common_n': len(pair_std.index.intersection(pair_str.index)),
            'common_anchor_r_difference': float(
                pair_std.loc[pair_std.index.intersection(pair_str.index)].iloc[:, 0].corr(
                    pair_std.loc[pair_std.index.intersection(pair_str.index)].iloc[:, 1])
                - pair_str.loc[pair_std.index.intersection(pair_str.index)].iloc[:, 0].corr(
                    pair_str.loc[pair_std.index.intersection(pair_str.index)].iloc[:, 1])
            ),
        })

    return pd.DataFrame(records)
