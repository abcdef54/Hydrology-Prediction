"""Train-only rainfall screening with target-boundary purging and shared anchors."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

WINDOWS = (1, 3, 6, 12, 24, 48)
DEFAULT_LAGS = tuple(range(0, 73, 2))  # 0 to 72 hours, 2-hour grid



def _pairwise_correlation(frame, target):
    """Pairwise-complete Pearson r; constant/absent series return NaN quietly."""
    x = frame.to_numpy(dtype=float)
    y = target.reindex(frame.index).to_numpy(dtype=float)[:, None]
    valid = np.isfinite(x) & np.isfinite(y)
    count = valid.sum(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        xm = np.where(valid, x, 0).sum(axis=0) / count
        ym = np.where(valid, y, 0).sum(axis=0) / count
        xc = np.where(valid, x - xm, 0)
        yc = np.where(valid, y - ym, 0)
        corr = (xc * yc).sum(axis=0) / np.sqrt((xc * xc).sum(axis=0) * (yc * yc).sum(axis=0))
    corr[count < 2] = np.nan
    return pd.Series(corr, index=frame.columns)


def evaluate_single_station_candidates(
    accum_standard: Dict[int, pd.DataFrame],
    targets: Dict[str, pd.DataFrame],
    water_stations: pd.DataFrame,
    rain_stations: pd.DataFrame,
    dist_matrix: pd.DataFrame,
    dem_df: pd.DataFrame,
    train_end_utc: str = '2026-08-01T00:00:00+07:00',
    windows: Tuple[int, ...] = WINDOWS,
    lags: Tuple[int, ...] = DEFAULT_LAGS
) -> pd.DataFrame:
    """Evaluate all rain station candidates strictly on the training period using vectorized correlation."""
    water_names = water_stations.set_index(water_stations['station_id'].astype(str))['name'].to_dict()
    rain_names = rain_stations.set_index(rain_stations['station_id'].astype(str))['name'].to_dict()
    elev_map = dem_df.set_index(dem_df['station_id'].astype(str))['elevation_m'].to_dict() if 'elevation_m' in dem_df.columns else {}

    train_end = pd.Timestamp(train_end_utc)
    records = []
    rain_sids = sorted(rain_stations['station_id'].astype(str).tolist())

    for wid in sorted(water_stations['station_id'].astype(str).tolist()):
        if wid not in targets:
            continue
        df_target = targets[wid]
        train_mask = df_target.index + pd.Timedelta(hours=12) <= train_end
        df_train = df_target.loc[train_mask]

        y12 = df_train['dh_p12h']
        y6 = df_train['dh_p6h']
        yh = df_train['water_level_mm']

        if len(y12.dropna()) < 100:
            continue

        w_name = water_names.get(wid, wid)

        best_c12 = pd.Series(-999.0, index=rain_sids)
        best_w12 = pd.Series(1, index=rain_sids)
        best_l12 = pd.Series(0, index=rain_sids)

        best_c6 = pd.Series(-999.0, index=rain_sids)
        best_w6 = pd.Series(1, index=rain_sids)
        best_l6 = pd.Series(0, index=rain_sids)

        best_ch = pd.Series(-999.0, index=rain_sids)
        best_wh = pd.Series(1, index=rain_sids)
        best_lh = pd.Series(0, index=rain_sids)

        for w in windows:
            if w not in accum_standard:
                continue
            rain_w = accum_standard[w]

            for l in lags:
                rain_wl = rain_w.shift(l).loc[train_mask]

                # Corrwith dh_p12h
                c12 = _pairwise_correlation(rain_wl, y12)
                imp12 = c12 > best_c12
                best_c12 = best_c12.where(~imp12, c12)
                best_w12 = best_w12.where(~imp12, w)
                best_l12 = best_l12.where(~imp12, l)

                # Corrwith dh_p6h
                c6 = _pairwise_correlation(rain_wl, y6)
                imp6 = c6 > best_c6
                best_c6 = best_c6.where(~imp6, c6)
                best_w6 = best_w6.where(~imp6, w)
                best_l6 = best_l6.where(~imp6, l)

                # Corrwith water_level_mm
                ch = _pairwise_correlation(rain_wl, yh)
                imph = ch > best_ch
                best_ch = best_ch.where(~imph, ch)
                best_wh = best_wh.where(~imph, w)
                best_lh = best_lh.where(~imph, l)

        for rsid in rain_sids:
            r_name = rain_names.get(rsid, rsid)
            dist_km = dist_matrix.loc[wid, rsid] if (wid in dist_matrix.index and rsid in dist_matrix.columns) else np.nan
            elev_m = elev_map.get(rsid, np.nan)

            records.append({
                'water_station_id': wid,
                'water_station_name': w_name,
                'rain_station_id': rsid,
                'rain_station_name': r_name,
                'distance_km': round(dist_km, 1) if np.isfinite(dist_km) else np.nan,
                'elevation_m': round(elev_m, 1) if np.isfinite(elev_m) else np.nan,
                'best_corr_dh12': round(float(best_c12[rsid]), 4) if best_c12[rsid] > -900 else np.nan,
                'best_window_dh12': int(best_w12[rsid]),
                'best_lag_dh12': int(best_l12[rsid]),
                'best_corr_dh6': round(float(best_c6[rsid]), 4) if best_c6[rsid] > -900 else np.nan,
                'best_window_dh6': int(best_w6[rsid]),
                'best_lag_dh6': int(best_l6[rsid]),
                'best_corr_h': round(float(best_ch[rsid]), 4) if best_ch[rsid] > -900 else np.nan,
                'best_window_h': int(best_wh[rsid]),
                'best_lag_h': int(best_lh[rsid]),
            })

    df_eval = pd.DataFrame(records)
    return df_eval


def build_exclusion_list(
    pivot_standard: pd.DataFrame,
    rain_stations: pd.DataFrame,
    train_end_utc: str = '2026-08-01T00:00:00+07:00',
    min_train_completeness_pct: float = 85.0,
    max_continuous_missing_hours: int = 500
) -> pd.DataFrame:
    """Identify stations that must be excluded from modeling due to quality or coverage."""
    train_end = pd.Timestamp(train_end_utc)
    train_mask = pivot_standard.index <= train_end
    p_train = pivot_standard.loc[train_mask]

    meta = rain_stations.set_index(rain_stations['station_id'].astype(str))['name'].to_dict()
    total_train_hours = len(p_train)

    records = []
    for sid in p_train.columns:
        s = p_train[sid]
        valid_cnt = s.notna().sum()
        comp_pct = valid_cnt / total_train_hours * 100.0

        # Check max continuous missing gap
        is_nan = s.isna()
        gap_lens = (~is_nan).cumsum()[is_nan].value_counts()
        max_gap = int(gap_lens.max()) if len(gap_lens) > 0 else 0

        # Check flatline / zero variance
        valid_s = s.dropna()
        variance = float(valid_s.var()) if len(valid_s) > 1 else 0.0

        reasons = []
        if comp_pct < min_train_completeness_pct:
            reasons.append(f'Low training completeness ({comp_pct:.1f}% < {min_train_completeness_pct}%)')
        if max_gap > max_continuous_missing_hours:
            reasons.append(f'Large continuous missing gap ({max_gap}h > {max_continuous_missing_hours}h)')
        if variance == 0.0:
            reasons.append('Zero variance (constant reading)')

        if reasons:
            records.append({
                'rain_station_id': sid,
                'rain_station_name': meta.get(sid, sid),
                'train_completeness_pct': round(comp_pct, 1),
                'max_missing_gap_hours': max_gap,
                'status': 'EXCLUDED',
                'reasons': '; '.join(reasons)
            })

    return pd.DataFrame(records)


def evaluate_multistation_combinations(
    accum_standard: Dict[int, pd.DataFrame],
    targets: Dict[str, pd.DataFrame],
    water_stations: pd.DataFrame,
    candidate_subsets: Dict[str, List[str]],  # label -> list of rain_sids
    water_station_id: str = '0074647403',
    train_end_utc: str = '2026-08-01T00:00:00+07:00',
    window: int = 6
) -> pd.DataFrame:
    """Evaluate multi-station combinations for correlation, redundancy, and regression skill."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error, r2_score

    train_end = pd.Timestamp(train_end_utc)
    df_t = targets[water_station_id]
    train_mask = df_t.index + pd.Timedelta(hours=12) <= train_end
    test_mask = df_t.index > train_end

    y_train = df_t.loc[train_mask, 'dh_p12h']
    y_test = df_t.loc[test_mask, 'dh_p12h']

    rain_w = accum_standard[window]

    # Every combination uses identical anchors, including the single-gauge baseline.
    required = sorted({sid for sids in candidate_subsets.values() for sid in sids})
    common = rain_w[required].notna().all(axis=1) & df_t['dh_p12h'].notna()
    train_mask = train_mask & common
    test_mask = test_mask & common
    y_train = df_t.loc[train_mask, 'dh_p12h']
    y_test = df_t.loc[test_mask, 'dh_p12h']
    records = []

    for name, sids in candidate_subsets.items():
        # Feature matrix: contemporaneous rain_w at lag 0
        X_all = rain_w[sids]
        
        # Collinearity: max pairwise correlation evaluated strictly on training period
        if len(sids) > 1:
            c_arr = X_all.loc[train_mask].corr().to_numpy(copy=True)
            np.fill_diagonal(c_arr, np.nan)
            max_collinearity = float(np.nanmax(c_arr))
            mean_collinearity = float(np.nanmean(c_arr))
        else:
            max_collinearity = 0.0
            mean_collinearity = 0.0

        # Fit simple ridge regression on train to test complementary value
        df_train_joint = pd.concat([y_train, X_all.loc[train_mask]], axis=1).dropna()
        df_test_joint = pd.concat([y_test, X_all.loc[test_mask]], axis=1).dropna()

        if len(df_train_joint) < 100 or len(df_test_joint) < 50:
            continue

        X_tr = df_train_joint.iloc[:, 1:].values
        y_tr = df_train_joint.iloc[:, 0].values
        X_te = df_test_joint.iloc[:, 1:].values
        y_te = df_test_joint.iloc[:, 0].values

        model = Ridge(alpha=1.0)
        model.fit(X_tr, y_tr)

        train_pred = model.predict(X_tr)
        test_pred = model.predict(X_te)

        train_r = float(np.corrcoef(y_tr, train_pred)[0, 1])
        test_r = float(np.corrcoef(y_te, test_pred)[0, 1])
        train_r2 = float(r2_score(y_tr, train_pred))
        test_r2 = float(r2_score(y_te, test_pred))

        records.append({
            'subset_name': name,
            'train_n': len(y_tr),
            'evaluation_n': len(y_te),
            'last_train_anchor_utc': str(df_train_joint.index.max()),
            'last_train_target_utc': str(df_train_joint.index.max() + pd.Timedelta(hours=12)),
            'first_evaluation_anchor_utc': str(df_test_joint.index.min()),
            'n_stations': len(sids),
            'station_ids': ', '.join(sids),
            'max_collinearity': round(max_collinearity, 3),
            'mean_collinearity': round(mean_collinearity, 3),
            'train_pearson_r': round(train_r, 4),
            'test_pearson_r': round(test_r, 4),
            'train_r2': round(train_r2, 4),
            'test_r2': round(test_r2, 4),
        })

    return pd.DataFrame(records).sort_values('train_pearson_r', ascending=False).reset_index(drop=True)
