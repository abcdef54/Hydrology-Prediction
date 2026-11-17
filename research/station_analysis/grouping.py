"""Exploratory station grouping from training-period dynamics, not datum offsets.

Clusters describe similarity, not measured hydraulic connectivity. DEM elevation
is assessed separately and never supplied to the clustering estimator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import RobustScaler


DYNAMIC_FEATURES = [
    "change_6h_iqr_ratio", "change_12h_iqr_ratio",
    "change_6h_abs_p95_iqr_ratio", "change_12h_abs_p95_iqr_ratio",
    "autocorr_6h", "autocorr_12h", "autocorr_24h",
    "zero_fraction", "change_6h_positive_fraction", "change_12h_positive_fraction",
]
SPECTRAL_FEATURES = ["diurnal_power_fraction_proxy", "semidiurnal_power_fraction_proxy"]


def _identity(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "station_id" not in result:
        if "id" in result:
            result["station_id"] = result.id.astype(str)
        else:
            result["station_id"] = result.index.astype(str)
    result["station_id"] = result.station_id.astype(str)
    if result.station_id.duplicated().any():
        raise ValueError("Station IDs must be unique")
    return result


def _fit(matrix: pd.DataFrame, max_k: int, seed: int) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Fit only complete, variable columns; do not impute station dynamics."""
    columns = [column for column in matrix if
               np.isfinite(matrix[column]).all() and matrix[column].nunique() > 1]
    matrix = matrix[columns]
    scores = []
    if len(matrix) < 3 or not columns:
        return pd.Series(dtype="Int64"), pd.DataFrame(), pd.DataFrame(index=matrix.index)
    transformed = pd.DataFrame(
        RobustScaler().fit_transform(matrix), index=matrix.index, columns=matrix.columns
    )
    unique = len(transformed.drop_duplicates())
    best_score, best_labels = -np.inf, None
    for k in range(2, min(max_k, len(matrix) - 1, unique) + 1):
        labels = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(transformed)
        if len(set(labels)) < 2:
            continue
        score = float(silhouette_score(transformed, labels))
        scores.append({"k": k, "silhouette": score, "station_count": len(matrix)})
        if score > best_score:
            best_score, best_labels = score, labels
    if best_labels is None:
        return pd.Series(dtype="Int64"), pd.DataFrame(scores), transformed
    return pd.Series(best_labels, index=matrix.index, dtype="Int64"), pd.DataFrame(scores), transformed


def _terrain_associations(features: pd.DataFrame, terrain: pd.DataFrame,
                          radius: float) -> pd.DataFrame:
    columns = ["terrain_feature", "dynamic_feature", "radius_m", "n", "spearman_rho",
               "descriptive_p_value", "scope", "interpretation"]
    if terrain.empty or not {"dem_source", "radius_m", "station_id"} <= set(terrain.columns):
        return pd.DataFrame(columns=columns)
    selected = terrain.loc[
        terrain.dem_source.astype(str).str.lower().str.contains("fabdem")
        & np.isclose(pd.to_numeric(terrain.radius_m, errors="coerce"), radius)
    ].copy()
    if selected.empty:
        return pd.DataFrame(columns=columns)
    selected["station_id"] = selected.station_id.astype(str)
    if selected.station_id.duplicated().any():
        raise ValueError("FABDEM terrain features must be unique by station and radius")
    dynamics = features.copy()
    for hours in (6, 12):
        normalized = f"change_{hours}h_iqr_ratio"
        if normalized in dynamics and "iqr_mm" in dynamics:
            dynamics[f"change_{hours}h_iqr_mm"] = dynamics[normalized] * dynamics.iqr_mm
    joined = dynamics.merge(selected, on="station_id", how="inner")
    rows = []
    for terrain_column in ("median_m", "relief_p90_p10_m", "slope_median_deg"):
        if terrain_column not in joined:
            continue
        for dynamic_column in (
            "change_6h_iqr_ratio", "change_12h_iqr_ratio",
            "change_6h_iqr_mm", "change_12h_iqr_mm",
            "autocorr_12h", "semidiurnal_power_fraction_proxy",
        ):
            if dynamic_column not in joined:
                continue
            pairs = joined[[terrain_column, dynamic_column]].replace([np.inf, -np.inf], np.nan).dropna()
            rho, p_value = np.nan, np.nan
            if len(pairs) >= 5 and (pairs.nunique() > 1).all():
                result = spearmanr(pairs.iloc[:, 0], pairs.iloc[:, 1])
                rho, p_value = float(result.statistic), float(result.pvalue)
            rows.append({
                "terrain_feature": terrain_column, "dynamic_feature": dynamic_column,
                "radius_m": radius, "n": len(pairs), "spearman_rho": rho,
                "descriptive_p_value": p_value,
                "scope": "pre_grouping_end_dynamics",
                "interpretation": "exploratory; no multiplicity correction, no causal or predictive claim",
            })
    return pd.DataFrame(rows, columns=columns)


def group_stations(features: pd.DataFrame, metadata: pd.DataFrame,
                   terrain: pd.DataFrame, config: dict) -> dict[str, pd.DataFrame]:
    """Return auditable assignments, silhouette scores and DEM associations.

    A grouping cutoff is mandatory for fitted clusters. Undefined quantities and
    sensor datums remain metadata warnings, not normalization claims. Units must
    be the project's approved mm convention. Sparse, empty, constant or incomplete
    dynamics are excluded rather than silently imputed into a group.
    """
    features = _identity(features)
    metadata = _identity(metadata)
    merged = metadata.merge(features, on="station_id", how="left", suffixes=("", "_feature"))
    min_hours = int(config.get("min_pair_hours", 168))
    max_k = int(config.get("cluster_max_k", 5))
    seed = int(config.get("seed", 42))
    cutoff = config.get("grouping_end")
    if max_k < 2 or min_hours < 1:
        raise ValueError("Clustering requires cluster_max_k>=2 and min_pair_hours>=1")
    cutoff_stamp = pd.Timestamp(cutoff) if cutoff else None
    if cutoff_stamp is not None and cutoff_stamp.tzinfo is None:
        raise ValueError("grouping_end must include a timezone")
    status, reasons = [], []
    for _, row in merged.iterrows():
        reason = []
        if cutoff_stamp is None or row.get("scope") != "before_grouping_end":
            reason.append("training_period_cutoff_required")
        else:
            recorded = row.get("grouping_end_exclusive")
            if pd.isna(recorded) or pd.Timestamp(recorded) != cutoff_stamp:
                reason.append("grouping_cutoff_mismatch")
        unit = str(row.get("measurement_unit", "")).strip().lower()
        if unit != "mm":
            reason.append("measurement_unit_not_project_mm")
        if pd.isna(row.get("observed_hours")) or row.get("observed_hours", 0) < min_hours:
            reason.append("insufficient_observed_hours")
        if pd.isna(row.get("scale_available")) or not bool(row.get("scale_available", False)):
            reason.append("constant_or_no_robust_scale")
        required = [column for column in DYNAMIC_FEATURES if column in features]
        if not required or any(not np.isfinite(row.get(column, np.nan)) for column in required):
            reason.append("incomplete_dynamic_features")
        status.append("excluded" if reason else "eligible")
        reasons.append(";".join(reason))
    merged["status"] = status
    merged["exclusion_reason"] = reasons
    merged["cluster"] = pd.Series(pd.NA, index=merged.index, dtype="Int64")
    eligible = merged.loc[merged.status == "eligible"].set_index("station_id")
    selected_columns = [column for column in DYNAMIC_FEATURES + SPECTRAL_FEATURES if column in eligible]
    matrix = eligible[selected_columns].apply(pd.to_numeric, errors="coerce")
    labels, scores, scaled = _fit(matrix, max_k, seed)
    if len(labels):
        label_map = labels.to_dict()
        merged["cluster"] = merged.station_id.map(label_map).astype("Int64")
        merged.loc[merged.station_id.isin(labels.index), "status"] = "clustered_exploratory"
    else:
        merged.loc[merged.status == "eligible", "status"] = "eligible_insufficient_group_structure"
    scaled.index.name = "station_id"
    scaled = scaled.reset_index()
    sensitivity_rows = []
    no_spectral = matrix.drop(columns=SPECTRAL_FEATURES, errors="ignore")
    alternative, alternate_scores, _ = _fit(no_spectral, max_k, seed)
    shared = labels.index.intersection(alternative.index)
    spectral_used = [column for column in SPECTRAL_FEATURES if column in scaled]
    if len(shared):
        sensitivity_rows.append({
            "comparison": "all_selected_features_vs_without_spectral_proxies",
            "n": len(shared), "adjusted_rand_index": adjusted_rand_score(labels.loc[shared], alternative.loc[shared]),
            "selected_k": int(labels.nunique()), "without_spectral_k": int(alternative.nunique()),
            "spectral_features_used": ";".join(spectral_used),
            "status": "exploratory" if spectral_used else "identical_feature_sets_no_spectral_available",
        })
    sensitivity = pd.DataFrame(sensitivity_rows, columns=[
        "comparison", "n", "adjusted_rand_index", "selected_k", "without_spectral_k",
        "spectral_features_used", "status",
    ])
    scores = scores.reindex(columns=["k", "silhouette", "station_count"])
    if len(scores):
        scores["selected"] = scores.k.eq(int(labels.nunique()))
    else:
        scores["selected"] = pd.Series(dtype=bool)
    # Terrain associations use only eligible pre-cutoff stations, including any
    # too-small eligible set where clustering itself was not possible.
    associations = _terrain_associations(
        eligible.reset_index(), terrain,
        float(config.get("terrain_association_radius_m", 200)),
    )
    return {
        "cluster_assignments": merged,
        "cluster_scores": scores,
        "cluster_features": scaled,
        "cluster_sensitivity": sensitivity,
        "terrain_associations": associations,
    }
