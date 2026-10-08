from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def to_numpy(arr: np.ndarray | torch.Tensor | pd.Series | list) -> np.ndarray:
    """Convert tensor, Series, or list to a 1D or 2D numpy array."""
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    if isinstance(arr, pd.Series):
        return arr.to_numpy()
    return np.asarray(arr)


def rmse(observed: np.ndarray | torch.Tensor, predicted: np.ndarray | torch.Tensor) -> float:
    """Root Mean Squared Error."""
    obs = to_numpy(observed)
    pred = to_numpy(predicted)
    valid = np.isfinite(obs) & np.isfinite(pred)
    if not valid.any():
        return np.nan
    return float(np.sqrt(np.mean((obs[valid] - pred[valid]) ** 2)))


def mae(observed: np.ndarray | torch.Tensor, predicted: np.ndarray | torch.Tensor) -> float:
    """Mean Absolute Error."""
    obs = to_numpy(observed)
    pred = to_numpy(predicted)
    valid = np.isfinite(obs) & np.isfinite(pred)
    if not valid.any():
        return np.nan
    return float(np.mean(np.abs(obs[valid] - pred[valid])))


def nse(observed: np.ndarray | torch.Tensor, predicted: np.ndarray | torch.Tensor) -> float:
    r"""
    Nash-Sutcliffe Efficiency (NSE).
    Range: (-inf, 1.0]. NSE = 1 is perfect prediction; NSE = 0 is equivalent to mean benchmark.
    
    Formula:
        NSE = 1 - \sum (obs - pred)^2 / \sum (obs - mean(obs))^2
    """
    obs = to_numpy(observed)
    pred = to_numpy(predicted)
    valid = np.isfinite(obs) & np.isfinite(pred)
    if not valid.any():
        return np.nan
    obs_v = obs[valid]
    pred_v = pred[valid]
    
    denominator = np.sum((obs_v - np.mean(obs_v)) ** 2)
    if denominator == 0:
        return np.nan
    numerator = np.sum((obs_v - pred_v) ** 2)
    return float(1.0 - (numerator / denominator))


def kge(
    observed: np.ndarray | torch.Tensor,
    predicted: np.ndarray | torch.Tensor,
    return_components: bool = False,
) -> float | tuple[float, float, float, float]:
    r"""
    Kling-Gupta Efficiency (KGE, Gupta et al. 2009).
    Range: (-inf, 1.0]. KGE = 1 is perfect prediction.

    Formula:
        KGE = 1 - \sqrt{(r - 1)^2 + (\alpha - 1)^2 + (\beta - 1)^2}
    where:
        r: Pearson correlation coefficient between observed and predicted
        alpha: ratio of standard deviations (\sigma_{pred} / \sigma_{obs})
        beta: ratio of means (\mu_{pred} / \mu_{obs})
    """
    obs = to_numpy(observed)
    pred = to_numpy(predicted)
    valid = np.isfinite(obs) & np.isfinite(pred)
    if not valid.any():
        return (np.nan, np.nan, np.nan, np.nan) if return_components else np.nan

    obs_v = obs[valid]
    pred_v = pred[valid]

    # Means
    mu_obs = np.mean(obs_v)
    mu_pred = np.mean(pred_v)
    if mu_obs == 0:
        beta = np.nan
    else:
        beta = mu_pred / mu_obs

    # Standard deviations
    std_obs = np.std(obs_v, ddof=0)
    std_pred = np.std(pred_v, ddof=0)
    if std_obs == 0:
        alpha = np.nan
        r = np.nan
    else:
        alpha = std_pred / std_obs
        if std_pred == 0:
            r = 0.0
        else:
            covariance = np.mean((obs_v - mu_obs) * (pred_v - mu_pred))
            r = np.clip(covariance / (std_obs * std_pred), -1.0, 1.0)

    if np.isnan(r) or np.isnan(alpha) or np.isnan(beta):
        kge_val = np.nan
    else:
        kge_val = float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))

    if return_components:
        return kge_val, float(r), float(alpha), float(beta)
    return kge_val


def peak_metrics(
    observed: np.ndarray | torch.Tensor | pd.Series,
    predicted: np.ndarray | torch.Tensor | pd.Series,
    timestamps: pd.Series | None = None,
) -> dict[str, float | None]:
    """
    Compute peak-specific metrics during event periods:
    - peak_absolute_error: |pred_at_peak - obs_at_peak| (or difference between highest points)
    - peak_timing_error_minutes: absolute time offset between observed peak and predicted peak
    """
    obs = to_numpy(observed)
    pred = to_numpy(predicted)
    valid = np.isfinite(obs) & np.isfinite(pred)
    if not valid.any():
        return {"peak_absolute_error": None, "peak_timing_error_minutes": None}

    obs_v = obs[valid]
    pred_v = pred[valid]

    obs_peak_idx = int(np.argmax(obs_v))
    pred_peak_idx = int(np.argmax(pred_v))

    peak_abs_err = float(abs(pred_v[pred_peak_idx] - obs_v[obs_peak_idx]))

    timing_err = None
    if timestamps is not None:
        ts = pd.to_datetime(pd.Series(timestamps).to_numpy())[valid]
        timing_err = float(abs((ts[pred_peak_idx] - ts[obs_peak_idx]).total_seconds() / 60))

    return {
        "peak_absolute_error": peak_abs_err,
        "peak_timing_error_minutes": timing_err,
    }


def flood_classification_metrics(
    observed: np.ndarray | torch.Tensor | pd.Series,
    predicted: np.ndarray | torch.Tensor | pd.Series,
    threshold: float | None = None,
    percentile: float = 0.90,
) -> dict[str, float]:
    """
    Compute binary classification metrics for flood warning exceedance.
    - threshold: specific water level alarm threshold (e.g. 575.2 cm).
      If None, uses the given percentile (e.g. 90th percentile of observed).
    Returns accuracy, precision, recall (POD), f1_score, and threshold used.
    """
    obs = to_numpy(observed)
    pred = to_numpy(predicted)
    valid = np.isfinite(obs) & np.isfinite(pred)
    if not valid.any():
        return {
            "flood_accuracy": np.nan,
            "flood_precision": np.nan,
            "flood_recall": np.nan,
            "flood_f1": np.nan,
            "flood_threshold": np.nan,
        }

    obs_v = obs[valid]
    pred_v = pred[valid]

    if threshold is None:
        threshold = float(np.quantile(obs_v, percentile))

    y_true = (obs_v >= threshold).astype(int)
    y_pred = (pred_v >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    accuracy = (tp + tn) / max(1, (tp + tn + fp + fn))
    precision = tp / max(1, (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = tp / max(1, (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "flood_accuracy": float(accuracy),
        "flood_precision": float(precision),
        "flood_recall": float(recall),
        "flood_f1": float(f1),
        "flood_threshold": float(threshold),
    }


def evaluate_all(
    observed: np.ndarray | torch.Tensor | pd.Series,
    predicted: np.ndarray | torch.Tensor | pd.Series,
    timestamps: pd.Series | None = None,
    flood_threshold: float | None = None,
    flood_percentile: float = 0.90,
) -> dict[str, float | None]:
    """Calculate the full hydrological metric suite for observed vs predicted series."""
    obs = to_numpy(observed)
    pred = to_numpy(predicted)
    
    kge_score, r, alpha, beta = kge(obs, pred, return_components=True)
    peaks = peak_metrics(obs, pred, timestamps=timestamps)
    flood = flood_classification_metrics(
        obs, pred, threshold=flood_threshold, percentile=flood_percentile
    )

    return {
        "rmse": rmse(obs, pred),
        "mae": mae(obs, pred),
        "nse": nse(obs, pred),
        "kge": kge_score,
        "kge_r": r,
        "kge_alpha": alpha,
        "kge_beta": beta,
        "peak_absolute_error": peaks["peak_absolute_error"],
        "peak_timing_error_minutes": peaks["peak_timing_error_minutes"],
        "flood_accuracy": flood["flood_accuracy"],
        "flood_precision": flood["flood_precision"],
        "flood_recall": flood["flood_recall"],
        "flood_f1": flood["flood_f1"],
        "flood_threshold": flood["flood_threshold"],
    }

