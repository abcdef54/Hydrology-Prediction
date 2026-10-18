"""Loss functions for direct multi-horizon forecasts in normalized target units."""
from collections.abc import Sequence
import math

import torch
from torch import nn


class HorizonWeightedMSE(nn.Module):
    """Average squared error with positive, normalized weights on the last axis.

    Scaling all weights by the same constant leaves loss and gradients unchanged.
    Equal weights reproduce mean MSE. Input shapes must match exactly, avoiding
    silent broadcasting across samples or forecast horizons.
    """

    def __init__(self, weights: Sequence[float]) -> None:
        super().__init__()
        values = torch.as_tensor(weights, dtype=torch.float64)
        if values.ndim != 1 or not values.numel():
            raise ValueError('Provide one weight per forecast horizon')
        if not torch.isfinite(values).all() or not (values > 0).all():
            raise ValueError('Horizon weights must be finite and positive')
        normalized = values / values.max()
        self.register_buffer('weights', normalized / normalized.sum())

    def reduce_per_horizon(self, errors: torch.Tensor) -> torch.Tensor:
        if errors.ndim < 1 or errors.shape[-1] != len(self.weights):
            raise ValueError('Weight count must match the last (horizon) dimension')
        weights = self.weights.to(device=errors.device, dtype=errors.dtype)
        return (errors * weights).sum(dim=-1).mean()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if predictions.shape != targets.shape:
            raise ValueError('Predictions and targets must have identical shapes')
        return self.reduce_per_horizon((predictions - targets).square())


def validate_correction_penalty(
    penalty: float, loss_func: nn.Module, num_horizons: int,
) -> None:
    """Reject a negative residual reward when the MSE objective is unbounded.

    With normalized horizon weights alpha, the coefficient of prediction[h]^2
    is alpha[h] + penalty / H. All coefficients must be strictly positive.
    Equal-weight mean MSE therefore requires penalty > -1.
    Validation/checkpoint selection should use data loss without this reward.
    """
    if not math.isfinite(penalty):
        raise ValueError("Correction penalty must be finite")
    if penalty >= 0:
        return
    if num_horizons < 1:
        raise ValueError("Output horizon count must be positive")
    if isinstance(loss_func, HorizonWeightedMSE):
        if len(loss_func.weights) != num_horizons:
            raise ValueError("Horizon weight count must match model outputs")
        lower_bound = -num_horizons * loss_func.weights.min().item()
    elif isinstance(loss_func, nn.MSELoss) and loss_func.reduction == "mean":
        lower_bound = -1.0
    else:
        raise ValueError("Negative correction penalty requires mean MSE or HorizonWeightedMSE")
    if penalty <= lower_bound:
        raise ValueError(
            f"Correction penalty must be > {lower_bound:.12g} for this loss; "
            "otherwise the training objective can be unbounded below"
        )
