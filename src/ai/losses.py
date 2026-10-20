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


class HorizonWeightedTailLoss(HorizonWeightedMSE):
    """Weighted normalized MSE + smooth physical-mm threshold penalty.

    StandardScaler inverse-transform differences are (prediction-target)*scale;
    target means cancel. Softplus gives a nonzero gradient around the threshold.
    """
    def __init__(self, weights, target_scales, strength=0.1,
                 threshold_mm=200.0, transition_mm=20.0):
        super().__init__(weights)
        scales = torch.as_tensor(target_scales, dtype=torch.float64)
        if scales.shape != self.weights.shape or not torch.isfinite(scales).all() or not (scales > 0).all():
            raise ValueError("One finite positive target scale is required per horizon")
        if not math.isfinite(strength) or strength < 0:
            raise ValueError("Tail penalty strength must be finite and nonnegative")
        if not all(math.isfinite(v) and v > 0 for v in (threshold_mm, transition_mm)):
            raise ValueError("Tail threshold and transition must be finite and positive")
        self.register_buffer('target_scales', scales)
        self.strength = float(strength)
        self.threshold_mm = float(threshold_mm)
        self.transition_mm = float(transition_mm)

    def components(self, predictions, targets):
        mse = super().forward(predictions, targets)
        scales = self.target_scales.to(device=predictions.device, dtype=predictions.dtype)
        error_mm = (predictions - targets).abs() * scales
        penalty = self.reduce_per_horizon(torch.nn.functional.softplus(
            (error_mm - self.threshold_mm) / self.transition_mm))
        return mse, penalty

    def forward(self, predictions, targets):
        if self.strength == 0:
            return super().forward(predictions, targets)
        mse, penalty = self.components(predictions, targets)
        return mse + self.strength * penalty


class CommonHorizonAuxiliaryLoss(nn.Module):
    """Keep B's five-horizon coefficients and add separately averaged auxiliaries.

    Unlike normalizing weights over all 24 outputs, the primary contribution
    is unchanged. The same reduction is used for residual regularization.
    """
    def __init__(self, target_columns, target_scales, auxiliary_weight=.25,
                 tail_strength=.5, threshold_mm=200., transition_mm=20.):
        super().__init__()
        if not math.isfinite(auxiliary_weight) or auxiliary_weight < 0:
            raise ValueError('Auxiliary weight must be finite and nonnegative')
        common=[f'target_water_level_plus_{h*60}m' for h in (1,3,6,12,24)]
        if len(set(target_columns))!=len(target_columns) or not set(common).issubset(target_columns):
            raise ValueError('Unique targets including the five common horizons are required')
        primary=[target_columns.index(c) for c in common]
        extra=[i for i in range(len(target_columns)) if i not in primary]
        scales=torch.as_tensor(target_scales,dtype=torch.float64)
        if scales.shape!=(len(target_columns),):
            raise ValueError('One target scale per output is required')
        self.register_buffer('primary_indices',torch.tensor(primary,dtype=torch.long))
        self.register_buffer('auxiliary_indices',torch.tensor(extra,dtype=torch.long))
        self.auxiliary_weight=float(auxiliary_weight)
        self.primary_loss=HorizonWeightedTailLoss([1]*len(primary),scales[primary],
            strength=tail_strength,threshold_mm=threshold_mm,transition_mm=transition_mm)
        self.auxiliary_loss=(HorizonWeightedTailLoss([1]*len(extra),scales[extra],
            strength=tail_strength,threshold_mm=threshold_mm,transition_mm=transition_mm)
            if extra else None)

    def components(self,predictions,targets):
        if predictions.ndim<1 or predictions.shape!=targets.shape or predictions.shape[-1]!=(len(self.primary_indices)+len(self.auxiliary_indices)):
            raise ValueError('Prediction and target shapes must match all configured outputs')
        primary=self.primary_loss(predictions.index_select(-1,self.primary_indices),
                                  targets.index_select(-1,self.primary_indices))
        extra=(self.auxiliary_loss(predictions.index_select(-1,self.auxiliary_indices),
                                  targets.index_select(-1,self.auxiliary_indices))
               if self.auxiliary_loss is not None else predictions.new_zeros(()))
        return primary,extra

    def forward(self,predictions,targets):
        primary,extra=self.components(predictions,targets)
        return primary+self.auxiliary_weight*extra

    def correction_loss(self,corrections):
        primary=corrections.index_select(-1,self.primary_indices).square().mean()
        extra=(corrections.index_select(-1,self.auxiliary_indices).square().mean()
               if len(self.auxiliary_indices) else corrections.new_zeros(()))
        return primary+self.auxiliary_weight*extra


class DirectionAmplitudeLoss(HorizonWeightedMSE):
    """B's data loss plus bounded direction reward, change weighting and excess risk.

    persistence is the observed anchor level expressed in each target's scaled
    units, not a learned prediction. No future observation is used as an input.
    """

    def __init__(self, weights, target_scales, direction_strength=.05,
                 amplitude_strength=.1, excess_strength=.1, neutral_mm=10.,
                 direction_scale_mm=50., reward_scale_mm=100.,
                 amplitude_scale_mm=200., excess_scale_mm=100., quartic=.25):
        super().__init__(weights)
        strengths = (direction_strength, amplitude_strength, excess_strength, quartic)
        scales = (neutral_mm, direction_scale_mm, reward_scale_mm,
                  amplitude_scale_mm, excess_scale_mm)
        if any(not math.isfinite(v) or v < 0 for v in strengths):
            raise ValueError('Strengths must be finite and nonnegative')
        if any(not math.isfinite(v) or v <= 0 for v in scales):
            raise ValueError('Physical scales must be finite and positive')
        self.base = HorizonWeightedTailLoss(weights, target_scales, strength=.5,
                                           threshold_mm=200., transition_mm=20.)
        self.direction_strength = direction_strength
        self.amplitude_strength = amplitude_strength
        self.excess_strength = excess_strength
        self.neutral_mm = neutral_mm
        self.direction_scale_mm = direction_scale_mm
        self.reward_scale_mm = reward_scale_mm
        self.amplitude_scale_mm = amplitude_scale_mm
        self.excess_scale_mm = excess_scale_mm
        self.quartic = quartic

    def components(self, predictions, targets, persistence):
        if predictions.shape != targets.shape or persistence.shape != targets.shape:
            raise ValueError('Predictions, targets and persistence must share a shape')
        scale = self.base.target_scales.to(predictions)
        error = (predictions - targets) * scale
        actual_change = (targets - persistence.detach()) * scale
        predicted_change = (predictions - persistence.detach()) * scale
        active = actual_change.abs() > self.neutral_mm
        direction_reward = .5 * (1 + torch.tanh(
            actual_change.sign() * predicted_change / self.direction_scale_mm))
        gate = torch.exp(-(error / self.reward_scale_mm).square())
        direction = self.reduce_per_horizon(active * (1 - direction_reward * gate))
        # Extra attention to strong real changes; not a duplicate unweighted MSE.
        amplitude = self.reduce_per_horizon(
            (error / self.amplitude_scale_mm).square()
            * (actual_change.abs() / self.amplitude_scale_mm).clamp(max=3))
        excess = torch.relu((error.abs() - 200.) / self.excess_scale_mm)
        excess = self.reduce_per_horizon(excess.square() + self.quartic * excess.pow(4))
        mse, soft_tail = self.base.components(predictions, targets)
        return dict(base_weighted_mse=mse, soft_tail=soft_tail,
                    direction=direction, amplitude=amplitude, excess=excess)

    def forward(self, predictions, targets, persistence):
        c = self.components(predictions, targets, persistence)
        return (c['base_weighted_mse'] + .5 * c['soft_tail']
                + self.direction_strength * c['direction']
                + self.amplitude_strength * c['amplitude']
                + self.excess_strength * c['excess'])


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
