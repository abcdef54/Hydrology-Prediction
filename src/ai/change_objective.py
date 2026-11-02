"""Opt-in change-risk objectives. All balancing constants come from train only."""
import math
import torch
from .losses import HorizonWeightedMSE


class ChangeRiskLoss(HorizonWeightedMSE):
    def __init__(self, weights, counts, scales, mode, scale_floor_mm=200.,
                 event_weight=1., under_weight=1., threshold_mm=200.,
                 large_error_penalty=0.2, large_error_threshold_mm=200.,
                 large_error_transition_mm=20., rise_event_weight=0.):
        super().__init__(weights)
        if mode not in ('station-balanced', 'event-weighted', 'asymmetric', 'combined', 'large-error-count', 'large-error-primary'):
            raise ValueError('Unknown change loss')
        for value in (scale_floor_mm, threshold_mm):
            if not math.isfinite(value) or value <= 0:
                raise ValueError('Physical scales must be positive and finite')
        for value in (event_weight, under_weight):
            if not math.isfinite(value) or value < 0:
                raise ValueError('Risk weights must be nonnegative and finite')
        if not math.isfinite(large_error_penalty) or large_error_penalty < 0:
            raise ValueError('Large-error penalty must be nonnegative and finite')
        if not math.isfinite(rise_event_weight) or rise_event_weight < 0:
            raise ValueError('Rise event weight must be nonnegative and finite')
        if rise_event_weight and mode != 'station-balanced':
            raise ValueError('Rise-event weighting requires station-balanced regression')
        if any(not math.isfinite(value) or value <= 0 for value in
               (large_error_threshold_mm, large_error_transition_mm)):
            raise ValueError('Large-error threshold and transition must be positive and finite')
        counts = torch.as_tensor(counts, dtype=torch.float64)
        scales = torch.as_tensor(scales, dtype=torch.float64)
        if (counts.ndim != 1 or scales.shape != (len(counts), len(self.weights))
                or not torch.isfinite(counts).all() or (counts < 0).any()
                or not (counts > 0).any() or not torch.isfinite(scales).all() or (scales <= 0).any()):
            raise ValueError('Expected valid training counts and station/horizon scales')
        active = counts > 0
        frequency = torch.zeros_like(counts)
        frequency[active] = counts.sum() / (active.sum() * counts[active])
        self.register_buffer('frequency', frequency)
        self.register_buffer('scales', scales)
        self.mode, self.scale_floor_mm = mode, scale_floor_mm
        self.event_weight, self.under_weight, self.threshold_mm = event_weight, under_weight, threshold_mm
        self.large_error_penalty = large_error_penalty
        self.large_error_threshold_mm = large_error_threshold_mm
        self.large_error_transition_mm = large_error_transition_mm
        self.rise_event_weight = rise_event_weight

    def elements(self, predictions, targets, persistence, station_ids, *, apply_rise_weight=True):
        if predictions.ndim != 2 or predictions.shape != targets.shape or persistence.shape != targets.shape:
            raise ValueError('Expected equal [batch,horizon] tensors')
        if station_ids.shape != predictions.shape[:1] or station_ids.dtype != torch.long:
            raise ValueError('Expected integer station IDs per sample')
        if predictions.shape[0] == 0 or predictions.shape[1] != len(self.weights):
            raise ValueError('Nonempty batch and configured horizons required')
        torch._assert_async(((station_ids >= 0) & (station_ids < len(self.scales))).all(),
                            'Station ID outside loss mapping')
        scale = self.scales[station_ids].to(predictions)
        error = (predictions-targets)*scale
        change = (targets-persistence.detach())*scale
        balanced = self.mode in ('station-balanced', 'combined', 'large-error-count')
        denominator = scale.clamp_min(self.scale_floor_mm) if balanced else scale
        base = (error/denominator).square()
        event = (change.abs() > self.threshold_mm).to(predictions.dtype)
        weighted = base * (1+self.event_weight*event) if self.mode in ('event-weighted', 'combined') else base
        # Only true rises activate the asymmetric cost. Wrong-sign predictions
        # incur full shortfall; excessive positive predictions remain penalized.
        under = ((-error).clamp_min(0)/denominator).square() * (change > self.threshold_mm)
        total = weighted+self.under_weight*under if self.mode in ('asymmetric', 'combined') else weighted
        if balanced:
            total = total*self.frequency[station_ids, None].to(predictions)
        if self.rise_event_weight and apply_rise_weight:
            total = total*(1+self.rise_event_weight*(change > self.threshold_mm).to(predictions.dtype))
        if self.mode in ('large-error-count', 'large-error-primary'):
            # Approximate the per-prediction physical-error count. Unlike the
            # base term, this term gives each prediction equal station weight.
            soft_count = torch.sigmoid((error.abs()-self.large_error_threshold_mm)
                                       / self.large_error_transition_mm)
            count_loss = self.large_error_penalty*soft_count
            total = total + count_loss if self.mode == 'large-error-count' else count_loss
        return total

    def forward(self, predictions, targets, persistence, station_ids, *, apply_rise_weight=True):
        return self.reduce_per_horizon(self.elements(predictions, targets, persistence, station_ids,
                                                     apply_rise_weight=apply_rise_weight))

    def specification(self):
        spec = dict(mode=self.mode, scale_floor_mm=self.scale_floor_mm, event_weight=self.event_weight,
                    under_weight=self.under_weight, threshold_mm=self.threshold_mm,
                    counts_policy='N/(active_stations*n_station), train only',
                    station_frequency=self.frequency.tolist(), target_scales=self.scales.tolist(),
                    horizon_weights=self.weights.tolist())
        if self.mode in ('large-error-count', 'large-error-primary'):
            spec.update(large_error_penalty=self.large_error_penalty,
                        large_error_threshold_mm=self.large_error_threshold_mm,
                        large_error_transition_mm=self.large_error_transition_mm,
                        large_error_count_policy='unbalanced prediction cells, horizon-weighted',
                        base_squared_error_weight=1. if self.mode == 'large-error-count' else 0.)
        if self.rise_event_weight:
            spec['rise_event_weight'] = self.rise_event_weight
        return spec


def from_training_dataset(dataset, **options):
    """Use saved train statistics and current train membership, never val/test."""
    import numpy as np
    if dataset.split != 'train':
        raise ValueError('Loss constants must be constructed from training membership only')
    count = len(dataset.view.mapping)
    ids = dataset.view.samples.station_index.to_numpy(dtype=np.int64)
    counts = np.bincount(ids, minlength=count)
    scales = np.ones((count, dataset.num_targets))
    for sid, stats in dataset.view.normalizers['stations'].items():
        i = dataset.view.mapping[sid]['station_index']
        scales[i] = [stats['targets'][f'h{h}']['scale'] for h in getattr(dataset, 'target_horizons', (6, 12))]
    return ChangeRiskLoss(counts=counts, scales=scales, **options)
