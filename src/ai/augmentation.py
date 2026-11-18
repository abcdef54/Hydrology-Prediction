"""Training-only online data augmentation for single-station regional residual datasets.

SurgeScaleAugmenter (``--augment-surge-scale KMIN:KMAX:PROB``): each training sample, with probability PROB, gets a
"bigger / smaller event" version of itself, redrawn every time the sample is loaded (a new k every epoch):

    level(τ) = tide(τ) + r(τ)               tide = the dataset's tide_now (7-day harmonic fit), r = non-tidal part
    r'(τ)    = r̄ + k (r(τ) - r̄)             r̄ = mean of r over the sample's history window, k ~ U[KMIN, KMAX]
    a(τ)     = r'(τ) - r(τ) = (k - 1)(r(τ) - r̄)

a is added to the level history (channel water_level_depth) and to every target (t + h), so the label stays consistent
with the inputs and the astronomical tide is unchanged. Channels derived from the station level follow exactly:

* obs_<name>_d<k> (neighbour change over k h): the neighbour is assumed to receive the same offset a (a river surge
  raises both gauges alike), so obs_<name>_minus_self is unchanged and obs_<name>_d<k> gets a(τ) - a(τ-k).
* flow_dvres += a, flow_dvres_d<k> += a(τ) - a(τ-k), flow_lp_dv += mean25(a) - a, flow_lp_dv_d<k> += change of
  mean25(a); flow_grad*, flow_lp_grad* are unchanged (the offset cancels between the two gauges).
* Exogenous inputs (rain, ECMWF, sea, wind, tide, tide phase, calendar) are kept as they are.
* Channels that would need a refit (lin_*, selffc_*, neighbour surge obs_*_surge*) are refused.

r is linearly interpolated over missing hours to compute the offsets of neighbour differences; channels whose raw
value is missing (masked to 0 by the reader) are never modified. The transform runs in the train loader's collate
function only, so validation, test and the per-epoch diagnostics (which build their own loaders) see original data.
"""
import re

import numpy as np
import torch
from torch.utils.data import get_worker_info

from src.ai.regional_training_data import RegionalBatch


def parse_surge_scale(text):
    try:
        k_min, k_max, prob = (float(v) for v in text.split(':'))
    except ValueError:
        raise ValueError(f'--augment-surge-scale expects KMIN:KMAX:PROB, got {text!r}') from None
    if not (0 < k_min <= k_max) or not (0 <= prob <= 1):
        raise ValueError('--augment-surge-scale needs 0 < KMIN <= KMAX and 0 <= PROB <= 1')
    return k_min, k_max, prob


class SurgeScaleAugmenter:
    """Collate function for the TRAIN loader (see module docstring)."""

    LOWPASS_HOURS = 25

    def __init__(self, dataset, k_min, k_max, prob, seed):
        self.dataset, self.k_min, self.k_max, self.prob, self.seed = dataset, float(k_min), float(k_max), float(prob), int(seed)
        self._rng, self._rng_owner = None, None
        view = dataset.view
        if len(view.mapping) != 1:
            raise ValueError('--augment-surge-scale supports single-station datasets only')
        sid = next(iter(view.mapping))
        if 'tide_now' not in view.rain:
            raise ValueError('--augment-surge-scale needs the tide group (tide_now) in the dataset')
        level = view.water[sid].to_numpy(float)
        tide = view.rain['tide_now'].to_numpy(float)
        r = level - tide
        self.r = r                                                     # raw non-tidal part (NaN where level missing)
        idx = np.arange(len(r))
        ok = np.isfinite(r)
        self.R = np.interp(idx, idx[ok], r[ok])                        # gap-filled, for neighbour / low-pass offsets
        csum = np.concatenate([[0.], np.cumsum(self.R)])
        lo = np.clip(idx - self.LOWPASS_HOURS + 1, 0, None)
        self.lpR = (csum[idx + 1] - csum[lo]) / (idx + 1 - lo)         # trailing 25 h mean of R
        stat = view.normalizers['stations'][sid]['water']
        self.water_scale = float(stat['scale'])
        shared = view.normalizers['shared']
        cols = dataset.feature_cols
        if cols[0] != 'water_level_depth':
            raise ValueError('expected water_level_depth as the first input channel')
        # (feature index, kind, lag, column scale, raw-present mask on the grid)
        self.channels = []
        for i, c in enumerate(cols):
            if c.startswith(('lin_', 'selffc_')) or re.fullmatch(r'obs_\w+_surge(_d\d+)?', c):
                raise ValueError(f'--augment-surge-scale cannot keep channel {c} consistent (needs a refit); '
                                 f'leave its group out of --rain-stations')
            m = re.fullmatch(r'obs_\w+_d(\d+)', c)
            kind = lag = None
            if m:
                kind, lag = 'diff', int(m.group(1))
            elif c == 'flow_dvres':
                kind, lag = 'level', 0
            elif (m := re.fullmatch(r'flow_dvres_d(\d+)', c)):
                kind, lag = 'diff', int(m.group(1))
            elif c == 'flow_lp_dv':
                kind, lag = 'lp_minus_level', 0
            elif (m := re.fullmatch(r'flow_lp_dv_d(\d+)', c)):
                kind, lag = 'lp_diff', int(m.group(1))
            if kind:
                present = np.isfinite(view.rain[c].to_numpy(float))
                self.channels.append((i, kind, lag, float(shared[c]['scale']), present))
        self.horizons = list(dataset.target_horizons)

    def _generator(self):
        info = get_worker_info()
        owner = (info.id if info else -1)
        if self._rng is None or self._rng_owner != owner:
            self._rng = np.random.default_rng([self.seed, owner + 1, 977])
            self._rng_owner = owner
        return self._rng

    def offsets(self, positions, k):
        """Per-sample offsets a(τ) on the history rows (n, L) and at the target hours (n, H), mm. k: (n,)."""
        d = self.dataset
        starts, ends = d._starts[positions], d._ends[positions]
        rows = starts[:, None] + d._history_offsets                         # (n, L)
        rbar = np.array([np.nanmean(self.r[s:e]) for s, e in zip(starts, ends)])
        c = (k - 1.)[:, None]
        hist = c * (self.R[rows] - rbar[:, None])
        anchor = ends - 1
        targets = np.stack([anchor + h for h in self.horizons], axis=1)
        tgt = c * (self.r[targets] - rbar[:, None])                          # targets exist -> r finite
        return rows, hist, tgt, c, rbar

    def __call__(self, batch):
        if not isinstance(batch, RegionalBatch) or batch.positions is None:
            raise TypeError('SurgeScaleAugmenter expects RegionalBatch with positions (RegionalTrainingDataset.__getitems__)')
        out = dict(batch.tensors)
        positions = batch.positions
        rng = self._generator()
        chosen = rng.random(len(positions)) < self.prob
        k = rng.uniform(self.k_min, self.k_max, size=len(positions))
        if not chosen.any():
            return out
        sel = np.flatnonzero(chosen)
        pos, k = positions[sel], k[sel]
        rows, hist, tgt, c, _ = self.offsets(pos, k)
        x = out['x'].clone()
        sel_t = torch.from_numpy(sel)
        level_present = (x[sel_t, :, 1] == 0).numpy()
        x[sel_t, :, 0] += torch.from_numpy((hist / self.water_scale * level_present).astype(np.float32))
        for i, kind, lag, scale, present in self.channels:
            back = np.clip(rows - lag, 0, None)
            if kind == 'level':
                delta = hist
            elif kind == 'diff':
                delta = c * (self.R[rows] - self.R[back])
            elif kind == 'lp_minus_level':
                delta = c * (self.lpR[rows] - self.R[rows])
            else:                                                            # lp_diff
                delta = c * (self.lpR[rows] - self.lpR[back])
            x[sel_t, :, i] += torch.from_numpy((delta / scale * present[rows]).astype(np.float32))
        out['x'] = x
        y = out['y'].clone()
        y[sel_t] += torch.from_numpy(tgt.astype(np.float32)) / out['target_scale'][sel_t]
        out['y'] = y
        return out
