"""Resample complete training sequences, balancing rare changes by event."""

import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler


class EventBalancedSampler(Sampler):
    """Keep epoch length fixed; add rise/fall draws without synthesizing data.

    Fractions apply to the whole epoch; the final shuffle mixes these draws
    across batches. Natural draws are without replacement. Each priority draw
    chooses an event uniformly, then an eligible anchor uniformly in that event.
    Future changes are training labels only, never model input features.
    """

    def __init__(self, frame, dataset, seed=42, rise_fraction=.1,
                 fall_fraction=.1, threshold_mm=200., event_gap_hours=12.):
        fractions = (rise_fraction, fall_fraction)
        if (not all(math.isfinite(x) and 0 <= x < 1 for x in fractions)
                or sum(fractions) >= 1):
            raise ValueError("Priority fractions must be nonnegative with sum <1")
        if not all(math.isfinite(x) and x > 0 for x in (threshold_mm, event_gap_hours)):
            raise ValueError("Change threshold and event gap must be positive")
        if not 0 <= seed < 2**32:
            raise ValueError("Seed must be in [0, 2**32)")
        self.size = len(dataset)
        if self.size == 0:
            raise ValueError("Cannot sample an empty training dataset")
        rows = frame.iloc[dataset.end_indices]
        times = pd.DatetimeIndex(pd.to_datetime(rows.timestamp_utc, utc=True))
        if not times.is_unique or not times.is_monotonic_increasing:
            raise ValueError("Training anchors must be unique and chronological")
        targets = [f"target_water_level_plus_{h * 60}m" for h in (12, 24)]
        if not set(targets).issubset(dataset.target_cols):
            raise ValueError("Event sampling requires +12h and +24h targets")
        changes = rows[targets].to_numpy(dtype=float) - rows.water_level_depth.to_numpy()[:, None]
        if not np.isfinite(changes).all():
            raise ValueError("Eligible training changes must be finite")
        dominant = changes[np.arange(self.size), np.abs(changes).argmax(axis=1)]
        rare = np.abs(dominant) > threshold_mm
        self.groups = {
            "rise": np.flatnonzero(dominant > threshold_mm),
            "fall": np.flatnonzero(dominant < -threshold_mm),
        }
        self.counts = {
            "rise": int(self.size * rise_fraction),
            "fall": int(self.size * fall_fraction),
        }
        self.counts["natural"] = self.size - sum(self.counts.values())
        rare_positions = np.flatnonzero(rare)
        event_ids = np.full(self.size, -1, dtype=int)
        if len(rare_positions):
            breaks = times[rare_positions].to_series().diff().gt(pd.Timedelta(hours=event_gap_hours))
            event_ids[rare_positions] = breaks.cumsum().to_numpy()
        self.weights = {}
        self.summary = {
            "eligible_anchors": self.size, "threshold_mm": threshold_mm,
            "event_gap_hours": event_gap_hours, "rare_anchors": int(rare.sum()),
            "dominant_horizon_policy": "largest absolute true change at +12/+24h; ties use +12h",
            "draws_per_epoch": dict(self.counts),
        }
        for name, positions in self.groups.items():
            if self.counts[name] and not len(positions):
                raise ValueError(f"No {name} anchors exceed the training threshold")
            ids = event_ids[positions]
            _, inverse, sizes = np.unique(ids, return_inverse=True, return_counts=True)
            self.weights[name] = torch.as_tensor(1. / sizes[inverse], dtype=torch.float64)
            self.summary[name] = {"anchors": len(positions), "events": len(sizes)}
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return self.size

    def set_epoch(self, epoch):
        if epoch < 0:
            raise ValueError("Epoch must be nonnegative")
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        natural = torch.randperm(self.size, generator=generator)[:self.counts["natural"]]
        draws = [natural]
        for name in ("rise", "fall"):
            if self.counts[name]:
                selected = torch.multinomial(
                    self.weights[name], self.counts[name], replacement=True, generator=generator,
                )
                draws.append(torch.as_tensor(self.groups[name])[selected])
        indices = torch.cat(draws)
        yield from indices[torch.randperm(self.size, generator=generator)].tolist()
