"""Cached training adapter with selectable history and train-only target scaling."""
import numpy as np
import hashlib
from collections.abc import Sequence
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, RandomSampler
from torch.utils.data import default_collate

from .regional_data import RegionalDataset
from .regional_data.training_selection import configure_view


class RegionalBatch(Sequence):
    """A gathered batch with lazy rows for ordinary DataLoader compatibility."""
    def __init__(self, tensors):
        self.tensors = tensors

    def __len__(self):
        return len(self.tensors['y'])

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index >= len(self) or index < -len(self):
            raise IndexError(index)
        return {key: value[index] for key, value in self.tensors.items()}


def regional_collate(samples):
    """Avoid creating and then stacking hundreds of individual tensor rows."""
    return samples.tensors if isinstance(samples, RegionalBatch) else default_collate(samples)


class RegionalTrainingDataset(Dataset):
    def __init__(self, path, *, split, fold=None, rain_stations='nham+bach_ma+ho_hoa_my+hong_thai',
                 use_weather_forecast=True, time_features='none', station_id=None,
                 sequence_length=None, target_horizons=None, target_normalizers=None):
        self.split = split
        self.view = RegionalDataset(path, split=split, fold=fold, rain_stations=rain_stations,
                                    use_weather_forecast=use_weather_forecast, time_features=time_features)
        selection = configure_view(self.view, sequence_length, target_horizons, target_normalizers)
        self.target_horizons = self.view.target_horizons
        if station_id is not None:
            if station_id not in self.view.mapping:
                raise ValueError(f'Unknown water station: {station_id}')
            self.view.samples = self.view.samples[self.view.samples.station_id.eq(station_id)].reset_index(drop=True)
        if self.view.samples.empty:
            raise ValueError(f'No samples for {split}, {fold}, station={station_id}')
        absent = set(self.view.samples.station_id)-set(self.view.normalizers['stations'])
        if absent:
            raise ValueError(f'No train-only normalizers for {sorted(absent)} in {fold}; select a later fold or eligible station')
        self.feature_cols = self.view.feature_columns
        self.target_cols = [f'target_water_level_plus_{h*60}m' for h in self.target_horizons]
        self.seq_len, self.num_features, self.num_targets = self.view.length, len(self.feature_cols), len(self.target_cols)
        self.rain_station_ids = self.view.rain_ids
        self.rain_station_ranking = []
        self.rain_station_combination = '+'.join(self.rain_station_ids)
        self.feature_set = 'water_level'+('+rain' if self.rain_station_ids else '')+'+flood_season'+('+weather_forecast' if use_weather_forecast else '')
        self.resolution_minutes = self.view.manifest.get('resolution_minutes', 60)
        self.feature_scaler = self.target_scaler = None
        self.timestamps = self.view.samples.anchor_utc
        self.regional_metadata = dict(dataset_path=str(self.view.path.resolve()), fold=self.view.fold,
            dataset_manifest_sha256=hashlib.sha256((self.view.path/'manifest.json').read_bytes()).hexdigest(),
            station_mapping=self.view.mapping, normalizers=self.view.normalizers,
            rain_stations=self.rain_station_ids, time_features=time_features,
            use_weather_forecast=use_weather_forecast, selected_station=station_id,
            training_selection=selection, target_horizons=self.target_horizons,
            test_status='post-cutoff exploratory; not independent final test')
        self._cache_samples()

    def _cache_samples(self):
        """Cache compact metadata, not a duplicated [samples,time,features] tensor."""
        frame = self.view.samples
        self._cached_frame = frame
        self._ids = frame.station_index.to_numpy(dtype=np.int64, copy=True)
        count = len(self.view.mapping)
        expected = frame.station_id.map({sid: value['station_index'] for sid, value in self.view.mapping.items()}).to_numpy()
        if not np.array_equal(self._ids, expected):
            raise ValueError('Station IDs do not match the saved mapping')
        if set(frame.station_id)-set(self.view.normalizers['stations']):
            raise ValueError('Station lacks training normalization')
        if len(self._ids) and (self._ids.min() < 0 or self._ids.max() >= count):
            raise ValueError('Station ID outside saved mapping')
        self._ends = frame.anchor_row.to_numpy(dtype=np.int64, copy=True)+1
        self._starts = frame.history_start_row.to_numpy(dtype=np.int64, copy=True)
        self._water = np.zeros((count, len(self.view.water), 2), dtype=np.float32)
        means, scales = np.zeros((count, self.num_targets), dtype=np.float32), np.ones((count, self.num_targets), dtype=np.float32)
        for sid, values in self.view.normalizers['stations'].items():
            i = self.view.mapping[sid]['station_index']
            self._water[i] = self.view.water_values[sid]
            means[i] = [values['targets'][f'h{h}']['mean'] for h in self.target_horizons]
            scales[i] = [values['targets'][f'h{h}']['scale'] for h in self.target_horizons]
        raw = frame[[f'target_h{h}_mm' for h in self.target_horizons]].to_numpy(dtype=np.float32)
        self._target_mean = torch.from_numpy(means[self._ids])
        self._target_scale = torch.from_numpy(scales[self._ids])
        self._targets = torch.from_numpy((raw-means[self._ids])/scales[self._ids])
        self._station_ids = torch.from_numpy(self._ids)
        self._history_offsets = np.arange(self.seq_len, dtype=np.int64)
        columns = self.view.schema['weather_columns'] if self.view.use_weather else []
        self._weather_positions = np.array([self.feature_cols.index(c) for c in columns], dtype=np.int64)
        self._weather_placeholders = np.array([1. if c.endswith('_missing') else 0. for c in columns], dtype=np.float32)
        self.station_ids_validated = True

    def _ensure_cache(self):
        # Selection utilities may replace the view's frame before iteration.
        if self.view.samples is not self._cached_frame:
            self._cache_samples()

    def _inputs(self, positions):
        times = self._starts[positions, None]+self._history_offsets
        result = np.empty((len(positions), self.seq_len, self.num_features), dtype=np.float32)
        result[:, :, :2] = self._water[self._ids[positions, None], times]
        result[:, :, 2:] = self.view.shared[times]
        if len(self._weather_positions):
            result[:, :-1, self._weather_positions] = self._weather_placeholders
        return torch.from_numpy(result)

    def __len__(self):
        return len(self.view)

    def __getitem__(self, item):
        self._ensure_cache()
        return dict(x=self._inputs(np.array([item]))[0], y=self._targets[item],
            station_ids=self._station_ids[item], target_scale=self._target_scale[item], target_mean=self._target_mean[item])

    def __getitems__(self, items):
        """Let DataLoader construct all windows in a batch with one array gather."""
        self._ensure_cache()
        positions = np.asarray(items, dtype=np.int64)
        indices = torch.from_numpy(positions)
        return RegionalBatch(dict(x=self._inputs(positions), y=self._targets[indices],
            station_ids=self._station_ids[indices], target_scale=self._target_scale[indices],
            target_mean=self._target_mean[indices]))


class RegionalShuffleSampler(Sampler):
    """Preserve the old epoch order when worker processes are kept alive.

    Previously DataLoader's worker-base-seed draw shared the shuffle generator.
    Reproduce that draw in a dedicated sampler, independent of worker restarts.
    """
    def __init__(self, dataset, seed):
        self.dataset = dataset
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        torch.empty((), dtype=torch.int64).random_(generator=self.generator)
        yield from RandomSampler(self.dataset, generator=self.generator)


def regional_loaders(path, *, batch_size, num_workers, seed, **selection):
    loaders = []
    target_normalizers = None
    for i, split in enumerate(['train', 'validation', 'test_exploratory']):
        dataset = RegionalTrainingDataset(path, split=split, target_normalizers=target_normalizers, **selection)
        if i == 0:
            target_normalizers = dataset.view.normalizers
        loaders.append(DataLoader(dataset, batch_size=batch_size,
            collate_fn=regional_collate,
            sampler=RegionalShuffleSampler(dataset, seed+i) if i == 0 else None,
            num_workers=num_workers, drop_last=False, pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0, generator=torch.Generator().manual_seed(seed+i)))
    return tuple(loaders)


def regional_persistence(dataset, regional=True):
    stats = dataset.view.normalizers['stations']
    if not regional:
        sid = dataset.view.samples.station_id.iloc[0]
        return stats[sid]['persistence_scale'], stats[sid]['persistence_offset'], 0
    count = len(dataset.view.mapping)
    scale, offset = np.ones((count, dataset.num_targets)), np.zeros((count, dataset.num_targets))
    for sid, values in stats.items():
        i = dataset.view.mapping[sid]['station_index']
        scale[i], offset[i] = values['persistence_scale'], values['persistence_offset']
    return scale.tolist(), offset.tolist(), count
