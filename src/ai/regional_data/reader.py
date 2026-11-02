"""Read selectable numpy sequences; training integration is intentionally separate."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.ai.rain_stations import resolve_rain_stations


class RegionalDataset:
    """One sample belongs to one station; shared rain/weather are joined by time."""

    def __init__(self, path, *, rain_stations='nham+bach_ma+ho_hoa_my+hong_thai',
                 use_weather_forecast=True, time_features='none', split='train', fold=None):
        self.path = Path(path)
        self.manifest = json.loads((self.path/'manifest.json').read_text())
        if self.manifest.get('status') != 'complete':
            raise ValueError('Dataset is incomplete')
        self.schema = json.loads((self.path/'feature_schema.json').read_text())
        self.mapping = json.loads((self.path/'station_mapping.json').read_text())
        self.fold = fold or self.manifest['default_fold']
        self.normalizers = json.loads((self.path/'normalizers.json').read_text())[self.fold]
        self.rain_ids = ([] if rain_stations in ('none', '') else
            resolve_rain_stations(rain_stations, list(self.schema['rain_groups']), self.schema['rain_aliases']))
        if not isinstance(use_weather_forecast, bool):
            raise ValueError('use_weather_forecast must be a boolean')
        self.use_weather = use_weather_forecast
        if time_features not in ('none', 'hour', 'day_of_year', 'all'):
            raise ValueError('time_features must be none, hour, day_of_year, or all')
        self.time_features = time_features
        self.calendar_columns = ['flood_season']
        if time_features in ('hour', 'all'):
            self.calendar_columns.extend(self.schema['calendar_groups']['hour'])
        if time_features in ('day_of_year', 'all'):
            self.calendar_columns.extend(self.schema['calendar_groups']['day_of_year'])
        self.feature_columns = list(self.schema['water_columns'])
        self.feature_columns.extend(self.calendar_columns)
        for sid in self.rain_ids:
            self.feature_columns.extend(self.schema['rain_groups'][sid])
        if self.use_weather:
            self.feature_columns.extend(self.schema['weather_columns'])
        self.samples = pd.read_parquet(self.path/'samples.parquet')
        if split in ('train', 'validation'):
            mask = self.samples[self.fold+'_'+split]
        elif split in ('all', 'test_exploratory', 'purged'):
            mask = np.ones(len(self.samples), dtype=bool) if split == 'all' else self.samples.split.eq(split)
        else:
            raise ValueError('Use train, validation, test_exploratory, purged, or all')
        self.samples = self.samples.loc[mask].reset_index(drop=True)
        self.water = pd.read_parquet(self.path/'water.parquet').set_index('timestamp_utc')
        self.rain = pd.read_parquet(self.path/'rainfall.parquet').set_index('timestamp_utc')
        self.weather = pd.read_parquet(self.path/'weather.parquet').set_index('timestamp_utc')
        self.calendar = pd.read_parquet(self.path/'calendar.parquet').set_index('timestamp_utc')
        self.length = self.manifest['sequence_length']
        # Transform shared channels once, identically for every target station.
        rain_columns = [c for sid in self.rain_ids for c in self.schema['rain_groups'][sid]]
        weather_columns = self.schema['weather_columns'] if self.use_weather else []
        self.shared_columns = self.calendar_columns+rain_columns+weather_columns
        frame = pd.concat([self.calendar[self.calendar_columns], self.rain[rain_columns], self.weather[weather_columns]], axis=1)
        for col in self.shared_columns:
            if col in self.normalizers['shared']:
                stat = self.normalizers['shared'][col]
                frame[col] = (frame[col]-stat['mean'])/stat['scale']
        self.shared = np.nan_to_num(frame.to_numpy(dtype=np.float32), nan=0.)
        self.water_values = {}
        for sid in self.mapping:
            if sid not in self.normalizers['stations']:
                continue
            raw = self.water[sid].to_numpy(dtype=float)
            stat = self.normalizers['stations'][sid]['water']
            self.water_values[sid] = np.column_stack([
                np.nan_to_num((raw-stat['mean'])/stat['scale'], nan=0.), ~np.isfinite(raw)]).astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        row = self.samples.iloc[item]
        station = self.normalizers['stations'].get(row.station_id)
        if station is None:
            raise ValueError(f'No training normalization for station {row.station_id} in fold {self.fold}')
        start, end = int(row.history_start_row), int(row.anchor_row)+1
        x = np.concatenate([self.water_values[row.station_id][start:end], self.shared[start:end]], axis=1).copy()
        if self.use_weather:
            # Preserve anchor_only: historical rows cannot reveal later forecast vintages.
            for col in self.schema['weather_columns']:
                i = self.feature_columns.index(col)
                x[:-1, i] = 1. if col.endswith('_missing') else 0.
        horizons = getattr(self, 'target_horizons', [6, 12])
        target_mm = np.array([row[f'target_h{h}_mm'] for h in horizons], dtype=np.float32)
        y = np.array([(target_mm[i]-station['targets'][f'h{h}']['mean'])/station['targets'][f'h{h}']['scale']
                      for i, h in enumerate(horizons)], dtype=np.float32)
        return dict(x=x, y=y, station_index=np.int64(row.station_index), station_id=row.station_id,
            anchor_utc=row.anchor_utc, target_mm=target_mm,
            delta_mm=target_mm-np.float32(row.water_now_mm), water_now_mm=np.float32(row.water_now_mm),
            persistence_scale=np.asarray(station['persistence_scale'], dtype=np.float32),
            persistence_offset=np.asarray(station['persistence_offset'], dtype=np.float32))
