"""Audit saved alignment, source vintages, masks and training-only statistics."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .builder import sha, write_json
from .preprocessing import check_hourly, context_rows, moments, calendar_features
from .reader import RegionalDataset


def validate_dataset(path, *, verify_sources=True):
    path = Path(path)
    manifest = json.loads((path/'manifest.json').read_text())
    config = json.loads((path/'config.json').read_text())
    schema = json.loads((path/'feature_schema.json').read_text())
    mapping = json.loads((path/'station_mapping.json').read_text())
    stats = json.loads((path/'normalizers.json').read_text())
    samples = pd.read_parquet(path/'samples.parquet')
    water = pd.read_parquet(path/'water.parquet').set_index('timestamp_utc')
    rain = pd.read_parquet(path/'rainfall.parquet').set_index('timestamp_utc')
    standard = pd.read_parquet(path/'rainfall_hourly_standard.parquet').set_index('timestamp_utc')
    strict = pd.read_parquet(path/'rainfall_hourly_strict.parquet').set_index('timestamp_utc')
    weather = pd.read_parquet(path/'weather.parquet').set_index('timestamp_utc')
    calendar = pd.read_parquet(path/'calendar.parquet').set_index('timestamp_utc')
    check_hourly(water.index)
    for frame in [rain, standard, strict, weather, calendar]:
        assert frame.index.equals(water.index), 'Hourly alignment differs'
    np.testing.assert_array_equal(calendar.flood_season, water.index.tz_convert('Asia/Ho_Chi_Minh').month.isin([9, 10, 11, 12]))
    pd.testing.assert_frame_equal(calendar, calendar_features(water.index), check_freq=False)
    assert not samples.duplicated(['station_id', 'anchor_utc']).any()
    assert sorted(s['station_index'] for s in mapping.values()) == list(range(len(mapping)))
    assert len(samples) == manifest['sample_count']
    assert schema['rain_aliases']['hong_thai'] == '750062'
    assert all(sid in schema['rain_groups'] for sid in ['750007', '750041', '682816', '750062'])
    for sid, part in samples.groupby('station_id'):
        series = water[sid]
        positions = part.anchor_row.to_numpy(dtype=int)
        assert (part.station_index == mapping[sid]['station_index']).all()
        assert water.index[positions].equals(pd.DatetimeIndex(part.anchor_utc))
        assert (positions-part.history_start_row.to_numpy() == manifest['sequence_length']-1).all()
        assert (part.history_start_row >= 0).all()
        np.testing.assert_array_equal(series.iloc[positions], part.water_now_mm)
        for horizon in [6, 12]:
            times = pd.DatetimeIndex(part.anchor_utc)+pd.Timedelta(hours=horizon)
            assert times.equals(pd.DatetimeIndex(part[f'target_h{horizon}_utc']))
            expected = series.reindex(times).to_numpy()
            assert np.isfinite(expected).all()
            np.testing.assert_array_equal(expected, part[f'target_h{horizon}_mm'])
            np.testing.assert_allclose(part[f'delta_h{horizon}_mm'], expected-part.water_now_mm, atol=1e-9)
        coverage = series.notna().rolling(manifest['sequence_length']).mean().iloc[positions].to_numpy()
        np.testing.assert_array_equal(coverage, part.history_coverage)
        assert (coverage >= config['minimum_water_history_fraction']).all()
    for col in schema['rain_value_columns']+schema['weather_value_columns']:
        frame = rain if col in rain else weather
        np.testing.assert_array_equal(frame[col+'_missing'].astype(bool), frame[col].isna())
        if col != 'weather_forecast_age_hours':
            assert (frame[col].dropna() >= 0).all()
    for sid in standard:
        for window in config['rain_windows']:
            expected = standard[sid].rolling(window, min_periods=window).sum()
            np.testing.assert_allclose(rain[f'rain_{sid}_sum_{window}h'], expected, equal_nan=True)
        np.testing.assert_array_equal(rain[f'rain_{sid}_hour_estimated'].astype(bool),
                                      standard[sid].notna() & strict[sid].isna())
    valid = weather.run_id.notna()
    assert weather.loc[valid, 'grid_id'].eq(manifest['weather_grid_id']).all()
    assert weather.loc[valid, 'cell_selection'].eq('land').all()
    assert weather.loc[valid, 'model'].eq('ecmwf_ifs').all()
    age = (weather.index.to_series()-weather.initialized_at)/pd.Timedelta(hours=1)
    np.testing.assert_allclose(age, weather.weather_forecast_age_hours, equal_nan=True)
    assert weather.loc[valid, 'weather_forecast_age_hours'].between(6, 24).all()
    known = weather.available_at.notna() & valid
    assert (weather.loc[known, 'available_at'] <= weather.index[known]).all()
    assert (weather.loc[valid, 'weather_forecast_observed_availability']
            +weather.loc[valid, 'weather_forecast_assumed_availability']).eq(1).all()
    assert not (weather.weather_forecast_observed_availability.astype(bool)
                & weather.weather_forecast_hindcast.astype(bool)).any()
    for fold in config['folds']:
        selected = samples[samples[fold['id']+'_train']]
        val = samples[samples[fold['id']+'_validation']]
        boundary = pd.Timestamp(fold['validation_start'])
        assert (selected.anchor_utc+pd.Timedelta(hours=config['purge_hours']) < boundary).all()
        assert (selected.target_h12_utc < boundary).all()
        assert (val.anchor_utc >= boundary).all()
        assert (val.anchor_utc+pd.Timedelta(hours=config['purge_hours']) < pd.Timestamp(fold['validation_end'])).all()
        assert not (samples[fold['id']+'_train'] & samples[fold['id']+'_validation']).any()
        for sid, part in selected.groupby('station_id'):
            rows = context_rows(part.anchor_row, manifest['sequence_length'], len(water))
            st = stats[fold['id']]['stations'][sid]
            assert st['water'] == moments(water[sid].iloc[rows]), 'Water scaler differs from train-only fit'
            for z in [6, 12]:
                target = st['targets'][f'h{z}']
                assert target == moments(part[f'target_h{z}_mm'])
                normalized_current = (part.water_now_mm.to_numpy()-st['water']['mean'])/st['water']['scale']
                persistence = normalized_current*st['persistence_scale'][[6, 12].index(z)]+st['persistence_offset'][[6, 12].index(z)]
                np.testing.assert_allclose(persistence, (part.water_now_mm-target['mean'])/target['scale'], atol=1e-10)
        fit_rows = context_rows(np.unique(selected.anchor_row), manifest['sequence_length'], len(water))
        anchor_rows = np.unique(selected.anchor_row)
        for frame, columns, rows in [(rain, schema['rain_value_columns'], fit_rows),
                                     (weather, schema['weather_value_columns'], anchor_rows)]:
            for col in columns:
                assert stats[fold['id']]['shared'][col] == moments(frame[col].iloc[rows])
    source_forecasts_checked = 0
    if verify_sources:
        for name, expected_hash in manifest['source_sha256'].items():
            assert sha(name) == expected_hash, f'Source changed: {name}'
        # Independent reconstruction of every forecast window from one run's hourly values.
        candidates = [name for name in manifest['source_sha256'] if name.endswith('/points.parquet')]
        if candidates:
            points = pd.read_parquet(candidates[0])
            points = points[points.run_id.isin(weather.loc[valid, 'run_id']) & points.quality_status.eq('forecast')]
            assert not points.duplicated(['run_id', 'valid_at']).any()
            values = points.set_index(['run_id', 'valid_at']).precipitation_mm
            selected = weather.loc[valid]
            for a, b in [(0, 3), (3, 6), (6, 12), (12, 24)]:
                expected = np.zeros(len(selected))
                for hours in range(a+1, b+1):
                    idx = pd.MultiIndex.from_arrays([selected.run_id, selected.index+pd.Timedelta(hours=hours)])
                    expected += values.reindex(idx).to_numpy()
                assert np.isfinite(expected).all()
                np.testing.assert_allclose(selected[f'weather_forecast_rain_{a}_{b}h'], expected, atol=1e-8)
            source_forecasts_checked = len(selected)
    # Feature toggles must never change population or station identity.
    views = [RegionalDataset(path, rain_stations=choice, use_weather_forecast=forecast, split='all')
             for choice, forecast in [('nham+bach_ma', False), ('nham+bach_ma+ho_hoa_my+hong_thai', True)]]
    assert views[0].samples.sample_id.equals(views[1].samples.sample_id)
    time_sizes = {}
    for mode in ['none', 'hour', 'day_of_year', 'all']:
        view = RegionalDataset(path, split='all', time_features=mode)
        assert view.samples.sample_id.equals(views[1].samples.sample_id)
        time_sizes[mode] = len(view.feature_columns)
        for position in [0, len(view)-1]:
            item = view[position]
            for col in view.calendar_columns:
                loc = view.feature_columns.index(col)
                row = view.samples.iloc[position]
                expected = calendar[col].iloc[int(row.history_start_row):int(row.anchor_row)+1]
                np.testing.assert_allclose(item['x'][:, loc], expected, atol=1e-7)
    second = views[1]
    for sid in mapping:
        for i in second.samples.index[second.samples.station_id.eq(sid)][::max(1, len(second.samples)//100)]:
            item = second[i]
            assert item['x'].shape == (manifest['sequence_length'], len(second.feature_columns))
            assert item['y'].shape == (2,)
            assert np.isfinite(item['x']).all() and np.isfinite(item['y']).all()
            current = item['x'][-1, 0]*item['persistence_scale']+item['persistence_offset']
            normalizer = second.normalizers['stations'][sid]
            expected = [(item['water_now_mm']-normalizer['targets'][f'h{z}']['mean'])/normalizer['targets'][f'h{z}']['scale'] for z in [6, 12]]
            np.testing.assert_allclose(current, expected, atol=2e-5)
    shared_positions = [second.feature_columns.index(c) for c in second.shared_columns]
    # Samples at identical anchors must have byte-identical rainfall/weather channels.
    for _, group in second.samples.groupby('anchor_row'):
        if len(group) < 2:
            continue
        positions = group.index.to_numpy()
        if int(group.iloc[0].anchor_row) % 168 != 0:
            continue
        reference = second[int(positions[0])]['x'][:, shared_positions]
        for position in positions[1:]:
            np.testing.assert_array_equal(reference, second[int(position)]['x'][:, shared_positions])
    result = dict(status='PASS', samples=len(samples), stations=len(mapping), rain_stations=len(standard.columns),
        exact_targets_checked=len(samples)*2, forecast_anchors_reconstructed=source_forecasts_checked,
        source_hashes_checked=len(manifest['source_sha256']) if verify_sources else 0,
        checks=['station boundary', 'hourly continuity', 'exact timestamp targets', 'no target interpolation',
            'shared feature identity', 'feature toggle preserves anchors', 'local-month flood season',
            'optional local-hour and leap-year-aware day-of-year channels', 'rain window and estimated masks',
            'single-vintage forecast windows', 'forecast availability and provenance', 'chronological purge',
            'all fold scalers fit on train only', 'station-specific persistence coefficients', 'finite reader output'])
    write_json(path/'validation.json', result)
    (path/'BUILDING.json').unlink(missing_ok=True)
    manifest['artifact_sha256'] = {str(p.relative_to(path)): sha(p) for p in sorted(path.rglob('*'))
                                   if p.is_file() and p.name != 'manifest.json'}
    write_json(path/'manifest.json', manifest)
    return result
