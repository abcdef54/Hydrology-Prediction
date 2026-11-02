import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.ai.regional_data.builder import build_dataset, make_rain_groups
from src.ai.regional_data.preprocessing import check_hourly, calendar_features, make_samples, fit_normalizers
from src.ai.regional_data.reader import RegionalDataset
from src.ai.regional_data.validation import validate_dataset
from research.scripts.build_regional_dataset import prepare_weather, GRID


def fixture():
    index = pd.date_range('2026-08-27', periods=260, freq='h', tz='UTC')
    level = pd.DataFrame({sid: 1000*k+np.sin(np.arange(len(index))/5)*100+np.arange(len(index))
                          for k, sid in enumerate(['001', '002', '003'])}, index=index)
    mapping = {sid: dict(station_index=k, name=sid, cluster=k) for k, sid in enumerate(level)}
    rain = pd.DataFrame({sid: np.arange(len(index)) % 7 for sid in ['750007', '750041', '682816', '750062']}, index=index, dtype=float)
    strict = rain.copy()
    strict.loc[index[120], '750062'] = np.nan
    rain.loc[index[100], '750041'] = np.nan
    strict.loc[index[100], '750041'] = np.nan
    audit, points = [], []
    for k, cycle in enumerate(pd.date_range(index.min()-pd.Timedelta(hours=6), index.max(), freq='6h')):
        rid = f'run{k}'
        audit.append(dict(location_id=GRID, run_id=rid, initialized_at=cycle,
            retrieved_at=pd.Timestamp('2026-10-27', tz='UTC'), available_at=pd.NaT,
            availability_basis='unknown_archive', provenance_class='hindcast_assumption_only',
            archive_kind='hindcast_49r1', strict_eligible_run=False, model='ecmwf_ifs',
            grid_latitude=16.274164, grid_longitude=107.264960))
        for h in range(1, 73):
            points.append(dict(run_id=rid, valid_at=cycle+pd.Timedelta(hours=h),
                               precipitation_mm=float(h), quality_status='forecast'))
    weather = prepare_weather(index, pd.DataFrame(audit), pd.DataFrame(points))
    config = dict(expected_station_count=3, sequence_length=96, minimum_water_history_fraction=.8,
                  purge_hours=24, cutoff='2026-09-05T00:00:00Z', weather_grid_id=GRID,
                  rain_windows=[1, 3, 6, 12, 24, 48],
                  folds=[dict(id='summer', validation_start='2026-09-03T00:00:00Z',
                              validation_end='2026-09-05T00:00:00Z')])
    catalog = pd.DataFrame([dict(station_id=sid, name=sid, latitude=16., longitude=107.) for sid in rain])
    return level, rain, strict, weather, mapping, config, catalog


class RegionalDataTests(unittest.TestCase):
    def test_fixed_requested_split_dates_and_purge_at_previous_split_end(self):
        index = pd.date_range('2025-11-15', '2026-10-15T14:00:00', freq='h', tz='Asia/Ho_Chi_Minh')
        level = pd.DataFrame({'001': np.arange(len(index), dtype=float)}, index=index)
        mapping = {'001': dict(station_index=0)}
        folds = [dict(id='fixed', validation_start='2026-07-31T00:00:00+07:00', validation_end='2026-09-15T00:00:00+07:00')]
        samples = make_samples(level, mapping, folds, '2026-09-15T00:00:00+07:00')
        for split, first, last in [
            ('train', None, '2026-07-29T23:00:00+07:00'),
            ('validation', '2026-07-31T00:00:00+07:00', '2026-09-13T23:00:00+07:00'),
            ('test_exploratory', '2026-09-15T00:00:00+07:00', '2026-10-15T02:00:00+07:00')]:
            times = samples.loc[samples.split.eq(split), 'anchor_utc']
            if first:
                self.assertEqual(times.min(), pd.Timestamp(first))
            self.assertEqual(times.max(), pd.Timestamp(last))

    def test_hourly_grid_rejects_gaps_and_duplicates(self):
        idx = pd.date_range('2026-01-01', periods=4, freq='h', tz='UTC')
        check_hourly(idx)
        for bad in [idx.delete(1), idx.insert(1, idx[0]), idx.tz_localize(None), idx[::-1]]:
            with self.assertRaises(ValueError):
                check_hourly(bad)

    def test_exact_targets_and_missing_history(self):
        level, _, _, _, mapping, c, _ = fixture()
        level.loc[level.index[140], '001'] = np.nan
        samples = make_samples(level, mapping, c['folds'], c['cutoff'])
        part = samples[samples.station_id.eq('001')]
        self.assertNotIn(134, part.anchor_row.to_list())  # +6 label missing
        self.assertNotIn(128, part.anchor_row.to_list())  # +12 label missing
        self.assertNotIn(140, part.anchor_row.to_list())  # H(t) missing
        self.assertIn(141, part.anchor_row.to_list())  # input missingness is allowed
        for z in [6, 12]:
            np.testing.assert_array_equal(part[f'target_h{z}_mm'], level['001'].reindex(part.anchor_utc+pd.Timedelta(hours=z)))
        level.loc[level.index[95:125], '002'] = np.nan
        changed = make_samples(level, mapping, c['folds'], c['cutoff'])
        self.assertNotIn(130, changed[changed.station_id.eq('002')].anchor_row.to_list())

    def test_purge_and_exact_split_boundary(self):
        level, _, _, _, mapping, c, _ = fixture()
        s = make_samples(level, mapping, c['folds'], c['cutoff'])
        boundary = pd.Timestamp(c['folds'][0]['validation_start'])
        self.assertTrue((s.loc[s.summer_train, 'target_h12_utc'] < boundary).all())
        at_gap = s.anchor_utc.eq(boundary-pd.Timedelta(hours=24))
        self.assertTrue(s.loc[at_gap, 'split'].eq('purged').all())
        self.assertFalse((s.summer_train & s.summer_validation).any())

    def test_calendar_uses_vietnamese_month(self):
        for when, expected in [('2026-08-31T16:00:00Z', 0), ('2026-08-31T17:00:00Z', 1),
                               ('2026-12-31T16:00:00Z', 1), ('2026-12-31T17:00:00Z', 0)]:
            self.assertEqual(int(calendar_features(pd.DatetimeIndex([when])).iloc[0, 0]), expected)

    def test_hour_cycles_use_local_time_and_wrap_at_midnight(self):
        times = pd.DatetimeIndex(['2026-08-31T16:00:00Z', '2026-08-31T17:00:00Z', '2026-08-31T23:00:00Z'])
        result = calendar_features(times)
        angles = 2*np.pi*np.array([23, 0, 6])/24
        np.testing.assert_allclose(result.hour_sin, np.sin(angles), atol=1e-12)
        np.testing.assert_allclose(result.hour_cos, np.cos(angles), atol=1e-12)

    def test_day_cycles_handle_leap_year_and_local_year_boundary(self):
        # Feb 29 local midnight exists only in leap years; Jan 1 must start at angle zero.
        times = pd.DatetimeIndex(['2024-02-28T17:00:00Z', '2024-02-29T17:00:00Z',
                                 '2025-02-28T17:00:00Z', '2024-12-31T16:00:00Z',
                                 '2024-12-31T17:00:00Z'])
        result = calendar_features(times)
        angles = 2*np.pi*np.array([59/366, 60/366, 59/365, 365/366, 0])
        np.testing.assert_allclose(result.day_sin, np.sin(angles), atol=1e-12)
        np.testing.assert_allclose(result.day_cos, np.cos(angles), atol=1e-12)

    def test_rain_windows_do_not_fill_missing_and_preserve_estimated_flag(self):
        _, rain, strict, _, _, c, _ = fixture()
        result, _, _ = make_rain_groups(rain, strict, c['rain_windows'])
        self.assertTrue(result.iloc[100]['rain_750041_sum_3h_missing'])
        self.assertTrue(result.iloc[102]['rain_750041_sum_3h_missing'])
        self.assertFalse(result.iloc[103]['rain_750041_sum_3h_missing'])
        self.assertEqual(result.iloc[120]['rain_750062_hour_estimated'], 1)
        self.assertEqual(result.iloc[121]['rain_750062_hour_estimated'], 0)

    def test_forecast_single_vintage_and_assumed_provenance(self):
        _, _, _, weather, _, _, _ = fixture()
        row = weather.iloc[0]
        self.assertEqual(row.weather_forecast_rain_0_3h, 7+8+9)
        self.assertEqual(row.weather_forecast_rain_12_24h, sum(range(19, 31)))
        self.assertEqual(row.weather_forecast_observed_availability, 0)
        self.assertEqual(row.weather_forecast_assumed_availability, 1)
        self.assertEqual(row.weather_forecast_hindcast, 1)
        self.assertEqual(row.weather_forecast_age_hours, 6)

    def test_scalers_ignore_future_values(self):
        level, rain, strict, weather, mapping, c, _ = fixture()
        samples = make_samples(level, mapping, c['folds'], c['cutoff'])
        features, _, columns = make_rain_groups(rain, strict, c['rain_windows'])
        schema = dict(rain_value_columns=columns, weather_value_columns=['weather_forecast_age_hours'])
        original = fit_normalizers(samples, level, features, weather, schema, c['folds'], 96)
        boundary = pd.Timestamp(c['folds'][0]['validation_start'])
        level.loc[level.index >= boundary] = 1e9
        features.loc[features.index >= boundary, columns] = 1e9
        weather.loc[weather.index >= boundary, 'weather_forecast_age_hours'] = 1e9
        altered = make_samples(level, mapping, c['folds'], c['cutoff'])
        updated = fit_normalizers(altered, level, features, weather, schema, c['folds'], 96)
        self.assertEqual(original, updated)

    def test_constant_training_targets_require_explicit_station_opt_in(self):
        level, rain, strict, weather, mapping, config, catalog = fixture()
        level['001'] = 0.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = dict(level=level, rain_standard=rain, rain_strict=strict, weather=weather,
                stations=mapping, clusters=pd.DataFrame({'station_id': list(mapping)}),
                source_paths=[], rain_catalog=catalog)
            with self.assertRaisesRegex(ValueError, 'No training target dynamics: 001'):
                build_dataset(root/'rejected', config=config, **args)
            allowed = dict(config, allow_constant_target_station_ids=['001'])
            build_dataset(root/'explicit', config=allowed, **args)
            manifest = json.loads((root/'explicit'/'manifest.json').read_text())
            self.assertEqual(manifest['constant_training_target_stations'], ['001'])
            self.assertEqual(validate_dataset(root/'explicit', verify_sources=False)['status'], 'PASS')

    def test_station_subsets_preserve_labels_and_reindex(self):
        from src.ai.build_station_subset_dataset import build_station_subset
        level, rain, strict, weather, mapping, config, catalog = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_dataset(root/'source', level=level, rain_standard=rain, rain_strict=strict,
                weather=weather, stations=mapping, clusters=pd.DataFrame({'station_id': list(mapping)}),
                config=config, source_paths=[], rain_catalog=catalog)
            validate_dataset(root/'source', verify_sources=False)
            for ids in (['001'], ['003', '001']):
                output = root/('subset'+str(len(ids)))
                self.assertEqual(build_station_subset(root/'source', output, ids)['stations'], len(ids))
                saved = json.loads((output/'station_mapping.json').read_text())
                self.assertEqual([saved[sid]['station_index'] for sid in ids], list(range(len(ids))))
            for ids in ([], ['001', '001'], ['999']):
                with self.assertRaises(ValueError):
                    build_station_subset(root/'source', root/'invalid', ids)
                self.assertFalse((root/'invalid').exists())
            with self.assertRaises(FileExistsError):
                build_station_subset(root/'source', root/'subset1', ['001'])

    def test_unclustered_station_requires_explicit_opt_in(self):
        level, rain, strict, weather, mapping, config, catalog = fixture()
        level['004'] = level['001']
        mapping['004'] = dict(station_index=3, name='004', cluster=-1)
        config['expected_station_count'] = 4
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = dict(level=level, rain_standard=rain, rain_strict=strict, weather=weather,
                stations=mapping, clusters=pd.DataFrame({'station_id': list(mapping)}),
                source_paths=[], rain_catalog=catalog)
            with self.assertRaises(ValueError):
                build_dataset(root/'rejected', config=config, **args)
            build_dataset(root/'explicit', config=dict(config, allow_unclustered_stations=True), **args)
            self.assertEqual(validate_dataset(root/'explicit', verify_sources=False)['status'], 'PASS')

    def test_build_read_toggle_station_boundaries_and_integrity(self):
        level, rain, strict, weather, mapping, c, catalog = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'dataset'
            kwargs = dict(level=level, rain_standard=rain, rain_strict=strict, weather=weather,
                stations=mapping, clusters=pd.DataFrame({'station_id': list(mapping)}), config=c,
                source_paths=[], rain_catalog=catalog)
            build_dataset(path, **kwargs)
            result = validate_dataset(path, verify_sources=False)
            self.assertEqual(result['status'], 'PASS')
            with self.assertRaises(FileExistsError):
                build_dataset(path, **kwargs)
            full = RegionalDataset(path, split='all')
            short = RegionalDataset(path, split='all', rain_stations='nham+bach_ma', use_weather_forecast=False)
            self.assertEqual(len(full), len(short))
            self.assertEqual(full[0]['x'].shape, (96, 68))
            self.assertEqual(short[0]['x'].shape, (96, 29))
            np.testing.assert_array_equal(full[0]['target_mm'], short[0]['target_mm'])
            for mode, width in [('none', 68), ('hour', 70), ('day_of_year', 70), ('all', 72)]:
                other = RegionalDataset(path, split='all', time_features=mode)
                self.assertEqual(other[0]['x'].shape, (96, width))
                pd.testing.assert_frame_equal(other.samples, full.samples)
                np.testing.assert_array_equal(other[0]['y'], full[0]['y'])
                self.assertIn('flood_season', other.feature_columns)
            with self.assertRaises(ValueError):
                RegionalDataset(path, time_features='weekly')
            for col in full.schema['weather_columns']:
                expected = 1 if col.endswith('_missing') else 0
                self.assertTrue((full[0]['x'][:-1, full.feature_columns.index(col)] == expected).all())
            with self.assertRaises(ValueError):
                RegionalDataset(path, rain_stations='bach_ma+750041')
            with self.assertRaises(ValueError):
                RegionalDataset(path, rain_stations='not_a_station')
            with self.assertRaises(ValueError):
                RegionalDataset(path, use_weather_forecast='false')
            # Corrupt a target; independent audit must reject it.
            samples = pd.read_parquet(path/'samples.parquet')
            samples.loc[0, 'target_h6_mm'] += 1
            samples.to_parquet(path/'samples.parquet', index=False)
            with self.assertRaises(AssertionError):
                validate_dataset(path, verify_sources=False)


if __name__ == '__main__':
    unittest.main()
