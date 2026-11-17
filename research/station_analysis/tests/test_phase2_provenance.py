"""Dataset forecast ancestry and archived value reconstruction tests."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from research.station_analysis.phase2.provenance import audit_dataset_provenance
from research.station_analysis.tests import test_phase2_forecasts as forecast_fixtures


class DatasetForecastProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        fixture = forecast_fixtures.ForecastIntegrityTests()
        fixture.setUp()
        self.init, self.fixture = fixture.init, fixture
        self.runs = pd.DataFrame([fixture.run_row()])
        self.points = fixture.points()
        self.anchor = self.init+pd.Timedelta(hours=6)
        self.config = fixture.config

    def tearDown(self):
        self.temporary.cleanup()

    def dataset(self, name='dataset', *, run_id='r1', rain_0_3=3, omit_weather24=False,
                retrieved=None, basis='assumed_6h_lag', kind='archive_50r1_unverified'):
        directory = self.root/name
        directory.mkdir()
        columns = {'timestamp_utc': [self.anchor], 'weather_forecast_age_hours': [6],
                   'weather_forecast_rain_0_3h': [rain_0_3], 'weather_forecast_rain_3_6h': [3],
                   'weather_forecast_rain_6_12h': [6],
                   'weather_forecast_rain_0_3h_missing': [0],
                   'weather_forecast_rain_3_6h_missing': [0],
                   'weather_forecast_rain_6_12h_missing': [0]}
        if not omit_weather24:
            columns['weather_forecast_rain_12_24h'] = [12]
            columns['weather_forecast_rain_12_24h_missing'] = [0]
        pd.DataFrame(columns).to_csv(directory/'hourly.csv', index=False)
        pd.DataFrame({'timestamp_utc': [self.anchor]}).to_csv(directory/'train.csv', index=False)
        provenance = pd.DataFrame({
            'timestamp_utc': [self.anchor], 'run_id': [run_id], 'initialized_at': [self.init],
            'source_retrieved_at': [retrieved if retrieved is not None else self.init+pd.Timedelta(days=20)],
            'age_hours': [6], 'was_filled': [False], 'archive_kind': [kind],
            'availability_basis': [basis],
        })
        provenance.to_csv(directory/'weather_provenance.csv.gz', index=False, compression='gzip')
        return directory

    def audit(self, directory, *, runs=None, points=True):
        return audit_dataset_provenance([directory], self.runs if runs is None else runs,
                                        self.config, self.points if points else None)

    def test_archive_values_match_without_proving_historical_availability(self):
        result = self.audit(self.dataset())
        row = result['dataset_forecast_rows'].iloc[0]
        self.assertTrue(row.feature_values_match_db_vintage)
        self.assertEqual(row.compared_forecast_values, 4)
        self.assertTrue(row.source_retrieved_after_anchor)
        self.assertTrue(row.assumed_availability)
        self.assertFalse(row.historical_availability_verified)
        self.assertEqual(set(result['dataset_forecast_integrity'].split), {'all_hourly', 'train'})

    def test_reconstruction_uses_recorded_vintage_not_latest_revision(self):
        earlier = self.init+pd.Timedelta(hours=5)
        later = self.init+pd.Timedelta(hours=8)
        runs = pd.DataFrame([
            self.fixture.run_row('r1', retrieved=earlier, available=earlier,
                                 kind='operational_single_run', basis='observed_retrieval'),
            self.fixture.run_row('revision', retrieved=later, available=later,
                                 kind='operational_single_run', basis='observed_retrieval')])
        points = pd.concat([self.points, self.fixture.points('revision', value=9)], ignore_index=True)
        directory = self.dataset(retrieved=earlier, kind='operational_single_run', basis='observed_retrieval')
        result = audit_dataset_provenance([directory], runs, self.config, points)
        row = result['dataset_forecast_rows'].iloc[0]
        self.assertTrue(row.historical_availability_verified)
        self.assertTrue(row.feature_values_match_db_vintage)

    def test_unknown_run_fails_closed_and_has_no_reconstruction_claim(self):
        row = self.audit(self.dataset(run_id='unknown'))['dataset_forecast_rows'].iloc[0]
        self.assertFalse(row.source_run_found)
        self.assertFalse(row.historical_availability_verified)
        self.assertFalse(row.feature_values_match_db_vintage)

    def test_weather12_ablation_only_compares_present_feature_windows(self):
        row = self.audit(self.dataset(omit_weather24=True))['dataset_forecast_rows'].iloc[0]
        self.assertEqual(row.compared_forecast_values, 3)
        self.assertTrue(row.feature_values_match_db_vintage)

    def test_mismatched_value_and_mask_are_reported(self):
        directory = self.dataset(rain_0_3=99)
        hourly = pd.read_csv(directory/'hourly.csv')
        hourly['weather_forecast_rain_3_6h_missing'] = 1
        hourly.to_csv(directory/'hourly.csv', index=False)
        row = self.audit(directory)['dataset_forecast_rows'].iloc[0]
        self.assertEqual(row.mismatched_forecast_values, 1)
        self.assertFalse(row.feature_values_match_db_vintage)
        self.assertFalse(row.feature_masks_match_values)

    def test_missing_provenance_file_is_not_replaced_by_assumed_metadata(self):
        directory = self.dataset()
        (directory/'weather_provenance.csv.gz').unlink()
        row = self.audit(directory)['dataset_forecast_rows'].iloc[0]
        self.assertFalse(row.provenance_present)
        self.assertFalse(row.historical_availability_verified)
        self.assertFalse(row.dataset_age_consistent)

    def test_duplicate_provenance_anchor_cannot_pass(self):
        directory = self.dataset()
        provenance = pd.read_csv(directory/'weather_provenance.csv.gz')
        pd.concat([provenance, provenance], ignore_index=True).to_csv(
            directory/'weather_provenance.csv.gz', index=False, compression='gzip')
        row = self.audit(directory)['dataset_forecast_rows'].iloc[0]
        self.assertTrue(row.duplicate_provenance_anchor)
        self.assertFalse(row.feature_values_match_db_vintage)
        self.assertFalse(row.historical_availability_verified)

    def test_unknown_timestamp_age_and_invalid_fallback_do_not_pass(self):
        directory = self.dataset()
        provenance = pd.read_csv(directory/'weather_provenance.csv.gz')
        provenance['initialized_at'] = None
        provenance['age_hours'] = np.nan
        provenance['was_filled'] = 'maybe'
        provenance.to_csv(directory/'weather_provenance.csv.gz', index=False, compression='gzip')
        row = self.audit(directory)['dataset_forecast_rows'].iloc[0]
        self.assertFalse(row.provenance_age_consistent)
        self.assertFalse(row.fallback_flag_valid)

    def test_no_points_does_not_claim_value_reconstruction(self):
        row = self.audit(self.dataset(), points=False)['dataset_forecast_rows'].iloc[0]
        self.assertEqual(row.compared_forecast_values, 0)
        self.assertFalse(row.feature_values_match_db_vintage)

    def test_empty_dataset_selection_keeps_exportable_table_schema(self):
        result = audit_dataset_provenance([], self.runs, self.config, self.points)
        self.assertTrue(result['dataset_forecast_rows'].empty)
        self.assertIn('historical_availability_verified', result['dataset_forecast_rows'])
        self.assertIn('anchor_count', result['dataset_forecast_integrity'])


if __name__ == '__main__':
    unittest.main()
