"""Cache and analysis-store safety independent from live PostgreSQL."""
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import pandas as pd

from research.station_analysis.config import load_config, sha256, write_json
from research.station_analysis.database import store_analysis_database
from research.station_analysis.pipeline import _fingerprint, _validate_prerequisites


class PipelineSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.work = Path(self.temp.name)
        self.default = Path('research/station_analysis/default.yaml')
        self.config = load_config(self.default, {'work_dir': str(self.work)})

    def tearDown(self):
        self.temp.cleanup()

    def test_extract_identity_ignores_report_and_grouping_preferences(self):
        baseline = _fingerprint('extract', self.config, self.work)
        changed = {**self.config, 'output_dir': 'different-report', 'seed': 7,
                   'grouping_end': '2026-07-01T00:00:00Z', 'radii_m': [200]}
        self.assertEqual(baseline, _fingerprint('extract', changed, self.work))
        changed['start'] = '2026-01-01T00:00:00Z'
        self.assertNotEqual(baseline, _fingerprint('extract', changed, self.work))

    def test_missing_or_modified_prerequisites_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'prerequisite'):
            _validate_prerequisites('dem', self.config, self.work)
        artifact = self.work/'snapshot.txt'
        artifact.write_text('frozen')
        write_json(self.work/'cache/extract.json', {
            'fingerprint': _fingerprint('extract', self.config, self.work),
            'outputs': {str(artifact): sha256(artifact)},
        })
        _validate_prerequisites('dem', self.config, self.work)
        artifact.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Changed or missing'):
            _validate_prerequisites('dem', self.config, self.work)

    def test_naive_or_inverted_period_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'timezone'):
            load_config(self.default, {'start': '2026-01-01'})
        with self.assertRaisesRegex(ValueError, 'precede'):
            load_config(self.default, {'start': '2026-02-01T00:00Z', 'end': '2026-01-01T00:00Z'})

    def test_sqlite_rerun_removes_stale_tables_and_preserves_ids(self):
        target = self.work/'analysis.sqlite'
        frame = pd.DataFrame({'station_id': ['0074647403'], 'value': [1]})
        store_analysis_database(target, {'old': frame, 'stations': frame})
        store_analysis_database(target, {'stations': frame})
        with sqlite3.connect(target) as conn:
            self.assertEqual(conn.execute('select station_id from stations').fetchone()[0], '0074647403')
            tables = conn.execute("select name from sqlite_master where type='table'").fetchall()
            self.assertEqual(tables, [('stations',)])

    def test_failed_sqlite_export_preserves_previous_complete_store(self):
        target = self.work/'analysis.sqlite'
        frame = pd.DataFrame({'value': [1]})
        store_analysis_database(target, {'complete': frame})
        original = target.read_bytes()
        with self.assertRaises(Exception):
            store_analysis_database(target, {'partial': frame, 'broken': pd.DataFrame()})
        self.assertEqual(target.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
