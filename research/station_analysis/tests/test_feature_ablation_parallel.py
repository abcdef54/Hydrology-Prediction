"""Parallel orchestration must merge safely and defer only reporting, not experiments."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from research.station_analysis.phase2.feature_ablation.parallel import merge_station
from research.station_analysis.phase2.feature_ablation.reporting import finalize


class ParallelAblationTests(unittest.TestCase):
    def fixture(self,root):
        child=Path(root)/'child';station=child/'stations'/'001';station.mkdir(parents=True)
        fit=child/'fits'/'key';fit.mkdir(parents=True)
        (fit/'identity.json').write_text('{"seed":42}')
        (fit/'model.ubj').write_bytes(b'unchanged model bytes')
        (station/'ledger.json').write_text(json.dumps([{'fit_key':'key','status':'completed'}]))
        (station/'metrics.csv').write_text('rmse\n1.0\n')
        (station/'complete.json').write_text('{"station_id":"001"}')
        work=Path(root)/'canonical';(work/'fits').mkdir(parents=True)
        return station,work

    def test_station_completion_published_with_models_and_predictions(self):
        with TemporaryDirectory() as root:
            station,work=self.fixture(root);merge_station(station,work)
            self.assertEqual((work/'fits/key/model.ubj').read_bytes(),b'unchanged model bytes')
            self.assertTrue((work/'stations/001/complete.json').exists())
            self.assertTrue((work/'stations/001/metrics.csv').exists())
            merge_station(station,work)  # Idempotent coordinator polling.

    def test_collision_cannot_overwrite_a_different_model_identity(self):
        with TemporaryDirectory() as root:
            station,work=self.fixture(root);dest=work/'fits/key';dest.mkdir()
            (dest/'identity.json').write_text('{"seed":17}')
            with self.assertRaises(ValueError):merge_station(station,work)

    def test_worker_only_defers_report_not_scientific_fit_state(self):
        with TemporaryDirectory() as root,patch.dict('os.environ',{'HYDROLOGY_STAGE4_COORDINATED_WORKER':'1'}):
            marker=finalize(Path(root),Path(root)/'unused',{})
            result=json.loads(marker.read_text())
            self.assertEqual(result['status'],'fitting_complete')
            self.assertEqual(result['reporting'],'deferred_to_coordinator')
            self.assertFalse((Path(root)/'unused').exists())
