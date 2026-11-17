"""Retained-target checks and Phase 2 stage/cache safety."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import pandas as pd

from research.station_analysis.phase2.artifacts import audit_datasets
from research.station_analysis.phase2.config import load_config
from research.station_analysis.phase2.pipeline import _cached,_snapshot
from research.station_analysis.phase2.snapshot import analysis_anchors,selection_identity,validate_snapshot
from research.station_analysis.config import sha256,write_json


class Phase2ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.folder=Path(self.temp.name)
        self.default=Path('research/station_analysis/phase2/config.yaml')

    def tearDown(self):
        self.temp.cleanup()

    def test_exact_target_alignment_does_not_bridge_timestamp_holes(self):
        d=self.folder/'dataset';d.mkdir()
        stamps=pd.to_datetime(['2026-01-01T00:00Z','2026-01-01T01:00Z','2026-01-01T03:00Z'])
        frame=pd.DataFrame({'timestamp_utc':stamps,'water_level_depth':[10.,20.,40.],
                            'target_water_level_plus_60m':[20.,np.nan,np.nan]})
        frame.to_csv(d/'hourly.csv',index=False)
        write_json(d/'manifest.json',{'feature_columns':['water_level_depth'],
                                     'target_columns':['target_water_level_plus_60m'],'purge_hours':24})
        result=audit_datasets([d])['dataset_target_alignment'].iloc[0]
        self.assertEqual(result.paired_finite,1)
        self.assertEqual(result.value_mismatches,0)
        frame.loc[0,'target_water_level_plus_60m']=999
        frame.to_csv(d/'hourly.csv',index=False)
        self.assertEqual(audit_datasets([d])['dataset_target_alignment'].iloc[0].value_mismatches,1)

    def test_right_end_anchor_includes_last_completed_hour_and_end_is_exclusive(self):
        stations=pd.DataFrame({'station_id':['w'],'station_type':['water_level']})
        raw=pd.DataFrame({'station_id':['w'],'observed_at':pd.to_datetime(['2026-01-01T02:30Z'])})
        config={'start':'2026-01-01T00:00Z','end':None}
        anchors=analysis_anchors(raw,stations,config)
        self.assertEqual(anchors[-1],pd.Timestamp('2026-01-01T02:00Z'))
        config['end']='2026-01-01T02:00Z'
        self.assertEqual(analysis_anchors(raw,stations,config)[-1],pd.Timestamp('2026-01-01T01:00Z'))

    def test_snapshot_hash_validation_and_changed_selection(self):
        p=self.folder/'stations.parquet';p.write_bytes(b'frozen')
        config={'water_station_ids':[],'rain_station_ids':[]}
        write_json(self.folder/'manifest.json',{'selection_identity':selection_identity(config),
                                               'sha256':{p.name:sha256(p)}})
        validate_snapshot(self.folder,config)
        with self.assertRaisesRegex(ValueError,'selection'):
            validate_snapshot(self.folder,{**config,'water_station_ids':['0074647403']})
        p.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'changed'):
            validate_snapshot(self.folder,config)

    def test_changed_snapshot_manifest_is_rejected_before_source_access(self):
        frozen=self.folder/'frozen';frozen.mkdir()
        manifest=frozen/'manifest.json';manifest.write_text('{}')
        write_json(self.folder/'current_snapshot.json',{
            'directory':str(frozen),'manifest_sha256':sha256(manifest)})
        manifest.write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError,'manifest changed'):
            _snapshot({},self.folder)

    def test_no_retained_datasets_still_produce_exportable_inventory_schemas(self):
        result=audit_datasets([])
        for frame in result.values():
            self.assertTrue(frame.empty)
            self.assertGreater(len(frame.columns),0)

    def test_cache_requires_outputs_to_remain_identical(self):
        out=self.folder/'report.md';out.write_text('complete')
        cache=self.folder/'cache.json';write_json(cache,{'fingerprint':'abc','outputs':{str(out):sha256(out)}})
        self.assertTrue(_cached(cache,'abc'))
        out.write_text('changed')
        self.assertFalse(_cached(cache,'abc'))

    def test_config_protects_original_directories_and_requires_timezone_ids(self):
        load_config(self.default)
        for overrides in [{'work_dir':'train_data/test'}, {'output_dir':'data/DEM/test'},
                          {'start':'2026-01-01'}, {'water_station_ids':[74647403]}]:
            with self.assertRaises(ValueError):
                load_config(self.default,overrides)

    def test_unreviewed_stage4_cannot_execute(self):
        result=subprocess.run(['.venv/bin/python','-m','research.station_analysis.phase2.pipeline',
                               '--stage','features'],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('requires review',result.stderr)


if __name__=='__main__':
    unittest.main()
