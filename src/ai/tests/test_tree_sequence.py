import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from src.ai.tree_sequence import sequence_table,ValidationSelector,train_horizon,TreeSequenceForecast
from research.tree_horizon_experiment import prepare,compare


class TreeSequenceTests(unittest.TestCase):
    def test_lag_order_missing_and_future_exclusion(self):
        f=pd.DataFrame(dict(timestamp_utc=pd.date_range('2026-01-01',periods=8,freq='h',tz='UTC'),
            water_level_depth=np.arange(8.),rain=[0,np.nan,2,3,4,5,6,7],target_water_level_plus_60m=np.arange(8.)+1))
        data=sequence_table(f,['water_level_depth','rain'],['target_water_level_plus_60m'],3)
        np.testing.assert_equal(data['X'][0],[2,2,1,np.nan,0,0])
        changed=f.copy();changed.loc[3:,'rain']=1000
        np.testing.assert_equal(sequence_table(changed,['water_level_depth','rain'],['target_water_level_plus_60m'],3)['X'][0],data['X'][0])
        self.assertEqual(data['names'][:3],['lag000__water_level_depth','lag000__rain','lag001__water_level_depth'])
        with self.assertRaises(ValueError):sequence_table(f,['target_water_level_plus_60m'],['target_water_level_plus_60m'],3)
        f.loc[2,'timestamp_utc']=f.loc[1,'timestamp_utc']
        with self.assertRaises(ValueError):sequence_table(f,['water_level_depth'],['target_water_level_plus_60m'],3)

    def test_checkpoint_tie_break_and_persistence_fallback(self):
        selector=ValidationSelector('large_error_rate',3,np.array([201.,0.]))
        self.assertFalse(selector.update(1,0,150))
        self.assertFalse(selector.update(2,0,140))
        self.assertFalse(selector.update(3,1,50))
        self.assertEqual(selector.best_round,2)
        self.assertTrue(selector.update(5,1,40))
        fallback=ValidationSelector('rmse',2,np.zeros(2))
        self.assertFalse(fallback.update(1,0,1))
        self.assertTrue(fallback.update(2,0,1))
        self.assertEqual(fallback.best_round,0)

    def test_native_training_and_reload_on_small_synthetic_data(self):
        rng=np.random.default_rng(42)
        train_x=rng.normal(size=(100,3)).astype('float32')
        val_x=rng.normal(size=(40,3)).astype('float32')
        train=dict(X=train_x,y=(50+30*train_x[:,0])[:,None],level=np.zeros(100))
        val=dict(X=val_x,y=(50+30*val_x[:,0])[:,None],level=np.zeros(40))
        train['X'][0,1]=np.nan
        options=dict(target_mode='residual',checkpoint_metric='large_error_rate',early_stopping=3,
            learning_rate=.1,max_depth=2,seed=42,device='cpu',n_jobs=1,num_leaves=3,n_estimators=8)
        for method in ['xgb','lgbm']:
            with self.subTest(method=method),tempfile.TemporaryDirectory() as name:
                folder=Path(name);path=folder/('model.json' if method=='xgb' else 'model.txt')
                result=train_horizon(method,train,val,0,options,path)
                self.assertEqual(result['kind'],'tree')
                self.assertGreater(result['best_round'],0)
                result['target']='target_water_level_plus_60m'
                spec=dict(train_method=method,target_mode='residual',horizons=[result],status='complete',
                    target_columns=['target_water_level_plus_60m'])
                (folder/'tree_model.json').write_text(json.dumps(spec))
                before=TreeSequenceForecast(folder).predict_table(val)
                after=TreeSequenceForecast(folder).predict_table(val)
                np.testing.assert_array_equal(before,after)
                self.assertEqual(before.shape,(40,1))
                error=before[:,0]-val['y'][:,0]
                self.assertAlmostEqual(np.sqrt(np.mean(error**2)),result['validation_score'][1],places=4)
                # Explicit persistence requires no native artifact.
                spec['horizons']=[dict(kind='persistence',path=None,best_round=0,target='target_water_level_plus_60m')]
                (folder/'tree_model.json').write_text(json.dumps(spec))
                np.testing.assert_array_equal(TreeSequenceForecast(folder).predict_table(val),np.zeros((40,1)))
                spec['status']='training';(folder/'tree_model.json').write_text(json.dumps(spec))
                with self.assertRaises(ValueError):TreeSequenceForecast(folder)

    def test_real_dataset_uses_same_96h_population_for_5_and_24(self):
        root=Path(__file__).resolve().parents[3]/'train_data/da_vien_1h_2026-10-15_forecast_mask_o24'
        if not root.exists():self.skipTest('Local snapshot absent')
        _,_,_,_,d5=prepare(root,5,96)
        _,_,_,_,d24=prepare(root,24,96)
        self.assertEqual([len(d5[s]['indices']) for s in ['train','val','test']],[6095,985,592])
        for s in ['train','val','test']:
            self.assertEqual(d5[s]['X'].shape[1],2496)
            np.testing.assert_array_equal(d5[s]['indices'],d24[s]['indices'])
            np.testing.assert_array_equal(d5[s]['X'],d24[s]['X'])

    def test_real_comparison_preserves_B_population_and_common_horizons(self):
        from research.finetune_direction_experiment import SOURCE
        from src.ai.comparison import load_candidate
        import joblib
        root=Path(__file__).resolve().parents[3]/'train_data/da_vien_1h_2026-10-15_forecast_mask_o24'
        if not root.exists() or not SOURCE.exists():self.skipTest('Local dataset/B absent')
        _,_,targets,frames,data=prepare(root,24,96)
        saved=joblib.load(SOURCE.with_suffix('.preprocessing.joblib'))
        ref=load_candidate(SOURCE.read_bytes(),saved,'cpu')
        values=np.repeat(data['test']['level'][:,None],24,axis=1)
        metrics,points,summary=compare(data['test'],values,frames['test'],targets,frames['test'],saved,ref)
        self.assertEqual(summary['common_anchors'],592)
        self.assertEqual(len(points),592*5)
        self.assertEqual(len(metrics),5)
        self.assertEqual(summary['reference']['above_200'],48)
        changed=frames['test'].copy();changed.loc[100,targets[0]]+=1
        with self.assertRaises(AssertionError):compare(data['test'],values,frames['test'],targets,changed,saved,ref)

    def test_full_cli_reporting_with_no_real_dataset_training(self):
        from unittest.mock import patch
        from contextlib import redirect_stdout
        import io
        from research.tree_horizon_experiment import main
        from research.finetune_direction_experiment import SOURCE
        root=Path(__file__).resolve().parents[3]
        dataset=root/'train_data/da_vien_1h_2026-10-15_forecast_mask_o24'
        if not dataset.exists() or not SOURCE.exists():self.skipTest('Local fixtures absent')
        def no_fit(*args,**kwargs):
            return dict(kind='persistence',best_round=0,validation_score=[0,0.],path=None,
                history=[dict(round=1,above_200=0,rmse_mm=0.)])
        with tempfile.TemporaryDirectory() as name:
            out=Path(name)/'run'
            argv=['tree_horizon_experiment','--train-method','xgb','--dataset-dir',str(dataset),
                '--output-dir',str(out),'--no-mlflow']
            with patch('sys.argv',argv),patch('research.tree_horizon_experiment.train_horizon',side_effect=no_fit),redirect_stdout(io.StringIO()):
                main()
            self.assertTrue((out/'REPORT.md').exists())
            self.assertEqual(json.loads((out/'tree_model.json').read_text())['status'],'complete')
            self.assertEqual(len(pd.read_csv(out/'all_horizon_metrics.csv')),5)
            self.assertEqual(len(pd.read_csv(out/'comparison_B_predictions.csv.gz')),2960)
            self.assertEqual(len(pd.read_csv(out/'comparison_production_sequences.csv')),10)


if __name__=='__main__':unittest.main()
