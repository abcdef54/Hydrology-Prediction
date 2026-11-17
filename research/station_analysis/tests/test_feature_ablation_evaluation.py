import unittest
import numpy as np
import pandas as pd
from research.station_analysis.phase2.feature_ablation.evaluation import aggregate_seeds,event_labels,factorial_interactions,KEYS

class FeatureAblationEvaluationTests(unittest.TestCase):
    def test_fold_rmse_is_weighted_by_sample_count(self):
        base=dict(station_id='x',station_name='X',experiment='E0',arm='W0',mode='normal',lag=6,version='all',diagnostic='principal',scope='validation',horizon=6,seed=42)
        frame=pd.DataFrame([{**base,'n':10,'rmse_mm':10,'mae_mm':8,'n200':1},{**base,'n':90,'rmse_mm':20,'mae_mm':16,'n200':2}])
        seed,summary=aggregate_seeds(frame)
        self.assertAlmostEqual(seed.rmse_mm.iloc[0],np.sqrt(370))
        self.assertEqual(seed.n200.iloc[0],3)
        self.assertEqual(seed.n.iloc[0],100)
        self.assertEqual(summary.seeds.iloc[0],1)

    def test_event_merge_rule_keeps_separated_episodes_apart(self):
        idx=pd.date_range('2026-01-01',periods=100,freq='h',tz='UTC');h=pd.Series(0.,index=idx)
        h.iloc[5:10]=500;h.iloc[70:76]=500
        labels,events=event_labels(h,400)
        self.assertEqual(len(events),2)
        self.assertNotEqual(labels.iloc[7],labels.iloc[72])
        self.assertEqual(labels.iloc[40],0)

    def test_factorial_ignores_release_time_diagnostics(self):
        arms=['W0','W1','W2','W3','W4','W5','W6','W7']
        rows=[dict(station_id='0074647403',experiment='E3',mode='normal',version='all',scope='validation',horizon=6,seed=42,lag=6,arm=a,rmse_mm=10,n200=1) for a in arms]
        rows.append({**rows[1],'lag':9,'rmse_mm':999})
        result=factorial_interactions(pd.DataFrame(rows))
        self.assertEqual(len(result),3)
        self.assertTrue(result.mse_gain_synergy.eq(0).all())
