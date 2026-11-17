"""Leakage, chronological purge, diagnostic semantics and safe equivalence tests."""
import unittest
import numpy as np
import pandas as pd
from research.station_analysis.phase2.feature_ablation.data import Source,BASE
from research.station_analysis.phase2.feature_ablation.pipeline import normalized
from research.station_analysis.phase2.feature_ablation.planning import choose_finalists
from research.station_analysis.phase2.feature_ablation.training import checkpoint_metric,scalar_metrics,reuse_key

class FeatureAblationTests(unittest.TestCase):
    def source(self):
        s=Source.__new__(Source)
        s.c={'folds':[{'id':'f','validation_start':'2026-02-01T00:00Z','validation_end':'2026-03-01T00:00Z'}],
             'purge_hours':24,'minimum_train':20,'minimum_validation':20,'rain_windows':[1,3,6,12,24],'rain_lags':[0,3,95]}
        s.index=pd.date_range('2026-01-01',periods=1500,freq='h',tz='UTC')
        s.rain=pd.DataFrame({'r':np.arange(len(s.index),dtype=float)},index=s.index);s.strict=s.rain.copy()
        frame=pd.DataFrame(index=s.index)
        for gid,value in [(BASE,2.),('other',9.)]:
            for a,b in [(0,3),(3,6),(6,12),(12,24)]:frame[f'{gid}|rain_{a}_{b}h_mm']=value
        frame['age_hours']=6;s.weather={6:frame}
        return s

    def test_labels_and_purge_do_not_cross_chronological_boundary(self):
        s=self.source();f=s.folds(s.index)[0]
        self.assertFalse((f['train']&f['val']).any())
        self.assertLess(s.index[f['train']].max()+pd.Timedelta(hours=24),s.index[f['val']].min())
        self.assertLess(s.index[f['val']].max()+pd.Timedelta(hours=24),pd.Timestamp(f['validation_end']))

    def test_rain_features_use_only_past_and_preserve_missing(self):
        s=self.source();a=s.rain_features(['r'],s.index)
        s.rain.loc[s.index[101]:,'r']=999999
        b=s.rain_features(['r'],s.index)
        pd.testing.assert_frame_equal(a.iloc[:101],b.iloc[:101])
        s.rain.loc[s.index[100],'r']=np.nan
        c=s.rain_features(['r'],s.index)
        self.assertTrue(np.isnan(c.loc[s.index[100],'rain_r_sum1']))
        self.assertEqual(c.loc[s.index[100],'rain_r_sum1_missing'],1)

    def test_weather_diagnostics_keep_slot_count_and_old_values(self):
        s=self.source();idx=s.index[:10]
        real=s.weather_features([BASE,'other'],idx)
        dup=s.weather_features([BASE,'other'],idx,'duplicate')
        empty=s.weather_features([BASE,'other'],idx,'availability')
        self.assertEqual(real.shape,dup.shape)
        self.assertEqual(real.shape,empty.shape)
        self.assertEqual(dup.weather_grid1_rain_0_3.iloc[0],2)
        self.assertTrue(empty.filter(regex='rain_.*(?<!missing)$').eq(0).all().all())
        self.assertTrue(empty.weather_age_hours.eq(6).all())
        short=s.weather_features([BASE],idx,window12=True)
        self.assertFalse(any('12_24' in c for c in short))

    def test_semantically_equal_cases_are_identical_not_retrained(self):
        d={'fixed_rain':['r'],'selected_rain':['r']}
        a=normalized({'experiment':'E1','arm':'W0','grids':[]},d)
        b=normalized({'experiment':'E3','arm':'W0','grids':[]},d)
        self.assertEqual(a,b)
        c=normalized({'experiment':'E3','grids':[BASE]},d)
        e=normalized({'experiment':'E5','grids':[BASE]},d)
        self.assertEqual(c,e)

    def test_finalist_selection_ignores_post_cutoff_metrics(self):
        rows=[]
        for arm,n,r in [('W0',20,80),('W1',10,85),('W2',21,90)]:
            for scope in ['validation','exploratory']:
                rows.append(dict(experiment='E3',arm=arm,mode='normal',scope=scope,n200=n,rmse_mm=r))
        before,_=choose_finalists(rows)
        rows[-1]['n200']=0;rows[-1]['rmse_mm']=0
        after,_=choose_finalists(rows)
        self.assertEqual(before,after)
        self.assertNotIn('W2',before)

    def test_reuse_ignores_only_run_and_fold_extension_not_schema_or_seed(self):
        old={'config':{'run_name':'v1','folds':[1,2,3],'eta':.05},'metadata':{'station_id':'x','execution_signature':'old'},'columns':['water0'],'seed':42}
        new={'config':{'run_name':'v2','folds':[0,1,2,3],'eta':.05,'reuse_previous_run':'v1'},'metadata':{'station_id':'x','execution_signature':'new'},'columns':['water0'],'seed':42}
        self.assertEqual(reuse_key(old),reuse_key(new))
        new['seed']=17
        self.assertNotEqual(reuse_key(old),reuse_key(new))
        new['seed']=42;new['columns']=['rain0']
        self.assertNotEqual(reuse_key(old),reuse_key(new))

    def test_large_errors_strict_and_checkpoint_priority(self):
        m=scalar_metrics([0,0,0],[200,-200,200.01]);self.assertEqual(m['n200'],1)
        class Labels:
            def get_label(self):return np.zeros(2)
        _,a=checkpoint_metric(np.array([201,0]),Labels())
        _,b=checkpoint_metric(np.array([200,200]),Labels())
        self.assertLess(b,a)
