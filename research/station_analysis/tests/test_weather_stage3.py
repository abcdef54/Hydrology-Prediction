"""Scientific alignment, missingness, city deduplication and train-only projections."""
import unittest
import numpy as np
import pandas as pd
from research.station_analysis.phase2.weather_metrics import future_amount,rain_metrics,partial_correlations
from research.station_analysis.phase2.weather import canonical_city_forecast


class WeatherStage3Tests(unittest.TestCase):
    def test_future_interval_uses_hour_ends_without_overlap(self):
        idx=pd.date_range('2026-01-01',periods=30,freq='h',tz='UTC')
        rain=pd.Series(np.arange(30,dtype=float),index=idx)
        self.assertEqual(future_amount(rain,0,3).iloc[0],6.)
        self.assertEqual(future_amount(rain,3,6).iloc[0],15.)
        self.assertEqual(future_amount(rain,6,12).iloc[0],sum(range(7,13)))
        self.assertEqual(future_amount(rain,12,24).iloc[0],sum(range(13,25)))
        self.assertEqual(sum(future_amount(rain,a,b).iloc[0] for a,b in [(0,3),(3,6),(6,12),(12,24)]),sum(range(1,25)))

    def test_missing_hour_and_end_are_not_zero_filled(self):
        idx=pd.date_range('2026-01-01',periods=10,freq='h',tz='UTC')
        rain=pd.Series(1.,index=idx);rain.iloc[2]=np.nan
        self.assertTrue(np.isnan(future_amount(rain,0,3).iloc[0]))
        self.assertTrue(np.isnan(future_amount(rain,0,3).iloc[-1]))
        self.assertEqual(future_amount(rain,3,6).iloc[0],3.)

    def test_irregular_or_naive_ground_truth_rejected(self):
        with self.assertRaises(ValueError):future_amount(pd.Series([1.,2.],index=pd.date_range('2026-01-01',periods=2,freq='h')),0,1)
        idx=pd.DatetimeIndex(['2026-01-01T00:00Z','2026-01-01T02:00Z'])
        with self.assertRaises(ValueError):future_amount(pd.Series([1.,2.],index=idx),0,1)

    def test_detection_uses_same_pairs_and_undefined_recall_is_nan(self):
        pred=pd.Series([0.,1.,20.,np.nan]);obs=pd.Series([0.,0.,10.,100.])
        r=rain_metrics(pred,obs,wet_mm=.1,heavy_mm=15.)
        self.assertEqual(r['n'],3)
        self.assertEqual(r['wet_tp'],1);self.assertEqual(r['wet_fp'],1)
        self.assertTrue(np.isnan(r['heavy_recall']))
        self.assertEqual(r['heavy_fp'],1)

    def test_zero_baseline_skill_and_bias(self):
        r=rain_metrics(pd.Series([0.,2.]),pd.Series([0.,2.]))
        self.assertEqual(r['bias_mm'],0.)
        self.assertEqual(r['mse_skill_vs_zero'],1.)
        self.assertTrue(np.isnan(rain_metrics(pd.Series([0.]),pd.Series([0.]))['mse_skill_vs_zero']))

    def test_city_forecast_preserves_one_vintage_and_never_duplicates_anchor(self):
        stamp=pd.Timestamp('2026-01-01T12:00Z')
        rows=[]
        for loc,complete,init,value in [('A',True,stamp-pd.Timedelta(hours=12),1.),('B',False,stamp-pd.Timedelta(hours=6),9.)]:
            rows.append({'location_id':loc,'policy':'legacy_assumed_release','anchor_utc':stamp,'model':'ecmwf',
                         'selected_run_id':loc,'complete_24h':complete,'selected_initialized_at':init,
                         'selected_retrieved_at':stamp,'rain_0_3h_mm':value,'rain_12_24h_mm':value*4})
        result=canonical_city_forecast(pd.DataFrame(rows))
        self.assertEqual(len(result),1)
        self.assertEqual(result.iloc[0].location_id,'hue_city_shared')
        self.assertEqual(result.iloc[0].request_location_id,'A')
        self.assertEqual(result.iloc[0].rain_12_24h_mm,4.)

    def test_city_selection_does_not_mix_models_silently(self):
        with self.assertRaises(ValueError):canonical_city_forecast(pd.DataFrame({'model':['A','B']}))

    def test_partial_projection_fit_does_not_see_evaluation_targets(self):
        rng=np.random.default_rng(42)
        idx=pd.date_range('2026-01-01',periods=300,freq='h',tz='UTC')
        controls=pd.DataFrame({'a':rng.normal(size=300),'b':rng.normal(size=300)},index=idx)
        feature=controls.a+pd.Series(rng.normal(size=300),index=idx)
        target=controls.b+feature+pd.Series(rng.normal(size=300),index=idx)
        train=np.arange(300)<200;ev=~train
        original=partial_correlations(feature,target,controls,train,ev)
        changed=target.copy();changed.loc[ev]+=10000
        changed_feature=feature.copy();changed_feature.loc[ev]*=1000
        again=partial_correlations(changed_feature,changed,controls,train,ev)
        self.assertEqual(original[0],again[0])
        self.assertEqual(again[0]['train_fit_n'],200)


if __name__=='__main__':unittest.main()
