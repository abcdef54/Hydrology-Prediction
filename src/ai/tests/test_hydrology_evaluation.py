import unittest
import numpy as np
import pandas as pd
from src.ai.hydrology_evaluation import (EvaluationConfig, _check_predictions,
                                         event_matching, observation_events, regression_detection)


class HydrologyEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.start=pd.Timestamp('2026-01-01',tz='UTC')
        self.cfg=EvaluationConfig(event_window_hours=2,event_merge_gap_hours=1)

    def predictions(self):
        a=self.start+pd.Timedelta(hours=2)
        return pd.DataFrame([dict(station_id='1',anchor_utc=a,target_utc=a+pd.Timedelta(hours=h),horizon_hours=h,
                                  current_mm=0.,actual_mm=actual,predicted_mm=pred)
                             for h,actual,pred in [(6,200.,210.),(12,250.,220.)]])

    def test_strict_and_gray_area(self):
        summary,_=regression_detection(self.predictions(),self.cfg)
        s=summary[summary.level=='pooled'].iloc[0]
        self.assertEqual((s.tp,s.fp,s.fn),(1,1,0));self.assertEqual(s.gray_ignored,1)
        self.assertEqual(s.gray_precision,1)

    def test_event_shared_between_horizons_and_matching(self):
        # Observed rises at t=8,9; same event is matched to either forecast horizon.
        times=pd.date_range(self.start,periods=16,freq='h')
        obs=pd.DataFrame(dict(station_id='1',timestamp_utc=times,water_level_mm=[0]*8+[250,300]+[300]*6))
        events=observation_events(obs,{'1':(times[2],times[-1])},self.cfg)
        self.assertEqual(len(events),1);self.assertEqual(events.iloc[0].start_utc,times[8])
        a=times[2];p=pd.DataFrame([dict(station_id='1',anchor_utc=a,target_utc=a+pd.Timedelta(hours=h),horizon_hours=h,
                                      current_mm=0.,actual_mm=250.,predicted_mm=250.) for h in (6,12)])
        e,w,m=event_matching(p,events,self.cfg)
        self.assertEqual((len(e),len(w),len(m)),(1,1,1));self.assertEqual(m.iloc[0].lead_to_event_start_hours,6)

    def test_competing_warning_is_not_labeled_false(self):
        times=pd.date_range(self.start,periods=16,freq='h')
        obs=pd.DataFrame(dict(station_id='1',timestamp_utc=times,water_level_mm=[0]*8+[250,300]+[300]*6))
        events=observation_events(obs,{'1':(times[2],times[-1])},self.cfg)
        # Two separated warning episodes target the same observed event.
        p=pd.DataFrame([dict(station_id='1',anchor_utc=times[a],target_utc=times[t],horizon_hours=t-a,
                             current_mm=0.,actual_mm=250.,predicted_mm=250.) for a,t in [(0,8),(2,8)]])
        _,warnings,matches=event_matching(p,events,self.cfg)
        self.assertEqual(len(matches),1)
        self.assertEqual(sorted(warnings.status.tolist()),['matched','unmatched_competing'])
        self.assertEqual(events.iloc[0].peak_time_utc,times[9])

    def test_warning_without_event_is_unmatched(self):
        empty=pd.DataFrame(columns=['station_id','event_id','start_utc','end_utc','peak_time_utc','peak_level_mm'])
        _,warnings,matches=event_matching(self.predictions(),empty,self.cfg)
        self.assertTrue(matches.empty);self.assertEqual(warnings.iloc[0].status,'unmatched_no_overlap')

    def test_missing_prevents_false_rolling_event(self):
        times=pd.date_range(self.start,periods=8,freq='h')
        obs=pd.DataFrame(dict(station_id='1',timestamp_utc=times,water_level_mm=[0,0,np.nan,300,300,300,300,300]))
        events=observation_events(obs,{'1':(times[0],times[-1])},self.cfg)
        self.assertTrue(events.empty)

    def test_missing_splits_two_events(self):
        times=pd.date_range(self.start,periods=13,freq='h')
        obs=pd.DataFrame(dict(station_id='1',timestamp_utc=times,water_level_mm=[0,0,250,300,np.nan,0,0,250,300,300,300,300,300]))
        cfg=EvaluationConfig(event_window_hours=2,event_merge_gap_hours=10)
        self.assertEqual(len(observation_events(obs,{'1':(times[0],times[-1])},cfg)),2)

    def test_invalid_keys_and_time_rejected(self):
        df=self.predictions();self.assertRaises(ValueError,_check_predictions,pd.concat([df,df.iloc[:1]]))
        df.loc[0,'target_utc']+=pd.Timedelta(hours=1);self.assertRaises(ValueError,_check_predictions,df)

    def test_no_positive_metrics_are_undefined(self):
        p=self.predictions();p.actual_mm=0;summary,_=regression_detection(p,self.cfg)
        row=summary[summary.level=='pooled'].iloc[0]
        self.assertTrue(np.isnan(row.average_precision));self.assertTrue(np.isnan(row.recall))


if __name__ == '__main__': unittest.main()
