"""Test forecast timing, whole-window identity and exact sequence missingness."""
import tempfile
from pathlib import Path
from unittest import TestCase
import pandas as pd
import numpy as np
from research.scripts.weather_dataset_readiness import select_windows, sequence_audit, metric_pair


class ReadinessTests(TestCase):
    def cube(self):
        start=pd.Timestamp('2026-09-01T00:00Z')
        rows=[]
        for shift in [0,6]:
            cycle=start+pd.Timedelta(hours=shift)
            for hour in range(1,72):
                rows.append(dict(run_id=str(shift),initialized_at=cycle,
                    valid_at=cycle+pd.Timedelta(hours=hour),precipitation_mm=float(hour if shift==0 else 999)))
        return pd.DataFrame(rows)

    def test_uses_one_available_run_and_interval_endpoints(self):
        cube=self.cube();anchor=pd.Timestamp('2026-09-01T06:00Z')
        result=select_windows(cube,pd.DatetimeIndex([anchor])).iloc[0]
        self.assertEqual(result.initialized_at,pd.Timestamp('2026-09-01T00:00Z'))
        self.assertEqual(result.forecast_0_3h,7+8+9)
        self.assertEqual(result.forecast_3_6h,10+11+12)
        self.assertEqual(result.forecast_12_24h,sum(range(19,31)))

    def test_null_does_not_become_zero_and_fallback_has_eligible_cycle(self):
        cube=self.cube();anchor=pd.Timestamp('2026-09-01T12:00Z')
        cube.loc[(cube.run_id=='6')&(cube.valid_at==anchor+pd.Timedelta(hours=1)),'precipitation_mm']=np.nan
        primary=select_windows(cube,pd.DatetimeIndex([anchor])).iloc[0]
        fallback=select_windows(cube,pd.DatetimeIndex([anchor]),max_age=24,fallback=True).iloc[0]
        self.assertTrue(pd.isna(primary.forecast_0_3h))
        self.assertEqual(primary.forecast_3_6h,3*999)
        self.assertTrue(fallback.fallback)
        self.assertEqual(fallback.forecast_0_3h,13+14+15)
        self.assertEqual(fallback.initialized_at,pd.Timestamp('2026-09-01T00:00Z'))

    def test_missing_one_timestep_affects_every_containing_sequence(self):
        index=pd.date_range('2026-01-01',periods=120,freq='h',tz='UTC')
        part=pd.DataFrame({'water_level_depth':1.,'target_water_level_plus_60m':2.},index=index)
        complete=pd.Series(True,index=index);complete.iloc[100]=False
        with tempfile.TemporaryDirectory() as tmp:
            result,_=sequence_audit({'train':part},complete,None,Path(tmp))
        self.assertEqual(result.iloc[0].baseline_sequences,25)
        self.assertEqual(result.iloc[0].complete_weather_all_96,5)
        self.assertEqual(result.iloc[0].sequences_lost_if_drop_missing,20)

    def test_metrics_use_common_pairs(self):
        result=metric_pair(pd.Series([1.,2.,np.nan]),pd.Series([2.,np.nan,4.]))
        self.assertEqual(result['n'],1)
        self.assertEqual(result['bias_mm'],1)

    def test_early_peak_is_history_but_not_a_training_target(self):
        from research.scripts.weather_dataset_readiness import high_water_events
        index=pd.date_range('2026-01-01',periods=180,freq='h',tz='UTC')
        levels=np.full(180,100.)
        levels[48:60]=1000.  # Peak before the first 96-step anchor.
        levels[130:140]=300.
        part=pd.DataFrame({'water_level_depth':levels,'rain_750007_depth':0.,'rain_750062_depth':0.},index=index)
        for h in [1,3,6,12,24]:part[f'target_water_level_plus_{h*60}m']=part.water_level_depth.shift(-h)
        complete=pd.Series(True,index=index)
        with tempfile.TemporaryDirectory() as tmp:
            _,retained=sequence_audit({'train':part},complete,None,Path(tmp))
            events,summary=high_water_events(part,{'train':part},retained,Path(tmp))
        early=events.loc[events.peak_water_mm==1000].iloc[0]
        self.assertFalse(early.peak_present_as_eligible_target)
        self.assertEqual(summary.iloc[0].max_water_mm,1000.)
        self.assertEqual(summary.iloc[0].max_water_in_eligible_targets_mm,300.)

    def test_future_window_report_counts_unavailable_forecasts(self):
        from research.scripts.weather_dataset_readiness import window_analysis
        index=pd.date_range('2026-09-01',periods=48,freq='h',tz='UTC')
        hourly=pd.DataFrame({'rain_750007_depth':1.,'rain_750062_depth':1.},index=index)
        splits={'train':hourly.iloc[:24],'val':hourly.iloc[24:36],'test':hourly.iloc[36:]}
        with tempfile.TemporaryDirectory() as tmp:
            complete,fallback,metrics=window_analysis(self.cube(),hourly,splits,Path(tmp))
            heavy=pd.read_csv(Path(tmp)/'02_heavy_rain_detection.csv')
            self.assertTrue((Path(tmp)/'02_future_windows.png').exists())
        self.assertEqual(len(metrics),16)
        self.assertFalse(complete.iloc[0])  # No forecast assumed available before 06 UTC.
        self.assertTrue(complete.iloc[6])
        row=heavy[(heavy.scope=='all')&(heavy.window=='0-3h')].iloc[0]
        self.assertGreater(row.actual_heavy_missing_forecast,0)
        self.assertGreaterEqual(row.actual_heavy_all,row.true_heavy)
