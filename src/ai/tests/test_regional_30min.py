import unittest
import numpy as np
import pandas as pd
from research.scripts.build_regional_30min_dataset import aggregate_water,aggregate_rain,prior_hour_forecast
from src.ai.regional_data.builder import make_rain_groups
from src.ai.regional_data.preprocessing import calendar_features,make_samples


class Regional30MinuteTests(unittest.TestCase):
    def test_30min_water_and_rain_use_only_past_slots(self):
        i=pd.date_range('2026-01-01T00:00Z',periods=13,freq='10min')
        raw=pd.DataFrame({'station_id':['x']*len(i),'observed_at':i,'depth':np.arange(len(i),dtype=float),'quality_status':'observed'})
        anchors=pd.date_range(i[0],i[-1],freq='30min');water=aggregate_water(raw,['x'],anchors);rain,strict=aggregate_rain(raw,['x'],anchors)
        self.assertEqual(water.loc[i[6],'x'],np.mean([4,5,6]));self.assertEqual(rain.loc[i[6],'x'],sum(range(1,7)))
        self.assertEqual(strict.loc[i[9],'x'],sum(range(4,10)))
        modified=raw.copy();modified.loc[modified.observed_at>i[6],'depth']=9999
        pd.testing.assert_series_equal(water.loc[:i[6],'x'],aggregate_water(modified,['x'],anchors).loc[:i[6],'x'])

    def test_rain_multihour_avoids_double_counting(self):
        idx=pd.date_range('2026-01-01T00:00Z',periods=9,freq='30min');a=pd.DataFrame({'x':range(9)},index=idx,dtype=float)
        result,_,_=make_rain_groups(a,a,[1,3],30)
        self.assertEqual(result.loc[idx[6],'rain_x_sum_3h'],2+4+6)
        self.assertEqual(result.loc[idx[7],'rain_x_sum_3h'],3+5+7)

    def test_calendar_minute_and_targets(self):
        idx=pd.date_range('2026-01-01T00:00Z',periods=180,freq='30min')
        c=calendar_features(idx);self.assertNotEqual(c.hour_sin.iloc[0],c.hour_sin.iloc[1])
        water=pd.DataFrame({'x':np.arange(len(idx),dtype=float)},index=idx)
        folds=[{'id':'fixed','validation_start':'2026-01-03T00:00Z','validation_end':'2026-01-04T00:00Z'}]
        s=make_samples(water,{'x':{'station_index':0}},folds,'2026-01-04T00:00Z',96,.8,24,30)
        r=s.iloc[0];self.assertEqual(r.delta_h6_mm,12);self.assertEqual(r.delta_h12_mm,24)
        self.assertEqual(r.anchor_utc-idx[r.history_start_row],pd.Timedelta(hours=47,minutes=30))

    def test_half_hour_forecast_never_selects_future_hour(self):
        anchors=pd.date_range('2026-01-01T12:00Z',periods=2,freq='30min');columns={
          'timestamp_utc':[anchors[0]],'initialized_at':[pd.Timestamp('2026-01-01T06:00Z')],
          'run_id':['r'],'weather_forecast_age_hours':[6.],
          'weather_forecast_observed_availability':[0],'weather_forecast_assumed_availability':[1],'weather_forecast_hindcast':[0],
          'archive_kind':['x'],'retrieved_at':[pd.NaT],'available_at':[pd.NaT],'assumed_available_at':[anchors[0]],
          'availability_basis':['unknown'],'provenance_class':['unknown'],'model':['m'],'grid_latitude':[1.],
          'grid_longitude':[2.],'grid_id':['g'],'cell_selection':['land'],'model_version':['x'],'strict_eligible_run':[False]}
        for c in ['weather_forecast_rain_0_3h','weather_forecast_rain_3_6h','weather_forecast_rain_6_12h','weather_forecast_rain_12_24h']:
            columns[c]=[10.];columns[c+'_missing']=[0]
        columns['weather_forecast_age_hours_missing']=[0]
        w=prior_hour_forecast(pd.DataFrame(columns),anchors)
        self.assertEqual(w.iloc[1].forecast_source_anchor_utc,anchors[0]);self.assertEqual(w.iloc[1].weather_forecast_age_hours,6.5)
        self.assertEqual(w.iloc[1].weather_forecast_rain_0_3h,10.)

if __name__=='__main__':unittest.main()
