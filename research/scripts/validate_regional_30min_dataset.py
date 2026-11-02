"""Validate the 30-minute regional dataset, without training or inference."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from src.ai.regional_data.builder import make_rain_groups,sha,write_json
from src.ai.regional_data.preprocessing import check_hourly,calendar_features,context_rows,moments
from src.ai.regional_training_data import RegionalTrainingDataset
from research.scripts.build_regional_30min_dataset import aggregate_water,aggregate_rain,prior_hour_forecast,SNAPSHOT,OLD


def validate(path):
    path=Path(path);manifest=json.loads((path/'manifest.json').read_text());config=json.loads((path/'config.json').read_text());schema=json.loads((path/'feature_schema.json').read_text());mapping=json.loads((path/'station_mapping.json').read_text());stats=json.loads((path/'normalizers.json').read_text())['fixed']
    assert manifest['resolution_minutes']==config['resolution_minutes']==30 and manifest['sequence_length']==config['sequence_length']==96
    water=pd.read_parquet(path/'water.parquet').set_index('timestamp_utc');rain=pd.read_parquet(path/'rainfall.parquet').set_index('timestamp_utc');standard=pd.read_parquet(path/'rainfall_hourly_standard.parquet').set_index('timestamp_utc');strict=pd.read_parquet(path/'rainfall_hourly_strict.parquet').set_index('timestamp_utc');weather=pd.read_parquet(path/'weather.parquet').set_index('timestamp_utc');calendar=pd.read_parquet(path/'calendar.parquet').set_index('timestamp_utc');samples=pd.read_parquet(path/'samples.parquet')
    check_hourly(water.index,30)
    assert all(frame.index.equals(water.index) for frame in [rain,standard,strict,weather,calendar])
    assert list(water.columns)==list(mapping) and len(mapping)==11
    pd.testing.assert_frame_equal(calendar,calendar_features(water.index),check_freq=False)
    assert not samples.duplicated(['station_id','anchor_utc']).any() and len(samples)==manifest['sample_count']
    for sid,part in samples.groupby('station_id'):
        pos=part.anchor_row.to_numpy(dtype=int);series=water[sid]
        assert (part.station_index==mapping[sid]['station_index']).all() and water.index[pos].equals(pd.DatetimeIndex(part.anchor_utc))
        assert (pos-part.history_start_row.to_numpy()==95).all() and (part.history_start_row>=0).all()
        np.testing.assert_allclose(series.iloc[pos],part.water_now_mm,rtol=0,atol=1e-10)
        for h in [6,12]:
            times=pd.DatetimeIndex(part.anchor_utc)+pd.Timedelta(hours=h)
            assert times.equals(pd.DatetimeIndex(part[f'target_h{h}_utc']))
            vals=series.reindex(times).to_numpy();assert np.isfinite(vals).all()
            np.testing.assert_allclose(vals,part[f'target_h{h}_mm'],rtol=0,atol=1e-10)
            np.testing.assert_allclose(vals-part.water_now_mm,part[f'delta_h{h}_mm'],rtol=0,atol=1e-10)
        cover=series.notna().rolling(96,min_periods=96).mean().iloc[pos].to_numpy();np.testing.assert_allclose(cover,part.history_coverage,rtol=0,atol=1e-12)
        assert (cover>=config['minimum_water_history_fraction']).all()
    bounds=config['folds'][0];start=pd.Timestamp(bounds['validation_start']);end=pd.Timestamp(bounds['validation_end']);train=samples[samples.fixed_train];val=samples[samples.fixed_validation]
    assert (train.anchor_utc+pd.Timedelta(hours=24)<start).all() and (val.anchor_utc>=start).all() and (val.anchor_utc+pd.Timedelta(hours=24)<end).all()
    assert not (samples.fixed_train&samples.fixed_validation).any()
    expected_rain,_,_=make_rain_groups(standard,strict,config['rain_windows'],30)
    pd.testing.assert_frame_equal(rain,expected_rain,check_freq=False)
    for sid,part in train.groupby('station_id'):
        indices=context_rows(part.anchor_row,96,len(water));st=stats['stations'][sid]
        assert st['water']==moments(water[sid].iloc[indices])
        for h in [6,12]:assert st['targets'][f'h{h}']==moments(part[f'target_h{h}_mm'])
    context=context_rows(np.unique(train.anchor_row),96,len(water));anchors=np.unique(train.anchor_row)
    for frame,cols,idx in [(rain,schema['rain_value_columns'],context),(weather,schema['weather_value_columns'],anchors)]:
        for col in cols:assert stats['shared'][col]==moments(frame[col].iloc[idx]),col
    valid=weather.run_id.notna();assert (weather.loc[valid,'forecast_source_anchor_utc']<=weather.index[valid]).all()
    assert (weather.index[valid]-weather.loc[valid,'forecast_source_anchor_utc']).isin([pd.Timedelta(0),pd.Timedelta(minutes=30)]).all()
    assert weather.loc[valid,'weather_forecast_age_hours'].between(6,24).all()
    assert (weather.loc[valid & weather.available_at.notna(),'available_at']<=weather.index[valid & weather.available_at.notna()]).all()
    raw=pd.read_parquet(SNAPSHOT/'observations.parquet',filters=[('station_id','in',list(mapping)+list(standard.columns))]);raw.station_id=raw.station_id.astype(str)
    pd.testing.assert_frame_equal(water,aggregate_water(raw,list(mapping),water.index),check_freq=False,check_names=False)
    s,t=aggregate_rain(raw,list(standard.columns),water.index)
    pd.testing.assert_frame_equal(standard,s,check_freq=False,check_names=False);pd.testing.assert_frame_equal(strict,t,check_freq=False,check_names=False)
    # Independent direct recomputation from individual 10-minute slots.
    raw['observed_at']=pd.to_datetime(raw.observed_at,utc=True)
    gauges=['750014','750041','750062']
    for sid in gauges:
        items=raw.loc[raw.station_id.eq(sid)].set_index('observed_at').sort_index()
        for anchor in samples.anchor_utc.drop_duplicates().sort_values().iloc[::max(1,len(water)//40)]:
            for hours in [1,3,6,12,24,48]:
                chunks=[]
                for lag in range(hours):
                    end=anchor-pd.Timedelta(hours=lag)
                    block=items[(items.index>end-pd.Timedelta(hours=1))&(items.index<=end)]
                    valid=block[block.quality_status.eq('observed')&np.isfinite(block.depth)&block.depth.ge(0)&(block.index==block.index.floor('10min'))]
                    if block.index.duplicated().any() or len(valid) not in (5,6): chunks.append(np.nan)
                    else: chunks.append(valid.depth.sum()*(6/5 if len(valid)==5 else 1))
                expected_total=sum(chunks) if np.isfinite(chunks).all() else np.nan
                actual_total=rain.loc[anchor,f'rain_{sid}_sum_{hours}h']
                assert (pd.isna(expected_total) and pd.isna(actual_total)) or np.isclose(expected_total,actual_total,rtol=0,atol=1e-8),(sid,anchor,hours,expected_total,actual_total)
    old=pd.read_parquet(OLD/'weather.parquet');expected=prior_hour_forecast(old,water.index)
    for col in ['run_id','weather_forecast_rain_0_3h','weather_forecast_rain_3_6h','weather_forecast_rain_6_12h','weather_forecast_rain_12_24h','weather_forecast_age_hours','forecast_source_anchor_utc']:
        pd.testing.assert_series_equal(weather[col],expected[col],check_names=False,check_freq=False)
    for source,h in manifest['source_sha256'].items():assert sha(source)==h,source
    dataset=RegionalTrainingDataset(path,split='train',fold='fixed',rain_stations='hong_thai+bach_ma+phu_bai',time_features='all',use_weather_forecast=True,sequence_length=96,target_horizons=[6,12])
    for idx in [0,len(dataset)//2,len(dataset)-1]:
        row=dataset[idx];assert row['x'].shape==(96,59) and row['y'].shape==(2,) and np.isfinite(row['x'].numpy()).all() and np.isfinite(row['y'].numpy()).all()
    assert dataset.resolution_minutes==30
    return dict(status='passed',sample_count=len(samples),splits={str(k):int(v) for k,v in samples.split.value_counts().items()},
        timestamps_checked=len(water),stations=len(mapping),target_horizons_hours=[6,12],sequence_length=96,resolution_minutes=30,
        rain_forecast_rule='At :30 use previous hourly-anchor forecast vector, shifted relative to current anchor; availability retrospective',
        checks=['all samples exact water and future labels','independent direct 10-minute rainfall window sampling','train/validation purge','raw-observation reconstruction','30-minute grid and calendars','rain 1/3/6/12/24/48h masks and totals','as-of forecast and age','train-only normalizers','source hashes','training adapter synthetic inspection'])


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('dataset',type=Path);args=ap.parse_args()
    result=validate(args.dataset);write_json(args.dataset/'validation.json',result)
    (args.dataset/'BUILDING.json').unlink(missing_ok=True)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
