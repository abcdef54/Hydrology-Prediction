"""Verify forecast amounts against explicit point-gauge comparators."""
import numpy as np
import pandas as pd
from .weather_metrics import future_amount,rain_metrics

WINDOWS=((0,3),(3,6),(6,12),(12,24))


def haversine_km(lat,lon,lats,lons):
    a,b=np.radians(lat),np.radians(np.asarray(lats,dtype=float))
    dlat=b-a;dlon=np.radians(np.asarray(lons,dtype=float)-lon)
    return 6371*2*np.arcsin(np.minimum(1,np.sqrt(np.sin(dlat/2)**2+np.cos(a)*np.cos(b)*np.sin(dlon/2)**2)))


def compare_forecasts(selected, rainfall, strict_rainfall, stations, grids, config):
    wet=config['weather_analysis']['wet_threshold_mm']
    heavy_rate=config['weather_analysis']['heavy_mean_rate_mm_h']
    nearest=config['weather_analysis']['nearby_gauges']
    boundary=pd.Timestamp(config['future_analysis']['grouping_end'])
    catalog=stations[stations.station_type.eq('rainfall')].set_index('station_id')
    comparator_rows=[];pairs=[];metrics=[]
    obs_cache={}
    for (location,policy),part in selected.groupby(['location_id','policy']):
        if part.anchor_utc.duplicated().any():raise ValueError('Multiple forecast models at one anchor require separate analysis')
        frame=part.set_index('anchor_utc').sort_index()
        grid=grids[grids.location_id.eq(location)].iloc[0]
        distances=pd.Series(haversine_km(grid.grid_latitude,grid.grid_longitude,catalog.latitude,catalog.longitude),index=catalog.index)
        primary=config['weather_analysis'].get('reference_gauges_by_location',{}).get(location,config['weather_analysis']['reference_gauge_ids'])
        ids=list(dict.fromkeys(primary+distances.sort_values().head(nearest).index.tolist()))
        ids=[s for s in ids if s in rainfall.columns]
        for sid in ids:
            comparator_rows.append({'location_id':location,'policy':policy,'gauge_id':sid,
                                    'gauge_name':catalog.loc[sid,'name'],'primary_reference_gauge':sid in primary,
                                    'grid_distance_km':float(distances[sid]),
                                    'comparison_support':'point gauge; grid-area representativeness unverified'})
            for a,b in WINDOWS:
                key=(sid,a,b)
                if key not in obs_cache:
                    obs_cache[key]=(future_amount(rainfall[sid],a,b),future_amount(strict_rainfall[sid],a,b))
                for obs_policy,truth in zip(['standard_5of6','strict_6of6'],obs_cache[key]):
                    df=frame[['selected_run_id','selected_archive_kind','age_hours']].copy()
                    df['forecast_mm']=frame[f'rain_{a}_{b}h_mm']
                    df['observed_mm']=truth.reindex(df.index)
                    df['scope']=np.where(df.index+pd.Timedelta(hours=24)<=boundary,'train',
                                          np.where(df.index>boundary,'evaluation','purge'))
                    df['season']=np.where(df.index.tz_convert('Asia/Ho_Chi_Minh').month.isin([9,10,11,12]),'flood_months','other_months')
                    df['age_bin']=pd.cut(df.age_hours,[-.001,6,9,12,24],labels=['0-6','6-9','9-12','12-24'],include_lowest=True).astype(str)
                    df['location_id']=location;df['policy']=policy;df['gauge_id']=sid
                    df['observation_policy']=obs_policy;df['window']=f'{a}_{b}h';df['window_hours']=b-a
                    df['archive_kind']=df.selected_archive_kind.fillna('unavailable')
                    pairs.append(df.reset_index())
                    # Cohorts are not pooled into a single evidential class.
                    for (archive,scope),base in df.groupby(['archive_kind','scope']):
                        if scope=='purge':continue
                        for breakdown,column in [('all',None),('season','season'),('age','age_bin')]:
                            groups=[('all',base)] if column is None else list(base.groupby(column))
                            for value,sub in groups:
                                if value=='nan':continue
                                result=rain_metrics(sub.forecast_mm,sub.observed_mm,wet_mm=wet,heavy_mm=heavy_rate*(b-a))
                                metrics.append({'location_id':location,'policy':policy,'gauge_id':sid,
                                                'observation_policy':obs_policy,'archive_kind':archive,'scope':scope,
                                                'window':f'{a}_{b}h','window_hours':b-a,
                                                'breakdown':breakdown,'stratum':value,**result})
    return pd.DataFrame(comparator_rows),pd.concat(pairs,ignore_index=True),pd.DataFrame(metrics)
