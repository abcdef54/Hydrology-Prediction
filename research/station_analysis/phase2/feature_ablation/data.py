"""Read frozen Stage 1-3B artifacts and assemble timestamp-exact past-only features."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import yaml
from research.station_analysis.config import ROOT
from ..rainfall_preprocessing import preprocess_canonical_water_levels
from ..spatial_weather.processing import synchronized_features
from ..weather_quality import haversine_km

BASE='ecmwf_ifs:16.274164:107.264960'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_config(path):
    c=yaml.safe_load(Path(path).read_text())
    for key in ['master_plan','spatial_pointer','output_root','work_root']:c[key]=str(ROOT/c[key])
    if c['horizons']!=[6,12] or c['purge_hours']<24:raise ValueError('Require horizons 6/12 and 24h purge')
    if c['forecast_history_policy']!='anchor_only':raise ValueError('Research tree schema is explicitly anchor_only')
    return c

class Source:
    def __init__(self,c,work):
        self.c=c;self.work=Path(work);self.work.mkdir(parents=True,exist_ok=True)
        pointer=json.loads(Path(c['spatial_pointer']).read_text());self.report=Path(pointer['report']).parent
        self.spec=json.loads((self.report/'stage4_spatial_ablation_spec.json').read_text())
        m=json.loads((self.report/'manifest.json').read_text());self.snapshot=Path(m['snapshot_directory'])
        self.spatial=Path(self.spec['synchronized_forecast_path']).parent
        self.stations=pd.read_parquet(self.snapshot/'stations.parquet')
        self.water=self.stations[self.stations.station_type.eq('water_level')].copy()
        self.rains=self.stations[self.stations.station_type.eq('rainfall')].copy()
        root=ROOT/'data/processed/station_analysis/phase2'
        self.rain=pd.read_parquet(root/'rainfall/tables/hourly_rainfall_standard.parquet')
        self.strict=pd.read_parquet(root/'rainfall/tables/hourly_rainfall_strict.parquet')
        self.index=self.rain.index
        if not self.index.equals(pd.date_range(self.index.min(),self.index.max(),freq='h')):raise ValueError('Hourly index has gaps')
        obs=pd.read_parquet(root/'inventory/tables/observation_availability.parquet')
        self.level,self.targets=preprocess_canonical_water_levels(obs,self.water,self.index)
        self.grids=pd.read_parquet(self.spatial/'grid_registry.parquet')
        self.weather={6:pd.read_parquet(self.spatial/'synchronized_features.parquet').set_index('anchor_utc')}
        points=None;audit=None
        for lag in c['forecast_release_lags']:
            if lag==6:continue
            cache=self.work/f'weather_lag{lag}.parquet'
            if cache.exists():frame=pd.read_parquet(cache)
            else:
                if points is None:
                    points=pd.read_parquet(self.spatial/'points.parquet')
                    audit=pd.read_parquet(self.spatial/'tables/forecast_run_integrity.parquet')
                frame=synchronized_features(pd.DataFrame(),self.grids.grid_id.tolist(),self.index,lag,c['max_forecast_age_hours'],audit,points)
                frame.to_parquet(cache,index=False)
            self.weather[lag]=frame.set_index('anchor_utc')
        # Identical eligibility even for release-time diagnostics; never let availability change the population silently.
        self.common=self.index
        for f in self.weather.values():self.common=self.common.intersection(f.index)
        self.rain_events=pd.read_csv(ROOT/'reports/station_analysis/phase2/stage2_rainfall/tables/05_independent_rainfall_events.csv')
        self.candidates=pd.read_csv(self.report/'tables/candidate_arms.csv',dtype={'water_station_id':str})
        self.input_paths=[Path(c['master_plan']),self.report/'manifest.json',self.report/'stage4_spatial_ablation_spec.json',
            self.snapshot/'stations.parquet',root/'inventory/tables/observation_availability.parquet',
            root/'rainfall/tables/hourly_rainfall_standard.parquet',root/'rainfall/tables/hourly_rainfall_strict.parquet',
            self.spatial/'synchronized_features.parquet',self.spatial/'runs.parquet',self.spatial/'points.parquet']
        self.receipt_scope='observations retrospective snapshot; sensor receipt timing not independently proven at all anchors'

    def folds(self,idx):
        result=[]
        for f in self.c['folds']:
            a=pd.Timestamp(f['validation_start']);b=pd.Timestamp(f['validation_end'])
            train=(idx<a-pd.Timedelta(hours=self.c['purge_hours']))
            val=(idx>=a)&(idx+pd.Timedelta(hours=self.c['purge_hours'])<b)
            if train.sum()>=self.c['minimum_train'] and val.sum()>=self.c['minimum_validation']:
                result.append({**f,'train':train,'val':val})
        return result

    def base(self,sid):
        h=self.level[sid];x=pd.DataFrame({f'water_lag_{lag}':h.shift(lag) for lag in range(self.c['sequence_length'])},index=self.index)
        for lag in [1,3,6,12,24,48,95]:x[f'water_change_{lag}']=h-h.shift(lag)
        for w in [3,6,12,24,48,96]:
            x[f'water_mean_{w}']=h.rolling(w,min_periods=max(1,int(w*.8))).mean()
            x[f'water_std_{w}']=h.rolling(w,min_periods=max(2,int(w*.8))).std()
        x['water_history_coverage']=h.notna().rolling(96,min_periods=96).mean()
        y=pd.DataFrame({f'h{z}':self.targets[sid][f'dh_p{z}h'] for z in [6,12]})
        good=h.notna()&y.notna().all(axis=1)&x.water_history_coverage.ge(self.c['minimum_water_history_fraction'])&self.index.isin(self.common)
        return x.loc[good].astype('float32'),y.loc[good].astype('float32'),h.loc[good]

    def choose_rain(self,sid,idx,train):
        row=self.water.set_index('station_id').loc[sid]
        distances=haversine_km(row.latitude,row.longitude,self.rains.latitude,self.rains.longitude)
        fixed=self.rains.iloc[np.argsort(distances)[:2]].station_id.tolist()
        if sid=='0074647403':fixed=['750062','750007']
        # Discovery uses the earliest eligible fold TRAIN only, before every validation period used for this station.
        y=self.targets[sid].dh_p12h.reindex(idx).loc[train]
        score={}
        for col in self.rain:
            x=self.rain[col].rolling(6,min_periods=6).sum().reindex(idx).loc[train]
            pair=pd.concat([x,y],axis=1).dropna()
            score[col]=float(pair.iloc[:,0].corr(pair.iloc[:,1])) if len(pair)>=100 and pair.std().gt(0).all() else np.nan
        ranked=sorted([s for s in score if np.isfinite(score[s])],key=lambda s:(-score[s],s))
        return fixed,ranked[:2] if len(ranked)>=2 else fixed,score

    def rain_features(self,ids,idx,strict=False):
        source=self.strict if strict else self.rain
        parts={}
        for sid in ids:
            for w in self.c['rain_windows']:
                v=source[sid].rolling(w,min_periods=w).sum().reindex(idx)
                parts[f'rain_{sid}_sum{w}']=v;parts[f'rain_{sid}_sum{w}_missing']=v.isna().astype(float)
            for lag in self.c['rain_lags']:
                v=source[sid].shift(lag).reindex(idx)
                parts[f'rain_{sid}_lag{lag}']=v;parts[f'rain_{sid}_lag{lag}_missing']=v.isna().astype(float)
        return pd.DataFrame(parts,index=idx,dtype='float32')

    def weather_features(self,ids,idx,mode='normal',lag=6,window12=False):
        f=self.weather[lag].reindex(idx);parts={}
        for k,gid in enumerate(ids):
            use=BASE if mode=='duplicate' else gid
            for a,b in [(0,3),(3,6),(6,12),(12,24)]:
                if window12 and a==12:continue
                key=f'weather_grid{k}_rain_{a}_{b}'
                v=f[f'{use}|rain_{a}_{b}h_mm']
                parts[key]=v*0 if mode=='availability' else v
                parts[key+'_missing']=v.isna().astype(float)
        if ids:parts['weather_age_hours']=f.age_hours
        return pd.DataFrame(parts,index=idx,dtype='float32')

    def features(self,base,ids,rain_ids,exp,mode='normal',lag=6):
        parts=[base]
        if exp not in ['E0','E2']:parts.append(self.rain_features(rain_ids,base.index,strict=mode=='strict'))
        if exp not in ['E0','E1']:parts.append(self.weather_features(ids,base.index,mode,lag,window12=exp=='E4'))
        result=pd.concat(parts,axis=1).replace([np.inf,-np.inf],np.nan)
        if result.columns.duplicated().any():raise ValueError('Duplicate feature columns')
        return result
