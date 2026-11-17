"""Describe measured feature effects without turning exploratory results into selection."""
from pathlib import Path
from .explanations import read_importance
import json
import pandas as pd
import numpy as np


def diagnostic_effects(incremental):
    paired=incremental.groupby(['station_id','arm','mode','lag','scope','horizon','diagnostic']).agg(
        delta_rmse_mm=('delta_rmse_mm','mean'),delta_n200=('delta_n200','mean'),seed_count=('seed','nunique')).reset_index()
    result=paired.groupby(['diagnostic','scope','horizon']).agg(
        comparisons=('station_id','size'),stations=('station_id','nunique'),
        mean_delta_rmse_mm=('delta_rmse_mm','mean'),median_delta_rmse_mm=('delta_rmse_mm','median'),
        minimum_delta_rmse_mm=('delta_rmse_mm','min'),maximum_delta_rmse_mm=('delta_rmse_mm','max'),
        comparisons_reducing_rmse=('delta_rmse_mm',lambda s:int(s.lt(0).sum())),
        comparisons_increasing_rmse=('delta_rmse_mm',lambda s:int(s.gt(0).sum())),
        mean_delta_n200=('delta_n200','mean')).reset_index()
    return result,paired


def paired_strata(work,strata):
    rows=[]
    for sd in sorted((Path(work)/'stations').glob('*')):
        if not (sd/'complete.json').exists():continue
        sid=sd.name;ledger=pd.DataFrame(json.loads((sd/'ledger.json').read_text()))
        for col,default in [('version','all'),('mode','normal'),('lag',6)]:
            if col not in ledger:ledger[col]=default
            else:ledger[col]=ledger[col].fillna(default)
        normal=ledger[ledger.status.eq('completed')&ledger['mode'].eq('normal')&ledger.lag.eq(6)&ledger.version.eq('all')]
        for r in normal[normal.experiment.isin(['E1','E3'])&~normal.arm.str.startswith('R')].itertuples():
            if r.experiment=='E3':reference='E1';reference_arm='W0';label='weather_added_to_observed_rain'
            else:reference='E0';reference_arm='W0';label='observed_rain_added'
            ref=normal[normal.experiment.eq(reference)&normal.arm.eq(reference_arm)&normal.seed.eq(r.seed)]
            if len(ref)!=1:continue
            a=strata[strata.fit_key.eq(r.fit_key)];b=strata[strata.fit_key.eq(ref.iloc[0].fit_key)]
            p=a.merge(b,on=['scope','fold','horizon','stratum'],suffixes=('_a','_b'),validate='one_to_one')
            if not p.n_a.eq(p.n_b).all():raise ValueError('Stratum pairing has different anchors')
            for (scope,h,group),g in p.groupby(['scope','horizon','stratum']):
                n=g.n_a.sum()
                if not n:continue
                rows.append(dict(station_id=sid,experiment=r.experiment,arm=r.arm,seed=int(r.seed),scope=scope,horizon=int(h),stratum=group,
                    effect=label,n=int(n),delta_rmse_mm=float(np.sqrt((g.rmse_mm_a**2*g.n_a).sum()/n)-np.sqrt((g.rmse_mm_b**2*g.n_b).sum()/n)),
                    delta_n200=int(g.n200_a.sum()-g.n200_b.sum())))
    return pd.DataFrame(rows)


def feature_windows(work,ledger):
    rows=[]
    for key in sorted(ledger.fit_key.dropna().unique()):
        folder=Path(work)/'fits'/key;m=json.loads((folder/'identity.json').read_text());frame=read_importance(folder/'importance.csv')
        for horizon,part in frame.groupby('horizon'):
            total=part.total_gain.sum()
            for r in part.itertuples():
                f=r.feature
                if f.endswith('_missing'):kind='missing_mask'
                elif f.startswith('weather_grid'):kind='forecast_interval_'+f.rsplit('_rain_',1)[-1]
                elif f.startswith('rain_') and '_sum' in f:kind='observed_accumulation_'+f.rsplit('_sum',1)[-1]
                elif f.startswith('rain_'):kind='observed_history_lag'
                elif f.startswith('water_lag_'):
                    lag=int(f.rsplit('_',1)[-1]);kind='water_lag_'+('0' if lag==0 else '1_6' if lag<=6 else '7_24' if lag<=24 else '25_48' if lag<=48 else '49_95')
                elif f.startswith('water_'):kind='water_history_statistics'
                else:kind=f
                rows.append(dict(station_id=m['metadata']['station_id'],fit_key=key,horizon=horizon,seed=m['seed'],feature_family=kind,
                                 gain_fraction=float(r.total_gain/total) if total else 0.))
    frame=pd.DataFrame(rows)
    return frame.groupby(['station_id','fit_key','horizon','seed','feature_family']).gain_fraction.sum().reset_index()


def predicted_change_summary(ledger,strata):
    """Compare correction amplitude on identical retrospective high-change groups."""
    cases=ledger[ledger.status.eq('completed')&ledger['mode'].eq('normal')&ledger.lag.eq(6)&ledger.version.eq('all')&ledger.experiment.isin(['E0','E1','E3'])]
    part=strata[strata.stratum.isin(['all','rapid_rise','rapid_fall','high_water','event_hours'])]
    paired=cases[['station_id','experiment','arm','seed','fit_key']].merge(part,on=['station_id','fit_key'],validate='many_to_many')
    rows=[]
    for key,g in paired.groupby(['station_id','experiment','arm','scope','horizon','stratum','seed']):
        n=g.n.sum()
        if not n:continue
        actual=float((g.mean_abs_actual_change_mm*g.n).sum()/n);pred=float((g.mean_abs_predicted_change_mm*g.n).sum()/n)
        rows.append(dict(zip(['station_id','experiment','arm','scope','horizon','stratum','seed'],key),n=int(n),
            actual_abs_change_mm=actual,predicted_abs_change_mm=pred,amplitude_ratio_pct=100*pred/actual if actual>0 else np.nan))
    seed=pd.DataFrame(rows)
    return seed.groupby(['station_id','experiment','arm','scope','horizon','stratum']).agg(
        seeds=('seed','nunique'),n=('n','first'),actual_abs_change_mm=('actual_abs_change_mm','mean'),
        predicted_abs_change_mm=('predicted_abs_change_mm','mean'),amplitude_ratio_pct=('amplitude_ratio_pct','mean')).reset_index()


def target_variability(work,cutoff):
    rows=[]
    for sd in sorted((Path(work)/'stations').glob('*')):
        if not (sd/'complete.json').exists():continue
        pop=pd.read_parquet(sd/'population.parquet');times=pd.to_datetime(pop.anchor_utc,utc=True)
        masks={c[:-6]:pop[c].to_numpy(dtype=bool) for c in pop if c.endswith('_train')}
        masks['full_pre_cutoff']=times<pd.Timestamp(cutoff)-pd.Timedelta(hours=24)
        for fold,mask in masks.items():
            for h in [6,12]:
                v=pop.loc[mask,f'dh{h}_mm']
                rows.append(dict(station_id=sd.name,fold=fold,horizon=h,n=len(v),std_mm=float(v.std()),
                    minimum_mm=float(v.min()),maximum_mm=float(v.max()),unique_values=int(v.nunique()),
                    zero_change_pct=float(100*v.eq(0).mean()),abs_change_ge200_pct=float(100*v.abs().ge(200).mean())))
    return pd.DataFrame(rows)
