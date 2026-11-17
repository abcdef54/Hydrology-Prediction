"""Paired seed/fold metrics and retrospective hydrological strata without selection leakage."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .training import scalar_metrics

KEYS=['station_id','station_name','experiment','arm','mode','lag','version','diagnostic']


def gather(work):
    parts=[];ledgers=[]
    for sd in sorted((Path(work)/'stations').glob('*')):
        if (sd/'metrics.csv').exists():parts.append(pd.read_csv(sd/'metrics.csv',dtype={'station_id':str}))
        if (sd/'ledger.json').exists():ledgers.extend(json.loads((sd/'ledger.json').read_text()))
    frame=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()
    for name,default in [('version','all'),('diagnostic','principal')]:
        if name not in frame:frame[name]=default
        else:frame[name]=frame[name].fillna(default)
    return frame,pd.DataFrame(ledgers)


def aggregate_seeds(metrics):
    rows=[]
    for key,g in metrics.groupby(KEYS+['scope','horizon','seed'],dropna=False):
        n=g.n.sum()
        rows.append(dict(zip(KEYS+['scope','horizon','seed'],key),n=int(n),
            rmse_mm=float(np.sqrt((g.rmse_mm**2*g.n).sum()/n)) if n else np.nan,
            mae_mm=float((g.mae_mm*g.n).sum()/n) if n else np.nan,
            n200=int(g.n200.sum()),n200_pct=float(100*g.n200.sum()/n) if n else np.nan))
    seed=pd.DataFrame(rows)
    summary=seed.groupby(KEYS+['scope','horizon'],dropna=False).agg(
        seeds=('seed','nunique'),n=('n','first'),rmse_mean=('rmse_mm','mean'),rmse_sd=('rmse_mm','std'),
        rmse_min=('rmse_mm','min'),rmse_max=('rmse_mm','max'),mae_mean=('mae_mm','mean'),
        n200_mean=('n200','mean'),n200_min=('n200','min'),n200_max=('n200','max'),n200_pct_mean=('n200_pct','mean')).reset_index()
    return seed,summary


def paired_incremental(seed):
    principal=seed[(seed['mode']=='normal')&(seed.lag==6)&seed.version.eq('all')]
    rows=[]
    for r in seed.itertuples():
        if r.version!='all':continue
        if r.mode!='normal' or r.lag!=6:
            e='E3';arm=r.arm;label=r.diagnostic
        elif r.experiment=='E3':e='E1';arm='W0';label='weather_added_to_observed_rain'
        elif r.experiment=='E2':e='E0';arm='W0';label='weather_added_to_water_only'
        elif r.experiment=='E4':e='E3';arm=r.arm;label='remove_12_24h_forecast'
        elif r.experiment=='E5':e='E3';arm=r.arm;label='train_selected_observed_rain'
        elif r.experiment=='E1':e='E0';arm='W0';label='observed_rain_added'
        else:continue
        before=principal[(principal.station_id==r.station_id)&(principal.experiment==e)&(principal.arm==arm)&
                         (principal.scope==r.scope)&(principal.horizon==r.horizon)&(principal.seed==r.seed)]
        if len(before)!=1:continue
        b=before.iloc[0]
        if r.n!=b.n:raise ValueError('Incremental comparison has mismatched sample counts')
        rows.append(dict(station_id=r.station_id,experiment=r.experiment,arm=r.arm,mode=r.mode,lag=r.lag,
                         diagnostic=label,scope=r.scope,horizon=r.horizon,seed=r.seed,n=r.n,
                         reference_experiment=e,reference_arm=arm,reference_rmse=b.rmse_mm,
                         delta_rmse_mm=r.rmse_mm-b.rmse_mm,delta_n200=r.n200-b.n200,
                         delta_mae_mm=r.mae_mm-b.mae_mm))
    return pd.DataFrame(rows)


def event_labels(level,threshold):
    """Retrospective episodes with explicit 12h merge; not independent flood declarations."""
    flagged=(level.ge(threshold)|(level-level.shift(6)).abs().ge(100)|(level-level.shift(12)).abs().ge(200))&level.notna()
    stamps=level.index[flagged];event=pd.Series(0,index=level.index,dtype=int);records=[]
    if not len(stamps):return event,records
    clusters=np.cumsum(np.r_[True,(stamps[1:]-stamps[:-1])>pd.Timedelta(hours=12)])
    for number in np.unique(clusters):
        part=stamps[clusters==number];a=part[0];b=part[-1]
        if b-a<pd.Timedelta(hours=2):continue
        event.loc[a:b]=int(number)
        records.append(dict(event=int(number),start_utc=str(a),end_utc=str(b),span_hours=int((b-a)/pd.Timedelta(hours=1))+1,
                            peak_water_mm=float(level.loc[a:b].max()),definition='train q95 OR abs past dh6>=100 OR abs past dh12>=200; merge gaps<=12h'))
    return event,records


def evaluate_strata(work,source):
    output=Path(work)/'analysis_cache';output.mkdir(exist_ok=True)
    rows=[];peaks=[];catalogue=[]
    for sd in sorted((Path(work)/'stations').glob('*')):
        if not (sd/'complete.json').exists():continue
        sid=sd.name;cached=output/f'events_change_v2_{sid}.parquet';peakcache=output/f'peaks_change_v2_{sid}.parquet'
        # Cache per completed station; source hashes are frozen in the execution manifest.
        pop=pd.read_parquet(sd/'population.parquet');pop['anchor_utc']=pd.to_datetime(pop.anchor_utc,utc=True)
        first_train=next(c for c in pop if c.endswith('_train'))
        threshold=float(pop.loc[pop[first_train],'water_now_mm'].quantile(.95))
        episode,catalog=event_labels(source.level[sid],threshold)
        catalogue.extend(dict(station_id=sid,threshold_q95_mm=threshold,**r) for r in catalog)
        fits=sorted(set(r['fit_key'] for r in json.loads((sd/'ledger.json').read_text()) if r.get('fit_key')))
        if cached.exists() and peakcache.exists():
            rows.extend(pd.read_parquet(cached).to_dict('records'));peaks.extend(pd.read_parquet(peakcache).to_dict('records'));continue
        station_rows=[];station_peaks=[]
        for key in fits:
            folder=Path(work)/'fits'/key;identity=json.loads((folder/'identity.json').read_text());meta=identity['metadata']
            pred=pd.read_parquet(folder/'predictions.parquet');pred['anchor_utc']=pd.to_datetime(pred.anchor_utc,utc=True)
            for (scope,fold,h),part in pred.groupby(['scope','fold','horizon']):
                actual=part.actual_mm.to_numpy();p=part.prediction_mm.to_numpy();base=part.persistence_mm.to_numpy()
                target=pd.DatetimeIndex(part.anchor_utc)+pd.Timedelta(hours=int(h));dh=actual-base
                kinds=source.weather[meta['lag']][f'{BASE}|archive_kind'].reindex(pd.DatetimeIndex(part.anchor_utc))
                labels=episode.reindex(target).fillna(0).to_numpy(dtype=int)
                season=target.tz_convert('Asia/Ho_Chi_Minh').month.isin([9,10,11,12])
                masks={'all':np.ones(len(part),dtype=bool),'rising':dh>0,'falling':dh<0,
                    'rapid_rise':dh>=(100 if h==6 else 200),'rapid_fall':dh<=(-100 if h==6 else -200),
                    'high_water':actual>=threshold,'flood_months':season,'other_months':~season,
                    'event_hours':labels>0,'hindcast_49r1':kinds.str.contains('49r1').to_numpy(),
                    'archive_50r1':kinds.str.contains('50r1').to_numpy()}
                for name,mask in masks.items():
                    stats=scalar_metrics(actual[mask],p[mask]);ref=scalar_metrics(actual[mask],base[mask])
                    station_rows.append(dict(station_id=sid,fit_key=key,scope=scope,fold=fold,horizon=int(h),stratum=name,
                        **stats,persistence_rmse=ref['rmse_mm'],persistence_n200=ref['n200'],
                        mean_abs_predicted_change_mm=float(np.mean(abs(p[mask]-base[mask]))) if mask.any() else None,
                        mean_abs_actual_change_mm=float(np.mean(abs(actual[mask]-base[mask]))) if mask.any() else None,
                        direction_accuracy=float((np.sign(p[mask]-base[mask])==np.sign(actual[mask]-base[mask])).mean()) if mask.any() else None,
                        errors_fixed=int(((abs(actual[mask]-base[mask])>200)&(abs(actual[mask]-p[mask])<=200)).sum()),
                        errors_introduced=int(((abs(actual[mask]-base[mask])<=200)&(abs(actual[mask]-p[mask])>200)).sum())))
                for ev in np.unique(labels[labels>0]):
                    mask=labels==ev
                    stats=scalar_metrics(actual[mask],p[mask]);ref=scalar_metrics(actual[mask],base[mask])
                    station_rows.append(dict(station_id=sid,fit_key=key,scope=scope,fold=fold,horizon=int(h),stratum=f'event_{ev}',
                        **stats,persistence_rmse=ref['rmse_mm'],persistence_n200=ref['n200']))
                    if mask.sum()>=6:
                        truth=actual[mask];forecast=p[mask];times=target[mask]
                        station_peaks.append(dict(station_id=sid,fit_key=key,scope=scope,fold=fold,horizon=int(h),event=int(ev),n=int(mask.sum()),
                            peak_amplitude_error_mm=float(forecast.max()-truth.max()),
                            error_at_actual_peak_mm=float(forecast[np.argmax(truth)]-truth.max()),
                            peak_timing_error_hours=float((times[np.argmax(forecast)]-times[np.argmax(truth)])/pd.Timedelta(hours=1))))
        pd.DataFrame(station_rows).to_parquet(cached,index=False);pd.DataFrame(station_peaks).to_parquet(peakcache,index=False)
        rows.extend(station_rows);peaks.extend(station_peaks)
    return pd.DataFrame(rows),pd.DataFrame(peaks),pd.DataFrame(catalogue)


def factorial_interactions(seed):
    d=seed[(seed.station_id=='0074647403')&(seed.experiment=='E3')&(seed['mode']=='normal')&seed.lag.eq(6)&seed.version.eq('all')]
    pairs=[('A_Luoi','Da_Vien','W1','W2','W6'),('A_Luoi','Thuan_An','W1','W5','W3'),('Da_Vien','Thuan_An','W2','W5','W7')]
    rows=[]
    for (scope,h,s),part in d.groupby(['scope','horizon','seed']):
        lookup=part.set_index('arm')
        for a,b,wa,wb,wab in pairs:
            if not {'W0',wa,wb,wab}<=set(lookup.index):continue
            mse=lookup.rmse_mm**2
            rows.append(dict(scope=scope,horizon=h,seed=s,grid_a=a,grid_b=b,
                mse_gain_synergy=float(mse[wa]+mse[wb]-mse[wab]-mse['W0']),
                n200_gain_synergy=float(lookup.n200[wa]+lookup.n200[wb]-lookup.n200[wab]-lookup.n200['W0']),
                interpretation='positive = joint gain beyond sum of separate refitted-model gains; not causal hydrological interaction'))
    return pd.DataFrame(rows)

# Shared explicit reference identity, not the current nearest requested coordinate.
from .data import BASE


def persistence_baselines(work):
    """Persistence needs no fitting or seed; reuse the exact common prediction anchors."""
    rows=[]
    for sd in sorted((Path(work)/'stations').glob('*')):
        if not (sd/'complete.json').exists():continue
        ledger=pd.DataFrame(json.loads((sd/'ledger.json').read_text()))
        chosen=ledger[ledger.status.eq('completed')&ledger.experiment.eq('E0')&ledger.seed.eq(42)].iloc[0]
        pred=pd.read_parquet(Path(work)/'fits'/chosen.fit_key/'predictions.parquet')
        for (scope,h),part in pred.groupby(['scope','horizon']):
            if part.anchor_utc.duplicated().any():raise ValueError('Persistence population contains overlapping validation anchors')
            rows.append(dict(station_id=sd.name,scope=scope,horizon=int(h),**scalar_metrics(part.actual_mm,part.persistence_mm)))
    return pd.DataFrame(rows)
