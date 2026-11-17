"""Rain-episode robustness of preselected feature effects, separate from tidal water episodes."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .training import scalar_metrics


def evaluate(work,source):
    folder=Path(work)/'analysis_cache';folder.mkdir(exist_ok=True);rows=[];effects=[]
    labels=pd.Series(0,index=source.index,dtype=int)
    for ep in source.rain_events.itertuples():
        start=pd.Timestamp(ep.start_utc);end=pd.Timestamp(ep.end_utc)
        if labels.loc[start:end].ne(0).any():raise ValueError('Rainfall episode windows overlap')
        labels.loc[start:end]=int(ep.event_id)
    for sd in sorted((Path(work)/'stations').glob('*')):
        if not (sd/'complete.json').exists():continue
        cache=folder/f'rain_events_{sd.name}.csv';effectcache=folder/f'rain_event_effects_{sd.name}.csv'
        if cache.exists() and effectcache.exists():
            rows.extend(pd.read_csv(cache,dtype={'station_id':str}).to_dict('records'))
            try:effects.extend(pd.read_csv(effectcache,dtype={'station_id':str}).to_dict('records'))
            except pd.errors.EmptyDataError:pass
            continue
        ledger=pd.DataFrame(json.loads((sd/'ledger.json').read_text()))
        selected=json.loads((sd/'finalists.json').read_text())['arms']
        if 'version' not in ledger:ledger['version']=None
        cases=ledger[ledger.status.eq('completed')&ledger['mode'].eq('normal')&ledger.lag.eq(6)&ledger.version.isna()&
                     (ledger.experiment.isin(['E0','E1'])|(ledger.experiment.eq('E3')&ledger.arm.isin(['W1',*selected])))]
        computed=[]
        for key in sorted(cases.fit_key.unique()):
            pred=pd.read_parquet(Path(work)/'fits'/key/'predictions.parquet')
            pred['anchor_utc']=pd.to_datetime(pred.anchor_utc,utc=True)
            target=pd.DatetimeIndex(pred.anchor_utc)+pd.to_timedelta(pred.horizon,unit='h')
            pred['rain_event']=labels.reindex(target).fillna(0).to_numpy(dtype=int)
            for (scope,h,ev),g in pred[pred.rain_event.gt(0)].groupby(['scope','horizon','rain_event']):
                if len(g)<6:continue
                computed.append(dict(station_id=sd.name,fit_key=key,scope=scope,horizon=int(h),rain_event=int(ev),
                    **scalar_metrics(g.actual_mm,g.prediction_mm)))
        frame=pd.DataFrame(computed)
        joined=cases[['station_id','experiment','arm','seed','fit_key']].merge(frame,on=['station_id','fit_key'],validate='many_to_many')
        local=[]
        for (scope,h,seed),part in joined.groupby(['scope','horizon','seed']):
            ref=part[part.experiment.eq('E1')][['rain_event','n','rmse_mm','n200']]
            for arm,g in part[part.experiment.eq('E3')].groupby('arm'):
                pair=g.merge(ref,on='rain_event',suffixes=('_a','_b'),validate='one_to_one')
                if not pair.n_a.eq(pair.n_b).all():raise ValueError('Rain-event comparison has different prediction anchors')
                for r in pair.itertuples():
                    local.append(dict(station_id=sd.name,arm=arm,scope=scope,horizon=int(h),seed=int(seed),rain_event=r.rain_event,n=int(r.n_a),
                        delta_rmse_mm=r.rmse_mm_a-r.rmse_mm_b,delta_n200=int(r.n200_a-r.n200_b),
                        reference_rmse_mm=r.rmse_mm_b,candidate_rmse_mm=r.rmse_mm_a))
        joined.to_csv(cache,index=False);pd.DataFrame(local).to_csv(effectcache,index=False)
        rows.extend(joined.to_dict('records'));effects.extend(local)
    raw=pd.DataFrame(rows);paired=pd.DataFrame(effects)
    mean=paired.groupby(['station_id','arm','scope','horizon','rain_event']).agg(n=('n','first'),delta_rmse_mm=('delta_rmse_mm','mean'),delta_n200=('delta_n200','mean')).reset_index()
    summary=mean.groupby(['station_id','arm','scope','horizon']).agg(
        evaluated_episodes=('rain_event','nunique'),episodes_lower_rmse=('delta_rmse_mm',lambda s:int(s.lt(0).sum())),
        episodes_higher_rmse=('delta_rmse_mm',lambda s:int(s.gt(0).sum())),
        min_episode_delta_rmse_mm=('delta_rmse_mm','min'),max_episode_delta_rmse_mm=('delta_rmse_mm','max')).reset_index()
    return raw,paired,summary
