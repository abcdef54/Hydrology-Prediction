"""Paired seed/event bootstrap for preselected finalists; no iid-hour confidence claims."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def event_bootstrap(work,strata,replicates=200):
    rows=[];rng=np.random.default_rng(20261027)
    event=strata[strata.stratum.str.startswith('event_')&strata.stratum.ne('event_hours')]
    for sd in sorted((Path(work)/'stations').glob('*')):
        if not (sd/'complete.json').exists():continue
        sid=sd.name;selected=json.loads((sd/'finalists.json').read_text())['arms']
        ledger=pd.DataFrame(json.loads((sd/'ledger.json').read_text()))
        if 'version' not in ledger:ledger['version']=None
        eligible=ledger[ledger.status.eq('completed')&ledger['mode'].eq('normal')&ledger.lag.eq(6)&ledger.version.isna()]
        for arm in list(dict.fromkeys(['W1',*selected])):
            for scope in ['validation','exploratory']:
                for h in [6,12]:
                    datasets=[]
                    for seed in [17,29,42,71,103]:
                        current=eligible[(eligible.experiment=='E3')&(eligible.arm==arm)&(eligible.seed==seed)]
                        ref=eligible[(eligible.experiment=='E1')&(eligible.seed==seed)]
                        if len(current)!=1 or len(ref)!=1:continue
                        a=event[(event.fit_key==current.iloc[0].fit_key)&(event.scope==scope)&(event.horizon==h)]
                        b=event[(event.fit_key==ref.iloc[0].fit_key)&(event.scope==scope)&(event.horizon==h)]
                        paired=a.merge(b,on=['scope','fold','horizon','stratum'],suffixes=('_a','_b'),validate='one_to_one')
                        if len(paired):
                            if not paired.n_a.eq(paired.n_b).all():raise ValueError('Paired event populations differ')
                            datasets.append(paired)
                    if not datasets:continue
                    n_events=min(d.stratum.nunique() for d in datasets)
                    if n_events<3:
                        rows.append(dict(station_id=sid,arm=arm,scope=scope,horizon=h,event_count=n_events,status='insufficient_events',rmse_delta_ci_low=np.nan,rmse_delta_ci_high=np.nan));continue
                    arrays=[]
                    for d in datasets:
                        v=d.assign(sse_a=d.rmse_mm_a**2*d.n_a,sse_b=d.rmse_mm_b**2*d.n_b)
                        arrays.append(v.groupby('stratum')[['n_a','sse_a','sse_b']].sum().to_numpy())
                    estimates=[]
                    for _ in range(replicates):
                        values=arrays[rng.integers(len(arrays))]
                        sample=values[rng.integers(len(values),size=len(values))].sum(axis=0)
                        estimates.append(np.sqrt(sample[1]/sample[0])-np.sqrt(sample[2]/sample[0]))
                    lo,hi=np.quantile(estimates,[.025,.975])
                    rows.append(dict(station_id=sid,arm=arm,scope=scope,horizon=h,event_count=n_events,
                        status='algorithmic_event_seed_bootstrap_not_physical_independence_proof',rmse_delta_ci_low=float(lo),rmse_delta_ci_high=float(hi)))
    return pd.DataFrame(rows)
