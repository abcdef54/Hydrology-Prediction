"""Explicit experiment ledger, train-only challengers and bounded diagnostic scope."""
import numpy as np
import pandas as pd
from .data import BASE
from .training import write_json


def arms_for(source,sid):
    if sid=='0074647403':return {r['arm']:r['grid_ids'] for r in source.spec['extended_design']['da_vien_factorial_arms']}
    return {r.arm:[] if pd.isna(r.grid_ids) else r.grid_ids.split(';') for r in source.candidates[source.candidates.water_station_id.eq(sid)].itertuples()}


def discover(source,sid,x,y,folds):
    train=folds[0]['train'];fixed,selected,rain_scores=source.choose_rain(sid,x.index,train)
    controls=pd.concat([x[['water_lag_0','water_change_3','water_change_6']],source.rain_features(fixed,x.index),source.weather_features([BASE],x.index)],axis=1)
    # No imputation fitted on validation: means and nuisance projection come only from earliest TRAIN.
    c=controls.loc[train].to_numpy(dtype=float);means=np.nanmean(c,axis=0);means=np.nan_to_num(means)
    c=np.where(np.isfinite(c),c,means);sd=c.std(axis=0);sd[sd==0]=1
    design=np.column_stack([np.ones(len(c)),(c-means)/sd])
    targets=y.loc[train].to_numpy(dtype=float)
    residual_y=targets-design@np.linalg.lstsq(design,targets,rcond=None)[0]
    scores={}
    for gid in source.grids.grid_id:
        if gid==BASE:continue
        f=source.weather_features([gid],x.index).filter(regex='rain_.*(?<!missing)$').loc[train].to_numpy(dtype=float)
        residual=f-design@np.linalg.lstsq(design,f,rcond=None)[0]
        correlations=[]
        for j in range(residual.shape[1]):
            for k in range(2):
                if residual[:,j].std()>1e-6 and residual_y[:,k].std()>1e-6:
                    correlations.append(abs(float(np.corrcoef(residual[:,j],residual_y[:,k])[0,1])))
        scores[gid]=float(np.mean(correlations)) if correlations else 0.
    arms=arms_for(source,sid);excluded=[g for g in scores if g not in arms.get('W4',[])]
    excluded.sort(key=lambda g:(-scores[g],g))
    challenger=excluded[0] if excluded and scores[excluded[0]]>=.05 else None
    return dict(station_id=sid,fixed_rain=fixed,selected_rain=selected,
                rainfall_train_correlations={k:float(v) if np.isfinite(v) else None for k,v in rain_scores.items()},
                incremental_train_scores=scores,challenger=challenger,challenger_rule='highest mean abs residual correlation across 4 windows/2 horizons among excluded grids; >=0.05',
                discovery_fold=folds[0]['id'],discovery_end=str(x.index[train].max()),
                validation_not_used=True,weather_candidate_history='Stage 3B W0-W4 were discovered in pre-cutoff EDA, so inner-fold estimates are selection-informed')


def principal(source,sid):
    tasks=[dict(experiment='E0',arm='W0',grids=[]),dict(experiment='E1',arm='W0',grids=[])]
    for arm,ids in arms_for(source,sid).items():
        if not ids:
            tasks.append(dict(experiment='E3',arm=arm,grids=[]))
            continue
        for e in ['E2','E3','E4']:tasks.append(dict(experiment=e,arm=arm,grids=ids))
    if sid=='751117':
        for arm,ids in [('R0',[]),('R1',[BASE]),('R2',['ecmwf_ifs:16.274164:107.863250']),('R3',[BASE,'ecmwf_ifs:16.274164:107.863250'])]:
            tasks.append(dict(experiment='E3',arm=arm,grids=ids,diagnostic='regional_Trung_Lang'))
    return tasks


def choose_finalists(records):
    frame=pd.DataFrame(records)
    frame=frame[frame.experiment.eq('E3')&frame.arm.str.startswith('W')&frame['mode'].eq('normal')&frame.scope.eq('validation')]
    scores=frame.groupby('arm').agg(n200=('n200','mean'),rmse=('rmse_mm','mean')).reset_index()
    keep=[]
    for r in scores.itertuples():
        dominated=((scores.n200<=r.n200)&(scores.rmse<=r.rmse)&((scores.n200<r.n200)|(scores.rmse<r.rmse))).any()
        if not dominated:keep.append(r.arm)
    return scores[scores.arm.isin(keep)].sort_values(['n200','rmse','arm']).arm.head(2).tolist(),scores


def diagnostics(source,sid,finalists,discovery):
    arms=arms_for(source,sid);chosen=list(dict.fromkeys(['W1',*finalists]));tasks=[]
    for arm in chosen:
        ids=arms[arm]
        if not ids:continue
        tasks.append(dict(experiment='E5',arm=arm,grids=ids,diagnostic='D5'))
        for mode,label in [('availability','D1'),('strict','D6')]:tasks.append(dict(experiment='E3',arm=arm,grids=ids,mode=mode,diagnostic=label))
        if len(ids)>=2:tasks.append(dict(experiment='E3',arm=arm,grids=ids,mode='duplicate',diagnostic='D2'))
        for lag in [9,12]:tasks.append(dict(experiment='E3',arm=arm,grids=ids,lag=lag,diagnostic='D7'))
    for arm in list(dict.fromkeys(['W1',*finalists])):
        ids=arms[arm]
        if ids:
            for version in ['49r1','50r1']:tasks.append(dict(experiment='E3',arm=arm,grids=ids,version=version,diagnostic='D8'))
    if discovery['challenger']:
        gid=discovery['challenger']
        tasks.extend([dict(experiment='E3',arm='C1',grids=[gid],diagnostic='D9'),dict(experiment='E3',arm='C2',grids=[BASE,gid],diagnostic='D9')])
    return tasks
