"""Grouped tree gain, block-permutation sensitivity, and TreeSHAP interactions."""
from pathlib import Path
import json,logging
import numpy as np
import pandas as pd
import xgboost as xgb
from .training import scalar_metrics
LOG=logging.getLogger(__name__)


def feature_group(feature,metadata):
    if feature=='weather_age_hours':return 'forecast_age'
    if feature.endswith('_missing'):return 'missing_masks'
    if feature.startswith('water_'):return 'water_history'
    if feature.startswith('rain_'):return 'observed_rain:'+feature.split('_')[1]
    if feature.startswith('weather_grid'):
        position=int(feature.split('_')[1][4:]);return 'forecast:'+metadata['grids'][position]
    return 'other'


def read_importance(path):
    try:return pd.read_csv(path)
    except pd.errors.EmptyDataError:return pd.DataFrame(columns=['horizon','feature','total_gain'])


def grouped_gain(work,ledger):
    rows=[]
    for key in sorted(set(ledger.fit_key.dropna())):
        folder=Path(work)/'fits'/key;identity=json.loads((folder/'identity.json').read_text());meta=identity['metadata']
        frame=read_importance(folder/'importance.csv')
        frame['group']=[feature_group(f,meta) for f in frame.feature]
        for h in [6,12]:
            part=frame[frame.horizon.eq(h)]
            if part.empty:
                rows.append(dict(station_id=meta['station_id'],fit_key=key,seed=identity['seed'],horizon=h,group='constant_prediction_no_splits',total_gain=0.,gain_fraction=0.))
                continue
            total=part.total_gain.sum()
            for group,g in part.groupby('group'):
                rows.append(dict(station_id=meta['station_id'],fit_key=key,seed=identity['seed'],horizon=h,
                    group=group,total_gain=float(g.total_gain.sum()),gain_fraction=float(g.total_gain.sum()/total) if total else 0.))
    return pd.DataFrame(rows)


def explain_finalists(work,source,c):
    folder=Path(work)/'explanations';folder.mkdir(exist_ok=True);shap_rows=[];inter_rows=[];perm_rows=[]
    for sd in sorted((Path(work)/'stations').glob('*')):
        if not (sd/'complete.json').exists():continue
        sid=sd.name
        cache=folder/f'{sid}.json'
        if cache.exists():
            saved=json.loads(cache.read_text());shap_rows+=saved['shap'];inter_rows+=saved['interactions'];perm_rows+=saved['permutation'];continue
        finalists=json.loads((sd/'finalists.json').read_text())['arms']
        ledger=json.loads((sd/'ledger.json').read_text())
        # One fixed seed for explanations, never the best seed; all five seeds already have gain/ablation outcomes.
        selected={r['fit_key'] for r in ledger if r.get('seed')==42 and r.get('status')=='completed' and r.get('mode')=='normal' and r.get('lag',6)==6
                  and r.get('version') is None and ((r['experiment']=='E1') or (r['experiment']=='E3' and r['arm'] in ['W1',*finalists]))}
        base,y,h=source.base(sid);local_shap=[];local_inter=[];local_perm=[]
        for key in sorted(selected):
            fit=Path(work)/'fits'/key;identity=json.loads((fit/'identity.json').read_text());meta=identity['metadata']
            x=source.features(base,meta['grids'],meta['rain_ids'],meta['experiment'],meta['mode'],meta['lag'])
            folds=source.folds(x.index);last=folds[-1]
            for horizon in c['horizons']:
                for scope in ['validation','exploratory']:
                    if scope=='validation':mask=last['val'];modelpath=fit/f'{last["id"]}_h{horizon}.ubj'
                    else:mask=x.index>=pd.Timestamp(c['cutoff']);modelpath=fit/f'final_h{horizon}.ubj'
                    if not mask.any():continue
                    model=xgb.Booster();model.load_model(modelpath);model.set_param({'device':'cpu','nthread':4})
                    values=x.loc[mask].to_numpy(dtype=np.float32);truth=y.loc[mask,f'h{horizon}'].to_numpy()
                    dm=xgb.DMatrix(values,feature_names=x.columns.tolist(),nthread=4);normal=model.predict(dm)
                    baseline=scalar_metrics(truth,normal)
                    groups={}
                    for j,col in enumerate(x.columns):groups.setdefault(feature_group(col,meta),[]).append(j)
                    # Circular 24/72/168-row shifts preserve blocks within this diagnostic scope;
                    # these perturbed forecasts are NOT operational forecasts or feature-selection trials.
                    for group,indices in groups.items():
                        for shift in [24,72,168]:
                            if shift>=len(values):continue
                            changed=values.copy();changed[:,indices]=np.roll(values[:,indices],shift,axis=0)
                            pred=model.predict(xgb.DMatrix(changed,feature_names=x.columns.tolist(),nthread=4))
                            stats=scalar_metrics(truth,pred)
                            local_perm.append(dict(station_id=sid,fit_key=key,horizon=horizon,scope=scope,group=group,
                                shift_rows=shift,n=len(truth),delta_rmse_mm=stats['rmse_mm']-baseline['rmse_mm'],
                                delta_n200=stats['n200']-baseline['n200'],interpretation='retrospective block-shift sensitivity; correlated features and nonlinear extrapolation limit causal meaning'))
                    positions=np.linspace(0,len(values)-1,min(c['explanation_samples'],len(values)),dtype=int)
                    ds=xgb.DMatrix(values[positions],feature_names=x.columns.tolist(),nthread=4)
                    contrib=model.predict(ds,pred_contribs=True)
                    for group,indices in groups.items():
                        local_shap.append(dict(station_id=sid,fit_key=key,horizon=horizon,scope=scope,group=group,n=len(positions),
                            mean_abs_shap_mm=float(np.abs(contrib[:,indices].sum(axis=1)).mean()),seed=42))
                    interaction=model.predict(ds,pred_interactions=True)
                    labels=list(groups)
                    for j,a in enumerate(labels):
                        for b in labels[j+1:]:
                            block=interaction[:,groups[a],:][:,:,groups[b]].sum(axis=(1,2))
                            local_inter.append(dict(station_id=sid,fit_key=key,horizon=horizon,scope=scope,group_a=a,group_b=b,n=len(positions),
                                mean_abs_interaction_mm=float(abs(block).mean()),seed=42))
        saved={'shap':local_shap,'interactions':local_inter,'permutation':local_perm}
        cache.write_text(json.dumps(saved,ensure_ascii=False,allow_nan=False))
        shap_rows+=local_shap;inter_rows+=local_inter;perm_rows+=local_perm
        LOG.info('EXPLANATIONS %s fits=%s',sid,len(selected))
    return pd.DataFrame(shap_rows),pd.DataFrame(inter_rows),pd.DataFrame(perm_rows)
