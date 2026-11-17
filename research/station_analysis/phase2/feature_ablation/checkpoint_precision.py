"""Fixed-seed numerical tie-break sensitivity; never select features from post-cutoff data."""
from pathlib import Path
import json,logging
import numpy as np
import pandas as pd
import xgboost as xgb
from .data import Source,load_config
from .training import Thermal,Guard,scalar_metrics,write_json
LOG=logging.getLogger(__name__)


def scaled_metric(predicted,dmatrix):
    e=np.asarray(predicted)-dmatrix.get_label();rmse=float(np.sqrt(np.mean(e**2)))
    # Count priority is unchanged: the bounded fractional part remains strictly below one.
    return 'n200_bounded_rmse',float((abs(e)>200).sum()+rmse/(rmse+1))



def summarize(work,folder,result):
    aggregated=[]
    for (sid,key,h),g in result.groupby(['station_id','fit_key','horizon']):
        n=g.n.sum()
        aggregated.append(dict(station_id=str(sid),fit_key=key,horizon=int(h),n=int(n),
            old_rmse=float(np.sqrt((g.old_rmse_mm**2*g.n).sum()/n)),new_rmse=float(np.sqrt((g.new_rmse_mm**2*g.n).sum()/n)),
            old_n200=int(g.old_n200.sum()),new_n200=int(g.new_n200.sum())))
    a=pd.DataFrame(aggregated);effects=[]
    for sd in sorted((work/'stations').glob('*')):
        if not (sd/'complete.json').exists():continue
        ledger=pd.DataFrame(json.loads((sd/'ledger.json').read_text()))
        normal=ledger[ledger.status.eq('completed')&ledger.seed.eq(42)&ledger['mode'].eq('normal')&ledger.lag.eq(6)]
        if 'version' in normal:normal=normal[normal.version.isna()]
        reference=normal[normal.experiment.eq('E1')].iloc[0].fit_key
        for row in normal[normal.experiment.eq('E3')&normal.arm.str.startswith('W')].itertuples():
            for h in [6,12]:
                m=a[a.fit_key.eq(row.fit_key)&a.horizon.eq(h)];r=a[a.fit_key.eq(reference)&a.horizon.eq(h)]
                if len(m)!=1 or len(r)!=1:continue
                m=m.iloc[0];r=r.iloc[0]
                effects.append(dict(station_id=sd.name,arm=row.arm,horizon=h,
                    old_delta_rmse=m.old_rmse-r.old_rmse,new_delta_rmse=m.new_rmse-r.new_rmse,
                    old_delta_n200=int(m.old_n200-r.old_n200),new_delta_n200=int(m.new_n200-r.new_n200)))
    e=pd.DataFrame(effects);e.to_csv(folder/'feature_effect_sensitivity.csv',index=False);a.to_csv(folder/'aggregated_comparison.csv',index=False)
    f=e[e.arm.ne('W0')]
    write_json(folder/'interpretation.json',{'models':len(result),'paired_feature_cases':len(f),'seed':42,
        'old_benefit_to_new_harm':int(((f.old_delta_rmse<-.05)&(f.new_delta_rmse>.05)).sum()),
        'old_harm_to_new_benefit':int(((f.old_delta_rmse>.05)&(f.new_delta_rmse<-.05)).sum()),
        'largest_pooled_rmse_reduction_mm':float((a.old_rmse-a.new_rmse).max()),
        'largest_n200_reduction':int((a.old_n200-a.new_n200).max()),
        'notes':'validation only; fixed seed; no primary model/feature reselection'})


def execute(c):
    work=Path(c['work_root'])/c['run_name'];folder=work/'checkpoint_precision';folder.mkdir(exist_ok=True)
    source=Source(c,work/'shared');thermal=Thermal(folder,c);all_rows=[]
    try:
        for sd in sorted((work/'stations').glob('*')):
            if not (sd/'complete.json').exists():continue
            cache=folder/f'{sd.name}.csv'
            if cache.exists():all_rows.extend(pd.read_csv(cache).to_dict('records'));continue
            base,y,h=source.base(sd.name);folds=source.folds(base.index)
            finalists=json.loads((sd/'finalists.json').read_text())['arms']
            ledger=pd.DataFrame(json.loads((sd/'ledger.json').read_text()))
            if 'version' not in ledger:ledger['version']=None
            chosen=ledger[ledger.status.eq('completed')&ledger.seed.eq(42)&ledger['mode'].eq('normal')&ledger.lag.eq(6)&ledger.version.isna()&
                (ledger.experiment.eq('E1')|(ledger.experiment.eq('E3')&ledger.arm.isin(['W1',*finalists])))]
            keys=sorted(chosen.fit_key.unique());rows=[]
            for key in keys:
                fit=work/'fits'/key;identity=json.loads((fit/'identity.json').read_text());meta=identity['metadata']
                features=source.features(base,meta['grids'],meta['rain_ids'],meta['experiment'],meta['mode'],meta['lag'])
                old=pd.read_csv(fit/'metrics.csv');values=features.to_numpy(dtype=np.float32)
                for horizon in [6,12]:
                    target=y[f'h{horizon}'].to_numpy(dtype=np.float32)
                    for fold in folds:
                        thermal.check();dt=xgb.DMatrix(values[fold['train']],label=target[fold['train']],feature_names=features.columns.tolist(),nthread=4)
                        dv=xgb.DMatrix(values[fold['val']],label=target[fold['val']],feature_names=features.columns.tolist(),nthread=4)
                        params={**c['xgboost'],'seed':42}
                        model=xgb.train(params,dt,c['num_boost_round'],evals=[(dv,'validation')],custom_metric=scaled_metric,
                            early_stopping_rounds=c['early_stopping_rounds'],verbose_eval=False,callbacks=[Guard(thermal)])
                        count=model.best_iteration+1;model=model[:count];pred=model.predict(dv)
                        metrics=scalar_metrics(target[fold['val']],pred)
                        ref=old[old.scope.eq('validation')&old.fold.eq(fold['id'])&old.horizon.eq(horizon)].iloc[0]
                        modelpath=folder/f'{key}_{fold["id"]}_h{horizon}.ubj';model.save_model(modelpath)
                        rows.append(dict(station_id=sd.name,fit_key=key,horizon=horizon,fold=fold['id'],seed=42,n=metrics['n'],
                            old_rmse_mm=float(ref.rmse_mm),new_rmse_mm=metrics['rmse_mm'],old_n200=int(ref.n200),new_n200=metrics['n200'],
                            old_rounds=int(ref.rounds),new_rounds=count,model_path=str(modelpath)))
            pd.DataFrame(rows).to_csv(cache,index=False);all_rows.extend(rows)
            LOG.info('CHECKPOINT PRECISION %s models=%s',sd.name,len(rows))
    finally:thermal.close()
    result=pd.DataFrame(all_rows);result.to_csv(folder/'fold_comparison.csv',index=False)
    summarize(work,folder,result)
    write_json(folder/'manifest.json',{'status':'completed_fixed_seed_sensitivity','seed':42,'new_models':len(result),
        'old_metric':'N200 + 0.001*RMSE/(RMSE+1)','sensitivity_metric':'N200 + RMSE/(RMSE+1)',
        'reason':'XGBoost custom evaluation is formatted with six decimal places; inspect sensitivity to fractional tie-break resolution',
        'selection_policy':'does not change primary features, finalists or five-seed results; validation-only diagnostic'})
    return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path(__file__).with_name('config.yaml'));a=p.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s');print(execute(load_config(a.config)).shape)
