"""Independent artifact audit: populations, five seeds, saved-model predictions and source hashes."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from .data import Source,load_config,sha
from .training import scalar_metrics,write_json
from .planning import principal


def audit(work,c,*,require_complete=True):
    work=Path(work);source=Source(c,work/'shared');record=json.loads((work/'execution_manifest.json').read_text())
    for path,expected in record['inputs'].items():
        if sha(path)!=expected:raise ValueError(f'Source changed: {path}')
    for path,expected in record['code'].items():
        if sha(path)!=expected:raise ValueError(f'Execution source changed: {path}')
    allowed_signatures={record['signature']}
    parallel=work/'parallel_manifest.json'
    if parallel.exists():
        for worker in json.loads(parallel.read_text())['workers']:
            child_path=Path(worker['work'])/'execution_manifest.json'
            if not child_path.exists():continue
            child=json.loads(child_path.read_text())
            ignore=['run_name','reuse_previous_run']
            if {k:v for k,v in child['config'].items() if k not in ignore}!={k:v for k,v in record['config'].items() if k not in ignore} or child['inputs']!=record['inputs'] or child['code']!=record['code']:
                raise ValueError('Parallel worker provenance differs from canonical experiment')
            allowed_signatures.add(child['signature'])
    eligibility=pd.DataFrame(json.loads((work/'eligibility.json').read_text()))
    if require_complete and len(eligibility)!=len(source.water):raise ValueError('Eligibility audit is incomplete')
    complete={p.parent.name for p in (work/'stations').glob('*/complete.json')}
    required=set(eligibility[eligibility.eligible].station_id)
    if require_complete and complete!=required:raise ValueError('Not all eligible stations completed')
    checks=[];used=set()
    for sid in sorted(complete):
        sd=work/'stations'/sid;population=pd.read_parquet(sd/'population.parquet');idx=pd.DatetimeIndex(pd.to_datetime(population.anchor_utc,utc=True))
        if idx.has_duplicates:raise ValueError('Duplicated anchors')
        base,y,h=source.base(sid)
        if not idx.equals(base.index):raise ValueError('Saved population differs from independently rebuilt population')
        np.testing.assert_allclose(population.dh6_mm,y.h6,rtol=0,atol=0)
        np.testing.assert_allclose(population.dh12_mm,y.h12,rtol=0,atol=0)
        for horizon in [6,12]:
            expected=source.level[sid].reindex(idx+pd.Timedelta(hours=horizon)).to_numpy()-source.level[sid].reindex(idx).to_numpy()
            np.testing.assert_allclose(population[f'dh{horizon}_mm'],expected,rtol=1e-6,atol=.001)
        for f in source.folds(idx):
            if (f['train']&f['val']).any() or idx[f['train']].max()+pd.Timedelta(hours=24)>=idx[f['val']].min():
                raise ValueError('Chronological purge violated')
        ledger=pd.DataFrame(json.loads((sd/'ledger.json').read_text()))
        for column,value in [('version','all'),('diagnostic','principal'),('mode','normal'),('lag',6)]:
            if column not in ledger:ledger[column]=value
            else:ledger[column]=ledger[column].fillna(value)
        finished=ledger[ledger.status.eq('completed')]
        for key,g in finished.groupby(['experiment','arm','mode','lag','version','diagnostic']):
            if set(g.seed.astype(int))!=set(c['seeds']):raise ValueError(f'Missing seeds: {sid} {key}')
        for task in principal(source,sid):
            found=finished[finished.experiment.eq(task['experiment'])&finished.arm.eq(task['arm'])]
            if set(found.seed.astype(int))!=set(c['seeds']):raise ValueError(f'Principal task missing {sid} {task}')
        all_keys=sorted(finished.fit_key.unique());used.update(all_keys)
        for fitkey in all_keys:
            folder=work/'fits'/fitkey;done=json.loads((folder/'done.json').read_text())
            identity_record=json.loads((folder/'identity.json').read_text())
            if identity_record['metadata']['execution_signature'] not in allowed_signatures:raise ValueError('Unknown scientific execution signature')
            for reused in done.get('reused_fold_models',[]):
                if sha(reused['source_model'])!=reused['sha256'] or sha(folder/f'{reused["fold"]}_h{reused["horizon"]}.ubj')!=reused['sha256']:
                    raise ValueError('Reused model hash mismatch')
            for name in done['files']:
                if not (folder/name).is_file():raise ValueError(f'Fit artifact missing {fitkey}/{name}')
            # Recompute metrics from every saved forecast, independently of model training logs.
            pred=pd.read_parquet(folder/'predictions.parquet');metrics=pd.read_csv(folder/'metrics.csv')
            meta=json.loads((folder/'identity.json').read_text())['metadata']
            local_idx=idx
            if meta['version']:
                kind=source.weather[6]['ecmwf_ifs:16.274164:107.264960|archive_kind'].reindex(idx)
                local_idx=idx[kind.str.contains(meta['version'],regex=False).fillna(False)]
            local_folds={f['id']:f for f in source.folds(local_idx)}
            for (scope,fold,horizon),part in pred.groupby(['scope','fold','horizon']):
                m=metrics[(metrics.scope==scope)&(metrics.fold==fold)&(metrics.horizon==horizon)].iloc[0]
                expected_times=local_idx[local_idx>=pd.Timestamp(c['cutoff'])] if scope=='exploratory' else local_idx[local_folds[fold]['val']]
                if not pd.DatetimeIndex(pd.to_datetime(part.anchor_utc,utc=True)).equals(expected_times):raise ValueError('Prediction anchors differ from common population')
                value=scalar_metrics(part.actual_mm,part.prediction_mm)
                if int(value['n200'])!=int(m.n200) or not np.isclose(value['rmse_mm'],m.rmse_mm,rtol=1e-10,atol=1e-8):
                    raise ValueError('Saved prediction metrics mismatch')
        # Fixed-seed, fixed-sample saved-model check per station, including all eligible validation folds.
        candidates=finished[(finished.seed==42)&finished.version.eq('all')]
        selected=sorted(candidates.fit_key.unique())
        for fitkey in [selected[0],selected[-1]] if len(selected)>1 else selected:
            folder=work/'fits'/fitkey;identity=json.loads((folder/'identity.json').read_text());meta=identity['metadata']
            x=source.features(base,meta['grids'],meta['rain_ids'],meta['experiment'],meta['mode'],meta['lag'])
            pred=pd.read_parquet(folder/'predictions.parquet');pred['anchor_utc']=pd.to_datetime(pred.anchor_utc,utc=True)
            for (scope,fold,horizon),part in pred.groupby(['scope','fold','horizon']):
                sample=part.sample(min(8,len(part)),random_state=42)
                model=xgb.Booster();model.load_model(folder/(f'final_h{horizon}.ubj' if scope=='exploratory' else f'{fold}_h{horizon}.ubj'))
                model.set_param({'device':'cpu','nthread':2})
                actual=model.predict(xgb.DMatrix(x.reindex(sample.anchor_utc).to_numpy(dtype=np.float32),feature_names=x.columns.tolist()))+sample.persistence_mm.to_numpy()
                np.testing.assert_allclose(actual,sample.prediction_mm,rtol=0,atol=.02)
        checks.append(dict(station_id=sid,population=len(idx),unique_fits=len(all_keys),saved_model_checks=len(selected[:2]),status='PASS'))
    result={'status':'PASS','complete':complete==required and len(eligibility)==len(source.water),
        'source_hashes_verified':len(record['inputs']),'execution_code_hashes_verified':len(record['code']),
        'unique_fits_verified':len(used),'stations':checks,'checked_at':pd.Timestamp.now(tz='UTC').isoformat()}
    write_json(work/'audit.json',result)
    return result


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path(__file__).with_name('config.yaml'));p.add_argument('--allow-partial',action='store_true')
    a=p.parse_args();c=load_config(a.config);print(json.dumps(audit(Path(c['work_root'])/c['run_name'],c,require_complete=not a.allow_partial),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
