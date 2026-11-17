"""Resumable local residual XGBoost fits, fixed checkpoint rule and GPU thermal guard."""
from pathlib import Path
import hashlib,json,logging,subprocess,threading,time,shutil
import numpy as np
import pandas as pd
import xgboost as xgb
LOG=logging.getLogger(__name__)
_REUSE_INDEX={}


def reuse_key(identity):
    config={k:v for k,v in identity['config'].items() if k not in ['run_name','folds','reuse_previous_run']}
    metadata={k:v for k,v in identity['metadata'].items() if k!='execution_signature'}
    return digest({'config':config,'metadata':metadata,'seed':identity['seed'],'columns':identity['columns']})


def previous_fit(folder,c,identity):
    """Reuse only completed legacy cases with identical sources, schema and fit parameters."""
    from research.station_analysis.config import ROOT
    if not c.get('reuse_previous_run'):return None
    previous=ROOT/c['reuse_previous_run'];index_key=str(previous)
    if index_key not in _REUSE_INDEX:
        old=json.loads((previous/'execution_manifest.json').read_text())
        new=json.loads((Path(folder)/'execution_manifest.json').read_text())
        if old['inputs']!=new['inputs']:raise ValueError('Cannot reuse fits from a different frozen dataset')
        for path,expected in old['code'].items():
            archived=previous/'execution_source'/Path(path).name
            if not archived.exists() or hashlib.sha256(archived.read_bytes()).hexdigest()!=expected:
                raise ValueError('Legacy execution code provenance is missing')
        index={}
        for done in sorted((previous/'fits').glob('*/done.json')):
            candidate=done.parent
            record=json.loads((candidate/'identity.json').read_text())
            index[reuse_key(record)]=candidate
        _REUSE_INDEX[index_key]=index
    return _REUSE_INDEX[index_key].get(reuse_key(identity))

def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,default=str).encode()).hexdigest()[:20]

def write_json(path,obj):
    p=Path(path);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str,allow_nan=False));tmp.replace(p)

def scalar_metrics(actual,pred):
    error=np.asarray(pred)-np.asarray(actual)
    if not len(error):return dict(n=0,rmse_mm=None,mae_mm=None,n200=0,n200_pct=None,bias_mm=None)
    return dict(n=len(error),rmse_mm=float(np.sqrt(np.mean(error**2))),mae_mm=float(np.mean(abs(error))),
                n200=int((abs(error)>200).sum()),n200_pct=float(100*(abs(error)>200).mean()),bias_mm=float(error.mean()))

def checkpoint_metric(predicted,dmatrix):
    e=np.asarray(predicted)-dmatrix.get_label();r=np.sqrt(np.mean(e**2))
    return 'n200_then_rmse',float((abs(e)>200).sum()+.001*r/(r+1))

class Thermal:
    def __init__(self,folder,c):
        self.c=c;self.value=None;self.error=None;self.process=None;self.samples=Path(folder)/'thermal.jsonl'
        if c['xgboost']['device'].startswith('cuda'):
            self.process=subprocess.Popen(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits','--loop=2'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1)
            threading.Thread(target=self._read,daemon=True).start()
            deadline=time.monotonic()+10
            while self.value is None and time.monotonic()<deadline and self.process.poll() is None:time.sleep(.1)
            if self.value is None:raise RuntimeError('GPU telemetry unavailable; do not start unmonitored GPU fits')
    def _read(self):
        for line in self.process.stdout:
            try:
                self.value=(time.monotonic(),int(line.strip()))
                with self.samples.open('a') as f:f.write(json.dumps({'utc':pd.Timestamp.now(tz='UTC').isoformat(),'c':self.value[1]})+'\n')
            except ValueError:self.error=line.strip()
    def check(self):
        if self.process is None:return
        if self.value is None or time.monotonic()-self.value[0]>20:raise RuntimeError('GPU telemetry stale')
        if self.value[1]>=self.c['thermal_pause_c']:
            LOG.info('THERMAL PAUSE at %s C; resume at <=%s C',self.value[1],self.c['thermal_resume_c'])
            while self.value[1]>self.c['thermal_resume_c']:
                time.sleep(5)
                if time.monotonic()-self.value[0]>20:raise RuntimeError('GPU telemetry stopped during thermal pause')
            LOG.info('THERMAL RESUME at %s C',self.value[1])
    def close(self):
        if self.process is not None:self.process.terminate()

class Guard(xgb.callback.TrainingCallback):
    def __init__(self,thermal):self.thermal=thermal
    def after_iteration(self,model,epoch,evals_log):
        if epoch%5==0:self.thermal.check()
        return False

def run_case(folder,c,thermal,x,y,current,folds,metadata,seed):
    identity={'config':c,'metadata':metadata,'seed':seed,'columns':x.columns.tolist()}
    key=digest(identity);out=Path(folder)/'fits'/key;done=out/'done.json'
    if done.exists():
        record=json.loads(done.read_text())
        if all((out/name).exists() for name in record['files']):return key
    out.mkdir(parents=True,exist_ok=True);write_json(out/'identity.json',identity)
    previous=previous_fit(folder,c,identity)
    old_metrics=pd.read_csv(previous/'metrics.csv') if previous else pd.DataFrame()
    old_predictions=pd.read_parquet(previous/'predictions.parquet') if previous else pd.DataFrame()
    reused=[]
    start=time.monotonic();rows=[];prediction_frames=[];importances=[]
    xvalues=x.to_numpy(dtype=np.float32);idx=x.index
    cutoff=pd.Timestamp(c['cutoff']);eval_mask=idx>=cutoff
    fulltrain=idx<cutoff-pd.Timedelta(hours=c['purge_hours'])
    params={**c['xgboost'],'seed':seed}
    for horizon in c['horizons']:
        target=y[f'h{horizon}'].to_numpy(dtype=np.float32);rounds=[]
        for fold in folds:
            thermal.check();name=fold['id']
            old=old_metrics[(old_metrics.horizon==horizon)&old_metrics.fold.eq(name)&old_metrics.scope.eq('validation')] if len(old_metrics) else pd.DataFrame()
            if len(old)==1 and (previous/f'{name}_h{horizon}.ubj').exists():
                subset=old_predictions[old_predictions.horizon.eq(horizon)&old_predictions.fold.eq(name)&old_predictions.scope.eq('validation')].copy()
                if not pd.DatetimeIndex(pd.to_datetime(subset.anchor_utc,utc=True)).equals(idx[fold['val']]):
                    raise ValueError('Legacy fold has different anchors; cannot reuse')
                if not np.allclose(subset.actual_mm-subset.persistence_mm,target[fold['val']],rtol=0,atol=.001):
                    raise ValueError('Legacy targets differ; cannot reuse')
                shutil.copy2(previous/f'{name}_h{horizon}.ubj',out/f'{name}_h{horizon}.ubj')
                rows.append(old.iloc[0].to_dict());prediction_frames.append(subset);rounds.append(int(old.iloc[0].rounds))
                reused.append({'fold':name,'horizon':horizon,'source_model':str(previous/f'{name}_h{horizon}.ubj'),
                               'sha256':hashlib.sha256((previous/f'{name}_h{horizon}.ubj').read_bytes()).hexdigest()})
                continue
            dt=xgb.DMatrix(xvalues[fold['train']],label=target[fold['train']],feature_names=x.columns.tolist(),nthread=4)
            dv=xgb.DMatrix(xvalues[fold['val']],label=target[fold['val']],feature_names=x.columns.tolist(),nthread=4)
            model=xgb.train(params,dt,c['num_boost_round'],evals=[(dv,'validation')],custom_metric=checkpoint_metric,
                            early_stopping_rounds=c['early_stopping_rounds'],verbose_eval=False,callbacks=[Guard(thermal)])
            count=model.best_iteration+1;model=model[:count];rounds.append(count)
            delta=model.predict(dv);truth=target[fold['val']]
            actual=current.to_numpy()[fold['val']]+truth;pred=current.to_numpy()[fold['val']]+delta
            rows.append(dict(horizon=horizon,scope='validation',fold=name,rounds=count,**scalar_metrics(actual,pred)))
            prediction_frames.append(pd.DataFrame(dict(anchor_utc=idx[fold['val']],horizon=horizon,scope='validation',fold=name,
                actual_mm=actual,prediction_mm=pred,persistence_mm=current.to_numpy()[fold['val']]),index=None))
            model.save_model(out/f'{name}_h{horizon}.ubj')
        thermal.check();count=max(1,int(np.median(rounds)))
        df=xgb.DMatrix(xvalues[fulltrain],label=target[fulltrain],feature_names=x.columns.tolist(),nthread=4)
        model=xgb.train(params,df,count,verbose_eval=False,callbacks=[Guard(thermal)])
        delta=model.predict(xgb.DMatrix(xvalues[eval_mask],feature_names=x.columns.tolist(),nthread=4)) if eval_mask.any() else np.array([],dtype=float)
        actual=current.to_numpy()[eval_mask]+target[eval_mask];pred=current.to_numpy()[eval_mask]+delta
        rows.append(dict(horizon=horizon,scope='exploratory',fold='post_cutoff',rounds=count,**scalar_metrics(actual,pred)))
        prediction_frames.append(pd.DataFrame(dict(anchor_utc=idx[eval_mask],horizon=horizon,scope='exploratory',fold='post_cutoff',
            actual_mm=actual,prediction_mm=pred,persistence_mm=current.to_numpy()[eval_mask])))
        model.save_model(out/f'final_h{horizon}.ubj')
        for feature,gain in model.get_score(importance_type='total_gain').items():
            importances.append(dict(horizon=horizon,feature=feature,total_gain=float(gain)))
    pd.concat(prediction_frames,ignore_index=True).to_parquet(out/'predictions.parquet',index=False)
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(importances).to_csv(out/'importance.csv',index=False)
    files=[p.name for p in out.iterdir() if p.is_file() and p.name!='done.json']
    write_json(done,{'key':key,'files':files,'elapsed_seconds':time.monotonic()-start,'feature_count':x.shape[1],
                     'xgboost_version':xgb.__version__,'trained_models':2*(len(folds)+1),
                     'newly_fitted_models':2*(len(folds)+1)-len(reused),'reused_fold_models':reused})
    LOG.info('FIT %s seed=%s features=%s folds=%s seconds=%.1f',key,seed,x.shape[1],len(folds),time.monotonic()-start)
    return key
