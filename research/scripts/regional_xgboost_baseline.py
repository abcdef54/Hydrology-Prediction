"""Regional residual XGBoost baseline; fixed anchors, train-only scaling."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from src.ai.regional_training_data import RegionalTrainingDataset

ROOT=Path('train_data/hue_regional_20261015_h6_h12_fixed_split_v1')
OUT=Path('reports/model_training/regional_xgboost_20261102'); OUT.mkdir(parents=True,exist_ok=True)
LAGS=[0,1,3,6,12,24,48,72,95]
def build(split):
 d=RegionalTrainingDataset(ROOT,split=split,fold='fixed',rain_stations='hong_thai+bach_ma+phu_bai',use_weather_forecast=True,time_features='all',sequence_length=96,target_horizons=[6,12])
 n=len(d); blocks=[]
 # Forecast is anchor-only; every lag includes the corresponding saved missingness masks.
 for start in range(0,n,256):
  ix=np.arange(start,min(start+256,n)); b=d.__getitems__(ix.tolist()).tensors
  x=b['x'].numpy(); lag=x[:,[95-k for k in LAGS],:].reshape(len(ix),-1)
  ids=b['station_ids'].numpy(); onehot=np.eye(11,dtype=np.float32)[ids]
  blocks.append(np.concatenate([lag,onehot],axis=1))
 frame=d.view.samples.reset_index(drop=True); stats=d.view.normalizers['stations']; N=int(len(d))
 return d,np.concatenate(blocks),frame,stats

def main():
 train,Xtr,tr,stats=build('train'); val,Xv,v,_=build('validation'); test,Xt,t,_=build('test_exploratory')
 assert len(tr)==62500 and len(v)==11880 and len(t)==7953
 counts=tr.station_id.value_counts(); weight=np.array([len(tr)/(11*counts[s]) for s in tr.station_id],dtype=np.float32)
 results=[]; predictions=[]
 for horizon in [6,12]:
  def scale(frame):return np.array([max(stats[s]['targets'][f'h{horizon}']['scale'],200.) for s in frame.station_id],dtype=np.float32)
  strn,sv,st=map(scale,[tr,v,t]); ytr=(tr[f'delta_h{horizon}_mm'].to_numpy()/strn).astype('float32'); yv=(v[f'delta_h{horizon}_mm'].to_numpy()/sv).astype('float32')
  # Same training-frequency weights as the Transformer's station-balanced loss.
  wv=np.array([len(tr)/(11*counts[s]) for s in v.station_id],dtype=np.float32)
  model=xgb.XGBRegressor(n_estimators=700,max_depth=6,learning_rate=.05,subsample=.8,colsample_bytree=.8,tree_method='hist',device='cpu',n_jobs=4,random_state=42,objective='reg:squarederror',early_stopping_rounds=50)
  print(f'Training +{horizon}h: {len(tr)} train, {len(v)} validation, {Xtr.shape[1]} features',flush=True)
  model.fit(Xtr,ytr,sample_weight=weight,eval_set=[(Xv,yv)],sample_weight_eval_set=[wv],verbose=False)
  print(f'+{horizon}h best iteration: {model.best_iteration}',flush=True)
  model.save_model(OUT/f'model_h{horizon}.json')
  for split,X,frame,sc in [('validation',Xv,v,sv),('test',Xt,t,st)]:
   pred=frame.water_now_mm.to_numpy()+model.predict(X)*sc; actual=frame[f'target_h{horizon}_mm'].to_numpy();delta=actual-frame.water_now_mm.to_numpy(); predicted=pred-frame.water_now_mm.to_numpy();err=pred-actual
   for station,g in pd.DataFrame(dict(station_id=frame.station_id,error=err,delta=delta,predicted=predicted)).groupby('station_id'):
    rise=g.delta>200; alarm=g.predicted>200
    results.append(dict(split=split,horizon=horizon,station_id=station,n=len(g),sse=float((g.error**2).sum()),sae=float(g.error.abs().sum()),large_errors=int((g.error.abs()>200).sum()),rises=int(rise.sum()),hits=int((rise&alarm).sum()),alarms=int(alarm.sum()),false_alarms=int((~rise&alarm).sum()),best_iteration=int(model.best_iteration)))
   predictions.append(pd.DataFrame(dict(split=split,horizon=horizon,station_id=frame.station_id,anchor_utc=frame.anchor_utc,actual_mm=actual,predicted_mm=pred,current_mm=frame.water_now_mm)))
 metrics=pd.DataFrame(results); metrics.to_csv(OUT/'station_metrics.csv',index=False);pd.concat(predictions).to_parquet(OUT/'predictions.parquet',index=False)
 (OUT/'config.json').write_text(json.dumps(dict(lags_hours=LAGS,features=train.feature_cols,rain=train.rain_station_ids,time='all',seed=42,target='physical change divided by max(train target scale, 200 mm)',train_weights='inverse training station frequency',validation_selection='weighted scaled-change RMSE per horizon',xgboost=dict(n_estimators=700,max_depth=6,learning_rate=.05,subsample=.8,colsample_bytree=.8,tree_method='hist',device='cpu',early_stopping_rounds=50)),indent=2))
 for split,g in metrics.groupby('split'):
  print(split,'n',int(g.n.sum()),'rmse',float(np.sqrt(g.sse.sum()/g.n.sum())),'mae',float(g.sae.sum()/g.n.sum()),'errors',int(g.large_errors.sum()),'hits',int(g.hits.sum()),'rises',int(g.rises.sum()),'false',int(g.false_alarms.sum()),'alarms',int(g.alarms.sum()),flush=True)
 print('Artifacts:',OUT,flush=True)
if __name__=='__main__': main()
