"""Compare local old/new checkpoints and persistence on identical held-out rows.

Reconstruct new scalers from its immutable train.csv and verify checkpoint buffer
coefficients and logged validation loss before scoring. No training/deployment.
"""
import os
import argparse
os.environ.setdefault('MLFLOW_DISABLE_AGENT_HINT','1')
import json
import hashlib
from pathlib import Path
import sqlite3
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.ai.dataset import HydrologyDataset
from src.ai.lstm import ResidualLSTM
from src.ai.losses import HorizonWeightedMSE
from src.ai.utils import select_feature_columns

OUT=ROOT/'reports/model_comparison_2026-10-15'
DATA=ROOT/'train_data/1h_top4_pg_2026-10-15_val_aug01_sep15'
NEW=ROOT/'model/1h_top4_pg_2026-10-15_val_aug01_sep15/1h/ResidualLSTM_water_level+rain+flood_season_best.pt'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model(path,saved,parameters):
    cols=saved['feature_columns'];fs=saved['feature_scaler'];ts=saved['target_scaler'];i=cols.index('water_level_depth')
    model=ResidualLSTM(input_size=len(cols),output_size=len(saved['target_columns']),level_feature_index=i,
        persistence_scale=(fs.scale_[i]/ts.scale_).tolist(),
        persistence_offset=((fs.mean_[i]-ts.mean_)/ts.scale_).tolist(),**parameters)
    state=torch.load(path,map_location='cpu',weights_only=True)
    for key in ['persistence_scale','persistence_offset']:
        torch.testing.assert_close(state[key],getattr(model,key))
    model.load_state_dict(state,strict=True);model.eval();model.requires_grad_(False)
    return model


def infer(model,dataset):
    with torch.inference_mode():
        outputs=[model(x).numpy() for x,_ in DataLoader(dataset,batch_size=128,shuffle=False)]
    return np.concatenate(outputs)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output_dir
    if (OUT / 'manifest.json').exists():
        raise FileExistsError('Use a new output directory to preserve prior model comparisons')
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True,parents=True)
    con=sqlite3.connect(f'file:{ROOT / "mlflow.db"}?mode=ro',uri=True)
    rid,status=con.execute("SELECT r.run_uuid,r.status FROM runs r JOIN experiments e ON e.experiment_id=r.experiment_id WHERE e.name='da_vien_weighted_loss' ORDER BY r.start_time DESC LIMIT 1").fetchone()
    params=dict(con.execute('SELECT key,value FROM params WHERE run_uuid=?',(rid,)))
    metrics=dict(con.execute('SELECT key,value FROM latest_metrics WHERE run_uuid=?',(rid,)))
    assert params['dataset_dir']==str(DATA)
    train=pd.read_csv(DATA/'train.csv');val=pd.read_csv(DATA/'val.csv');test=pd.read_csv(DATA/'test.csv')
    cols=select_feature_columns(train.columns,params['feature_set']);targets=[c for c in train if c.startswith('target_')];seq=int(params['sequence_length'])
    fit=HydrologyDataset(train,feature_cols=cols,target_cols=targets,fit_scalers=True,seq_len=seq,require_current_level=True)
    saved=dict(feature_scaler=fit.feature_scaler,target_scaler=fit.target_scaler,feature_columns=cols,target_columns=targets,sequence_length=seq,missing_scaled_value=0.)
    joblib.dump(saved,OUT/'new_model_reconstructed_preprocessing.joblib')
    architecture=dict(hidden_size=int(params['Hidden size']),num_layers=int(params['Layers']),dropout=float(params['LSTM dropout']),input_dropout=float(params['Input dropout']),head_dropout=float(params['Head dropout']))
    new=load_model(NEW,saved,architecture)
    def dataset(frame,pre):return HydrologyDataset(frame,feature_cols=pre['feature_columns'],target_cols=pre['target_columns'],feature_scaler=pre['feature_scaler'],target_scaler=pre['target_scaler'],seq_len=pre['sequence_length'],require_current_level=True)
    vd=dataset(val,saved);vp=infer(new,vd)
    loss=HorizonWeightedMSE(json.loads(params['horizon_weights']))(torch.from_numpy(vp),torch.from_numpy(vd.y[vd.end_indices])).item()
    assert abs(loss-metrics['best_val_loss'])<1e-5,(loss,metrics['best_val_loss'])
    old_cfg=json.loads((ROOT/'src/backend/prediction/model_configs.json').read_text())[0]
    oldpath=ROOT/old_cfg['model_path'];oldprep=ROOT/old_cfg['preprocessing_path'];old_saved=joblib.load(oldprep)
    assert old_saved['feature_columns']==cols and old_saved['target_columns']==targets and old_saved['sequence_length']==seq
    old=load_model(oldpath,old_saved,old_cfg['model_parameters'])
    nd=dataset(test,saved);od=dataset(test,old_saved)
    np.testing.assert_array_equal(nd.end_indices,od.end_indices)
    new_values=saved['target_scaler'].inverse_transform(infer(new,nd))
    old_values=old_saved['target_scaler'].inverse_transform(infer(old,od))
    observed=test[targets].to_numpy()[nd.end_indices]
    base=test.water_level_depth.to_numpy()[nd.end_indices]
    anchors=pd.to_datetime(test.timestamp_utc.iloc[nd.end_indices],utc=True).reset_index(drop=True)
    # Confirm the new model reproduces the recorded test run in physical units.
    for j,t in enumerate(targets):
        actual_mae=np.abs(new_values[:,j]-observed[:,j]).mean()
        assert abs(actual_mae-metrics[f'test/{t}/mae'])<.02,(t,actual_mae,metrics[f'test/{t}/mae'])
    frames=[];results=[]
    for j,target in enumerate(targets):
        horizon=int(target.rsplit('_',1)[-1].removesuffix('m'))
        frame=pd.DataFrame(dict(anchor_utc=anchors,target_utc=anchors+pd.Timedelta(minutes=horizon),horizon_hours=horizon//60,actual=observed[:,j],new_model=new_values[:,j],old_model=old_values[:,j],persistence=base))
        for name in ['new_model','old_model','persistence']:
            err=frame[name]-frame.actual;ae=err.abs()
            frame[f'{name}_error']=err
            results.append(dict(horizon_hours=horizon//60,model=name,n=len(frame),mae=ae.mean(),rmse=np.sqrt((err**2).mean()),bias=err.mean(),max_ae=ae.max(),above200=int(ae.gt(200).sum()),above200_pct=ae.gt(200).mean()*100))
        frames.append(frame)
    points=pd.concat(frames,ignore_index=True);result=pd.DataFrame(results)
    points.to_csv(OUT/'paired_predictions.csv',index=False);result.to_csv(OUT/'metrics.csv',index=False)
    # Per-horizon line figures, all methods share exactly the same target rows.
    fig,axes=plt.subplots(5,1,figsize=(17,16),sharey=True)
    styles=[('actual','Observed','#222222','-'),('new_model','New weighted, penalty 0','#1677b8','-'),('old_model','Old serving model','#ce4525','-'),('persistence','Persistence','#7750a2','--')]
    for ax,(h,g) in zip(axes,points.groupby('horizon_hours')):
        times=pd.to_datetime(g.target_utc,utc=True).dt.tz_convert('Asia/Ho_Chi_Minh')
        for col,label,color,ls in styles:ax.plot(times,g[col],label=label,color=color,ls=ls,lw=1.3,alpha=.9)
        ax.set_title(f'+{h} hours | {len(g)} paired predictions',fontsize=11);ax.set_ylabel('Water level (mm)');ax.grid(alpha=.2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m',tz=times.dt.tz));ax.xaxis.set_major_locator(mdates.DayLocator(interval=3,tz=times.dt.tz))
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.965),ncol=4)
    fig.suptitle('Da Vien: new model vs old model vs persistence | same samples, same truth',fontsize=15)
    axes[-1].set_xlabel('Forecast target date, Vietnam time');fig.tight_layout(rect=(0,0,1,.945))
    for ext in ['png','pdf','svg']:fig.savefig(OUT/f'paired_forecasts_all_horizons.{ext}',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,metric in zip(axes,['mae','above200_pct']):
        for name,label,color,_ in styles[1:]:
            g=result[result.model.eq(name)];ax.plot(g.horizon_hours,g[metric],marker='o',label=label,color=color)
        ax.set(xlabel='Forecast horizon (hours)',ylabel='MAE (mm)' if metric=='mae' else 'Absolute error >200 mm (%)',xticks=[1,3,6,12,24]);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'comparison_metrics.png',dpi=170);plt.close(fig)
    provenance=dict(new_run_id=rid,new_run_status=status,best_epoch=metrics['best_checkpoint_epoch'],logged_best_val=metrics['best_val_loss'],recomputed_best_val=loss,new_checkpoint_sha256=digest(NEW),old_checkpoint_sha256=digest(oldpath),old_scaler_sha256=digest(oldprep),new_scaler='Reconstructed from exact train.csv; checkpoint persistence buffers, validation loss and all five test MAEs verified',data_manifest_sha256=digest(DATA/'manifest.json'),test_csv_sha256=digest(DATA/'test.csv'),n_anchors=len(anchors),first_anchor_vietnam=str(anchors.min().tz_convert('Asia/Ho_Chi_Minh')),last_anchor_vietnam=str(anchors.max().tz_convert('Asia/Ho_Chi_Minh')),new_params=params,old_model_version=old_cfg['model_version'],caveats=['Same data but models differ in train split, fitted scalers, loss and penalty; not a single-factor ablation','September/October has been explored and is not a pristine holdout','96-step windows begin inside test.csv, matching original training evaluation; first 95 test hours are warmup'])
    (OUT/'manifest.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
    print(result.round(3).to_string(index=False));print(json.dumps({k:provenance[k] for k in ['new_run_id','new_run_status','best_epoch','recomputed_best_val','n_anchors','first_anchor_vietnam','last_anchor_vietnam']},ensure_ascii=False))


if __name__=='__main__':main()
