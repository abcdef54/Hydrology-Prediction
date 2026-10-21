"""Train B from scratch with train-label change weighting, plus an exact control.

Weight = 1 + min(abs(H(t+h)-H(t))/200, 1), symmetric for rise/fall.
Divide by each horizon's mean TRAIN weight to retain the overall loss scale.
Validation and checkpoint selection always use B's unchanged unweighted loss.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:2')
os.environ.setdefault('MPLCONFIGDIR','/tmp/hydrology-matplotlib')
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.ai.losses import HorizonWeightedTailLoss
from src.ai.comparison import evaluation_dataset,load_candidate,paired_metrics
from research.finetune_direction_experiment import SOURCE,DATA,make_model,evaluate


def raw_change_weights(change_mm,boost=1.,scale_mm=200.):
    if not math.isfinite(boost) or boost<0 or not math.isfinite(scale_mm) or scale_mm<=0:
        raise ValueError('Nonnegative finite boost and positive finite scale required')
    return 1+boost*(change_mm.detach().abs()/scale_mm).clamp(max=1)


def weighted_loss(base,pred,target,persistence,normalizer,boost=1.):
    if boost==0:return base(pred,target)
    if pred.shape!=target.shape or pred.shape!=persistence.shape:
        raise ValueError('Prediction, label and persistence shapes must match')
    scales=base.target_scales.to(pred)
    weights=raw_change_weights((target-persistence).detach()*scales,boost)
    weights=weights/torch.as_tensor(normalizer,device=pred.device,dtype=pred.dtype)
    mse=(pred-target).square()
    tail=torch.nn.functional.softplus(((pred-target).abs()*scales-base.threshold_mm)/base.transition_mm)
    return base.reduce_per_horizon(weights*(mse+base.strength*tail))


def sequence_stats(points,error_column):
    rows=[]
    for h,g in points.groupby('horizon_minutes'):
        bad=g[g[error_column].abs().gt(200)].sort_values('target_utc')
        groups=pd.to_datetime(bad.target_utc,utc=True).diff().ne(pd.Timedelta(hours=1)).cumsum()
        runs=bad.groupby(groups).size()
        rows.append(dict(horizon_hours=int(h//60),n=len(g),bad=len(bad),
            under=int(g[error_column].lt(-200).sum()),over=int(g[error_column].gt(200).sum()),
            runs=len(runs),longest_run=int(runs.max()) if len(runs) else 0,
            multi_run_points=int(runs[runs.ge(2)].sum())))
    return pd.DataFrame(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--patience',type=int,default=25)
    p.add_argument('--no-mlflow',action='store_true')
    args=p.parse_args()
    if args.epochs<1 or args.patience<1:p.error('Epochs/patience must be positive')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU training fallback')
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);device='cuda'
    saved=joblib.load(SOURCE.with_suffix('.preprocessing.joblib'))
    frames={s:pd.read_csv(DATA/f'{s}.csv') for s in ['train','val','test']}
    datasets={s:evaluation_dataset(frames[s],saved) for s in frames}
    assert [len(datasets[s]) for s in frames]==[6095,985,592]
    train=frames['train'].iloc[datasets['train'].end_indices]
    changes=train[saved['target_columns']].to_numpy()-train.water_level_depth.to_numpy()[:,None]
    assert np.isfinite(changes).all()
    base=HorizonWeightedTailLoss([1]*5,saved['target_scaler'].scale_,strength=.5,threshold_mm=200,transition_mm=20).to(device)
    b=load_candidate(SOURCE.read_bytes(),saved,device)
    settings=dict(seed=42,epochs=args.epochs,patience=args.patience,lr=.0009,batch_size=128,
        weight_decay=.001,correction_penalty=.1,gradient_clip=1.,source=str(SOURCE),dataset=str(DATA),
        initialization='scratch; no B model weights',scaler_source='unchanged train-fitted B preprocessing',
        device=torch.cuda.get_device_name(0),weight_formula='1 + boost * min(abs(true_change_mm)/200,1)',
        normalization='fixed mean raw weight per horizon, train eligible anchors only',
        weight_scope='MSE and soft tail only; correction regularization unchanged',
        validation='unweighted B tail loss; select lexicographic (N200, pooled RMSE), exactly B rule',
        no_validation_or_test_label_weighting=True)
    (out/'settings.json').write_text(json.dumps(settings,indent=2))
    import mlflow
    from contextlib import nullcontext
    if not args.no_mlflow:
        mlflow.set_tracking_uri('http://localhost:5000');mlflow.set_experiment('da_vien_change_weight')
    results={}
    for arm,boost in [('control_B',0.),('change_weight_2x',1.)]:
        folder=out/arm;folder.mkdir()
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        torch.use_deterministic_algorithms(True)
        loader=DataLoader(datasets['train'],batch_size=128,shuffle=True,drop_last=True,
            pin_memory=True,generator=torch.Generator().manual_seed(42))
        val=DataLoader(datasets['val'],batch_size=128,shuffle=False,generator=torch.Generator().manual_seed(43))
        model=make_model(saved,device,initialize_from_B=False)
        normalizer=raw_change_weights(torch.as_tensor(changes),boost).mean(0).numpy()
        (folder/'weight_normalizer.json').write_text(json.dumps(dict(zip(saved['target_columns'],normalizer.tolist())),indent=2))
        checkpoint=folder/'model_best.pt'
        for name in ['model_best','model_lowest_loss']:joblib.dump(saved,folder/f'{name}.preprocessing.joblib')
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0009,weight_decay=.001)
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=.9)
        best=evaluate(model,val,base,device,saved['target_scaler'].scale_)
        best_epoch=0;wait=0;lowest=best['loss'];lowest_epoch=0
        for name in ['model_best','model_lowest_loss']:torch.save(model.state_dict(),folder/f'{name}.pt')
        history=[dict(epoch=0,**best)]
        with (nullcontext(None) if args.no_mlflow else mlflow.start_run(run_name=arm)) as run:
            if run:mlflow.log_params({**settings,'arm':arm,'boost':boost,'checkpoint_path':str(checkpoint)})
            print(f'{arm}: GPU={settings["device"]}; initial validation={best}',flush=True)
            for epoch in range(1,args.epochs+1):
                model.train();total=0.;batches=0
                for x,y in loader:
                    x,y=x.to(device),y.to(device);optimizer.zero_grad(set_to_none=True)
                    pred=model(x);persistence=model.persistence(x)
                    data=weighted_loss(base,pred,y,persistence,normalizer,boost)
                    value=data+.1*(pred-persistence).square().mean()
                    if not torch.isfinite(value):raise ValueError('Non-finite objective')
                    value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    optimizer.step();total+=data.item();batches+=1
                scheduler.step();current=evaluate(model,val,base,device,saved['target_scaler'].scale_)
                history.append(dict(epoch=epoch,train_loss=total/batches,**current))
                if current['loss']<lowest:
                    lowest=current['loss'];lowest_epoch=epoch;torch.save(model.state_dict(),folder/'model_lowest_loss.pt')
                score=lambda m:(m['above_200'],m['rmse_mm'])
                if score(current)<score(best):
                    best=dict(current);best_epoch=epoch;wait=0;torch.save(model.state_dict(),checkpoint)
                else:wait+=1
                pd.DataFrame(history).to_csv(folder/'validation_history.csv',index=False)
                if run:mlflow.log_metrics({**{f'val/{k}':v for k,v in current.items()},'train/loss':total/batches},step=epoch)
                print(f'{arm} epoch {epoch}: val N200={current["above_200"]}, RMSE={current["rmse_mm"]:.3f}, best={best_epoch}',flush=True)
                if wait>=args.patience:break
            result=dict(run_id=run.info.run_id if run else None,selected_epoch=best_epoch,selected_validation=best,
                lowest_loss_epoch=lowest_epoch,lowest_loss=lowest,epochs_completed=epoch,boost=boost)
            model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True));model.eval()
            for scope,frame in [('strict_test',frames['test']),('full',pd.read_csv(DATA/'hourly.csv'))]:
                m,s,points,c=paired_metrics(frame,model,saved,b,saved,device=device)
                m.to_csv(folder/f'{scope}_metrics.csv',index=False)
                points.to_csv(folder/f'{scope}_predictions.csv.gz',index=False)
                sequence_stats(points,'candidate_error_mm').to_csv(folder/f'{scope}_sequences.csv',index=False)
                sequence_stats(points,'serving_error_mm').to_csv(folder/f'{scope}_B_sequences.csv',index=False)
                old=points.serving_error_mm.abs().gt(200);new=points.candidate_error_mm.abs().gt(200)
                result[scope]=dict(summary=s,coverage=c,fixed=int((old&~new).sum()),new=int((~old&new).sum()))
                if scope=='strict_test':assert s['serving']['above_200']==48
                print(arm,scope,json.dumps(s['candidate']),flush=True)
            (folder/'result.json').write_text(json.dumps(result,indent=2))
            results[arm]=result
        del model,optimizer;torch.cuda.empty_cache()
    (out/'results.json').write_text(json.dumps(results,indent=2))
    print('Completed',out,flush=True)


if __name__=='__main__':main()
