"""BL-01: controlled fine-tuning of restored B, using its unchanged scalers.

Run: python -m research.finetune_direction_experiment --output-dir <new directory>
Each arm starts from B (not from the other arm). Test is scored after validation
selection only. No model upload or production registry update.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:2')

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import mlflow

from src.ai.comparison import evaluation_dataset, load_candidate, paired_metrics
from src.ai.losses import DirectionAmplitudeLoss, HorizonWeightedTailLoss
from src.ai.lstm import ResidualLSTM

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'model/experiments/model_b_retrain_20261019/rain_750007-750062/tail_0.5_threshold_200_transition_20_large_error_rate/1h/ResidualLSTM_water_level+rain+flood_season+weather_forecast_best.pt'
DATA=ROOT/'train_data/da_vien_1h_2026-10-15_forecast_mask'


def objective(loss,model,x,p,y):
    if isinstance(loss,DirectionAmplitudeLoss):
        return loss(p,y,model.persistence(x).detach())
    return loss(p,y)


def evaluate(model,loader,loss,device,scales):
    model.eval(); n=0; squared=0.; bad={200:0,400:0}; total=0.; components={}
    scale=torch.as_tensor(scales,device=device,dtype=torch.float64)
    with torch.inference_mode():
        for x,y in loader:
            x,y=x.to(device),y.to(device);p=model(x)
            value=objective(loss,model,x,p,y)
            if not torch.isfinite(value):raise ValueError('Non-finite validation loss')
            total+=value.item()*y.numel();n+=y.numel()
            error=(p.double()-y.double())*scale
            squared+=error.square().sum().item()
            for threshold in bad:bad[threshold]+=error.abs().gt(threshold).sum().item()
            if isinstance(loss,DirectionAmplitudeLoss):
                c=loss.components(p,y,model.persistence(x))
            else:
                mse,tail=loss.components(p,y);c=dict(base_weighted_mse=mse,soft_tail=tail)
            for key,v in c.items():components[key]=components.get(key,0.)+v.item()*y.numel()
    return dict(loss=total/n,rmse_mm=math.sqrt(squared/n),n=n,
                above_200=bad[200],above_400=bad[400],
                **{f'component/{k}':v/n for k,v in components.items()})


def choose_checkpoint(current,baseline,best):
    # Both error thresholds must remain no worse than B on validation.
    eligible=(current['above_200']<=baseline['above_200']
              and current['above_400']<=baseline['above_400'])
    score=lambda m:(m['above_200'],m['above_400'],m['rmse_mm'])
    return eligible and score(current)<score(best)


def make_model(saved,device,initialize_from_B=True):
    fs,ts=saved['feature_scaler'],saved['target_scaler']
    index=saved['feature_columns'].index('water_level_depth')
    policy=saved.get('forecast_history_policy','all_timesteps')
    weather=[i for i,c in enumerate(saved['feature_columns']) if c.startswith('weather_forecast_')]
    masks=[i for i in weather if saved['feature_columns'][i].endswith('_missing')]
    model=ResidualLSTM(input_size=len(saved['feature_columns']),hidden_size=32,num_layers=3,
        output_size=5,dropout=.7,input_dropout=.2,head_dropout=.2,level_feature_index=index,
        forecast_history_policy=policy,weather_feature_indices=weather,weather_mask_indices=masks,
        weather_mask_values=((1.-fs.mean_[masks])/fs.scale_[masks]).tolist(),
        persistence_scale=(fs.scale_[index]/ts.scale_).tolist(),
        persistence_offset=((fs.mean_[index]-ts.mean_)/ts.scale_).tolist())
    if initialize_from_B:
        model.load_state_dict(torch.load(SOURCE,map_location='cpu',weights_only=True),strict=True)
    return model.to(device)


def loss_arms(tail_strengths=None):
    if tail_strengths is None:
        return [('control_B_loss', .5), ('direction_amplitude_excess', None)]
    if not tail_strengths or any(not math.isfinite(v) or v<0 for v in tail_strengths):
        raise ValueError('Tail strengths must be finite and nonnegative')
    if len(set(tail_strengths))!=len(tail_strengths):
        raise ValueError('Duplicate tail strengths would share an output directory')
    return [(f'tail_{v:g}',v) for v in tail_strengths]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--patience',type=int,default=10)
    parser.add_argument('--lr',type=float,default=.0001)
    parser.add_argument('--mode',choices=['finetune','scratch'],default='finetune')
    parser.add_argument('--tail-strengths',type=float,nargs='+',
                        help='BL-02: use only B original loss, one arm per strength')
    args=parser.parse_args()
    if args.epochs<1 or args.patience<1 or not math.isfinite(args.lr) or args.lr<=0:
        parser.error('Epochs/patience/lr must be finite and positive')
    try:
        arms=loss_arms(args.tail_strengths)
    except ValueError as exc:
        parser.error(str(exc))
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; refusing silent CPU fallback')
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    device='cuda';torch.set_num_threads(2)
    saved=joblib.load(SOURCE.with_suffix('.preprocessing.joblib'))
    frames={s:pd.read_csv(DATA/f'{s}.csv') for s in ['train','val','test']}
    datasets={s:evaluation_dataset(frame,saved) for s,frame in frames.items()}
    assert [len(datasets[s]) for s in ['train','val','test']]==[6095,985,592]
    scales=saved['target_scaler'].scale_
    bmodel=load_candidate(SOURCE.read_bytes(),saved,device)
    # Only validation is evaluated before choosing either fine-tuned model.
    settings=dict(mode=args.mode,seed=42,lr=args.lr,epochs=args.epochs,patience=args.patience,
        batch_size=128,weight_decay=.001,correction_penalty=.1,gradient_clip=1.,
        direction_strength=.05,amplitude_strength=.1,excess_strength=.1,
        neutral_mm=10.,direction_scale_mm=50.,reward_scale_mm=100.,
        amplitude_scale_mm=200.,excess_scale_mm=100.,quartic=.25,
        source_checkpoint=str(SOURCE) if args.mode=='finetune' else 'none (random initialization)',
        scaler_source=str(SOURCE.with_suffix('.preprocessing.joblib')),
        dataset=str(DATA),gpu=torch.cuda.get_device_name(0),
        checkpoint_rule='min (N200,N400,RMSE), N200/N400 <= epoch-0 validation (B for finetune, persistence for scratch)',
        minimum_loss_checkpoint='saved separately; not selected using test')
    if args.tail_strengths is not None:
        settings.update(experiment='BL-02',tail_strengths=args.tail_strengths,
                        direction_strength=0.,amplitude_strength=0.,excess_strength=0.)
    (out/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    mlflow.set_tracking_uri('http://localhost:5000')
    mlflow.set_experiment('da_vien_tail_strength_finetune' if args.tail_strengths is not None
                         else 'da_vien_direction_amplitude_'+args.mode)
    results={}
    for arm,tail_strength in arms:
        folder=out/arm;folder.mkdir()
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False
        torch.use_deterministic_algorithms(True)
        generator=torch.Generator().manual_seed(42)
        train=DataLoader(datasets['train'],batch_size=128,shuffle=True,drop_last=True,
                         generator=generator,pin_memory=True)
        loader=DataLoader(datasets['val'],batch_size=128,shuffle=False,
                          generator=torch.Generator().manual_seed(43))
        model=make_model(saved,device,initialize_from_B=args.mode=='finetune')
        loss=(HorizonWeightedTailLoss([1]*5,scales,strength=tail_strength) if tail_strength is not None
              else DirectionAmplitudeLoss([1]*5,scales)).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.001)
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=.9)
        baseline=evaluate(model,loader,loss,device,scales)
        best=dict(baseline);best_epoch=0;wait=0;loss_best=baseline['loss'];loss_epoch=0
        checkpoint=folder/'model_best.pt'
        torch.save(model.state_dict(),checkpoint)
        torch.save(model.state_dict(),folder/'model_lowest_loss.pt')
        for name in ['model_best','model_lowest_loss']:
            joblib.dump(saved,folder/f'{name}.preprocessing.joblib')
        log=[dict(epoch=0,**baseline)]
        with mlflow.start_run(run_name=arm) as run:
            mlflow.log_params({**settings,'arm':arm,'loss_function':type(loss).__name__,
                               'tail_penalty':tail_strength if tail_strength is not None else .5,
                               'effective_direction_strength':0. if tail_strength is not None else .05,
                               'effective_amplitude_strength':0. if tail_strength is not None else .1,
                               'effective_excess_strength':0. if tail_strength is not None else .1,
                               'checkpoint_path':str(checkpoint)})
            mlflow.log_dict({'feature_columns':saved['feature_columns']},'features.json')
            mlflow.log_metrics({f'val/{k}':v for k,v in baseline.items()},step=0)
            print(f'{arm}: baseline validation {baseline}',flush=True)
            for epoch in range(1,args.epochs+1):
                model.train();sum_loss=0.;batches=0
                for x,y in train:
                    x,y=x.to(device),y.to(device);optimizer.zero_grad(set_to_none=True)
                    p=model(x);data_loss=objective(loss,model,x,p,y)
                    value=data_loss+.1*(p-model.persistence(x)).square().mean()
                    if not torch.isfinite(value):raise ValueError('Non-finite training loss')
                    value.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    optimizer.step();sum_loss+=data_loss.item();batches+=1
                scheduler.step()
                current=evaluate(model,loader,loss,device,scales)
                log.append(dict(epoch=epoch,train_loss=sum_loss/batches,**current))
                mlflow.log_metrics({**{f'val/{k}':v for k,v in current.items()},
                                    'train/data_loss':sum_loss/batches},step=epoch)
                if current['loss']<loss_best:
                    loss_best=current['loss'];loss_epoch=epoch
                    torch.save(model.state_dict(),folder/'model_lowest_loss.pt')
                if choose_checkpoint(current,baseline,best):
                    best=dict(current);best_epoch=epoch;wait=0
                    torch.save(model.state_dict(),checkpoint)
                else:wait+=1
                print(f'{arm} epoch {epoch}: N200={current["above_200"]}, '
                      f'N400={current["above_400"]}, RMSE={current["rmse_mm"]:.3f}, '
                      f'loss={current["loss"]:.4f}; selected={best_epoch}',flush=True)
                pd.DataFrame(log).to_csv(folder/'validation_history.csv',index=False)
                if wait>=args.patience:break
            model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True))
            metrics,summary,points,coverage=paired_metrics(frames['test'],model,saved,bmodel,saved,device=device)
            metrics.to_csv(folder/'test_metrics.csv',index=False)
            points.to_csv(folder/'test_predictions.csv',index=False)
            result=dict(run_id=run.info.run_id,selected_epoch=best_epoch,
                        tail_penalty=tail_strength,
                        baseline_validation=baseline,selected_validation=best,
                        lowest_loss_epoch=loss_epoch,lowest_loss=loss_best,
                        test=summary,coverage=coverage,epochs_completed=epoch)
            (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n')
            mlflow.log_metrics({'selected_epoch':best_epoch,'test/above_200':summary['candidate']['above_200'],
                               'test/pooled_rmse_mm':summary['candidate']['pooled_rmse_mm']})
            results[arm]=result
            print(arm+' TEST '+json.dumps(summary['candidate']),flush=True)
        del model,loss,optimizer
        torch.cuda.empty_cache()
    (out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print('Completed '+str(out),flush=True)


if __name__=='__main__':main()
