"""O5/O24 training CLI: primary-horizon loss plus auxiliary hourly supervision.

Dataset preparation: python -m research.scripts.build_output24_dataset
Training is only started by main(); importing helpers never trains or contacts MLflow.
"""
import argparse
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import random
from contextlib import nullcontext
from uuid import uuid4
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:2')
os.environ.setdefault('MPLCONFIGDIR','/tmp/hydrology-matplotlib')
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import mlflow
from src.ai.dataset import HydrologyDataset
from src.ai.lstm import LSTM,ResidualLSTM
from src.ai.losses import CommonHorizonAuxiliaryLoss
from src.ai.comparison import load_candidate,paired_metrics
from research.scripts.build_output24_dataset import DEFAULT_OUTPUT,COMMON,TARGETS,read_frame,assert_matched_population
from research.scripts.finetune_direction_experiment import SOURCE
from src.ai.metrics import markdown_table as table
from src.ai.rain_stations import (
    available_rain_ids, combination_name, resolve_rain_stations, select_rain_columns,
)

ROOT=Path(__file__).resolve().parents[2]


def prepare_data(directory,outputs,sequence_length,rain_stations=None):
    directory=Path(directory)
    spec=json.loads((directory/'manifest.json').read_text())
    if spec.get('experiment')!='O5_O24' or spec['target_columns']!=TARGETS:
        raise ValueError('Build the O5/O24 dataset first with research.scripts.build_output24_dataset')
    features=spec['feature_columns'];selected=COMMON if outputs==5 else TARGETS
    rain_ids = resolve_rain_stations(
        rain_stations, available_rain_ids(features), spec.get('rain_station_aliases'),
    )
    features = select_rain_columns(features, rain_ids)
    frames={};datasets={}
    for split in ['train','val','test']:
        d=read_frame(directory/f'{split}.csv')
        assert_matched_population(d[TARGETS],d.water_level_depth,sequence_length)
        frames[split]=d.reset_index()
    fit=HydrologyDataset(frames['train'],feature_cols=features,target_cols=selected,
        fit_scalers=True,seq_len=sequence_length,require_current_level=True)
    saved=dict(feature_columns=features,target_columns=selected,sequence_length=sequence_length,
        feature_scaler=fit.feature_scaler,target_scaler=fit.target_scaler,
        missing_scaled_value=0.,raw_missing_value=None,forecast_history_policy='all_timesteps',
        rain_station_ids=available_rain_ids(features),experiment='O5_O24',
        rain_station_combination=combination_name(rain_ids),
        dataset_dir=str(directory.resolve()),eligibility_policy='matched complete 24h population',
        common_target_columns=COMMON)
    datasets['train']=fit
    for split in ['val','test']:
        datasets[split]=HydrologyDataset(frames[split],feature_cols=features,target_cols=selected,
            feature_scaler=fit.feature_scaler,target_scaler=fit.target_scaler,
            seq_len=sequence_length,require_current_level=True)
    if not all(len(d)>0 for d in datasets.values()):raise ValueError('Empty sequence population')
    return frames,datasets,saved


def make_model(saved,args,device):
    options=dict(input_size=len(saved['feature_columns']),output_size=len(saved['target_columns']),
        hidden_size=args.lstm_hidden_size,num_layers=args.lstm_num_layers,dropout=args.lstm_drop_out,
        input_dropout=args.lstm_input_dropout,head_dropout=args.lstm_head_dropout)
    architectures = {'lstm': LSTM, 'residual_lstm': ResidualLSTM}
    if args.train_method not in architectures:
        raise ValueError(f'Unsupported sequence model: {args.train_method}')
    architecture = architectures[args.train_method]
    if not issubclass(architecture,ResidualLSTM):
        return architecture(**options).to(device)
    fs,ts=saved['feature_scaler'],saved['target_scaler']
    index=saved['feature_columns'].index('water_level_depth')
    return architecture(**options,
        level_feature_index=index,persistence_scale=(fs.scale_[index]/ts.scale_).tolist(),
        persistence_offset=((fs.mean_[index]-ts.mean_)/ts.scale_).tolist()).to(device)


def evaluate(model,loader,loss,saved,device):
    model.eval();n=0;total=0.;primary_total=0.;extra_total=0.
    squares=torch.zeros(len(saved['target_columns']),dtype=torch.float64)
    bad200=torch.zeros_like(squares,dtype=torch.long);bad400=torch.zeros_like(bad200)
    scales=torch.as_tensor(saved['target_scaler'].scale_,device=device,dtype=torch.float64)
    with torch.inference_mode():
        for x,y in loader:
            x,y=x.to(device),y.to(device);pred=model(x)
            primary,extra=loss.components(pred,y);value=primary+loss.auxiliary_weight*extra
            if not torch.isfinite(value):raise ValueError('Non-finite validation loss')
            rows=len(y);n+=rows;total+=value.item()*rows
            primary_total+=primary.item()*rows;extra_total+=extra.item()*rows
            e=(pred.double()-y.double())*scales
            squares+=e.square().sum(0).cpu();bad200+=e.abs().gt(200).sum(0).cpu();bad400+=e.abs().gt(400).sum(0).cpu()
    if not n:raise ValueError('No eligible validation rows')
    primary=loss.primary_indices.cpu()
    result=dict(loss=total/n,primary_loss=primary_total/n,auxiliary_loss=extra_total/n,
        common_n=n*5,common_above_200=int(bad200[primary].sum()),common_above_400=int(bad400[primary].sum()),
        common_rmse_mm=math.sqrt(squares[primary].sum().item()/(n*5)),
        all_n=n*len(squares),all_above_200=int(bad200.sum()),all_above_400=int(bad400.sum()),
        all_rmse_mm=math.sqrt(squares.sum().item()/(n*len(squares))))
    for name,s,b,c in zip(saved['target_columns'],squares,bad200,bad400):
        result[f'horizon/{name}/rmse_mm']=math.sqrt(s.item()/n)
        result[f'horizon/{name}/above_200']=int(b)
        result[f'horizon/{name}/above_400']=int(c)
    return result


def validation_score(metrics,metric):
    return ((metrics['common_above_200'],metrics['common_rmse_mm'])
            if metric=='large_error_rate' else (metrics['loss'],))


def all_predictions(model,saved,frame,dataset,device):
    scaled=[]
    with torch.inference_mode():
        model.eval()
        for x,_ in DataLoader(dataset,batch_size=128,shuffle=False):scaled.append(model(x.to(device)).cpu().numpy())
    values=saved['target_scaler'].inverse_transform(np.concatenate(scaled))
    indices=dataset.end_indices
    timestamps=pd.to_datetime(frame.timestamp_utc,utc=True).iloc[indices].reset_index(drop=True)
    rows=[];points=[]
    for j,col in enumerate(saved['target_columns']):
        minutes=int(col.rsplit('_',1)[-1][:-1]);actual=frame[col].to_numpy()[indices]
        e=values[:,j].astype(np.float64)-actual
        if not np.isfinite(e).all():raise ValueError('Non-finite test predictions')
        rows.append(dict(horizon_hours=minutes//60,n=len(e),rmse_mm=np.sqrt(np.mean(e**2)),mae_mm=np.abs(e).mean(),
            above_200=int((abs(e)>200).sum()),above_200_pct=100*(abs(e)>200).mean(),above_400=int((abs(e)>400).sum())))
        points.append(pd.DataFrame(dict(anchor_utc=timestamps,target_utc=timestamps+pd.Timedelta(minutes=minutes),
            horizon_minutes=minutes,actual_mm=actual,prediction_mm=values[:,j],error_mm=e)))
    return pd.DataFrame(rows),pd.concat(points,ignore_index=True)


def write_comparisons(out,model,saved,frames,datasets,args):
    """Score shared horizons against B/production and all candidate horizons separately."""
    metrics,points=all_predictions(model,saved,frames['test'],datasets['test'],args.device)
    metrics.to_csv(out/'all_horizon_metrics.csv',index=False)
    points.to_csv(out/'all_horizon_predictions.csv.gz',index=False)
    references=[('B',args.reference_checkpoint,None)]
    registry=json.loads((ROOT/'src/backend/prediction/model_configs.json').read_text())
    config=next(c for c in registry if c['station_id']=='0074647403')
    references.append(('production',ROOT/config['model_path'],
        pd.read_csv(ROOT/'train_data/da_vien_1h_2026-10-15/test.csv')))
    errors={};summaries={};common_points=None;common_metrics=None
    for label,path,reference_frame in references:
        try:
            pre=(ROOT/config['preprocessing_path'] if label=='production'
                 else path.with_suffix('.preprocessing.joblib'))
            reference_saved=joblib.load(pre)
            if len(reference_saved['target_columns'])!=5 or set(reference_saved['target_columns'])!=set(COMMON):
                raise ValueError('Comparison reference must have exactly the five common horizons')
            reference=load_candidate(path.read_bytes(),reference_saved,args.device)
            m,s,p,c=paired_metrics(frames['test'],model,saved,reference,reference_saved,
                device=args.device,serving_frame=reference_frame)
            for name in ['candidate','serving']:
                s[name]['above_400']=int(p[f'{name}_error_mm'].abs().gt(400).sum())
            m.to_csv(out/f'comparison_{label}_metrics.csv',index=False)
            p.to_csv(out/f'comparison_{label}_predictions.csv.gz',index=False)
            summaries[label]=dict(summary=s,coverage=c,reference_checkpoint=str(path))
            if label=='B':
                common_points=p;common_metrics=m
                from src.ai.metrics import extrema_timing as timing
                detail,temporal=timing(p)
                detail.to_csv(out/'common_extrema.csv',index=False)
                temporal.to_csv(out/'common_extrema_summary.csv',index=False)
                diagnostics=[]
                for minutes,g in p.groupby('horizon_minutes'):
                    bad=g[g.candidate_error_mm.abs().gt(200)].sort_values('target_utc')
                    groups=pd.to_datetime(bad.target_utc,utc=True).diff().ne(pd.Timedelta(hours=1)).cumsum()
                    lengths=bad.groupby(groups).size()
                    diagnostics.append(dict(horizon_hours=minutes//60,
                        under=int(g.candidate_error_mm.lt(-200).sum()),over=int(g.candidate_error_mm.gt(200).sum()),
                        max_abs_error_mm=g.candidate_error_mm.abs().max(),
                        longest_bad_run=int(lengths.max()) if len(lengths) else 0))
                pd.DataFrame(diagnostics).to_csv(out/'common_error_runs.csv',index=False)
        except Exception as exc:
            errors[label]=f'{type(exc).__name__}: {exc}'
    (out/'comparison_summary.json').write_text(json.dumps(dict(references=summaries,errors=errors),ensure_ascii=False,indent=2)+'\n')
    text=f'# O{len(saved["target_columns"])} — dự báo mực nước trực tiếp theo giờ\n\n'
    text+='Primary loss giữ hệ số của 5 horizon +1/+3/+6/+12/+24. Các horizon phụ có loss trung bình riêng nhân auxiliary weight. Checkpoint được chọn trên validation trước khi chấm test. Không thay registry production.\n\n'
    if 'B' in summaries:
        r=summaries['B'];c=r['coverage'];s=r['summary']
        text+=f"## So với model 5 output tham chiếu (mặc định B)\n\nReference: `{args.reference_checkpoint}`.\n\n{c['common_anchors']} anchor, {c['first_anchor_vietnam']} → {c['last_anchor_vietnam']}.\n\n"
        display=[]
        for name,part in [('B',s['serving']),('Mới',s['candidate'])]:
            display.append(dict(model=name,RMSE=part['pooled_rmse_mm'],MAE=part['mae_mm'],
                n=part['n_predictions'],above_200=part['above_200'],percent=part['above_200_pct'],above_400=part['above_400']))
        text+=table(pd.DataFrame(display),['model','RMSE','MAE','n','above_200','percent','above_400'])+'\n\n'
        common=common_points
        pbad=common.serving_error_mm.abs().gt(200);cbad=common.candidate_error_mm.abs().gt(200)
        text+=f"Sửa {int((pbad&~cbad).sum())} lỗi B, tạo {int((~pbad&cbad).sum())} lỗi mới; so cùng các anchor.\n\n"
        text+=table(common_metrics,['horizon_hours','model','n','rmse_mm','above_200','above_200_pct'])+'\n\n'
    if 'production' in summaries:
        s=summaries['production']['summary']
        text+=f"## So với production trên năm horizon chung\n\nRMSE gộp production {s['serving']['pooled_rmse_mm']:.2f} mm, mới {s['candidate']['pooled_rmse_mm']:.2f} mm; lỗi >200 mm {s['serving']['above_200']} → {s['candidate']['above_200']}. Chi tiết ở comparison_production_metrics.csv.\n\n"
    text+='## Tất cả output của model mới\n\n'+table(metrics,['horizon_hours','n','rmse_mm','mae_mm','above_200','above_200_pct','above_400'])+'\n\n'
    text+='Không trộn tổng lỗi trên 24 horizon với tổng trên 5 horizon để kết luận cải thiện. B/production chỉ có 5 mốc để đối chiếu. Dataset/test hiện đã xem nhiều lần, chỉ là đánh giá khám phá; cần dữ liệu mới để xác nhận. Không so trực tiếp val_loss tổng O5 và O24.\n'
    if common_points is not None:
        text+='\nChuỗi lỗi thiếu/thừa ở common_error_runs.csv. Đỉnh/đáy ở common_extrema_summary.csv: prominence 50 mm, distance 6 giờ, ghép gần nhất một-một trong ±6 giờ, tách đoạn thiếu; lag dương là muộn. Đọc kèm số đỉnh không ghép được, không kết luận cải thiện độ trễ chỉ từ lag trung bình của các đỉnh ghép được.\n'
    if errors:text+='\nLỗi so sánh (training vẫn giữ nguyên checkpoint): '+json.dumps(errors,ensure_ascii=False)+'\n'
    if common_points is not None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        fig,axes=plt.subplots(5,1,figsize=(16,16))
        for ax,(minutes,g) in zip(axes,common_points.groupby('horizon_minutes')):
            g=g.sort_values('target_utc');t=pd.to_datetime(g.target_utc,utc=True).dt.tz_convert('Asia/Ho_Chi_Minh')
            # Reindex so a missing eligible hour cannot be silently connected.
            g=g.assign(time=t).set_index('time').reindex(pd.date_range(t.min(),t.max(),freq='h'))
            ax.plot(g.index,g.actual_mm,label='Thực tế',color='black')
            ax.plot(g.index,g.serving_mm,label='B',color='#777777')
            ax.plot(g.index,g.candidate_mm,label=f'O{len(saved["target_columns"])}',color='#1677b8')
            ax.set(ylabel='mm',title=f'+{minutes//60} giờ');ax.legend();ax.grid(alpha=.2)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m',tz=t.dt.tz))
        fig.tight_layout();fig.savefig(out/'common_horizon_predictions.png',dpi=150);plt.close(fig)
        text+='\n[Figure năm horizon chung](common_horizon_predictions.png)\n'
    (out/'REPORT.md').write_text(text)
    return summaries,errors


def training_objective(model,x,pred,data_loss,loss,penalty):
    if penalty==0:
        return data_loss
    if not isinstance(model,ResidualLSTM):
        raise ValueError('Nonzero correction penalty requires residual_lstm')
    return data_loss+penalty*loss.correction_loss(pred-model.persistence(x))


def correction_penalty(value):
    penalty=float(value)
    # MSE and residual reward use identical group weights, leaving a positive
    # quadratic coefficient exactly when 1 + penalty > 0. Tail loss is >= 0.
    if not math.isfinite(penalty) or penalty<=-1:
        raise argparse.ArgumentTypeError('correction penalty must be finite and > -1')
    return penalty


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--rain-stations',
                   help='Named combination, e.g. hong_thai+vp+bach_ma+phu_bai; omitted keeps dataset stations')
    p.add_argument('--outputs',type=int,choices=[5,24],required=True)
    p.add_argument('--train-method',choices=['residual_lstm','lstm'],default='residual_lstm')
    p.add_argument('--output-dir',type=Path,help='New model directory; default creates a unique directory')
    p.add_argument('--device',default='cuda')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--num-epochs',type=int,default=50)
    p.add_argument('--early-stopping',type=int,default=25)
    p.add_argument('--lstm-seq-len',type=int,default=96)
    p.add_argument('--lstm-hidden-size',type=int,default=32)
    p.add_argument('--lstm-num-layers',type=int,default=3)
    p.add_argument('--lstm-drop-out',type=float,default=.7)
    p.add_argument('--lstm-input-dropout',type=float,default=.2)
    p.add_argument('--lstm-head-dropout',type=float,default=.2)
    p.add_argument('--lstm-lr',type=float,default=.0009)
    p.add_argument('--lstm-weight-decay',type=float,default=.001)
    p.add_argument('--lstm-correction-penalty',type=correction_penalty,default=None,
                   help='Finite value > -1; default 0.1 for residual models, 0 for direct models')
    p.add_argument('--lstm-tail-penalty',type=float,default=.5)
    p.add_argument('--lstm-tail-threshold-mm',type=float,default=200.)
    p.add_argument('--lstm-tail-transition-mm',type=float,default=20.)
    p.add_argument('--aux-weight',type=float,default=.25)
    p.add_argument('--checkpoint-metric',choices=['large_error_rate','val_loss'],default='large_error_rate')
    p.add_argument('--reference-checkpoint',type=Path,default=SOURCE)
    p.add_argument('--mlflow-ex-name',default='da_vien_output_horizons')
    p.add_argument('--no-mlflow',action='store_true')
    p.add_argument('--check-only',action='store_true',help='Validate data/model/loss/backward; no optimizer step, training, checkpoint or MLflow')
    return p


def main():
    p=parser();args=p.parse_args()
    residual=args.train_method == 'residual_lstm'
    if args.lstm_correction_penalty is None:
        args.lstm_correction_penalty=.1 if residual else 0.
    if not residual and args.lstm_correction_penalty!=0:
        p.error(f'--train-method {args.train_method} requires --lstm-correction-penalty 0')
    for name in ['batch_size','num_epochs','early_stopping','lstm_seq_len','lstm_hidden_size','lstm_num_layers']:
        if getattr(args,name)<1:p.error(f'{name} must be positive')
    for name in ['lstm_weight_decay','lstm_tail_penalty','aux_weight']:
        if not math.isfinite(getattr(args,name)) or getattr(args,name)<0:p.error(f'{name} must be finite and nonnegative')
    for name in ['lstm_lr','lstm_tail_threshold_mm','lstm_tail_transition_mm']:
        if not math.isfinite(getattr(args,name)) or getattr(args,name)<=0:p.error(f'{name} must be finite and positive')
    if not 0<=args.seed<2**32:p.error('seed outside uint32 range')
    if args.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; explicitly use --device cpu if intended')
    if args.outputs==24 and args.aux_weight<=0:p.error('O24 requires --aux-weight >0 to supervise intermediate outputs')
    if not args.mlflow_ex_name.strip():p.error('MLflow experiment name must not be empty')
    if args.output_dir is not None and args.output_dir.exists():raise FileExistsError(args.output_dir)
    torch.set_num_threads(2);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    frames,datasets,saved=prepare_data(args.dataset_dir,args.outputs,args.lstm_seq_len,args.rain_stations)
    saved['train_method']=args.train_method
    model=make_model(saved,args,args.device)
    loss=CommonHorizonAuxiliaryLoss(saved['target_columns'],saved['target_scaler'].scale_,
        auxiliary_weight=args.aux_weight if args.outputs==24 else 0.,tail_strength=args.lstm_tail_penalty,
        threshold_mm=args.lstm_tail_threshold_mm,transition_mm=args.lstm_tail_transition_mm).to(args.device)
    settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    settings.update(input_size=len(saved['feature_columns']),output_size=len(saved['target_columns']),
        rain_station_ids=saved['rain_station_ids'],rain_station_combination=saved['rain_station_combination'],
        trainable_parameters=sum(v.numel() for v in model.parameters()),
        sequences={k:len(v) for k,v in datasets.items()},common_target_columns=COMMON,
        target_columns=saved['target_columns'],effective_aux_weight=loss.auxiliary_weight,
        loss_formula='B_loss(common5) + aux_weight * B_loss(extra19); same partition for correction regularization',
        checkpoint_scope='five common horizons only when large_error_rate',
        initialization='random; no B weights loaded',forecast_history_policy='all_timesteps')
    print(json.dumps(settings,ensure_ascii=False,indent=2),flush=True)
    if args.check_only:
        x,y=next(iter(DataLoader(datasets['train'],batch_size=2,shuffle=False)))
        x,y=x.to(args.device),y.to(args.device);pred=model(x)
        value=training_objective(model,x,pred,loss(pred,y),loss,args.lstm_correction_penalty)
        value.backward()
        if not torch.isfinite(value) or not all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None):
            raise ValueError('Non-finite forward/backward')
        print('CHECK ONLY passed: no optimizer step, no training, no checkpoint, no MLflow',flush=True)
        return
    if not args.reference_checkpoint.is_file() or not args.reference_checkpoint.with_suffix('.preprocessing.joblib').is_file():
        raise FileNotFoundError('Reference B checkpoint/preprocessing missing')
    default_root = ROOT/'model/experiments'
    if args.rain_stations is not None:
        default_root /= 'rain_' + saved['rain_station_combination']
    out=args.output_dir or default_root/f'{args.train_method}_o{args.outputs}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid4().hex[:6]}'
    out=out.resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'settings.json').write_text(json.dumps(settings,ensure_ascii=False,indent=2)+'\n')
    saved.update(auxiliary_weight=loss.auxiliary_weight,loss_config=dict(tail_strength=args.lstm_tail_penalty,
        threshold_mm=args.lstm_tail_threshold_mm,transition_mm=args.lstm_tail_transition_mm),checkpoint_scope=COMMON)
    for name in ['model_best','model_lowest_loss']:joblib.dump(saved,out/f'{name}.preprocessing.joblib')
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lstm_lr,weight_decay=args.lstm_weight_decay)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=.9)
    train=DataLoader(datasets['train'],batch_size=args.batch_size,shuffle=True,drop_last=True,
        pin_memory=args.device.startswith('cuda'),generator=torch.Generator().manual_seed(args.seed))
    val=DataLoader(datasets['val'],batch_size=args.batch_size,shuffle=False,generator=torch.Generator().manual_seed(args.seed+1))
    if not len(train):raise ValueError('Batch size exceeds training population')
    if not args.no_mlflow:
        mlflow.set_tracking_uri('http://localhost:5000');mlflow.set_experiment(args.mlflow_ex_name)
    with (nullcontext(None) if args.no_mlflow else mlflow.start_run(run_name=f'{args.train_method}_o{args.outputs}_aux{loss.auxiliary_weight:g}_seed{args.seed}')) as run:
        if run:
            mlflow.log_params(settings);mlflow.log_param('checkpoint_path',str(out/'model_best.pt'))
        baseline=evaluate(model,val,loss,saved,args.device)
        best=dict(baseline);best_epoch=0;wait=0;lowest_loss=baseline['loss'];lowest_epoch=0
        for name in ['model_best','model_lowest_loss']:torch.save(model.state_dict(),out/f'{name}.pt')
        history=[dict(epoch=0,**baseline)]
        if run:mlflow.log_metrics({f'val/{k}':v for k,v in baseline.items()},step=0)
        initial_label='persistence' if residual else f'untrained {args.train_method}'
        print(f'Epoch 0 {initial_label}: common N200={baseline["common_above_200"]}, '
              f'RMSE={baseline["common_rmse_mm"]:.3f}; combined loss={baseline["loss"]:.5f}',flush=True)
        for epoch in range(1,args.num_epochs+1):
            model.train();total=0.;count=0
            for x,y in train:
                x,y=x.to(args.device),y.to(args.device);optimizer.zero_grad(set_to_none=True)
                pred=model(x);data_loss=loss(pred,y)
                value=training_objective(model,x,pred,data_loss,loss,args.lstm_correction_penalty)
                if not torch.isfinite(value):raise ValueError('Non-finite objective')
                value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();total+=data_loss.item();count+=1
            scheduler.step();current=evaluate(model,val,loss,saved,args.device)
            history.append(dict(epoch=epoch,train_loss=total/count,**current))
            if current['loss']<lowest_loss:
                lowest_loss=current['loss'];lowest_epoch=epoch;torch.save(model.state_dict(),out/'model_lowest_loss.pt')
            if validation_score(current,args.checkpoint_metric)<validation_score(best,args.checkpoint_metric):
                best=dict(current);best_epoch=epoch;wait=0;torch.save(model.state_dict(),out/'model_best.pt')
            else:wait+=1
            pd.DataFrame(history).to_csv(out/'validation_history.csv',index=False)
            if run:mlflow.log_metrics({**{f'val/{k}':v for k,v in current.items()},'train/data_loss':total/count},step=epoch)
            print(f'Epoch {epoch}: common N200={current["common_above_200"]}, N400={current["common_above_400"]}, '
                  f'RMSE={current["common_rmse_mm"]:.3f}; combined loss={current["loss"]:.5f}; best={best_epoch}',flush=True)
            if wait>=args.early_stopping:break
        result=dict(run_id=run.info.run_id if run else None,epochs_completed=epoch,selected_epoch=best_epoch,
            baseline_validation=baseline,selected_validation=best,lowest_loss_epoch=lowest_epoch,
            lowest_loss=lowest_loss,checkpoint=str(out/'model_best.pt'))
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        model.load_state_dict(torch.load(out/'model_best.pt',map_location=args.device,weights_only=True))
        model.eval()
        summaries,errors=write_comparisons(out,model,saved,frames,datasets,args)
        for label,r in summaries.items():
            before=r['summary']['serving'];after=r['summary']['candidate']
            print(f'Test common5 vs {label}: RMSE {before["pooled_rmse_mm"]:.2f} → {after["pooled_rmse_mm"]:.2f} mm; '
                  f'N200 {before["above_200"]} → {after["above_200"]}',flush=True)
        if errors:print('Comparison errors:',errors,flush=True)
        if run:
            mlflow.log_metrics({'selected_epoch':best_epoch,'lowest_loss_epoch':lowest_epoch})
            for label,r in summaries.items():
                mlflow.log_metrics({f'test_common/{label}/candidate_rmse_mm':r['summary']['candidate']['pooled_rmse_mm'],
                    f'test_common/{label}/candidate_above_200':r['summary']['candidate']['above_200']})
    print('Checkpoint:',out/'model_best.pt',flush=True)
    print('Report:',out/'REPORT.md',flush=True)


if __name__=='__main__':main()
