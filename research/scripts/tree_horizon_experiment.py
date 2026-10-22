"""Tree baselines on the same forecast dataset/history as B, without app changes."""
import argparse
from contextlib import nullcontext
from datetime import datetime,timezone
import json
from pathlib import Path
from uuid import uuid4
import joblib
import numpy as np
import pandas as pd
import torch
from src.ai.tree_sequence import sequence_table,train_horizon,TreeSequenceForecast
from src.ai.comparison import load_candidate,evaluation_dataset,predict,DEFAULT_REGISTRY
from research.scripts.build_output24_dataset import COMMON,TARGETS,read_frame,assert_matched_population
from research.scripts.finetune_direction_experiment import SOURCE
from src.ai.metrics import markdown_table as table

ROOT=Path(__file__).resolve().parents[2]


def prepare(directory,outputs,sequence_length):
    spec=json.loads((directory/'manifest.json').read_text())
    if spec.get('variant','mask')!='mask':raise ValueError('Use mask dataset for native-NaN tree comparison')
    features=spec['feature_columns'];targets=COMMON if outputs==5 else TARGETS
    if len(set(features))!=len(features):raise ValueError('Duplicate features')
    frames={};data={}
    for name in ['train','val','test']:
        f=read_frame(directory/f'{name}.csv')
        if not set(targets).issubset(f):raise ValueError('Requested targets missing; build O24 dataset for --outputs 24')
        if set(TARGETS).issubset(f):assert_matched_population(f[TARGETS],f.water_level_depth,sequence_length)
        frames[name]=f.reset_index();data[name]=sequence_table(frames[name],features,targets,sequence_length)
        if not len(data[name]['indices']):raise ValueError('Empty sequence population')
    for left,right in [('train','val'),('val','test')]:
        if pd.to_datetime(frames[left].timestamp_utc,utc=True).max()+pd.Timedelta(hours=24)>=pd.to_datetime(frames[right].timestamp_utc,utc=True).min():
            raise ValueError('24-hour chronological purge required')
    return spec,features,targets,frames,data


def prediction_points(data,values,targets):
    parts=[]
    for j,col in enumerate(targets):
        minutes=int(col.rsplit('_',1)[-1][:-1])
        parts.append(pd.DataFrame(dict(anchor_utc=data['times'],target_utc=data['times']+pd.Timedelta(minutes=minutes),
            horizon_minutes=minutes,actual_mm=data['y'][:,j],prediction_mm=values[:,j],error_mm=values[:,j]-data['y'][:,j])))
    return pd.concat(parts,ignore_index=True)


def summarize(points,error='error_mm'):
    values=points[error].to_numpy(dtype=float)
    return dict(n=len(values),rmse_mm=float(np.sqrt(np.mean(values**2))),mae_mm=float(np.mean(abs(values))),
        above_200=int((abs(values)>200).sum()),above_200_pct=float(100*(abs(values)>200).mean()))


def compare(data,values,frame,targets,reference_frame,reference_saved,reference_model):
    ds=evaluation_dataset(reference_frame,reference_saved)
    if not pd.to_datetime(frame.timestamp_utc,utc=True).equals(pd.to_datetime(reference_frame.timestamp_utc,utc=True)):
        raise ValueError('Reference timestamps differ')
    indices=np.intersect1d(data['indices'],ds.end_indices)
    if not len(indices):raise ValueError('No shared anchors')
    reference=predict(reference_model,ds,indices,'cpu',128)
    selected=np.searchsorted(data['indices'],indices);parts=[]
    for j,col in enumerate(targets):
        if col not in reference_saved['target_columns']:continue
        np.testing.assert_allclose(frame[col].to_numpy()[indices],reference_frame[col].to_numpy()[indices],rtol=0,atol=1e-9)
        minutes=int(col.rsplit('_',1)[-1][:-1]);actual=frame[col].to_numpy()[indices]
        parts.append(pd.DataFrame(dict(anchor_utc=pd.to_datetime(frame.timestamp_utc,utc=True).iloc[indices].to_numpy(),
            horizon_minutes=minutes,actual_mm=actual,candidate_mm=values[selected,j],
            candidate_error_mm=values[selected,j]-actual,
            reference_error_mm=reference[:,reference_saved['target_columns'].index(col)]-actual)))
    if not parts:raise ValueError('No shared horizons')
    points=pd.concat(parts,ignore_index=True)
    rows=[]
    for h,g in points.groupby('horizon_minutes'):
        before=summarize(g,'reference_error_mm');after=summarize(g,'candidate_error_mm')
        rows.append(dict(horizon_hours=h//60,n=len(g),reference_rmse=before['rmse_mm'],candidate_rmse=after['rmse_mm'],
            delta_rmse=after['rmse_mm']-before['rmse_mm'],reference_bad=before['above_200'],reference_pct=before['above_200_pct'],
            candidate_bad=after['above_200'],candidate_pct=after['above_200_pct']))
    return pd.DataFrame(rows),points,dict(common_anchors=len(indices),candidate_eligible=len(data['indices']),reference_eligible=len(ds),
        first_anchor=str(pd.to_datetime(frame.timestamp_utc,utc=True).iloc[indices[0]].tz_convert('Asia/Ho_Chi_Minh')),
        last_anchor=str(pd.to_datetime(frame.timestamp_utc,utc=True).iloc[indices[-1]].tz_convert('Asia/Ho_Chi_Minh')),
        candidate=summarize(points,'candidate_error_mm'),reference=summarize(points,'reference_error_mm'))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-method',choices=['xgb','lgbm'],required=True)
    p.add_argument('--dataset-dir',type=Path,default=ROOT/'train_data/da_vien_1h_2026-10-15_forecast_mask_o24')
    p.add_argument('--outputs',type=int,choices=[5,24],default=5)
    p.add_argument('--sequence-length','--lstm-seq-len',dest='sequence_length',type=int,default=96)
    p.add_argument('--target-mode',choices=['residual','direct'],default='residual')
    p.add_argument('--checkpoint-metric',choices=['large_error_rate','rmse'],default='large_error_rate')
    p.add_argument('--n-estimators',type=int,default=500)
    p.add_argument('--early-stopping',type=int,default=50)
    p.add_argument('--learning-rate',type=float,default=.05)
    p.add_argument('--max-depth',type=int,default=4)
    p.add_argument('--num-leaves',type=int,default=15,help='LightGBM only')
    p.add_argument('--n-jobs',type=int,default=4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu',help='cuda for XGBoost only; B comparison uses CPU')
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--reference-checkpoint',type=Path,default=SOURCE)
    p.add_argument('--mlflow-ex-name',default='da_vien_tree_horizons')
    p.add_argument('--no-mlflow',action='store_true')
    p.add_argument('--check-only',action='store_true',help='Validate dataset/history only; no fitting, checkpoint or MLflow')
    return p


def main():
    p=parser();args=p.parse_args()
    for name in ['sequence_length','n_estimators','early_stopping','max_depth','num_leaves','n_jobs']:
        if getattr(args,name)<1:p.error(f'{name} must be positive')
    if not np.isfinite(args.learning_rate) or args.learning_rate<=0:p.error('Learning rate must be finite and positive')
    if not 0<=args.seed<2**31:p.error('Seed outside supported range')
    if args.train_method=='lgbm' and args.device!='cpu':p.error('LightGBM runner requires --device cpu')
    if args.device=='cuda' and not torch.cuda.is_available():p.error('CUDA unavailable; explicitly select --device cpu')
    spec,features,targets,frames,data=prepare(args.dataset_dir,args.outputs,args.sequence_length)
    print(json.dumps(dict(method=args.train_method,sequences={s:len(v['indices']) for s,v in data.items()},
        raw_features=len(features),lagged_features=data['train']['X'].shape[1],outputs=len(targets),target_mode=args.target_mode),indent=2),flush=True)
    if args.check_only:print('CHECK ONLY passed: no fitting or output files');return
    if not args.reference_checkpoint.is_file():raise FileNotFoundError('Reference checkpoint missing')
    import mlflow
    if not args.no_mlflow:
        mlflow.set_tracking_uri('http://localhost:5000');mlflow.set_experiment(args.mlflow_ex_name)
    out=args.output_dir or ROOT/'model/experiments'/f'{args.train_method}_o{args.outputs}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid4().hex[:6]}'
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    options=vars(args).copy();settings={k:str(v) if isinstance(v,Path) else v for k,v in options.items()}
    settings.update(objective='squared error in mm; not LSTM tail/custom loss',
        checkpoint_rule='per-horizon validation (N200, RMSE) or RMSE; persistence retained if no booster beats it',
        sequence_order='lag0 newest first, then lag1..95; all selected features at every lag',
        feature_columns=features,target_columns=targets,lagged_feature_count=len(data['train']['names']),
        missing_policy='native NaN plus original mask features; no imputer/scaler fitted',
        sequences={s:len(v['indices']) for s,v in data.items()})
    (out/'settings.json').write_text(json.dumps(settings,indent=2))
    (out/'lagged_features.json').write_text(json.dumps(data['train']['names'],indent=2))
    artifact=dict(train_method=args.train_method,target_mode=args.target_mode,sequence_length=args.sequence_length,
        feature_columns=features,target_columns=targets,horizons=[],status='training')
    with (nullcontext(None) if args.no_mlflow else mlflow.start_run(run_name=f'{args.train_method}_o{args.outputs}_{args.target_mode}')) as run:
        if run:mlflow.log_params({k:v for k,v in settings.items() if k not in ['feature_columns','target_columns']})
        for j,target in enumerate(targets):
            h=int(target.rsplit('_',1)[-1][:-1])//60
            path=out/f'horizon_{h:02d}h.{"json" if args.train_method=="xgb" else "txt"}'
            result=train_horizon(args.train_method,data['train'],data['val'],j,options,path)
            pd.DataFrame(result.pop('history')).to_csv(out/f'validation_h{h:02d}.csv',index=False)
            result['target']=target;artifact['horizons'].append(result)
            (out/'tree_model.json').write_text(json.dumps(artifact,indent=2))
            print(f'+{h}h: {result["kind"]}, selected rounds={result["best_round"]}, validation={result["validation_score"]}',flush=True)
            if run:mlflow.log_metric(f'selected_rounds_h{h}',result['best_round'])
        artifact['status']='complete'
        (out/'tree_model.json').write_text(json.dumps(artifact,indent=2))
        # Reload native artifacts before scoring, exercising the user inference path.
        model=TreeSequenceForecast(out);values=model.predict_table(data['test'])
        points=prediction_points(data['test'],values,targets);points.to_csv(out/'all_predictions.csv.gz',index=False)
        all_metrics=pd.DataFrame([dict(horizon_hours=h//60,**summarize(g)) for h,g in points.groupby('horizon_minutes')])
        all_metrics.to_csv(out/'all_horizon_metrics.csv',index=False)
        config=next(c for c in json.loads(DEFAULT_REGISTRY.read_text()) if c['station_id']=='0074647403')
        references=[('B',args.reference_checkpoint,args.reference_checkpoint.with_suffix('.preprocessing.joblib'),frames['test']),
            ('production',ROOT/config['model_path'],ROOT/config['preprocessing_path'],pd.read_csv(ROOT/'train_data/da_vien_1h_2026-10-15/test.csv'))]
        summaries={};text='# Tree so với B và production\n\n'
        text+=f'{args.train_method}, {args.outputs} outputs, lịch sử {args.sequence_length}h, target {args.target_mode}. Objective squared error; early stopping/chọn round bằng validation {args.checkpoint_metric} riêng từng horizon. Không dùng tail loss/correction penalty/aux-weight của LSTM. Không thay registry.\n\n'
        for name,path,pre,reference_frame in references:
            saved=joblib.load(pre);reference=load_candidate(path.read_bytes(),saved,'cpu')
            metrics,paired,coverage=compare(data['test'],values,frames['test'],targets,reference_frame,saved,reference)
            metrics.to_csv(out/f'comparison_{name}_metrics.csv',index=False);paired.to_csv(out/f'comparison_{name}_predictions.csv.gz',index=False)
            sequence_rows=[]
            for minutes,g in paired.groupby('horizon_minutes'):
                target_times=pd.to_datetime(g.anchor_utc,utc=True)+pd.Timedelta(minutes=int(minutes))
                for label,error_col in [(name,'reference_error_mm'),('tree','candidate_error_mm')]:
                    errors=pd.Series(g[error_col].to_numpy(),index=target_times).sort_index()
                    bad=errors[errors.abs().gt(200)]
                    lengths=bad.groupby(bad.index.to_series().diff().ne(pd.Timedelta(hours=1)).cumsum()).size()
                    sequence_rows.append(dict(horizon_hours=minutes//60,model=label,under=int(errors.lt(-200).sum()),
                        over=int(errors.gt(200).sum()),longest_run=int(lengths.max()) if len(lengths) else 0,
                        multi_run_points=int(lengths[lengths.ge(2)].sum())))
            sequence_metrics=pd.DataFrame(sequence_rows)
            sequence_metrics.to_csv(out/f'comparison_{name}_sequences.csv',index=False)
            summaries[name]=coverage
            text+=f'## So với {name}\n\nMốc đầu vào {coverage["first_anchor"]} → {coverage["last_anchor"]}; {coverage["common_anchors"]} anchor chung.\n\n'
            text+=table(metrics,list(metrics.columns))+'\n\n'
            text+='Chuỗi lỗi tại +12/+24 giờ:\n\n'+table(sequence_metrics[sequence_metrics.horizon_hours.isin([12,24])],list(sequence_metrics.columns))+'\n\n'
            text+=f'Tổng năm horizon chung: reference {coverage["reference"]["above_200"]} lỗi, tree {coverage["candidate"]["above_200"]} lỗi; RMSE {coverage["reference"]["rmse_mm"]:.2f} → {coverage["candidate"]["rmse_mm"]:.2f} mm.\n\n'
            print(name,json.dumps(coverage),flush=True)
            if run:mlflow.log_metrics({f'test_vs_{name}/above_200':coverage['candidate']['above_200'],f'test_vs_{name}/rmse_mm':coverage['candidate']['rmse_mm']})
        text+='Mỗi model dùng đúng preprocessing của nó; tree giữ raw NaN, reference dùng scaler đã lưu. Chỉ chấm giao anchor/horizon và kiểm tra actual khớp. Mẫu hiện tại là strict test, reset warmup theo từng split; test đã xem nhiều lần là khám phá. 24 output tree là 24 booster độc lập, không chia sẻ head/backbone. Những horizon chọn persistence được ghi rõ, không giả là booster cải thiện.\n\n'
        text+='[Metric toàn bộ horizon](all_horizon_metrics.csv) · [Metadata](tree_model.json)\n'
        (out/'REPORT.md').write_text(text);(out/'comparison_summary.json').write_text(json.dumps(summaries,indent=2))
    print('Report:',out/'REPORT.md',flush=True)


if __name__=='__main__':main()
