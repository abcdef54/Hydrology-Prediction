"""Write reviewable experiment commands; this program NEVER launches training."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex


def experiment_jobs(dataset, output, *, method='transformer-late-fusion', seeds=(17,29,42),
                    python='.venv/bin/python', tracking_uri='http://localhost:5000'):
    common = [python,'-m','src.ai.train','--dataset-dir',str(Path(dataset).resolve()),
        '--train-method',method,'--horizon','1h','--target-horizons','6','12',
        '--feature-set','water_level+rain+flood_season+weather_forecast','--rain-station','nham+hong_thai',
        '--use-weather-forecast','true','--regional-fold','fixed','--batch-size','256','--num-worker','0',
        '--num-epochs','20','--early-stopping','30','--gradient-accumulation','1',
        '--lstm-seq-len','96','--lstm-hidden-size','64','--lstm-num-layers','3',
        '--transformer-heads','4','--transformer-ff-size','128','--lstm-drop-out','0.3',
        '--lstm-input-dropout','0.2','--lstm-head-dropout','0.2','--station-embedding-size','8',
        '--forecast-hidden-size','32','--lstm-lr','0.0009','--lstm-weight-decay','0.001',
        '--lstm-correction-penalty','0','--lstm-horizon-weights','1','1','--lstm-tail-penalty','0',
        '--lstm-tail-threshold-mm','200','--lstm-tail-transition-mm','20','--lstm-checkpoint-metric','val_loss',
        '--change-scale-floor-mm','200','--change-under-weight','0','--change-threshold-mm','200',
        '--diagnose-change-dynamics','--dynamics-every','1','--skip-test','--device','cuda',
        '--mlflow-tracking-uri',tracking_uri,'--mlflow-ex-name','regional_change_dynamics']
    jobs = []
    def add(stage, name, seed, time='none', head='shared', loss='station-balanced', alpha=0., gradients=False):
        job_id = f'{name}_s{seed}'
        checkpoint_root = Path(output).resolve()/'runs'/job_id
        argv = common+['--seed',str(seed),'--time-features',time,'--head-mode',head,
            '--change-loss',loss,'--change-event-weight',str(alpha),'--checkpoint-dir',str(checkpoint_root)]
        if gradients:
            argv += ['--diagnose-station-gradients','--gradient-probe-samples','32']
        jobs.append(dict(job_id=job_id,stage=stage,seed=seed,time_features=time,head_mode=head,
            change_loss=loss,event_weight=alpha,checkpoint_root=str(checkpoint_root),argv=argv,
            script=f'commands/{job_id}.sh'))
    add('diagnostic','D0',42,gradients=True)
    for seed in seeds:
        for time in ['none','hour','day_of_year','all']:
            add('calendar',f'T_{time}',seed,time=time)
        for head in ['shared','station']:
            add('heads',f'H_{head}',seed,head=head,gradients=True)
        for name,loss,alpha in [('B1','station-balanced',0.),('E025','combined',.25),('E050','combined',.5)]:
            add('loss',f'L_{name}',seed,loss=loss,alpha=alpha)
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--dataset-dir',type=Path,default=Path('train_data/hue_regional_20261015_h6_h12_fixed_split_v1'))
    parser.add_argument('--method',default='transformer-late-fusion')
    parser.add_argument('--seeds',type=int,nargs='+',default=[17,29,42])
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(not 0<=s<2**32 for s in args.seeds):
        parser.error('Seeds must be unique and in [0,2**32)')
    jobs = experiment_jobs(args.dataset_dir,args.output_dir,method=args.method,seeds=args.seeds)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    (args.output_dir/'commands').mkdir()
    repository = Path(__file__).resolve().parents[2]
    for job in jobs:
        path = args.output_dir/job['script']
        path.write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd '+shlex.quote(str(repository))+
            '\n# Refuse reuse of an existing result directory.\n'+
            'if [ -e '+shlex.quote(job['checkpoint_root'])+' ]; then echo "Result directory already exists" >&2; exit 1; fi\n'+
            shlex.join(job['argv'])+'\n')
    manifest = dict(format_version=1,created_utc=datetime.now(timezone.utc).isoformat(),
        repository=str(repository),dataset=str(args.dataset_dir.resolve()),
        status='PREPARED_ONLY_NO_TRAINING_EXECUTED',jobs=jobs,
        gate='Run D0 first; review diagnostics before calendar, heads or loss stages. No automatic stage promotion.')
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (args.output_dir/'RUNBOOK.md').write_text(
        '# Thí nghiệm phần 4 — chỉ mới chuẩn bị\n\n'
        'Chưa chạy training. Mỗi script là một lệnh độc lập; không có script tự chạy cả ma trận.\n\n'
        'Chạy các lệnh từ root repository.\n\n'
        '1. Sau khi review code, chỉ chạy `bash '+str((args.output_dir/'commands/D0_s42.sh').resolve())+'`.\n'
        '2. Đọc sai số train_eval/val_eval, dự báo từng thời điểm và gradient trước khi chọn bước tiếp.\n'
        '3. Calendar: 4 lựa chọn × 3 seeds. Heads: shared/station × 3 seeds. Loss: B1/alpha .25/.5 × 3 seeds.\n'
        '4. Giữ tập dữ liệu, mưa, history và hai horizon cố định. Tất cả lệnh bỏ qua test.\n'
        '5. Tổng hợp bằng `.venv/bin/python -m research.scripts.summarize_change_experiments --manifest '+
        str((args.output_dir/'manifest.json').resolve())+' --output '+str((args.output_dir/'summary.csv').resolve())+'`.\n\n'
        'Chẩn đoán thêm một lượt đọc train và validation ở eval mode mỗi epoch, lưu snapshot và Parquet. '
        'Thời gian/dung lượng tăng; không dùng số đo này để so throughput với các run cũ. '
        'Checkpoint là weights + preprocessing, không phải trạng thái resume optimizer. '
        'Nhiều anchors trong cùng một sự kiện không phải các sự kiện độc lập.\n')
    print(f'Prepared {len(jobs)} commands in {args.output_dir}; NO training launched.')


if __name__ == '__main__':
    main()
