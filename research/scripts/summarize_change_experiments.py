"""Summarize saved eval-mode metrics only. Never loads weights or runs inference."""
import argparse
import json
from pathlib import Path
import pandas as pd


def summarize(manifest):
    rows = []
    for job in manifest['jobs']:
        root = Path(job['checkpoint_root'])
        completed = sorted(root.glob('**/*.dynamics/epoch_*/summary.json')) if root.exists() else []
        row = {k:job[k] for k in ['job_id','stage','seed','time_features','head_mode','change_loss','event_weight']}
        row['status'] = 'not_run'
        if completed:
            roots = {path.parent.parent for path in completed}
            if len(roots) != 1:
                raise ValueError(f'Multiple result trees for {job["job_id"]}; do not mix reruns')
            records = [(json.loads(path.read_text()),path) for path in completed]
            records.sort(key=lambda item:item[0]['epoch'])
            last,path = records[-1]
            epoch = last['selected_best_epoch']
            selected = path.parent.parent/f'epoch_{epoch:03d}'/'summary.json'
            row.update(status='available' if last.get('training_finished') else 'partial',
                last_epoch=last['epoch'],best_epoch=epoch,
                validation_fingerprint=last['validation_fingerprint'],best_val_loss=last['best_val_loss'])
            if not selected.is_file():
                row['status'] = 'selected_epoch_not_saved'
            else:
                for split in ['train_eval','val_eval']:
                    table = pd.read_csv(selected.parent/f'{split}_metrics.csv')
                    for group in ['all','extreme','rise','fall','non_extreme']:
                        r = table.loc[table.level.eq('pooled_all')&table.group.eq(group)].iloc[0]
                        for field in ['n','rmse_mm','mae_mm','bias_mm','rise_recall','rise_false_alarm_ratio',
                                      'rise_shortfall_mm','objective_contribution','wrong_sign_rate',
                                      'predicted_abs_mm','actual_abs_mm','large_errors','rise_false_positive_rate']:
                            row[f'{split}/{group}/{field}'] = r[field]
                row['snapshot'] = str(selected.parent/'model.pt')
        rows.append(row)
    # No ranking: costs differ across loss variants and probes do not measure
    # independent events. Explicit fingerprints reveal incompatible anchor sets.
    return pd.DataFrame(rows)


def history(manifest):
    """Keep every epoch/station/regime, not only the chosen checkpoint."""
    tables = []
    for job in manifest['jobs']:
        root = Path(job['checkpoint_root'])
        fingerprints = set()
        for path in sorted(root.glob('**/*.dynamics/epoch_*/summary.json')) if root.exists() else []:
            metadata = json.loads(path.read_text())
            fingerprints.add(metadata['validation_fingerprint'])
            if len(fingerprints)>1:
                raise ValueError(f'Validation anchors changed within {job["job_id"]}')
            for split in ['train_eval','val_eval']:
                frame=pd.read_csv(path.parent/f'{split}_metrics.csv',dtype={'station_id':str})
                frame.insert(0,'job_id',job['job_id'])
                frame['validation_fingerprint']=metadata['validation_fingerprint']
                tables.append(frame)
    return pd.concat(tables,ignore_index=True) if tables else pd.DataFrame(columns=['job_id','epoch','split'])


def gradient_history(manifest,filename):
    tables=[]
    for job in manifest['jobs']:
        root=Path(job['checkpoint_root'])
        for path in sorted(root.glob('**/*.dynamics/epoch_*/summary.json')) if root.exists() else []:
            artifact=path.parent/filename
            if artifact.is_file():
                frame=pd.read_csv(artifact,dtype={'station_id':str,'station_a':str,'station_b':str})
                frame.insert(0,'epoch',json.loads(path.read_text())['epoch'])
                frame.insert(0,'job_id',job['job_id'])
                tables.append(frame)
    return pd.concat(tables,ignore_index=True) if tables else pd.DataFrame(columns=['job_id','epoch'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    table = summarize(manifest)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    table.to_csv(args.output,index=False)
    history(manifest).to_csv(args.output.with_suffix('.history.csv'),index=False)
    gradient_history(manifest,'gradient_norms.csv').to_csv(args.output.with_suffix('.gradient_norms.csv'),index=False)
    gradient_history(manifest,'gradient_cosines.csv').to_csv(args.output.with_suffix('.gradient_cosines.csv'),index=False)
    print(table[['job_id','status']].to_string(index=False))
    print('Saved',args.output,'; no models or test metrics evaluated.')


if __name__ == '__main__':
    main()
