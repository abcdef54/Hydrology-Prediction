"""Audit saved epoch metrics and labels. Never constructs or runs a model.

Usage: .venv/bin/python -m research.scripts.analyze_checkpoint_dynamics --output-dir PATH
SQLite is opened read-only in one transaction. Train metrics historically have
step=0; recover their epoch from their timestamp preceding the validation log.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

import joblib
import numpy as np
import pandas as pd

RUNS = {
    'transformer_none_event0': '6b979576c9014d358a042a1633ce1659',
    'transformer_none_event2': 'fedeac3495f44cca807f40e1ba4d73c1',
    'transformer_all_event2': '55bd0ec9ba884acd8f40aa0d35fc3dc7',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def label_statistics(samples, mapping):
    rows = []
    for (split, sid), group in samples.groupby(['split', 'station_id']):
        if split not in ('train', 'validation', 'test_exploratory'):
            continue
        for horizon in (6, 12):
            d = group[f'delta_h{horizon}_mm']
            extreme, rise = d.abs() > 200, d > 200
            times = group.loc[extreme, 'anchor_utc'].sort_values()
            blocks = int(times.diff().ne(pd.Timedelta(hours=1)).sum()) if len(times) else 0
            rows.append(dict(split=split, station_id=sid, station=mapping[sid]['name'],
                horizon=horizon, n=len(d), extreme_n=int(extreme.sum()), rise_n=int(rise.sum()),
                extreme_pct=100*extreme.mean(), extreme_actual_abs_mm=d[extreme].abs().mean(),
                consecutive_extreme_blocks=blocks, delta_std_mm=d.std(ddof=0),
                first=str(group.anchor_utc.min()), last=str(group.anchor_utc.max())))
    return pd.DataFrame(rows)


def plot_dynamics(epochs, details, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    for ax, label, best, title in [(axes[0,0], 'transformer_none_event0',7,'Không bật time features'),
                                  (axes[0,1], 'transformer_all_event2',6,'Bật time features')]:
        v=epochs.loc[epochs.label.eq(label)&epochs.split.eq('val')]
        t=epochs.loc[epochs.label.eq(label)&epochs.split.eq('train')]
        ax.plot(v.epoch,v.logged_loss,label='Validation tổng',color='black')
        ax.plot(v.epoch,v.extreme_objective,label='Phần |ΔH| >200 mm',color='#c05b30')
        ax.plot(v.epoch,v.normal_objective,label='Phần |ΔH| ≤200 mm',color='#286a99')
        ax.plot(t.epoch,t.reconstructed_sample_weighted_loss,':',label='Train online, có dropout',color='#608954')
        ax.axvline(best,color='gray',ls='--',lw=1)
        ax.set(title=title,xlabel='Epoch',ylabel='Đóng góp objective B1')
        ax.legend(fontsize=8);ax.grid(alpha=.15)
    v=details.loc[details.label.eq('transformer_all_event2')&details.split.eq('val')]
    h=v.loc[v.station_id.eq('0074647606')&v.horizon.eq(12)]
    axes[1,0].plot(h.epoch,h.extreme_pred_abs_mm,label='Dự báo: |ΔH| TB khi extreme',color='#c05b30')
    axes[1,0].axhline(1341.298,color='gray',ls='--',label='Thực tế: |ΔH| TB khi extreme')
    axes[1,0].set(title='Huỳnh Liên +12h: biên độ trên validation',xlabel='Epoch',ylabel='mm')
    axes[1,0].legend(fontsize=8);axes[1,0].grid(alpha=.15)
    a=v.loc[v.epoch.eq(6)].groupby('station').objective_contribution.sum()
    b=v.loc[v.epoch.eq(17)].groupby('station').objective_contribution.sum()
    delta=(b-a).sort_values()
    names={'Cầu Huỳnh Liên-Văn Trạch Hòa(Sông Ô Lâu)':'Huỳnh Liên',
           'Cầu Hòa Xuân 1(Sông Ô Lâu)':'Hòa Xuân','Trạm Tứ Phú (Sông Bồ)':'Tứ Phú',
           'Trạm Ngã Ba Sình (Sông Hương)':'Ngã Ba Sình','Cảng Cá Thuận An':'Thuận An',
           'Thôn Phò Nam A, xã Quảng Thọ':'Phò Nam A','Thôn Trung Làng, xã Quảng Thái':'Trung Làng'}
    axes[1,1].barh([names.get(name,name) for name in delta.index],delta.values,
                  color=['#608954' if value<0 else '#c05b30' for value in delta])
    axes[1,1].axvline(0,color='gray',lw=1)
    axes[1,1].set(title='Time features: epoch 17 trừ epoch 6',xlabel='Thay đổi đóng góp validation loss')
    fig.suptitle('Dự báo lớn hơn: extreme cải thiện ở Huỳnh Liên, chi phí tăng ở nhóm khác',fontsize=13)
    fig.savefig(output,dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, default=Path('mlflow.db'))
    parser.add_argument('--dataset', type=Path,
                        default=Path('train_data/hue_regional_20261015_h6_h12_fixed_split_v1'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source = dict(dataset=str(args.dataset.resolve()), dataset_files={}, runs={}, code_files={})
    for name in ['samples.parquet', 'station_mapping.json', 'manifest.json', 'normalizers.json', 'config.json']:
        source['dataset_files'][name] = digest(args.dataset/name)
    for name in ['src/ai/trainers.py', 'src/ai/change_objective.py', 'src/ai/change_diagnostics.py',
                 'src/ai/residual_sequence.py', 'src/ai/regional_data/preprocessing.py', __file__]:
        source['code_files'][str(Path(name).resolve())] = digest(name)
    samples = pd.read_parquet(args.dataset/'samples.parquet')
    samples['station_id'] = samples.station_id.astype(str)
    mapping = json.loads((args.dataset/'station_mapping.json').read_text())
    stats = label_statistics(samples, mapping)
    stats.to_csv(args.output_dir/'label_distribution.csv', index=False)
    db = sqlite3.connect(f'file:{args.db.resolve()}?mode=ro', uri=True)
    db.execute('BEGIN')
    all_details, all_epochs, raw_metrics = [], [], []
    for label, run in RUNS.items():
        params = dict(db.execute('SELECT key,value FROM params WHERE run_uuid=?', (run,)))
        checkpoint = Path(params['checkpoint_path'])
        loss_path = checkpoint.with_suffix('.loss.json')
        loss_spec = json.loads(loss_path.read_text())
        assert loss_spec['mode'] == 'station-balanced'
        assert loss_spec['training_provenance']['dataset_manifest_sha256'] == source['dataset_files']['manifest.json']
        metrics = pd.read_sql_query('SELECT key,value,timestamp,step,is_nan FROM metrics WHERE run_uuid=?',
                                    db, params=(run,))
        metrics['label'] = label
        raw_metrics.append(metrics)
        val = metrics.loc[metrics.key.eq('val_loss')].sort_values('step')
        assert val.step.tolist() == list(range(1, 21))
        end_stamps = val.timestamp.to_numpy()
        train = metrics.loc[metrics.key.str.startswith('train/')].copy()
        assert train.step.eq(0).all()
        train['epoch'] = np.searchsorted(end_stamps, train.timestamp.to_numpy(), side='left')+1
        assert train.epoch.between(1, 20).all()
        for metric, group in train.groupby('key'):
            assert not group.epoch.duplicated().any(), metric
            # Undefined ratios (e.g. no predicted alarm) are intentionally not logged.
            if metric.endswith('/rmse_mm'):
                assert sorted(group.epoch) == list(range(1, 21)), metric
        validate = metrics.loc[metrics.key.str.startswith('val/') & metrics.step.between(1, 20)].copy()
        validate['epoch'] = validate.step
        lookup = {prefix: frame.set_index(['epoch', 'key']).value
                  for prefix, frame in [('train', train), ('val', validate)]}
        source['runs'][label] = dict(run_id=run, params=params,
            source_files={str(path): digest(path) for path in [checkpoint, loss_path,
                checkpoint.with_suffix('.architecture.json'), checkpoint.with_suffix('.preprocessing.joblib'),
                checkpoint.with_suffix('.train_change_metrics.csv'), checkpoint.with_suffix('.val_change_metrics.csv')]},
            train_epoch_alignment='Unique train metric timestamp in interval (previous val timestamp, current val timestamp]',
            loss_spec=loss_spec)
        saved = joblib.load(checkpoint.with_suffix('.preprocessing.joblib'))
        source['runs'][label]['features'] = saved['feature_columns']
        source['runs'][label]['calendar_features'] = [name for name in saved['feature_columns']
            if name in ('hour_sin', 'hour_cos', 'day_sin', 'day_cos')]
        best = int(metrics.loc[metrics.key.eq('best_checkpoint_epoch'), 'value'].iloc[-1])
        source['runs'][label]['best_epoch'] = best
        for prefix, split in [('train', 'train'), ('val', 'validation')]:
            split_stats = stats.loc[stats.split.eq(split)].set_index(['station_id','horizon'])
            total_n = int(samples.split.eq(split).sum())
            for epoch in range(1, 21):
                details = []
                for sid, meta in mapping.items():
                    for h_index, horizon in enumerate((6, 12)):
                        s = split_stats.loc[(sid,horizon)]
                        key = f'{prefix}/{sid}/target_water_level_plus_{horizon*60}m/'
                        row = dict(label=label, split=prefix, epoch=epoch, station_id=sid,
                                   station=meta['name'], horizon=horizon, n=s.n,
                                   extreme_n=s.extreme_n, rise_n=s.rise_n)
                        for field in ['rmse_mm','mae_mm','extreme_rmse_mm','extreme_pred_abs_mm',
                                      'correction_abs_mm','rise_bias_mm','rise_recall','rise_false_alarm_ratio',
                                      'rise_false_positive_rate','objective_share','large_errors']:
                            row[field] = lookup[prefix].get((epoch,key+field), np.nan)
                        assert np.isfinite(row['rmse_mm'])
                        i = meta['station_index']
                        denominator = max(loss_spec['target_scales'][i][h_index], loss_spec['scale_floor_mm'])
                        coefficient = loss_spec['station_frequency'][i]*loss_spec['horizon_weights'][h_index]/denominator**2
                        sse = row['rmse_mm']**2*s.n
                        extreme_sse = row['extreme_rmse_mm']**2*s.extreme_n if s.extreme_n else 0.
                        normal_sse = sse-extreme_sse
                        assert normal_sse >= -1e-5
                        row.update(denominator_mm=denominator, coefficient=coefficient,
                            sse=sse, extreme_sse=extreme_sse, normal_sse=normal_sse,
                            normal_rmse_mm=(normal_sse/(s.n-s.extreme_n))**.5,
                            objective_contribution=coefficient*sse/total_n,
                            extreme_objective_contribution=coefficient*extreme_sse/total_n,
                            normal_objective_contribution=coefficient*normal_sse/total_n)
                        details.append(row)
                frame = pd.DataFrame(details)
                objective = frame.objective_contribution.sum()
                np.testing.assert_allclose(frame.objective_share,
                    frame.objective_contribution/objective, atol=1e-7, rtol=1e-6)
                actual_loss = metrics.loc[metrics.key.eq('val_loss' if prefix=='val' else 'train_loss')
                                          & metrics.step.eq(epoch), 'value'].iloc[0]
                if prefix == 'val':
                    np.testing.assert_allclose(objective, actual_loss, rtol=1e-6, atol=1e-8)
                all_details.extend(details)
                all_epochs.append(dict(label=label, split=prefix, epoch=epoch,
                    logged_loss=actual_loss, reconstructed_sample_weighted_loss=objective,
                    extreme_objective=frame.extreme_objective_contribution.sum(),
                    normal_objective=frame.normal_objective_contribution.sum(),
                    pooled_rmse_mm=(frame.sse.sum()/frame.n.sum())**.5,
                    extreme_rmse_mm=(frame.extreme_sse.sum()/frame.extreme_n.sum())**.5,
                    normal_rmse_mm=(frame.normal_sse.sum()/(frame.n.sum()-frame.extreme_n.sum()))**.5,
                    large_errors=frame.large_errors.sum()))
    # Inventory of finished related runs, without using test metrics for selection.
    inventory = []
    for (run,) in db.execute("SELECT run_uuid FROM runs WHERE status='FINISHED'"):
        params = dict(db.execute('SELECT key,value FROM params WHERE run_uuid=?',(run,)))
        if not params.get('dataset_dir','').endswith(args.dataset.name):
            continue
        m = dict(db.execute("SELECT key,value FROM metrics WHERE run_uuid=? AND key IN ('best_checkpoint_epoch','epochs_completed','best_val_loss')",(run,)))
        if 'best_checkpoint_epoch' not in m:
            continue
        inventory.append(dict(run_id=run,method=params.get('train_method'), hidden=params.get('Hidden size'),
            sequence_length=params.get('sequence_length'), targets=params.get('target_horizons_hours'),
            input_size=params.get('input_size'),batch=params.get('batch_size'),loss=params.get('change_loss'),**m))
    db.rollback()
    db.close()
    pd.DataFrame(inventory).to_csv(args.output_dir/'run_inventory.csv',index=False)
    raw = pd.concat(raw_metrics,ignore_index=True)
    raw.to_csv(args.output_dir/'source_metrics.csv',index=False)
    source['metrics_sha256'] = digest(args.output_dir/'source_metrics.csv')
    details = pd.DataFrame(all_details)
    details.to_csv(args.output_dir/'station_epoch_metrics.csv',index=False)
    epochs = pd.DataFrame(all_epochs)
    epochs.to_csv(args.output_dir/'epoch_summary.csv',index=False)
    deltas = []
    for label in RUNS:
        best = source['runs'][label]['best_epoch']
        frame = details.loc[details.label.eq(label) & details.split.eq('val')]
        before = frame.loc[frame.epoch.eq(best)].set_index(['station_id','horizon'])
        for epoch in [17,20]:
            after = frame.loc[frame.epoch.eq(epoch)].set_index(['station_id','horizon'])
            for field in ['objective_contribution','extreme_objective_contribution','normal_objective_contribution','sse']:
                for (sid,horizon),delta in (after[field]-before[field]).items():
                    deltas.append(dict(label=label,from_epoch=best,to_epoch=epoch,station_id=sid,
                        station=mapping[sid]['name'],horizon=horizon,field=field,delta=delta))
    pd.DataFrame(deltas).to_csv(args.output_dir/'validation_deltas.csv',index=False)
    # Check that changing the inactive event flag leaves every saved learning
    # curve unchanged. This is descriptive replication, not an independent seed.
    first = details.loc[details.label.eq('transformer_none_event0')].drop(columns='label').reset_index(drop=True)
    second = details.loc[details.label.eq('transformer_none_event2')].drop(columns='label').reset_index(drop=True)
    source['inactive_event_weight_comparison'] = dict(identical_derived_station_rows=first.equals(second),
        compared_rows=len(first),
        identical_checkpoint_bytes=(digest(Path(source['runs']['transformer_none_event0']['params']['checkpoint_path']))
            == digest(Path(source['runs']['transformer_none_event2']['params']['checkpoint_path']))))
    plot_dynamics(epochs, details, args.output_dir/'checkpoint_dynamics.png')
    source['outputs'] = {p.name:digest(p) for p in args.output_dir.glob('*.csv')}
    (args.output_dir/'provenance.json').write_text(json.dumps(source,ensure_ascii=False,indent=2)+'\n')
    print('Saved audit:',args.output_dir)
    print('No model was loaded, trained or executed. All validation objectives reconstructed within 1e-6 relative tolerance.')


if __name__ == '__main__':
    main()
