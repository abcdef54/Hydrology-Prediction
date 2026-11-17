"""Compare two frozen regional checkpoints with saved local forecasts on common anchors."""
import hashlib
import json
from pathlib import Path
import sqlite3

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from src.ai.regional_training_data import RegionalTrainingDataset, regional_collate
from src.ai.sequence_checkpoint import SequenceRuntime


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out = Path('reports/model_comparisons/regional_shortlist_vs_local_da_vien_20261030')
    out.mkdir(parents=True, exist_ok=False)
    local = Path('reports/model_comparisons/o2_h6_h12_20261023T072347886733052Z_vs_b/paired_predictions.csv')
    local_manifest = local.with_name('manifest.json')
    data_path = Path('train_data/hue_regional_20261015_h6_h12_fixed_split_v1')
    conn = sqlite3.connect('file:mlflow.db?mode=ro', uri=True)
    runs = {'cnn_gru_shared_s42': 'cbe5a23a76204adebd4b5606e24b293b',
            'gru_shared_s42': '6062cef58656406db91b6ef86251391c',
            'gru_late_station_s17': 'a1aead1527854d7bab637009bd172ac6',
            'transformer_station_h32_s42': '5c90d3c3360547c79dfb9bf64b5c1105',
            'transformer_station_h64_s42': '86a00b8689a94a518c5b5632bb387c3c'}
    original = pd.read_csv(local)
    original['anchor_utc'] = pd.to_datetime(original.anchor_utc, utc=True)
    original['horizon_hours'] = original.horizon_minutes // 60
    assert not original.duplicated(['anchor_utc', 'horizon_hours']).any()
    data = RegionalTrainingDataset(data_path, split='test_exploratory', fold='fixed',
        rain_stations='nham+hong_thai', use_weather_forecast=True, time_features='none',
        station_id='0074647403', sequence_length=96, target_horizons=[6, 12])
    frame = data.view.samples.copy()
    frame['anchor_utc'] = pd.to_datetime(frame.anchor_utc, utc=True)
    assert not frame.anchor_utc.duplicated().any()
    frame['dataset_index'] = np.arange(len(frame))
    rows = []
    for h in [6, 12]:
        subset = original[original.horizon_hours == h].merge(frame[['anchor_utc', 'dataset_index', 'water_now_mm', f'target_h{h}_mm']],
            on='anchor_utc', validate='one_to_one')
        assert np.allclose(subset.actual_mm, subset[f'target_h{h}_mm'], rtol=0, atol=1e-4)
        subset['current_mm'] = subset.water_now_mm
        rows.append(subset)
    paired = pd.concat(rows, ignore_index=True)
    anchors = sorted(set(paired[paired.horizon_hours == 6].anchor_utc) & set(paired[paired.horizon_hours == 12].anchor_utc))
    paired = paired[paired.anchor_utc.isin(anchors)].sort_values(['anchor_utc', 'horizon_hours']).reset_index(drop=True)
    assert len(paired) == len(anchors) * 2 and anchors
    indices = paired.drop_duplicates('anchor_utc').dataset_index.astype(int).tolist()
    loader = DataLoader(Subset(data, indices), batch_size=128, collate_fn=regional_collate, shuffle=False)
    paired['local_model_b'] = paired.serving_mm
    provenance = dict(dataset=str(data_path.resolve()), dataset_manifest_sha256=sha(data_path/'manifest.json'),
        local_saved_predictions=str(local.resolve()), local_saved_predictions_sha256=sha(local),
        local_source_manifest=str(local_manifest.resolve()), local_source_manifest_sha256=sha(local_manifest),
        local_reference={k: json.loads(local_manifest.read_text())[k] for k in
                         ['reference_model_id', 'serving_model_version', 'serving_checkpoint',
                          'serving_sha256', 'serving_preprocessing_sha256']},
        station_id='0074647403', anchors=len(anchors), models={}, tolerance_mm=1e-4,
        interpretation='Exploratory previously examined test')
    torch.set_num_threads(4)
    for name, rid in runs.items():
        params = dict(conn.execute('select key,value from params where run_uuid=?', (rid,)))
        ckpt = Path(params['checkpoint_path']); pre = ckpt.with_suffix('.preprocessing.joblib')
        saved = joblib.load(pre)
        assert saved['feature_columns'] == data.feature_cols
        assert saved['target_columns'] == data.target_cols
        assert saved['sequence_length'] == data.seq_len
        assert saved['regional']['dataset_manifest_sha256'] == provenance['dataset_manifest_sha256']
        assert saved['regional']['normalizers'] == data.view.normalizers
        runtime = SequenceRuntime(ckpt, device='cpu')
        forecasts = []
        for batch in loader:
            forecasts.append(runtime.predict_mm(batch['x'], batch['station_ids']))
        values = np.concatenate(forecasts)
        assert values.shape == (len(anchors), 2) and np.isfinite(values).all()
        paired[name] = values.reshape(-1)
        provenance['models'][name] = dict(mlflow_run_id=rid, checkpoint=str(ckpt.resolve()),
            checkpoint_sha256=sha(ckpt), preprocessing_sha256=sha(pre), learning_rate=params['learning_rate'])
    metrics = []
    for name in ['local_model_b', *runs]:
        for h in [6, 12]:
            d = paired[paired.horizon_hours == h];error = d[name]-d.actual_mm
            actual_rise = d.actual_mm-d.current_mm > 200
            alarm = d[name]-d.current_mm > 200
            hits = (actual_rise & alarm).sum(); false = (~actual_rise & alarm).sum()
            metrics.append(dict(model=name,horizon_hours=h,n=len(d),mae_mm=error.abs().mean(),
                rmse_mm=np.sqrt((error**2).mean()),bias_mm=error.mean(),large_errors=int((error.abs()>200).sum()),
                actual_rises=int(actual_rise.sum()),hits=int(hits),false_alarms=int(false),
                rise_recall=hits/actual_rise.sum() if actual_rise.sum() else np.nan,
                rise_false_alarm_ratio=false/alarm.sum() if alarm.sum() else np.nan,
                persistence_rmse_mm=np.sqrt(((d.current_mm-d.actual_mm)**2).mean())))
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    paired[['anchor_utc', 'target_utc', 'horizon_hours', 'actual_mm', 'current_mm',
            'local_model_b', *runs]].to_parquet(out/'paired_predictions.parquet',index=False)
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(pd.DataFrame(metrics).round(3).to_string(index=False))
    print('Saved to',out)


if __name__ == '__main__':
    main()
