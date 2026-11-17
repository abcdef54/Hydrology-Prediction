"""Evaluate frozen shortlist across all stations on the existing common timestamps."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from src.ai.regional_training_data import RegionalTrainingDataset, regional_collate
from src.ai.sequence_checkpoint import SequenceRuntime


def main():
    reference = Path('reports/model_comparisons/regional_shortlist_vs_local_da_vien_20261030')
    output = Path('reports/model_comparisons/regional_shortlist_all_station_errors_20261030')
    output.mkdir(parents=True, exist_ok=False)
    provenance = json.loads((reference/'provenance.json').read_text())
    previous = pd.read_parquet(reference/'paired_predictions.parquet')
    anchors = pd.DatetimeIndex(pd.to_datetime(previous.anchor_utc, utc=True).unique()).sort_values()
    data = RegionalTrainingDataset(provenance['dataset'], split='test_exploratory', fold='fixed',
        rain_stations='nham+hong_thai', use_weather_forecast=True, time_features='none',
        sequence_length=96, target_horizons=[6, 12])
    frame = data.view.samples.copy()
    frame['anchor_utc'] = pd.to_datetime(frame.anchor_utc, utc=True)
    frame['dataset_index'] = np.arange(len(frame))
    frame = frame[frame.anchor_utc.isin(anchors)].sort_values(['station_id', 'anchor_utc']).reset_index(drop=True)
    assert not frame.duplicated(['station_id', 'anchor_utc']).any()
    assert len(frame.station_id.unique()) == 11
    sizes = frame.groupby('station_id').size()
    assert (sizes == len(anchors)).all(), sizes
    loader = DataLoader(Subset(data, frame.dataset_index.tolist()), batch_size=128,
                        collate_fn=regional_collate, shuffle=False)
    names = {sid: item.get('name', sid) for sid, item in data.view.mapping.items()}
    torch.set_num_threads(4)
    all_errors = []
    for model, info in provenance['models'].items():
        saved = joblib.load(Path(info['checkpoint']).with_suffix('.preprocessing.joblib'))
        assert saved['feature_columns'] == data.feature_cols and saved['target_columns'] == data.target_cols
        assert saved['sequence_length'] == data.seq_len
        assert saved['regional']['dataset_manifest_sha256'] == provenance['dataset_manifest_sha256']
        assert saved['regional']['normalizers'] == data.view.normalizers
        runtime = SequenceRuntime(info['checkpoint'], device='cpu')
        output_values = np.concatenate([runtime.predict_mm(batch['x'], batch['station_ids']) for batch in loader])
        assert output_values.shape == (len(frame), 2) and np.isfinite(output_values).all()
        davi = frame.station_id.eq('0074647403').to_numpy()
        check = frame.loc[davi, ['anchor_utc']].copy()
        check['h6'] = output_values[davi, 0]; check['h12'] = output_values[davi, 1]
        for h in [6, 12]:
            pair = previous[previous.horizon_hours == h][['anchor_utc', model]].merge(
                check[['anchor_utc', f'h{h}']], on='anchor_utc', validate='one_to_one')
            assert len(pair) == len(anchors) and np.allclose(pair[model], pair[f'h{h}'], atol=1e-3, rtol=0)
        for j,h in enumerate([6,12]):
            actual = frame[f'target_h{h}_mm'].to_numpy(); pred = output_values[:,j]; error = pred-actual
            mask = np.abs(error)>200
            selected = frame.loc[mask, ['station_id', 'anchor_utc']].copy()
            selected['station'] = selected.station_id.map(names)
            selected['model'] = model; selected['horizon_hours'] = h
            selected['target_utc'] = selected.anchor_utc+pd.Timedelta(hours=h)
            selected['actual_mm'] = actual[mask]; selected['predicted_mm'] = pred[mask]
            selected['error_mm'] = error[mask]; selected['absolute_error_mm'] = np.abs(error[mask])
            all_errors.append(selected)
        print(model, 'complete', flush=True)
    errors = pd.concat(all_errors, ignore_index=True)
    counts = (errors.groupby(['model','station_id','station','horizon_hours']).size()
              .unstack('horizon_hours', fill_value=0).reindex(columns=[6,12],fill_value=0))
    index = pd.MultiIndex.from_product([list(provenance['models']), list(names)], names=['model','station_id'])
    full = counts.reset_index().drop(columns='station').set_index(['model','station_id']).reindex(index, fill_value=0).reset_index()
    full['station'] = full.station_id.map(names);full = full.rename(columns={6:'h6_errors',12:'h12_errors'})
    full['total_errors'] = full.h6_errors+full.h12_errors
    full.sort_values(['model','total_errors','station'], ascending=[True,False,True]).to_csv(output/'station_counts.csv', index=False)
    errors.sort_values(['model','station','anchor_utc','horizon_hours']).to_csv(output/'individual_errors.csv', index=False)
    meta = dict(source_comparison=str(reference.resolve()), anchors_per_station=len(anchors),
        first_anchor_utc=str(anchors.min()), last_anchor_utc=str(anchors.max()),
        station_count=len(names), horizons=[6,12], strict_error_threshold_mm=200,
        dataset=provenance['dataset'], dataset_manifest_sha256=provenance['dataset_manifest_sha256'],
        models=provenance['models'], note='Previously explored test period; model checkpoints are frozen')
    (output/'provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('Saved',output,'totals:',full.groupby('model').total_errors.sum().to_dict())


if __name__ == '__main__':
    main()
