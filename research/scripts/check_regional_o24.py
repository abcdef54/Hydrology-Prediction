"""Audit an O24 view and untrained model interfaces; never train or write artifacts."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.ai.regional_training_data import RegionalTrainingDataset, regional_loaders, regional_persistence
from src.ai.residual_sequence import residual_model
from src.ai.change_objective import from_training_dataset

DATA = Path('train_data/hue_regional_20261015_h6_h12_fixed_split_v1')
HORIZONS = list(range(1, 25))
MODELS = [('R2', 'gru_high_precision', 'gru-late-fusion', '750007+750062', 'none', 64, 'station', 17),
          ('R1', 'transformer_primary', 'transformer-late-fusion', '750062+750041+750014', 'all', 32, 'station-mlp', 42)]

def main():
    torch.set_num_threads(4)
    saved_samples = pd.read_parquet(DATA/'samples.parquet').set_index('sample_id')
    config = json.loads((DATA/'config.json').read_text())
    reports = []
    for name, folder, arch, rain, time, hidden, head, seed in MODELS:
        loaders = regional_loaders(DATA, batch_size=8, num_workers=0, seed=seed, fold='fixed',
                                   rain_stations=rain, time_features=time, sequence_length=96, target_horizons=HORIZONS)
        train = loaders[0].dataset
        previous = RegionalTrainingDataset(DATA, split='train', fold='fixed', rain_stations=rain,
                                           time_features=time, sequence_length=96, target_horizons=[6, 12])
        assert train.feature_cols == previous.feature_cols
        assert train.num_features == (42 if name == 'R2' else 59)
        assert train.seq_len == previous.seq_len == 96
        old_rows = {sid: i for i, sid in enumerate(previous.view.samples.sample_id)}
        for i in (0, len(train)//2, len(train)-1):
            old_index = old_rows[train.view.samples.iloc[i].sample_id]
            torch.testing.assert_close(train[i]['x'], previous[old_index]['x'], rtol=0, atol=0)
        assert train.view.normalizers['shared'] == previous.view.normalizers['shared']
        for sid, stats in train.view.normalizers['stations'].items():
            old = previous.view.normalizers['stations'][sid]
            assert stats['water'] == old['water']
            assert all(stats['targets'][f'h{h}'] == old['targets'][f'h{h}'] for h in (6, 12))
            part = train.view.samples[train.view.samples.station_id == sid]
            for h in set(HORIZONS)-{6, 12}:
                values = part[f'target_h{h}_mm'].to_numpy()
                stat = stats['targets'][f'h{h}']
                assert stat['count'] == len(values) and np.isclose(stat['mean'], values.mean())
                assert np.isclose(stat['scale'], values.std()) if values.std() > 0 else stat['scale'] == 1
        totals = {}
        for loader in loaders:
            d = loader.dataset; frame = d.view.samples; old = saved_samples.loc[frame.sample_id]
            assert np.array_equal(old.station_id.to_numpy(), frame.station_id.to_numpy())
            assert np.array_equal(old.anchor_utc.to_numpy(), frame.anchor_utc.to_numpy())
            flag = 'fixed_'+d.split
            if d.split in ('train', 'validation'):
                assert old[flag].all()
                boundary = pd.Timestamp(config['folds'][0]['validation_start' if d.split == 'train' else 'validation_end'])
                assert (frame.anchor_utc + pd.Timedelta(hours=24) < boundary).all()
            else:
                assert old.split.eq('test_exploratory').all()
            assert d.view.normalizers == train.view.normalizers
            for sid, part in frame.groupby('station_id'):
                for h in HORIZONS:
                    expected = d.view.water[sid].reindex(pd.DatetimeIndex(part.anchor_utc) + pd.Timedelta(hours=h)).to_numpy()
                    assert np.isfinite(expected).all() and np.allclose(expected, part[f'target_h{h}_mm'], rtol=0, atol=1e-9)
            totals[d.split] = {'anchors': len(d), 'start_utc': str(frame.anchor_utc.min()), 'end_utc': str(frame.anchor_utc.max()),
                               'excluded_from_saved_pool': d.regional_metadata['training_selection']['dropped_samples']}
        scale, offset, stations = regional_persistence(train)
        weather = [i for i,c in enumerate(train.feature_cols) if c.startswith('weather_forecast_')]
        masks = [i for i in weather if train.feature_cols[i].endswith('_missing')]
        model = residual_model(arch, input_size=train.num_features, output_size=24, hidden_size=hidden, num_layers=3,
            dropout=.3, input_dropout=.2, head_dropout=.2, num_stations=stations, station_embedding_size=8,
            forecast_hidden_size=32, head_mode=head, station_mlp_hidden_size=32, transformer_heads=4,
            transformer_ff_size=128, persistence_scale=scale, persistence_offset=offset,
            weather_feature_indices=weather, weather_mask_indices=masks, weather_mask_values=[1.]*len(masks),
            forecast_history_policy='anchor_only')
        model.eval(); batch = next(iter(loaders[1]))
        with torch.no_grad():
            pred = model(batch['x'], batch['station_ids'])
            persistence = model.persistence(batch['x'], batch['station_ids'])
            loss = from_training_dataset(train, weights=[1.]*24, mode='station-balanced', event_weight=0., under_weight=0.)
            assert pred.shape == batch['y'].shape == (len(batch['x']), 24)
            assert torch.isfinite(loss(pred, batch['y'], persistence, batch['station_ids']))
        reports.append({'model': name, 'input_shape': list(batch['x'].shape), 'output_shape': list(pred.shape),
                        'selected_epochs_weights_used': False, 'split_coverage': totals})
        print(json.dumps(reports[-1], ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
