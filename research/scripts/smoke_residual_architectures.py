"""Bounded real-data training/inference smoke matrix, never a performance study."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import json
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.ai.residual_sequence import NEW_METHODS
from src.ai.sequence_checkpoint import SequenceRuntime
from src.ai.train import parser as train_parser, validate_args, build_sequence_trainer
from src.ai.utils import set_seed


def cap_dataset(dataset, maximum):
    if hasattr(dataset, 'view'):
        # Exercise all station embeddings rather than only the first station.
        frame = dataset.view.samples
        each = max(1, maximum//frame.station_id.nunique())
        positions = frame.groupby('station_id', sort=False).head(each).index
        dataset.view.samples = frame.loc[positions].reset_index(drop=True)
        dataset.timestamps = dataset.view.samples.anchor_utc
    else:
        positions = np.linspace(0, len(dataset)-1, min(maximum, len(dataset)), dtype=int)
        dataset.end_indices = dataset.end_indices[positions]
        if dataset.timestamps is not None:
            dataset.timestamps = dataset.timestamps.iloc[positions].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-dataset', type=Path, default=Path('train_data/da_vien_1h_2026-10-15_forecast_mask_o2_h6_h12'))
    parser.add_argument('--regional-dataset', type=Path, default=Path('train_data/hue_regional_20261021_h6_h12_v3'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--samples-per-split', type=int, default=22)
    parser.add_argument('--station-specific-only', action='store_true', help='Exercise one target station from the regional pool')
    parser.add_argument('--include-station-specific', action='store_true', help='Also exercise one target station from the regional pool')
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Choose a new smoke output directory')
    args.output_dir.mkdir(parents=True)
    torch.set_num_threads(2)
    results = []
    modes = [(True, True)] if args.station_specific_only else [(False, False), (True, False)]
    if args.include_station_specific and not args.station_specific_only:
        modes.append((True, True))
    for regional, single_station in modes:
        for weather in [False, True]:
            for method in NEW_METHODS:
                set_seed(42)
                name = f'{"regional_single_station" if single_station else "regional" if regional else "local"}_{method}_weather{int(weather)}'
                folder = args.output_dir/name
                folder.mkdir()
                argv = ['--train-method', method, '--dataset-dir', str(args.regional_dataset if regional else args.local_dataset),
                    '--horizon', '1h', '--mlflow-ex-name', 'architecture_smoke', '--device', 'cpu',
                    '--num-epochs', '1', '--early-stopping', '2', '--batch-size', '8', '--num-worker', '0',
                    '--hidden-size', '8', '--num-layers', '1', '--dropout', '0', '--lstm-seq-len', '96',
                    '--tcn-channels', '8', '8', '8', '8', '8', '--cnn-channels', '8', '8',
                    '--forecast-hidden-size', '8', '--station-embedding-size', '4',
                    '--lstm-tail-penalty', '.5', '--lstm-correction-penalty', '.1',
                    '--lstm-checkpoint-metric', 'large_error_rate', '--target-horizons', '6', '12',
                    '--use-weather-forecast', str(weather).lower(), '--checkpoint-dir', str(folder)]
                if regional:
                    argv += ['--time-features', 'all']
                if single_station:
                    argv += ['--water-station', '0074647404']
                options = train_parser.parse_args(argv)
                validate_args(options)
                with (folder/'execution.log').open('w') as log, redirect_stdout(log), redirect_stderr(log):
                    trainer = build_sequence_trainer(options)
                    for attr in ['train_dataloader', 'eval_dataloader', 'test_dataloader']:
                        old = getattr(trainer, attr)
                        cap_dataset(old.dataset, args.samples_per_split)
                        setattr(trainer, attr, DataLoader(old.dataset, batch_size=8,
                            shuffle=(attr == 'train_dataloader'), generator=torch.Generator().manual_seed(42)))
                    with patch.multiple('src.ai.trainers.mlflow', log_metrics=Mock(), log_metric=Mock(),
                                        log_table=Mock(), set_tag=Mock()):
                        trainer.train()
                    runtime = SequenceRuntime(trainer.checkpoint_path)
                    batch = next(iter(trainer.test_dataloader))
                    inputs, _, scale, mean = trainer.prepare_batch(batch, trainer.test_dataloader.dataset)
                    with torch.inference_mode():
                        prediction = trainer.predict_batch(trainer.model, inputs)
                    x = inputs['x'] if isinstance(inputs, dict) else inputs
                    ids = inputs.get('station_ids') if isinstance(inputs, dict) else None
                    restored = runtime.predict_mm(x, ids)
                    expected = (prediction.double()*scale.double()+mean.double()).numpy()
                    np.testing.assert_allclose(restored, expected, atol=.005, rtol=1e-6)
                row = dict(case=name, architecture=method, regional=regional, weather=weather,
                    status='PASS', checkpoint=str(trainer.checkpoint_path.resolve()),
                    input_size=trainer.model.input_size, parameters=sum(p.numel() for p in trainer.model.parameters()),
                    samples={attr: len(getattr(trainer, attr).dataset) for attr in ['train_dataloader', 'eval_dataloader', 'test_dataloader']},
                    selected_epoch=trainer.best_checkpoint_epoch, validation_metrics=trainer.best_validation_metrics)
                results.append(row)
                (args.output_dir/'results.json').write_text(json.dumps(results, indent=2))
                print(name, 'PASS', flush=True)
    print(f'{len(results)} real-data smoke cases passed; no full training or tuning.', flush=True)


if __name__ == '__main__':
    main()
