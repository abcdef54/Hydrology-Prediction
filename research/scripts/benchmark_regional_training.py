"""Compare archived and optimized full train+validation loops on disposable weights."""
import argparse
import gc
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

import torch

from src.ai.losses import HorizonWeightedMSE
from src.ai.regional_training_data import regional_loaders, regional_persistence
from src.ai.residual_sequence import residual_model
from src.ai.trainers import ResidualLSTMTrainer
from src.ai.utils import set_seed
from research.station_analysis.phase2.feature_ablation.training import Thermal


class CheckedLoader:
    def __init__(self, loader, thermal):
        self.loader, self.dataset, self.thermal = loader, loader.dataset, thermal
        self.paused = 0.
        self.order_hash = None

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        digest = hashlib.sha256()
        for batch in self.loader:
            start = time.perf_counter()
            self.thermal.check()
            waited = time.perf_counter()-start
            if waited > 1:
                self.paused += waited
            # Small, deterministic fingerprints: no device copies or full-window hashing.
            for key in ['station_ids', 'y']:
                digest.update(batch[key].numpy().tobytes())
            digest.update(batch['x'][:, -1, 0].numpy().tobytes())
            yield batch
        self.order_hash = digest.hexdigest()


def baseline_modules(snapshot, output):
    runtime = output/'baseline_runtime'/'performance_baseline'
    runtime.mkdir(parents=True)
    (runtime/'__init__.py').write_text('')
    for name in ['reader.py', 'regional_training_data.py', 'residual_sequence.py', 'trainers.py']:
        text = (snapshot/name).read_text()
        text = text.replace('from .regional_data import RegionalDataset', 'from .reader import RegionalDataset')
        text = text.replace('from src.ai.residual_sequence import ResidualSequence', 'from performance_baseline.residual_sequence import ResidualSequence')
        (runtime/name).write_text(text)
    sys.path.insert(0, str(runtime.parent.resolve()))
    return tuple(importlib.import_module('performance_baseline.'+name) for name in ['regional_training_data', 'residual_sequence', 'trainers'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dataset-dir', type=Path, default=Path('train_data/hue_regional_20261015_h6_h12_fixed_split_v1'))
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--workers', type=int, nargs='+', default=[0, 2])
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Use a new benchmark output directory')
    args.output_dir.mkdir(parents=True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for the end-to-end GPU benchmark')
    old_data, old_model, old_trainer = baseline_modules(args.baseline_dir, args.output_dir)
    thermal = Thermal(args.output_dir, dict(xgboost={'device': 'cuda'}, thermal_pause_c=65, thermal_resume_c=59))
    metadata = dict(torch=torch.__version__, device=torch.cuda.get_device_name(), epochs=args.epochs,
        batch_size=256, hidden_size=64, layers=3, dropout=.3, seed=42,
        baseline_hashes=json.loads((args.baseline_dir/'sha256.json').read_text()),
        optimized_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            Path('src/ai/regional_training_data.py'), Path('src/ai/residual_sequence.py'), Path('src/ai/trainers.py')]})
    records, states = [], {}
    try:
        for workers in args.workers:
            for variant in ['baseline', 'optimized']:
                set_seed(42)
                loaders = (old_data.regional_loaders if variant == 'baseline' else regional_loaders)(
                    args.dataset_dir, batch_size=256, num_workers=workers, seed=42,
                    rain_stations='nham+hong_thai', time_features='none', use_weather_forecast=True)
                train, val, test = loaders
                dataset = train.dataset
                weather = [i for i, name in enumerate(dataset.feature_cols) if name.startswith('weather_forecast_')]
                masks = [i for i in weather if dataset.feature_cols[i].endswith('_missing')]
                scale, offset, count = regional_persistence(dataset)
                factory = old_model.residual_model if variant == 'baseline' else residual_model
                model = factory('lstm', input_size=dataset.num_features, output_size=2, hidden_size=64,
                    num_layers=3, dropout=.3, input_dropout=.2, head_dropout=.2, num_stations=count,
                    station_embedding_size=8, persistence_scale=scale, persistence_offset=offset,
                    weather_feature_indices=weather, weather_mask_indices=masks, weather_mask_values=[1.]*len(masks))
                optimizer = torch.optim.AdamW(model.parameters(), lr=.0009, weight_decay=.001)
                cls = old_trainer.ResidualLSTMTrainer if variant == 'baseline' else ResidualLSTMTrainer
                trainer = cls(model=model, horizon='1h', num_epochs=args.epochs,
                    train_dataloader=CheckedLoader(train, thermal), eval_dataloader=CheckedLoader(val, thermal),
                    test_dataloader=test, loss_func=HorizonWeightedMSE([1, 1]), optimizer=optimizer,
                    scheduler=None, early_stopping=30, correction_penalty=0., device='cuda',
                    checkpoint_dir=args.output_dir/variant, checkpoint_metric='val_loss')
                print(f'START {variant} workers={workers}', flush=True)
                for epoch in range(1, args.epochs+1):
                    torch.cuda.synchronize()
                    paused = trainer.train_dataloader.paused+trainer.eval_dataloader.paused
                    start = time.perf_counter()
                    train_loss = trainer.train_epoch(epoch)
                    torch.cuda.synchronize()
                    after_train = time.perf_counter()
                    val_loss = trainer.evaluate_loss(trainer.eval_dataloader)
                    torch.cuda.synchronize()
                    end = time.perf_counter()
                    cooldown = trainer.train_dataloader.paused+trainer.eval_dataloader.paused-paused
                    row = dict(variant=variant, workers=workers, epoch=epoch, train_seconds=after_train-start,
                        validation_seconds=end-after_train, wall_seconds=end-start, cooldown_seconds=cooldown,
                        active_seconds=end-start-cooldown, train_loss=train_loss, val_loss=val_loss,
                        train_order_sha256=trainer.train_dataloader.order_hash,
                        val_order_sha256=trainer.eval_dataloader.order_hash,
                        validation_metrics=trainer.last_validation_metrics)
                    records.append(row)
                    (args.output_dir/'results.json').write_text(json.dumps(dict(metadata=metadata, epochs=records), indent=2))
                    print(json.dumps(row), flush=True)
                final = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if variant == 'baseline':
                    states[workers] = final
                else:
                    difference = max((final[k]-states[workers][k]).abs().max().item() for k in final)
                    old = [r for r in records if r['variant'] == 'baseline' and r['workers'] == workers]
                    new = [r for r in records if r['variant'] == 'optimized' and r['workers'] == workers]
                    assert all(a['train_order_sha256'] == b['train_order_sha256'] for a, b in zip(old, new)), 'Shuffle order changed'
                    assert all(abs(a['val_loss']-b['val_loss']) < 1e-6 for a, b in zip(old, new)), 'Validation loss changed'
                    assert difference < 1e-6, f'Weights changed: {difference}'
                    metadata[f'workers{workers}_max_weight_difference'] = difference
                    print(f'PARITY workers={workers} max_weight_difference={difference}', flush=True)
                del trainer, model, optimizer, loaders, train, val, test, dataset
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        thermal.close()
    (args.output_dir/'results.json').write_text(json.dumps(dict(metadata=metadata, epochs=records), indent=2))
    print('BENCHMARK COMPLETE', flush=True)


if __name__ == '__main__':
    main()
