"""Two-epoch before/after performance check; disposable models, no test scoring."""
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import time
from unittest.mock import patch

import pandas as pd
import torch

import src.ai.train as training
from src.ai.change_dynamics import ChangeDynamics
from src.ai.regional_training_data import regional_loaders
from src.ai.utils import set_seed


def load_before(path, name):
    spec = importlib.util.spec_from_file_location('src.ai.' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--manifest', type=Path, required=True)
    cli.add_argument('--source-before', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    old_data = load_before(args.source_before / 'regional_training_data.py', '_old_regional_data')
    old_dynamics = load_before(args.source_before / 'change_dynamics.py', '_old_change_dynamics')
    job = json.loads(args.manifest.read_text())['jobs'][-1]
    records, states, predictions = [], {}, {}

    def build(label):
        options = training.parser.parse_args(job['args'][3:])
        options.checkpoint_dir = args.output / label
        set_seed(options.seed)
        with patch.object(training, 'regional_loaders', old_data.regional_loaders if label == 'before' else regional_loaders):
            trainer = training.build_sequence_trainer(options)
        if label == 'before':
            trainer.dynamics = old_dynamics.ChangeDynamics(every=1, gradients=True, probe_samples=32)
        return trainer

    # Initialize CUDA/cuDNN before either measured loop. Reset all RNGs on build.
    warm = build('warmup')
    batch = next(iter(warm.train_dataloader))
    inputs, y, scale, _ = warm.prepare_batch(batch, warm.train_dataloader.dataset)
    loss = warm.data_loss(warm.predict_batch(warm.model, inputs), y, scale, inputs)
    loss.backward()
    torch.cuda.synchronize()
    del warm, inputs, y, scale, loss, batch
    gc.collect()
    for label in ['before', 'after']:
        trainer = build(label)
        for epoch in [1, 2]:
            def timed(fn):
                torch.cuda.synchronize()
                start = time.perf_counter()
                value = fn()
                torch.cuda.synchronize()
                return value, time.perf_counter() - start
            train_loss, train_seconds = timed(lambda: trainer.train_epoch(epoch))
            val_loss, val_seconds = timed(lambda: trainer.evaluate_loss(trainer.eval_dataloader))
            trainer.best_val_loss, trainer.best_checkpoint_epoch = val_loss, epoch
            folder, diagnostic_seconds = timed(lambda: trainer.dynamics.run(trainer, epoch))
            row = dict(variant=label, epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                       train_seconds=train_seconds, val_seconds=val_seconds,
                       diagnostic_seconds=diagnostic_seconds,
                       total_seconds=train_seconds + val_seconds + diagnostic_seconds)
            if label == 'before':
                states[epoch] = {k: v.detach().cpu().clone() for k, v in trainer.model.state_dict().items()}
                predictions[epoch] = folder
            else:
                row['max_weight_difference'] = max((v.detach().cpu() - states[epoch][k]).abs().max().item()
                                                   for k, v in trainer.model.state_dict().items())
                assert row['max_weight_difference'] == 0, 'Training weights changed'
                for filename in ['train_eval_metrics.csv', 'val_eval_metrics.csv', 'validation_predictions.parquet']:
                    read = pd.read_parquet if filename.endswith('parquet') else pd.read_csv
                    pd.testing.assert_frame_equal(read(predictions[epoch] / filename), read(folder / filename),
                                                  check_exact=True)
                reference = next(r for r in records if r['variant'] == 'before' and r['epoch'] == epoch)
                assert train_loss == reference['train_loss'] and val_loss == reference['val_loss']
                row['exact_diagnostic_parity'] = True
            records.append(row)
            (args.output / 'results.json').write_text(json.dumps(records, indent=2))
            print(json.dumps(row), flush=True)
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
    print('PARITY BENCHMARK COMPLETE', flush=True)


if __name__ == '__main__':
    main()
