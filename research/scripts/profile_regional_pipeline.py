"""Bounded GPU profiling on disposable weights; never evaluates the test split."""
import argparse
import gc
import itertools
import json
from pathlib import Path
import time

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from src.ai.train import parser, validate_args, build_sequence_trainer
from src.ai.utils import set_seed


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--manifest', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--steps', type=int, default=40)
    options = cli.parse_args()
    options.output.mkdir(parents=True, exist_ok=False)
    job = json.loads(options.manifest.read_text())['jobs'][-1]
    arguments = job['args'][3:]
    args = parser.parse_args(arguments)
    args.checkpoint_dir = options.output / 'disposable'
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    result = dict(device=torch.cuda.get_device_name(), torch=torch.__version__,
                  architecture=args.train_method, steps=options.steps, timings={}, cases=[])

    def timed(name, operation):
        torch.cuda.synchronize()
        start = time.perf_counter()
        value = operation()
        torch.cuda.synchronize()
        result['timings'][name] = time.perf_counter() - start
        print(name, result['timings'][name], flush=True)
        return value

    set_seed(args.seed)
    trainer = timed('setup_seconds', lambda: build_sequence_trainer(args))
    timed('train_epoch_seconds', lambda: trainer.train_epoch(1))
    loss = timed('validation_seconds', lambda: trainer.evaluate_loss(trainer.eval_dataloader))
    trainer.best_val_loss, trainer.best_checkpoint_epoch = loss, 1
    dynamics = trainer.dynamics
    # Record the diagnostics sub-stages without altering their actual work.
    original = dynamics.evaluate_split
    def evaluate(*pos, **kw):
        return timed('diagnostics_' + pos[2] + '_seconds', lambda: original(*pos, **kw))
    dynamics.evaluate_split = evaluate
    probe = dynamics.gradient_probe
    dynamics.gradient_probe = lambda *a, **k: timed('gradient_probes_seconds', lambda: probe(*a, **k))
    timed('all_diagnostics_seconds', lambda: dynamics.run(trainer, 1))
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    for workers, cached in [(0, False), (2, False), (0, True)]:
        set_seed(args.seed)
        args.num_worker = workers
        trainer = build_sequence_trainer(args)
        dataset = trainer.train_dataloader.dataset
        # Identical ordered prefix in all cases. Caching is a diagnostic upper
        # bound, not a proposed change to sampling or model semantics.
        batches = list(itertools.islice(trainer.train_dataloader, options.steps)) if cached else None
        if cached:
            batches = [{k: v.to('cuda') for k, v in batch.items()} for batch in batches]
        trainer.model.train()
        torch.cuda.synchronize()
        start = time.perf_counter()
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        with profile(activities=activities) as prof:
            iterator = iter(batches if cached else trainer.train_dataloader)
            for _ in range(options.steps):
                with record_function('pipeline/data_wait'):
                    batch = next(iterator)
                with record_function('pipeline/prepare_transfer'):
                    inputs, y, scales, _ = trainer.prepare_batch(batch, dataset, need_mean=False)
                trainer.optimizer.zero_grad()
                with record_function('pipeline/forward_loss'):
                    p = trainer.predict_batch(trainer.model, inputs)
                    loss = trainer.data_loss(p, y, scales, inputs)
                with record_function('pipeline/backward'):
                    loss.backward()
                with record_function('pipeline/diagnostic_meter'):
                    meter = trainer.new_change_diagnostics(dataset)
                    trainer.update_change_diagnostics(meter, p.detach(), y, scales, inputs)
                with record_function('pipeline/optimizer'):
                    torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 1.)
                    trainer.optimizer.step()
                prof.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        name = f'workers{workers}_' + ('gpu_cached' if cached else 'host')
        prof.export_chrome_trace(str(options.output / f'{name}.json'))
        (options.output / f'{name}.txt').write_text(prof.key_averages().table(
            sort_by='self_cpu_time_total', row_limit=45))
        events = [{key: getattr(event, key, None) for key in
                   ['key', 'count', 'self_cpu_time_total', 'cpu_time_total',
                    'self_device_time_total', 'device_time_total']}
                  for event in prof.key_averages()]
        result['cases'].append(dict(name=name, wall_seconds=elapsed, events=events))
        (options.output / 'results.json').write_text(json.dumps(result, indent=2))
        print(name, elapsed, flush=True)
        del trainer, batches, iterator, prof, batch, inputs, p, loss, meter
        gc.collect()
        torch.cuda.empty_cache()
    print('PROFILE COMPLETE', flush=True)


if __name__ == '__main__':
    main()
