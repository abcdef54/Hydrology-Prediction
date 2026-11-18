"""Per-horizon equal-weight ensembles of packaged station checkpoints (e.g. models/da_vien/v1/package.json).

Usage: PYTHONPATH=. .venv/bin/python research/scripts/member_ensemble.py PACKAGE.json DATASET_DIR [OUT.parquet]
PACKAGE.json: {"horizons": {"<h>": {"members": [dir, ...]}}} with paths relative to the package folder; each member dir
holds model.pt + model.preprocessing.joblib. For horizon h the prediction is the mean (mm) of the members' +h outputs.
DATASET_DIR must hold the members' input columns (same config; an evaluation grid built with --normalizers-from works).
Splits: train / validation / purged / test (<= 2026-10-14 19:00 UTC) / new (later issue times).
Prints n200 (|error| > 200 mm), MAE and max per split and horizon; writes per-issue-time predictions to OUT.parquet.
"""
import json, sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.ai.regional_training_data import RegionalTrainingDataset, regional_collate
from src.ai.sequence_checkpoint import SequenceRuntime

NEW_AFTER = pd.Timestamp('2026-10-14 19:00', tz='UTC')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def member_predictions(member, dataset_dir, h):
    pt = Path(member) / 'model.pt'
    saved = joblib.load(pt.with_suffix('.preprocessing.joblib'))['regional']
    kw = dict(rain_stations='+'.join(saved['rain_stations']) or 'none', use_weather_forecast=saved['use_weather_forecast'],
              time_features=saved['time_features'], target_horizons=saved['target_horizons'],
              sequence_length=saved['training_selection']['sequence_length'])
    j = saved['target_horizons'].index(h)
    rt, tn, frames = SequenceRuntime(pt, device=DEV), None, []
    for split in ('train', 'validation', 'purged', 'test_exploratory'):
        ds = RegionalTrainingDataset(dataset_dir, split=split, target_normalizers=tn, **kw)
        tn = tn or ds.view.normalizers
        fr = ds.view.samples
        pred = np.concatenate([rt.predict_mm(b['x'], b['station_ids'])[:, j]
                               for b in DataLoader(ds, batch_size=1024, collate_fn=regional_collate)])
        anchor = pd.to_datetime(fr.anchor_utc, utc=True)
        name = np.where(split != 'test_exploratory', split, np.where(anchor > NEW_AFTER, 'new', 'test'))
        frames.append(pd.DataFrame(dict(anchor=anchor, split=name, now=fr.water_now_mm.to_numpy(), y=fr[f'target_h{h}_mm'].to_numpy(), pred=pred)))
    return pd.concat(frames, ignore_index=True)


def run(package, dataset_dir):
    package = Path(package)
    spec = json.loads(package.read_text())
    out = []
    for h, hs in spec['horizons'].items():
        h = int(h)
        preds = [member_predictions(package.parent / m, dataset_dir, h) for m in hs['members']]
        base = preds[0].copy()
        for p in preds[1:]:
            if not (p.anchor.equals(base.anchor) and np.allclose(p.y, base.y)):
                raise ValueError('members cover different samples')
        base['pred'] = np.mean([p.pred.to_numpy() for p in preds], axis=0)
        base['h'] = h
        out.append(base)
    df = pd.concat(out, ignore_index=True)
    df['err'] = df.pred - df.y
    return df


if __name__ == '__main__':
    df = run(sys.argv[1], sys.argv[2])
    order = ['train', 'validation', 'purged', 'test', 'new']
    for h, g in df.groupby('h'):
        for s in order:
            q = g[g.split == s].err.abs()
            if len(q):
                print(f'+{h} h {s:10s} n={len(q):5d} n200={int((q > 200).sum()):3d} mae={q.mean():6.1f} max={q.max():6.1f}')
    if len(sys.argv) > 3:
        df.to_parquet(sys.argv[3])
