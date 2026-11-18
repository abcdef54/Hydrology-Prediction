"""Export a run's multi-horizon forecasts (mean of chosen epochs) for every issue time of a dataset, as a wide table.

Usage: PYTHONPATH=. .venv/bin/python research/scripts/export_self_forecasts.py RUN_DIR DATASET_DIR OUT.parquet [--epochs 91-100]
OUT.parquet: timestamp_utc (issue time) + p<h> = forecast level (mm) for issue + h hours, h = the run's target horizons.
Used as the `self_fc` group (src/ai/station_dataset/features.self_forecast_features). DATASET_DIR should cover the
whole grid you want (e.g. an evaluation grid built with --normalizers-from); issue times without a valid sample are absent.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from research.scripts.member_ensemble import member_predictions

ap = argparse.ArgumentParser()
ap.add_argument('run_dir'); ap.add_argument('dataset'); ap.add_argument('out'); ap.add_argument('--epochs', default='91-100')
a = ap.parse_args()
lo, hi = map(int, a.epochs.split('-'))
dirs = []
for e in range(lo, hi + 1):
    d = [p.parent for p in Path(a.run_dir).rglob(f'epoch_{e:03d}/model.pt')]
    assert len(d) == 1, (e, d)
    dirs.append(d[0])
horizons = joblib.load(dirs[0] / 'model.preprocessing.joblib')['regional']['target_horizons']
cols = {}
for h in horizons:
    preds = [member_predictions(d, a.dataset, h) for d in dirs]
    base = preds[0]
    cols[f'p{h}'] = pd.Series(np.mean([p.pred.to_numpy() for p in preds], axis=0), index=base.anchor)
out = pd.DataFrame(cols).sort_index()
out = out[~out.index.duplicated()].rename_axis('timestamp_utc').reset_index()
out.to_parquet(a.out, index=False)
print(a.out, out.shape, out.timestamp_utc.min(), out.timestamp_utc.max())
