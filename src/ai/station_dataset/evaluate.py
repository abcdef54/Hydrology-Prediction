"""Score station checkpoints in mm and combine models (equal-weight ensemble).

    # every epoch snapshot of one run (train.py --diagnose-change-dynamics) on validation and exploratory test
    .venv/bin/python -m src.ai.station_dataset.evaluate epochs --dataset train_data/thuan_an/ta_water_only_wind_tide30_fc \
        --run-dir model/experiments/thuan_an/ta_water_only_wind_tide30_fc/<config> --out reports/<config>_epochs.csv

    # one or more model folders (each with model.pt + model.preprocessing.joblib) and their mean
    .venv/bin/python -m src.ai.station_dataset.evaluate ensemble \
        --member train_data/thuan_an/ta_water_only_wind_tide30_fc models/thuan_an/wind_tide30_fc_gru32x3_seq12_cosine_ep77 \
        --member train_data/thuan_an/ta_water_only_wind models/thuan_an/wind_gru32x2_seq25_ep33

Metrics per horizon: n200 = number of |prediction - observation| > 200 mm, MAE and max absolute error (mm).
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.ai.regional_training_data import regional_loaders
from src.ai.sequence_checkpoint import SequenceRuntime

SPLITS = {'validation': 1, 'test_exploratory': 2}


def predict(model_pt, dataset_dir, split='validation'):
    """Predictions (mm) of one checkpoint on one split ('validation', 'test_exploratory', 'purged' or 'all'); the
    selection (groups, window, horizons) is read from the model's preprocessing bundle, the data from `dataset_dir`
    (which must hold the same columns)."""
    model_pt = Path(model_pt)
    saved = joblib.load(model_pt.with_suffix('.preprocessing.joblib'))['regional']
    horizons = saved['target_horizons']
    loaders = regional_loaders(str(dataset_dir), batch_size=512, num_workers=0, seed=0,
                               rain_stations='+'.join(saved['rain_stations']) or 'none', use_weather_forecast=saved['use_weather_forecast'],
                               time_features=saved['time_features'], target_horizons=horizons,
                               sequence_length=saved['training_selection']['sequence_length'])
    if split in SPLITS:
        loader = loaders[SPLITS[split]]
    else:                                   # 'purged' (anchors between the splits) or 'all'
        from torch.utils.data import DataLoader
        from src.ai.regional_training_data import RegionalTrainingDataset, regional_collate
        view = loaders[0].dataset.view
        dataset = RegionalTrainingDataset(str(dataset_dir), split=split, target_normalizers=view.normalizers,
                                          rain_stations='+'.join(saved['rain_stations']) or 'none', use_weather_forecast=saved['use_weather_forecast'],
                                          time_features=saved['time_features'], target_horizons=horizons,
                                          sequence_length=saved['training_selection']['sequence_length'])
        loader = DataLoader(dataset, batch_size=512, collate_fn=regional_collate)
    runtime = SequenceRuntime(model_pt)
    pred = np.concatenate([runtime.predict_mm(b['x'], b['station_ids']) for b in loader])
    frame = loader.dataset.view.samples
    rows = [pd.DataFrame(dict(anchor_utc=frame.anchor_utc.to_numpy(), horizon_hours=h, predicted_mm=pred[:, j],
                              actual_mm=frame[f'target_h{h}_mm'].to_numpy(), current_mm=frame.water_now_mm.to_numpy()))
            for j, h in enumerate(horizons)]
    return pd.concat(rows, ignore_index=True)


def metrics(df):
    out = {}
    err = (df.predicted_mm - df.actual_mm).abs()
    for h, e in list(err.groupby(df.horizon_hours)) + [('all', err)]:
        out[f'h{h}' if h != 'all' else 'all'] = dict(n200=int((e > 200).sum()), mae=round(float(e.mean()), 2), max=round(float(e.max()), 1))
    return out


def score_epochs(dataset_dir, run_dir, out=None):
    """Validation/test metrics for every epoch_XXX/model.pt of a run (snapshots from --diagnose-change-dynamics)."""
    epochs = sorted(Path(run_dir).rglob('epoch_*/model.pt'))
    epochs = [p for p in epochs if p.parent.name != 'epoch_000']
    rows = []
    for p in epochs:
        r = {'epoch': int(p.parent.name.split('_')[1])}
        for split in SPLITS:
            for k, v in metrics(predict(p, dataset_dir, split)).items():
                r.update({f'{split}_{k}_{m}': x for m, x in v.items()})
        rows.append(r)
        print(r['epoch'], 'val n200', r['validation_all_n200'], 'test n200', r['test_exploratory_all_n200'],
              'val MAE', r['validation_all_mae'], 'test MAE', r['test_exploratory_all_mae'], flush=True)
    table = pd.DataFrame(rows).sort_values('epoch')
    if out:
        table.to_csv(out, index=False)
    zero = table[(table.validation_all_n200 == 0)]
    print('epochs with 0 validation errors:', zero.epoch.tolist())
    print('... and also 0 test errors:', zero[zero.test_exploratory_all_n200 == 0].epoch.tolist())
    return table


def ensemble(members, splits=tuple(SPLITS)):
    """members: [(dataset_dir, model_dir_or_pt)]; equal-weight mean of the predicted water level (mm)."""
    for split in splits:
        preds = []
        for dataset_dir, model in members:
            pt = Path(model) / 'model.pt' if Path(model).is_dir() else Path(model)
            p = predict(pt, dataset_dir, split).set_index(['anchor_utc', 'horizon_hours']).sort_index()
            preds.append(p)
            print(split, Path(model).name, metrics(p.reset_index()))
        base = preds[0]
        if any(not p.index.equals(base.index) for p in preds):
            raise ValueError('members cover different samples; build the datasets from the same config')
        mean = base.assign(predicted_mm=sum(p.predicted_mm for p in preds) / len(preds))
        print(split, 'ENSEMBLE', metrics(mean.reset_index()))


def combined(members, start, end, exclude_splits=('excluded_sensor_step',), out=None):
    """Per-horizon combination over a time window of issue times.

    members: [(dataset_dir, model_dir_or_pt, horizons)]; each horizon is taken from exactly one member (no averaging).
    start/end: issue-time bounds (inclusive), any timezone-aware ISO string, e.g. '2026-09-01T00:00:00+07:00'.
    All anchors of the datasets are used (train/validation/purged/test), except splits in `exclude_splits`.
    Prints MAE, RMSE, bias, max |error|, count > 200 mm and the persistence MAE per horizon.
    """
    a0, a1 = pd.Timestamp(start).tz_convert('UTC'), pd.Timestamp(end).tz_convert('UTC')
    parts = []
    for dataset_dir, model, horizons in members:
        pt = Path(model) / 'model.pt' if Path(model).is_dir() else Path(model)
        d = predict(pt, dataset_dir, 'all')
        d['anchor_utc'] = pd.to_datetime(d.anchor_utc, utc=True)
        s = pd.read_parquet(Path(dataset_dir) / 'samples.parquet')[['anchor_utc', 'split']]
        s['anchor_utc'] = pd.to_datetime(s.anchor_utc, utc=True)
        d = d.merge(s, on='anchor_utc', how='left')
        d = d[(d.anchor_utc >= a0) & (d.anchor_utc <= a1) & d.horizon_hours.isin(horizons) & ~d.split.isin(exclude_splits)]
        d['member'] = Path(model).name
        parts.append(d)
    x = pd.concat(parts, ignore_index=True)
    if x.duplicated(['anchor_utc', 'horizon_hours']).any():
        raise ValueError('a horizon is assigned to more than one member')
    x['error_mm'] = x.predicted_mm - x.actual_mm
    rows = []
    for h, g in x.groupby('horizon_hours'):
        e = g.error_mm
        rows.append(dict(horizon_hours=h, member=g.member.iloc[0], issue_times=g.anchor_utc.nunique(), mae=round(e.abs().mean(), 1),
                         rmse=round(float(np.sqrt((e ** 2).mean())), 1), bias=round(e.mean(), 1), max_abs=round(e.abs().max(), 1),
                         n200=int((e.abs() > 200).sum()), persistence_mae=round((g.current_mm - g.actual_mm).abs().mean(), 1)))
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    print('all horizons MAE', round(x.error_mm.abs().mean(), 1))
    if out:
        x.to_parquet(out)
    return table


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('epochs'); e.add_argument('--dataset', required=True); e.add_argument('--run-dir', required=True); e.add_argument('--out')
    m = sub.add_parser('ensemble'); m.add_argument('--member', nargs=2, action='append', required=True, metavar=('DATASET', 'MODEL'))
    c = sub.add_parser('combined', help='per-horizon combination over a window of issue times')
    c.add_argument('--member', nargs=3, action='append', required=True, metavar=('DATASET', 'MODEL', 'HORIZONS'),
                   help='HORIZONS comma-separated, e.g. 3,6')
    c.add_argument('--start', required=True); c.add_argument('--end', required=True)
    c.add_argument('--keep-sensor-step', action='store_true', help='also score the excluded_sensor_step anchors')
    c.add_argument('--out', help='parquet with every prediction')
    a = p.parse_args()
    torch.set_num_threads(2)
    if a.cmd == 'epochs':
        score_epochs(a.dataset, a.run_dir, a.out)
    elif a.cmd == 'combined':
        combined([(d, m_, [int(h) for h in hs.split(',')]) for d, m_, hs in a.member], a.start, a.end,
                 exclude_splits=() if a.keep_sensor_step else ('excluded_sensor_step',), out=a.out)
    else:
        ensemble([tuple(x) for x in a.member])


if __name__ == '__main__':
    main()
