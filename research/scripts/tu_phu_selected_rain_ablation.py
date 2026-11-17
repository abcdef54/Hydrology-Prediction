"""Retrain Tứ Phú with train-selected rain gauges and no weather forecast."""
from pathlib import Path
import hashlib
import json
import logging

import numpy as np
import pandas as pd
import xgboost as xgb

from research.station_analysis.phase2.feature_ablation.data import Source
from research.station_analysis.phase2.feature_ablation.training import Thermal, run_case, scalar_metrics, write_json


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    original = Path('data/processed/station_analysis/phase2/feature_ablation/comprehensive_v2')
    execution = json.loads((original / 'execution_manifest.json').read_text())
    config = json.loads(json.dumps(execution['config']))
    stamp = pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%SZ')
    work = original.parent / ('tu_phu_selected_rain_no_weather_' + stamp)
    work.mkdir()
    config['run_name'] = work.name
    config['reuse_previous_run'] = None
    source = Source(config, original / 'shared')
    sid = '0074647404'
    base, targets, current = source.base(sid)
    population = pd.read_parquet(original / 'stations' / sid / 'population.parquet')
    assert base.index.equals(pd.DatetimeIndex(pd.to_datetime(population.anchor_utc, utc=True)))
    for horizon in config['horizons']:
        np.testing.assert_array_equal(targets[f'h{horizon}'], population[f'dh{horizon}_mm'])
    discovery_path = original / 'stations' / sid / 'discovery.json'
    rain = json.loads(discovery_path.read_text())['selected_rain']
    features = source.features(base, [], rain, 'E1')
    assert not any('weather' in name for name in features.columns)
    folds = source.folds(base.index)
    write_json(work / 'execution_manifest.json', {
        'config': config, 'source_execution_manifest': str(original / 'execution_manifest.json'),
        'source_inputs': execution['inputs'], 'selected_rain': rain,
        'discovery_sha256': hashlib.sha256(discovery_path.read_bytes()).hexdigest(),
        'definition': 'E5 selected gauges, no forecast (selected-rain W0); fresh fits',
        'feature_columns': features.columns.tolist(), 'anchors': len(base),
    })
    ledger = json.loads((original / 'stations' / sid / 'ledger.json').read_text())
    metrics, predictions, fits = [], [], []
    thermal = Thermal(work, config)
    try:
        for seed in config['seeds']:
            meta = dict(station_id=sid, experiment='E1', grids=[], rain_ids=rain,
                        mode='normal', lag=6, version=None)
            key = run_case(work, config, thermal, features, targets, current, folds, meta, seed)
            fits.append(dict(seed=seed, fit_key=key))
            reference = [r for r in ledger if r['experiment'] == 'E5' and r['arm'] == 'W1'
                         and r['seed'] == seed and r['mode'] == 'normal' and r['lag'] == 6
                         and not r.get('version')]
            assert len(reference) == 1
            ref_folder = original / 'fits' / reference[0]['fit_key']
            ref_meta = json.loads((ref_folder / 'identity.json').read_text())['metadata']
            ref_features = source.features(base, ref_meta['grids'], ref_meta['rain_ids'],
                                           ref_meta['experiment'], ref_meta['mode'], ref_meta['lag'])
            for variant, folder, design in [('no_weather', work / 'fits' / key, features),
                                            ('E5-W1', ref_folder, ref_features)]:
                dm = xgb.DMatrix(design.to_numpy(dtype=np.float32), feature_names=design.columns.tolist(), nthread=4)
                for horizon in config['horizons']:
                    model = xgb.Booster()
                    model.load_model(folder / f'final_h{horizon}.ubj')
                    model.set_param({'device': 'cpu', 'nthread': 4})
                    predicted = current.to_numpy() + model.predict(dm)
                    actual = current.to_numpy() + targets[f'h{horizon}'].to_numpy()
                    for scope, mask in [('all', np.ones(len(base), dtype=bool)),
                                        ('pre_cutoff', base.index < pd.Timestamp(config['cutoff'])),
                                        ('post_cutoff', base.index >= pd.Timestamp(config['cutoff']))]:
                        metrics.append(dict(variant=variant, seed=seed, horizon=horizon, scope=scope,
                                            **scalar_metrics(actual[mask], predicted[mask])))
                    predictions.append(pd.DataFrame(dict(anchor_utc=base.index, variant=variant,
                        seed=seed, horizon=horizon, actual_mm=actual, prediction_mm=predicted)))
            logging.info('Finished seed %s', seed)
    finally:
        thermal.close()
    frame = pd.DataFrame(metrics)
    frame.to_csv(work / 'full_dataset_metrics.csv', index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(work / 'full_dataset_predictions.parquet', index=False)
    write_json(work / 'selected_fits.json', fits)
    summary = frame.groupby(['variant', 'scope', 'horizon']).agg(
        rmse_mean=('rmse_mm', 'mean'), rmse_sd=('rmse_mm', 'std'), mae_mean=('mae_mm', 'mean'),
        n200_mean=('n200', 'mean'), n=('n', 'first')).reset_index()
    summary.to_csv(work / 'summary.csv', index=False)
    print(summary.to_string(index=False), flush=True)
    print('OUTPUT', work.resolve(), flush=True)


if __name__ == '__main__':
    main()
