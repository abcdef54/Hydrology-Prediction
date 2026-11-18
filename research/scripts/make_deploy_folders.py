"""Assemble deploy/<station>/ : model bundle (plain dict/numpy, format v2), training datasets, Open-Meteo data,
predict.py, worker.py. README.md files are written separately.

Usage: PYTHONPATH=. .venv/bin/python research/scripts/make_deploy_folders.py
"""
import hashlib, json, shutil
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import torch

OM = Path('data/external/open_meteo')
SOURCES = {'wind': 'wind_previous_runs.json', 'ecmwf': 'ecmwf_single_runs', 'marine': 'marine_sea_level.json'}


def member(model_dir, feature_config, dataset_name):
    model_dir = Path(model_dir)
    saved = joblib.load(model_dir / 'model.preprocessing.joblib')
    raw = (model_dir / 'model.pt').read_bytes()
    state = torch.load(BytesIO(raw), map_location='cpu', weights_only=True)
    reg = saved['regional']; spec = saved['model_spec']['options']
    assert spec['architecture'] == 'gru' and spec.get('head_mode') == 'station' and not spec['weather_feature_indices']
    sid = reg['selected_station'] or next(iter(reg['station_mapping']))
    norm = reg['normalizers']
    cfg = {k: v for k, v in feature_config.items()}
    return dict(
        architecture=dict(input_size=spec['input_size'], hidden_size=spec['hidden_size'], num_layers=spec['num_layers'],
                          output_size=spec['output_size'], num_stations=spec['num_stations'],
                          station_embedding_size=spec['station_embedding_size']),
        state_dict={k: v.numpy().copy() for k, v in state.items()},
        feature_columns=list(saved['feature_columns']), sequence_length=reg['training_selection']['sequence_length'],
        horizons=list(reg['target_horizons']), groups=list(reg['rain_stations']),
        station_index=reg['station_mapping'][sid]['station_index'],
        normalizers=dict(water=norm['stations'][sid]['water'],
                         targets={k: v for k, v in norm['stations'][sid]['targets'].items()},
                         shared=norm['shared']),
        train_anchor_first=norm['train_anchor_first'], train_anchor_last=norm['train_anchor_last'],
        feature_config=cfg, training_dataset=dataset_name, source_model_dir=str(model_dir),
        model_pt_sha256=hashlib.sha256(raw).hexdigest())


def copy_open_meteo(dst, marine):
    d = dst / 'data' / 'open_meteo'; d.mkdir(parents=True, exist_ok=True)
    shutil.copy(OM / 'prev_offshore_to_20261116.json', d / SOURCES['wind'])
    if (d / SOURCES['ecmwf']).exists():
        shutil.rmtree(d / SOURCES['ecmwf'])
    shutil.copytree(OM / 'ecmwf_single_runs_soil', d / SOURCES['ecmwf'])
    if marine:
        shutil.copy(OM / 'marine_offshore_ta_to_20261116.json', d / SOURCES['marine'])


def assemble(station_dir, name, station_id, station_name, members, combination, datasets, marine):
    dst = Path('deploy') / station_dir
    (dst / 'model').mkdir(parents=True, exist_ok=True)
    bundle = dict(format='hydrology_station_bundle_v2', name=name, station_id=station_id, station_name=station_name,
                  combination=combination, members=members,
                  sources={k: v for k, v in SOURCES.items() if marine or k != 'marine'})
    joblib.dump(bundle, dst / 'model' / f'{name}.joblib', compress=3)
    for ds in datasets:
        tgt = dst / 'data' / 'train_dataset' / Path(ds).name
        if tgt.exists():
            shutil.rmtree(tgt)
        shutil.copytree(ds, tgt)
    copy_open_meteo(dst, marine)
    for f in ('predict.py', 'worker.py'):
        shutil.copy(Path('src/ai/station_dataset/deploy_templates') / f, dst / f)
    print(dst, sorted(p.name for p in dst.iterdir()))


ta_cfg = json.loads(Path('models/thuan_an/dataset_config.json').read_text())
ta_cfg['wind'] = dict(ta_cfg['wind'], request_point=[16.85, 107.85])
assemble('thuan_an', 'thuan_an_ensemble_34k', '0074647402', 'Cảng Cá Thuận An',
         {'wind_tide30_fc_gru32x3_seq12': member('models/thuan_an/wind_tide30_fc_gru32x3_seq12_cosine_ep77', ta_cfg, 'ta_water_only_wind_tide30_fc'),
          'wind_gru32x2_seq25': member('models/thuan_an/wind_gru32x2_seq25_ep33', ta_cfg, 'ta_water_only_wind')},
         {'rule': 'mean', 'horizons': {3: ['wind_tide30_fc_gru32x3_seq12', 'wind_gru32x2_seq25'],
                                       6: ['wind_tide30_fc_gru32x3_seq12', 'wind_gru32x2_seq25'],
                                       12: ['wind_tide30_fc_gru32x3_seq12', 'wind_gru32x2_seq25']}},
         ['train_data/thuan_an/ta_water_only_wind_tide30_fc', 'train_data/thuan_an/ta_water_only_wind'], marine=False)

nbs = json.loads(Path('models/nga_ba_sinh/combined_v1/combined.json').read_text())['members']
cfg = lambda ds: json.loads((Path(ds) / 'manifest.json').read_text())['config']
assemble('nga_ba_sinh', 'nga_ba_sinh_combined_v1', '0074647705', 'Trạm Ngã Ba Sình (sông Hương)',
         {'short_3h_6h': member('models/nga_ba_sinh/combined_v1/short_3h_6h', cfg(nbs['short_3h_6h']['dataset']), 'nbs_onshore_tide7_fc_sea7_v3'),
          'long_12h': member('models/nga_ba_sinh/combined_v1/long_12h', cfg(nbs['long_12h']['dataset']), 'nbs_onshore_tide7_fc_sea_v3')},
         {'rule': 'per_horizon', 'horizons': {3: ['short_3h_6h'], 6: ['short_3h_6h'], 12: ['long_12h']}},
         [nbs['short_3h_6h']['dataset'], nbs['long_12h']['dataset']], marine=True)
