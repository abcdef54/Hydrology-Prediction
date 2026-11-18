"""In-repo joblib bundles (format v1, needs this repository and torch to load) for the station models (Thuận An,
Ngã Ba Sình) and their reference inference. The standalone hand-off folders deploy/thuan_an and deploy/nga_ba_sinh
(bundle format v2: plain dicts / numpy only, self-contained predict.py and worker.py) are built by
research/scripts/make_deploy_folders.py from src/ai/station_dataset/deploy_templates/.

A bundle (one .joblib per station) holds everything a deployment needs except the raw input data:
  members[name] = {state_dict (numpy arrays), preprocessing (the training model.preprocessing.joblib dict: feature
                   columns, normalizers, architecture spec, horizons, window), feature_config (dataset config used to
                   build that member's inputs), groups}
  combination   = {'rule': 'mean' | 'per_horizon', 'horizons': {h: [member names]}}
plus station metadata, metrics and caveats. Loading needs torch and this repository (model classes in src/ai).

    # build the bundles (once)
    PYTHONPATH=. .venv/bin/python -m src.ai.station_dataset.deploy build-all --out-dir models/deploy

    # forecast at issue time t (UTC hour) from raw sources
    from src.ai.station_dataset.deploy import StationBundle
    b = StationBundle.load('models/deploy/nga_ba_sinh_combined_v1.joblib')
    out = b.predict('2026-11-10T03:00Z', sources=dict(
        wind_previous_runs_json='data/external/open_meteo/prev_offshore_to_20261116.json',
        ecmwf_single_runs_dir='data/external/open_meteo/ecmwf_single_runs_soil',
        marine_json='data/external/open_meteo/marine_offshore_ta_to_20261116.json'))
    # out = {3: mm, 6: mm, 12: mm}; level is read from PostgreSQL unless sources['level'] (hourly Series, mm) is given

Inputs per member are rebuilt with the same functions as training (src/ai/station_dataset/features.py), on an hourly
grid ending at t, then standardized with the member's own train-only normalizers (NaN -> 0, masks kept raw).
"""
import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.ai.regional_data.preprocessing import calendar_features
from src.ai.sequence_checkpoint import load_model_state
from src.ai.station_dataset import features as F
from src.ai.station_dataset.open_meteo import load_single_runs

FORMAT = 'hydrology_station_bundle_v1'
SOURCE_KEYS = {'wind': ('wind', 'previous_runs_json', 'wind_previous_runs_json'),
               'fc_ecmwf': ('fc_ecmwf', 'single_runs_dir', 'ecmwf_single_runs_dir'),
               'sea': ('sea', 'marine_json', 'marine_json')}
HISTORY_DAYS = 32          # longest need: 30-day tide window + 25-h input window + margin


def _member(model_dir, feature_config):
    model_dir = Path(model_dir)
    saved = joblib.load(model_dir / 'model.preprocessing.joblib')
    raw = (model_dir / 'model.pt').read_bytes()
    state = torch.load(BytesIO(raw), map_location='cpu', weights_only=True)
    return dict(source_dir=str(model_dir), model_pt_sha256=hashlib.sha256(raw).hexdigest(),
                state_dict={k: v.numpy() for k, v in state.items()}, preprocessing=saved,
                groups=list(saved['regional']['rain_stations']), feature_config=feature_config)


def make_bundle(name, station_id, members, combination, out, info=None):
    """members: {member_name: (model_dir, feature_config_dict)}; combination: {'rule', 'horizons': {h: [names]}}."""
    bundle = dict(format=FORMAT, name=name, station_id=station_id, combination=combination,
                  members={m: _member(d, cfg) for m, (d, cfg) in members.items()},
                  created=pd.Timestamp.now(tz='UTC').isoformat(), info=info or {})
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out, compress=3)
    return bundle


class StationBundle:
    def __init__(self, bundle, device='cpu'):
        if bundle.get('format') != FORMAT:
            raise ValueError('not a station bundle')
        self.bundle, self.device = bundle, device
        self.models = {m: load_model_state({k: torch.from_numpy(np.asarray(v)) for k, v in b['state_dict'].items()},
                                           b['preprocessing'], device) for m, b in bundle['members'].items()}

    @classmethod
    def load(cls, path, device='cpu'):
        return cls(joblib.load(path), device)

    # ------------------------------------------------------------------ inputs
    def level(self, cfg, index, sources):
        if sources.get('level') is not None:
            return sources['level'].reindex(index).astype(float)
        from src.ai.station_dataset.build import hourly_level_from_db
        return hourly_level_from_db(cfg['station_id'], index, despike=cfg.get('despike'),
                                    invalid_ranges=cfg.get('invalid_ranges'))

    @staticmethod
    def group_frames(cfg, groups, index, level, sources):
        """Same glue as build.build() for the groups used by the deployed members."""
        frames = {}
        for g in groups:
            if g in SOURCE_KEYS:
                block, key, src = SOURCE_KEYS[g]
                path = sources.get(src) or cfg[block][key]
            if g == 'wind':
                w = F.wind_features(path, index, onshore_pos=cfg['wind'].get('onshore_pos', False))
                if cfg['wind'].get('channels'):
                    w = w[[c for c in w.columns if c in cfg['wind']['channels'] or c == 'wind_missing_frac']]
                frames[g] = w
            elif g == 'tide':
                # fallback fit (hours without a valid trailing window) uses all available history here, the
                # train-only fit in training; with >= window_days of data before t the fallback is never used.
                frames[g] = F.tide_features(level, np.isfinite(level.to_numpy()), window_days=cfg['tide']['window_days'])
            elif g == 'fc_ecmwf':
                fc = cfg['fc_ecmwf']
                cells = sorted(fc['all_cells'])
                sync = F.synchronized_rain(load_single_runs(path), index, cells, fc.get('lag_hours', 6), fc.get('max_age_hours', 24))
                r = F.forecast_rain_features(sync, index, fc['named_cells'], cells, fc.get('transform', 'log1p'))
                if fc.get('channels'):
                    r = r[[c for c in r.columns if c in fc['channels'] or c == 'fc_missing_frac']]
                frames[g] = r
            elif g == 'sea':
                frames[g] = F.sea_level_features(path, index, cfg['sea'].get('week_hours', 168), cfg['sea'].get('direct', False))
            else:
                raise ValueError(f'group {g} not supported by the deployment glue')
        return frames

    def member_input(self, name, t, sources, frames_cache=None):
        """Standardized input window (1, L, F) of member `name` for issue time t (UTC hour)."""
        b = self.bundle['members'][name]
        saved, cfg = b['preprocessing'], b['feature_config']
        reg = saved['regional']
        L = reg['training_selection']['sequence_length']
        t = pd.Timestamp(t).tz_convert('UTC') if pd.Timestamp(t).tzinfo else pd.Timestamp(t, tz='UTC')
        index = pd.date_range(t - pd.Timedelta(days=HISTORY_DAYS), t, freq='h')
        key = (id(cfg), t)
        if frames_cache is not None and key in frames_cache:
            level, frames = frames_cache[key]
        else:
            level = self.level(cfg, index, sources)
            frames = self.group_frames(cfg, b['groups'], index, level, sources)
            if frames_cache is not None:
                frames_cache[key] = (level, frames)
        sid = self.bundle['station_id']
        norm = reg['normalizers']
        w = norm['stations'][sid]['water']
        table = pd.concat([calendar_features(index)[['flood_season']]] + [frames[g] for g in b['groups']], axis=1)
        cols = saved['feature_columns']
        x = np.zeros((len(index), len(cols)), dtype=np.float32)
        x[:, 0] = np.nan_to_num((level.to_numpy() - w['mean']) / w['scale'], nan=0.)
        x[:, 1] = ~np.isfinite(level.to_numpy())
        for j, c in enumerate(cols[2:], start=2):
            v = table[c].to_numpy(dtype=float)
            if c in norm['shared']:
                v = (v - norm['shared'][c]['mean']) / norm['shared'][c]['scale']
            x[:, j] = np.nan_to_num(v, nan=0.)
        window = x[-L:]
        coverage = float(np.isfinite(level.to_numpy()[-L:]).mean())
        return window[None], coverage

    # ------------------------------------------------------------------ forecast
    def member_predict(self, name, x):
        reg = self.bundle['members'][name]['preprocessing']['regional']
        sid = self.bundle['station_id']
        ids = torch.full((len(x),), reg['station_mapping'][sid]['station_index'], dtype=torch.long)
        with torch.inference_mode():
            out = self.models[name](torch.as_tensor(x, dtype=torch.float32, device=self.device), ids.to(self.device)).cpu().numpy()
        stats = reg['normalizers']['stations'][sid]['targets']
        hs = reg['target_horizons']
        return {h: out[:, i] * stats[f'h{h}']['scale'] + stats[f'h{h}']['mean'] for i, h in enumerate(hs)}

    def predict(self, t, sources=None, min_coverage=0.8):
        """{horizon: forecast level (mm)} at issue time t, combined per the bundle rule."""
        sources = sources or {}
        cache, per = {}, {}
        for name in self.bundle['members']:
            x, cov = self.member_input(name, t, sources, cache)
            if cov < min_coverage:
                raise ValueError(f'{name}: only {cov:.0%} of the {x.shape[1]}-h level window is observed')
            per[name] = self.member_predict(name, x)
        comb = self.bundle['combination']
        return {int(h): float(np.mean([per[m][int(h)][0] for m in names])) for h, names in comb['horizons'].items()}


# ---------------------------------------------------------------------- the two deployed stations
def _cfg_from_manifest(dataset_dir):
    return json.loads((Path(dataset_dir) / 'manifest.json').read_text())['config']


def build_all(out_dir):
    out_dir = Path(out_dir)
    ta_cfg = json.loads(Path('models/thuan_an/dataset_config.json').read_text())
    ta = json.loads(Path('models/thuan_an/ensemble_34k.json').read_text())
    make_bundle('thuan_an_ensemble_34k', '0074647402',
                {'wind_tide30_fc_gru32x3_seq12_cosine_ep77': ('models/thuan_an/wind_tide30_fc_gru32x3_seq12_cosine_ep77', ta_cfg),
                 'wind_gru32x2_seq25_ep33': ('models/thuan_an/wind_gru32x2_seq25_ep33', ta_cfg)},
                {'rule': 'mean', 'horizons': {h: list(ta['members']) for h in (3, 6, 12)}},
                out_dir / 'thuan_an_ensemble_34k.joblib',
                info=dict(selection=ta['selection'], metrics_sep_oct_2026=ta['sep_oct_2026'], caveats=ta['caveats'],
                          docs='models/thuan_an/README.md, models/deploy/README.md'))
    nbs = json.loads(Path('models/nga_ba_sinh/combined_v1/combined.json').read_text())
    make_bundle('nga_ba_sinh_combined_v1', '0074647705',
                {'short_3h_6h': ('models/nga_ba_sinh/combined_v1/short_3h_6h', _cfg_from_manifest(nbs['members']['short_3h_6h']['dataset'])),
                 'long_12h': ('models/nga_ba_sinh/combined_v1/long_12h', _cfg_from_manifest(nbs['members']['long_12h']['dataset']))},
                {'rule': 'per_horizon', 'horizons': {3: ['short_3h_6h'], 6: ['short_3h_6h'], 12: ['long_12h']}},
                out_dir / 'nga_ba_sinh_combined_v1.joblib',
                info=dict(rule=nbs['rule'], metrics=nbs['metrics'], periods=nbs['periods'], caveats=nbs['caveats'],
                          docs='models/nga_ba_sinh/REPRODUCE.md, models/deploy/README.md'))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build-all'); b.add_argument('--out-dir', default='models/deploy')
    a = p.parse_args()
    if a.cmd == 'build-all':
        build_all(a.out_dir)
        for f in sorted(Path(a.out_dir).glob('*.joblib')):
            print(f, f.stat().st_size, hashlib.sha256(f.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
