"""Initialise a residual sequence model from another station's checkpoint (transfer / finetuning).

Every weight is copied unchanged, except:
- persistence_scale / persistence_offset stay those of the NEW dataset (its own level and target normalisers);
- the output head keeps only the rows of the horizons the new model predicts (e.g. source +3/+6/+12 h -> target +6 h).
The input columns (names and order), sequence architecture and head layout must match exactly; the new dataset's own
input normalisers are used (inputs are standardised per station, so the weights act on the same relative units).
"""
from pathlib import Path

import joblib
import torch

ARCH_KEYS = ('architecture', 'input_size', 'hidden_size', 'num_layers', 'num_stations', 'station_embedding_size',
             'head_mode', 'horizon_heads', 'horizon_head_size', 'horizon_head_layers', 'horizon_gru_layers', 'prediction_mode')


def init_from_checkpoint(model, checkpoint, feature_columns, target_horizons):
    checkpoint = Path(checkpoint)
    saved = joblib.load(checkpoint.with_suffix('.preprocessing.joblib'))
    if list(saved['feature_columns']) != list(feature_columns):
        raise ValueError(f'Input columns differ from {checkpoint}:\n  source {list(saved["feature_columns"])}\n  target {list(feature_columns)}')
    src_opts = saved['model_spec']['options']
    dst_opts = model.architecture_config
    diff = {k: (src_opts.get(k), dst_opts.get(k)) for k in ARCH_KEYS if src_opts.get(k) != dst_opts.get(k)}
    if diff:
        raise ValueError(f'Architecture differs from {checkpoint}: {diff}')
    src_h = list(saved['regional']['target_horizons'])
    missing = [h for h in target_horizons if h not in src_h]
    if missing:
        raise ValueError(f'Source checkpoint has no output for horizons {missing} (has {src_h})')
    rows = torch.tensor([src_h.index(h) for h in target_horizons])
    state = torch.load(checkpoint, map_location='cpu')
    new = model.state_dict()
    copied, sliced = [], []
    for name, value in new.items():
        if name.startswith('persistence_'):
            continue
        if name not in state:
            raise ValueError(f'{name} not in source checkpoint')
        src = state[name]
        if src.shape == value.shape and not (name.startswith('linear.') and len(src_h) != len(target_horizons)):
            new[name] = src.clone()
            copied.append(name)
            continue
        if not name.startswith('linear.'):
            raise ValueError(f'{name}: shape {tuple(src.shape)} vs {tuple(value.shape)}')
        axis = 1 if dst_opts.get('head_mode') in ('station', 'station-mlp') else 0   # station heads: [stations, horizons, ...]
        out = src.index_select(axis, rows)
        if out.shape != value.shape:
            raise ValueError(f'{name}: sliced shape {tuple(out.shape)} vs {tuple(value.shape)}')
        new[name] = out.clone()
        sliced.append(name)
    model.load_state_dict(new)
    return dict(checkpoint=str(checkpoint), source_horizons=src_h, target_horizons=list(target_horizons),
                copied=len(copied), sliced=sliced, kept=[k for k in new if k.startswith('persistence_')])
