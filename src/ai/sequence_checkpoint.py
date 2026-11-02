"""Explicit architecture metadata beside raw state_dicts; legacy weights stay loadable."""
from io import BytesIO
from pathlib import Path
import json
import re

import joblib
import numpy as np
import torch

from .lstm import LSTM, ResidualLSTM
from .residual_sequence import ResidualSequence, residual_model


def model_spec(model):
    if isinstance(model, ResidualSequence):
        return dict(format_version=1, family='residual_sequence', options=model.architecture_config)
    if isinstance(model, LSTM):
        options = {k: getattr(model, k) for k in ['input_size', 'output_size', 'hidden_size', 'num_layers',
                                                'dropout', 'input_dropout', 'head_dropout']}
        if isinstance(model, ResidualLSTM):
            options.update(level_feature_index=model.level_feature_index,
                persistence_scale=model.persistence_scale.detach().cpu().tolist(),
                persistence_offset=model.persistence_offset.detach().cpu().tolist(),
                forecast_history_policy=model.forecast_history_policy,
                weather_feature_indices=model._weather_indices.cpu().tolist(),
                weather_mask_indices=model._weather_mask_indices.cpu().tolist(),
                weather_mask_values=model._weather_mask_values.cpu().tolist())
        return dict(format_version=1, family='residual_lstm' if isinstance(model, ResidualLSTM) else 'lstm', options=options)
    return None


def model_from_spec(spec):
    if spec.get('format_version') != 1:
        raise ValueError('Unsupported architecture metadata version')
    options = dict(spec['options'])
    if spec['family'] == 'residual_sequence':
        return residual_model(options.pop('architecture'), **options)
    if spec['family'] == 'residual_lstm':
        return ResidualLSTM(**options)
    if spec['family'] == 'lstm':
        return LSTM(**options)
    raise ValueError('Unsupported model family')


def save_model_metadata(model, checkpoint):
    spec = model_spec(model)
    if spec is not None:
        target = Path(checkpoint).with_suffix('.architecture.json')
        target.write_text(json.dumps(spec, indent=2, allow_nan=False)+'\n')


def load_model_state(state, saved, device='cpu'):
    """Reconstruct new models exactly, or infer the unchanged legacy LSTM layout."""
    spec = saved.get('model_spec')
    if spec is not None:
        model = model_from_spec(spec)
    else:
        layers = [k for k in state if re.fullmatch(r'lstm.weight_ih_l\d+', k)]
        if not layers:
            raise ValueError('Architecture metadata is required for this checkpoint')
        options = dict(input_size=state['lstm.weight_ih_l0'].shape[1],
            output_size=state['linear.weight'].shape[0], hidden_size=state['lstm.weight_hh_l0'].shape[1],
            num_layers=len(layers), dropout=0.)
        if 'persistence_scale' in state:
            policy = saved.get('forecast_history_policy', 'all_timesteps')
            weather = [i for i, name in enumerate(saved['feature_columns']) if name.startswith('weather_forecast_')]
            masks = [i for i in weather if saved['feature_columns'][i].endswith('_missing')]
            fs = saved.get('feature_scaler')
            values = ((1.-fs.mean_[masks])/fs.scale_[masks]).tolist() if fs is not None else [1.]*len(masks)
            options.update(level_feature_index=saved['feature_columns'].index('water_level_depth'),
                persistence_scale=state['persistence_scale'].tolist(), persistence_offset=state['persistence_offset'].tolist(),
                forecast_history_policy=policy, weather_feature_indices=weather,
                weather_mask_indices=masks, weather_mask_values=values)
            model = ResidualLSTM(**options)
        else:
            model = LSTM(**options)
    for key in ['persistence_scale', 'persistence_offset']:
        if hasattr(model, key) and (key not in state or not torch.allclose(state[key], getattr(model, key))):
            raise ValueError('Saved architecture persistence coefficients disagree with weights')
    model.load_state_dict(state, strict=True)
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError('Checkpoint contains nonfinite weights')
    return model.to(device).eval()


class SequenceRuntime:
    """Offline inference from the same checkpoint/preprocessing pair used by training.

    Inputs must already follow saved feature order and preprocessing, just as
    the existing app runtime receives a prepared input tensor.
    """
    def __init__(self, checkpoint, device='cpu'):
        checkpoint = Path(checkpoint)
        self.saved = joblib.load(checkpoint.with_suffix('.preprocessing.joblib'))
        self.device = torch.device(device)
        state = torch.load(BytesIO(checkpoint.read_bytes()), map_location='cpu', weights_only=True)
        self.model = load_model_state(state, self.saved, device)

    def predict_mm(self, x, station_ids=None):
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        ids = None if station_ids is None else torch.as_tensor(station_ids, dtype=torch.long, device=self.device)
        with torch.inference_mode():
            prediction = self.model(x) if ids is None else self.model(x, ids)
        values = prediction.cpu().numpy()
        regional = self.saved.get('regional')
        if regional:
            if ids is None and not regional.get('selected_station'):
                raise ValueError('Regional inference requires station IDs')
            reverse = {v['station_index']: sid for sid, v in regional['station_mapping'].items()}
            stats = regional['normalizers']['stations']
            sids = [regional['selected_station']]*len(values) if ids is None else [reverse[int(i)] for i in ids.cpu()]
            horizons = regional.get('target_horizons')
            if horizons is None:
                columns = self.saved.get('target_columns')
                horizons = ([int(col.rsplit('_', 1)[-1][:-1])//60 for col in columns]
                            if columns else [6, 12])
            scale = np.array([[stats[sid]['targets'][f'h{h}']['scale'] for h in horizons] for sid in sids])
            mean = np.array([[stats[sid]['targets'][f'h{h}']['mean'] for h in horizons] for sid in sids])
            return values*scale+mean
        return self.saved['target_scaler'].inverse_transform(values)

    def predict_with_diagnostics_mm(self, x, station_ids):
        """Return event-aware predictions and components, without target data."""
        if not getattr(self.model, 'event_aware', False):
            raise ValueError('Event diagnostics require an event-aware checkpoint')
        regional = self.saved.get('regional')
        if not regional or station_ids is None:
            raise ValueError('Event diagnostics require regional station IDs')
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        ids = torch.as_tensor(station_ids, dtype=torch.long, device=self.device)
        with torch.inference_mode():
            predicted, aux = self.model(x, ids, return_diagnostics=True)
        reverse = {v['station_index']: sid for sid, v in regional['station_mapping'].items()}
        horizons = regional.get('target_horizons', [6, 12])
        stats = regional['normalizers']['stations']
        sids = [reverse[int(i)] for i in ids.cpu()]
        scale = np.array([[stats[s]['targets'][f'h{h}']['scale'] for h in horizons] for s in sids])
        mean = np.array([[stats[s]['targets'][f'h{h}']['mean'] for h in horizons] for s in sids])
        return dict(predicted_mm=predicted.cpu().numpy()*scale+mean,
                    rise_logit=aux['rise_logits'].cpu().numpy(),
                    rise_probability=aux['rise_probability'].cpu().numpy(),
                    base_delta_mm=aux['base_delta_normalized'].cpu().numpy()*scale,
                    amplitude_mm=aux['amplitude_normalized'].cpu().numpy()*scale,
                    correction_mm=aux['correction_normalized'].cpu().numpy()*scale,
                    final_delta_mm=aux['final_delta_normalized'].cpu().numpy()*scale)
