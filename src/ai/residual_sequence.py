"""Residual sequence encoders with early/late forecast fusion and station conditioning."""
import math
import torch
from torch import nn
from torch.nn import functional as F

NEW_METHODS = ('gru', 'tcn', 'cnn-gru', 'gru-late-fusion', 'tcn-late-fusion', 'cnn-gru-late-fusion',
               'transformer', 'transformer-late-fusion')


class TransformerHistory(nn.Module):
    """Causal self-attention over observed history, with dynamic sinusoidal positions."""
    def __init__(self, input_size, hidden_size, num_layers, dropout, heads, feedforward_size):
        super().__init__()
        self.projection = nn.Linear(input_size, hidden_size)
        layer = nn.TransformerEncoderLayer(hidden_size, heads, feedforward_size,
            dropout, activation='gelu', batch_first=True, norm_first=True)
        self.layers = nn.TransformerEncoder(layer, num_layers,
            norm=nn.LayerNorm(hidden_size), enable_nested_tensor=False)
        # TransformerEncoder clones one layer; give each layer independent weights.
        for block in self.layers.layers:
            for parameter in block.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
        self.position_dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.projection(x)
        length, width = x.shape[1:]
        position = torch.arange(length, device=x.device, dtype=torch.float32)[:, None]
        frequency = torch.exp(torch.arange(0, width, 2, device=x.device,
            dtype=torch.float32)*(-math.log(10000.)/width))
        angles = position*frequency
        encoding = torch.zeros(length, width, device=x.device, dtype=torch.float32)
        encoding[:, 0::2] = angles.sin()
        encoding[:, 1::2] = angles[:, :width//2].cos()
        x = self.position_dropout(x+encoding.to(dtype=x.dtype))
        mask = torch.ones(length, length, device=x.device, dtype=torch.bool).triu(1)
        return self.layers(x, mask=mask, is_causal=True)


class CausalConv1d(nn.Conv1d):
    """Left padding only; output j cannot depend on inputs after j."""
    def __init__(self, input_size, output_size, kernel_size, dilation=1):
        if kernel_size < 1 or dilation < 1:
            raise ValueError('kernel_size and dilation must be positive')
        super().__init__(input_size, output_size, kernel_size, dilation=dilation)
        self.left_padding = (kernel_size-1)*dilation

    def forward(self, x):
        return super().forward(F.pad(x, (self.left_padding, 0)))


class TemporalBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, dilation, dropout):
        super().__init__()
        self.layers = nn.Sequential(CausalConv1d(input_size, output_size, kernel_size, dilation),
            nn.ReLU(), nn.Dropout(dropout), CausalConv1d(output_size, output_size, kernel_size, dilation),
            nn.ReLU(), nn.Dropout(dropout))
        self.shortcut = nn.Identity() if input_size == output_size else nn.Conv1d(input_size, output_size, 1)

    def forward(self, x):
        return F.relu(self.layers(x)+self.shortcut(x))


class SequenceEncoder(nn.Module):
    def __init__(self, kind, input_size, hidden_size, num_layers, dropout,
                 tcn_channels, tcn_kernel_size, cnn_channels, cnn_kernel_size,
                 transformer_heads=4, transformer_ff_size=128):
        super().__init__()
        self.kind = kind
        channels = input_size
        if kind == 'transformer':
            self.temporal = TransformerHistory(input_size, hidden_size, num_layers, dropout,
                                               transformer_heads, transformer_ff_size)
            self.output_size = hidden_size
        elif kind == 'tcn':
            blocks = []
            for i, width in enumerate(tcn_channels):
                blocks.append(TemporalBlock(channels, width, tcn_kernel_size, 2**i, dropout))
                channels = width
            self.temporal = nn.Sequential(*blocks)
            self.output_size = channels
            self.receptive_field = 1+2*(tcn_kernel_size-1)*sum(2**i for i in range(len(tcn_channels)))
        else:
            layers = []
            if kind == 'cnn-gru':
                for i, width in enumerate(cnn_channels):
                    layers.extend([CausalConv1d(channels, width, cnn_kernel_size, 2**i), nn.ReLU(), nn.Dropout(dropout)])
                    channels = width
            self.temporal = nn.Sequential(*layers)
            cls = nn.LSTM if kind == 'lstm' else nn.GRU
            self.recurrent = cls(channels, hidden_size, num_layers, batch_first=True,
                                dropout=dropout if num_layers > 1 else 0.)
            self.output_size = hidden_size

    def forward_sequence(self, x):
        if self.kind == 'transformer':
            return self.temporal(x)
        x = self.temporal(x.transpose(1, 2)).transpose(1, 2)
        if self.kind != 'tcn':
            x, _ = self.recurrent(x)
        return x

    def forward(self, x):
        return self.forward_sequence(x)[:, -1]


class StationCorrectionHead(nn.Module):
    """One zero-initialized correction projection per station."""
    def __init__(self, stations, hidden_size, output_size):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(stations, output_size, hidden_size))
        self.bias = nn.Parameter(torch.zeros(stations, output_size))

    def forward(self, features, station_ids):
        weights = self.weight[station_ids]
        if features.ndim == 3:
            return (weights * features).sum(dim=-1) + self.bias[station_ids]
        return torch.bmm(weights, features.unsqueeze(-1)).squeeze(-1)+self.bias[station_ids]


class StationMLPCorrectionHead(nn.Module):
    """Three hidden layers and a separate correction for each station."""
    def __init__(self, stations, input_size, output_size, mlp_hidden_size=32):
        super().__init__()
        widths = (input_size, mlp_hidden_size, mlp_hidden_size, mlp_hidden_size, output_size)
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for i, (input_width, output_width) in enumerate(zip(widths, widths[1:])):
            weight = nn.Parameter(torch.empty(stations, output_width, input_width))
            bias = nn.Parameter(torch.empty(stations, output_width))
            if i == len(widths)-2:
                nn.init.zeros_(weight)
                nn.init.zeros_(bias)
            else:
                for station_weight in weight:
                    nn.init.kaiming_uniform_(station_weight, a=math.sqrt(5))
                bound = 1 / math.sqrt(input_width)
                nn.init.uniform_(bias, -bound, bound)
            self.weights.append(weight)
            self.biases.append(bias)

    def forward(self, features, station_ids):
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            features = torch.bmm(weight[station_ids], features.unsqueeze(-1)).squeeze(-1) + bias[station_ids]
            if i < len(self.weights)-1:
                features = torch.relu(features)
        return features


class ResidualSequence(nn.Module):
    """Station-aware persistence plus a zero-initialized learned correction."""
    def __init__(self, *, architecture='gru', input_size, output_size=2, hidden_size=32,
                 num_layers=3, dropout=.2, input_dropout=0., head_dropout=0.,
                 level_feature_index=0, persistence_scale=None, persistence_offset=None,
                 weather_feature_indices=None, weather_mask_indices=None, weather_mask_values=None,
                 forecast_history_policy='anchor_only', num_stations=0, station_embedding_size=8,
                 forecast_hidden_size=32, tcn_channels=(32, 32, 32, 32, 32), tcn_kernel_size=3,
                 cnn_channels=(32, 32), cnn_kernel_size=3,
                 transformer_heads=4, transformer_ff_size=128, head_mode='shared', horizon_decoder='none',
                 station_mlp_hidden_size=32, prediction_mode='residual', event_aware=False):
        super().__init__()
        kind = architecture.removesuffix('-late-fusion')
        if kind not in ('lstm', 'gru', 'tcn', 'cnn-gru', 'transformer'):
            raise ValueError('Unknown sequence architecture')
        if prediction_mode not in ('residual', 'direct'):
            raise ValueError('Prediction mode must be residual or direct')
        if event_aware and (kind != 'transformer' or prediction_mode != 'residual' or horizon_decoder != 'none'):
            raise ValueError('Event-aware mode requires a residual Transformer without a horizon decoder')
        for name, value in [('input_size', input_size), ('output_size', output_size), ('hidden_size', hidden_size),
                            ('num_layers', num_layers), ('station_embedding_size', station_embedding_size),
                            ('forecast_hidden_size', forecast_hidden_size), ('tcn_kernel_size', tcn_kernel_size),
                            ('cnn_kernel_size', cnn_kernel_size)]:
            if not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if kind == 'transformer' or horizon_decoder == 'transformer':
            if (not isinstance(transformer_heads, int) or transformer_heads < 1
                    or not isinstance(transformer_ff_size, int) or transformer_ff_size < 1):
                raise ValueError('Transformer heads and feed-forward size must be positive integers')
            if hidden_size % transformer_heads:
                raise ValueError('Transformer hidden_size must be divisible by transformer_heads')
        if not isinstance(num_stations, int) or num_stations < 0:
            raise ValueError('num_stations must be nonnegative')
        if head_mode not in ('shared', 'station', 'station-mlp') or (head_mode != 'shared' and not num_stations):
            raise ValueError('Station heads require a regional station-conditioned model')
        if not isinstance(station_mlp_hidden_size, int) or isinstance(station_mlp_hidden_size, bool) or station_mlp_hidden_size < 1:
            raise ValueError('Station MLP hidden size must be a positive integer')
        if head_mode != 'station-mlp' and station_mlp_hidden_size != 32:
            raise ValueError('Station MLP hidden size requires station-mlp head mode')
        if horizon_decoder not in ('none', 'transformer'):
            raise ValueError('Unknown horizon decoder')
        if horizon_decoder == 'transformer' and (head_mode != 'station' or not architecture.endswith('-late-fusion')):
            raise ValueError('Horizon decoder requires late fusion and station heads')
        self.head_mode = head_mode
        self.prediction_mode = prediction_mode
        self.event_aware = bool(event_aware)
        self.horizon_decoder_mode = horizon_decoder
        for rate in [dropout, input_dropout, head_dropout]:
            if not 0 <= rate < 1:
                raise ValueError('Dropout must be in [0,1)')
        if not tcn_channels or not cnn_channels or any(not isinstance(n, int) or n < 1 for n in [*tcn_channels, *cnn_channels]):
            raise ValueError('Convolution channels must be positive integers')
        weather = list(weather_feature_indices or [])
        masks, values = list(weather_mask_indices or []), list(weather_mask_values or [])
        if (len(set(weather)) != len(weather) or len(set(masks)) != len(masks)
                or any(i < 0 or i >= input_size for i in weather)
                or not set(masks).issubset(weather) or len(masks) != len(values)
                or level_feature_index in weather or not 0 <= level_feature_index < input_size):
            raise ValueError('Invalid weather/persistence feature mapping')
        if forecast_history_policy not in ('anchor_only', 'all_timesteps'):
            raise ValueError('Invalid forecast history policy')
        if forecast_history_policy == 'anchor_only' and weather and not masks:
            raise ValueError('Weather channels require explicit missing masks')
        scale = torch.as_tensor([1.]*output_size if persistence_scale is None else persistence_scale, dtype=torch.float32)
        offset = torch.as_tensor([0.]*output_size if persistence_offset is None else persistence_offset, dtype=torch.float32)
        expected = (num_stations, output_size) if num_stations else (output_size,)
        if (scale.shape != expected or offset.shape != expected or not torch.isfinite(scale).all()
                or not torch.isfinite(offset).all() or not (scale > 0).all()
                or not torch.isfinite(torch.tensor(values)).all()):
            raise ValueError('Invalid station/target persistence coefficients')
        self.input_size, self.output_size = input_size, output_size
        self.hidden_size, self.num_layers = hidden_size, num_layers
        self.dropout, self.input_dropout, self.head_dropout = dropout, input_dropout, head_dropout
        self.level_feature_index, self.num_stations = level_feature_index, num_stations
        self.forecast_history_policy = forecast_history_policy
        self.late_fusion = architecture.endswith('-late-fusion')
        self.architecture_config = dict(architecture=architecture, input_size=input_size, output_size=output_size,
            hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, input_dropout=input_dropout,
            head_dropout=head_dropout, level_feature_index=level_feature_index, persistence_scale=scale.tolist(),
            persistence_offset=offset.tolist(), weather_feature_indices=weather, weather_mask_indices=masks,
            weather_mask_values=values, forecast_history_policy=forecast_history_policy, num_stations=num_stations,
            station_embedding_size=station_embedding_size, forecast_hidden_size=forecast_hidden_size,
            tcn_channels=list(tcn_channels), tcn_kernel_size=tcn_kernel_size,
            cnn_channels=list(cnn_channels), cnn_kernel_size=cnn_kernel_size)
        if kind == 'transformer' or horizon_decoder == 'transformer':
            self.architecture_config.update(transformer_heads=transformer_heads,
                                           transformer_ff_size=transformer_ff_size)
        if head_mode != 'shared':
            self.architecture_config['head_mode'] = head_mode
        if head_mode == 'station-mlp':
            self.architecture_config['station_mlp_hidden_size'] = station_mlp_hidden_size
        if horizon_decoder != 'none':
            self.architecture_config['horizon_decoder'] = horizon_decoder
        if prediction_mode != 'residual':
            self.architecture_config['prediction_mode'] = prediction_mode
        if event_aware:
            self.architecture_config['event_aware'] = True
        self.register_buffer('persistence_scale', scale)
        self.register_buffer('persistence_offset', offset)
        self.register_buffer('_weather_indices', torch.tensor(weather, dtype=torch.long), persistent=False)
        self.register_buffer('_mask_indices', torch.tensor(masks, dtype=torch.long), persistent=False)
        self.register_buffer('_mask_values', torch.tensor(values, dtype=torch.float32), persistent=False)
        history = [i for i in range(input_size) if not self.late_fusion or i not in weather]
        self.register_buffer('_history_indices', torch.tensor(history, dtype=torch.long), persistent=False)
        self.input_dropout_layer = nn.Dropout1d(input_dropout)
        self.encoder = SequenceEncoder(kind, len(history), hidden_size, num_layers, dropout,
                                       tcn_channels, tcn_kernel_size, cnn_channels, cnn_kernel_size,
                                       transformer_heads, transformer_ff_size)
        self.station_embedding = nn.Embedding(num_stations, station_embedding_size) if num_stations else None
        self.forecast_encoder = (nn.Sequential(nn.Dropout(input_dropout), nn.Linear(len(weather), forecast_hidden_size),
            nn.ReLU(), nn.Dropout(head_dropout), nn.Linear(forecast_hidden_size, forecast_hidden_size), nn.ReLU())
            if self.late_fusion and weather else None)
        head_size = self.encoder.output_size+(station_embedding_size if num_stations else 0)+(forecast_hidden_size if self.forecast_encoder else 0)
        # Nonlinear fusion allows station identity and forecast to modulate the
        # history response, rather than contributing only additive biases.
        self.fusion = nn.Sequential(nn.Linear(head_size, hidden_size), nn.ReLU())
        self.head_dropout_layer = nn.Dropout(head_dropout)
        self.linear = nn.Linear(hidden_size, output_size)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        # Keep initialization/RNG of shared components identical to the control.
        if head_mode == 'station':
            self.linear = StationCorrectionHead(num_stations, hidden_size, output_size)
        elif head_mode == 'station-mlp':
            self.linear = StationMLPCorrectionHead(num_stations, hidden_size, output_size, station_mlp_hidden_size)
        if event_aware:
            self.rise_head = nn.Linear(hidden_size, output_size)
            self.amplitude_head = nn.Linear(hidden_size, output_size)
            nn.init.zeros_(self.rise_head.weight)
            nn.init.constant_(self.rise_head.bias, -2.)
            nn.init.zeros_(self.amplitude_head.weight)
            nn.init.constant_(self.amplitude_head.bias, -5.)
        if horizon_decoder == 'transformer':
            self.memory_projection = (nn.Identity() if self.encoder.output_size == hidden_size
                                      else nn.Linear(self.encoder.output_size, hidden_size))
            self.horizon_queries = nn.Parameter(torch.empty(output_size, hidden_size))
            nn.init.normal_(self.horizon_queries, std=0.02)
            self.horizon_decoder = nn.TransformerDecoderLayer(
                d_model=hidden_size, nhead=transformer_heads, dim_feedforward=transformer_ff_size,
                dropout=dropout, activation='gelu', batch_first=True, norm_first=True)

    def _batch(self, x, station_ids, validate_station_ids=True):
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim != 3 or x.shape[-1] != self.input_size or x.shape[1] == 0:
            raise ValueError('Expected [batch,time,input_size] or [time,input_size]')
        if self.num_stations:
            if station_ids is None or station_ids.dtype != torch.long or station_ids.shape != (len(x),):
                raise ValueError('Regional models require one int64 station ID per sample')
            if validate_station_ids and ((station_ids < 0).any() or (station_ids >= self.num_stations).any()):
                raise ValueError('Station ID outside saved mapping')
        elif station_ids is not None:
            raise ValueError('Single-station model has no station embedding')
        return x

    def _persistence(self, x, station_ids):
        scale = self.persistence_scale if station_ids is None else self.persistence_scale[station_ids]
        offset = self.persistence_offset if station_ids is None else self.persistence_offset[station_ids]
        return x[:, -1, self.level_feature_index, None]*scale+offset

    def persistence(self, x, station_ids=None, *, _validate_station_ids=True):
        x = self._batch(x, station_ids, _validate_station_ids)
        return self._persistence(x, station_ids)

    def prepare_inputs(self, x):
        if self.late_fusion or self.forecast_history_policy != 'anchor_only' or not len(self._weather_indices):
            return x
        x = x.clone()
        x[:, :-1, self._weather_indices] = 0.
        x[:, :-1, self._mask_indices] = self._mask_values.to(x)
        return x

    def forward(self, x, station_ids=None, *, _validate_station_ids=True, return_diagnostics=False):
        x = self._batch(x, station_ids, _validate_station_ids)
        prepared = self.prepare_inputs(x)
        history = prepared.index_select(-1, self._history_indices)
        history = self.input_dropout_layer(history.transpose(1, 2)).transpose(1, 2)
        sequence = self.encoder.forward_sequence(history) if self.horizon_decoder_mode == 'transformer' else None
        parts = [sequence[:, -1] if sequence is not None else self.encoder(history)]
        if self.forecast_encoder is not None:
            parts.append(self.forecast_encoder(x[:, -1].index_select(-1, self._weather_indices)))
        if self.station_embedding is not None:
            parts.append(self.station_embedding(station_ids))
        features = self.fusion(torch.cat(parts, dim=-1))
        if sequence is not None:
            queries = features[:, None, :] + self.horizon_queries[None, :, :]
            features = self.horizon_decoder(queries, self.memory_projection(sequence))
        features = self.head_dropout_layer(features)
        output = self.linear(features, station_ids) if self.head_mode != 'shared' else self.linear(features)
        if self.event_aware:
            logits = self.rise_head(features)
            amplitude = F.softplus(self.amplitude_head(features))
            correction = torch.sigmoid(logits)*amplitude
            prediction = self._persistence(x, station_ids)+output+correction
            if return_diagnostics:
                return prediction, dict(rise_logits=logits, rise_probability=torch.sigmoid(logits),
                    base_delta_normalized=output, amplitude_normalized=amplitude,
                    correction_normalized=correction, final_delta_normalized=output+correction)
            return prediction
        if return_diagnostics:
            raise ValueError('Diagnostics require an event-aware model')
        if self.prediction_mode == 'direct':
            return output
        return self._persistence(x, station_ids)+output


class ResidualGRU(ResidualSequence):
    def __init__(self, **kwargs):
        super().__init__(architecture='gru', **kwargs)


class ResidualTCN(ResidualSequence):
    def __init__(self, **kwargs):
        super().__init__(architecture='tcn', **kwargs)


class ResidualCNNGRU(ResidualSequence):
    def __init__(self, **kwargs):
        super().__init__(architecture='cnn-gru', **kwargs)


class ResidualGRULateFusion(ResidualSequence):
    def __init__(self, **kwargs):
        super().__init__(architecture='gru-late-fusion', **kwargs)


class ResidualTCNLateFusion(ResidualSequence):
    def __init__(self, **kwargs):
        super().__init__(architecture='tcn-late-fusion', **kwargs)


class ResidualCNNGRULateFusion(ResidualSequence):
    def __init__(self, **kwargs):
        super().__init__(architecture='cnn-gru-late-fusion', **kwargs)


class ResidualTransformer(ResidualSequence):
    def __init__(self, **kwargs):
        super().__init__(architecture='transformer', **kwargs)


class ResidualTransformerLateFusion(ResidualSequence):
    def __init__(self, **kwargs):
        super().__init__(architecture='transformer-late-fusion', **kwargs)


def residual_model(method, **kwargs):
    classes = dict(zip(NEW_METHODS, [ResidualGRU, ResidualTCN, ResidualCNNGRU,
                    ResidualGRULateFusion, ResidualTCNLateFusion, ResidualCNNGRULateFusion,
                    ResidualTransformer, ResidualTransformerLateFusion]))
    if method == 'lstm':
        return ResidualSequence(architecture='lstm', **kwargs)
    if method not in classes:
        raise ValueError(f'Unknown residual architecture: {method}')
    return classes[method](**kwargs)
