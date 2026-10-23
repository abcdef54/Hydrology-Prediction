"""Shared PyTorch LSTM architectures for training and inference."""

import torch
from torch import nn


class LSTM(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 3,
        dropout: float = 0.2,
        input_dropout: float = 0.0,
        head_dropout: float = 0.0,
    ):
        super().__init__()

        for name, rate in (
            ("dropout", dropout),
            ("input_dropout", input_dropout),
            ("head_dropout", head_dropout),
        ):
            if not 0.0 <= rate < 1.0:
                raise ValueError(f"{name} must be between 0 (inclusive) and 1 (exclusive)")

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.head_dropout = head_dropout
        # Drop whole input channels for each sequence during training.
        self.input_dropout_layer = nn.Dropout1d(p=input_dropout)
        self.head_dropout_layer = nn.Dropout(p=head_dropout)

        self.lstm = torch.nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True,
        )
        self.linear = torch.nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.input_dropout_layer(x.transpose(1, 2)).transpose(1, 2)
        sequence_output, _ = self.lstm(x)

        final_hidden_state = sequence_output[:, -1, :]

        predictions = self.linear(self.head_dropout_layer(final_hidden_state))
        return predictions


class ResidualLSTM(LSTM):
    """Forecast a correction to the current water level in target-scaled units."""

    def __init__(
        self,
        *args,
        level_feature_index: int,
        persistence_scale: list[float],
        persistence_offset: list[float],
        forecast_history_policy: str = 'all_timesteps',
        weather_feature_indices: list[int] | None = None,
        weather_mask_indices: list[int] | None = None,
        weather_mask_values: list[float] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not 0 <= level_feature_index < self.input_size:
            raise ValueError("level_feature_index is outside the input features")
        if len(persistence_scale) != self.output_size or len(persistence_offset) != self.output_size:
            raise ValueError("persistence coefficients must match output_size")
        self.level_feature_index = level_feature_index
        self.register_buffer("persistence_scale", torch.tensor(persistence_scale, dtype=torch.float32))
        self.register_buffer("persistence_offset", torch.tensor(persistence_offset, dtype=torch.float32))
        if forecast_history_policy not in ('all_timesteps', 'anchor_only'):
            raise ValueError('Unknown forecast history policy')
        self.forecast_history_policy = forecast_history_policy
        weather = list(weather_feature_indices or [])
        masks = list(weather_mask_indices or [])
        values = list(weather_mask_values or [])
        if (len(set(weather)) != len(weather) or len(set(masks)) != len(masks)
                or any(i < 0 or i >= self.input_size for i in weather + masks)
                or not set(masks).issubset(weather) or len(masks) != len(values)
                or self.level_feature_index in weather):
            raise ValueError('Invalid weather policy feature mapping')
        if forecast_history_policy == 'anchor_only' and (not weather or not masks):
            raise ValueError('Anchor-only policy requires weather features and explicit masks')
        if not torch.isfinite(torch.tensor(values)).all():
            raise ValueError('Weather mask values must be finite')
        # Derived from saved feature order/scaler; no learned weights or legacy
        # state_dict keys are added. Reload requires the preprocessing policy.
        self.register_buffer('_weather_indices', torch.tensor(weather, dtype=torch.long), persistent=False)
        self.register_buffer('_weather_mask_indices', torch.tensor(masks, dtype=torch.long), persistent=False)
        self.register_buffer('_weather_mask_values', torch.tensor(values, dtype=torch.float32), persistent=False)
        # The untrained model is exactly the persistence forecast.
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def persistence(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        current_level = x[:, -1, self.level_feature_index].unsqueeze(-1)
        return current_level * self.persistence_scale + self.persistence_offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prepare_inputs(x)
        return self.persistence(x) + super().forward(x)

    def prepare_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """Withhold past weather per sequence, preserving current weather and observations."""
        if getattr(self, 'forecast_history_policy', 'all_timesteps') == 'all_timesteps':
            return x
        if x.dim() not in (2, 3):
            raise ValueError('Expected [time,features] or [batch,time,features]')
        two_dimensional = x.dim() == 2
        result = (x.unsqueeze(0) if two_dimensional else x).clone()
        # Zero here is a neutral standardized placeholder, never raw zero rain:
        # the existing forecast masks explicitly mark these inputs unavailable.
        result[:, :-1, self._weather_indices] = 0.
        result[:, :-1, self._weather_mask_indices] = self._weather_mask_values.to(result)
        return result.squeeze(0) if two_dimensional else result


class RegionalResidualLSTM(ResidualLSTM):
    def __init__(
            self,
            *args,
            num_stations: int,
            dynamic_input_size: int,
            station_embedding_size: int = 8,
            **kwargs
    ) -> None:
        super().__init__(input_size=dynamic_input_size+station_embedding_size, *args, **kwargs)
        self.station_embedding = nn.Embedding(num_embeddings=num_stations, embedding_dim=station_embedding_size)

    def forward(self, x: torch.Tensor, station_ids: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Expected input shape [batch, time, feature]")

        if station_ids.shape != x.shape[0]:
            raise ValueError("Expected one station ID per sample")

        station_features = self.station_embedding(station_ids)
        repeated_station_features = station_features.unsqueeze(1).expand(-1, x.shape[1], -1)
        conditioned_inputs = torch.cat([x, repeated_station_features], dim=-1)

        return super().forward(conditioned_inputs)