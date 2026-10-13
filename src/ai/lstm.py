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
        # The untrained model is exactly the persistence forecast.
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def persistence(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        current_level = x[:, -1, self.level_feature_index].unsqueeze(-1)
        return current_level * self.persistence_scale + self.persistence_offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.persistence(x) + super().forward(x)


