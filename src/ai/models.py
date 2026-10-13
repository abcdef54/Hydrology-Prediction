from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Protocol
import torch
import torch.nn as nn
import xgboost as xgb
import lightgbm as gbm
import pandas as pd
from googlehydrology.modelzoo.mean_embedding_forecast_lstm import MeanEmbeddingForecastLSTM
from src.ai.utils import select_feature_columns
from src.ai.lstm import LSTM, ResidualLSTM


class WaterLevelPredictionModel(Protocol):
    def __call__(self, *arg, **kwargs) -> torch.Tensor:
        ...


class MeanEmbeddingForecastLSTMWithAdapter(nn.Module):
    """Adapt observed history to a pretrained MEF recurrent backbone.

    This dataset has no future forecasts. The forecast LSTM therefore receives
    a zero forecast embedding plus the last hindcast state.
    """

    def __init__(
        self,
        pretrained_model: MeanEmbeddingForecastLSTM,
        local_hindcast_size: int,
        output_size: int,
        freeze_backbone: bool = True,
        input_dropout: float = 0.0,
        head_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        for name, rate in (("input_dropout", input_dropout), ("head_dropout", head_dropout)):
            if not 0.0 <= rate < 1.0:
                raise ValueError(f"{name} must be between 0 (inclusive) and 1 (exclusive)")

        self.input_dropout = input_dropout
        self.head_dropout = head_dropout
        self.input_dropout_layer = nn.Dropout1d(p=input_dropout)
        self.head_dropout_layer = nn.Dropout(p=head_dropout)
        self.backbone = pretrained_model
        self.hindcast_adapter = nn.Sequential(
            nn.Linear(local_hindcast_size, self.backbone.hindcast_lstm.input_size),
            nn.LayerNorm(self.backbone.hindcast_lstm.input_size),
        )
        hidden_size = self.backbone.hindcast_lstm.hidden_size
        self.forecast_embedding_size = self.backbone.forecast_lstm.input_size - hidden_size
        if self.forecast_embedding_size < 0:
            raise ValueError("MEF forecast input is smaller than the hindcast state")
        self.output_head = nn.Linear(hidden_size, output_size)

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.hindcast_adapter.parameters():
            param.requires_grad = True
        for param in self.output_head.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x_hindcast_local: torch.Tensor) -> torch.Tensor:
        if x_hindcast_local.dim() == 2:
            x_hindcast_local = x_hindcast_local.unsqueeze(1)
        # Drop whole feature channels across the history of each training sample.
        x_hindcast_local = self.input_dropout_layer(
            x_hindcast_local.transpose(1, 2)
        ).transpose(1, 2)
        x_hindcast_encoded = self.hindcast_adapter(x_hindcast_local)
        hindcast_output, (h_hind, c_hind) = self.backbone.hindcast_lstm(x_hindcast_encoded)
        no_future_forecast = hindcast_output.new_zeros(
            (len(x_hindcast_local), 1, self.forecast_embedding_size)
        )
        forecast_input = torch.cat(
            [no_future_forecast, hindcast_output[:, -1:, :]], dim=-1
        )
        forecast_output, _ = self.backbone.forecast_lstm(
            forecast_input,
            (h_hind, c_hind)
        )
        final_hidden_state = self.head_dropout_layer(forecast_output[:, -1, :])
        return self.output_head(final_hidden_state)



class AttentionLSTM1(LSTM):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.attention_score = nn.Linear(self.hidden_size, 1)        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.input_dropout_layer(x.transpose(1, 2)).transpose(1, 2)
        sequence_output, _ = self.lstm(x)

        attention_logits = self.attention_score(sequence_output)
        attention_weights = torch.softmax(attention_logits, dim=1)

        context = torch.sum(sequence_output * attention_weights, dim=1)
        out = self.linear(self.head_dropout_layer(context))

        return out


class AttentionLSTM2(ResidualLSTM):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attention_score = nn.Linear(self.hidden_size, 1)        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        persistence = self.persistence(x)
        x = self.input_dropout_layer(x.transpose(1, 2)).transpose(1, 2)
        sequence_output, _ = self.lstm(x)

        attention_logits = self.attention_score(sequence_output)
        attention_weights = torch.softmax(attention_logits, dim=1)

        context = torch.sum(sequence_output * attention_weights, dim=1)
        out = self.linear(self.head_dropout_layer(context))

        return persistence + out



class AttentionLSTM3(AttentionLSTM2):
    def __init__(
        self,
        head_dim: int,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.head_dim = head_dim

        self.horizon_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size, self.head_dim),
                nn.ReLU(),
                self.head_dropout_layer,
                nn.Linear(self.head_dim, 1)
            )
            for _ in range(self.output_size)
        ])

        for head in self.horizon_heads:
            output_layer = head[-1]
            nn.init.zeros_(output_layer.weight)
            nn.init.zeros_(output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        persistence = self.persistence(x)
        x = self.input_dropout_layer(x.transpose(1, 2)).transpose(1, 2)
        sequence_output, _ = self.lstm(x)
        
        attention_logits = self.attention_score(sequence_output)
        attention_weights = torch.softmax(attention_logits, dim=1)

        context = torch.sum(sequence_output * attention_weights, dim=1)
        out = torch.cat(
            [head(context) for head in self.horizon_heads],
            dim=-1
        )
        return persistence + out        


@dataclass(frozen=True)
class XGBoostSettings:
    n_estimators: int = 2000
    learning_rate: float = 0.5
    max_depth: int = 9
    min_child_weight: float = 5.0
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_alpha: float = 1.0
    reg_lambda: float = 1.5
    early_stopping_rounds: int = 100
    n_jobs: int = -1


@dataclass(frozen=True)
class LightGBMSettings:
    n_estimators: int = 2000
    learning_rate: float = 0.05
    num_leaves: int = 31
    max_depth: int = 6
    subsample: float = 0.75
    subsample_freq: int = 1
    colsample_bytree: float = 0.75
    reg_alpha: float = 0.5
    reg_lambda: float = 1.5
    early_stopping_rounds: int = 150
    n_jobs: int = -1
    verbose: int = 1


def tree_feature_columns(train_dataset: pd.DataFrame, feature_set: str) -> list[str]:
    selected = select_feature_columns(train_dataset.columns, feature_set)
    if selected is not None:
        return selected
    return [
        column for column in train_dataset.columns
        if not column.startswith("target_") and not column.startswith("timestamp")
    ]


class XGBoost:
    def __init__(self, settings: XGBoostSettings | None = None, feature_set: str = "all") -> None:
        self.settings = settings or XGBoostSettings()
        self.feature_set = feature_set
        self.models: dict[str, xgb.XGBRegressor] = {}
        self.feature_columns: list[str] = []
        self.target_columns: list[str] = []

    def train(self, train_dataset: pd.DataFrame, eval_dataset: pd.DataFrame, random_state: int = 42) -> XGBoost:
        self.target_columns = [col for col in train_dataset.columns if col.startswith("target_")]
        self.feature_columns = tree_feature_columns(train_dataset, self.feature_set)
        print(f"Training XGBoost: {self.feature_set} feature set, {len(self.feature_columns)} inputs")
        print(f"XGBoost settings: {self.settings}")
        X = train_dataset[self.feature_columns]
        y = train_dataset[self.target_columns]
        model_parameters = asdict(self.settings)
        stopping_rounds = model_parameters.pop("early_stopping_rounds")
        model_parameters["early_stopping_rounds"] = stopping_rounds or None
        for target_column in self.target_columns:
            model = xgb.XGBRegressor(
                objective="reg:squarederror",
                random_state=random_state,
                **model_parameters,
            )
            print(f"Fitting {target_column}")
            model.fit(
                X,
                y[target_column],
                eval_set=[(eval_dataset[self.feature_columns], eval_dataset[target_column])],
                verbose=False,
            )
            self.models[target_column] = model
        return self
    
    def predict(self, test_dataset: pd.DataFrame) -> pd.DataFrame:
        X_test = test_dataset[self.feature_columns]
        result_df = test_dataset.copy()
        for target_column in self.target_columns:
            prediction = self.models[target_column].predict(X_test)
            result_df.loc[:, target_column] = prediction
        return result_df


class LightGBM:
    def __init__(self, settings: LightGBMSettings | None = None, feature_set: str = "all") -> None:
        self.settings = settings or LightGBMSettings()
        self.feature_set = feature_set
        self.models: dict[str, gbm.LGBMRegressor] = {}
        self.target_columns: list[str] = []
        self.feature_columns: list[str] = []
    
    def train(self, train_dataset: pd.DataFrame, eval_dataset: pd.DataFrame, random_state: int = 42) -> LightGBM:
        self.target_columns = [col for col in train_dataset.columns if col.startswith("target_")]
        self.feature_columns = tree_feature_columns(train_dataset, self.feature_set)
        print(f"Training LightGBM: {self.feature_set} feature set, {len(self.feature_columns)} inputs")
        print(f"LightGBM settings: {self.settings}")
        X = train_dataset[self.feature_columns]
        y = train_dataset[self.target_columns]
        model_parameters = asdict(self.settings)
        stopping_rounds = model_parameters.pop("early_stopping_rounds")
        for target_column in self.target_columns:
            callbacks = [gbm.early_stopping(stopping_rounds, verbose=False)] if stopping_rounds else []
            model = gbm.LGBMRegressor(
                objective='regression',
                random_state=random_state,
                **model_parameters,
            )
            print(f"Fitting {target_column}")
            model.fit(
                X, y[target_column],
                eval_set=[(eval_dataset[self.feature_columns], eval_dataset[target_column])],
                callbacks=callbacks,
            )
            self.models[target_column] = model
        return self
    
    def predict(self, test_dataset: pd.DataFrame) -> pd.DataFrame:
        X_test = test_dataset[self.feature_columns]
        result_df = test_dataset.copy()
        for target_column in self.target_columns:
            prediction = self.models[target_column].predict(X_test)
            result_df.loc[:, target_column] = prediction
        return result_df
