from __future__ import annotations
import torch
import torch.nn as nn
import xgboost as xgb
import lightgbm as gbm
import pandas as pd
from googlehydrology.modelzoo.mean_embedding_forecast_lstm import MeanEmbeddingForecastLSTM


class MeanEmbeddingForecastLSTMWithAdapter(nn.Module):
    def __init__(
        self,
        pretrained_model: MeanEmbeddingForecastLSTM,
        local_hindcast_size: int,
        pretrained_hindcast_size: int,
        local_forecast_size: int,
        pretrained_forecast_size: int,
        output_size: int,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        
        self.backbone = pretrained_model

        self.hindcast_adapter = nn.Sequential(
            nn.Linear(local_hindcast_size, pretrained_hindcast_size),
            nn.LayerNorm(pretrained_hindcast_size)
        )
        
        self.forecast_adapter = nn.Sequential(
            nn.Linear(local_forecast_size, pretrained_forecast_size),
            nn.LayerNorm(pretrained_forecast_size)
        )

        hidden_size = self.backbone.config_data.hidden_size

        self.output_head = nn.Linear(hidden_size, output_size)

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        for param in self.hindcast_adapter.parameters():
            param.requires_grad = True
        
        for param in self.forecast_adapter.parameters():
            param.requires_grad = True
        
        for param in self.output_head.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(
        self, 
        x_hindcast_local: torch.Tensor,
        x_forecast_local: torch.Tensor,
    ) -> torch.Tensor:
        # Ensure 3D shape (batch_size, seq_len, num_features)
        if x_hindcast_local.dim() == 2:
            x_hindcast_local = x_hindcast_local.unsqueeze(1)
        if x_forecast_local.dim() == 2:
            x_forecast_local = x_forecast_local.unsqueeze(1)

        # 1. Project local features to backbone input dimension
        x_hindcast_encoded = self.hindcast_adapter(x_hindcast_local)
        x_forecast_encoded = self.forecast_adapter(x_forecast_local)

        # 2. Process hindcast sequence via pretrained hindcast LSTM
        hindcast_output, (h_hind, c_hind) = self.backbone.hindcast_lstm(x_hindcast_encoded)

        # 3. Align forecast input dimensions with backbone forecast LSTM
        # If backbone expects hindcast context concatenated along feature axis:
        if x_forecast_encoded.shape[-1] == self.backbone.forecast_lstm.input_size:
            forecast_input = x_forecast_encoded
        elif x_forecast_encoded.shape[-1] + self.backbone.config_data.hidden_size == self.backbone.forecast_lstm.input_size:
            hindcast_context = hindcast_output[:, -1:, :].expand(-1, x_forecast_encoded.shape[1], -1)
            forecast_input = torch.cat([x_forecast_encoded, hindcast_context], dim=-1)
        else:
            forecast_input = x_forecast_encoded

        # 4. Process forecast sequence initialized with hindcast hidden state
        forecast_output, _ = self.backbone.forecast_lstm(
            forecast_input,
            (h_hind, c_hind)
        )

        # 5. Map final forecast representation to downstream targets
        final_representation = forecast_output[:, -1, :]
        predictions = self.output_head(final_representation)
        return predictions



class LSTM(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        sequence_output, _ = self.lstm(x)

        final_hidden_state = sequence_output[:, -1, :]

        predictions = self.linear(final_hidden_state)
        return predictions

class XGBoost:
    def __init__(self) ->  None:
        self.models: dict[str, xgb.XGBRegressor] = {}
        self.feature_columns: list[str] = None
        self.target_columns: list[str] = None

    def train(self, train_dataset: pd.DataFrame, eval_dataset: pd.DataFrame, random_state: int = 42) -> XGBoost:
        self.target_columns = [col for col in train_dataset.columns if col.startswith("target_")]
        self.feature_columns = [
            col for col in train_dataset.columns
            if col not in self.target_columns and not col.startswith("timestamp")
        ]
        X = train_dataset[self.feature_columns]
        y = train_dataset[self.target_columns]
        for target_column in self.target_columns:
            model = xgb.XGBRegressor(
                objective="reg:squarederror",

                n_estimators=2000,
                learning_rate=0.5,

                max_depth=9,
                min_child_weight=5,

                subsample=0.9,
                colsample_bytree=0.9,

                reg_alpha=1.0,
                reg_lambda=1.5,

                early_stopping_rounds=100,

                n_jobs=-1,
                random_state=random_state,
            )
            model.fit(
                X, 
                y[target_column], 
                eval_set=[(eval_dataset[self.feature_columns], eval_dataset[target_column])]
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
    def __init__(self) -> None:
        self.models: dict[str, gbm.LGBMRegressor] = {}
        self.target_columns: list[str] = None
        self.feature_columns: list[str] = None
    
    def train(self, train_dataset: pd.DataFrame, eval_dataset: pd.DataFrame, random_state: int = 42) -> LightGBM:
        self.target_columns = [col for col in train_dataset.columns if col.startswith("target_")]
        self.feature_columns = [
            col for col in train_dataset.columns
            if col not in self.target_columns and not col.startswith("timestamp")
        ]
        X = train_dataset[self.feature_columns]
        y = train_dataset[self.target_columns]
        
        for target_column in self.target_columns:
            model = gbm.LGBMRegressor(
                objective='regression',
                n_estimators=2000,
                learning_rate=0.05,
                subsample=0.75,
                subsample_freq=1,
                colsample_bytree=0.75,
                reg_alpha=0.5,
                reg_lambda=1.5,
                num_leaves=31,
                max_depth=6,
                verbose=1,
                early_stopping_rounds=150,
                n_jobs=-1,
                random_state=random_state,
            )
            model.fit(
                X, y[target_column],
                eval_set=[(eval_dataset[self.feature_columns], eval_dataset[target_column])],
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