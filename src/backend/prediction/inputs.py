"""Read-only PostgreSQL service for the Residual LSTM's historical inputs."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.engine import Connection, Engine
import torch

from src.backend.db.models import Observation
from src.backend.prediction.features import (
    build_hourly_features,
    utc_timestamp,
)
from src.backend.prediction.preprocessing import InputPreprocessor


class InsufficientHistoryError(ValueError):
    def __init__(self, available: int, required: int):
        self.available = available
        self.required = required
        super().__init__(
            f"INSUFFICIENT_HISTORY: {available} valid hourly observations; {required} required."
        )


@dataclass(frozen=True)
class PreparedInput:
    station_id: str
    model_version: str
    feature_schema_version: str
    tensor: torch.Tensor
    raw_features: pd.DataFrame
    timestamps: pd.DatetimeIndex
    requested_reference_time: datetime | None
    hourly_reference_time: datetime
    anchor_observed_at: datetime


class PredictionInputService:
    def __init__(self, engine: Engine, preprocessor: InputPreprocessor):
        self.engine = engine
        self.preprocessor = preprocessor
        self.config = preprocessor.config

    def prepare(
        self,
        reference_time: datetime | None = None,
        overrides: Mapping[str, float | None] | None = None,
    ) -> PreparedInput:
        requested = utc_timestamp(reference_time) if reference_time is not None else None
        # Keep all expanding history queries on one consistent, read-only snapshot.
        with self.engine.connect().execution_options(
            isolation_level="REPEATABLE READ"
        ) as connection:
            with connection.begin():
                connection.execute(sa.text("SET TRANSACTION READ ONLY"))
                reference = self._resolve_reference(connection, requested)
                selected = self._select_history(connection, reference)
        prepared, tensor = self.preprocessor.prepare_features(selected, overrides)
        return PreparedInput(
            station_id=self.config.station_id,
            model_version=self.config.model_version,
            feature_schema_version=self.config.feature_schema_version,
            tensor=tensor,
            raw_features=prepared,
            timestamps=prepared.index.copy(),
            requested_reference_time=requested.to_pydatetime() if requested is not None else None,
            hourly_reference_time=reference.to_pydatetime(),
            anchor_observed_at=prepared.index[-1].to_pydatetime(),
        )

    def _resolve_reference(
        self, connection: Connection, requested: pd.Timestamp | None
    ) -> pd.Timestamp:
        if requested is not None:
            return requested.floor("h")
        latest = connection.scalar(
            sa.select(sa.func.max(Observation.observed_at)).where(
                Observation.station_id == self.config.station_id,
                Observation.depth.is_not(None),
                Observation.observed_at <= datetime.now(timezone.utc),
            )
        )
        if latest is None:
            raise InsufficientHistoryError(0, self.preprocessor.sequence_length)
        return utc_timestamp(latest).floor("h")

    def _select_history(self, connection: Connection, reference: pd.Timestamp) -> pd.DataFrame:
        earliest = connection.scalar(
            sa.select(sa.func.min(Observation.observed_at)).where(
                Observation.station_id == self.config.station_id,
                Observation.depth.is_not(None),
                Observation.observed_at <= reference.to_pydatetime(),
            )
        )
        if earliest is None:
            raise InsufficientHistoryError(0, self.preprocessor.sequence_length)
        # Include this model's rainfall windows and physical-time water lags.
        earliest_required = utc_timestamp(earliest).floor("h") - pd.Timedelta(
            hours=self.config.warmup_hours
        )
        history_hours = self.preprocessor.sequence_length + self.config.warmup_hours
        while True:
            start = reference - pd.Timedelta(hours=history_hours)
            query = (
                sa.select(Observation.station_id, Observation.observed_at, Observation.depth)
                .where(
                    Observation.station_id.in_(
                        (self.config.station_id, *self.config.rain_station_ids)
                    ),
                    Observation.observed_at > start.to_pydatetime(),
                    Observation.observed_at <= reference.to_pydatetime(),
                )
                .order_by(Observation.observed_at)
            )
            observations = pd.read_sql(query, connection)
            features = build_hourly_features(observations, reference.to_pydatetime(), self.config)
            # Valid serving timesteps need observed water. Missing rain/lag features
            # remain allowed and use the saved training imputation after scaling.
            valid = features.loc[np.isfinite(features["water_level_depth"])]
            if len(valid) >= self.preprocessor.sequence_length:
                selected = valid.tail(self.preprocessor.sequence_length).copy()
                # Gaps can move the first selected timestep closer to the query
                # boundary. Include enough rain history before accepting it.
                warmup_start = selected.index[0] - pd.Timedelta(hours=self.config.warmup_hours)
                if start <= warmup_start or start <= earliest_required:
                    return selected
            if start <= earliest_required:
                raise InsufficientHistoryError(len(valid), self.preprocessor.sequence_length)
            history_hours *= 2
