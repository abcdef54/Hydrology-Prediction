"""Deterministic forecast identities and atomic PostgreSQL storage for original inputs."""

from dataclasses import asdict, replace
from datetime import timedelta
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Connection, Engine

from src.backend.db.models import Forecast, ForecastPoint as StoredForecastPoint
from src.backend.prediction.config import StationModelConfig
from src.backend.prediction.inputs import PreparedInput
from src.backend.prediction.results import ForecastPoint, ForecastResult


def build_pipeline_version(
    config: StationModelConfig, artifact_hashes: tuple[str, str], device: str
) -> str:
    """Identify loaded checkpoint/scaler bytes, configuration and serving source code.

    Paths are excluded so relocating identical artifacts does not change identity.
    Deploy code changes with a worker restart, as for any Python application.

    Args:
        config: Station-specific model, feature, calendar and horizon settings.
        artifact_hashes: SHA-256 digests of the loaded checkpoint and preprocessing
            file, in that order. These must describe the bytes used by the runtime.
        device: Runtime device identifier, such as "cpu" or "cuda:0".

    Returns:
        A 64-character lowercase SHA-256 digest identifying the serving pipeline.

    Raises:
        OSError: If a pipeline source file cannot be read.
    """
    configuration = asdict(config)
    configuration.pop("model_path")
    configuration.pop("preprocessing_path")
    source_names = (
        "features.py",
        "inputs.py",
        "preprocessing.py",
        "inference.py",
        "persistence.py",
        "results.py",
    )
    source_hashes = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in source_names
    }
    contract = {
        "identity_version": 1,
        "config": configuration,
        "artifacts": artifact_hashes,
        "source": source_hashes,
        "device": device,
    }
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()


def build_input_hash(prepared: PreparedInput, config: StationModelConfig) -> str:
    """Hash every original-unit feature, scaled tensor and timestep in model order.

    Missing values have canonical bytes and a separate mask. Request time is
    excluded: different requests selecting exactly the same inputs may reuse a
    forecast, while their response context and freshness must remain separate.

    Args:
        prepared: Selected input chain, including original-unit features, scaled
            tensor, timezone-aware timestamps and the actual forecast anchor.
        config: Model contract defining the station, sequence length, feature
            order and forecast horizons expected for this input.

    Returns:
        A 64-character lowercase SHA-256 digest of the complete input chain.
        Neither prepared features nor the tensor are modified.

    Raises:
        ValueError: If input identity, timestamps, shape or numeric values do not
            match the model contract. Missing raw values are allowed.
    """
    if (
        prepared.station_id != config.station_id
        or prepared.model_version != config.model_version
        or prepared.feature_schema_version != config.feature_schema_version
        or tuple(prepared.raw_features.columns) != config.feature_columns
        or len(prepared.timestamps) != config.sequence_length
        or not prepared.raw_features.index.equals(prepared.timestamps)
        or prepared.timestamps.tz is None
        or prepared.timestamps[-1].to_pydatetime() != prepared.anchor_observed_at
    ):
        raise ValueError("Prepared input does not match the forecast identity contract.")
    raw = prepared.raw_features.to_numpy(dtype="<f8", copy=True)
    tensor = prepared.tensor.detach().cpu().numpy().astype("<f4", copy=False)
    if raw.shape != (config.sequence_length, len(config.feature_columns)) or tensor.shape != (
        1,
        *raw.shape,
    ):
        raise ValueError("Prepared input has an invalid shape.")
    if np.isinf(raw).any() or not np.isfinite(tensor).all():
        raise ValueError("Prepared input contains invalid values.")
    missing = np.isnan(raw)
    raw[missing] = 0
    header = {
        "version": 1,
        "station": config.station_id,
        "model": config.model_version,
        "schema": config.feature_schema_version,
        "columns": config.feature_columns,
        "shape": raw.shape,
        "horizons": config.forecast_horizons_minutes,
        "anchor": prepared.timestamps[-1].tz_convert("UTC").isoformat(),
    }
    digest = hashlib.sha256(json.dumps(header, sort_keys=True).encode())
    for values in (
        prepared.timestamps.tz_convert("UTC").as_unit("ns").asi8.astype("<i8"),
        missing,
        raw,
        tensor,
    ):
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def load_forecast(
    connection: Connection,
    config: StationModelConfig,
    prepared: PreparedInput,
    pipeline_version: str,
    input_hash: str,
) -> ForecastResult | None:
    """Read a complete matching forecast, rebuilding context from this request.

    Corrupt/incomplete stored horizons raise ValueError instead of being returned
    as successful forecasts. No database rows are changed by this function.

    Args:
        connection: Open SQLAlchemy connection; its lifecycle belongs to the caller.
        config: Station/model contract and the required forecast horizons.
        prepared: Current request's input chain, anchor and reference-time context.
        pipeline_version: Digest identifying the currently loaded serving pipeline.
        input_hash: Digest returned by build_input_hash for this chain and config.

    Returns:
        Matching forecast in millimetres with reused=True and the current request
        times, or None when no matching parent exists. Original ID and generation
        time are preserved; freshness is calculated separately by the API.

    Raises:
        ValueError: If a matching stored forecast has an incorrect anchor, missing
            or unexpected horizons, incorrect forecast times or nonfinite values.
        sqlalchemy.exc.SQLAlchemyError: If a database read fails.
    """
    parent = (
        connection.execute(
            sa.select(Forecast.__table__).where(
                Forecast.station_id == config.station_id,
                Forecast.model_version == config.model_version,
                Forecast.feature_schema_version == config.feature_schema_version,
                Forecast.pipeline_version == pipeline_version,
                Forecast.input_hash == input_hash,
            )
        )
        .mappings()
        .first()
    )
    if parent is None:
        return None
    points = (
        connection.execute(
            sa.select(StoredForecastPoint.__table__).where(
                StoredForecastPoint.forecast_id == parent["id"]
            )
        )
        .mappings()
        .all()
    )
    by_horizon = {point["horizon_minutes"]: point for point in points}
    if parent["anchor_observed_at"] != prepared.anchor_observed_at or set(by_horizon) != set(
        config.forecast_horizons_minutes
    ):
        raise ValueError("Stored forecast is incomplete or has an incorrect anchor.")
    predictions = []
    for horizon in config.forecast_horizons_minutes:
        point = by_horizon[horizon]
        expected_time = prepared.anchor_observed_at + timedelta(minutes=horizon)
        if point["forecast_at"] != expected_time or not np.isfinite(point["water_level_mm"]):
            raise ValueError("Stored forecast contains an invalid horizon point.")
        predictions.append(ForecastPoint(horizon, expected_time, point["water_level_mm"]))
    return ForecastResult(
        station_id=config.station_id,
        model_version=config.model_version,
        feature_schema_version=config.feature_schema_version,
        requested_reference_time=prepared.requested_reference_time,
        hourly_reference_time=prepared.hourly_reference_time,
        anchor_observed_at=prepared.anchor_observed_at,
        predictions=tuple(predictions),
        forecast_id=parent["id"],
        generated_at=parent["generated_at"],
        pipeline_version=pipeline_version,
        reused=True,
    )


def save_forecast(
    engine: Engine,
    config: StationModelConfig,
    prepared: PreparedInput,
    pipeline_version: str,
    input_hash: str,
    forecast: ForecastResult,
) -> ForecastResult:
    """Commit a parent and all horizons together, or reuse a concurrent winner.

    Database failures propagate to the API's 503 handler. Inference happens
    before this transaction, so no DB lock is held while the model runs.

    Args:
        engine: PostgreSQL engine used to open and commit the storage transaction.
        config: Station/model contract defining the output horizon order.
        prepared: Original observation inputs used to generate this forecast;
            callers must exclude requests containing user overrides.
        pipeline_version: Digest identifying the loaded model, scaler and pipeline.
        input_hash: Digest of the full prepared input chain for this config.
        forecast: Inference result in millimetres, with times based on the anchor.

    Returns:
        Committed forecast with its ID, generation time, pipeline version and
        reused=False; or the existing winner with reused=True on an identity
        conflict. Existing stored predictions are never overwritten.

    Raises:
        ValueError: If the result violates the model contract or a conflicting
            stored result cannot be read and validated.
        sqlalchemy.exc.SQLAlchemyError: If storage or commit fails. The new parent
            and all new horizon points are rolled back together.
    """
    if (
        forecast.station_id != config.station_id
        or forecast.model_version != config.model_version
        or forecast.feature_schema_version != config.feature_schema_version
        or forecast.anchor_observed_at != prepared.anchor_observed_at
        or tuple(point.horizon_minutes for point in forecast.predictions)
        != config.forecast_horizons_minutes
        or any(
            point.forecast_at
            != prepared.anchor_observed_at + timedelta(minutes=point.horizon_minutes)
            or not np.isfinite(point.water_level_mm)
            for point in forecast.predictions
        )
    ):
        raise ValueError("Model result does not match the forecast storage contract.")
    with engine.begin() as connection:
        forecast_id = uuid4()
        statement = (
            insert(Forecast)
            .values(
                id=forecast_id,
                station_id=config.station_id,
                model_version=config.model_version,
                feature_schema_version=config.feature_schema_version,
                pipeline_version=pipeline_version,
                input_hash=input_hash,
                anchor_observed_at=prepared.anchor_observed_at,
            )
            .on_conflict_do_nothing(constraint="uq_forecasts_input_identity")
            .returning(Forecast.generated_at)
        )
        generated_at = connection.scalar(statement)
        if generated_at is None:
            # Under READ COMMITTED the unique conflict waits for the winner's
            # complete transaction; the next SELECT sees its committed horizons.
            stored = load_forecast(connection, config, prepared, pipeline_version, input_hash)
            if stored is None:
                raise ValueError("Concurrent forecast could not be read.")
            return stored
        connection.execute(
            sa.insert(StoredForecastPoint),
            [
                {
                    "forecast_id": forecast_id,
                    "horizon_minutes": point.horizon_minutes,
                    "forecast_at": point.forecast_at,
                    "water_level_mm": point.water_level_mm,
                }
                for point in forecast.predictions
            ],
        )
    return replace(
        forecast,
        forecast_id=forecast_id,
        generated_at=generated_at,
        pipeline_version=pipeline_version,
        reused=False,
    )
