"""Result caching orchestration; numerical pipeline source identity remains unchanged."""

from src.backend.prediction.inference import PredictionService
from src.backend.prediction.persistence import build_input_hash, load_forecast, save_forecast


class AnchorUnavailableError(ValueError):
    """The selected chain does not end at the exact hourly anchor requested."""


class CachedPredictionService(PredictionService):
    """Reuse Redis then PostgreSQL before inference for unchanged original inputs.

    Args:
        engine: Engine owned by the app, used for current inputs and atomic storage.
        runtime: Existing station runtime; numerical code and artifact hashes unchanged.
        result_cache: Optional committed-output cache, bypassed entirely for overrides.
        input_cache: Optional shared original-input cache used also by auto-fill.
    """

    def __init__(self, engine, runtime, result_cache, input_cache=None):
        super().__init__(engine, runtime)
        self.result_cache = result_cache
        self.input_cache = input_cache

    def predict(self, reference_time=None, overrides=None, *, required_anchor=None):
        """Prepare inputs, check cache/storage, then infer and atomically save a miss.

        Args:
            reference_time: Historical hour or None for latest.
            overrides: Last-step scenario edits; nonempty bypasses both cache and DB.
            required_anchor: Range mode requires this exact anchor, avoiding a shifted
                forecast being mislabeled as a different hour when observations gap.

        Returns:
            Forecast in mm with committed identity for original input, transient for edits.

        Raises:
            AnchorUnavailableError: Prepared anchor differs from required range hour.
            InsufficientHistoryError: The input chain is incomplete.
            ValueError, RuntimeError: Invalid input/output or inference execution.
            sqlalchemy.exc.SQLAlchemyError: DB read/commit failed; no cache is published.
        """
        if overrides:
            return super().predict(reference_time, overrides)
        prepared = (
            self.input_cache.prepare(self.inputs, reference_time)
            if self.input_cache is not None
            else self.inputs.prepare(reference_time)
        )
        if required_anchor is not None and prepared.anchor_observed_at != required_anchor:
            raise AnchorUnavailableError("Exact hourly anchor is unavailable.")
        config = self.runtime.config
        pipeline = self.runtime.pipeline_version
        input_hash = build_input_hash(prepared, config)
        cached = self.result_cache.read(config, prepared, pipeline, input_hash)
        if cached is not None:
            return cached
        with self.engine.connect() as connection:
            stored = load_forecast(connection, config, prepared, pipeline, input_hash)
        if stored is None:
            stored = save_forecast(
                self.engine, config, prepared, pipeline, input_hash, self.runtime.predict(prepared)
            )
        self.result_cache.store(input_hash, stored)
        return stored
