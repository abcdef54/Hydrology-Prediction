"""Raw observations; no hourly aggregation or model preprocessing here."""

from datetime import datetime
from uuid import UUID, uuid4

import sqlalchemy as sa
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    metadata = sa.MetaData(
        schema="hydrology",
        naming_convention={
            "pk": "pk_%(table_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
        },
    )


class Station(Base):
    """Store one station's metadata and source identity, not model weights.

    Attributes:
        id: Stable string ID preserving leading zeros.
        station_type: water_level or rainfall, determining the source collection.
        measurement_unit: Unit of raw depth; water is mm and unconfirmed rain units
            remain None. Alarm thresholds are stored separately in metres.
        latitude/longitude: Nullable coordinates in decimal degrees.
        alarm_level_1_m/alarm_level_2_m/alarm_level_3_m: Nullable stored thresholds.
        enabled: Whether the station appears in the public enabled-station list.
        prediction_enabled: Database permission to predict; a separate model registry
            entry and valid history are also needed at serving time.
        source_name/source_station_id/observation_collection: Upstream identity used
            to map observations to the correct source station and collection.

    The schema enforces source mapping and valid metadata; SQLAlchemy sessions
    and migrations manage persistence rather than the class constructor.
    """
    __tablename__ = "stations"
    __table_args__ = (
        sa.CheckConstraint("btrim(id) <> ''", name="id_not_blank"),
        sa.CheckConstraint("btrim(name) <> ''", name="name_not_blank"),
        sa.CheckConstraint("station_type IN ('water_level', 'rainfall')", name="station_type"),
        sa.CheckConstraint("latitude BETWEEN -90 AND 90", name="latitude"),
        sa.CheckConstraint("longitude BETWEEN -180 AND 180", name="longitude"),
        sa.CheckConstraint(
            "alarm_level_1_m >= 0 AND alarm_level_1_m < 'Infinity'::float8", name="alarm_1_finite"
        ),
        sa.CheckConstraint(
            "alarm_level_2_m >= 0 AND alarm_level_2_m < 'Infinity'::float8", name="alarm_2_finite"
        ),
        sa.CheckConstraint(
            "alarm_level_3_m >= 0 AND alarm_level_3_m < 'Infinity'::float8", name="alarm_3_finite"
        ),
        sa.CheckConstraint("alarm_level_1_m < alarm_level_2_m", name="alarm_1_before_2"),
        sa.CheckConstraint("alarm_level_2_m < alarm_level_3_m", name="alarm_2_before_3"),
        sa.CheckConstraint("alarm_level_1_m < alarm_level_3_m", name="alarm_1_before_3"),
        sa.CheckConstraint(
            "NOT prediction_enabled OR station_type = 'water_level'", name="prediction_water_only"
        ),
        sa.CheckConstraint("btrim(source_name) <> ''", name="source_not_blank"),
        sa.CheckConstraint("btrim(source_station_id) <> ''", name="source_station_not_blank"),
        sa.CheckConstraint(
            "measurement_unit IS NULL OR btrim(measurement_unit) <> ''", name="unit_not_blank"
        ),
        sa.CheckConstraint(
            "(station_type = 'water_level' AND observation_collection = 'water_level_depth') "
            "OR (station_type = 'rainfall' AND observation_collection = 'rain_water_depth')",
            name="collection_matches_type",
        ),
        sa.UniqueConstraint(
            "source_name",
            "observation_collection",
            "source_station_id",
            name="uq_stations_source_station",
        ),
        sa.UniqueConstraint(
            "id", "source_name", "observation_collection", name="uq_stations_source_mapping"
        ),
    )
    id: Mapped[str] = mapped_column(sa.Text, primary_key=True)
    name: Mapped[str] = mapped_column(sa.Text)
    station_type: Mapped[str] = mapped_column(sa.Text)
    river_name: Mapped[str | None] = mapped_column(sa.Text)
    address: Mapped[str | None] = mapped_column(sa.Text)
    latitude: Mapped[float | None] = mapped_column(sa.Double)
    longitude: Mapped[float | None] = mapped_column(sa.Double)
    alarm_level_1_m: Mapped[float | None] = mapped_column(sa.Double)
    alarm_level_2_m: Mapped[float | None] = mapped_column(sa.Double)
    alarm_level_3_m: Mapped[float | None] = mapped_column(sa.Double)
    enabled: Mapped[bool] = mapped_column(server_default=sa.true())
    prediction_enabled: Mapped[bool] = mapped_column(server_default=sa.false())
    source_name: Mapped[str] = mapped_column(sa.Text)
    source_station_id: Mapped[str] = mapped_column(sa.Text)
    observation_collection: Mapped[str] = mapped_column(sa.Text)
    measurement_unit: Mapped[str | None] = mapped_column(
        sa.Text, comment="Water mm confirmed by project owner; rain unit unconfirmed (NULL)."
    )
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), server_default=sa.func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), server_default=sa.func.now(), server_onupdate=sa.FetchedValue()
    )


class Observation(Base):
    """Store a raw reading at an absolute instant before model preprocessing.

    Attributes:
        station_id/observed_at: Composite key identifying a station's reading.
            observed_at is timezone-aware; connections use UTC.
        depth: Original source value, in Station.measurement_unit; None means
            missing, while zero and negative water levels can be actual readings.
        quality_status: observed for finite readings or missing when depth is None.
        source_name/source_collection/source_record_id: Provenance and source
            uniqueness fields used when importing or updating upstream records.
        received_at: First database insertion time, preserved across source corrections.
        updated_at: Database-maintained time of the latest row update.

    Hourly averaging, rain accumulation, scaling and user overrides belong to
    the prediction pipeline; they must not overwrite these raw readings.
    """
    __tablename__ = "observations"
    __table_args__ = (
        sa.ForeignKeyConstraint(
            ["station_id", "source_name", "source_collection"],
            [
                "hydrology.stations.id",
                "hydrology.stations.source_name",
                "hydrology.stations.observation_collection",
            ],
        ),
        sa.UniqueConstraint(
            "source_name",
            "source_collection",
            "source_record_id",
            name="uq_observations_source_record",
        ),
        sa.CheckConstraint("isfinite(observed_at)", name="observed_at_finite"),
        sa.CheckConstraint("quality_status IN ('observed', 'missing')", name="quality_status"),
        sa.CheckConstraint(
            "(quality_status = 'missing' AND depth IS NULL) "
            "OR (quality_status = 'observed' AND depth IS NOT NULL "
            "AND depth > '-Infinity'::float8 AND depth < 'Infinity'::float8)",
            name="depth_matches_quality",
        ),
    )
    station_id: Mapped[str] = mapped_column(sa.Text, primary_key=True)
    observed_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        primary_key=True,
        comment="Original source time_point; absolute instant, UTC connection.",
    )
    depth: Mapped[float | None] = mapped_column(
        sa.Double, comment="Original API depth, without conversion, imputation or scaling."
    )
    source_name: Mapped[str] = mapped_column(sa.Text)
    source_collection: Mapped[str] = mapped_column(sa.Text)
    source_record_id: Mapped[int | None] = mapped_column(sa.BigInteger)
    quality_status: Mapped[str] = mapped_column(sa.Text, server_default="observed")
    received_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        server_default=sa.func.now(),
        comment="First insertion time; preserve during source corrections.",
    )
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), server_default=sa.func.now(), server_onupdate=sa.FetchedValue()
    )


class Forecast(Base):
    """A saved forecast from unedited observations and a specific model/input contract.

    Input hashes are SHA-256 hexadecimal digests. Pipeline versions must identify
    the scaler and preprocessing; generating that identity belongs to the service.
    User override forecasts are not stored here.
    """

    __tablename__ = "forecasts"
    __table_args__ = (
        sa.CheckConstraint("btrim(model_version) <> ''", name="model_version_not_blank"),
        sa.CheckConstraint("btrim(feature_schema_version) <> ''", name="schema_version_not_blank"),
        sa.CheckConstraint("btrim(pipeline_version) <> ''", name="pipeline_version_not_blank"),
        sa.CheckConstraint("input_hash ~ '^[0-9a-f]{64}$'", name="input_hash_sha256"),
        sa.CheckConstraint("isfinite(anchor_observed_at)", name="anchor_finite"),
        sa.CheckConstraint("isfinite(generated_at)", name="generated_finite"),
        sa.UniqueConstraint(
            "station_id",
            "model_version",
            "feature_schema_version",
            "pipeline_version",
            "input_hash",
            name="uq_forecasts_input_identity",
        ),
        sa.Index(
            "ix_forecasts_station_model_anchor", "station_id", "model_version", "anchor_observed_at"
        ),
    )
    id: Mapped[UUID] = mapped_column(sa.Uuid, primary_key=True, default=uuid4)
    station_id: Mapped[str] = mapped_column(sa.ForeignKey("hydrology.stations.id"))
    model_version: Mapped[str] = mapped_column(sa.Text)
    feature_schema_version: Mapped[str] = mapped_column(sa.Text)
    pipeline_version: Mapped[str] = mapped_column(sa.Text)
    input_hash: Mapped[str] = mapped_column(sa.Text)
    anchor_observed_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True))
    generated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), server_default=sa.func.now()
    )


class ForecastPoint(Base):
    """One horizon in millimetres; measured truth stays in observations.

    The service must write every horizon with the parent in one transaction and
    calculate forecast_at from anchor + horizon. Foreign keys prevent orphaned
    points and accidental deletion of a parent that still has points.
    """

    __tablename__ = "forecast_points"
    __table_args__ = (
        sa.CheckConstraint("horizon_minutes > 0", name="horizon_positive"),
        sa.CheckConstraint("isfinite(forecast_at)", name="forecast_at_finite"),
        sa.CheckConstraint(
            "water_level_mm > '-Infinity'::float8 AND water_level_mm < 'Infinity'::float8",
            name="water_level_finite",
        ),
    )
    forecast_id: Mapped[UUID] = mapped_column(
        sa.ForeignKey("hydrology.forecasts.id"), primary_key=True
    )
    horizon_minutes: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
    forecast_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True))
    water_level_mm: Mapped[float] = mapped_column(sa.Double)
