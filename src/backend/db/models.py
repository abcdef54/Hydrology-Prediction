"""Raw observations; no hourly aggregation or model preprocessing here."""

from datetime import datetime

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
