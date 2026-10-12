"""Initial raw observation schema; frozen independently of live ORM models."""

from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.execute(sa.schema.CreateSchema("hydrology"))
    op.create_table(
        "stations",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("station_type", sa.Text(), nullable=False),
        sa.Column("river_name", sa.Text(), nullable=True),
        sa.Column("address", sa.Text(), nullable=True),
        sa.Column("latitude", sa.Double(), nullable=True),
        sa.Column("longitude", sa.Double(), nullable=True),
        sa.Column("alarm_level_1_m", sa.Double(), nullable=True),
        sa.Column("alarm_level_2_m", sa.Double(), nullable=True),
        sa.Column("alarm_level_3_m", sa.Double(), nullable=True),
        sa.Column("enabled", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("prediction_enabled", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("source_name", sa.Text(), nullable=False),
        sa.Column("source_station_id", sa.Text(), nullable=False),
        sa.Column("observation_collection", sa.Text(), nullable=False),
        sa.Column(
            "measurement_unit",
            sa.Text(),
            nullable=True,
            comment="Water mm confirmed by project owner; rain unit unconfirmed (NULL).",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "(station_type = 'water_level' AND observation_collection = 'water_level_depth') OR (station_type = 'rainfall' AND observation_collection = 'rain_water_depth')",
            name=op.f("ck_stations_collection_matches_type"),
        ),
        sa.CheckConstraint(
            "NOT prediction_enabled OR station_type = 'water_level'",
            name=op.f("ck_stations_prediction_water_only"),
        ),
        sa.CheckConstraint(
            "alarm_level_1_m >= 0 AND alarm_level_1_m < 'Infinity'::float8",
            name=op.f("ck_stations_alarm_1_finite"),
        ),
        sa.CheckConstraint(
            "alarm_level_2_m >= 0 AND alarm_level_2_m < 'Infinity'::float8",
            name=op.f("ck_stations_alarm_2_finite"),
        ),
        sa.CheckConstraint(
            "alarm_level_3_m >= 0 AND alarm_level_3_m < 'Infinity'::float8",
            name=op.f("ck_stations_alarm_3_finite"),
        ),
        sa.CheckConstraint("btrim(id) <> ''", name=op.f("ck_stations_id_not_blank")),
        sa.CheckConstraint("btrim(name) <> ''", name=op.f("ck_stations_name_not_blank")),
        sa.CheckConstraint("btrim(source_name) <> ''", name=op.f("ck_stations_source_not_blank")),
        sa.CheckConstraint(
            "btrim(source_station_id) <> ''", name=op.f("ck_stations_source_station_not_blank")
        ),
        sa.CheckConstraint(
            "measurement_unit IS NULL OR btrim(measurement_unit) <> ''",
            name=op.f("ck_stations_unit_not_blank"),
        ),
        sa.CheckConstraint(
            "station_type IN ('water_level', 'rainfall')", name=op.f("ck_stations_station_type")
        ),
        sa.CheckConstraint(
            "alarm_level_1_m < alarm_level_2_m", name=op.f("ck_stations_alarm_1_before_2")
        ),
        sa.CheckConstraint(
            "alarm_level_1_m < alarm_level_3_m", name=op.f("ck_stations_alarm_1_before_3")
        ),
        sa.CheckConstraint(
            "alarm_level_2_m < alarm_level_3_m", name=op.f("ck_stations_alarm_2_before_3")
        ),
        sa.CheckConstraint("latitude BETWEEN -90 AND 90", name=op.f("ck_stations_latitude")),
        sa.CheckConstraint("longitude BETWEEN -180 AND 180", name=op.f("ck_stations_longitude")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_stations")),
        sa.UniqueConstraint(
            "id", "source_name", "observation_collection", name="uq_stations_source_mapping"
        ),
        sa.UniqueConstraint(
            "source_name",
            "observation_collection",
            "source_station_id",
            name="uq_stations_source_station",
        ),
        schema="hydrology",
    )
    op.create_table(
        "observations",
        sa.Column("station_id", sa.Text(), nullable=False),
        sa.Column(
            "observed_at",
            sa.DateTime(timezone=True),
            nullable=False,
            comment="Original source time_point; absolute instant, UTC connection.",
        ),
        sa.Column(
            "depth",
            sa.Double(),
            nullable=True,
            comment="Original API depth, without conversion, imputation or scaling.",
        ),
        sa.Column("source_name", sa.Text(), nullable=False),
        sa.Column("source_collection", sa.Text(), nullable=False),
        sa.Column("source_record_id", sa.BigInteger(), nullable=True),
        sa.Column("quality_status", sa.Text(), server_default="observed", nullable=False),
        sa.Column(
            "received_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
            comment="First insertion time; preserve during source corrections.",
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "(quality_status = 'missing' AND depth IS NULL) OR (quality_status = 'observed' AND depth IS NOT NULL AND depth > '-Infinity'::float8 AND depth < 'Infinity'::float8)",
            name=op.f("ck_observations_depth_matches_quality"),
        ),
        sa.CheckConstraint(
            "quality_status IN ('observed', 'missing')", name=op.f("ck_observations_quality_status")
        ),
        sa.CheckConstraint(
            "isfinite(observed_at)", name=op.f("ck_observations_observed_at_finite")
        ),
        sa.ForeignKeyConstraint(
            ["station_id", "source_name", "source_collection"],
            [
                "hydrology.stations.id",
                "hydrology.stations.source_name",
                "hydrology.stations.observation_collection",
            ],
            name=op.f("fk_observations_station_id_stations"),
        ),
        sa.PrimaryKeyConstraint("station_id", "observed_at", name=op.f("pk_observations")),
        sa.UniqueConstraint(
            "source_name",
            "source_collection",
            "source_record_id",
            name="uq_observations_source_record",
        ),
        schema="hydrology",
    )

    op.execute("""CREATE FUNCTION hydrology.set_updated_at() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            NEW.updated_at = clock_timestamp();
            RETURN NEW;
        END;
        $$""")
    for table in ("stations", "observations"):
        op.execute(
            f"CREATE TRIGGER {table}_updated_at BEFORE UPDATE ON hydrology.{table} "
            "FOR EACH ROW EXECUTE FUNCTION hydrology.set_updated_at()"
        )


def downgrade():
    op.drop_table("observations", schema="hydrology")
    op.drop_table("stations", schema="hydrology")
    op.execute("DROP FUNCTION hydrology.set_updated_at()")
    op.execute(sa.schema.DropSchema("hydrology"))
