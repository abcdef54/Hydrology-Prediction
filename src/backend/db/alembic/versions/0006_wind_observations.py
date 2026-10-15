"""Add raw wind observations and independent sync permission; keep serving data intact."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade():
    """Add wind storage without converting any raw measurements or enabling predictions."""
    op.drop_constraint(op.f("ck_stations_station_type"), "stations", schema="hydrology")
    op.create_check_constraint(
        op.f("ck_stations_station_type"),
        "stations",
        "station_type IN ('water_level','rainfall','wind')",
        schema="hydrology",
    )
    op.drop_constraint(op.f("ck_stations_collection_matches_type"), "stations", schema="hydrology")
    op.create_check_constraint(
        op.f("ck_stations_collection_matches_type"),
        "stations",
        "(station_type='water_level' AND observation_collection='water_level_depth') OR "
        "(station_type='rainfall' AND observation_collection='rain_water_depth') OR "
        "(station_type='wind' AND observation_collection='iot_wind_speed')",
        schema="hydrology",
    )
    op.add_column(
        "stations",
        sa.Column("sync_enabled", sa.Boolean(), nullable=False, server_default=sa.false()),
        schema="hydrology",
    )
    op.add_column("stations", sa.Column("measurement_units", JSONB()), schema="hydrology")
    op.create_table(
        "wind_observations",
        sa.Column("station_id", sa.Text(), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        *(sa.Column(field, sa.Double()) for field in ("ws", "wd", "wsg", "wdg")),
        sa.Column("source_name", sa.Text(), nullable=False),
        sa.Column("source_collection", sa.Text(), nullable=False),
        sa.Column("source_record_id", sa.BigInteger(), nullable=False),
        sa.Column("quality_status", sa.Text(), nullable=False),
        sa.Column(
            "received_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.PrimaryKeyConstraint("station_id", "observed_at", name=op.f("pk_wind_observations")),
        sa.ForeignKeyConstraint(
            ["station_id", "source_name", "source_collection"],
            [
                "hydrology.stations.id",
                "hydrology.stations.source_name",
                "hydrology.stations.observation_collection",
            ],
            name=op.f("fk_wind_observations_station_id_stations"),
        ),
        sa.UniqueConstraint(
            "source_name",
            "source_collection",
            "source_record_id",
            name="uq_wind_observations_source_record",
        ),
        sa.CheckConstraint(
            "isfinite(observed_at)", name=op.f("ck_wind_observations_observed_at_finite")
        ),
        sa.CheckConstraint(
            "quality_status IN ('observed','missing')",
            name=op.f("ck_wind_observations_quality_status"),
        ),
        sa.CheckConstraint(
            "(quality_status = 'missing' AND ws IS NULL AND wd IS NULL AND wsg IS NULL AND wdg IS NULL) OR (quality_status = 'observed' AND (ws IS NOT NULL OR wd IS NOT NULL OR wsg IS NOT NULL OR wdg IS NOT NULL))",
            name=op.f("ck_wind_observations_values_match_quality"),
        ),
        *(
            sa.CheckConstraint(
                f"{field} > '-Infinity'::float8 AND {field} < 'Infinity'::float8",
                name=op.f(f"ck_wind_observations_{field}_finite"),
            )
            for field in ("ws", "wd", "wsg", "wdg")
        ),
        schema="hydrology",
    )
    op.execute(
        "CREATE TRIGGER wind_observations_updated_at BEFORE UPDATE ON hydrology.wind_observations FOR EACH ROW EXECUTE FUNCTION hydrology.set_updated_at()"
    )


def downgrade():
    """Require operators to remove wind data explicitly before reverting its schema."""
    if op.get_bind().scalar(
        sa.text("SELECT EXISTS(SELECT 1 FROM hydrology.stations WHERE station_type='wind')")
    ):
        raise RuntimeError("Wind stations exist; refusing destructive downgrade.")
    op.drop_table("wind_observations", schema="hydrology")
    op.drop_column("stations", "measurement_units", schema="hydrology")
    op.drop_column("stations", "sync_enabled", schema="hydrology")
    op.drop_constraint(op.f("ck_stations_collection_matches_type"), "stations", schema="hydrology")
    op.create_check_constraint(
        op.f("ck_stations_collection_matches_type"),
        "stations",
        "(station_type='water_level' AND observation_collection='water_level_depth') OR (station_type='rainfall' AND observation_collection='rain_water_depth')",
        schema="hydrology",
    )
    op.drop_constraint(op.f("ck_stations_station_type"), "stations", schema="hydrology")
    op.create_check_constraint(
        op.f("ck_stations_station_type"),
        "stations",
        "station_type IN ('water_level','rainfall')",
        schema="hydrology",
    )
