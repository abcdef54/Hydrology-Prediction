"""Keep SOHOA infrastructure and operational reports separate from DCU observations."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade():
    """Create linked facility metadata and raw operational storage without data conversion."""
    op.create_table(
        "facilities",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("source_id", sa.BigInteger(), nullable=False),
        sa.Column("source_name", sa.Text(), nullable=False, server_default="hue_sohoa"),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("kind_label", sa.Text(), nullable=False),
        sa.Column("station_id", sa.Text()),
        sa.Column("latitude", sa.Double()),
        sa.Column("longitude", sa.Double()),
        sa.Column("specifications", JSONB(), nullable=False),
        sa.Column("measurement_units", JSONB(), nullable=False),
        sa.Column("sync_enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("last_full_sync_at", sa.DateTime(timezone=True)),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint("source_id", name=op.f("uq_facilities_source_id")),
        sa.ForeignKeyConstraint(
            ["station_id"],
            ["hydrology.stations.id"],
            name=op.f("fk_facilities_station_id_stations"),
        ),
        sa.CheckConstraint("source_id > 0", name=op.f("ck_facilities_positive_source_id")),
        sa.CheckConstraint(
            "kind IN ('hydropower','reservoir','dam','river')", name=op.f("ck_facilities_kind")
        ),
        sa.CheckConstraint("latitude BETWEEN -90 AND 90", name=op.f("ck_facilities_latitude")),
        sa.CheckConstraint("longitude BETWEEN -180 AND 180", name=op.f("ck_facilities_longitude")),
        sa.CheckConstraint("btrim(name) <> ''", name=op.f("ck_facilities_name_not_blank")),
        schema="hydrology",
    )
    op.create_table(
        "facility_operations",
        sa.Column("source_record_id", sa.BigInteger(), primary_key=True),
        sa.Column("facility_id", sa.Text(), nullable=False),
        sa.Column("function_code", sa.Text(), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True)),
        sa.Column("time_status", sa.Text(), nullable=False),
        *(
            sa.Column(field, sa.Text(), nullable=False)
            for field in ("observation_date_text", "observation_hour_text", "source_updated_text")
        ),
        sa.Column("source_updated_at", sa.DateTime(timezone=True)),
        sa.Column("source_status", sa.Text()),
        sa.Column("source_status_label", sa.Text()),
        *(
            sa.Column(field, sa.Double())
            for field in ("htl", "hhl", "qden", "qdi", "mucnuocsong", "luuluongdieutiet")
        ),
        *(
            sa.Column(field, sa.Text())
            for field in ("gate_opening_text", "operation_start_text", "operation_end_text")
        ),
        sa.Column(
            "received_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.ForeignKeyConstraint(
            ["facility_id"],
            ["hydrology.facilities.id"],
            name=op.f("fk_facility_operations_facility_id_facilities"),
        ),
        sa.CheckConstraint(
            "source_record_id > 0", name=op.f("ck_facility_operations_positive_source_id")
        ),
        sa.CheckConstraint(
            "observed_at IS NULL OR isfinite(observed_at)",
            name=op.f("ck_facility_operations_observed_at_finite"),
        ),
        sa.CheckConstraint(
            "time_status IN ('valid','invalid')", name=op.f("ck_facility_operations_time_status")
        ),
        sa.CheckConstraint(
            "(time_status='valid') = (observed_at IS NOT NULL)",
            name=op.f("ck_facility_operations_time_matches_status"),
        ),
        *(
            sa.CheckConstraint(
                f"{field} > '-Infinity'::float8 AND {field} < 'Infinity'::float8",
                name=op.f(f"ck_facility_operations_{field}_finite"),
            )
            for field in ("htl", "hhl", "qden", "qdi", "mucnuocsong", "luuluongdieutiet")
        ),
        schema="hydrology",
    )
    op.create_index(
        "ix_facility_operations_facility_time",
        "facility_operations",
        ["facility_id", "observed_at", "source_record_id"],
        schema="hydrology",
    )
    for table in ("facilities", "facility_operations"):
        op.execute(
            f"CREATE TRIGGER {table}_updated_at BEFORE UPDATE ON hydrology.{table} FOR EACH ROW EXECUTE FUNCTION hydrology.set_updated_at()"
        )


def downgrade():
    """Refuse removal of imported operational history; empty schemas may be reverted."""
    if op.get_bind().scalar(sa.text("SELECT EXISTS(SELECT 1 FROM hydrology.facilities)")):
        raise RuntimeError("Facilities exist; refusing destructive downgrade.")
    op.drop_table("facility_operations", schema="hydrology")
    op.drop_table("facilities", schema="hydrology")
