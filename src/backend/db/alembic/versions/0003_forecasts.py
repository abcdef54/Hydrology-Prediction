"""Store reusable forecasts and their horizon points; no predictions are generated."""

from alembic import op
import sqlalchemy as sa

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "forecasts",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("station_id", sa.Text(), nullable=False),
        sa.Column("model_version", sa.Text(), nullable=False),
        sa.Column("feature_schema_version", sa.Text(), nullable=False),
        sa.Column("pipeline_version", sa.Text(), nullable=False),
        sa.Column("input_hash", sa.Text(), nullable=False),
        sa.Column("anchor_observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "generated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "btrim(model_version) <> ''", name=op.f("ck_forecasts_model_version_not_blank")
        ),
        sa.CheckConstraint(
            "btrim(feature_schema_version) <> ''",
            name=op.f("ck_forecasts_schema_version_not_blank"),
        ),
        sa.CheckConstraint(
            "btrim(pipeline_version) <> ''", name=op.f("ck_forecasts_pipeline_version_not_blank")
        ),
        sa.CheckConstraint(
            "input_hash ~ '^[0-9a-f]{64}$'", name=op.f("ck_forecasts_input_hash_sha256")
        ),
        sa.CheckConstraint("isfinite(anchor_observed_at)", name=op.f("ck_forecasts_anchor_finite")),
        sa.CheckConstraint("isfinite(generated_at)", name=op.f("ck_forecasts_generated_finite")),
        sa.ForeignKeyConstraint(
            ["station_id"], ["hydrology.stations.id"], name=op.f("fk_forecasts_station_id_stations")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_forecasts")),
        sa.UniqueConstraint(
            "station_id",
            "model_version",
            "feature_schema_version",
            "pipeline_version",
            "input_hash",
            name="uq_forecasts_input_identity",
        ),
        schema="hydrology",
    )
    op.create_index(
        "ix_forecasts_station_model_anchor",
        "forecasts",
        ["station_id", "model_version", "anchor_observed_at"],
        schema="hydrology",
    )
    op.create_table(
        "forecast_points",
        sa.Column("forecast_id", sa.Uuid(), nullable=False),
        sa.Column("horizon_minutes", sa.Integer(), nullable=False),
        sa.Column("forecast_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("water_level_mm", sa.Double(), nullable=False),
        sa.CheckConstraint("horizon_minutes > 0", name=op.f("ck_forecast_points_horizon_positive")),
        sa.CheckConstraint(
            "isfinite(forecast_at)", name=op.f("ck_forecast_points_forecast_at_finite")
        ),
        sa.CheckConstraint(
            "water_level_mm > '-Infinity'::float8 AND water_level_mm < 'Infinity'::float8",
            name=op.f("ck_forecast_points_water_level_finite"),
        ),
        sa.ForeignKeyConstraint(
            ["forecast_id"],
            ["hydrology.forecasts.id"],
            name=op.f("fk_forecast_points_forecast_id_forecasts"),
        ),
        sa.PrimaryKeyConstraint("forecast_id", "horizon_minutes", name=op.f("pk_forecast_points")),
        schema="hydrology",
    )


def downgrade():
    op.drop_table("forecast_points", schema="hydrology")
    op.drop_index("ix_forecasts_station_model_anchor", table_name="forecasts", schema="hydrology")
    op.drop_table("forecasts", schema="hydrology")
