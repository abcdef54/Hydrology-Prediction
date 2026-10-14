"""Record the project owner's mm decision for the four supported rain stations."""

from alembic import op
import sqlalchemy as sa

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None

RAIN_STATION_IDS = ("750003", "750007", "750011", "750062")
UNIT_COMMENT = "Raw depth unit; supported water and rain stations use mm by project decision."


def upgrade():
    """Set supported rainfall units to mm without converting any observations.

    Returns:
        None. Updates only unset units for the four known Hue DCU rain stations.
    """
    stations = sa.table(
        "stations",
        sa.column("id", sa.Text),
        sa.column("station_type", sa.Text),
        sa.column("source_name", sa.Text),
        sa.column("observation_collection", sa.Text),
        sa.column("measurement_unit", sa.Text),
        schema="hydrology",
    )
    op.execute(
        stations.update()
        .where(
            stations.c.id.in_(RAIN_STATION_IDS),
            stations.c.station_type == "rainfall",
            stations.c.source_name == "hue_dcu",
            stations.c.observation_collection == "rain_water_depth",
            stations.c.measurement_unit.is_(None),
        )
        .values(measurement_unit="mm")
    )
    op.alter_column(
        "stations",
        "measurement_unit",
        schema="hydrology",
        existing_type=sa.Text(),
        comment=UNIT_COMMENT,
    )


def downgrade():
    """Restore the original unknown units for the four seeded rain stations.

    Returns:
        None. Leaves raw readings and other station metadata unchanged.
    """
    op.execute(
        sa.text(
            "UPDATE hydrology.stations SET measurement_unit = NULL "
            "WHERE id IN ('750003', '750007', '750011', '750062') "
            "AND station_type = 'rainfall' AND source_name = 'hue_dcu' "
            "AND observation_collection = 'rain_water_depth' AND measurement_unit = 'mm'"
        )
    )
    op.alter_column(
        "stations",
        "measurement_unit",
        schema="hydrology",
        existing_type=sa.Text(),
        comment="Water mm confirmed by project owner; rain unit unconfirmed (NULL).",
    )
