"""Apply owner-approved mm units to water stations and display the four wind stations."""

from alembic import op
import sqlalchemy as sa

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade():
    """Update station metadata only; preserve all raw measurements and wind units.

    Returns:
        None. Unset Hue water units become mm; four imported wind stations become
        visible. No observations or prediction permissions are changed.
    """
    op.execute(
        sa.text(
            "UPDATE hydrology.stations SET measurement_unit='mm' "
            "WHERE source_name='hue_dcu' AND station_type='water_level' "
            "AND observation_collection='water_level_depth' AND measurement_unit IS NULL"
        )
    )
    op.execute(
        sa.text(
            "UPDATE hydrology.stations SET enabled=true "
            "WHERE source_name='hue_dcu' AND station_type='wind' "
            "AND id IN ('wind:759002','wind:759001','wind:228005','wind:228006') "
            "AND enabled=false"
        )
    )


def downgrade():
    """Retain owner-approved metadata; a schema downgrade does not undo data decisions."""
