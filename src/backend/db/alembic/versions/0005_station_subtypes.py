"""Preserve source station subtypes without changing observations or serving flags."""

from alembic import op
import sqlalchemy as sa

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade():
    """Add nullable metadata for measurement stations and flood towers."""
    op.add_column("stations", sa.Column("station_subtype", sa.Text()), schema="hydrology")
    op.add_column("stations", sa.Column("station_subtype_label", sa.Text()), schema="hydrology")


def downgrade():
    """Remove subtype metadata without changing station identity or observations."""
    op.drop_column("stations", "station_subtype_label", schema="hydrology")
    op.drop_column("stations", "station_subtype", schema="hydrology")
