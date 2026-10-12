"""Seed verified metadata (2026-10-12); no observations or alarm thresholds."""

from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

STATIONS = [
    {
        "id": "0074647403",
        "name": "Dã Viên",
        "station_type": "water_level",
        "river_name": "Sông Hương",
        "address": "Cầu Dã Viên, thành phố Huế",
        "latitude": 16.458944,
        "longitude": 107.571361,
        "prediction_enabled": True,
        "source_name": "hue_dcu",
        "source_station_id": "0074647403",
        "observation_collection": "water_level_depth",
        "measurement_unit": "mm",
    },
    {
        "id": "750011",
        "name": "Hồ A Lá (A Lưới)",
        "station_type": "rainfall",
        "river_name": None,
        "address": "Công trình đầu mối hồ A Lá",
        "latitude": 16.260278,
        "longitude": 107.245,
        "prediction_enabled": False,
        "source_name": "hue_dcu",
        "source_station_id": "750011",
        "observation_collection": "rain_water_depth",
        "measurement_unit": None,
    },
    {
        "id": "750007",
        "name": "Nhâm (A Lưới)",
        "station_type": "rainfall",
        "river_name": None,
        "address": "Đồn biên phòng Nhâm",
        "latitude": 16.221111,
        "longitude": 107.179444,
        "prediction_enabled": False,
        "source_name": "hue_dcu",
        "source_station_id": "750007",
        "observation_collection": "rain_water_depth",
        "measurement_unit": None,
    },
    {
        "id": "750062",
        "name": "Hồng Thái (A Lưới)",
        "station_type": "rainfall",
        "river_name": None,
        "address": "UBND xã Hồng Thái",
        "latitude": 16.23981,
        "longitude": 107.24161,
        "prediction_enabled": False,
        "source_name": "hue_dcu",
        "source_station_id": "750062",
        "observation_collection": "rain_water_depth",
        "measurement_unit": None,
    },
    {
        "id": "750003",
        "name": "Thị trấn A Lưới (A Lưới)",
        "station_type": "rainfall",
        "river_name": None,
        "address": "Phòng NN và PTNT huyện A Lưới",
        "latitude": 16.272222,
        "longitude": 107.2325,
        "prediction_enabled": False,
        "source_name": "hue_dcu",
        "source_station_id": "750003",
        "observation_collection": "rain_water_depth",
        "measurement_unit": None,
    },
]


def upgrade():
    table = sa.table(
        "stations",
        sa.column("id", sa.Text),
        sa.column("name", sa.Text),
        sa.column("station_type", sa.Text),
        sa.column("river_name", sa.Text),
        sa.column("address", sa.Text),
        sa.column("latitude", sa.Double),
        sa.column("longitude", sa.Double),
        sa.column("prediction_enabled", sa.Boolean),
        sa.column("source_name", sa.Text),
        sa.column("source_station_id", sa.Text),
        sa.column("observation_collection", sa.Text),
        sa.column("measurement_unit", sa.Text),
        schema="hydrology",
    )
    op.bulk_insert(table, STATIONS)


def downgrade():
    table = sa.table("stations", sa.column("id", sa.Text), schema="hydrology")
    # Foreign keys deliberately block removing a station that already has observations.
    op.execute(table.delete().where(table.c.id.in_([row["id"] for row in STATIONS])))
