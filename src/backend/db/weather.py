"""Weather forecast tables registered with the application's database metadata."""
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from src.backend.db.models import Base

locations = sa.Table('weather_locations', Base.metadata,
    sa.Column('id', sa.Text, primary_key=True),
    sa.Column('station_id', sa.Text, sa.ForeignKey('hydrology.stations.id'), nullable=False),
    sa.Column('latitude', sa.Double, nullable=False),
    sa.Column('longitude', sa.Double, nullable=False),
    sa.CheckConstraint('latitude BETWEEN -90 AND 90', name='latitude'),
    sa.CheckConstraint('longitude BETWEEN -180 AND 180', name='longitude'))
runs = sa.Table('weather_forecast_runs', Base.metadata,
    sa.Column('id', sa.Uuid, primary_key=True),
    sa.Column('location_id', sa.Text, sa.ForeignKey('hydrology.weather_locations.id'), nullable=False),
    sa.Column('model', sa.Text, nullable=False),
    sa.Column('initialized_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('retrieved_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('available_at', sa.DateTime(timezone=True)),
    sa.Column('availability_basis', sa.Text, nullable=False),
    sa.Column('archive_kind', sa.Text, nullable=False),
    sa.Column('grid_latitude', sa.Double, nullable=False),
    sa.Column('grid_longitude', sa.Double, nullable=False),
    sa.Column('checksum', sa.Text, nullable=False),
    sa.Column('payload', JSONB, nullable=False),
    sa.Column('request', JSONB, nullable=False),
    sa.UniqueConstraint('location_id','model','initialized_at','checksum', name='uq_weather_run_vintage'),
    sa.Index('ix_weather_run_lookup','location_id','model','initialized_at'))
hourly = sa.Table('weather_forecast_hourly', Base.metadata,
    sa.Column('run_id', sa.Uuid, sa.ForeignKey('hydrology.weather_forecast_runs.id'), primary_key=True),
    sa.Column('valid_at', sa.DateTime(timezone=True), primary_key=True),
    sa.Column('precipitation_mm', sa.Double),
    sa.Column('quality_status', sa.Text, nullable=False),
    sa.CheckConstraint("(quality_status = 'missing' AND precipitation_mm IS NULL) OR "
                       "(quality_status = 'forecast' AND precipitation_mm >= 0 AND precipitation_mm < 'Infinity'::float8)", name='value_quality'),
    sa.Index('ix_weather_hourly_valid','valid_at'))
fetches = sa.Table('weather_fetches', Base.metadata,
    sa.Column('id', sa.Uuid, primary_key=True),
    sa.Column('location_id', sa.Text, sa.ForeignKey('hydrology.weather_locations.id'), nullable=False),
    sa.Column('initialized_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('retrieved_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('http_status', sa.Integer),
    sa.Column('status', sa.Text, nullable=False),
    sa.Column('checksum', sa.Text),
    sa.Column('request', JSONB, nullable=False),
    sa.Column('error', sa.Text))
