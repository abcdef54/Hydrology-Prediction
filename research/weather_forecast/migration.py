"""Store forecast weather separately from measured observations."""
from alembic import op
revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None

def upgrade():
    op.execute('\nCREATE TABLE hydrology.weather_locations (\n\tid TEXT NOT NULL, \n\tstation_id TEXT NOT NULL, \n\tlatitude DOUBLE PRECISION NOT NULL, \n\tlongitude DOUBLE PRECISION NOT NULL, \n\tCONSTRAINT pk_weather_locations PRIMARY KEY (id), \n\tCONSTRAINT ck_weather_locations_latitude CHECK (latitude BETWEEN -90 AND 90), \n\tCONSTRAINT ck_weather_locations_longitude CHECK (longitude BETWEEN -180 AND 180), \n\tCONSTRAINT fk_weather_locations_station_id_stations FOREIGN KEY(station_id) REFERENCES hydrology.stations (id)\n)\n\n')
    op.execute('\nCREATE TABLE hydrology.weather_forecast_runs (\n\tid UUID NOT NULL, \n\tlocation_id TEXT NOT NULL, \n\tmodel TEXT NOT NULL, \n\tinitialized_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tretrieved_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tavailable_at TIMESTAMP WITH TIME ZONE, \n\tavailability_basis TEXT NOT NULL, \n\tarchive_kind TEXT NOT NULL, \n\tgrid_latitude DOUBLE PRECISION NOT NULL, \n\tgrid_longitude DOUBLE PRECISION NOT NULL, \n\tchecksum TEXT NOT NULL, \n\tpayload JSONB NOT NULL, \n\trequest JSONB NOT NULL, \n\tCONSTRAINT pk_weather_forecast_runs PRIMARY KEY (id), \n\tCONSTRAINT uq_weather_run_vintage UNIQUE (location_id, model, initialized_at, checksum), \n\tCONSTRAINT fk_weather_forecast_runs_location_id_weather_locations FOREIGN KEY(location_id) REFERENCES hydrology.weather_locations (id)\n)\n\n')
    op.execute('CREATE INDEX ix_weather_run_lookup ON hydrology.weather_forecast_runs (location_id, model, initialized_at)')
    op.execute("\nCREATE TABLE hydrology.weather_forecast_hourly (\n\trun_id UUID NOT NULL, \n\tvalid_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tprecipitation_mm DOUBLE PRECISION, \n\tquality_status TEXT NOT NULL, \n\tCONSTRAINT pk_weather_forecast_hourly PRIMARY KEY (run_id, valid_at), \n\tCONSTRAINT ck_weather_forecast_hourly_value_quality CHECK ((quality_status = 'missing' AND precipitation_mm IS NULL) OR (quality_status = 'forecast' AND precipitation_mm >= 0 AND precipitation_mm < 'Infinity'::float8)), \n\tCONSTRAINT fk_weather_forecast_hourly_run_id_weather_forecast_runs FOREIGN KEY(run_id) REFERENCES hydrology.weather_forecast_runs (id)\n)\n\n")
    op.execute('CREATE INDEX ix_weather_hourly_valid ON hydrology.weather_forecast_hourly (valid_at)')
    op.execute('\nCREATE TABLE hydrology.weather_fetches (\n\tid UUID NOT NULL, \n\tlocation_id TEXT NOT NULL, \n\tinitialized_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tretrieved_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\thttp_status INTEGER, \n\tstatus TEXT NOT NULL, \n\tchecksum TEXT, \n\trequest JSONB NOT NULL, \n\terror TEXT, \n\tCONSTRAINT pk_weather_fetches PRIMARY KEY (id), \n\tCONSTRAINT fk_weather_fetches_location_id_weather_locations FOREIGN KEY(location_id) REFERENCES hydrology.weather_locations (id)\n)\n\n')

def downgrade():
    op.drop_table('weather_fetches', schema="hydrology")
    op.drop_table('weather_forecast_hourly', schema="hydrology")
    op.drop_table('weather_forecast_runs', schema="hydrology")
    op.drop_table('weather_locations', schema="hydrology")


if __name__ == '__main__':
    from dotenv import load_dotenv
    from alembic import command
    from alembic.config import Config
    load_dotenv()
    command.upgrade(Config('src/backend/db/alembic.ini'), 'head')
