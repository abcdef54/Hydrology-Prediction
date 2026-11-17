"""Freeze water, rain and forecast inventories in one read-only transaction."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

from research.station_analysis.config import ROOT, digest, sha256, write_json

FRAMES = ('stations', 'observations', 'weather_locations', 'forecast_runs', 'forecast_points')


def selection_identity(config):
    return digest({key: config.get(key) for key in ('water_station_ids', 'rain_station_ids')})


def database_identity():
    """Include a credential-free source identity; never serialize the connection URL."""
    from src.backend.db.session import get_database_url
    load_dotenv(ROOT / '.env', override=False)
    url = get_database_url()
    return digest({'host': url.host, 'port': url.port, 'database': url.database})


def validate_snapshot(folder, config):
    folder = Path(folder)
    manifest = json.loads((folder / 'manifest.json').read_text())
    if manifest['selection_identity'] != selection_identity(config):
        raise ValueError('Snapshot station selection changed; use --refresh-snapshot or another work directory')
    for name, expected in manifest['sha256'].items():
        path = folder / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f'Snapshot file changed or missing: {name}')
    return manifest


def extract_snapshot(config, destination):
    """All vintages and source dates are retained; analysis ranges are applied later."""
    from src.backend.db.session import build_engine
    load_dotenv(ROOT / '.env', override=False)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Snapshot destination must be a new staging directory')
    destination.mkdir(parents=True)
    engine = build_engine()
    try:
        with engine.connect().execution_options(isolation_level='REPEATABLE READ') as connection:
            connection.execute(text('SET TRANSACTION READ ONLY'))
            connection.execute(text("SET LOCAL statement_timeout='180s'"))
            snapshot_at = connection.execute(text('SELECT transaction_timestamp()')).scalar_one()
            stations = pd.read_sql(text(
                "SELECT id AS station_id,name,station_type,station_subtype,station_subtype_label,"
                "river_name,address,latitude,longitude,measurement_unit,"
                "alarm_level_1_m,alarm_level_2_m,alarm_level_3_m,source_name,source_station_id,"
                "enabled,sync_enabled,updated_at AS catalog_updated_at FROM hydrology.stations "
                "WHERE station_type IN ('water_level','rainfall') ORDER BY station_type,station_id"
            ), connection)
            selected = []
            for kind, key in [('water_level', 'water_station_ids'), ('rainfall', 'rain_station_ids')]:
                part = stations.loc[stations.station_type.eq(kind)]
                if config[key]:
                    missing = set(config[key]) - set(part.station_id)
                    if missing:
                        raise ValueError(f'Unknown {kind} station IDs: {sorted(missing)}')
                    part = part.loc[part.station_id.isin(config[key])]
                selected.append(part)
            stations = pd.concat(selected, ignore_index=True)
            stations.to_parquet(destination / 'stations.parquet', index=False)
            query = text(
                'SELECT station_id,observed_at,depth,quality_status,received_at,updated_at '
                'FROM hydrology.observations WHERE station_id=ANY(:ids) '
                'ORDER BY station_id,observed_at')
            # Server-side streaming avoids materializing all raw Python records at once.
            import pyarrow as pa
            import pyarrow.parquet as pq
            writer = None
            count = 0
            try:
                stream = connection.execution_options(stream_results=True)
                for frame in pd.read_sql(query, stream, params={'ids': stations.station_id.tolist()},
                                         chunksize=100000):
                    for col in ('observed_at', 'received_at', 'updated_at'):
                        frame[col] = pd.to_datetime(frame[col], utc=True)
                    table = pa.Table.from_pandas(frame, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(destination / 'observations.parquet', table.schema)
                    writer.write_table(table.cast(writer.schema))
                    count += len(frame)
            finally:
                if writer:
                    writer.close()
                connection.execution_options(stream_results=False)
            if writer is None:
                pd.DataFrame(columns=['station_id','observed_at','depth','quality_status',
                                      'received_at','updated_at']).to_parquet(
                    destination / 'observations.parquet', index=False)
            locations = pd.read_sql(text(
                'SELECT id AS location_id,station_id,latitude,longitude FROM hydrology.weather_locations '
                'ORDER BY id'), connection)
            locations.to_parquet(destination / 'weather_locations.parquet', index=False)
            runs = pd.read_sql(text(
                "SELECT id::text AS run_id,location_id,model,initialized_at,retrieved_at,available_at,"
                "availability_basis,archive_kind,grid_latitude,grid_longitude,checksum,"
                "payload->'hourly_units' AS hourly_units,"
                "payload->>'utc_offset_seconds' AS utc_offset_seconds "
                'FROM hydrology.weather_forecast_runs ORDER BY initialized_at,retrieved_at,id'
            ), connection)
            runs['hourly_units'] = runs.hourly_units.map(lambda value: json.dumps(value, sort_keys=True))
            runs.to_parquet(destination / 'forecast_runs.parquet', index=False)
            points = pd.read_sql(text(
                'SELECT run_id::text AS run_id,valid_at,precipitation_mm,quality_status '
                'FROM hydrology.weather_forecast_hourly ORDER BY run_id,valid_at'
            ), connection)
            points.to_parquet(destination / 'forecast_points.parquet', index=False)
    finally:
        engine.dispose()
    manifest = {
        'snapshot_utc': str(snapshot_at), 'transaction': 'REPEATABLE READ / READ ONLY',
        'source_database_identity': database_identity(), 'selection_identity': selection_identity(config),
        'observation_records': count, 'station_count': len(stations), 'forecast_run_count': len(runs),
        'forecast_point_count': len(points), 'source_dates': 'all retained source dates, no future interpretation',
        'receipt_semantics': 'received_at is first local insertion; updated_at is latest local revision, not original publication',
        'sha256': {p.name: sha256(p) for p in destination.glob('*.parquet')},
    }
    write_json(destination / 'manifest.json', manifest)
    return manifest


def load_frames(folder):
    return {name: pd.read_parquet(Path(folder) / f'{name}.parquet') for name in FRAMES}


def analysis_anchors(observations, stations, config):
    water_ids = stations.loc[stations.station_type.eq('water_level'), 'station_id']
    times = pd.to_datetime(observations.loc[observations.station_id.isin(water_ids), 'observed_at'], utc=True)
    if times.empty:
        raise ValueError('No water observations define an analysis period')
    start = pd.Timestamp(config['start']).ceil('h')
    end = (pd.Timestamp(config['end']) - pd.Timedelta(nanoseconds=1)).floor('h') \
        if config.get('end') else times.max().floor('h')
    if end < start:
        raise ValueError('No complete right-labeled water hours in requested range')
    return pd.date_range(start, end, freq='h', tz='UTC', name='timestamp_utc')
