"""Read a frozen repeatable-read PostgreSQL snapshot; persist only analysis SQLite."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import tempfile

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

from .config import ROOT, write_json


def extract(config):
    from src.backend.db.session import build_engine

    load_dotenv(ROOT / '.env', override=False)
    work = Path(config['work_dir'])
    (work / 'snapshot').mkdir(parents=True, exist_ok=True)
    engine = build_engine()
    try:
        with engine.connect().execution_options(isolation_level='REPEATABLE READ') as connection:
            connection.execute(text('SET TRANSACTION READ ONLY'))
            connection.execute(text("SET LOCAL statement_timeout='120s'"))
            stations = pd.read_sql(text(
                "SELECT id AS station_id,name,station_subtype,station_subtype_label,"
                "river_name,address,latitude,longitude,measurement_unit,"
                "alarm_level_1_m,alarm_level_2_m,alarm_level_3_m,enabled,sync_enabled,"
                "source_name,source_station_id,updated_at AS catalog_updated_at "
                "FROM hydrology.stations WHERE station_type='water_level' ORDER BY id"
            ), connection)
            if config['station_ids']:
                unknown = set(config['station_ids']) - set(stations.station_id)
                if unknown:
                    raise ValueError(f'Unknown water stations: {sorted(unknown)}')
                stations = stations[stations.station_id.isin(config['station_ids'])].copy()
            if stations.empty:
                raise ValueError('No water stations selected')
            params = {'ids': stations.station_id.tolist(), 'start': config.get('start'),
                      'end': config.get('end')}
            conditions = ['station_id=ANY(:ids)']
            if params['start']:
                conditions.append('observed_at>=:start')
            if params['end']:
                conditions.append('observed_at<:end')
            raw = pd.read_sql(text(
                'SELECT station_id,observed_at,depth,quality_status FROM hydrology.observations '
                f"WHERE {' AND '.join(conditions)} ORDER BY station_id,observed_at"
            ), connection, params=params)
            snapshot_time = connection.execute(text('SELECT transaction_timestamp()')).scalar_one()
    finally:
        engine.dispose()
    if raw.empty:
        raise ValueError('No observations in requested period')
    raw['observed_at'] = pd.to_datetime(raw.observed_at, utc=True)
    raw['depth'] = pd.to_numeric(raw.depth, errors='coerce')
    raw['finite'] = np.isfinite(raw.depth)
    units = stations.set_index('station_id').measurement_unit
    factors = {'mm': 1.0, 'cm': 10.0, 'm': 1000.0}
    # Unknown units retain raw summary but do not enter numerical comparisons.
    raw['depth_mm'] = raw.depth * raw.station_id.map(units).map(factors)
    raw.loc[~raw.finite | raw.quality_status.ne('observed'), 'depth_mm'] = np.nan
    start = pd.Timestamp(config['start']) if config.get('start') else raw.observed_at.min().floor('h')
    # Last, potentially partial hour is excluded for all stations.
    end = (pd.Timestamp(config['end']) if config.get('end') else raw.observed_at.max()).floor('h')
    hourly_index = pd.date_range(start.ceil('h'), end, freq='h', inclusive='left', tz='UTC')
    hourly = pd.DataFrame(index=hourly_index)
    summaries = []
    for station_id in stations.station_id:
        values = raw[raw.station_id.eq(station_id)]
        series = values.set_index('observed_at').depth_mm
        hourly[station_id] = series.resample('h', closed='left', label='left').mean().reindex(hourly_index)
        summaries.append({'station_id': station_id, 'raw_records': len(values),
                          'raw_finite': int(values.finite.sum()),
                          'raw_explicit_missing': int((~values.finite).sum()),
                          'raw_first': str(values.observed_at.min()) if len(values) else None,
                          'raw_last': str(values.observed_at.max()) if len(values) else None,
                          'unit_usable': units[station_id] in factors})
    hourly.index.name = 'timestamp_utc'
    stations.to_parquet(work / 'snapshot/stations.parquet', index=False)
    raw.drop(columns=['finite']).to_parquet(work / 'snapshot/raw_observations.parquet', index=False)
    pd.DataFrame(summaries).to_parquet(work / 'snapshot/raw_summary.parquet', index=False)
    hourly.to_parquet(work / 'snapshot/hourly.parquet')
    write_json(work / 'snapshot/manifest.json', {
        'snapshot_utc': snapshot_time, 'analysis_start_utc': start, 'exclusive_end_utc': end,
        'last_raw_utc': raw.observed_at.max(), 'station_count': len(stations),
        'raw_records': len(raw), 'hourly_rows': len(hourly), 'hourly_method':
        'mean of finite quality=observed source readings in [hour,hour+1h); no fill',
        'database_transaction': 'REPEATABLE READ, READ ONLY',
        'measurement_units': 'DB/project contract; physical quantity and station zero unverified',
        'selection': {key: config.get(key) for key in ('start', 'end', 'station_ids')},
        'created_utc': datetime.now(timezone.utc).isoformat(),
    })


def store_analysis_database(path, tables):
    """Transactional replace in a dedicated local file, never the source database."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent, prefix='.analysis-') as temporary_dir:
        temporary = Path(temporary_dir) / 'analysis.sqlite'
        with sqlite3.connect(temporary) as connection:
            for name, frame in tables.items():
                frame = frame.copy()
                for column in frame.columns:
                    if isinstance(frame[column].dtype, pd.DatetimeTZDtype):
                        frame[column] = frame[column].astype(str)
                    elif frame[column].dtype == 'object':
                        frame[column] = frame[column].map(
                            lambda value: str(value) if isinstance(value, (dict, list, tuple)) else value)
                frame.to_sql(name, connection, if_exists='replace', index=False)
            connection.commit()
        temporary.replace(path)
