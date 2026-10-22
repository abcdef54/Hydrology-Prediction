"""Collect recent, fixed ECMWF runs for the production rainfall inputs."""

from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from uuid import uuid4

import pandas as pd
import requests
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert

from src.backend.db.models import Station
from src.backend.db.weather import locations, runs, hourly, fetches
from src.backend.ingestion.weather_forecast import (
    ENDPOINT, MODEL, parse_response, request_params, save_response,
)
from src.backend.prediction.config import StationModelRegistry

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOCK_KEYS = (20261022, 1)


class WeatherSourceError(RuntimeError):
    """Sanitized weather failure; never include provider response bodies or URLs."""

    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


def configured_locations():
    """Use production model locations, deduplicating shared rainfall grids."""
    path = Path(os.environ.get(
        'MODEL_CONFIG_PATH', PROJECT_ROOT / 'src/backend/prediction/model_configs.json'
    ))
    registry = StationModelRegistry.from_json(path, PROJECT_ROOT)
    # The registry file is the source of station/model choices, not a dataset.
    station_ids = {item['station_id'] for item in json.loads(path.read_text())}
    selected = {}
    for station_id in station_ids:
        for config in registry.models_for(station_id):
            if config.weather_location_id is None:
                continue
            if config.weather_forecast_model != MODEL:
                raise ValueError('Live weather collection currently supports ecmwf_ifs only.')
            key = config.weather_location_id
            if key not in selected:
                provider, source_id, latitude, longitude = key.split(':')
                if provider != 'dcu':
                    raise ValueError('Unsupported weather location identifier.')
                latitude, longitude = float(latitude), float(longitude)
                if not math.isfinite(latitude) or not -90 <= latitude <= 90:
                    raise ValueError('Invalid weather latitude.')
                if not math.isfinite(longitude) or not -180 <= longitude <= 180:
                    raise ValueError('Invalid weather longitude.')
                selected[key] = {
                    'location': dict(id=key, station_id=source_id,
                                     latitude=latitude, longitude=longitude),
                    'release_lag_hours': config.weather_release_lag_hours,
                    'max_age_hours': config.weather_max_age_hours,
                }
            else:
                selected[key]['release_lag_hours'] = min(
                    selected[key]['release_lag_hours'], config.weather_release_lag_hours
                )
                selected[key]['max_age_hours'] = max(
                    selected[key]['max_age_hours'], config.weather_max_age_hours
                )
    return list(selected.values())


def recent_cycles(now, release_lag_hours, max_age_hours):
    """Newest eligible six-hour cycle first, with bounded older-run fallback."""
    start = (pd.Timestamp(now) - pd.Timedelta(hours=max_age_hours)).ceil('6h')
    end = (pd.Timestamp(now) - pd.Timedelta(hours=release_lag_hours)).floor('6h')
    return list(reversed(pd.date_range(start, end, freq='6h').to_pydatetime()))


def fetch_run(session, location, cycle, stop, endpoint=ENDPOINT, timeout=(5, 30)):
    """Fetch one exact run, retrying transient failures with cancellable waits."""
    params = request_params(location, cycle)
    for attempt in range(3):
        if stop.is_set():
            raise WeatherSourceError('Weather sync cancelled.')
        retry_delay = 2 ** (attempt + 1)
        try:
            response = session.get(endpoint, params=params, timeout=timeout)
        except (requests.ConnectionError, requests.Timeout):
            if attempt == 2:
                raise WeatherSourceError('Weather transport failed after retries.') from None
        except requests.RequestException:
            raise WeatherSourceError('Weather request failed.') from None
        else:
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == 2:
                    raise WeatherSourceError('Weather provider unavailable.', response.status_code)
                try:
                    retry_delay = min(30, max(retry_delay, float(response.headers.get('Retry-After', 0))))
                except (ValueError, OverflowError):
                    pass
            elif response.status_code != 200:
                raise WeatherSourceError('Weather run not available.', response.status_code)
            else:
                try:
                    payload = response.json()
                    points = parse_response(payload, cycle)
                    if points[-1]['valid_at'] < cycle + timedelta(hours=48):
                        raise ValueError('Insufficient live forecast horizon.')
                except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
                    raise WeatherSourceError('Invalid weather forecast response.') from None
                checksum = hashlib.sha256(response.content).hexdigest()
                return params, payload, checksum
        if stop.wait(retry_delay):
            raise WeatherSourceError('Weather sync cancelled.')
    raise WeatherSourceError('Weather request failed.')


def complete_run(connection, location_id, cycle):
    """Skip downloaded complete runs; partial/null vintages remain retryable."""
    count = (
        sa.select(sa.func.count())
        .select_from(hourly)
        .where(hourly.c.run_id == runs.c.id,
               hourly.c.valid_at > cycle,
               hourly.c.valid_at <= cycle + timedelta(hours=48),
               hourly.c.quality_status == 'forecast')
        .correlate(runs)
        .scalar_subquery()
    )
    latest_count = connection.scalar(sa.select(count).where(
        runs.c.location_id == location_id, runs.c.model == MODEL,
        runs.c.initialized_at == cycle,
    ).order_by(runs.c.retrieved_at.desc()).limit(1))
    return latest_count == 48


def sync_weather_forecasts(engine, stop, dry_run=False, *, now=None):
    """Poll production grids once; each saved run and its fetch log are atomic.

    Actual successful retrieval time is a conservative availability timestamp.
    Old archive rows retain their original unknown-availability metadata.
    A separate PostgreSQL advisory lock prevents concurrent weather collectors.
    """
    enabled = os.environ.get('WEATHER_SYNC_ENABLED', 'true').strip().lower()
    if enabled not in {'true', 'false', '1', '0'}:
        raise ValueError('WEATHER_SYNC_ENABLED must be true or false.')
    if enabled in {'false', '0'}:
        return {'status': 'disabled'}
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError('Weather sync time must include a timezone.')
    grids = configured_locations()
    report = {'status': 'validated' if dry_run else 'committed', 'locations': []}
    if dry_run:
        report['locations'] = [item['location']['id'] for item in grids]
        return report
    endpoint = os.environ.get('WEATHER_SYNC_ENDPOINT', ENDPOINT)
    with engine.connect() as connection:
        locked = connection.scalar(sa.text('SELECT pg_try_advisory_lock(:a, :b)'),
                                   {'a': LOCK_KEYS[0], 'b': LOCK_KEYS[1]})
        connection.commit()
        if not locked:
            return {'status': 'skipped_writer_busy'}
        try:
            with requests.Session() as session:
                for grid in grids:
                    location = grid['location']
                    item = {'location_id': location['id'], 'fetched': 0, 'skipped': 0, 'incomplete': 0,
                            'failures': [], 'latest_initialized_at': None}
                    report['locations'].append(item)
                    with connection.begin():
                        station = connection.scalar(sa.select(Station.id).where(
                            Station.id == location['station_id'], Station.station_type == 'rainfall'
                        ))
                        if station is None:
                            raise ValueError('Configured weather rainfall station not found.')
                        connection.execute(insert(locations).values(**location).on_conflict_do_nothing())
                    for cycle in recent_cycles(now, grid['release_lag_hours'], grid['max_age_hours']):
                        if stop.is_set():
                            report['status'] = 'cancelled'
                            return report
                        with connection.begin():
                            complete = complete_run(connection, location['id'], cycle)
                        if complete:
                            item['skipped'] += 1
                            continue
                        params = request_params(location, cycle)
                        try:
                            params, payload, checksum = fetch_run(
                                session, location, cycle, stop, endpoint=endpoint
                            )
                        except WeatherSourceError as error:
                            if stop.is_set():
                                report['status'] = 'cancelled'
                                return report
                            report['status'] = 'partial'
                            item['failures'].append({'initialized_at': cycle.isoformat(),
                                                     'http_status': error.status_code,
                                                     'error': str(error)})
                            with connection.begin():
                                connection.execute(insert(fetches).values(
                                    id=uuid4(), location_id=location['id'],
                                    initialized_at=cycle, retrieved_at=datetime.now(timezone.utc),
                                    http_status=error.status_code, status='failed',
                                    request=params, error=str(error),
                                ))
                            continue
                        retrieved = datetime.now(timezone.utc)
                        if stop.is_set():
                            report['status'] = 'cancelled'
                            return report
                        if any(point['precipitation_mm'] is None
                               for point in parse_response(payload, cycle)[:48]):
                            item['incomplete'] += 1
                            report['status'] = 'partial'
                        with connection.begin():
                            changed = save_response(
                                connection, location, cycle, params, payload, checksum, retrieved,
                                available_at=retrieved, availability_basis='observed_retrieval',
                                archive_kind='operational_single_run',
                            )
                            connection.execute(insert(fetches).values(
                                id=uuid4(), location_id=location['id'],
                                initialized_at=cycle, retrieved_at=retrieved, http_status=200,
                                status='success', checksum=checksum, request=params,
                            ))
                        item['fetched'] += int(changed)
                        if stop.wait(0.15):
                            report['status'] = 'cancelled'
                            return report
                    with connection.begin():
                        latest = connection.scalar(sa.select(sa.func.max(runs.c.initialized_at)).where(
                            runs.c.location_id == location['id'], runs.c.model == MODEL,
                            runs.c.initialized_at <= now,
                        ))
                    if latest is not None:
                        item['latest_initialized_at'] = latest.isoformat()
                        item['latest_age_hours'] = round((now - latest).total_seconds() / 3600, 2)
                        item['stale'] = now - latest > timedelta(hours=grid['max_age_hours'])
                    else:
                        item['stale'] = True
                    if item['stale']:
                        report['status'] = 'partial'
        finally:
            if connection.in_transaction():
                connection.rollback()
            if not connection.invalidated:
                connection.execute(sa.text('SELECT pg_advisory_unlock(:a, :b)'),
                                   {'a': LOCK_KEYS[0], 'b': LOCK_KEYS[1]})
                connection.commit()
    return report
