"""Operational run planning, isolation and real HTTP provider-contract checks."""

from datetime import datetime, timezone
import os
from threading import Event
from unittest import TestCase, skipUnless
from unittest.mock import MagicMock, patch
from uuid import uuid4

import requests

from src.backend.ingestion.weather_live import (
    configured_locations, fetch_run, recent_cycles, sync_weather_forecasts, WeatherSourceError,
)

NOW = datetime(2026, 10, 22, 13, tzinfo=timezone.utc)
LOCATION = {'id': 'dcu:750007:16.221111:107.179444', 'station_id': '750007',
            'latitude': 16.221111, 'longitude': 107.179444}


class WeatherLiveTests(TestCase):
    def test_recent_cycles_are_bounded_and_newest_first(self):
        cycles = recent_cycles(NOW, 6, 24)
        self.assertEqual([item.hour for item in cycles], [6, 0, 18])
        self.assertTrue(all(6 <= (NOW - item).total_seconds() / 3600 <= 24 for item in cycles))

    def test_production_models_share_one_configured_location(self):
        grids = configured_locations()
        self.assertEqual(len(grids), 1)
        self.assertEqual(grids[0]['location'], LOCATION)
        self.assertEqual(grids[0]['release_lag_hours'], 6)
        self.assertEqual(grids[0]['max_age_hours'], 24)

    def test_dry_run_and_disabled_mode_make_no_database_or_http_calls(self):
        engine = MagicMock()
        with patch.dict(os.environ, {'WEATHER_SYNC_ENABLED': 'false'}):
            self.assertEqual(sync_weather_forecasts(engine, Event()), {'status': 'disabled'})
        with patch.dict(os.environ, {'WEATHER_SYNC_ENABLED': 'true'}):
            result = sync_weather_forecasts(engine, Event(), dry_run=True)
        self.assertEqual(result['status'], 'validated')
        engine.connect.assert_not_called()

    def test_second_writer_skips_source_and_releases_connection(self):
        engine = MagicMock()
        connection = engine.connect.return_value.__enter__.return_value
        connection.scalar.return_value = False
        with patch('src.backend.ingestion.weather_live.requests.Session') as session:
            result = sync_weather_forecasts(engine, Event())
        self.assertEqual(result['status'], 'skipped_writer_busy')
        session.assert_not_called()
        engine.connect.return_value.__exit__.assert_called_once()


@skipUnless(os.environ.get('WEATHER_SIMULATION_URL'), 'Requires local Postman simulation')
class WeatherHttpTests(TestCase):
    def setUp(self):
        self.stop = MagicMock()
        self.stop.is_set.return_value = False
        self.stop.wait.return_value = False
        self.session = requests.Session()
        self.session.headers['x-mock-session'] = str(uuid4())
        self.addCleanup(self.session.close)
        self.cycle = recent_cycles(NOW, 6, 24)[0]
        self.base = os.environ['WEATHER_SIMULATION_URL'].rstrip('/') + '/weather'

    def fetch(self, path):
        return fetch_run(self.session, LOCATION, self.cycle, self.stop,
                         endpoint=self.base + path, timeout=(1, 2))

    def test_real_http_success_and_retry_after_rate_limit(self):
        params, payload, checksum = self.fetch('/retry')
        self.assertEqual(params['run'], '2026-10-22T06:00')
        self.assertEqual(payload['hourly']['precipitation'][1], 0.)
        self.assertEqual(len(checksum), 64)
        self.stop.wait.assert_called_once()

    def test_unavailable_run_can_fall_back_without_retry_storm(self):
        with self.assertRaises(WeatherSourceError) as error:
            self.fetch('/unavailable')
        self.assertEqual(error.exception.status_code, 404)
        self.stop.wait.assert_not_called()

    def test_html_error_retries_and_hides_response_body(self):
        with self.assertRaises(WeatherSourceError) as error:
            self.fetch('/html')
        self.assertEqual(error.exception.status_code, 502)
        self.assertNotIn('<html>', str(error.exception))
        self.assertEqual(self.stop.wait.call_count, 2)

    def test_invalid_rainfall_is_rejected_before_storage(self):
        with self.assertRaisesRegex(WeatherSourceError, 'Invalid weather forecast response'):
            self.fetch('/invalid')

    def test_stop_requested_during_retry_cancels_promptly(self):
        self.stop.wait.return_value = True
        with self.assertRaisesRegex(WeatherSourceError, 'cancelled'):
            self.fetch('/html')
