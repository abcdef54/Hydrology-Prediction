"""Full live collector against Postman HTTP and rollback-only PostgreSQL writes."""

from datetime import datetime, timezone
import os
from threading import Event
from unittest import TestCase, skipUnless
from unittest.mock import MagicMock, patch

from dotenv import load_dotenv
import sqlalchemy as sa

from src.backend.db.session import build_engine
from src.backend.db.weather import runs
from src.backend.ingestion.weather_live import sync_weather_forecasts


@skipUnless(os.environ.get('WEATHER_SIMULATION_URL'), 'Requires local Postman simulation')
class LiveCollectorDatabaseTests(TestCase):
    def test_collection_fallback_resume_and_atomic_storage(self):
        load_dotenv()
        engine = build_engine()
        location = dict(id='test-live-weather-rollback', station_id='750007',
                        latitude=16.221111, longitude=107.179444)
        grid = dict(location=location, release_lag_hours=6, max_age_hours=24)
        endpoint = os.environ['WEATHER_SIMULATION_URL'].rstrip('/') + '/weather/fallback'
        with engine.connect() as connection:
            outer = connection.begin()
            # Each production transaction uses a savepoint inside our outer
            # rollback. The collector still executes its real PostgreSQL SQL.
            borrowed = MagicMock()
            borrowed.execute.side_effect = connection.execute
            borrowed.scalar.side_effect = connection.scalar
            borrowed.begin.side_effect = connection.begin_nested
            borrowed.in_transaction.return_value = False
            borrowed.invalidated = False
            proxy_engine = MagicMock()
            proxy_engine.connect.return_value.__enter__.return_value = borrowed
            try:
                with patch('src.backend.ingestion.weather_live.configured_locations', return_value=[grid]), \
                     patch.dict(os.environ, {'WEATHER_SYNC_ENDPOINT': endpoint,
                                            'WEATHER_SYNC_ENABLED': 'true'}):
                    now = datetime(2026, 10, 20, 13, tzinfo=timezone.utc)
                    result = sync_weather_forecasts(proxy_engine, Event(), now=now)
                    self.assertEqual(result['status'], 'partial')
                    self.assertEqual(result['locations'][0]['fetched'], 2)
                    self.assertEqual(result['locations'][0]['failures'][0]['http_status'], 404)
                    self.assertEqual(result['locations'][0]['latest_age_hours'], 13.)
                    self.assertFalse(result['locations'][0]['stale'])
                    again = sync_weather_forecasts(proxy_engine, Event(), now=now)
                    self.assertEqual(again['locations'][0]['fetched'], 0)
                    self.assertEqual(again['locations'][0]['skipped'], 2)
                rows = connection.execute(sa.select(runs).where(
                    runs.c.location_id == location['id'])).mappings().all()
                self.assertEqual(len(rows), 2)
                self.assertTrue(all(row['availability_basis'] == 'observed_retrieval' for row in rows))
                self.assertTrue(all(row['available_at'] == row['retrieved_at'] for row in rows))
            finally:
                outer.rollback()
        engine.dispose()
