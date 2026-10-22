"""Forecast contracts and rollback-only PostgreSQL vintage tests."""
from datetime import datetime, timedelta, timezone
import unittest
import pandas as pd
import sqlalchemy as sa
from dotenv import load_dotenv
from src.backend.db.session import build_engine
from src.backend.db.weather import locations, runs, hourly
from src.backend.ingestion.weather_forecast import cycles, parse_response, save_response, coverage

T = datetime(2025,11,14,12,tzinfo=timezone.utc)

def payload():
    return dict(latitude=16.27, longitude=107.26, utc_offset_seconds=0,
                hourly_units={'precipitation':'mm', 'time':'unixtime'}, hourly={
                    'time':[int((T+timedelta(hours=i)).timestamp()) for i in range(73)],
                    'precipitation':[None]+[0.0]*72})

class ContractTests(unittest.TestCase):
    def test_cycles_include_first_available_and_no_future_run(self):
        stamps = cycles(pd.Timestamp('2025-11-14T19:00Z'), pd.Timestamp('2025-11-15T07:00Z'))
        self.assertEqual(stamps[0], T)
        self.assertEqual(stamps[-1], T+timedelta(hours=12))

    def test_null_and_zero_are_distinct_and_time_is_interval_end(self):
        p = payload()
        p['hourly']['precipitation'][3] = None
        rows = parse_response(p,T)
        self.assertEqual(rows[0]['valid_at'], T+timedelta(hours=1))
        self.assertEqual(rows[0]['precipitation_mm'],0)
        self.assertEqual(rows[2]['quality_status'],'missing')

    def test_bad_unit_negative_nan_duplicates_wrong_run_rejected(self):
        for mutation in ('unit','negative','nan','duplicate','wrongrun'):
            with self.subTest(mutation=mutation):
                p = payload()
                if mutation == 'unit': p['hourly_units']['precipitation']='inch'
                if mutation == 'negative': p['hourly']['precipitation'][2]=-1
                if mutation == 'nan': p['hourly']['precipitation'][2]=float('nan')
                if mutation == 'duplicate': p['hourly']['time'][2]=p['hourly']['time'][1]
                if mutation == 'wrongrun': p['hourly']['time']=[t+86400 for t in p['hourly']['time']]
                with self.assertRaises(ValueError): parse_response(p,T)

class DatabaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_dotenv()
        cls.engine = build_engine()
    @classmethod
    def tearDownClass(cls): cls.engine.dispose()
    def test_idempotence_revisions_and_missing_coverage(self):
        with self.engine.connect() as c:
            tx=c.begin()
            try:
                location=dict(id='test-weather-rollback',station_id='750007',latitude=16.22,longitude=107.17)
                c.execute(sa.insert(locations).values(**location))
                self.assertTrue(save_response(c,location,T,{},payload(),'a'*64,T))
                self.assertFalse(save_response(c,location,T,{},payload(),'a'*64,T))
                p=payload(); p['hourly']['precipitation'][8]=None
                self.assertTrue(save_response(c,location,T,{},p,'b'*64,T+timedelta(days=1)))
                self.assertEqual(c.scalar(sa.select(sa.func.count()).select_from(runs).where(runs.c.location_id==location['id'])),2)
                parts={'train':pd.DatetimeIndex([T+timedelta(hours=6)])}
                report=coverage(c,location,parts,6)
                self.assertEqual(report['train']['complete_24h'],0)
                self.assertIsNone(c.scalar(sa.select(runs.c.available_at).where(runs.c.location_id==location['id']).limit(1)))
            finally: tx.rollback()

    def test_operational_availability_is_preserved_without_backdating(self):
        with self.engine.connect() as connection:
            transaction = connection.begin()
            try:
                location = dict(id='test-weather-live-rollback', station_id='750007',
                                latitude=16.22, longitude=107.17)
                connection.execute(sa.insert(locations).values(**location))
                retrieved = T + timedelta(hours=7, minutes=15)
                options = dict(available_at=retrieved, availability_basis='observed_retrieval',
                               archive_kind='operational_single_run')
                self.assertTrue(save_response(connection, location, T, {}, payload(),
                                              'c' * 64, retrieved, **options))
                self.assertFalse(save_response(connection, location, T, {}, payload(),
                                               'c' * 64, retrieved + timedelta(hours=1), **options))
                row = connection.execute(sa.select(runs).where(
                    runs.c.location_id == location['id'])).mappings().one()
                self.assertEqual(row['available_at'], retrieved)
                self.assertEqual(row['availability_basis'], 'observed_retrieval')
                self.assertEqual(row['archive_kind'], 'operational_single_run')
                self.assertGreater(row['available_at'], T + timedelta(hours=6))
                from src.backend.ingestion.weather_live import complete_run

                self.assertTrue(complete_run(connection, location['id'], T))
                revised = payload()
                revised['hourly']['precipitation'][8] = None
                self.assertTrue(save_response(connection, location, T, {}, revised,
                                              'd' * 64, retrieved + timedelta(hours=1),
                                              available_at=retrieved + timedelta(hours=1),
                                              availability_basis='observed_retrieval',
                                              archive_kind='operational_single_run'))
                self.assertFalse(complete_run(connection, location['id'], T))
            finally:
                transaction.rollback()
