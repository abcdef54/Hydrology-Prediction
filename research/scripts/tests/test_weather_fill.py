"""Missing forecast resolution retains provenance and never uses a newer run."""
from unittest import TestCase
import numpy as np
import pandas as pd
from research.scripts.fill_missing_weather import fill_from_older
from research.scripts.weather_dataset_readiness import WINDOWS


class FillTests(TestCase):
    def frames(self):
        idx=pd.DatetimeIndex([pd.Timestamp('2026-09-01T18:00Z')])
        p=pd.DataFrame({'initialized_at':[pd.Timestamp('2026-09-01T06:00Z')],
                        'age_hours':[12.], 'fallback':[False]},index=idx)
        for a,b in WINDOWS:p[f'forecast_{a}_{b}h']=np.nan
        f=p.copy();f['initialized_at']=pd.Timestamp('2026-09-01T00:00Z');f['age_hours']=18.
        for a,b in WINDOWS:f[f'forecast_{a}_{b}h']=float(b-a)
        return p,f

    def test_replaces_whole_missing_anchor_and_records_source_age(self):
        p,f=self.frames();r=fill_from_older(p,f)
        self.assertTrue(r.iloc[0].was_filled)
        self.assertEqual(r.iloc[0].age_hours,18.)
        self.assertEqual(r.iloc[0].initialized_at,pd.Timestamp('2026-09-01T00:00Z'))
        self.assertFalse(r.filter(like='_missing').any().any())
        self.assertTrue(p.filter(like='forecast_').isna().all().all())

    def test_does_not_replace_complete_primary_with_alternative(self):
        p,f=self.frames()
        for a,b in WINDOWS:p[f'forecast_{a}_{b}h']=100.
        r=fill_from_older(p,f)
        self.assertFalse(r.iloc[0].was_filled)
        self.assertEqual(r.iloc[0].forecast_0_3h,100.)

    def test_rejects_newer_run_even_if_assumed_available(self):
        p,f=self.frames();f['initialized_at']=pd.Timestamp('2026-09-01T12:00Z')
        with self.assertRaises(ValueError):fill_from_older(p,f)
