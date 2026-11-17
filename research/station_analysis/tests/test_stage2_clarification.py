"""Checks for the focused clarification audit's diagnostic definitions."""
import unittest
import numpy as np
import pandas as pd
from research.station_analysis.phase2.stage2_clarification import contiguous_count, paired_stats


class ClarificationTests(unittest.TestCase):
    def test_contiguous_runs_are_not_hour_counts(self):
        mask=pd.Series([True,True,False,True,np.nan,True])
        self.assertEqual(contiguous_count(mask),3)

    def test_pairwise_count_preserves_missingness_and_scope(self):
        x=pd.Series([1.,2.,np.nan,4.,5.])
        y=pd.Series([2.,4.,6.,8.,np.nan])
        n,r=paired_stats(x,y,pd.Series([True,True,True,False,True]))
        self.assertEqual(n,2)
        self.assertTrue(np.isnan(r))
        n,r=paired_stats(x,y)
        self.assertEqual(n,3)
        self.assertAlmostEqual(r,1.)

    def test_constant_signal_does_not_get_spurious_correlation(self):
        n,r=paired_stats(pd.Series([1.,1.,1.]),pd.Series([1.,2.,3.]))
        self.assertEqual(n,3)
        self.assertTrue(np.isnan(r))
