"""Paired forecast schemas preserve labels, missing semantics and scaler fitting."""
import tempfile
from pathlib import Path
from unittest import TestCase

import numpy as np
import pandas as pd

from research.weather_forecast.build_datasets import encode_pair
from src.ai.dataset import HydrologyDataset
from src.ai.utils import get_dataloader, load_dataset


class WeatherDatasetTests(TestCase):
    def test_sentinel_drops_flags_keeps_zero_and_missing_labels(self):
        frame=pd.DataFrame({'water_level_depth':[100.,np.nan], 'water_level_missing':[0,1],
                            'rain_750007_depth':[0.,np.nan], 'rain_750007_missing':[0,1],
                            'target_water_level_plus_60m':[101.,np.nan]})
        features=[c for c in frame if not c.startswith('target_')]
        mask,minus,flags=encode_pair(frame,features)
        self.assertEqual(len(flags),2)
        self.assertFalse(any(c.endswith('_missing') for c in minus))
        self.assertEqual(minus.rain_750007_depth.iloc[0],0)
        self.assertEqual(minus.rain_750007_depth.iloc[1],-1)
        self.assertTrue(pd.isna(minus.target_water_level_plus_60m.iloc[1]))
        self.assertTrue(pd.isna(mask.water_level_depth.iloc[1]))
        frame.loc[0,'water_level_depth']=-1
        with self.assertRaises(ValueError):encode_pair(frame,features)

    def test_sentinel_excluded_from_scaler_and_survives_saved_scaler(self):
        train=pd.DataFrame({'water_level_depth':[100.,200.,-1.],
                            'target_water_level_plus_60m':[101.,201.,301.]})
        train.attrs.update(raw_missing_value=-1.,normalized_missing_value=-1.)
        ds=HydrologyDataset(train,fit_scalers=True,seq_len=1,require_current_level=True)
        self.assertEqual(ds.feature_scaler.mean_[0],150.)
        self.assertEqual(ds.X[-1,0],-1.)
        self.assertEqual(len(ds),2)  # Missing current water never becomes a valid residual anchor.
        test=pd.DataFrame({'water_level_depth':[-1.,1000.],
                           'target_water_level_plus_60m':[1.,2.]})
        restored=HydrologyDataset(test,feature_scaler=ds.feature_scaler,
            target_scaler=ds.target_scaler,seq_len=1,require_current_level=True)
        self.assertEqual(restored.X[0,0],-1.)
        self.assertEqual(restored.feature_scaler.mean_[0],150.)
        self.assertEqual(len(restored),1)

    def test_built_datasets_are_trainable_with_identical_anchor_targets(self):
        root=Path(__file__).resolve().parents[3]/'train_data'
        loaders={}
        for mode,width in [('mask',26),('neg1',19)]:
            folder=root/f'da_vien_1h_2026-10-15_forecast_{mode}'
            self.assertTrue(folder.exists())
            loaders[mode]=get_dataloader('1h',128,num_worker=0,pin_memory=False,drop_last=False,
                seq_len=96,require_current_level=True,
                feature_set='water_level+rain+flood_season+weather_forecast',dataset_dir=folder)
            for loader,count in zip(loaders[mode],[6095,985,592]):
                ds=loader.dataset
                self.assertEqual(ds.num_features,width)
                self.assertEqual(len(ds),count)
                self.assertTrue(np.isfinite(ds.X).all())
                self.assertEqual(ds.num_targets,5)
        for a,b in zip(loaders['mask'],loaders['neg1']):
            np.testing.assert_array_equal(a.dataset.end_indices,b.dataset.end_indices)
            np.testing.assert_array_equal(a.dataset.y,b.dataset.y)
            pd.testing.assert_series_equal(a.dataset.timestamps,b.dataset.timestamps)
        old=['water_level_missing','rain_750062_missing','rain_750007_missing']
        self.assertFalse(set(old)&set(loaders['neg1'][0].dataset.feature_cols))
        self.assertTrue(set(old)<=set(loaders['mask'][0].dataset.feature_cols))
        # Removing flags changes no scaler statistics for shared valid observations.
        a,b=loaders['mask'][0].dataset,loaders['neg1'][0].dataset
        for column in b.feature_cols:
            np.testing.assert_allclose(a.feature_scaler.mean_[a.feature_cols.index(column)],
                                       b.feature_scaler.mean_[b.feature_cols.index(column)])


    def test_comparison_reads_original_observations_for_app_schema(self):
        from src.ai.comparison import paired_metrics
        from src.ai.tests.test_comparison import example, LastLevel
        original,new,app=example()
        candidate=original.drop(columns=['rain_750007_depth'])
        _,_,_,coverage=paired_metrics(candidate,LastLevel(new,[0.,0.]),new,
            LastLevel(app,[0.,0.]),app,serving_frame=original)
        self.assertEqual(coverage['common_anchors'],3)
        wrong=original.copy();wrong.loc[0,'target_water_level_plus_60m']+=1
        with self.assertRaisesRegex(ValueError,'different target'):
            paired_metrics(candidate,LastLevel(new,[0.,0.]),new,
                LastLevel(app,[0.,0.]),app,serving_frame=wrong)
