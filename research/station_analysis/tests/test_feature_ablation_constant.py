import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import pandas as pd
from research.station_analysis.phase2.feature_ablation.explanations import grouped_gain

class ConstantAblationTests(unittest.TestCase):
    def test_empty_importance_means_constant_model_not_missing_experiment(self):
        with TemporaryDirectory() as root:
            p=Path(root)/'fits/key';p.mkdir(parents=True)
            (p/'identity.json').write_text(json.dumps({'metadata':{'station_id':'001'},'seed':42}))
            (p/'importance.csv').write_text('\n')
            result=grouped_gain(root,pd.DataFrame({'fit_key':['key']}))
            self.assertEqual(set(result.horizon),{6,12})
            self.assertTrue(result.group.eq('constant_prediction_no_splits').all())
            self.assertTrue(result.gain_fraction.eq(0).all())
