"""MAE is a physical-unit aggregate; threshold counts use individual errors."""
import math
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from src.ai.trainers import LSTMTrainer


class ValidationMAETests(unittest.TestCase):
    def trainer(self, model=None):
        trainer=object.__new__(LSTMTrainer)
        trainer.model=nn.Identity() if model is None else model
        trainer.device=torch.device('cpu')
        trainer.loss_func=nn.MSELoss()
        trainer.validation_threshold_mm=200.
        trainer.checkpoint_metric='large_error_rate'
        return trainer

    def test_pooled_and_horizon_mae_count_strict_and_partial_batch(self):
        errors=torch.tensor([[-100.,0.],[200.,-200.],[300.,100.]],dtype=torch.float64)
        scale=torch.tensor([10.,100.],dtype=torch.float64)
        data=TensorDataset(errors/scale,torch.zeros_like(errors))
        data.target_cols=['h6','h12']
        data.target_scaler=SimpleNamespace(scale_=np.array([10.,100.]),mean_=np.array([800.,-200.]))
        trainer=self.trainer()
        trainer.evaluate_loss(DataLoader(data,batch_size=2))
        m=trainer.last_validation_metrics
        self.assertAlmostEqual(m['val/mae_mm'],150.)
        self.assertAlmostEqual(m['val/h6/mae_mm'],200.)
        self.assertAlmostEqual(m['val/h12/mae_mm'],100.)
        self.assertAlmostEqual(m['val/rmse_mm'],math.sqrt(190000/6))
        self.assertEqual(m['val/large_error_count'],1)
        self.assertAlmostEqual(m['val/large_error_pct'],100/6)
        # Pooled RMSE is below 200, but one prediction still exceeds 200 mm.
        self.assertLess(m['val/rmse_mm'],200.)
        self.assertEqual(trainer.validation_score(0.),(1,m['val/rmse_mm']))

    def test_regional_per_sample_scales(self):
        class Echo(nn.Module):
            def forward(self,x):return x
        class Loader:
            dataset=SimpleNamespace(target_cols=['h6','h12'])
            def __iter__(self):
                yield dict(x=torch.ones(2,2),y=torch.zeros(2,2),
                           target_scale=torch.tensor([[10.,100.],[1000.,20.]]),
                           target_mean=torch.zeros(2,2))
        trainer=self.trainer(Echo())
        trainer.evaluate_loss(Loader())
        m=trainer.last_validation_metrics
        self.assertAlmostEqual(m['val/mae_mm'],282.5)
        self.assertAlmostEqual(m['val/h6/mae_mm'],505.)
        self.assertAlmostEqual(m['val/h12/mae_mm'],60.)
        self.assertEqual(m['val/large_error_count'],1)
        self.assertEqual(m['val/large_error_pct'],25.)

if __name__=='__main__':unittest.main()
