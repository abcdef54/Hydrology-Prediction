import unittest
import numpy as np
import torch
from research.change_weight_experiment import raw_change_weights,weighted_loss,sequence_stats
from src.ai.losses import HorizonWeightedTailLoss


class ChangeWeightTests(unittest.TestCase):
    def test_weights_symmetric_bounded_and_no_label_gradient(self):
        change=torch.tensor([-400.,-100.,0.,100.,400.],requires_grad=True)
        weight=raw_change_weights(change)
        torch.testing.assert_close(weight,torch.tensor([2.,1.5,1.,1.5,2.]))
        self.assertFalse(weight.requires_grad)
        normalized=weight/weight.mean()
        self.assertAlmostEqual(normalized.mean().item(),1.,places=6)

    def test_control_exact_and_physical_delta_uses_target_scale(self):
        loss=HorizonWeightedTailLoss([1,1],[100.,200.],strength=.5)
        pred=torch.tensor([[.5,.25]],requires_grad=True)
        target=torch.tensor([[1.,.5]])
        persistence=torch.zeros_like(pred)
        normalizer=np.array([1.5,1.5])
        torch.testing.assert_close(weighted_loss(loss,pred,target,persistence,normalizer,0),loss(pred,target),rtol=0,atol=0)
        # Both physical changes are 100 mm, so weights / normalizer == 1.
        value=weighted_loss(loss,pred,target,persistence,normalizer,1)
        torch.testing.assert_close(value,loss(pred,target))
        value.backward();self.assertTrue(torch.isfinite(pred.grad).all())
        with self.assertRaises(ValueError):raw_change_weights(target,boost=-1)


if __name__=='__main__':unittest.main()
