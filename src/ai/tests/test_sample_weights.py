import unittest
from types import SimpleNamespace

import torch

from src.ai.change_objective import ChangeRiskLoss
from src.ai.trainers import LSTMTrainer


def loss():
    return ChangeRiskLoss([1., 1., 1.], counts=[10], scales=[[200., 200., 200.]], mode='combined',
                          event_weight=0., under_weight=0.3)


class Persistence(torch.nn.Module):
    def forward(self, x, station_ids, **_):
        return x[:, -1, :3]


def trainer():
    t = LSTMTrainer.__new__(LSTMTrainer)       # only data_loss is exercised
    t.loss_func = loss()
    t.model = SimpleNamespace(persistence=Persistence())
    return t


class SampleWeights(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(6, 4, 5)
        self.inputs = dict(x=self.x, station_ids=torch.zeros(6, dtype=torch.long))
        self.y = torch.randn(6, 3)

    def test_no_weight_is_identical(self):
        t, p = trainer(), torch.randn(6, 3)
        self.assertTrue(torch.equal(t.data_loss(p, self.y, None, self.inputs),
                                    t.data_loss(p, self.y, None, self.inputs, sample_weight=None)))

    def test_unit_weights_equal_unweighted(self):
        t, p = trainer(), torch.randn(6, 3)
        a = t.data_loss(p, self.y, None, self.inputs)
        b = t.data_loss(p, self.y, None, self.inputs, sample_weight=torch.ones(6))
        self.assertTrue(torch.allclose(a, b))

    def test_zero_weight_removes_gradient(self):
        t = trainer()
        p = torch.randn(6, 3, requires_grad=True)
        w = torch.tensor([1., 0., 1., 0.3, 1., 0.])
        t.data_loss(p, self.y, None, self.inputs, sample_weight=w).backward()
        self.assertTrue(torch.all(p.grad[[1, 5]] == 0))
        self.assertTrue(torch.all(p.grad[[0, 2, 4]].abs().sum(dim=1) > 0))

    def test_weight_scales_contribution(self):
        t, p = trainer(), torch.randn(6, 3)
        cells = t.loss_func.elements(p, self.y, self.x[:, -1, :3], self.inputs['station_ids'])
        w = torch.tensor([0.3, 1., 1., 1., 1., 1.])
        expected = t.loss_func.reduce_per_horizon(cells * w[:, None])
        self.assertTrue(torch.allclose(t.data_loss(p, self.y, None, self.inputs, sample_weight=w), expected))

    def test_rejects_other_losses(self):
        t = trainer()
        t.loss_func = torch.nn.MSELoss()
        with self.assertRaises(ValueError):
            t.data_loss(torch.randn(6, 3), self.y, None, self.inputs, sample_weight=torch.ones(6))


if __name__ == '__main__':
    unittest.main()
