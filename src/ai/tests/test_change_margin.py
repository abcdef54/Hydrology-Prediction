import unittest

import torch

from src.ai.change_objective import ChangeRiskLoss


def loss(margin):
    return ChangeRiskLoss([1., 1., 1.], counts=[10], scales=[[100., 100., 100.]], mode='combined',
                          event_weight=0., under_weight=0.3, margin_mm=margin)


class ChangeMargin(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.y = torch.randn(8, 3)
        self.base = torch.zeros(8, 3)
        self.ids = torch.zeros(8, dtype=torch.long)

    def test_zero_margin_is_original(self):
        p = torch.randn(8, 3)
        a = ChangeRiskLoss([1., 1., 1.], counts=[10], scales=[[100.] * 3], mode='combined', event_weight=0., under_weight=0.3)
        self.assertTrue(torch.equal(a(p, self.y, self.base, self.ids), loss(0.)(p, self.y, self.base, self.ids)))

    def test_small_errors_cost_nothing(self):
        p = (self.y + 0.4).requires_grad_()                      # 0.4 * 100 mm = 40 mm < 50 mm margin
        out = loss(50.)(p, self.y, self.base, self.ids)
        out.backward()
        self.assertEqual(out.item(), 0.)
        self.assertTrue(torch.all(p.grad == 0))

    def test_large_errors_penalised_on_excess(self):
        p = self.y + 3.                                           # 300 mm error
        cells = loss(50.).elements(p, self.y, self.base, self.ids)
        ref = loss(0.).elements(self.y + 2.5, self.y, self.base, self.ids)    # 250 mm = 300 - 50
        self.assertTrue(torch.allclose(cells, ref))

    def test_invalid_margin(self):
        with self.assertRaises(ValueError):
            loss(-1.)


if __name__ == '__main__':
    unittest.main()
