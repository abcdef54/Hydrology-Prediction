"""Tail penalty of ChangeRiskLoss: tail_weight * (max(|e| - tail_start_mm, 0) / d)^2 on the raw error."""
import unittest

import torch

from src.ai.change_objective import ChangeRiskLoss


def loss(tail_weight=0., tail_start_mm=150., margin_mm=0., under_weight=0.):
    # scale 100 mm per normalized unit; d = max(scale, floor 200) = 200 mm
    return ChangeRiskLoss([1.], counts=[10], scales=[[100.]], mode='combined', event_weight=0.,
                          under_weight=under_weight, margin_mm=margin_mm,
                          tail_weight=tail_weight, tail_start_mm=tail_start_mm)


def value(fn, error_mm):
    y = torch.zeros(1, 1)
    p = torch.full((1, 1), error_mm / 100.)
    return fn(p, y, torch.zeros(1, 1), torch.zeros(1, dtype=torch.long)).item()


class ChangeTail(unittest.TestCase):
    def test_zero_weight_is_original(self):
        torch.manual_seed(0)
        p, y = torch.randn(8, 1), torch.randn(8, 1)
        args = (p, y, torch.zeros(8, 1), torch.zeros(8, dtype=torch.long))
        a = ChangeRiskLoss([1.], counts=[10], scales=[[100.]], mode='combined', event_weight=0., under_weight=0.3)
        self.assertTrue(torch.equal(a(*args), loss(0., under_weight=.3)(*args)))

    def test_penalty_starts_at_threshold_and_grows(self):
        for e in (100., 150.):                                  # below/at start: no extra cost
            self.assertAlmostEqual(value(loss(3.), e), value(loss(0.), e), places=6)
        for e, excess in ((200., 50.), (300., 150.), (-260., 110.)):
            extra = value(loss(3.), e) - value(loss(0.), e)
            self.assertAlmostEqual(extra, 3. * (excess / 200.) ** 2, places=5)

    def test_gradient_keeps_growing_beyond_200(self):
        grads = []
        for e in (210., 300.):
            p = torch.full((1, 1), e / 100., requires_grad=True)
            loss(3.)(p, torch.zeros(1, 1), torch.zeros(1, 1), torch.zeros(1, dtype=torch.long)).backward()
            grads.append(p.grad.item())
        self.assertGreater(grads[1], grads[0] > 0)

    def test_uses_raw_error_with_margin(self):
        # margin 100 shrinks the base term, but the tail still sees the raw 250 mm error (excess 100)
        extra = value(loss(2., margin_mm=100.), 250.) - value(loss(0., margin_mm=100.), 250.)
        self.assertAlmostEqual(extra, 2. * (100. / 200.) ** 2, places=5)

    def test_specification_and_validation(self):
        self.assertNotIn('tail_weight', loss(0.).specification())
        spec = loss(3., 140.).specification()
        self.assertEqual((spec['tail_weight'], spec['tail_start_mm']), (3., 140.))
        with self.assertRaises(ValueError):
            loss(-1.)
        with self.assertRaises(ValueError):
            loss(1., tail_start_mm=float('nan'))


if __name__ == '__main__':
    unittest.main()
