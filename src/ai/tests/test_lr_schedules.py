import unittest
import torch

from src.ai.lr_schedules import warmup_cosine


class LearningRateScheduleTests(unittest.TestCase):
    def test_three_epoch_warmup_and_cosine_reaches_minimum(self):
        param = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.AdamW([param], lr=2e-4, weight_decay=1e-3)
        scheduler = warmup_cosine(optimizer, total_epochs=60, warmup_epochs=3, min_lr=1e-6)
        rates = []
        for _ in range(60):
            rates.append(optimizer.param_groups[0]['lr'])
            optimizer.step()
            scheduler.step()
        for actual, expected in zip(rates[:4], [2e-4/3, 4e-4/3, 2e-4, 2e-4]):
            self.assertAlmostEqual(actual, expected, places=12)
        self.assertAlmostEqual(rates[-1], 1e-6, places=12)
        self.assertTrue(all(a >= b for a, b in zip(rates[3:], rates[4:])))
        self.assertEqual(optimizer.defaults['weight_decay'], 1e-3)

    def test_invalid_schedule_is_rejected(self):
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(()))], lr=2e-4)
        for args in [dict(total_epochs=4, warmup_epochs=3, min_lr=1e-6),
                     dict(total_epochs=60, warmup_epochs=0, min_lr=1e-6),
                     dict(total_epochs=60, warmup_epochs=3, min_lr=3e-4)]:
            with self.assertRaises(ValueError):
                warmup_cosine(optimizer, **args)


if __name__ == '__main__':
    unittest.main()
