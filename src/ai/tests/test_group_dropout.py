import unittest

import torch

from src.ai.residual_sequence import residual_model
from src.ai.sequence_checkpoint import model_from_spec, model_spec


def make(**extra):
    torch.manual_seed(0)
    return residual_model('gru-late-fusion', input_size=8, hidden_size=8, num_layers=1, output_size=3, num_stations=1,
                          head_mode='station', persistence_scale=[[1., 1., 1.]], persistence_offset=[[0., 0., 0.]], **extra)


GROUPS = dict(group_dropout_indices=[[3, 4], [6]], group_dropout_mask_indices=[[5], []], group_dropout_rates=[0.5, 0.3])


class GroupDropout(unittest.TestCase):
    def test_default_has_no_group_config(self):
        self.assertNotIn('group_dropout_indices', make().architecture_config)

    def test_eval_mode_is_identity(self):
        m = make(**GROUPS).eval()
        x = torch.randn(16, 12, 8)
        self.assertTrue(torch.equal(m.apply_group_dropout(x), x))

    def test_train_mode_replaces_whole_group_over_history(self):
        m = make(group_dropout_indices=[[3, 4]], group_dropout_mask_indices=[[5]], group_dropout_rates=[0.999999]).train()
        x = torch.randn(32, 12, 8)
        y = m.apply_group_dropout(x)
        self.assertTrue(torch.all(y[:, :, [3, 4]] == 0))
        self.assertTrue(torch.all(y[:, :, 5] == 1))
        keep = [0, 1, 2, 6, 7]
        self.assertTrue(torch.equal(y[:, :, keep], x[:, :, keep]))
        self.assertFalse(torch.all(x[:, :, [3, 4]] == 0))             # input tensor not modified in place

    def test_rate_is_per_sample(self):
        m = make(group_dropout_indices=[[3]], group_dropout_mask_indices=[[]], group_dropout_rates=[0.5]).train()
        torch.manual_seed(1)
        x = torch.ones(4000, 5, 8)
        dropped = (m.apply_group_dropout(x)[:, :, 3] == 0).all(dim=1).float().mean().item()
        self.assertAlmostEqual(dropped, 0.5, delta=0.05)

    def test_forward_and_spec_roundtrip(self):
        m = make(**GROUPS)
        spec = model_spec(m)
        self.assertEqual(spec['options']['group_dropout_rates'], [0.5, 0.3])
        r = model_from_spec(spec)
        r.load_state_dict(m.state_dict(), strict=True)
        x, ids = torch.randn(5, 12, 8), torch.zeros(5, dtype=torch.long)
        m.eval(); r.eval()
        self.assertTrue(torch.allclose(m(x, ids), r(x, ids)))
        m.train()
        self.assertEqual(m(x, ids).shape, (5, 3))

    def test_invalid_groups(self):
        with self.assertRaises(ValueError):
            make(group_dropout_indices=[[0]], group_dropout_mask_indices=[[]], group_dropout_rates=[0.5])     # level channel
        with self.assertRaises(ValueError):
            make(group_dropout_indices=[[3], [3]], group_dropout_mask_indices=[[], []], group_dropout_rates=[0.5, 0.5])
        with self.assertRaises(ValueError):
            make(group_dropout_indices=[[3]], group_dropout_mask_indices=[[]], group_dropout_rates=[1.0])


if __name__ == '__main__':
    unittest.main()
