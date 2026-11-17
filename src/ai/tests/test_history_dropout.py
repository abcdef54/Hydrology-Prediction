import unittest

import torch

from src.ai.residual_sequence import residual_model
from src.ai.sequence_checkpoint import model_from_spec, model_spec

# channels: 0 level, 1 level missing mask, 2 flood_season (kept), 3 rain value, 4 rain missing_frac, 5 tide (kept)
HD = dict(history_dropout_rate=0.5, history_dropout_min=4, history_dropout_value_indices=[0, 3], history_dropout_mask_indices=[1, 4])


def make(**extra):
    torch.manual_seed(0)
    return residual_model('gru-late-fusion', input_size=6, hidden_size=8, num_layers=1, output_size=3, num_stations=1,
                          head_mode='station', persistence_scale=[[1., 1., 1.]], persistence_offset=[[0., 0., 0.]], **extra)


class HistoryDropout(unittest.TestCase):
    def test_default_off_and_eval_identity(self):
        self.assertNotIn('history_dropout_rate', make().architecture_config)
        m = make(**HD).eval()
        x = torch.randn(8, 24, 6)
        self.assertTrue(torch.equal(m.apply_history_dropout(x), x))

    def test_hides_only_older_steps_and_keeps_anchor(self):
        m = make(**{**HD, 'history_dropout_rate': 0.999999}).train()
        torch.manual_seed(3)
        x = torch.randn(256, 24, 6) + 5.
        y = m.apply_history_dropout(x)
        self.assertTrue(torch.equal(y[:, -4:], x[:, -4:]))                 # at least `min` recent steps intact, anchor kept
        self.assertTrue(torch.equal(y[:, :, [2, 5]], x[:, :, [2, 5]]))     # known channels never hidden
        hidden = (y[:, :, 0] == 0)
        for i in range(len(x)):                                            # hidden steps form one leading block (no gaps)
            k = int(hidden[i].sum())
            self.assertTrue(hidden[i, :k].all() and not hidden[i, k:].any())
            self.assertTrue(torch.all(y[i, :k, [1, 4]] == 1) and torch.all(y[i, :k, 3] == 0))
            self.assertLessEqual(k, 20)
        self.assertGreater(int(hidden.any(dim=1).sum()), 200)

    def test_rate_and_kept_length_distribution(self):
        m = make(**HD).train()
        torch.manual_seed(1)
        x = torch.ones(6000, 12, 6)
        y = m.apply_history_dropout(x)
        hidden = (y[:, :, 0] == 0).sum(dim=1)
        frac = (hidden > 0).float().mean().item()
        self.assertAlmostEqual(frac, 0.5 * (1 - 1 / 9), delta=0.04)        # k = T (12) hides nothing: 1 of 9 draws
        self.assertLessEqual(int(hidden.max()), 12 - 4)

    def test_spec_roundtrip_and_forward(self):
        m = make(**HD)
        spec = model_spec(m)
        self.assertEqual(spec['options']['history_dropout_min'], 4)
        r = model_from_spec(spec)
        r.load_state_dict(m.state_dict(), strict=True)
        x, ids = torch.randn(5, 24, 6), torch.zeros(5, dtype=torch.long)
        m.eval(); r.eval()
        self.assertTrue(torch.allclose(m(x, ids), r(x, ids)))
        m.train()
        self.assertEqual(m(x, ids).shape, (5, 3))

    def test_short_window_shorter_than_min(self):
        m = make(**{**HD, 'history_dropout_min': 50, 'history_dropout_rate': 0.999999}).train()
        x = torch.randn(4, 6, 6)
        self.assertTrue(torch.equal(m.apply_history_dropout(x), x))       # nothing to hide when min >= T

    def test_invalid(self):
        with self.assertRaises(ValueError):
            make(history_dropout_rate=1.0, history_dropout_min=4, history_dropout_value_indices=[0])
        with self.assertRaises(ValueError):
            make(history_dropout_rate=0.5, history_dropout_min=0, history_dropout_value_indices=[0])
        with self.assertRaises(ValueError):
            make(history_dropout_rate=0.5, history_dropout_min=2, history_dropout_value_indices=[0], history_dropout_mask_indices=[0])


if __name__ == '__main__':
    unittest.main()
