"""Linear baseline: prediction = persistence + standardized forecast channel mapped to target units + correction."""
import unittest

import torch

from src.ai.residual_sequence import ResidualGRU
from src.ai.sequence_checkpoint import model_from_spec, model_spec

SCALE, OFFSET = [[1.3, 1.4]], [[.1, .2]]
B_IDX, B_SCALE, B_OFFSET = [3, 4], [[.6, .7]], [[-.05, .02]]


def make(**kw):
    opts = dict(input_size=6, output_size=2, hidden_size=8, num_layers=2, dropout=.2, input_dropout=.1, head_dropout=.1,
                num_stations=1, persistence_scale=SCALE, persistence_offset=OFFSET, head_mode='shared',
                station_embedding_size=0)
    opts.update(kw)
    return ResidualGRU(**opts)


def with_baseline(**kw):
    return make(baseline_feature_indices=B_IDX, baseline_scale=B_SCALE, baseline_offset=B_OFFSET, **kw)


class LinearBaseline(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x, self.sid = torch.randn(5, 12, 6), torch.zeros(5, dtype=torch.long)

    def expected_base(self):
        pers = self.x[:, -1, 0, None] * torch.tensor(SCALE[0]) + torch.tensor(OFFSET[0])
        return pers + self.x[:, -1, B_IDX] * torch.tensor(B_SCALE[0]) + torch.tensor(B_OFFSET[0])

    def test_untrained_model_equals_linear_baseline(self):
        m = with_baseline().eval()
        self.assertTrue(torch.allclose(m(self.x, self.sid), self.expected_base(), atol=1e-6))
        self.assertTrue(torch.allclose(m.base(self.x, self.sid), self.expected_base(), atol=1e-6))
        # persistence() stays plain persistence (used by the loss to define the true change)
        pers = self.x[:, -1, 0, None] * torch.tensor(SCALE[0]) + torch.tensor(OFFSET[0])
        self.assertTrue(torch.allclose(m.persistence(self.x, self.sid), pers))

    def test_correction_adds_to_baseline_and_dropout_never_hides_it(self):
        m = with_baseline(group_dropout_indices=[[5]], group_dropout_mask_indices=[[]], group_dropout_rates=[.9],
                          input_dropout=.5)
        with torch.no_grad():
            for p in m.parameters():
                p.add_(torch.randn_like(p) * .1)
        m.eval()
        self.assertFalse(torch.allclose(m(self.x, self.sid), self.expected_base()))   # trained correction is added
        with torch.no_grad():                                                          # zero head -> output = base,
            m.linear.weight.zero_(); m.linear.bias.zero_()                             # even with every dropout on
        m.train()
        for seed in range(5):
            torch.manual_seed(seed)
            self.assertTrue(torch.allclose(m(self.x, self.sid), self.expected_base(), atol=1e-6))

    def test_default_unchanged_and_state_dict_identical(self):
        torch.manual_seed(1)
        plain = make().eval()
        torch.manual_seed(1)
        based = with_baseline().eval()
        self.assertEqual(set(plain.state_dict()), set(based.state_dict()))   # coefficients live in the spec only
        self.assertNotIn('baseline_feature_indices', plain.architecture_config)
        self.assertTrue(torch.allclose(plain(self.x, self.sid), plain.persistence(self.x, self.sid)))

    def test_spec_round_trip(self):
        m = with_baseline().eval()
        with torch.no_grad():
            for p in m.parameters():
                p.add_(torch.randn_like(p) * .1)
        r = model_from_spec(model_spec(m)).eval()
        r.load_state_dict(m.state_dict())
        self.assertTrue(torch.allclose(r(self.x, self.sid), m(self.x, self.sid)))
        self.assertEqual(r.architecture_config['baseline_feature_indices'], B_IDX)

    def test_single_station_shapes(self):
        m = make(num_stations=0, persistence_scale=SCALE[0], persistence_offset=OFFSET[0],
                 baseline_feature_indices=B_IDX, baseline_scale=B_SCALE[0], baseline_offset=B_OFFSET[0]).eval()
        self.assertTrue(torch.allclose(m(self.x), self.expected_base(), atol=1e-6))

    def test_invalid(self):
        for kw in [dict(baseline_feature_indices=[3]),                                   # one index per output
                   dict(baseline_feature_indices=[0, 4]),                                # level channel
                   dict(baseline_feature_indices=[3, 9]),                                # out of range
                   dict(baseline_feature_indices=[3, 4], baseline_scale=[[.6]]),         # wrong shape
                   dict(baseline_feature_indices=[3, 4], group_dropout_indices=[[3]],
                        group_dropout_mask_indices=[[]], group_dropout_rates=[.3]),      # baseline channel dropped
                   dict(baseline_feature_indices=[3, 4], prediction_mode='direct')]:
            opts = dict(baseline_scale=B_SCALE, baseline_offset=B_OFFSET)
            opts.update(kw)
            with self.assertRaises(ValueError, msg=str(kw)):
                make(**opts)
        with self.assertRaises(ValueError):
            make(baseline_scale=B_SCALE)


if __name__ == '__main__':
    unittest.main()
