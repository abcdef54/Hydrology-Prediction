"""horizon_heads='separate': per-horizon fusion layer and head on top of the shared encoder."""
import unittest

import torch

from src.ai.residual_sequence import HorizonCorrectionHeads, ResidualGRU
from src.ai.sequence_checkpoint import model_from_spec, model_spec


def make(**kw):
    opts = dict(input_size=5, output_size=3, hidden_size=8, num_layers=2, dropout=.2, input_dropout=.1, head_dropout=.1,
                num_stations=1, persistence_scale=[[1.3, 1.4, 1.5]], persistence_offset=[[.1, .2, .3]],
                head_mode='shared', station_embedding_size=0)
    opts.update(kw)
    return ResidualGRU(**opts)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


class SeparateHorizonHeads(unittest.TestCase):
    def test_structure_and_size(self):
        shared, sep = make(), make(horizon_heads='separate')
        self.assertEqual(len(sep.fusion), 3)
        self.assertIsInstance(sep.linear, HorizonCorrectionHeads)
        self.assertEqual(tuple(sep.linear.weight.shape), (3, 8))
        # two extra fusion layers (8x8 + 8 each); the heads have the same size (3x8 + 3)
        self.assertEqual(n_params(sep) - n_params(shared), 2 * (8 * 8 + 8))
        self.assertNotIn('horizon_heads', shared.architecture_config)

    def test_starts_at_persistence(self):
        torch.manual_seed(0)
        m = make(horizon_heads='separate').eval()
        x, sid = torch.randn(4, 12, 5), torch.zeros(4, dtype=torch.long)
        self.assertTrue(torch.allclose(m(x, sid), m.persistence(x, sid)))

    def test_horizons_are_decoupled(self):
        # changing horizon 0's fusion layer and head must leave the other horizons' outputs untouched
        torch.manual_seed(0)
        m = make(horizon_heads='separate').eval()
        with torch.no_grad():
            m.linear.weight.normal_()
        x, sid = torch.randn(4, 12, 5), torch.zeros(4, dtype=torch.long)
        before = m(x, sid)
        with torch.no_grad():
            m.fusion[0][0].weight.add_(1.)
            m.linear.weight[0].add_(1.)
        after = m(x, sid)
        self.assertFalse(torch.allclose(before[:, 0], after[:, 0]))
        self.assertTrue(torch.allclose(before[:, 1:], after[:, 1:]))

    def test_spec_round_trip(self):
        torch.manual_seed(1)
        m = make(horizon_heads='separate').eval()
        with torch.no_grad():
            for p in m.parameters():
                p.add_(torch.randn_like(p) * .1)
        r = model_from_spec(model_spec(m)).eval()
        r.load_state_dict(m.state_dict())
        x, sid = torch.randn(3, 12, 5), torch.zeros(3, dtype=torch.long)
        self.assertEqual(r.horizon_heads, 'separate')
        self.assertTrue(torch.allclose(r(x, sid), m(x, sid)))

    def test_head_size_and_layers(self):
        m = make(horizon_heads='separate', horizon_head_size=16, horizon_head_layers=2)
        for head in m.fusion:
            linears = [l for l in head if isinstance(l, torch.nn.Linear)]
            self.assertEqual([(l.in_features, l.out_features) for l in linears], [(8, 16), (16, 16)])
        self.assertEqual(tuple(m.linear.weight.shape), (3, 16))
        # per horizon: 8*16+16 + 16*16+16 hidden, 16+1 output; shared encoder unchanged
        base = make(horizon_heads='separate')
        self.assertEqual(n_params(m) - n_params(base),
                         3 * ((8 * 16 + 16 + 16 * 16 + 16 + 16 + 1) - (8 * 8 + 8 + 8 + 1)))
        self.assertEqual((m.architecture_config['horizon_head_size'], m.architecture_config['horizon_head_layers']), (16, 2))
        torch.manual_seed(0)
        x, sid = torch.randn(4, 12, 5), torch.zeros(4, dtype=torch.long)
        m.eval()
        self.assertTrue(torch.allclose(m(x, sid), m.persistence(x, sid)))
        r = model_from_spec(model_spec(m))
        r.load_state_dict(m.state_dict())

    def test_default_size_keeps_layer_names(self):
        # default separate heads (1 layer, width = hidden size) keep the parameter names fusion.<k>.0.*
        names = set(make(horizon_heads='separate').state_dict())
        self.assertIn('fusion.2.0.weight', names)
        self.assertNotIn('fusion.2.2.weight', names)

    def test_horizon_gru_branches(self):
        # 3 GRU layers: 1 shared, 2 per horizon
        m = make(horizon_heads='separate', num_layers=3, horizon_gru_layers=2)
        self.assertEqual(m.encoder.recurrent.num_layers, 1)
        self.assertEqual(len(m.horizon_grus), 3)
        self.assertTrue(all(g.num_layers == 2 for g in m.horizon_grus))
        gru_layer = lambda i, h: 3 * (i * h + h * h + 2 * h)
        shared_3 = make(horizon_heads='separate', num_layers=3)
        # extra parameters = 2 more copies of the top two GRU layers (8 -> 8)
        self.assertEqual(n_params(m) - n_params(shared_3), 2 * 2 * gru_layer(8, 8))
        torch.manual_seed(0)
        m.eval()
        x, sid = torch.randn(4, 12, 5), torch.zeros(4, dtype=torch.long)
        self.assertTrue(torch.allclose(m(x, sid), m.persistence(x, sid)))
        # decoupled: perturbing horizon 2's GRU branch leaves horizons 0 and 1 untouched
        with torch.no_grad():
            m.linear.weight.normal_()
            before = m(x, sid)
            for p in m.horizon_grus[2].parameters():
                p.add_(.5)
            after = m(x, sid)
        self.assertTrue(torch.allclose(before[:, :2], after[:, :2]))
        self.assertFalse(torch.allclose(before[:, 2], after[:, 2]))
        r = model_from_spec(model_spec(m)).eval()
        r.load_state_dict(m.state_dict())
        self.assertTrue(torch.allclose(r(x, sid), m(x, sid)))
        # the shared encoder is used by every horizon: perturbing it changes all outputs
        with torch.no_grad():
            for p in m.encoder.parameters():
                p.add_(.5)
        self.assertFalse(torch.allclose(m(x, sid)[:, 0], after[:, 0]))

    def test_horizon_gru_invalid(self):
        with self.assertRaises(ValueError):
            make(horizon_heads='separate', num_layers=3, horizon_gru_layers=3)      # nothing shared
        with self.assertRaises(ValueError):
            make(num_layers=3, horizon_gru_layers=1)                                # needs separate heads

    def test_invalid_combinations(self):
        with self.assertRaises(ValueError):
            make(horizon_heads='other')
        with self.assertRaises(ValueError):
            make(horizon_heads='separate', head_mode='station', station_embedding_size=8)
        with self.assertRaises(ValueError):
            make(horizon_head_size=16)                                  # needs separate heads
        with self.assertRaises(ValueError):
            make(horizon_heads='separate', horizon_head_layers=0)


if __name__ == '__main__':
    unittest.main()
