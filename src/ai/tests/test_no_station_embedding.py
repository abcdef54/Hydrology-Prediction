"""station_embedding_size=0: single-station residual models without the (constant) station embedding."""
import unittest

import torch

from src.ai.residual_sequence import ResidualGRU
from src.ai.sequence_checkpoint import model_from_spec, model_spec


def make(**kw):
    opts = dict(input_size=5, output_size=3, hidden_size=8, num_layers=2, dropout=.2, input_dropout=.1, head_dropout=.1,
                num_stations=1, persistence_scale=[[1.3, 1.4, 1.5]], persistence_offset=[[.1, .2, .3]])
    opts.update(kw)
    return ResidualGRU(**opts)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


class NoStationEmbedding(unittest.TestCase):
    def test_structure_and_size(self):
        with_emb = make(head_mode='station')
        no_emb = make(head_mode='shared', station_embedding_size=0)
        self.assertIsNone(no_emb.station_embedding)
        self.assertEqual(no_emb.fusion[0].in_features, 8)
        self.assertIsInstance(no_emb.linear, torch.nn.Linear)
        # removed: the embedding (1x8) and its 8 columns in the fusion layer (8*H); both heads have 3*H+3 parameters
        self.assertEqual(n_params(with_emb) - n_params(no_emb), 8 + 8 * 8)

    def test_starts_at_persistence(self):
        torch.manual_seed(0)
        m = make(head_mode='shared', station_embedding_size=0).eval()
        x, sid = torch.randn(4, 12, 5), torch.zeros(4, dtype=torch.long)
        self.assertTrue(torch.allclose(m(x, sid), m.persistence(x, sid)))
        expected = x[:, -1, 0, None] * torch.tensor([1.3, 1.4, 1.5]) + torch.tensor([.1, .2, .3])
        self.assertTrue(torch.allclose(m.persistence(x, sid), expected))

    def test_spec_round_trip(self):
        torch.manual_seed(1)
        m = make(head_mode='shared', station_embedding_size=0).eval()
        with torch.no_grad():
            for p in m.parameters():
                p.add_(torch.randn_like(p) * .1)
        r = model_from_spec(model_spec(m)).eval()
        r.load_state_dict(m.state_dict())
        x, sid = torch.randn(3, 12, 5), torch.zeros(3, dtype=torch.long)
        self.assertIsNone(r.station_embedding)
        self.assertTrue(torch.allclose(r(x, sid), m(x, sid)))

    def test_default_unchanged_and_invalid_combinations(self):
        self.assertEqual(tuple(make(head_mode='station').station_embedding.weight.shape), (1, 8))
        with self.assertRaises(ValueError):
            make(head_mode='station', station_embedding_size=0)
        with self.assertRaises(ValueError):
            make(head_mode='shared', station_embedding_size=-1)


if __name__ == '__main__':
    unittest.main()
