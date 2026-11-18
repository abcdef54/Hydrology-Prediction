"""init_from_checkpoint: copy every weight, keep the new persistence coefficients, select head rows per horizon."""
import tempfile
import unittest
from pathlib import Path

import joblib
import torch

from src.ai.finetune import init_from_checkpoint
from src.ai.residual_sequence import ResidualGRU
from src.ai.sequence_checkpoint import model_spec

COLS = ['water_level_depth', 'water_level_missing', 'flood_season', 'a', 'b']


def make(out, scale, head_mode='station', **kw):
    return ResidualGRU(input_size=5, output_size=out, hidden_size=8, num_layers=2, num_stations=1, head_mode=head_mode,
                       persistence_scale=[scale], persistence_offset=[[0.] * out], **kw)


class Finetune(unittest.TestCase):
    def save(self, m, horizons, cols=COLS):
        d = Path(tempfile.mkdtemp())
        torch.save(m.state_dict(), d / 'model.pt')
        joblib.dump(dict(feature_columns=cols, model_spec=model_spec(m), regional=dict(target_horizons=horizons)),
                    d / 'model.preprocessing.joblib')
        return d / 'model.pt'

    def test_select_horizon_and_keep_persistence(self):
        torch.manual_seed(0)
        src = make(3, [1.3, 1.4, 1.5])
        with torch.no_grad():
            for p in src.parameters():
                p.add_(torch.randn_like(p) * .1)
        src.eval()
        path = self.save(src, [3, 6, 12])
        dst = make(1, [2.0])
        rep = init_from_checkpoint(dst, path, COLS, [6])
        dst.eval()
        self.assertEqual(rep['sliced'], ['linear.weight', 'linear.bias'])
        self.assertTrue(torch.equal(dst.persistence_scale, torch.tensor([[2.0]])))      # new dataset's coefficient
        x, sid = torch.randn(4, 12, 5), torch.zeros(4, dtype=torch.long)
        corr_src = src(x, sid)[:, 1] - src.persistence(x, sid)[:, 1]
        corr_dst = dst(x, sid)[:, 0] - dst.persistence(x, sid)[:, 0]
        self.assertTrue(torch.allclose(corr_src, corr_dst, atol=1e-6))                  # same learned +6 h correction

    def test_shared_head_and_same_horizons(self):
        torch.manual_seed(1)
        src = make(3, [1., 1., 1.], head_mode='shared', station_embedding_size=0)
        path = self.save(src, [3, 6, 12])
        dst = make(1, [1.], head_mode='shared', station_embedding_size=0)
        init_from_checkpoint(dst, path, COLS, [12])
        self.assertTrue(torch.equal(dst.linear.weight, src.linear.weight[2:3]))
        full = make(3, [1., 1., 1.], head_mode='shared', station_embedding_size=0)
        rep = init_from_checkpoint(full, path, COLS, [3, 6, 12])
        self.assertEqual(rep['sliced'], [])

    def test_mismatch_raises(self):
        src = make(3, [1., 1., 1.])
        path = self.save(src, [3, 6, 12])
        with self.assertRaises(ValueError):
            init_from_checkpoint(make(1, [1.]), path, COLS[:-1] + ['c'], [6])             # columns
        with self.assertRaises(ValueError):
            init_from_checkpoint(make(1, [1.]), path, COLS, [24])                          # horizon
        with self.assertRaises(ValueError):
            init_from_checkpoint(make(1, [1.], head_mode='shared', station_embedding_size=0), path, COLS, [6])


if __name__ == '__main__':
    unittest.main()
