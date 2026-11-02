import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
import torch
from src.ai.change_objective import ChangeRiskLoss, from_training_dataset
from src.ai.change_diagnostics import ChangeDiagnostics
from src.ai.residual_sequence import ResidualSequence
from src.ai.trainers import ResidualLSTMTrainer


class ChangeObjectiveTests(unittest.TestCase):
    def loss(self, mode='combined', counts=(2, 2), scales=((100., 300.), (10., 600.)), **kw):
        return ChangeRiskLoss([1, 1], counts, scales, mode, **kw)

    def test_scaling_roundtrip_and_persistence(self):
        # Different station means/std and horizon coefficients; last timestep only.
        water_mean = torch.tensor([100., -700.])
        water_std = torch.tensor([20., 500.])
        target_mean = torch.tensor([[80., 200.], [-800., -600.]])
        target_std = torch.tensor([[10., 40.], [200., 800.]])
        m = ResidualSequence(architecture='lstm', input_size=2, hidden_size=4, num_layers=1,
            num_stations=2, persistence_scale=water_std[:, None]/target_std,
            persistence_offset=(water_mean[:, None]-target_mean)/target_std)
        x = torch.randn(2, 4, 2)
        ids = torch.arange(2)
        p = m.persistence(x, ids)
        physical = p*target_std+target_mean
        expected = x[:, -1, 0]*water_std+water_mean
        torch.testing.assert_close(physical, expected[:, None].expand(-1, 2), atol=1e-4, rtol=1e-5)
        torch.testing.assert_close(m(x, ids), p)

    def test_frequency_balance_independent_of_batch_composition(self):
        loss = self.loss('station-balanced', counts=(3, 1), scales=((200., 200.), (200., 200.)))
        y, b = torch.zeros(4, 2), torch.zeros(4, 2)
        p = torch.tensor([[1., 1.], [1., 1.], [1., 1.], [3., 3.]])
        ids = torch.tensor([0, 0, 0, 1])
        self.assertAlmostEqual(loss(p, y, b, ids).item(), 5.)
        # Equal station mean losses: (1+9)/2, not present-station minibatch mean.
        full = loss(p, y, b, ids)
        pieces = sum(loss(p[i:i+1], y[i:i+1], b[i:i+1], ids[i:i+1]) for i in range(4))/4
        torch.testing.assert_close(full, pieces)

    def test_floor_controls_small_scale(self):
        loss = self.loss('station-balanced')
        p = torch.tensor([[1., 0.], [10., 0.]]) # same physical 100 mm error
        elements = loss.elements(p, torch.zeros_like(p), torch.zeros_like(p), torch.tensor([0, 1]))
        self.assertEqual(elements[0, 0], elements[1, 0])

    def test_rise_fall_flat_wrong_sign_and_gradient(self):
        ids = torch.tensor([0, 0, 0])
        y = torch.tensor([[3., 0.], [-3., 0.], [0., 0.]])
        p = torch.tensor([[-1., 0.], [1., 0.], [0., 0.]], requires_grad=True)
        b = torch.zeros_like(y)
        asym, base = self.loss('asymmetric'), self.loss('event-weighted', event_weight=0.)
        a, c = asym.elements(p, y, b, ids), base.elements(p, y, b, ids)
        self.assertGreater(a[0, 0], c[0, 0])
        torch.testing.assert_close(a[1:], c[1:])
        asym(p, y, b, ids).backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertLess(p.grad[0, 0], 0.)
        # Overprediction is never rewarded; optimum remains the observed value.
        self.assertEqual(asym(y, y, b, ids).item(), 0.)

    def test_threshold_strict_and_zero_event_weight(self):
        loss = self.loss('event-weighted')
        y = torch.tensor([[2., 0.], [2.01, 0.]])
        b, p = torch.zeros_like(y), torch.zeros_like(y)
        e = loss.elements(p, y, b, torch.tensor([0, 0]))
        self.assertEqual(e[0, 0].item(), 4.)
        self.assertGreater(e[1, 0].item(), 8.)
        baseline = self.loss('event-weighted', event_weight=0.)
        torch.testing.assert_close(baseline(p, y, b, torch.tensor([0, 0])), (p-y).square().mean())

    def test_invalid_configuration(self):
        for kwargs in [dict(scale_floor_mm=0), dict(event_weight=-1), dict(under_weight=float('nan')),
                       dict(threshold_mm=-1), dict(counts=(0, 0))]:
            with self.assertRaises(ValueError):
                self.loss(**kwargs)

    def test_large_error_count_penalty_uses_physical_error_and_has_gradients(self):
        # Both predictions are 200 mm off, despite different target scales.
        loss = self.loss('large-error-count', counts=(3, 1),
                         scales=((100., 200.), (400., 200.)),
                         large_error_penalty=0.2, large_error_transition_mm=20.)
        baseline = self.loss('station-balanced', counts=(3, 1),
                             scales=((100., 200.), (400., 200.)))
        p = torch.tensor([[2., 0.], [-0.5, 0.]], requires_grad=True)
        y = torch.zeros_like(p)
        ids = torch.tensor([0, 1])
        extra = loss.elements(p, y, y, ids) - baseline.elements(p, y, y, ids)
        torch.testing.assert_close(extra[:, 0], torch.tensor([0.1, 0.1]))
        # The new term pulls both signs toward the observed value.
        extra.sum().backward()
        self.assertGreater(p.grad[0, 0].item(), 0.)
        self.assertLess(p.grad[1, 0].item(), 0.)
        self.assertEqual(loss.specification()['large_error_threshold_mm'], 200.)

    def test_large_error_count_zero_strength_matches_station_balanced(self):
        loss = self.loss('large-error-count', large_error_penalty=0.)
        baseline = self.loss('station-balanced')
        p, y = torch.randn(2, 2), torch.randn(2, 2)
        ids = torch.tensor([0, 1])
        torch.testing.assert_close(loss(p, y, y, ids), baseline(p, y, y, ids))
        for kwargs in [dict(large_error_penalty=-1),
                       dict(large_error_threshold_mm=0),
                       dict(large_error_transition_mm=0)]:
            with self.assertRaises(ValueError):
                self.loss('large-error-count', **kwargs)

    def test_large_error_primary_is_only_a_soft_count(self):
        loss = self.loss('large-error-primary', counts=(3, 1),
                         scales=((100., 200.), (400., 200.)),
                         large_error_penalty=1., large_error_transition_mm=20.)
        # Same physical errors at stations with different frequencies and scales.
        p = torch.tensor([[2., -1.], [-0.5, 1.]], requires_grad=True)
        y = torch.zeros_like(p)
        ids = torch.tensor([0, 1])
        torch.testing.assert_close(loss.elements(p, y, y, ids), torch.full((2, 2), 0.5))
        torch.testing.assert_close(loss(p, y, y, ids), torch.tensor(0.5))
        loss(p, y, y, ids).backward()
        self.assertGreater(p.grad[0, 0].item(), 0.)
        self.assertLess(p.grad[1, 0].item(), 0.)
        self.assertEqual(loss.specification()['base_squared_error_weight'], 0.)

    def test_diagnostics_empty_and_false_alarm(self):
        meter = ChangeDiagnostics(1, 2, 'cpu')
        p, y, b = torch.tensor([[3., -3.]]), torch.tensor([[0., -4.]]), torch.zeros(1, 2)
        meter.update(p, y, b, torch.tensor([[100., 100.]]), torch.tensor([0]))
        frame = meter.table({'s': {'station_index': 0, 'name': 'Station'}}, ['h6', 'h12'])
        self.assertEqual(frame.iloc[0].rise_false_alarm_ratio, 1.)
        self.assertTrue(frame.iloc[0].rise_recall != frame.iloc[0].rise_recall) # undefined
        self.assertEqual(frame.iloc[1].extreme_actual_abs_mm, 400.)

    def test_training_only_factory_and_constants(self):
        import pandas as pd
        view = SimpleNamespace(mapping={'a': {'station_index': 0}, 'b': {'station_index': 1}},
            samples=pd.DataFrame({'station_index': [0, 0, 0, 1]}),
            normalizers={'stations': {
                'a': {'targets': {'h6': {'scale': 10.}, 'h12': {'scale': 20.}}},
                'b': {'targets': {'h6': {'scale': 30.}, 'h12': {'scale': 40.}}}}})
        data = SimpleNamespace(split='train', view=view, num_targets=2)
        loss = from_training_dataset(data, weights=[1, 1], mode='station-balanced')
        torch.testing.assert_close(loss.frequency, torch.tensor([2/3, 2.], dtype=torch.float64))
        data.split = 'validation'
        with self.assertRaises(ValueError):
            from_training_dataset(data, weights=[1, 1], mode='station-balanced')

    def test_empty_invalid_id_and_serialization(self):
        loss = self.loss()
        with self.assertRaises(ValueError):
            loss(torch.empty(0, 2), torch.empty(0, 2), torch.empty(0, 2), torch.empty(0, dtype=torch.long))
        with self.assertRaises(RuntimeError):
            loss(torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, 2), torch.tensor([-1]))
        clone = self.loss()
        clone.load_state_dict(loss.state_dict())
        self.assertEqual(clone.specification(), loss.specification())

    def test_synthetic_trainer_smoke_all_modes(self):
        from torch.utils.data import Dataset, DataLoader
        class Fake(Dataset):
            num_targets, num_features, seq_len = 2, 2, 4
            target_cols = ['target_water_level_plus_360m', 'target_water_level_plus_720m']
            feature_set = 'water_level'
            regional_metadata = {'station_mapping': {'s': {'station_index': 0, 'name': 'Synthetic'}}}
            station_ids_validated = True
            def __len__(self): return 4
            def __getitem__(self, i):
                return dict(x=torch.ones(4, 2), y=torch.tensor([3., -4.]), station_ids=torch.tensor(0),
                            target_scale=torch.tensor([100., 100.]), target_mean=torch.zeros(2))
        loader = DataLoader(Fake(), batch_size=2)
        for mode in ['station-balanced', 'event-weighted', 'asymmetric', 'combined']:
            with tempfile.TemporaryDirectory() as tmp:
                model = ResidualSequence(architecture='lstm', input_size=2, hidden_size=4,
                                         num_layers=1, num_stations=1, dropout=0.,
                                         persistence_scale=[[1., 1.]], persistence_offset=[[0., 0.]])
                loss = self.loss(mode, counts=(4,), scales=((100., 100.),))
                trainer = ResidualLSTMTrainer(model=model, correction_penalty=0., horizon='1h', num_epochs=1,
                    train_dataloader=loader, eval_dataloader=loader, test_dataloader=loader, loss_func=loss,
                    optimizer=torch.optim.Adam(model.parameters(), lr=.001), scheduler=None,
                    early_stopping=2, checkpoint_dir=tmp)
                trainer.train_epoch(1) # synthetic only
                self.assertTrue(torch.isfinite(torch.tensor(trainer.evaluate_loss(loader))))
                trainer.save_checkpoint()
                self.assertTrue(trainer.checkpoint_path.with_suffix('.loss.json').exists())
                model.load_state_dict(torch.load(trainer.checkpoint_path, weights_only=True))

if __name__ == '__main__': unittest.main()
