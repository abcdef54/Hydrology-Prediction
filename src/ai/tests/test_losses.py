"""Check weighted loss math, gradients, and persistence checkpoint scoring."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main

import torch
from torch import nn
from src.ai.losses import HorizonWeightedMSE
from src.ai.trainers import ResidualLSTMTrainer


class WeightedLossTests(TestCase):
    def test_equal_weights_reproduce_mse_and_gradients(self):
        p = torch.tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True)
        y = torch.zeros_like(p)
        actual = HorizonWeightedMSE([1, 1, 1])(p, y)
        expected = nn.MSELoss()(p, y)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual, p)[0], torch.autograd.grad(expected, p)[0])

    def test_weighted_values_and_gradients_and_scale_invariance(self):
        p = torch.tensor([[1., 2.]], requires_grad=True)
        y = torch.zeros_like(p)
        loss = HorizonWeightedMSE([1, 3])(p, y)
        self.assertAlmostEqual(loss.item(), 3.25)
        torch.testing.assert_close(torch.autograd.grad(loss, p)[0], torch.tensor([[.5, 3.]]))
        torch.testing.assert_close(loss, HorizonWeightedMSE([10, 30])(p, y))

    def test_reject_invalid_weights_and_broadcasting(self):
        for weights in [[], [0, 1], [-1, 2], [float('nan'), 1], [float('inf'), 1]]:
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                HorizonWeightedMSE(weights)
        with self.assertRaises(ValueError):
            HorizonWeightedMSE([1, 2])(torch.zeros(2, 3), torch.zeros(2, 3))
        with self.assertRaises(ValueError):
            HorizonWeightedMSE([1, 2])(torch.zeros(2, 2), torch.zeros(1, 2))

    def test_persistence_checkpoint_uses_same_weighted_validation_objective(self):
        class Baseline(nn.Module):
            output_size = 2
            def persistence(self, x): return torch.zeros_like(x)
        class Loader:
            dataset = SimpleNamespace(target_cols=['near', 'far'])
            def __iter__(self):
                yield torch.zeros(1, 2), torch.tensor([[1., 2.]])
        trainer = object.__new__(ResidualLSTMTrainer)
        trainer.model = Baseline()
        trainer.eval_dataloader = Loader()
        trainer.device = torch.device('cpu')
        trainer.loss_func = HorizonWeightedMSE([1, 3])
        with tempfile.TemporaryDirectory() as directory:
            trainer.checkpoint_path = Path(directory) / 'baseline.pt'
            trainer.prepare_checkpoint()
            self.assertAlmostEqual(trainer.best_val_loss, 3.25)
            self.assertEqual(trainer.best_checkpoint_epoch, 0)
            self.assertTrue(trainer.checkpoint_path.exists())


class NegativeCorrectionPenaltyTests(TestCase):
    def test_reject_unbounded_weighted_and_unweighted_objectives(self):
        from src.ai.losses import validate_correction_penalty
        for coefficient in [-1., -2., float('nan'), float('inf'), -float('inf')]:
            with self.subTest(coefficient=coefficient), self.assertRaises(ValueError):
                validate_correction_penalty(coefficient, nn.MSELoss(), 2)
        weighted = HorizonWeightedMSE([1, 3])  # strict lower bound is -0.5
        for coefficient in [-.5, -.6]:
            with self.subTest(coefficient=coefficient), self.assertRaises(ValueError):
                validate_correction_penalty(coefficient, weighted, 2)
        validate_correction_penalty(-.1, nn.MSELoss(), 2)
        validate_correction_penalty(-.1, weighted, 2)
        validate_correction_penalty(0., weighted, 2)
        validate_correction_penalty(.1, weighted, 2)
        with self.assertRaises(ValueError):
            validate_correction_penalty(-.1, nn.L1Loss(), 2)
        with self.assertRaises(ValueError):
            validate_correction_penalty(-.1, nn.MSELoss(reduction='sum'), 2)

    def test_negative_penalty_favors_larger_correction_without_changing_validation(self):
        from src.ai.models import ResidualLSTM
        from src.ai.losses import validate_correction_penalty
        model = ResidualLSTM(input_size=1, hidden_size=2, num_layers=1,
                             output_size=2, level_feature_index=0,
                             persistence_scale=[1., 1.], persistence_offset=[0., 0.])
        loader = SimpleNamespace(dataset=SimpleNamespace(feature_set='all'))
        for loss in [nn.MSELoss(), HorizonWeightedMSE([1, 3])]:
            validate_correction_penalty(-.1, loss, 2)
            trainer = ResidualLSTMTrainer(
                model=model, correction_penalty=-.1, loss_func=loss,
                horizon='1h', num_epochs=1, train_dataloader=loader,
                eval_dataloader=loader, test_dataloader=loader,
                optimizer=torch.optim.SGD(model.parameters(), lr=.01), scheduler=None,
                early_stopping=1,
            )
            inputs = torch.zeros(1, 1, 1)
            targets = torch.ones(1, 2)
            # Under mean MSE optimal correction is 1/(1-0.1) > 1.
            prediction = torch.ones(1, 2, requires_grad=True)
            data_loss = loss(prediction, targets)
            objective = trainer.training_loss(prediction, inputs, data_loss)
            self.assertAlmostEqual(data_loss.item(), 0.)
            self.assertAlmostEqual(objective.item(), -.1, places=6)
            self.assertTrue((torch.autograd.grad(objective, prediction)[0] < 0).all())
            self.assertEqual(trainer.model_settings()['Correction penalty'], -.1)
        with self.assertRaises(ValueError):
            ResidualLSTMTrainer(model=model, correction_penalty=-1., loss_func=nn.MSELoss())


if __name__ == '__main__': main()
