"""Physical-unit tail gradients, persistence scoring and validation-only selection."""
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
import numpy as np
import torch
from torch import nn

from src.ai.losses import HorizonWeightedMSE, HorizonWeightedTailLoss, validate_correction_penalty
from src.ai.trainers import LSTMTrainer, ResidualLSTMTrainer


class TailLossTests(TestCase):
    def test_threshold_is_physical_and_zero_strength_is_exact_mse(self):
        p=torch.tensor([[2.,.5]],dtype=torch.float64,requires_grad=True)
        y=torch.zeros_like(p)
        loss=HorizonWeightedTailLoss([1,2],[100,400],strength=.1)
        mse,penalty=loss.components(p,y)
        self.assertAlmostEqual(penalty.item(),math.log(2))  # Both errors exactly 200 mm.
        self.assertAlmostEqual(loss(p,y).item(),mse.item()+.1*math.log(2))
        zero=HorizonWeightedTailLoss([1,2],[100,400],strength=0)(p,y)
        base=HorizonWeightedMSE([1,2])(p,y)
        torch.testing.assert_close(zero,base,rtol=0,atol=0)
        torch.testing.assert_close(torch.autograd.grad(zero,p)[0],torch.autograd.grad(base,p)[0],rtol=0,atol=0)

    def test_extra_gradient_near_threshold_and_stability_far_above(self):
        p=torch.tensor([[1.99,2.,2.01]],dtype=torch.float64,requires_grad=True)
        y=torch.zeros_like(p)
        tail=HorizonWeightedTailLoss([1,1,1],[100,100,100],strength=.1)
        base=HorizonWeightedMSE([1,1,1])(p,y)
        extra=torch.autograd.grad(tail(p,y)-base,p)[0]
        self.assertTrue((extra>0).all())
        self.assertTrue((extra[:,1:]>extra[:,:-1]).all())
        p2=torch.tensor([[1e5,1e5,1e5]],requires_grad=True)
        loss=tail(p2,torch.zeros_like(p2));self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(torch.autograd.grad(loss,p2)[0]).all())

    def test_parameter_validation_and_negative_correction_guard(self):
        for options in [dict(target_scales=[0,1]),dict(target_scales=[1]),
                        dict(strength=-.1),dict(transition_mm=0),dict(threshold_mm=float('nan'))]:
            params=dict(weights=[1,2],target_scales=[100,200]);params.update(options)
            with self.subTest(options=options),self.assertRaises(ValueError):HorizonWeightedTailLoss(**params)
        loss=HorizonWeightedTailLoss([1,3],[100,200])
        validate_correction_penalty(-.1,loss,2)
        with self.assertRaises(ValueError):validate_correction_penalty(-.5,loss,2)


class TailCheckpointTests(TestCase):
    def loader(self):
        class Loader:
            dataset=SimpleNamespace(target_cols=['near','far'],
                target_scaler=SimpleNamespace(scale_=np.array([100.,400.])),feature_set='all')
            def __iter__(self):
                yield torch.tensor([[2.,.5]],dtype=torch.float64),torch.zeros(1,2,dtype=torch.float64)
                yield torch.tensor([[2.01,.5025]],dtype=torch.float64),torch.zeros(1,2,dtype=torch.float64)
        return Loader()

    def test_validation_uses_mm_strict_threshold_and_same_objective(self):
        trainer=object.__new__(LSTMTrainer)
        trainer.model=nn.Identity();trainer.device=torch.device('cpu')
        trainer.loss_func=HorizonWeightedTailLoss([1,3],[100,400])
        trainer.checkpoint_metric='large_error_rate';trainer.validation_threshold_mm=200.
        loss=trainer.evaluate_loss(self.loader())
        predictions=torch.tensor([[2.,.5],[2.01,.5025]],dtype=torch.float64)
        expected=trainer.loss_func(predictions,torch.zeros_like(predictions)).item()
        self.assertAlmostEqual(loss,expected)
        self.assertEqual(trainer.last_validation_metrics['val/large_error_count'],2)
        self.assertEqual(trainer.last_validation_metrics['val/large_error_pct'],50.)
        self.assertAlmostEqual(trainer.last_validation_metrics['val/rmse_mm'],math.sqrt((200**2+201**2)/2))

    def test_checkpoint_and_patience_follow_count_then_rmse_not_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            model=nn.Linear(1,1)
            loader=SimpleNamespace(dataset=SimpleNamespace(feature_set='all'))
            trainer=LSTMTrainer(model,'1h',5,loader,loader,loader,nn.MSELoss(),
                torch.optim.SGD(model.parameters(),lr=.01),None,3,checkpoint_dir=directory,
                checkpoint_metric='large_error_rate')
            trainer.checkpoint_path.parent.mkdir(parents=True)
            for epoch,loss,count,rmse,expected_epoch in [
                (1,.5,5,70,1),(2,.1,6,60,1),(3,.8,4,90,3),(4,.9,4,80,4),(5,.01,4,80,4)]:
                trainer.last_validation_metrics={'val/large_error_count':count,'val/rmse_mm':rmse}
                trainer.record_validation(epoch,loss)
                self.assertEqual(trainer.best_checkpoint_epoch,expected_epoch)
            self.assertEqual(trainer.epochs_without_improvement,1)
            self.assertEqual(trainer.best_val_loss,.9)
            self.assertEqual(trainer.best_checkpoint_score,(4,80))

    def test_persistence_includes_tail_loss_and_sets_same_selection_score(self):
        class Baseline(nn.Module):
            output_size=2
            def persistence(self,x):return x
        trainer=object.__new__(ResidualLSTMTrainer)
        trainer.model=Baseline();trainer.device=torch.device('cpu')
        trainer.eval_dataloader=self.loader()
        trainer.loss_func=HorizonWeightedTailLoss([1,3],[100,400])
        trainer.checkpoint_metric='large_error_rate';trainer.validation_threshold_mm=200.
        with tempfile.TemporaryDirectory() as directory:
            trainer.checkpoint_path=Path(directory)/'baseline.pt'
            trainer.prepare_checkpoint()
            self.assertEqual(trainer.best_checkpoint_epoch,0)
            self.assertEqual(trainer.best_checkpoint_score[0],2)
            self.assertEqual(trainer.best_validation_metrics['val/large_error_pct'],50.)
            expected=trainer.evaluate_loss(trainer.eval_dataloader,predictor=trainer.model.persistence)
            self.assertAlmostEqual(trainer.best_val_loss,expected)

    def test_cli_wires_scaler_and_separates_experiment_checkpoints(self):
        from src.ai.train import parser,validate_args,build_sequence_trainer
        root=Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as directory:
            argv=['train','--mlflow-ex-name','tail-test','--horizon','1h','--train-method','residual_lstm',
                '--dataset-dir',str(root/'train_data/da_vien_1h_2026-10-15_forecast_mask'),
                '--feature-set','water_level+rain+flood_season+weather_forecast',
                '--num-rain-station','2','--device','cpu','--num-worker','0',
                '--lstm-seq-len','96','--lstm-hidden-size','2','--lstm-num-layers','1',
                '--lstm-drop-out','0','--lstm-horizon-weights','1','1','1.5','1.5','2',
                '--lstm-correction-penalty','.1','--lstm-tail-penalty','.1',
                '--lstm-checkpoint-metric','large_error_rate','--checkpoint-dir',directory]
            args=parser.parse_args(argv[1:])
            validate_args(args)
            trainer=build_sequence_trainer(args)
            self.assertIsInstance(trainer.loss_func,HorizonWeightedTailLoss)
            np.testing.assert_allclose(trainer.loss_func.target_scales.numpy(),trainer.train_dataloader.dataset.target_scaler.scale_)
            self.assertIn('tail_0.1',str(trainer.checkpoint_path))
            x,y=next(iter(trainer.train_dataloader))
            trainer.optimizer.zero_grad()
            p=trainer.model(x)
            loss=trainer.training_loss(p,x,trainer.loss_func(p,y))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
