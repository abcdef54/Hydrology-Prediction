"""Transformer integration tests use only generated observations and tensors."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import torch

from src.ai.regional_data.builder import build_dataset
from src.ai.residual_sequence import residual_model
from src.ai.sequence_checkpoint import SequenceRuntime, model_spec, model_from_spec
from src.ai.tests.test_regional_data import fixture
from src.ai.tests.test_residual_architectures import options


class TransformerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_direct_head_predicts_level_without_persistence_and_roundtrips(self):
        model = residual_model('transformer-late-fusion', **options(True, True),
                               head_mode='station-mlp', prediction_mode='direct',
                               transformer_heads=2).eval()
        x = torch.randn(3, 12, 8)
        ids = torch.tensor([0, 1, 0])
        # Zero initialized head is zero in target-normalized level space,
        # not the current water-level persistence estimate.
        torch.testing.assert_close(model(x, ids), torch.zeros(3, 2), rtol=0, atol=0)
        self.assertFalse(torch.equal(model(x, ids), model.persistence(x, ids)))
        with torch.no_grad():
            model.linear.biases[-1].fill_(0.7)
        torch.testing.assert_close(model(x, ids), torch.full((3, 2), 0.7))
        clone = model_from_spec(model_spec(model)).eval()
        clone.load_state_dict(model.state_dict())
        self.assertEqual(clone.prediction_mode, 'direct')
        torch.testing.assert_close(clone(x, ids), model(x, ids))

    def test_direct_initial_checkpoint_scores_actual_model(self):
        from src.ai.train import parser, validate_args, build_sequence_trainer
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            level, rain, strict, weather, mapping, config, catalog = fixture()
            build_dataset(root/'pool', level=level, rain_standard=rain, rain_strict=strict,
                weather=weather, stations=mapping,
                clusters=pd.DataFrame({'station_id': list(mapping)}), config=config,
                source_paths=[], rain_catalog=catalog)
            args = parser.parse_args(['--dataset-dir', str(root/'pool'),
                '--train-method', 'transformer-late-fusion', '--horizon', '1h',
                '--regional-fold', 'summer', '--rain-station', 'nham+hong_thai',
                '--head-mode', 'station-mlp', '--prediction-mode', 'direct',
                '--lstm-hidden-size', '8', '--transformer-heads', '2',
                '--lstm-num-layers', '2', '--device', 'cpu',
                '--mlflow-ex-name', 'synthetic', '--checkpoint-dir', str(root/'models')])
            validate_args(args)
            trainer = build_sequence_trainer(args)
            self.assertIn('prediction_direct', trainer.checkpoint_path.parts)
            expected = trainer.evaluate_loss(trainer.eval_dataloader)
            with contextlib.redirect_stdout(io.StringIO()):
                trainer.prepare_checkpoint()
            self.assertAlmostEqual(trainer.best_val_loss, expected, places=6)
            self.assertEqual(trainer.best_checkpoint_epoch, 0)
            self.assertTrue(trainer.checkpoint_path.exists())
            message = io.StringIO()
            with contextlib.redirect_stdout(message):
                trainer.restore_best_checkpoint()
            self.assertIn('initial direct model', message.getvalue())

    def test_gru_horizon_decoder_station_heads_and_checkpoint_roundtrip(self):
        torch.manual_seed(12)
        model = residual_model('gru-late-fusion', **options(True, True), head_mode='station',
                               horizon_decoder='transformer', transformer_heads=2,
                               transformer_ff_size=19).eval()
        x = torch.randn(3, 12, 8)
        ids = torch.tensor([0, 1, 0])
        torch.testing.assert_close(model(x, ids), model.persistence(x, ids), rtol=0, atol=0)
        with torch.no_grad():
            model.linear.weight.normal_(0, .1)
        prediction = model(x, ids)
        self.assertEqual(prediction.shape, (3, 2))
        prediction.square().sum().backward()
        self.assertGreater(model.horizon_queries.grad.abs().sum().item(), 0)
        self.assertGreater(model.horizon_decoder.multihead_attn.in_proj_weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.encoder.recurrent.weight_ih_l0.grad.abs().sum().item(), 0)
        clone = model_from_spec(model_spec(model))
        clone.load_state_dict(model.state_dict())
        clone.eval()
        torch.testing.assert_close(clone(x, ids), prediction, rtol=0, atol=0)
        changed = x.clone(); changed[:, 0, 0] += 2
        self.assertFalse(torch.equal(model(changed, ids), prediction))

    def test_horizon_decoder_rejects_invalid_configuration(self):
        for method, head, heads in [('gru-late-fusion','shared',2), ('gru','station',2),
                                    ('gru-late-fusion','station',3)]:
            with self.subTest(method=method,head=head,heads=heads), self.assertRaises(ValueError):
                residual_model(method, **options(True, True), head_mode=head,
                               horizon_decoder='transformer', transformer_heads=heads)

    def test_decoder_cli_constructs_and_isolates_checkpoint_path(self):
        from src.ai.train import parser, validate_args, build_sequence_trainer
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            level, rain, strict, weather, mapping, config, catalog = fixture()
            build_dataset(root/'pool', level=level, rain_standard=rain, rain_strict=strict,
                weather=weather, stations=mapping,
                clusters=pd.DataFrame({'station_id': list(mapping)}), config=config,
                source_paths=[], rain_catalog=catalog)
            args = parser.parse_args(['--dataset-dir', str(root/'pool'),
                '--train-method', 'gru-late-fusion', '--horizon', '1h',
                '--target-horizons', '6', '12', '--regional-fold', 'summer',
                '--rain-station', 'nham+hong_thai', '--head-mode', 'station',
                '--horizon-decoder', 'transformer', '--transformer-heads', '2',
                '--lstm-hidden-size', '8', '--device', 'cpu',
                '--mlflow-ex-name', 'synthetic', '--checkpoint-dir', str(root/'models')])
            validate_args(args)
            trainer = build_sequence_trainer(args)
            self.assertIn('decoder_transformer', trainer.checkpoint_path.parts)
            batch = next(iter(trainer.train_dataloader))
            inputs, targets, _, _ = trainer.prepare_batch(batch, trainer.train_dataloader.dataset)
            self.assertEqual(trainer.predict_batch(trainer.model, inputs).shape, targets.shape)
            trainer.save_checkpoint()
            restored = model_from_spec(model_spec(trainer.model))
            restored.load_state_dict(torch.load(trainer.checkpoint_path, map_location='cpu', weights_only=True))
            self.assertEqual(restored.horizon_decoder_mode, 'transformer')

    def test_positions_dynamic_lengths_odd_width_and_independent_layers(self):
        config = options(False)
        config.update(hidden_size=9, num_layers=2, transformer_heads=3, transformer_ff_size=17)
        model = residual_model('transformer', **config).eval()
        layers = model.encoder.temporal.layers.layers
        self.assertFalse(torch.equal(layers[0].self_attn.in_proj_weight,
                                     layers[1].self_attn.in_proj_weight))
        for length in [1, 48, 96, 145]:
            x = torch.zeros(2, length, 4)
            encoded = model.encoder.forward_sequence(x)
            self.assertEqual(encoded.shape, (2, length, 9))
            self.assertTrue(torch.isfinite(encoded).all())
            if length > 1:
                self.assertFalse(torch.allclose(encoded[:, 0], encoded[:, -1]))
        clone = model_from_spec(model_spec(model))
        clone.load_state_dict(model.state_dict())
        self.assertEqual(clone.architecture_config['transformer_heads'], 3)
        self.assertEqual(clone.architecture_config['transformer_ff_size'], 17)

    def test_invalid_heads_and_cli_validation(self):
        for heads, width in [(0, 16), (3, 16), (4, 0)]:
            with self.subTest(heads=heads, width=width), self.assertRaises(ValueError):
                residual_model('transformer', **options(False),
                               transformer_heads=heads, transformer_ff_size=width)
        from src.ai.train import parser, validate_args
        argv = ['--train-method', 'transformer-late-fusion', '--horizon', '1h',
                '--mlflow-ex-name', 'synthetic', '--device', 'cpu']
        args = parser.parse_args(argv+['--lstm-hidden-size', '64', '--transformer-heads', '4'])
        validate_args(args)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            validate_args(parser.parse_args(argv+['--transformer-heads', '3']))

    def test_synthetic_cli_training_b1_and_checkpoint_physical_inference(self):
        from src.ai.train import parser, validate_args, build_sequence_trainer
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            level, rain, strict, weather, mapping, config, catalog = fixture()
            build_dataset(root/'pool', level=level, rain_standard=rain, rain_strict=strict,
                weather=weather, stations=mapping,
                clusters=pd.DataFrame({'station_id': list(mapping)}), config=config,
                source_paths=[], rain_catalog=catalog)
            for method in ['transformer', 'transformer-late-fusion']:
                with self.subTest(method=method):
                    torch.manual_seed(42)
                    args = parser.parse_args([
                        '--dataset-dir', str(root/'pool'), '--train-method', method,
                        '--horizon', '1h', '--target-horizons', '3', '6', '12', '18',
                        '--lstm-horizon-weights', '1', '1', '1', '1', '--lstm-seq-len', '48',
                        '--rain-station', 'nham+hong_thai', '--regional-fold', 'summer',
                        '--feature-set', 'water_level+rain+flood_season+weather_forecast',
                        '--use-weather-forecast', 'true', '--batch-size', '8', '--num-worker', '0',
                        '--lstm-hidden-size', '8', '--lstm-num-layers', '2',
                        '--transformer-heads', '2', '--transformer-ff-size', '19',
                        '--lstm-drop-out', '0.1', '--lstm-input-dropout', '0.1',
                        '--lstm-head-dropout', '0.1', '--change-loss', 'station-balanced',
                        '--num-epochs', '1', '--device', 'cpu', '--mlflow-ex-name', 'synthetic',
                        '--checkpoint-dir', str(root/method)])
                    validate_args(args)
                    trainer = build_sequence_trainer(args)
                    before = trainer.model.encoder.temporal.projection.weight.detach().clone()
                    trained = []
                    restore = trainer.restore_best_checkpoint
                    def capture_then_restore():
                        trained.append(trainer.model.encoder.temporal.projection.weight.detach().clone())
                        restore()
                    with contextlib.redirect_stdout(io.StringIO()), patch('src.ai.trainers.mlflow', Mock()), \
                            patch.object(trainer, 'restore_best_checkpoint', side_effect=capture_then_restore):
                        trainer.train()
                    self.assertTrue(trainer.checkpoint_path.is_file())
                    self.assertIn('val/mae_mm', trainer.last_validation_metrics)
                    # The head starts at zero; subsequent batches must reach attention/projection.
                    # Validation may legitimately select the initial persistence checkpoint.
                    self.assertFalse(torch.equal(before, trained[0]))
                    batch = next(iter(trainer.test_dataloader))
                    inputs, _, scales, means = trainer.prepare_batch(batch, trainer.test_dataloader.dataset)
                    trainer.model.eval()
                    with torch.inference_mode():
                        expected = (trainer.predict_batch(trainer.model, inputs).double()*scales.double()
                                    +means.double()).numpy()
                    runtime = SequenceRuntime(trainer.checkpoint_path)
                    actual = runtime.predict_mm(batch['x'], batch['station_ids'])
                    self.assertEqual(actual.shape, (len(batch['y']), 4))
                    np.testing.assert_allclose(actual, expected, atol=.005, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
