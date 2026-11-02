import io
import json
import contextlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from src.ai.lstm import ResidualLSTM
from src.ai.residual_sequence import NEW_METHODS, CausalConv1d, ResidualSequence, residual_model
from src.ai.sequence_checkpoint import model_spec, model_from_spec, load_model_state, save_model_metadata, SequenceRuntime
from src.ai.comparison import load_candidate
from src.ai.losses import HorizonWeightedTailLoss


def options(weather=True, regional=False):
    return dict(input_size=8 if weather else 4, output_size=2, hidden_size=8, num_layers=1,
        dropout=0., input_dropout=0., head_dropout=0., level_feature_index=0,
        persistence_scale=[[1., 2.], [3., 4.]] if regional else [1., 2.],
        persistence_offset=[[.5, -.5], [1., -1.]] if regional else [.5, -.5],
        weather_feature_indices=[4, 5, 6, 7] if weather else [],
        weather_mask_indices=[6] if weather else [], weather_mask_values=[1.] if weather else [],
        num_stations=2 if regional else 0, station_embedding_size=3, forecast_hidden_size=5,
        tcn_channels=[8]*5, tcn_kernel_size=3, cnn_channels=[8, 8], cnn_kernel_size=3)


class ResidualArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_forward_backward_persistence_and_embeddings_all_methods(self):
        for method in NEW_METHODS:
            for weather in [False, True]:
                for regional in [False, True]:
                    with self.subTest(method=method, weather=weather, regional=regional):
                        torch.manual_seed(42)
                        model = residual_model(method, **options(weather, regional))
                        x = torch.randn(2, 96, model.input_size)
                        ids = torch.tensor([0, 1]) if regional else None
                        expected = x[:, -1, 0, None]*model.persistence_scale+model.persistence_offset
                        torch.testing.assert_close(model(x, ids), expected)
                        self.assertEqual(model(x, ids).shape, (2, 2))
                        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
                        for _ in range(2):
                            optimizer.zero_grad()
                            loss = (model(x, ids)-torch.ones(2, 2)).square().mean()
                            loss.backward()
                            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
                            optimizer.step()
                        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters()))
                        if regional:
                            self.assertGreater(model.station_embedding.weight.grad.abs().sum().item(), 0)
                        if not weather:
                            self.assertIsNone(model.forecast_encoder)
                        else:
                            self.assertEqual(model.forecast_encoder is not None, method.endswith('-late-fusion'))

    def test_save_load_exact_architecture_and_predictions(self):
        for method in NEW_METHODS:
            for weather in [False, True]:
                for regional in [False, True]:
                    with self.subTest(method=method, weather=weather, regional=regional):
                        model = residual_model(method, **options(weather, regional)).eval()
                        torch.nn.init.normal_(model.linear.weight)
                        x = torch.randn(2, 96, model.input_size)
                        ids = torch.tensor([0, 1]) if regional else None
                        saved = dict(model_spec=model_spec(model))
                        restored = load_model_state(model.state_dict(), json.loads(json.dumps(saved)))
                        self.assertEqual(type(model), type(restored))
                        torch.testing.assert_close(model(x, ids), restored(x, ids), rtol=0, atol=0)
                        if not regional:
                            buf = io.BytesIO(); torch.save(model.state_dict(), buf)
                            standalone = load_candidate(buf.getvalue(), saved, 'cpu')
                            torch.testing.assert_close(model(x), standalone(x), rtol=0, atol=0)

    def test_encoder_causality_including_even_kernel(self):
        for method in ['gru', 'tcn', 'cnn-gru', 'transformer']:
            for kernel in [2, 3]:
                config = options(False)
                config.update(tcn_kernel_size=kernel, cnn_kernel_size=kernel)
                model = residual_model(method, **config).eval()
                x = torch.randn(2, 96, 4)
                changed = x.clone(); changed[:, 40:] += 100
                first = model.encoder.forward_sequence(x)
                second = model.encoder.forward_sequence(changed)
                self.assertEqual(first.shape[:2], (2, 96))
                torch.testing.assert_close(first[:, :40], second[:, :40], rtol=0, atol=0)
                if method == 'transformer':
                    with torch.inference_mode():
                        first = model.encoder.forward_sequence(x)
                        second = model.encoder.forward_sequence(changed)
                    torch.testing.assert_close(first[:, :40], second[:, :40], rtol=0, atol=0)
        conv = CausalConv1d(2, 3, 3, dilation=4)
        self.assertEqual(conv(torch.randn(2, 2, 96)).shape, (2, 3, 96))

    def test_late_fusion_never_passes_weather_to_history_encoder(self):
        for method in [m for m in NEW_METHODS if m.endswith('-late-fusion')]:
            model = residual_model(method, **options()).eval()
            torch.nn.init.normal_(model.linear.weight)
            received = []
            handle = model.encoder.register_forward_pre_hook(lambda m, args: received.append(args[0].detach().clone()))
            x = torch.randn(2, 96, 8, requires_grad=True)
            before = x.detach().clone()
            output = model(x)
            output.sum().backward()
            self.assertEqual(received[0].shape, (2, 96, 4))
            torch.testing.assert_close(received[0], x.detach()[:, :, :4])
            self.assertTrue(x.grad[:, :-1, 4:].eq(0).all())
            changed = before.clone(); changed[:, :-1, 4:] += 10000
            torch.testing.assert_close(model(changed), output.detach(), rtol=0, atol=0)
            torch.testing.assert_close(x.detach(), before)
            handle.remove()

    def test_early_anchor_only_withholds_historical_weather(self):
        for method in ['gru', 'tcn', 'cnn-gru', 'transformer']:
            model = residual_model(method, **options()).eval()
            torch.nn.init.normal_(model.linear.weight)
            x = torch.randn(2, 96, 8, requires_grad=True)
            model(x).sum().backward()
            self.assertTrue(x.grad[:, :-1, 4:].eq(0).all())
            prepared = model.prepare_inputs(x)
            self.assertTrue(prepared[:, :-1, 6].eq(1).all())
            torch.testing.assert_close(prepared[:, -1], x[:, -1])

    def test_station_validation_and_two_dimensional_single_sequence(self):
        model = residual_model('gru', **options(False, True))
        x = torch.randn(2, 96, 4)
        for bad in [None, torch.tensor([0]), torch.tensor([0., 1.]), torch.tensor([0, 2])]:
            with self.assertRaises(ValueError):
                model(x, bad)
        single = residual_model('gru', **options(False))
        self.assertEqual(single(x[0]).shape, (1, 2))
        torch.testing.assert_close(single(x[0]), single(x[:1]))
        with self.assertRaises(ValueError):
            single(x, torch.tensor([0, 0]))

    def test_dynamic_physical_scales_for_regional_tail_loss(self):
        criterion = HorizonWeightedTailLoss([1, 1], [1, 1], strength=.5)
        predicted, actual = torch.tensor([[1., 2.], [1., 2.]], requires_grad=True), torch.zeros(2, 2)
        scales = torch.tensor([[10., 20.], [100., 200.]])
        together = criterion(predicted, actual, target_scales=scales)
        separately = sum(criterion(predicted[i:i+1], actual[i:i+1], target_scales=scales[i]) for i in range(2))/2
        torch.testing.assert_close(together, separately)
        together.backward(); self.assertTrue(torch.isfinite(predicted.grad).all())

    def test_legacy_residual_lstm_state_dict_unchanged_and_loads_without_metadata(self):
        fs = StandardScaler().fit(np.arange(24).reshape(6, 4))
        ts = StandardScaler().fit(np.arange(12).reshape(6, 2))
        model = ResidualLSTM(input_size=4, hidden_size=8, num_layers=1, output_size=2,
            level_feature_index=0, persistence_scale=(fs.scale_[0]/ts.scale_).tolist(),
            persistence_offset=((fs.mean_[0]-ts.mean_)/ts.scale_).tolist()).eval()
        torch.nn.init.normal_(model.linear.weight)
        saved = dict(feature_columns=['water_level_depth', 'a', 'b', 'c'], feature_scaler=fs, target_scaler=ts)
        restored = load_model_state(model.state_dict(), saved)
        x = torch.randn(2, 96, 4)
        torch.testing.assert_close(model(x), restored(x))
        self.assertFalse(any('encoder' in key or 'weather' in key for key in model.state_dict()))

    def test_runtime_and_architecture_sidecar(self):
        fs = StandardScaler().fit(np.arange(24).reshape(6, 4))
        ts = StandardScaler().fit(np.arange(12).reshape(6, 2))
        for method in NEW_METHODS:
            with tempfile.TemporaryDirectory() as directory:
                model = residual_model(method, **options(False)).eval()
                torch.nn.init.normal_(model.linear.weight)
                checkpoint = Path(directory)/'model.pt'
                torch.save(model.state_dict(), checkpoint)
                save_model_metadata(model, checkpoint)
                joblib.dump(dict(model_spec=model_spec(model), feature_scaler=fs, target_scaler=ts), checkpoint.with_suffix('.preprocessing.joblib'))
                runtime = SequenceRuntime(checkpoint)
                x = torch.randn(2, 96, 4)
                # Match runtime mode: Transformer uses a different fused eval kernel
                # when autograd is disabled, with small floating-point differences.
                with torch.inference_mode():
                    expected = ts.inverse_transform(model(x).numpy())
                np.testing.assert_allclose(runtime.predict_mm(x), expected, rtol=0, atol=0)
                self.assertEqual(json.loads(checkpoint.with_suffix('.architecture.json').read_text())['options']['architecture'], method)

    def test_regional_lstm_baseline_and_station_coefficient_lookup(self):
        model = residual_model('lstm', **options(True, True)).eval()
        x = torch.randn(2, 96, 8)
        ids = torch.tensor([1, 0])
        expected = x[:, -1, 0, None]*model.persistence_scale[ids]+model.persistence_offset[ids]
        torch.testing.assert_close(model(x, ids), expected)
        restored = load_model_state(model.state_dict(), {'model_spec': model_spec(model)})
        torch.testing.assert_close(restored(x, ids), expected)

    def test_station_mlp_shape_persistence_isolation_and_roundtrip(self):
        torch.manual_seed(42)
        model = residual_model('gru-late-fusion', **options(True, True), head_mode='station-mlp').eval()
        x = torch.randn(3, 10, model.input_size)
        ids = torch.tensor([1, 0, 1])
        expected = x[:, -1, 0, None]*model.persistence_scale[ids]+model.persistence_offset[ids]
        torch.testing.assert_close(model(x, ids), expected)
        self.assertEqual([tuple(w.shape) for w in model.linear.weights],
                         [(2, 32, 8), (2, 32, 32), (2, 32, 32), (2, 2, 32)])
        with torch.no_grad():
            model.linear.weights[-1].normal_()
            model.linear.biases[-1][0].fill_(1.)
        before = model(x, ids).detach()
        model(x[:1], ids[:1]).sum().backward()
        for parameter in model.linear.parameters():
            self.assertTrue(parameter.grad[0].eq(0).all())
        self.assertGreater(model.linear.weights[-1].grad[1].abs().sum().item(), 0)
        restored = load_model_state(model.state_dict(), json.loads(json.dumps({'model_spec': model_spec(model)})))
        torch.testing.assert_close(restored(x, ids), before, rtol=0, atol=0)
        self.assertEqual(model_spec(model)['options']['head_mode'], 'station-mlp')
        with self.assertRaises(ValueError):
            residual_model('gru', **options(False), head_mode='station-mlp')
        with self.assertRaises(ValueError):
            residual_model('gru-late-fusion', **options(True, True), head_mode='station-mlp',
                           horizon_decoder='transformer', transformer_heads=2)

    def test_station_mlp_parameter_count(self):
        config = dict(input_size=42, output_size=2, hidden_size=64, num_layers=3, num_stations=11,
                      persistence_scale=[[1., 1.]]*11, persistence_offset=[[0., 0.]]*11,
                      weather_feature_indices=list(range(30, 42)), weather_mask_indices=[41],
                      weather_mask_values=[1.], station_embedding_size=8, forecast_hidden_size=32)
        linear = residual_model('gru-late-fusion', **config, head_mode='station')
        mlp = residual_model('gru-late-fusion', **config, head_mode='station-mlp')
        self.assertEqual(sum(p.numel() for p in linear.linear.parameters()), 1430)
        self.assertEqual(sum(p.numel() for p in mlp.linear.parameters()), 46838)
        self.assertEqual(sum(p.numel() for p in mlp.parameters()) - sum(p.numel() for p in linear.parameters()), 45408)

    def test_station_mlp_width_and_legacy_checkpoint_metadata(self):
        kwargs = options(True, True)
        for width in (16, 32, 64):
            model = residual_model('transformer-late-fusion', **kwargs, head_mode='station-mlp',
                                   transformer_heads=2, station_mlp_hidden_size=width).eval()
            self.assertEqual([tuple(w.shape) for w in model.linear.weights],
                [(2, width, 8), (2, width, width), (2, width, width), (2, 2, width)])
            saved = json.loads(json.dumps({'model_spec': model_spec(model)}))
            self.assertEqual(saved['model_spec']['options']['station_mlp_hidden_size'], width)
            if width == 32:
                del saved['model_spec']['options']['station_mlp_hidden_size']
            with torch.no_grad():
                model.linear.weights[-1].normal_()
            restored = load_model_state(model.state_dict(), saved)
            x, ids = torch.randn(2, 16, model.input_size), torch.tensor([0, 1])
            torch.testing.assert_close(restored(x, ids), model(x, ids), rtol=0, atol=0)
        for width in (0, -1):
            with self.assertRaises(ValueError):
                residual_model('gru', **kwargs, head_mode='station-mlp', station_mlp_hidden_size=width)
        with self.assertRaises(ValueError):
            residual_model('gru', **kwargs, head_mode='station', station_mlp_hidden_size=64)

    def test_station_mlp_width_cli_validation(self):
        from src.ai.train import parser, validate_args
        base = ['--train-method', 'gru-late-fusion', '--horizon', '1h', '--mlflow-ex-name', 'test']
        with patch('src.ai.train.is_regional_dataset', return_value=True):
            args = parser.parse_args(base + ['--head-mode', 'station-mlp', '--station-mlp-hidden-size', '64'])
            validate_args(args)
            self.assertEqual(args.station_mlp_hidden_size, 64)
            with contextlib.redirect_stderr(io.StringIO()):
                for extra in (['--head-mode', 'station-mlp', '--station-mlp-hidden-size', '0'],
                              ['--head-mode', 'station', '--station-mlp-hidden-size', '64']):
                    with self.assertRaises(SystemExit):
                        validate_args(parser.parse_args(base + extra))

    def test_cli_residual_alias_and_legacy_vanilla_option(self):
        from src.ai.train import parser, validate_args, build_sequence_trainer
        from src.ai.lstm import LSTM
        root = Path(__file__).resolve().parents[3]
        dataset = root/'train_data/da_vien_1h_2026-10-15_forecast_mask_o2_h6_h12'
        if not dataset.exists():
            self.skipTest('Local dataset snapshot is not installed')
        for method, expected in [('lstm', ResidualSequence), ('residual_lstm', ResidualLSTM), ('vanilla_lstm', LSTM)]:
            args = parser.parse_args(['--train-method', method, '--dataset-dir', str(dataset),
                '--horizon', '1h', '--mlflow-ex-name', 'test', '--device', 'cpu',
                '--hidden-size', '3', '--num-layers', '1', '--dropout', '0', '--lstm-seq-len', '96',
                '--target-horizons', '6', '12'])
            validate_args(args)
            trainer = build_sequence_trainer(args)
            self.assertIs(type(trainer.model), expected)


if __name__ == '__main__':
    unittest.main()
