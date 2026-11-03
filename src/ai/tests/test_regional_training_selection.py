"""Synthetic checks for configurable history and exact-timestamp output targets."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from src.ai.tests.test_regional_data import fixture
from src.ai.regional_data.builder import build_dataset
from src.ai.regional_data.training_selection import select_samples
from src.ai.regional_training_data import RegionalTrainingDataset, regional_loaders, regional_persistence
from src.ai.change_objective import from_training_dataset
from src.ai.residual_sequence import residual_model


class RegionalSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temp.name)/'pool'
        level, rain, strict, weather, mapping, config, catalog = fixture()
        cls.config, cls.level = config, level
        build_dataset(cls.path, level=level, rain_standard=rain, rain_strict=strict, weather=weather,
            stations=mapping, clusters=pd.DataFrame({'station_id': list(mapping)}), config=config,
            source_paths=[], rain_catalog=catalog)

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def test_short_history_is_last_48_hours(self):
        old = RegionalTrainingDataset(self.path, split='train')
        new = RegionalTrainingDataset(self.path, split='train', sequence_length=48)
        self.assertEqual(len(old), len(new))
        torch.testing.assert_close(new[0]['x'], old[0]['x'][-48:], rtol=0, atol=0)
        self.assertEqual(new[0]['x'].shape, (48, new.num_features))
        torch.testing.assert_close(new[0]['y'], old[0]['y'], rtol=0, atol=0)

    def test_new_targets_alignment_scaling_and_all_encoders(self):
        train, val, test = regional_loaders(self.path, batch_size=4, num_workers=0, seed=42,
            sequence_length=48, target_horizons=[18, 3, 12, 6])
        data = train.dataset
        self.assertEqual(data.target_horizons, [3, 6, 12, 18])
        self.assertEqual(data.num_targets, 4)
        self.assertEqual(val.dataset.view.normalizers, data.view.normalizers)
        for d in [data, val.dataset, test.dataset]:
            for i in [0, len(d)-1]:
                sample, row = d[i], d.view.samples.iloc[i]
                raw = sample['y']*sample['target_scale']+sample['target_mean']
                expected = [d.view.water[row.station_id].loc[row.anchor_utc+pd.Timedelta(hours=h)] for h in d.target_horizons]
                np.testing.assert_allclose(raw.numpy(), expected, rtol=1e-6, atol=1e-3)
                scale, offset, count = regional_persistence(d)
                for method in ['lstm','gru','tcn','cnn-gru','gru-late-fusion','tcn-late-fusion','cnn-gru-late-fusion']:
                    weather = [j for j,c in enumerate(d.feature_cols) if c.startswith('weather_forecast_')]
                    masks = [j for j in weather if d.feature_cols[j].endswith('_missing')]
                    model = residual_model(method,input_size=d.num_features,output_size=4,hidden_size=4,
                        num_layers=1,dropout=0.,num_stations=count,persistence_scale=scale,persistence_offset=offset,
                        weather_feature_indices=weather,weather_mask_indices=masks,weather_mask_values=[1.]*len(masks),
                        tcn_channels=[4,4],cnn_channels=[4])
                    output = model(sample['x'][None], sample['station_ids'][None])
                    self.assertEqual(output.shape,(1,4))
                    physical = output[0]*sample['target_scale']+sample['target_mean']
                    np.testing.assert_allclose(physical.detach().numpy(), [row.water_now_mm]*4, rtol=1e-5,atol=1e-3)
        batch = next(iter(train)); p=batch['y'].clone().requires_grad_()
        loss = from_training_dataset(data, weights=[1]*4, mode='station-balanced')
        loss(p,batch['y'],torch.zeros_like(p),batch['station_ids']).backward()
        self.assertTrue(torch.isfinite(p.grad).all())

    def test_hourly_24_targets_alignment_split_scalers_and_models(self):
        horizons = list(range(1, 25))
        train, val, test = regional_loaders(self.path, batch_size=4, num_workers=0, seed=17,
            sequence_length=96, target_horizons=horizons)
        data = train.dataset
        self.assertEqual(data.target_horizons, horizons)
        self.assertEqual(val.dataset.view.normalizers, data.view.normalizers)
        self.assertEqual(test.dataset.view.normalizers, data.view.normalizers)
        self.assertEqual(data.num_targets, 24)
        config = self.config
        for loader in (train, val, test):
            d = loader.dataset
            for _, row in d.view.samples.iterrows():
                for h in horizons:
                    self.assertEqual(row[f'target_h{h}_utc'], row.anchor_utc + pd.Timedelta(hours=h))
                    self.assertAlmostEqual(row[f'target_h{h}_mm'], d.view.water[row.station_id].loc[row[f'target_h{h}_utc']])
            for i in (0, len(d)-1):
                sample, row = d[i], d.view.samples.iloc[i]
                raw = sample['y'] * sample['target_scale'] + sample['target_mean']
                np.testing.assert_allclose(raw.numpy(), [row[f'target_h{h}_mm'] for h in horizons], atol=1e-3)
            if d.split in ('train', 'validation'):
                fold = config['folds'][0]
                boundary = pd.Timestamp(fold['validation_start' if d.split == 'train' else 'validation_end'])
                self.assertTrue((d.view.samples.target_h24_utc < boundary).all())
        batch = next(iter(train))
        scale, offset, count = regional_persistence(data)
        weather = [i for i, c in enumerate(data.feature_cols) if c.startswith('weather_forecast_')]
        masks = [i for i in weather if data.feature_cols[i].endswith('_missing')]
        for method, head in [('gru-late-fusion', 'station'), ('transformer-late-fusion', 'station-mlp')]:
            model = residual_model(method, input_size=data.num_features, output_size=24, hidden_size=8,
                num_layers=1, dropout=0., num_stations=count, persistence_scale=scale, persistence_offset=offset,
                weather_feature_indices=weather, weather_mask_indices=masks, weather_mask_values=[1.]*len(masks),
                transformer_heads=4, transformer_ff_size=16, head_mode=head)
            model.eval()
            output = model(batch['x'], batch['station_ids'])
            self.assertEqual(tuple(output.shape), (len(batch['y']), 24))
            loss = from_training_dataset(data, weights=[1]*24, mode='station-balanced')
            value = loss(output, batch['y'], model.persistence(batch['x'], batch['station_ids']), batch['station_ids'])
            self.assertTrue(torch.isfinite(value))

    def test_regional_outputs_24_shorthand(self):
        from src.ai.train import parser, validate_args
        args = parser.parse_args(['--dataset-dir', str(self.path), '--train-method', 'gru-late-fusion',
                                  '--horizon', '1h', '--mlflow-ex-name', 'test', '--outputs', '24'])
        validate_args(args)
        self.assertEqual(args.target_horizons, list(range(1, 25)))
        self.assertIsNone(args.outputs)

    def test_o24_trainer_checkpoint_runtime_for_both_selected_architectures(self):
        from src.ai.train import parser, validate_args, build_sequence_trainer
        from src.ai.sequence_checkpoint import SequenceRuntime
        for method, head in [('gru-late-fusion', 'station'), ('transformer-late-fusion', 'station-mlp')]:
            args = parser.parse_args(['--dataset-dir', str(self.path), '--train-method', method,
                '--horizon', '1h', '--mlflow-ex-name', 'test', '--outputs', '24', '--device', 'cpu',
                '--head-mode', head, '--lstm-hidden-size', '8', '--lstm-num-layers', '1',
                '--transformer-heads', '4', '--transformer-ff-size', '16', '--batch-size', '4',
                '--rain-station', 'nham+hong_thai+bach_ma', '--lstm-seq-len', '96',
                '--change-loss', 'station-balanced', '--change-event-weight', '0', '--change-under-weight', '0',
                '--checkpoint-dir', str(self.path.parent/'o24-checkpoints')])
            validate_args(args)
            trainer = build_sequence_trainer(args)
            batch = next(iter(trainer.eval_dataloader))
            inputs, y, scales, _ = trainer.prepare_batch(batch, trainer.eval_dataloader.dataset)
            trainer.model.eval()
            with torch.no_grad():
                pred = trainer.predict_batch(trainer.model, inputs)
                expected = (pred*batch['target_scale']+batch['target_mean']).numpy()
                self.assertTrue(torch.isfinite(trainer.data_loss(pred, y, scales, inputs)))
            trainer.save_preprocessing()
            trainer.save_checkpoint()
            runtime = SequenceRuntime(trainer.checkpoint_path)
            actual = runtime.predict_mm(batch['x'], batch['station_ids'])
            self.assertEqual(actual.shape, (len(y), 24))
            np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-5)

    def test_scalers_ignore_validation_values(self):
        original_read = pd.read_parquet
        baseline = RegionalTrainingDataset(self.path, split='train', target_horizons=[3, 18])
        start = pd.Timestamp(self.config['folds'][0]['validation_start'])
        def changed(path, *a, **kw):
            frame = original_read(path,*a,**kw)
            if Path(path).name == 'water.parquet':
                frame.loc[frame.timestamp_utc.ge(start), ['001','002','003']] += 1e8
            return frame
        with patch('pandas.read_parquet',side_effect=changed):
            modified = RegionalTrainingDataset(self.path, split='validation', target_horizons=[3,18])
        self.assertEqual(baseline.view.normalizers, modified.view.normalizers)

    def test_missing_future_target_is_dropped_not_interpolated(self):
        frame = pd.read_parquet(self.path/'samples.parquet')
        water = self.level.copy()
        sample = frame.loc[frame.station_id.eq('001')].iloc[0]
        water.loc[sample.anchor_utc+pd.Timedelta(hours=18),'001'] = np.nan
        selected,_ = select_samples(frame,water,48,[3,6,12,18],self.config,'summer')
        self.assertFalse(((selected.station_id=='001') & selected.anchor_utc.eq(sample.anchor_utc)).any())

    def test_longer_context_and_purge_for_long_horizon(self):
        data = RegionalTrainingDataset(self.path, split='test_exploratory',sequence_length=144,target_horizons=[6,12])
        self.assertEqual(data[0]['x'].shape[0],144)
        train = RegionalTrainingDataset(self.path,split='train',target_horizons=[30])
        boundary = pd.Timestamp(self.config['folds'][0]['validation_start'])
        self.assertTrue((train.view.samples.anchor_utc+pd.Timedelta(hours=30)<boundary).all())
        self.assertEqual(train.num_targets,1)
        self.assertEqual(train.regional_metadata['training_selection']['effective_purge_hours'],30)

    def test_cli_accepts_selection_and_rejects_wrong_weight_count(self):
        import contextlib,io
        from src.ai.train import parser,validate_args,build_sequence_trainer
        options=['--dataset-dir',str(self.path),'--train-method','gru-late-fusion',
            '--horizon','1h','--target-horizons','3','6','12','18','--lstm-seq-len','48',
            '--rain-station','nham+hong_thai+bach_ma','--feature-set','water_level+rain+flood_season+weather_forecast',
            '--regional-fold','summer','--batch-size','4','--device','cpu','--num-epochs','1',
            '--lstm-hidden-size','4','--lstm-num-layers','1','--lstm-drop-out','0',
            '--change-loss','station-balanced','--checkpoint-dir',str(self.path.parent/'model'),
            '--mlflow-ex-name','synthetic-test']
        args=parser.parse_args(options+['--lstm-horizon-weights','1','1','1','1'])
        validate_args(args)
        trainer=build_sequence_trainer(args)
        data=trainer.train_dataloader.dataset
        self.assertEqual((data.seq_len,data.num_targets),(48,4))
        self.assertEqual(len(data.rain_station_ids),3)
        batch=next(iter(trainer.train_dataloader))
        inputs,y,scales,_=trainer.prepare_batch(batch,data)
        out=trainer.predict_batch(trainer.model,inputs)
        loss=trainer.data_loss(out,y,scales,inputs)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        from src.ai.sequence_checkpoint import model_spec,model_from_spec
        clone=model_from_spec(model_spec(trainer.model))
        clone.load_state_dict(trainer.model.state_dict())
        trainer.model.eval();clone.eval()
        torch.testing.assert_close(trainer.predict_batch(clone,inputs),trainer.predict_batch(trainer.model,inputs))
        # Save/load the complete preprocessing + architecture + weights path.
        trainer.save_preprocessing()
        trainer.save_checkpoint()
        from src.ai.sequence_checkpoint import SequenceRuntime
        runtime=SequenceRuntime(trainer.checkpoint_path)
        actual=runtime.predict_mm(batch['x'],batch['station_ids'])
        expected=(trainer.predict_batch(trainer.model,inputs).detach()*batch['target_scale']+batch['target_mean']).numpy()
        np.testing.assert_allclose(actual,expected,atol=1e-3,rtol=1e-5)
        self.assertEqual(actual.shape,(len(batch['y']),4))
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            validate_args(parser.parse_args(options+['--lstm-horizon-weights','1','1']))

    def test_short_history_missingness_and_validation_boundaries(self):
        frame=pd.read_parquet(self.path/'samples.parquet')
        water=self.level.copy()
        row=frame.loc[frame.station_id.eq('001')].iloc[-1]
        water.loc[row.anchor_utc-pd.Timedelta(hours=15):row.anchor_utc,'001']=np.nan
        selected,_=select_samples(frame,water,48,[6,12],self.config,'summer')
        self.assertFalse(((selected.station_id=='001') & selected.anchor_utc.eq(row.anchor_utc)).any())
        selected,_=select_samples(frame,self.level,96,[30],self.config,'summer')
        end=pd.Timestamp(self.config['folds'][0]['validation_end'])
        self.assertTrue((selected.loc[selected.summer_validation,'target_h30_utc']<end).all())

    def test_invalid_arguments_and_default_parity(self):
        for kw in [dict(sequence_length=0),dict(target_horizons=[0]),dict(target_horizons=[6,6]),dict(target_horizons=[])]:
            with self.assertRaises(ValueError):RegionalTrainingDataset(self.path,split='train',**kw)
        a=RegionalTrainingDataset(self.path,split='train')
        b=RegionalTrainingDataset(self.path,split='train',sequence_length=96,target_horizons=[6,12])
        self.assertEqual(len(a),len(b))
        for key in a[0]:torch.testing.assert_close(a[0][key],b[0][key],rtol=0,atol=0)

if __name__=='__main__':unittest.main()
