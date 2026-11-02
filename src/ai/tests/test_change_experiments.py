"""Generated tensors only: diagnostic forward/gradient checks, NO training loop."""
import contextlib
import io
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from src.ai.change_dynamics import ChangeDynamics, RegimeMeter, isolated_evaluation, fixed_probe_selection
from src.ai.change_objective import ChangeRiskLoss
from src.ai.residual_sequence import residual_model
from src.ai.sequence_checkpoint import model_spec, model_from_spec, SequenceRuntime
from src.ai.trainers import ResidualLSTMTrainer
from research.scripts.prepare_change_experiments import experiment_jobs
from research.scripts.summarize_change_experiments import summarize, history, gradient_history


class FakeRegional(Dataset):
    def __init__(self,split='train'):
        self.split=split
        self.target_horizons=[6,12]
        self.num_targets=2
        self.num_features=2
        self.seq_len=4
        self.feature_set='water_level+rain'
        self.feature_cols=['water_level_depth','rain']
        self.target_cols=['target_water_level_plus_360m','target_water_level_plus_720m']
        self.feature_scaler=self.target_scaler=None
        self.rain_station_ids=['rain']
        self.station_ids_validated=True
        self.scales=torch.tensor([[10.,100.],[1000.,20.]])
        self.means=torch.tensor([[500.,-30.],[80.,150.]])
        self.water_means=torch.tensor([100.,200.])
        self.water_scales=torch.tensor([10.,30.])
        self.ids=torch.tensor([0,0,0,1,1,1])
        self.x=torch.arange(48,dtype=torch.float32).reshape(6,4,2)/10
        current=self.x[:,-1,0]*self.water_scales[self.ids]+self.water_means[self.ids]
        delta=torch.tensor([[300.,-300.],[200.,-200.],[0.,0.],[100.,500.],[-500.,0.],[0.,-300.]])
        targets=current[:,None]+delta
        self.y=(targets-self.means[self.ids])/self.scales[self.ids]
        mapping={sid:dict(station_index=i,name=name) for i,(sid,name) in enumerate([('001','Station A'),('002','Station B')])}
        stats={sid:dict(targets={f'h{h}':dict(mean=float(self.means[i,j]),scale=float(self.scales[i,j]))
                               for j,h in enumerate(self.target_horizons)}) for sid,i in [('001',0),('002',1)]}
        self.regional_metadata=dict(station_mapping=mapping,normalizers={'stations':stats},
                                   target_horizons=self.target_horizons,time_features='none',dataset_manifest_sha256='synthetic')
        self.view=SimpleNamespace(samples=pd.DataFrame(dict(
            sample_id=np.arange(6)+(100 if split=='validation' else 0),
            station_id=['001']*3+['002']*3,station_index=self.ids.numpy(),
            anchor_utc=pd.date_range('2026-01-01',periods=6,freq='h',tz='UTC'),
            water_now_mm=current.numpy(),target_h6_mm=targets[:,0].numpy(),target_h12_mm=targets[:,1].numpy())))

    def __len__(self):return 6

    def __getitem__(self,index):
        sid=self.ids[index]
        return dict(x=self.x[index],y=self.y[index],station_ids=sid,
                    target_scale=self.scales[sid],target_mean=self.means[sid])


def make_model(data,head='shared'):
    return residual_model('gru-late-fusion',input_size=2,output_size=2,hidden_size=4,num_layers=1,
        dropout=.2,input_dropout=.2,head_dropout=.2,num_stations=2,
        persistence_scale=(data.water_scales[:,None]/data.scales).tolist(),
        persistence_offset=((data.water_means[:,None]-data.means)/data.scales).tolist(),head_mode=head)


def make_trainer(directory,head='shared',mode='station-balanced',alpha=0.):
    data=FakeRegional()
    model=make_model(data,head)
    # Nonzero synthetic weights exercise the encoder; no optimizer update is used.
    with torch.no_grad():
        model.linear.weight.fill_(.03)
        # Keep the synthetic ReLU active so gradient tests do not depend on
        # random station embeddings accidentally shutting off the whole head.
        model.fusion[0].weight.zero_()
        model.fusion[0].weight[:,:model.encoder.output_size].fill_(.1)
        model.fusion[0].bias.fill_(1.)
    loss=ChangeRiskLoss([1.,1.],[3,3],data.scales,mode=mode,event_weight=alpha,under_weight=0.)
    return ResidualLSTMTrainer(model=model,horizon='1h',num_epochs=20,
        train_dataloader=DataLoader(data,batch_size=4,shuffle=True,generator=torch.Generator().manual_seed(42)),
        eval_dataloader=DataLoader(FakeRegional('validation'),batch_size=4),
        test_dataloader=Mock(),loss_func=loss,optimizer=Mock(),scheduler=None,early_stopping=30,
        checkpoint_dir=directory,skip_test=True)


class ChangeExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_split_diagnostics_reject_nonfinite_predictions_before_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = make_trainer(tmp)
            with torch.no_grad():
                trainer.model.linear.weight.fill_(float('nan'))
            folder = Path(tmp) / 'invalid'
            folder.mkdir()
            with isolated_evaluation(trainer.model):
                with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                    ChangeDynamics().evaluate_split(trainer, trainer.eval_dataloader.dataset,
                                                    'val_eval', folder, 1)
            self.assertFalse((folder / 'validation_predictions.parquet').exists())
            self.assertFalse((folder / 'val_eval_metrics.csv').exists())

    def test_regime_metrics_strict_threshold_and_empty_groups(self):
        meter=RegimeMeter(2,1,'cpu',200.,200.)
        p=torch.tensor([[100.],[-500.],[200.],[400.]])
        y=torch.tensor([[300.],[-300.],[200.],[0.]])
        cells=(p-y).square()
        meter.update(p,y,torch.zeros_like(y),torch.ones_like(y),torch.tensor([0,0,1,1]),cells)
        table=meter.table({'001':{'station_index':0},'002':{'station_index':1}},[6],4)
        all_rows=table[table.level.eq('pooled_all')].set_index('group')
        self.assertEqual(all_rows.loc['extreme','n'],2)
        self.assertEqual(all_rows.loc['non_extreme','n'],2)
        self.assertEqual(all_rows.loc['all','large_errors'],1)
        self.assertAlmostEqual(all_rows.loc['all','rmse_mm'],60000**.5)
        self.assertEqual(all_rows.loc['all','rise_recall'],0)
        self.assertEqual(all_rows.loc['all','rise_false_alarm_ratio'],1)
        self.assertEqual(all_rows.loc['all','rise_shortfall_mm'],200)
        self.assertAlmostEqual(all_rows.loc['all','objective_contribution'],cells.mean().item())
        self.assertAlmostEqual(all_rows.loc['all','sse'],all_rows.loc['extreme','sse']+all_rows.loc['non_extreme','sse'])
        empty=table[table.station_id.eq('002')&table.group.eq('rise')].iloc[0]
        self.assertEqual(empty.n,0);self.assertTrue(np.isnan(empty.rmse_mm))

    def test_isolation_restores_randomness_and_mixed_modes_on_error(self):
        model=make_model(FakeRegional()).train();model.encoder.eval()
        modes=[m.training for m in model.modules()]
        cpu=torch.get_rng_state().clone();python=random.getstate();numpy=np.random.get_state()
        with self.assertRaisesRegex(RuntimeError,'probe'):
            with isolated_evaluation(model):
                self.assertFalse(any(m.training for m in model.modules()))
                random.random();np.random.rand();torch.rand(3)
                raise RuntimeError('probe')
        self.assertTrue(torch.equal(cpu,torch.get_rng_state()))
        self.assertEqual(python,random.getstate())
        self.assertTrue(np.array_equal(numpy[1],np.random.get_state()[1]))
        self.assertEqual(modes,[m.training for m in model.modules()])

    def test_probe_uses_train_only_and_population_weights(self):
        data=FakeRegional();a=fixed_probe_selection(data,200,4);b=fixed_probe_selection(data,200,4)
        pd.testing.assert_frame_equal(a,b)
        self.assertLessEqual(a.groupby('station_id').size().max(),4)
        self.assertAlmostEqual(a.objective_weight.sum(),1.)
        self.assertEqual(len(a.dataset_index.unique()),len(a))
        with self.assertRaises(ValueError):fixed_probe_selection(FakeRegional('validation'),200,4)

    def test_full_probe_gradient_matches_actual_station_loss_contribution(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer=make_trainer(tmp)
            folder=Path(tmp)/'probe';folder.mkdir()
            dynamics=ChangeDynamics(gradients=True,probe_samples=8)
            with isolated_evaluation(trainer.model):
                dynamics.gradient_probe(trainer,folder)
                expected=[]
                data=trainer.train_dataloader.dataset
                parameters=list(trainer.model.encoder.parameters())
                for ids in [[0,1,2],[3,4,5]]:
                    batch=next(iter(DataLoader(torch.utils.data.Subset(data,ids),batch_size=3)))
                    inputs,y,scales,_=trainer.prepare_batch(batch,data)
                    loss=trainer.data_loss(trainer.predict_batch(trainer.model,inputs),y,scales,inputs)*.5
                    grads=torch.autograd.grad(loss,parameters)
                    expected.append(torch.cat([g.detach().flatten() for g in grads]).double())
            norms=pd.read_csv(folder/'gradient_norms.csv')
            np.testing.assert_allclose(norms.encoder_gradient_norm,
                [v.norm().item() for v in expected],rtol=1e-5)
            cosine=(expected[0]@expected[1])/(expected[0].norm()*expected[1].norm())
            self.assertAlmostEqual(pd.read_csv(folder/'gradient_cosines.csv').iloc[0].cosine,cosine.item(),places=5)

    def test_epoch_snapshot_physical_predictions_metrics_and_no_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer=make_trainer(tmp)
            trainer.best_checkpoint_epoch=1;trainer.best_val_loss=.2
            trainer.last_validation_metrics={'val/rmse_mm':123.}
            trainer.last_validation_mse=torch.tensor([2.,3.])
            for parameter in trainer.model.parameters():parameter.grad=torch.ones_like(parameter)
            weights={k:v.detach().clone() for k,v in trainer.model.state_dict().items()}
            grads=[p.grad.clone() for p in trainer.model.parameters()]
            state=trainer.train_dataloader.generator.get_state().clone()
            global_rng=torch.get_rng_state().clone()
            trainer.model.train()
            with contextlib.redirect_stdout(io.StringIO()),patch('mlflow.active_run',return_value=None):
                folder=ChangeDynamics(gradients=True,probe_samples=4).run(trainer,1)
            self.assertTrue(torch.equal(global_rng,torch.get_rng_state()))
            self.assertTrue(torch.equal(state,trainer.train_dataloader.generator.get_state()))
            self.assertTrue(trainer.model.training)
            self.assertEqual(trainer.last_validation_metrics,{'val/rmse_mm':123.})
            torch.testing.assert_close(trainer.last_validation_mse,torch.tensor([2.,3.]))
            self.assertEqual(trainer.best_checkpoint_epoch,1)
            for k,v in trainer.model.state_dict().items():torch.testing.assert_close(v,weights[k],rtol=0,atol=0)
            for p,g in zip(trainer.model.parameters(),grads):torch.testing.assert_close(p.grad,g,rtol=0,atol=0)
            trainer.optimizer.step.assert_not_called();trainer.optimizer.zero_grad.assert_not_called()
            trainer.test_dataloader.assert_not_called()
            self.assertFalse(trainer.checkpoint_path.exists())
            frame=pd.read_parquet(folder/'validation_predictions.parquet')
            self.assertEqual(len(frame),12)
            self.assertFalse(frame.duplicated(['sample_id','horizon_hours']).any())
            self.assertTrue(frame.station_id.isin(['001','002']).all())
            self.assertTrue((frame.target_utc-frame.anchor_utc).eq(pd.to_timedelta(frame.horizon_hours,unit='h')).all())
            np.testing.assert_allclose(frame.predicted_mm-frame.actual_mm,frame.error_mm)
            table=pd.read_csv(folder/'val_eval_metrics.csv')
            overall=table[table.level.eq('pooled_all')&table.group.eq('all')].iloc[0]
            self.assertAlmostEqual(overall.objective_contribution,frame.objective_cell.sum()/6,places=6)
            self.assertAlmostEqual(overall.mae_mm,frame.error_mm.abs().mean(),places=6)
            runtime=SequenceRuntime(folder/'model.pt')
            data=trainer.eval_dataloader.dataset
            actual=runtime.predict_mm(data.x,data.ids)
            np.testing.assert_allclose(actual,frame.predicted_mm.to_numpy().reshape(6,2),atol=.001,rtol=1e-5)
            norm=pd.read_csv(folder/'gradient_norms.csv')
            self.assertTrue(norm.encoder_gradient_norm.gt(0).all())
            pairs=pd.read_csv(folder/'gradient_cosines.csv')
            self.assertTrue(pairs.valid.all())
            self.assertTrue(pairs.cosine.between(-1.000001,1.000001).all())
            job=experiment_jobs('/tmp/synthetic',tmp)[0];job['checkpoint_root']=tmp
            history_table=gradient_history({'jobs':[job]},'gradient_norms.csv')
            self.assertEqual(set(history_table.station_id),{'001','002'})
            self.assertEqual(set(history_table.epoch),{1})

    def test_mlflow_side_effects_are_isolated_and_evaluation_is_repeatable(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer=make_trainer(tmp)
            trainer.best_checkpoint_epoch=0;trainer.best_val_loss=1.
            rng=torch.get_rng_state().clone()
            def random_logger(*args,**kwargs):
                torch.rand(2);np.random.rand();random.random()
            with contextlib.redirect_stdout(io.StringIO()),patch('mlflow.active_run',return_value=object()), \
                    patch('mlflow.log_metrics',side_effect=random_logger):
                first=ChangeDynamics().run(trainer,0)
                second=ChangeDynamics().run(trainer,1)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            a=pd.read_parquet(first/'validation_predictions.parquet').drop(columns='epoch')
            b=pd.read_parquet(second/'validation_predictions.parquet').drop(columns='epoch')
            pd.testing.assert_frame_equal(a,b)
            trainer.model.eval();data=trainer.eval_dataloader.dataset
            batch=next(iter(DataLoader(data,batch_size=6)))
            inputs,y,scale,_=trainer.prepare_batch(batch,data)
            with torch.inference_mode():
                expected=trainer.data_loss(trainer.predict_batch(trainer.model,inputs),y,scale,inputs).item()
            self.assertAlmostEqual(a.objective_cell.sum()/6,expected,places=6)
            job=experiment_jobs('/tmp/synthetic',tmp)[0];job['checkpoint_root']=tmp
            manifest={'jobs':[job]}
            summary=summarize(manifest)
            self.assertEqual(summary.iloc[0].best_epoch,0)
            self.assertEqual(summary.iloc[0].status,'partial')
            h=history(manifest)
            self.assertEqual(set(h.epoch),{0,1})
            self.assertEqual(set(h.split),{'train_eval','val_eval'})

    def test_cli_builds_diagnostic_station_heads_without_training(self):
        from src.ai.train import parser,validate_args,build_sequence_trainer
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'samples.parquet').touch()
            data=FakeRegional();validation=FakeRegional('validation')
            data.view.mapping=data.regional_metadata['station_mapping']
            data.view.normalizers=data.regional_metadata['normalizers']
            for sid,i in [('001',0),('002',1)]:
                data.view.normalizers['stations'][sid].update(
                    persistence_scale=(data.water_scales[i]/data.scales[i]).tolist(),
                    persistence_offset=((data.water_means[i]-data.means[i])/data.scales[i]).tolist())
            args=parser.parse_args(['--dataset-dir',tmp,'--train-method','transformer-late-fusion',
                '--horizon','1h','--mlflow-ex-name','synthetic','--device','cpu',
                '--lstm-hidden-size','4','--lstm-num-layers','1','--head-mode','station',
                '--diagnose-change-dynamics','--diagnose-station-gradients','--skip-test',
                '--change-loss','station-balanced','--change-event-weight','0','--change-under-weight','0',
                '--target-horizons','6','12','--checkpoint-dir',str(Path(tmp)/'model')])
            validate_args(args)
            loaders=(DataLoader(data,batch_size=4),DataLoader(validation,batch_size=4),Mock())
            with patch('src.ai.train.regional_loaders',return_value=loaders):
                trainer=build_sequence_trainer(args)
            self.assertEqual(trainer.model.head_mode,'station')
            self.assertTrue(trainer.skip_test)
            self.assertIsInstance(trainer.dynamics,ChangeDynamics)
            self.assertTrue(trainer.dynamics.gradients)
            self.assertIn('head_station',str(trainer.checkpoint_path))
            self.assertFalse(trainer.checkpoint_path.exists())

    def test_mocked_orchestration_saves_final_diagnostics_before_early_stop(self):
        # No fit: the epoch update and validation are mocks, as is the diagnostic.
        with tempfile.TemporaryDirectory() as tmp:
            trainer=make_trainer(tmp)
            trainer.train_epoch=Mock(return_value=.1)
            trainer.evaluate_loss=Mock(return_value=.2)
            trainer.early_stopping=1
            def record(epoch,loss):trainer.epochs_without_improvement=epoch-1
            trainer.record_validation=Mock(side_effect=record)
            trainer.dynamics=Mock();trainer.dynamics.every=10
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()), \
                    patch('src.ai.trainers.mlflow',Mock()):
                last=trainer.run_epochs(1,20)
            self.assertEqual(last,2)
            trainer.dynamics.run.assert_called_once_with(trainer,2)
            trainer.optimizer.step.assert_not_called()

    def test_mocked_training_entrypoint_skip_test_does_not_call_test(self):
        # Control-flow test only: no training, forward pass or optimizer update.
        with tempfile.TemporaryDirectory() as tmp:
            trainer=make_trainer(tmp)
            trainer.print_settings=Mock();trainer.prepare_checkpoint=Mock()
            trainer.train_epochs=Mock(return_value=20);trainer.restore_best_checkpoint=Mock()
            trainer.save_preprocessing=Mock();trainer.test=Mock(side_effect=AssertionError('Test must be skipped'))
            trainer.dynamics=Mock()
            with patch('src.ai.trainers.mlflow',Mock()):trainer.train()
            trainer.test.assert_not_called()
            trainer.dynamics.run.assert_called_once_with(trainer,0)
            trainer.optimizer.step.assert_not_called()

    def test_no_iteration_of_training_or_test_loaders_and_zero_gradient_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer=make_trainer(tmp)
            with torch.no_grad():trainer.model.linear.weight.zero_();trainer.model.linear.bias.zero_()
            trainer.best_checkpoint_epoch=0;trainer.best_val_loss=1.
            class NeverIterate:
                dataset=trainer.train_dataloader.dataset
                batch_size=4
                def __iter__(self):raise AssertionError('Training sampler must not be consumed')
            trainer.train_dataloader=NeverIterate()
            with contextlib.redirect_stdout(io.StringIO()),patch('mlflow.active_run',return_value=None):
                folder=ChangeDynamics(gradients=True).run(trainer,0)
            pairs=pd.read_csv(folder/'gradient_cosines.csv')
            self.assertTrue(pairs.cosine.isna().all())
            self.assertTrue(pairs.status.eq('zero_gradient').all())
            self.assertFalse(pairs.valid.any())

    def test_station_head_lookup_gradient_and_checkpoint_compatibility(self):
        data=FakeRegional()
        torch.manual_seed(42);shared=make_model(data);rng=torch.get_rng_state().clone()
        torch.manual_seed(42);station=make_model(data,'station')
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertNotIn('head_mode',model_spec(shared)['options'])
        for name,p in shared.state_dict().items():
            if not name.startswith('linear.'):
                torch.testing.assert_close(p,station.state_dict()[name],rtol=0,atol=0)
        station.eval()
        ids=torch.tensor([1,0])
        x=data.x[:2]
        torch.testing.assert_close(station(x,ids),station.persistence(x,ids))
        with torch.no_grad():station.linear.bias[0].fill_(1);station.linear.bias[1].fill_(2)
        torch.testing.assert_close(station(x,ids)-station.persistence(x,ids),torch.tensor([[2.,2.],[1.,1.]]))
        station(x[:1],torch.tensor([0])).sum().backward()
        self.assertTrue(station.linear.bias.grad[1].eq(0).all())
        clone=model_from_spec(json.loads(json.dumps(model_spec(station)))).eval()
        clone.load_state_dict(station.state_dict())
        torch.testing.assert_close(clone(x,ids),station(x,ids),rtol=0,atol=0)
        for kind in ['lstm','gru','tcn','cnn-gru','transformer-late-fusion']:
            model=residual_model(kind,input_size=2,num_stations=2,hidden_size=4,num_layers=1,
                persistence_scale=[[1.,1.],[1.,1.]],persistence_offset=[[0.,0.],[0.,0.]],head_mode='station')
            self.assertEqual(model(data.x[:2],ids).shape,(2,2))
        with self.assertRaises(ValueError):residual_model('gru',input_size=2,head_mode='station')

    def test_station_head_snapshot_restores_physical_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer=make_trainer(tmp,'station','combined',.25)
            trainer.best_checkpoint_epoch=0;trainer.best_val_loss=1.
            with contextlib.redirect_stdout(io.StringIO()),patch('mlflow.active_run',return_value=None):
                folder=ChangeDynamics().run(trainer,0)
            data=trainer.eval_dataloader.dataset
            frame=pd.read_parquet(folder/'validation_predictions.parquet')
            runtime=SequenceRuntime(folder/'model.pt')
            np.testing.assert_allclose(runtime.predict_mm(data.x,data.ids),
                frame.predicted_mm.to_numpy().reshape(6,2),atol=.001,rtol=1e-5)

    def test_prepared_matrix_is_opt_in_and_has_no_test_or_unused_weights(self):
        from src.ai.train import parser
        jobs=experiment_jobs('/tmp/synthetic-regional-dataset','/tmp/synthetic-prepared')
        self.assertEqual(len(jobs),28)
        self.assertEqual(len({j['checkpoint_root'] for j in jobs}),28)
        for job in jobs:
            args=parser.parse_args(job['argv'][3:])
            self.assertTrue(args.skip_test and args.diagnose_change_dynamics)
            self.assertEqual(args.change_under_weight,0)
            if args.change_loss=='station-balanced':self.assertEqual(args.change_event_weight,0)
            self.assertEqual(args.target_horizons,[6,12])
            self.assertEqual(args.lstm_seq_len,96)
            self.assertEqual(args.num_epochs,20)
        result=summarize({'jobs':jobs})
        self.assertTrue(result.status.eq('not_run').all())

    def test_cli_rejects_invalid_diagnostic_combinations_and_warns_unused_flags(self):
        from src.ai.train import parser,validate_args
        basic=['--train-method','gru','--horizon','1h','--mlflow-ex-name','synthetic']
        for extra in [['--head-mode','station'],['--diagnose-change-dynamics'],
                      ['--diagnose-station-gradients'],['--dynamics-every','0']]:
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                validate_args(parser.parse_args(basic+extra))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'samples.parquet').touch()
            args=parser.parse_args(basic+['--dataset-dir',tmp,'--change-loss','station-balanced',
                                         '--change-event-weight','2'])
            with self.assertWarnsRegex(UserWarning,'ignores event/under'):
                validate_args(args)


if __name__=='__main__':unittest.main()
