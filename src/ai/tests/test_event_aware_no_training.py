"""Synthetic forward/backward checks only: no optimizer, training loop, or data fitting."""
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import joblib
import numpy as np
import pandas as pd
import torch

from src.ai.residual_sequence import residual_model
from src.ai.sequence_checkpoint import model_spec, model_from_spec, SequenceRuntime
from src.ai.change_objective import ChangeRiskLoss
from src.ai.trainers import ResidualLSTMTrainer
from src.ai.event_evaluation import event_report, match_budget, plot_event_comparison
from src.ai.change_dynamics import ChangeDynamics, classification_elements


class EventAwareNoTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def model(self):
        torch.manual_seed(7)
        return residual_model('transformer-late-fusion',input_size=4,output_size=2,hidden_size=8,
            num_layers=1,dropout=0.,num_stations=2,station_embedding_size=2,
            persistence_scale=[[2.,4.],[1.,3.]],persistence_offset=[[.5,-1.],[2.,1.]],
            transformer_heads=2,transformer_ff_size=16,head_mode='station-mlp',event_aware=True).eval()

    def loss(self, weight=0.):
        return ChangeRiskLoss([1.,1.],[3,1],[[100.,400.],[200.,300.]],'station-balanced',
                              event_weight=0.,under_weight=0.,rise_event_weight=weight)

    def test_forward_components_initialization_and_negative_base(self):
        m=self.model();x=torch.randn(4,8,4);ids=torch.tensor([0,1,0,1]);p,a=m(x,ids,return_diagnostics=True)
        self.assertEqual(p.shape,(4,2)); self.assertTrue(torch.isfinite(p).all())
        self.assertTrue((a['amplitude_normalized']>=0).all())
        self.assertTrue((a['correction_normalized']<.001).all())
        torch.testing.assert_close(p,m.persistence(x,ids)+a['final_delta_normalized'])
        torch.testing.assert_close(m(x,ids),p)
        with torch.no_grad():m.linear.biases[-1].fill_(-1.)
        p,a=m(x,ids,return_diagnostics=True)
        self.assertTrue((a['base_delta_normalized']<0).all());self.assertTrue((a['final_delta_normalized']<0).all())

    def test_original_transformer_defaults_and_weights_are_unchanged(self):
        event=self.model()
        kwargs=dict(event.architecture_config);kwargs.pop('event_aware');kind=kwargs.pop('architecture')
        torch.manual_seed(7);original=residual_model(kind,**kwargs).eval()
        self.assertFalse(original.event_aware)
        for name,value in original.state_dict().items():
            torch.testing.assert_close(value,event.state_dict()[name],rtol=0,atol=0)
        x=torch.randn(2,8,4);ids=torch.tensor([0,1]);p,aux=event(x,ids,return_diagnostics=True)
        torch.testing.assert_close(p-original(x,ids),aux['correction_normalized'],rtol=1e-4,atol=1e-6)

    def test_event_loss_strict_threshold_per_horizon_and_weights(self):
        loss=self.loss(2.);ids=torch.tensor([0,1]);p=torch.zeros(2,2);base=torch.zeros(2,2)
        # station 0: exactly 200 mm is negative; +204 mm at h12 is positive.
        y=torch.tensor([[2.,.51],[-2.,1.]])
        raw=loss.elements(p,y,base,ids,apply_rise_weight=False)
        weighted=loss.elements(p,y,base,ids)
        torch.testing.assert_close(weighted[0,0],raw[0,0]);torch.testing.assert_close(weighted[0,1],raw[0,1]*3)
        torch.testing.assert_close(weighted[1,0],raw[1,0]);torch.testing.assert_close(weighted[1,1],raw[1,1]*3)
        # Classification uses raw logits and exactly the same > threshold definition.
        trainer=object.__new__(ResidualLSTMTrainer);trainer.model=SimpleNamespace(persistence=lambda **kw:base)
        trainer.loss_func=loss;trainer.rise_positive_weight=1.
        trainer.rise_classification_weight=.1
        logits=torch.tensor([[0.,1.],[-1.,2.]])
        actual=trainer.rise_classification_loss(logits,y,torch.tensor([[100.,400.],[200.,300.]]),{'x':torch.zeros(2,1,1),'station_ids':ids})
        labels=torch.tensor([[0.,1.],[0.,1.]])
        expected=torch.nn.functional.binary_cross_entropy_with_logits(logits,labels)
        torch.testing.assert_close(actual,expected)
        cells=classification_elements(trainer,logits,y,base,torch.tensor([[100.,400.],[200.,300.]]))
        torch.testing.assert_close(cells.sum(dim=1).mean(),expected*.1)

    def test_validation_regression_ignores_optional_rise_weight(self):
        ids=torch.tensor([0,1]);pred=torch.tensor([[0.,0.],[1.,-1.]])
        target=torch.tensor([[3.,1.],[2.,1.]])
        base=torch.zeros_like(target)
        standard=self.loss(0.);weighted=self.loss(2.)
        torch.testing.assert_close(weighted(pred,target,base,ids,apply_rise_weight=False),
                                   standard(pred,target,base,ids))
        self.assertGreater(weighted(pred,target,base,ids).item(),
                           standard(pred,target,base,ids).item())

    def test_event_validation_selects_unweighted_regression(self):
        m=self.model();ids=torch.tensor([0,1]);x=torch.randn(2,8,4)
        y=m.persistence(x,ids).detach()+torch.tensor([[3.,1.],[2.,1.]])
        batch={'x':x,'y':y,'station_ids':ids,'target_scale':torch.tensor([[100.,400.],[200.,300.]])}
        class Loader:
            dataset=SimpleNamespace(target_cols=['h6','h12'],target_scaler=None)
            def __iter__(self):return iter([batch])
        trainer=object.__new__(ResidualLSTMTrainer);trainer.model=m;trainer.device=torch.device('cpu')
        trainer.loss_func=self.loss(2.);trainer.rise_positive_weight=1.;trainer.rise_classification_weight=.1
        trainer.validation_threshold_mm=200.;trainer.checkpoint_metric='val_loss'
        with torch.inference_mode():
            p=m(x,ids);expected=self.loss(0.)(p,y,m.persistence(x,ids),ids).item()
        measured=trainer.evaluate_loss(Loader())
        self.assertAlmostEqual(measured,expected,places=6)
        self.assertIn('val/rise_classification_loss',trainer.last_validation_metrics)

    def test_variant_flags_parse_without_running_trainer(self):
        from src.ai.train import parser, validate_args
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'samples.parquet').touch()
            common=['--dataset-dir',tmp,'--train-method','transformer-late-fusion','--horizon','1h',
                    '--mlflow-ex-name','synthetic','--head-mode','station-mlp',
                    '--change-loss','station-balanced','--change-event-weight','0','--change-under-weight','0']
            for extra in [[],['--event-aware'],['--rise-event-weight','1'],
                          ['--event-aware','--rise-event-weight','2']]:
                args=parser.parse_args(common+extra);validate_args(args)
            args=parser.parse_args(common+['--event-aware','--prediction-mode','direct'])
            import contextlib
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):validate_args(args)

    def test_gradients_reach_heads_and_backbone(self):
        m=self.model();x=torch.randn(4,8,4);ids=torch.tensor([0,1,0,1])
        with torch.no_grad():
            m.rise_head.weight.fill_(.05);m.amplitude_head.weight.fill_(.05);m.linear.weights[-1].fill_(.05)
        p,a=m(x,ids,return_diagnostics=True)
        loss=p.square().mean()+.1*torch.nn.functional.binary_cross_entropy_with_logits(a['rise_logits'],torch.ones_like(p))
        loss.backward()
        for name,tensor in [('rise',m.rise_head.weight),('amplitude',m.amplitude_head.weight),
                            ('base',m.linear.weights[-1]),('encoder',m.encoder.temporal.projection.weight)]:
            self.assertIsNotNone(tensor.grad,name);self.assertTrue(torch.isfinite(tensor.grad).all(),name)
            self.assertGreater(tensor.grad.abs().sum().item(),0.,name)

    def test_checkpoint_and_runtime_physical_units(self):
        m=self.model();x=torch.randn(2,8,4);ids=torch.tensor([0,1]);clone=model_from_spec(model_spec(m)).eval();clone.load_state_dict(m.state_dict())
        torch.testing.assert_close(m(x,ids),clone(x,ids))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.pt';torch.save(m.state_dict(),path)
            stats={'s0':{'targets':{'h6':{'scale':100.,'mean':10.},'h12':{'scale':200.,'mean':20.}}},
                   's1':{'targets':{'h6':{'scale':300.,'mean':30.},'h12':{'scale':400.,'mean':40.}}}}
            joblib.dump({'model_spec':model_spec(m),'target_columns':['target_water_level_plus_360m','target_water_level_plus_720m'],
                'regional':{'station_mapping':{'s0':{'station_index':0},'s1':{'station_index':1}},
                            'normalizers':{'stations':stats},'target_horizons':[6,12]}},path.with_suffix('.preprocessing.joblib'))
            runtime=SequenceRuntime(path);a=runtime.predict_with_diagnostics_mm(x,ids);p=runtime.predict_mm(x,ids)
            np.testing.assert_allclose(a['predicted_mm'],p);np.testing.assert_allclose(a['final_delta_mm'],a['base_delta_mm']+a['correction_mm'],rtol=1e-6)
            self.assertTrue(np.isfinite(a['amplitude_mm']).all())

    def test_validation_snapshot_exports_event_components_without_fitting(self):
        model=self.model()
        class Dataset(torch.utils.data.Dataset):
            regional_metadata={'station_mapping':{'s0':{'station_index':0,'name':'Zero'},'s1':{'station_index':1,'name':'One'}}}
            target_horizons=[6,12]
            view=SimpleNamespace(samples=pd.DataFrame({'sample_id':[0,1], 'station_id':['s0','s1'],
                'anchor_utc':pd.date_range('2026-01-01',periods=2,freq='h',tz='UTC')}))
            def __len__(self):return 2
            def __getitem__(self,i):
                return {'x':torch.ones(8,4)*i,'y':torch.zeros(2),'station_ids':torch.tensor(i),
                        'target_scale':torch.tensor([100.,200.]),'target_mean':torch.tensor([10.,20.])}
        trainer=object.__new__(ResidualLSTMTrainer);trainer.model=model;trainer.device=torch.device('cpu')
        trainer.train_dataloader=SimpleNamespace(batch_size=2);trainer.loss_func=self.loss(1.)
        trainer.validation_threshold_mm=200.;trainer.rise_positive_weight=1.;trainer.rise_classification_weight=.1
        with tempfile.TemporaryDirectory() as tmp:
            ChangeDynamics().evaluate_split(trainer,Dataset(),'val_eval',Path(tmp),3)
            saved=pd.read_parquet(Path(tmp)/'validation_predictions.parquet')
            self.assertEqual(len(saved),4)
            for name in ['rise_logit','rise_probability','base_delta_mm','amplitude_mm','correction_mm']:
                self.assertTrue(np.isfinite(saved[name]).all(),name)
            np.testing.assert_allclose(saved.predicted_mm-saved.current_mm,
                                       saved.base_delta_mm+saved.correction_mm,atol=1e-4)

    def test_evaluation_thresholds_regimes_calibration_and_plot(self):
        f=pd.DataFrame({'station_id':['s']*5,'horizon_hours':[6]*5,'anchor_utc':pd.date_range('2026-01-01',periods=5,freq='h'),
          'current_mm':[100.]*5,'actual_mm':[50.,100.,150.,300.,301.],
          'predicted_mm':[60.,110.,220.,320.,360.],'rise_probability':[.1,.2,.3,.7,.9]})
        r=event_report(f,probability_thresholds=(.5,.8));self.assertEqual(len(r['summary']),1)
        c=r['confusion'];reg=c[(c.method=='regression')&(c.threshold==200)].iloc[0];self.assertEqual((reg.tp,reg.fp,reg.fn),(1,1,0))
        self.assertEqual(int(r['regimes']['n'].sum()),5);self.assertIn('classification',set(r['precision_recall'].method));self.assertIn('regression',set(r['precision_recall'].method))
        self.assertFalse(r['calibration'].empty);self.assertEqual(len(match_budget(c,max_false_positive_rate=.5)),1)
        fig=plot_event_comparison(f.assign(correction_mm=1.,amplitude_mm=2.),'s',6);self.assertEqual(len(fig.axes),4)
        import matplotlib.pyplot as plt;plt.close(fig)

if __name__=='__main__':unittest.main()
