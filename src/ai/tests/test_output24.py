"""Direct label construction, matched O5/O24 rows, coefficients and reload."""
import copy
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from src.ai.losses import CommonHorizonAuxiliaryLoss,HorizonWeightedTailLoss
from src.ai.comparison import load_candidate,paired_metrics
from research.scripts.build_output24_dataset import COMMON,TARGETS,add_targets,assert_matched_population,build,DEFAULT_OUTPUT
from research.scripts.output_horizon_experiment import prepare_data,make_model,parser,evaluate,validation_score,write_comparisons,training_objective


class OutputLossTests(unittest.TestCase):
    def test_signed_correction_penalty_cli_and_loss_backward(self):
        for outputs,columns in [(5,COMMON),(24,TARGETS)]:
            args=parser().parse_args(['--outputs',str(outputs),'--lstm-correction-penalty','-0.1'])
            self.assertEqual(args.lstm_correction_penalty,-.1)
            loss=CommonHorizonAuxiliaryLoss(columns,[100]*outputs,auxiliary_weight=2.)
            pred=torch.ones(2,outputs,requires_grad=True)
            value=loss(pred,torch.zeros_like(pred))+args.lstm_correction_penalty*loss.correction_loss(pred)
            value.backward()
            self.assertTrue(torch.isfinite(value))
            self.assertTrue(torch.isfinite(pred.grad).all())
            self.assertTrue(pred.grad.gt(0).all())
        for value in ['-1','-1.1','nan','inf','-inf']:
            with self.subTest(value=value),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                parser().parse_args(['--outputs','24',f'--lstm-correction-penalty={value}'])

    def test_common_coefficients_and_gradients_do_not_shrink_with_24_outputs(self):
        p=torch.randn(3,24,dtype=torch.float64,requires_grad=True);y=torch.randn_like(p)
        scales=np.arange(100.,124.)
        loss=CommonHorizonAuxiliaryLoss(TARGETS,scales,auxiliary_weight=.25)
        idx=loss.primary_indices
        main=HorizonWeightedTailLoss([1]*5,scales[idx.numpy()],strength=.5)
        main_value=main(p.index_select(1,idx),y.index_select(1,idx))
        components=loss.components(p,y)
        torch.testing.assert_close(components[0],main_value,rtol=0,atol=0)
        all_grad=torch.autograd.grad(loss(p,y),p,retain_graph=True)[0]
        main_grad=torch.autograd.grad(main_value,p)[0]
        torch.testing.assert_close(all_grad[:,idx],main_grad[:,idx],rtol=0,atol=0)
        self.assertTrue(all_grad[:,loss.auxiliary_indices].abs().gt(0).any())
        c=torch.randn(2,24,dtype=torch.float64,requires_grad=True)
        extra=loss.correction_loss(c)
        expected=c[:,idx].square().mean()+.25*c[:,loss.auxiliary_indices].square().mean()
        torch.testing.assert_close(extra,expected,rtol=0,atol=0)

    def test_five_output_loss_and_correction_exactly_match_B(self):
        p=torch.randn(2,5,dtype=torch.float64);y=torch.randn_like(p)
        loss=CommonHorizonAuxiliaryLoss(COMMON,[100]*5)
        b=HorizonWeightedTailLoss([1]*5,[100]*5,strength=.5)
        torch.testing.assert_close(loss(p,y),b(p,y),rtol=0,atol=0)
        torch.testing.assert_close(loss.correction_loss(p),p.square().mean(),rtol=0,atol=0)
        _,aux=loss.components(p,y);self.assertEqual(aux.item(),0.)

    def test_24h_has_same_coefficient_as_five_output_control(self):
        loss=CommonHorizonAuxiliaryLoss(TARGETS,[100]*24,tail_strength=0.)
        p=torch.zeros(1,24);p[0,-1]=1.
        self.assertAlmostEqual(loss(p,torch.zeros_like(p)).item(),1/5)
        with self.assertRaises(ValueError):CommonHorizonAuxiliaryLoss(TARGETS,[100]*24,auxiliary_weight=-1)
        with self.assertRaises(ValueError):CommonHorizonAuxiliaryLoss(TARGETS[:-1],[100]*23)

    def test_checkpoint_score_ignores_extra_horizons_for_fair_selection(self):
        a=dict(common_above_200=5,common_rmse_mm=70.,loss=.5,all_above_200=100)
        b=dict(common_above_200=6,common_rmse_mm=50.,loss=.1,all_above_200=6)
        self.assertLess(validation_score(a,'large_error_rate'),validation_score(b,'large_error_rate'))
        self.assertGreater(validation_score(a,'val_loss'),validation_score(b,'val_loss'))

    def test_validation_logs_common_five_and_all_24_separately(self):
        loss=CommonHorizonAuxiliaryLoss(TARGETS,[100]*24)
        class Constant(nn.Module):
            def forward(self,x):
                out=torch.full((len(x),24),3.)
                out[:,loss.primary_indices]=0.
                return out
        loader=DataLoader(TensorDataset(torch.zeros(3,2,1),torch.zeros(3,24)),batch_size=2)
        saved=dict(target_columns=TARGETS,target_scaler=type('Scale',(),dict(scale_=np.full(24,100.)))())
        result=evaluate(Constant(),loader,loss,saved,'cpu')
        self.assertEqual(result['common_n'],15)
        self.assertEqual(result['common_above_200'],0)
        self.assertEqual(result['common_rmse_mm'],0)
        self.assertEqual(result['all_n'],72)
        self.assertEqual(result['all_above_200'],57)


class OutputDatasetTests(unittest.TestCase):
    def test_plain_lstm_training_and_reload_for_five_and_24_outputs(self):
        if not DEFAULT_OUTPUT.exists():self.skipTest('Build local O24 dataset first')
        from src.ai.lstm import LSTM,ResidualLSTM
        for outputs in [5,24]:
            _,datasets,saved=prepare_data(DEFAULT_OUTPUT,outputs,96)
            args=parser().parse_args(['--outputs',str(outputs),'--train-method','lstm'])
            saved['train_method']='lstm'
            model=make_model(saved,args,'cpu')
            self.assertIs(type(model),LSTM)
            self.assertNotIsInstance(model,ResidualLSTM)
            x,y=datasets['train'][0];x=x[None];y=y[None]
            loss=CommonHorizonAuxiliaryLoss(saved['target_columns'],saved['target_scaler'].scale_)
            pred=model(x);data_loss=loss(pred,y)
            objective=training_objective(model,x,pred,data_loss,loss,0.)
            self.assertIs(objective,data_loss)
            objective.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
            with self.assertRaisesRegex(ValueError,'requires residual_lstm'):
                training_objective(model,x,pred,data_loss,loss,-.1)
            model.eval();buffer=io.BytesIO();torch.save(model.state_dict(),buffer)
            reloaded=load_candidate(buffer.getvalue(),saved,'cpu')
            self.assertIs(type(reloaded),LSTM)
            torch.testing.assert_close(model(x),reloaded(x))
            self.assertEqual(model(x).shape,(1,outputs))

    def source(self,n=300):
        index=pd.date_range('2026-01-01',periods=n,freq='h',tz='UTC',name='timestamp_utc')
        d=pd.DataFrame({'water_level_depth':np.arange(n,dtype=float)},index=index)
        for h,c in zip((1,3,6,12,24),COMMON):d[c]=d.water_level_depth.shift(-h)
        return d

    def test_labels_are_exact_observed_hour_means_not_interpolation(self):
        source=self.source();labels=add_targets(source)
        self.assertEqual(labels.iloc[0]['target_water_level_plus_120m'],2.)
        self.assertEqual(labels.iloc[0][TARGETS[-1]],24.)
        self.assertTrue(labels.iloc[-1].isna().all())
        self.assertEqual(labels[TARGETS[-1]].isna().sum(),24)
        source.iloc[2,source.columns.get_loc('water_level_depth')]=np.nan
        for h,c in zip((1,3,6,12,24),COMMON):source[c]=source.water_level_depth.shift(-h)
        # Existing five labels do not cover this specific future-hour hole.
        labels=add_targets(source)
        with self.assertRaisesRegex(ValueError,'different complete-target'):
            assert_matched_population(labels,source.water_level_depth)

    def test_misaligned_common_targets_are_rejected(self):
        d=self.source();d.iloc[0,d.columns.get_loc(COMMON[0])]+=1
        with self.assertRaisesRegex(ValueError,'Existing target'):add_targets(d)

    def test_builder_preserves_source_features_and_purge(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source';source.mkdir()
            d=self.source()
            features=['water_level_depth']+[f'feature_{i}' for i in range(25)]
            for col in features[1:]:d[col]=1.
            d.to_csv(source/'hourly.csv')
            for split,rows in [('train',slice(0,100)),('val',slice(124,224)),('test',slice(248,None))]:
                d.iloc[rows].to_csv(source/f'{split}.csv')
            spec=dict(variant='mask',feature_columns=features,feature_set='water_level+weather_forecast',
                availability_policy='test',rain_station_ids=['a','b'],observation_cutoff_utc=str(d.index.max()))
            (source/'manifest.json').write_text(json.dumps(spec))
            original=(source/'hourly.csv').read_bytes()
            out=root/'out';build(source,out)
            self.assertEqual((source/'hourly.csv').read_bytes(),original)
            made=pd.read_csv(out/'train.csv')
            np.testing.assert_equal(made.water_level_depth.to_numpy(),np.arange(100.))
            self.assertEqual(made.filter(like='target_').shape[1],24)
            with self.assertRaises(FileExistsError):build(source,out)
            d.iloc[110:224].to_csv(source/'val.csv')
            with self.assertRaisesRegex(ValueError,'purge'):build(source,root/'bad')
            self.assertFalse((root/'bad').exists())

    def test_real_dataset_has_identical_sequences_scalers_and_24_output_reload(self):
        if not DEFAULT_OUTPUT.exists():self.skipTest('Build local O24 dataset first')
        f5,d5,s5=prepare_data(DEFAULT_OUTPUT,5,96)
        f24,d24,s24=prepare_data(DEFAULT_OUTPUT,24,96)
        self.assertEqual([len(d24[s]) for s in ['train','val','test']],[6095,985,592])
        indices=[s24['target_columns'].index(c) for c in COMMON]
        for split in ['train','val','test']:
            np.testing.assert_equal(d5[split].end_indices,d24[split].end_indices)
            np.testing.assert_equal(d5[split].X,d24[split].X)
        np.testing.assert_equal(s5['target_scaler'].mean_,s24['target_scaler'].mean_[indices])
        np.testing.assert_equal(s5['target_scaler'].scale_,s24['target_scaler'].scale_[indices])
        from research.scripts.finetune_direction_experiment import SOURCE
        if SOURCE.exists():
            import joblib
            original=joblib.load(SOURCE.with_suffix('.preprocessing.joblib'))
            self.assertEqual(s5['feature_columns'],original['feature_columns'])
            np.testing.assert_equal(s5['feature_scaler'].mean_,original['feature_scaler'].mean_)
            np.testing.assert_equal(s5['feature_scaler'].scale_,original['feature_scaler'].scale_)
            np.testing.assert_equal(s5['target_scaler'].mean_,original['target_scaler'].mean_)
            np.testing.assert_equal(s5['target_scaler'].scale_,original['target_scaler'].scale_)
        args=parser().parse_args(['--outputs','24','--device','cpu','--check-only'])
        model=make_model(s24,args,'cpu').eval()
        x,y=d24['train'][0];pred=model(x[None])
        self.assertEqual(pred.shape,(1,24))
        loss=CommonHorizonAuxiliaryLoss(TARGETS,s24['target_scaler'].scale_)
        value=loss(pred,y[None]);value.backward()
        self.assertTrue(torch.isfinite(value))
        buffer=io.BytesIO();torch.save(model.state_dict(),buffer)
        reload=load_candidate(buffer.getvalue(),s24,'cpu')
        torch.testing.assert_close(model(x[None]),reload(x[None]))
        metrics,summary,points,coverage=paired_metrics(f24['test'],reload,s24,
            make_model(s5,args,'cpu').eval(),s5)
        self.assertEqual(coverage['common_anchors'],592)
        self.assertEqual(len(coverage['common_targets']),5)
        self.assertEqual(len(coverage['candidate_only_targets']),19)
        self.assertEqual(summary['candidate']['n_predictions'],2960)

    def test_report_compares_only_common_five_and_writes_all_24(self):
        if not DEFAULT_OUTPUT.exists():self.skipTest('Build local O24 dataset first')
        from research.scripts.finetune_direction_experiment import SOURCE
        if not SOURCE.exists():self.skipTest('Local B artifact not available')
        frames,datasets,saved=prepare_data(DEFAULT_OUTPUT,24,96)
        args=parser().parse_args(['--outputs','24','--device','cpu','--check-only'])
        model=make_model(saved,args,'cpu').eval()
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            summaries,errors=write_comparisons(out,model,saved,frames,datasets,args)
            self.assertEqual(errors,{})
            self.assertEqual(summaries['B']['summary']['serving']['above_200'],48)
            self.assertEqual(summaries['production']['summary']['serving']['above_200'],97)
            self.assertEqual(len(pd.read_csv(out/'all_horizon_metrics.csv')),24)
            self.assertEqual(len(pd.read_csv(out/'all_horizon_predictions.csv.gz')),592*24)
            self.assertEqual(len(pd.read_csv(out/'comparison_B_predictions.csv.gz')),592*5)
            self.assertTrue((out/'common_horizon_predictions.png').exists())


if __name__=='__main__':unittest.main()
