"""Direction/amplitude incentives and physical-unit extreme-error gradients."""
import unittest
import torch
from src.ai.losses import DirectionAmplitudeLoss, HorizonWeightedTailLoss


class DirectionLossTests(unittest.TestCase):
    def parts(self, pred, true=300.):
        loss = DirectionAmplitudeLoss([1], [200.])
        p = torch.tensor([[pred/200]], dtype=torch.float64, requires_grad=True)
        y = torch.tensor([[true/200]], dtype=torch.float64)
        anchor = torch.zeros_like(y)
        return loss, p, y, anchor, loss.components(p, y, anchor)

    def test_correct_direction_is_insufficient_without_amplitude(self):
        good = self.parts(300)[-1]
        weak = self.parts(50)[-1]
        strong = self.parts(550)[-1]
        wrong = self.parts(-50)[-1]
        for c in [weak, strong, wrong]:
            self.assertGreater(c['direction'].item(), good['direction'].item())
            self.assertGreater(c['amplitude'].item(), good['amplitude'].item())
        self.assertAlmostEqual(weak['amplitude'].item(), strong['amplitude'].item())

    def test_gradient_corrects_under_and_overshoot_in_both_directions(self):
        for true in [300., -300.]:
            for delta in [-250.,250.]:
                loss,p,y,a,_ = self.parts(true+delta,true)
                grad=torch.autograd.grad(loss(p,y,a),p)[0].item()
                self.assertGreater(grad*delta,0)

    def test_neutral_zone_and_bounded_direction_reward(self):
        for pred in [-1000.,0.,1000.]:
            c=self.parts(pred,true=5.)[-1]
            self.assertEqual(c['direction'].item(),0)
        for pred in [-1000.,0.,300.,1000.]:
            v=self.parts(pred)[-1]['direction'].item()
            self.assertTrue(0<=v<=1)

    def test_physical_threshold_and_superquadratic_excess(self):
        near=self.parts(500)[-1]['excess'].item()  # error exactly 200
        self.assertEqual(near,0)
        vals=[self.parts(300+e)[-1]['excess'].item() for e in [300,400,600]]
        self.assertGreater(vals[1],4*vals[0])
        self.assertGreater(vals[2],4*vals[1])
        loss=DirectionAmplitudeLoss([1,1],[100.,400.])
        p=torch.tensor([[3.,.75]],requires_grad=True)
        c=loss.components(p,torch.zeros_like(p),torch.zeros_like(p))
        self.assertAlmostEqual(c['excess'].item(),1.25)

    def test_disabled_new_terms_exactly_reproduce_B_and_invalid_parameters(self):
        loss=DirectionAmplitudeLoss([1],[200.],direction_strength=0,
                                    amplitude_strength=0,excess_strength=0)
        p=torch.tensor([[2.]],requires_grad=True);y=torch.zeros_like(p)
        torch.testing.assert_close(loss(p,y,y),loss.base(p,y),rtol=0,atol=0)
        for kw in [dict(excess_strength=-1),dict(neutral_mm=0),dict(quartic=float('nan'))]:
            with self.assertRaises(ValueError):DirectionAmplitudeLoss([1],[200.],**kw)
        with self.assertRaises(ValueError):loss(p,y,torch.zeros(2,1))

    def test_finite_large_error_and_saved_loss_roundtrip(self):
        import io
        loss,p,y,a,_=self.parts(100000.)
        value=loss(p,y,a)
        self.assertTrue(torch.isfinite(value))
        self.assertTrue(torch.isfinite(torch.autograd.grad(value,p)[0]).all())
        buffer=io.BytesIO();torch.save(loss.state_dict(),buffer);buffer.seek(0)
        other=DirectionAmplitudeLoss([1],[200.]);other.load_state_dict(torch.load(buffer,weights_only=True))
        torch.testing.assert_close(value,other(p,y,a))

    def test_checkpoint_guard_keeps_B_if_either_threshold_worsens(self):
        from research.scripts.finetune_direction_experiment import choose_checkpoint
        baseline=dict(above_200=90,above_400=10,rmse_mm=70.)
        self.assertFalse(choose_checkpoint(dict(above_200=80,above_400=11,rmse_mm=60.),baseline,baseline))
        self.assertFalse(choose_checkpoint(dict(above_200=91,above_400=0,rmse_mm=60.),baseline,baseline))
        self.assertTrue(choose_checkpoint(dict(above_200=89,above_400=10,rmse_mm=70.),baseline,baseline))
        self.assertTrue(choose_checkpoint(dict(above_200=90,above_400=9,rmse_mm=71.),baseline,baseline))

    def test_real_checkpoint_scalers_and_physical_delta(self):
        from pathlib import Path
        import joblib
        import pandas as pd
        from src.ai.comparison import evaluation_dataset
        from research.scripts.finetune_direction_experiment import SOURCE,DATA,make_model
        if not SOURCE.exists():self.skipTest('Local B checkpoint not available')
        saved=joblib.load(SOURCE.with_suffix('.preprocessing.joblib'))
        ds=evaluation_dataset(pd.read_csv(DATA/'train.csv'),saved)
        model=make_model(saved,'cpu').eval()
        x,y=ds[0];x,y=x[None],y[None]
        persistence=model.persistence(x)
        # The physical anchor level must agree across all five target scales.
        mm=persistence.detach().numpy()*saved['target_scaler'].scale_+saved['target_scaler'].mean_
        self.assertLess(float(mm.max()-mm.min()),.001)
        loss=DirectionAmplitudeLoss([1]*5,saved['target_scaler'].scale_)
        value=loss(model(x),y,persistence)
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_scratch_initialization_does_not_load_B_weights(self):
        import joblib
        from research.scripts.finetune_direction_experiment import SOURCE,make_model
        if not SOURCE.exists():self.skipTest('Local B checkpoint not available')
        saved=joblib.load(SOURCE.with_suffix('.preprocessing.joblib'))
        random_model=make_model(saved,'cpu',initialize_from_B=False).eval()
        trained=make_model(saved,'cpu',initialize_from_B=True).eval()
        self.assertTrue(random_model.linear.weight.eq(0).all())
        self.assertFalse(trained.linear.weight.eq(0).all())
        x=torch.zeros(2,96,len(saved['feature_columns']))
        torch.testing.assert_close(random_model(x),random_model.persistence(x),rtol=0,atol=0)

    def test_BL02_arms_only_change_tail_strength_and_reject_bad_configuration(self):
        from research.scripts.finetune_direction_experiment import loss_arms
        self.assertEqual(loss_arms([.5,.75,1.]),[('tail_0.5',.5),('tail_0.75',.75),('tail_1',1.)])
        for strengths in [[],[-1],[float('nan')],[.5,.5]]:
            with self.assertRaises(ValueError):loss_arms(strengths)
        p=torch.tensor([[2.]],requires_grad=True);y=torch.zeros_like(p)
        a=HorizonWeightedTailLoss([1],[200],strength=.5)
        b=HorizonWeightedTailLoss([1],[200],strength=1.)
        mse,penalty=a.components(p,y)
        torch.testing.assert_close(b(p,y)-a(p,y),.5*penalty)


if __name__=='__main__':unittest.main()
