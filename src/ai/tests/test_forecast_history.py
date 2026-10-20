"""Per-window weather ablation, observation parity and policy-aware reload."""
import io
import unittest
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from src.ai.lstm import ResidualLSTM
from src.ai.comparison import load_candidate


class ForecastHistoryTests(unittest.TestCase):
    def model(self, policy='anchor_only'):
        return ResidualLSTM(input_size=5,hidden_size=3,num_layers=1,output_size=1,
            level_feature_index=0,persistence_scale=[1.],persistence_offset=[0.],
            forecast_history_policy=policy,weather_feature_indices=[1,3,4],
            weather_mask_indices=[3],weather_mask_values=[1.])

    def test_only_past_weather_is_withheld_and_input_is_not_mutated(self):
        x=torch.arange(2*4*5,dtype=torch.float32).reshape(2,4,5)
        original=x.clone();m=self.model();y=m.prepare_inputs(x)
        torch.testing.assert_close(x,original)
        torch.testing.assert_close(y[:,-1],x[:,-1])
        torch.testing.assert_close(y[:,:,[0,2]],x[:,:,[0,2]])
        self.assertTrue(y[:,:-1,1].eq(0).all())
        self.assertTrue(y[:,:-1,4].eq(0).all())
        self.assertTrue(y[:,:-1,3].eq(1).all())
        torch.testing.assert_close(m.persistence(y),m.persistence(x))
        torch.testing.assert_close(self.model('all_timesteps').prepare_inputs(x),x)

    def test_mask_depends_on_position_in_each_window_not_global_timestamp(self):
        m=self.model();a=torch.randn(5,5)
        first=m.prepare_inputs(a[:4]);second=m.prepare_inputs(a[1:])
        torch.testing.assert_close(first[-1],a[3])  # Same row is current here.
        self.assertEqual(second[-2,1].item(),0.)   # And withheld in the next anchor.
        self.assertEqual(second[-2,3].item(),1.)
        torch.testing.assert_close(m.prepare_inputs(a[-1:]),a[-1:])

    def test_gradient_cannot_use_withheld_weather(self):
        x=torch.randn(2,4,5,requires_grad=True)
        self.model().prepare_inputs(x).sum().backward()
        self.assertTrue(x.grad[:,:-1,[1,3,4]].eq(0).all())
        self.assertTrue(x.grad[:,-1].eq(1).all())
        self.assertTrue(x.grad[:,:,[0,2]].eq(1).all())

    def test_reload_restores_policy_and_actual_scaled_mask_value(self):
        fs=StandardScaler().fit(np.array([[0,2,0,0,5],[1,4,1,0,7]],dtype=float))
        ts=StandardScaler().fit(np.array([[0.],[1.]]))
        saved=dict(feature_columns=['water_level_depth','weather_forecast_rain_0_3h',
                  'rain_750007_depth','weather_forecast_rain_0_3h_missing','weather_forecast_age_hours'],
                  target_columns=['target_water_level_plus_60m'],feature_scaler=fs,
                  target_scaler=ts,sequence_length=4,forecast_history_policy='anchor_only')
        model=ResidualLSTM(input_size=5,hidden_size=3,num_layers=1,output_size=1,
            level_feature_index=0,persistence_scale=[fs.scale_[0]/ts.scale_[0]],
            persistence_offset=[(fs.mean_[0]-ts.mean_[0])/ts.scale_[0]],
            forecast_history_policy='anchor_only',weather_feature_indices=[1,3,4],
            weather_mask_indices=[3],weather_mask_values=[(1-fs.mean_[3])/fs.scale_[3]]).eval()
        torch.nn.init.normal_(model.linear.weight)  # Exercise learned, non-persistence output.
        buffer=io.BytesIO();torch.save(model.state_dict(),buffer)
        reloaded=load_candidate(buffer.getvalue(),saved,'cpu')
        x=torch.randn(2,4,5)
        torch.testing.assert_close(model(x),reloaded(x))
        torch.testing.assert_close(model.prepare_inputs(x),reloaded.prepare_inputs(x))
        self.assertFalse(any('weather_' in key for key in model.state_dict()))
        self.assertEqual(reloaded.forecast_history_policy,'anchor_only')
        saved['forecast_history_policy']='bogus'
        with self.assertRaises(ValueError):load_candidate(buffer.getvalue(),saved,'cpu')

    def test_invalid_mapping_is_rejected(self):
        for options in [dict(forecast_history_policy='bad'),
                        dict(forecast_history_policy='anchor_only'),
                        dict(weather_feature_indices=[0]),dict(weather_feature_indices=[99]),
                        dict(weather_feature_indices=[1],weather_mask_indices=[3],weather_mask_values=[1.])]:
            with self.assertRaises(ValueError):
                ResidualLSTM(input_size=5,hidden_size=3,num_layers=1,output_size=1,
                    level_feature_index=0,persistence_scale=[1.],persistence_offset=[0.],**options)


if __name__=='__main__':unittest.main()
