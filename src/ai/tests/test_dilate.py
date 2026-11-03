import math
import unittest
import torch
from src.ai.dilate import dilate_components, StationBalancedDILATE

class DILATETests(unittest.TestCase):
    def test_two_point_exact_paths(self):
        x=torch.tensor([[0.,1.]],dtype=torch.double);y=torch.tensor([[.2,.7]],dtype=torch.double);gamma=.3
        shape,time=dilate_components(x,y,gamma)
        d=(x[:,:,None]-y[:,None,:]).square()[0];paths=torch.stack([d[0,0]+d[1,1],d[0,0]+d[0,1]+d[1,1],d[0,0]+d[1,0]+d[1,1]])
        expected=-gamma*torch.logsumexp(-paths/gamma,0); probs=torch.softmax(-paths/gamma,0)
        torch.testing.assert_close(shape[0],expected);torch.testing.assert_close(time[0],(probs[1]+probs[2])/4)
    def test_gradient_and_temporal_gradient(self):
        x=torch.tensor([[.1,.8,.3]],dtype=torch.double,requires_grad=True);y=torch.tensor([[0.,.2,.9]],dtype=torch.double)
        self.assertTrue(torch.autograd.gradcheck(lambda a: dilate_components(a,y,.2),(x,),eps=1e-6,atol=1e-4))
        t=dilate_components(x,y,.2)[1].sum();self.assertGreater(torch.autograd.grad(t,x)[0].abs().sum().item(),0)
    def test_edge_and_balancing(self):
        x=torch.tensor([[1.],[2.]],requires_grad=True);shape,t=dilate_components(x,torch.zeros_like(x),.01);torch.testing.assert_close(t,torch.zeros_like(t));torch.testing.assert_close(shape,x[:,0]**2)
        loss=StationBalancedDILATE([1,3],[[200.],[200.]],alpha=1.);v=loss(x,torch.zeros_like(x),torch.zeros_like(x),torch.tensor([0,1]));self.assertTrue(torch.isfinite(v));v.backward();self.assertTrue(torch.isfinite(x.grad).all())
        for gamma in [0,-1,float('nan')]:
            with self.assertRaises(ValueError):dilate_components(x,x,gamma)
        with self.assertRaises(ValueError):StationBalancedDILATE([1],[[200.]],alpha=2)
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda(self):
        x=torch.randn(2,24,device='cuda',requires_grad=True);y=torch.randn_like(x);s,t=dilate_components(x,y,.01);(s+t).mean().backward();self.assertTrue(torch.isfinite(x.grad).all())

    def test_24_horizons_finite_backward(self):
        torch.manual_seed(2)
        x=(torch.randn(2,24)*.1).requires_grad_();y=torch.randn(2,24)*.1
        loss=StationBalancedDILATE([2,2],[[150.]*24,[300.]*24],alpha=.5,gamma=.01)
        value=loss(x,y,torch.zeros_like(x),torch.tensor([0,1]));value.backward()
        self.assertTrue(torch.isfinite(value));self.assertTrue(torch.isfinite(x.grad).all());self.assertGreater(x.grad.abs().sum().item(),0)
