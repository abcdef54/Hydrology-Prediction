"""Differentiable DILATE with station-balanced, dimensionless water-level changes."""
import math
import torch
from .change_objective import ChangeRiskLoss


def dilate_components(predicted, target, gamma):
    """Return per-series Soft-DTW shape and soft-alignment temporal distortion.

    Inputs are dimensionless [batch,time]. Omega=(i-j)^2/time^2, matching
    the original implementation. Soft-DTW shape is not a divergence and may be negative.
    """
    if predicted.ndim != 2 or predicted.shape != target.shape or not predicted.numel():
        raise ValueError('Expected nonempty equal [batch,time] tensors')
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError('gamma must be finite and positive')
    n = predicted.shape[1]
    cost = (predicted[:,:,None]-target[:,None,:]).square()
    inf = torch.full_like(cost[:,0,0], float('inf'))
    zero = torch.zeros_like(inf)
    grid = [[inf for _ in range(n+1)] for _ in range(n+1)]
    grid[0][0] = zero
    transitions = {}
    for i in range(1,n+1):
        for j in range(1,n+1):
            prev = torch.stack((grid[i-1][j],grid[i][j-1],grid[i-1][j-1]),dim=-1)
            logits = -prev/gamma
            transitions[i,j] = torch.softmax(logits,dim=-1)
            grid[i][j] = cost[:,i-1,j-1]-gamma*torch.logsumexp(logits,dim=-1)
    # Expected path occupancy, propagated backwards through the soft minimum.
    visits = [[zero for _ in range(n+1)] for _ in range(n+1)]
    visits[n][n] = torch.ones_like(zero)
    temporal = zero
    for i in range(n,0,-1):
        for j in range(n,0,-1):
            q = visits[i][j]; temporal = temporal + q*((i-j)**2/(n*n))
            probs = transitions[i,j]
            if i>1: visits[i-1][j] = visits[i-1][j]+q*probs[:,0]
            if j>1: visits[i][j-1] = visits[i][j-1]+q*probs[:,1]
            if i>1 and j>1: visits[i-1][j-1] = visits[i-1][j-1]+q*probs[:,2]
    return grid[n][n], temporal


class StationBalancedDILATE(ChangeRiskLoss):
    def __init__(self, counts, scales, alpha=.5, gamma=.01, scale_floor_mm=200.):
        if not math.isfinite(alpha) or not 0 <= alpha <= 1: raise ValueError('alpha must be in [0,1]')
        if not math.isfinite(gamma) or gamma <= 0: raise ValueError('gamma must be finite and positive')
        super().__init__([1.]*len(scales[0]),counts,scales,'station-balanced',scale_floor_mm,0.,0.)
        self.alpha,self.gamma=alpha,gamma

    def forward(self,predictions,targets,persistence,station_ids,**kwargs):
        if predictions.ndim!=2 or predictions.shape!=targets.shape or predictions.shape!=persistence.shape or predictions.shape[1]!=len(self.weights):
            raise ValueError('Expected aligned [batch,horizon] tensors')
        if station_ids.shape!=(len(predictions),) or station_ids.dtype!=torch.long: raise ValueError('Expected station IDs')
        scale=self.scales[station_ids].to(predictions);denom=scale.clamp_min(self.scale_floor_mm)
        p=(predictions-persistence)*scale/denom; y=(targets-persistence)*scale/denom
        shape,temporal=dilate_components(p,y,self.gamma)
        return ((self.alpha*shape+(1-self.alpha)*temporal)*self.frequency[station_ids].to(predictions)).mean()

    def specification(self):
        spec=super().specification();spec.update(mode='dilate',alpha=self.alpha,gamma=self.gamma,normalization='station/horizon physical delta divided by max(target scale, floor)',temporal_omega='(i-j)^2 / horizon_count^2');return spec
