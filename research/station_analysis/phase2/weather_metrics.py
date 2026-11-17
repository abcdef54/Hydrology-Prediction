"""Timestamp-exact precipitation verification and missing-safe scalar metrics."""
import numpy as np
import pandas as pd


def future_amount(series, a, b):
    """Sum interval-ending observations at t+a+1 .. t+b; require every hour."""
    if not 0 <= a < b:
        raise ValueError('Expected 0 <= a < b')
    if series.index.tz is None or not series.index.equals(pd.date_range(series.index.min(),series.index.max(),freq='h')):
        raise ValueError('Ground truth requires a continuous timezone-aware hourly index')
    return pd.concat([series.shift(-h) for h in range(a+1,b+1)],axis=1).sum(axis=1,min_count=b-a)


def rain_metrics(predicted, observed, *, wet_mm=.1, heavy_mm=15.):
    data=pd.concat([predicted.rename('p'),observed.rename('o')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
    n=len(data)
    result={'n':n}
    if not n:
        return {**result,**{k:np.nan for k in ['mae_mm','rmse_mm','bias_mm','zero_forecast_rmse_mm','mse_skill_vs_zero']}}
    e=data.p-data.o
    base=float(np.mean(data.o**2))
    result.update(mae_mm=float(e.abs().mean()),rmse_mm=float(np.sqrt(np.mean(e**2))),
                  bias_mm=float(e.mean()),zero_forecast_rmse_mm=float(np.sqrt(base)),
                  mse_skill_vs_zero=1-float(np.mean(e**2))/base if base>0 else np.nan)
    for label,threshold in [('wet',wet_mm),('heavy',heavy_mm)]:
        p=data.p>=threshold;o=data.o>=threshold
        tp=int((p&o).sum());fp=int((p&~o).sum());fn=int((~p&o).sum());tn=int((~p&~o).sum())
        result.update({f'{label}_threshold_mm':threshold,f'{label}_tp':tp,f'{label}_fp':fp,
                       f'{label}_fn':fn,f'{label}_tn':tn,
                       f'{label}_precision':tp/(tp+fp) if tp+fp else np.nan,
                       f'{label}_recall':tp/(tp+fn) if tp+fn else np.nan,
                       f'{label}_f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan})
    return result


def correlations(x,y):
    data=pd.concat([x.rename('x'),y.rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
    n=len(data)
    if n<20 or data.x.std()==0 or data.y.std()==0:
        return n,np.nan,np.nan
    return n,float(data.x.corr(data.y)),float(data.x.corr(data.y,method='spearman'))


def partial_correlations(feature, target, controls, train_mask, evaluation_mask):
    """Fit nuisance projections on train only; evaluate residual correlations on each split."""
    frame=controls.copy()
    frame['feature']=feature;frame['target']=target
    valid=np.isfinite(frame.to_numpy()).all(axis=1)
    fit=train_mask&valid
    if int(fit.sum())<100:
        return []
    x=controls.loc[fit].to_numpy(dtype=float)
    mean=x.mean(axis=0);scale=x.std(axis=0);scale[scale==0]=1
    design=np.column_stack([np.ones(len(x)),(x-mean)/scale])
    coefficients=np.linalg.lstsq(design,frame.loc[fit,['feature','target']].to_numpy(),rcond=None)[0]
    all_x=np.column_stack([np.ones(len(controls)),(controls.to_numpy(dtype=float)-mean)/scale])
    residual=frame[['feature','target']].to_numpy()-all_x@coefficients
    residual=pd.DataFrame(residual,index=controls.index,columns=['feature','target'])
    rows=[]
    for split,mask in [('train',train_mask),('evaluation',evaluation_mask)]:
        n,r,s=correlations(residual.feature.loc[mask&valid],residual.target.loc[mask&valid])
        rows.append({'scope':split,'train_fit_n':int(fit.sum()),'n':n,'partial_pearson_r':r,'partial_spearman_r':s})
    return rows
