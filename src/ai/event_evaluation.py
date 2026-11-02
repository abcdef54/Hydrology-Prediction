"""Offline diagnostics for saved predictions; threshold fitting belongs on validation only.

One row per station, anchor and horizon. All errors and changes are in mm.
Adjacent forecast windows are correlated and are not independent events.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve


def _ratio(a, b):
    return float(a / b) if b else float('nan')


def event_report(frame, *, rise_threshold_mm=200., stable_mm=20., moderate_mm=100.,
                 probability_thresholds=(.1, .25, .5, .75, .9),
                 regression_thresholds_mm=(100., 150., 200., 250., 300.),
                 error_thresholds=(100., 200., 500.)):
    """Return station/horizon summaries, regimes, confusion rows and PR curves.

    Required columns: station_id, horizon_hours, current_mm, actual_mm, predicted_mm.
    Optional rise_probability is a supervised-head score; final forecast uses delta>200.
    Thresholds must be chosen without examining the held-out test set.
    """
    if not (0 <= stable_mm < moderate_mm < rise_threshold_mm):
        raise ValueError('Require 0 <= stable < moderate < rise threshold')
    if any(not 0 <= t <= 1 for t in probability_thresholds):
        raise ValueError('Probability thresholds must be between 0 and 1')
    required = {'station_id','horizon_hours','current_mm','actual_mm','predicted_mm'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Missing columns: {sorted(required-set(frame.columns))}')
    data = frame.copy()
    for col in ['current_mm','actual_mm','predicted_mm']:
        if not np.isfinite(data[col]).all():
            raise ValueError(f'Nonfinite {col}')
    data['actual_delta'] = data.actual_mm-data.current_mm
    data['predicted_delta'] = data.predicted_mm-data.current_mm
    data['error'] = data.predicted_mm-data.actual_mm
    data['large_rise'] = data.actual_delta > rise_threshold_mm
    d=data.actual_delta
    data['regime'] = np.select([d < -stable_mm, d.abs() <= stable_mm,
                                (d > stable_mm)&(d <= moderate_mm),
                                (d > moderate_mm)&(d <= rise_threshold_mm), d > rise_threshold_mm],
                               ['falling','stable','small_rise','moderate_rise','large_rise'], default='stable')
    summaries,regimes,confusions,curves,calibration=[],[],[],[],[]
    def confusion(g, pred, method, threshold):
        truth=g.large_rise.to_numpy();pred=np.asarray(pred,dtype=bool)
        tp=int((truth & pred).sum());fp=int((~truth & pred).sum());fn=int((truth & ~pred).sum());tn=int((~truth & ~pred).sum())
        precision=_ratio(tp,tp+fp);recall=_ratio(tp,tp+fn)
        f1=_ratio(2*tp,2*tp+fp+fn)
        return dict(method=method,threshold=threshold,tp=tp,fp=fp,fn=fn,tn=tn,precision=precision,
                    recall=recall,f1=f1,false_discovery_rate=_ratio(fp,tp+fp),false_positive_rate=_ratio(fp,fp+tn))
    for (station,h),g in data.groupby(['station_id','horizon_hours'],sort=True):
        key=dict(station_id=station,horizon_hours=h); e=g.error
        summaries.append(dict(**key,n=len(g),mae=float(e.abs().mean()),rmse=float(np.sqrt((e**2).mean())),
            bias=float(e.mean()),abs_error_p50=float(e.abs().quantile(.5)),abs_error_p90=float(e.abs().quantile(.9)),
            abs_error_p95=float(e.abs().quantile(.95)),abs_error_p99=float(e.abs().quantile(.99)),
            actual_delta_mean=float(g.actual_delta.mean()),predicted_delta_mean=float(g.predicted_delta.mean()),
            **{f'error_gt_{v:g}mm':int((e.abs()>v).sum()) for v in error_thresholds}))
        for threshold in regression_thresholds_mm:
            confusions.append(dict(**key,**confusion(g,g.predicted_delta>threshold,'regression',threshold)))
        truth=g.large_rise.to_numpy()
        if truth.any():
            precision,recall,thresholds=precision_recall_curve(truth,g.predicted_delta.to_numpy())
            ap=float(average_precision_score(truth,g.predicted_delta.to_numpy()))
            for i in range(len(thresholds)):
                curves.append(dict(**key,method='regression',threshold=float(thresholds[i]),
                                   precision=float(precision[i]),recall=float(recall[i]),auprc=ap))
        for regime,z in g.groupby('regime',sort=True):
            regimes.append(dict(**key,regime=regime,n=len(z),mae=float(z.error.abs().mean()),
                rmse=float(np.sqrt((z.error**2).mean())),bias=float(z.error.mean()),
                actual_delta_mean=float(z.actual_delta.mean()),predicted_delta_mean=float(z.predicted_delta.mean()),
                underprediction_rate=float((z.error<0).mean()),predicted_rise_rate=float((z.predicted_delta>rise_threshold_mm).mean())))
        if 'rise_probability' in g:
            score=g.rise_probability.to_numpy(dtype=float)
            if not np.isfinite(score).all() or (score<0).any() or (score>1).any():
                raise ValueError('rise_probability must be finite and within [0,1]')
            truth=g.large_rise.to_numpy()
            auprc=float(average_precision_score(truth,score)) if truth.any() else float('nan')
            for bin_index in range(10):
                mask=(score >= bin_index/10) & (score < (bin_index+1)/10 if bin_index < 9 else score <= 1.)
                if mask.any():
                    calibration.append(dict(**key,bin=bin_index,n=int(mask.sum()),
                        mean_probability=float(score[mask].mean()),observed_rate=float(truth[mask].mean()),
                        brier=float(np.mean((score-truth)**2))))
            for threshold in probability_thresholds:
                confusions.append(dict(**key,**confusion(g,score>=threshold,'classification',threshold),auprc=auprc))
            if truth.any():
                precision,recall,thresholds=precision_recall_curve(truth,score)
                for i in range(len(thresholds)):
                    curves.append(dict(**key,method='classification',threshold=float(thresholds[i]),precision=float(precision[i]),recall=float(recall[i]),auprc=auprc))
    return dict(summary=pd.DataFrame(summaries),regimes=pd.DataFrame(regimes),
                confusion=pd.DataFrame(confusions),precision_recall=pd.DataFrame(curves),calibration=pd.DataFrame(calibration))


def match_budget(confusion, *, method='classification', max_false_positive_rate=None, min_precision=None):
    """Select per-station/horizon validation thresholds; apply those thresholds elsewhere explicitly."""
    if (max_false_positive_rate is None) == (min_precision is None):
        raise ValueError('Specify exactly one budget')
    if method not in ('classification','regression'):
        raise ValueError('Unknown prediction method')
    rows=confusion[confusion.method.eq(method)].copy()
    if max_false_positive_rate is not None:
        rows=rows[rows.false_positive_rate <= max_false_positive_rate]
    else:
        rows=rows[rows.precision >= min_precision]
    return rows.sort_values(['station_id','horizon_hours','recall','threshold'],ascending=[True,True,False,False]).drop_duplicates(['station_id','horizon_hours'])


def plot_event_comparison(frame, station_id, horizon_hours):
    """Plot aligned held-out predictions; baseline_predicted_mm is optional."""
    import matplotlib.pyplot as plt
    z=frame[(frame.station_id==station_id)&(frame.horizon_hours==horizon_hours)].sort_values('anchor_utc')
    if z.empty: raise ValueError('No matching station/horizon rows')
    fig,axes=plt.subplots(4,1,sharex=True,figsize=(14,10),constrained_layout=True)
    x=pd.to_datetime(z.anchor_utc)
    axes[0].plot(x,z.actual_mm,label='Observed');axes[0].plot(x,z.predicted_mm,label='Event-aware')
    if 'baseline_predicted_mm' in z: axes[0].plot(x,z.baseline_predicted_mm,label='Baseline')
    axes[0].legend();axes[0].set_ylabel('Level (mm)')
    if 'rise_probability' in z: axes[1].plot(x,z.rise_probability)
    axes[1].set_ylabel('Rise probability')
    if 'correction_mm' in z: axes[2].plot(x,z.correction_mm,label='Applied correction')
    if 'amplitude_mm' in z: axes[2].plot(x,z.amplitude_mm,label='Amplitude',alpha=.6)
    if 'correction_mm' in z or 'amplitude_mm' in z: axes[2].legend()
    axes[2].set_ylabel('Correction (mm)')
    axes[3].plot(x,z.predicted_mm-z.actual_mm);axes[3].axhline(0,color='black',linewidth=.5);axes[3].set_ylabel('Error (mm)')
    return fig


def main():
    import argparse
    from pathlib import Path
    parser=argparse.ArgumentParser(description='Evaluate saved event-aware predictions; never fits models')
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,help='Optional saved predictions aligned by station/anchor/horizon')
    parser.add_argument('--rise-threshold-mm',type=float,default=200.)
    parser.add_argument('--stable-mm',type=float,default=20.)
    parser.add_argument('--moderate-mm',type=float,default=100.)
    parser.add_argument('--probability-thresholds',type=float,nargs='+',default=[.1,.25,.5,.75,.9])
    parser.add_argument('--regression-thresholds-mm',type=float,nargs='+',default=[100.,150.,200.,250.,300.])
    args=parser.parse_args()
    frame=pd.read_parquet(args.predictions)
    if args.baseline:
        b=pd.read_parquet(args.baseline);keys=['station_id','anchor_utc','horizon_hours']
        if b.duplicated(keys).any() or frame.duplicated(keys).any():raise ValueError('Duplicate prediction keys')
        b=b[keys+['predicted_mm','actual_mm']].rename(columns={'predicted_mm':'baseline_predicted_mm','actual_mm':'baseline_actual_mm'})
        frame=frame.merge(b,on=keys,how='left',validate='one_to_one')
        if frame.baseline_predicted_mm.isna().any() or not np.allclose(frame.actual_mm,frame.baseline_actual_mm,atol=.01,rtol=0):
            raise ValueError('Baseline anchors/labels do not match')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    report=event_report(frame,rise_threshold_mm=args.rise_threshold_mm,stable_mm=args.stable_mm,
                        moderate_mm=args.moderate_mm,probability_thresholds=args.probability_thresholds,
                        regression_thresholds_mm=args.regression_thresholds_mm)
    for key,table in report.items():table.to_csv(args.output_dir/f'{key}.csv',index=False)
    if args.baseline:
        baseline=frame.copy();baseline['predicted_mm']=baseline.baseline_predicted_mm
        for key,table in event_report(baseline.drop(columns=['rise_probability'],errors='ignore'),rise_threshold_mm=args.rise_threshold_mm,
                                      stable_mm=args.stable_mm,moderate_mm=args.moderate_mm).items():
            table.to_csv(args.output_dir/f'baseline_{key}.csv',index=False)
    print(f'Wrote metrics to {args.output_dir}')

if __name__=='__main__':main()
