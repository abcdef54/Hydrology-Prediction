r"""Evaluate saved water-level forecasts; never train or run model inference.

Events are derived once per station from observations, not per forecast horizon.
The default research definition is provisional and is not a flood-alert level.

# Offline evaluation of regional water-level predictions

This evaluator reads **existing prediction and observation artifacts**. It does not train, fit a threshold, invoke inference, or alter model/data artifacts. It provides regression, large-rise detection and observation-defined event analysis. The rise event is not a flood warning and no official water-level warning threshold is inferred.

## Usage

```bash
.venv/bin/python -m src.ai.hydrology_evaluation \
  --predictions path/to/validation_predictions.parquet \
  --observations train_data/hue_regional_20261015_h6_h12_fixed_split_v1/water.parquet \
  --output-dir reports/model_training/my_evaluation \
  --event-window-hours 12 \
  --rise-threshold-mm 200 \
  --event-merge-gap-hours 6 \
  --warning-delta-mm 200 \
  --warning-merge-gap-hours 1 \
  --tolerance-mm 20
```

Use the same predeclared arguments and observation coverage when comparing models. Prediction parquet must contain `station_id`, `anchor_utc`, `target_utc`, `horizon_hours`, `current_mm`, `actual_mm`, `predicted_mm`, one unique row per station/anchor/horizon. Horizons must have contiguous hourly target coverage. `target_utc` must equal issue time plus horizon. Observations are hourly wide parquet (`timestamp_utc` and one column per station) or long parquet (`timestamp_utc`, `station_id`, `water_level_mm`). UTC timestamps are normalized by pandas. Predictions, labels, current levels and observations must match within 0.01 mm, including every anchor/target. Duplicate keys, missing observations at anchors/targets, nonfinite predictions, inconsistent timestamps and gaps in forecast coverage raise errors. `water.parquet` contains dataset hourly values, not verified raw sensor records.

## Window-level evaluation

Regression: pooled, per horizon, per station/horizon MAE, RMSE, bias, absolute error p90 and count >200 mm; regime tables contain observed/predicted changes and ratio for observed rises >200. Labels use **actual future change >200 mm**, not absolute change. Detection is from predicted delta, with strict precision, recall, F1 and sklearn average precision (AP). `precision_recall.csv` gives station/horizon PR points; AP is undefined when there are no positives. FP categories are based on observed delta relative to the configured threshold: within 20 mm below the threshold; 20–100 mm below; more than 100 mm below. Conditional mean/median forecast error are also exported.

Gray-zone sensitivity ignores true changes in the closed interval [threshold−tolerance, threshold+tolerance], then computes precision and recall on the remaining samples, with predicted-delta warning threshold unchanged. This changes the evaluated population and is not a calibrated operating threshold. Use an independent calibration period and held-out evaluation; repeated validation inspection is exploratory.

## Observation-defined events

The user-selected research rule flags observation time `t` when `H(t)−H(t−12h)>200 mm`, with all 13 hourly observation values finite. The window and threshold are configurable. Flagged times for a station are merged when there are no more than six consecutive unflagged hours between them (configurable). An unknown/missing observation or invalid rolling window always breaks a cluster. Each station receives one shared event list; there is no duplicate event merely because a forecast has two horizons. Events are not cross-station merged. Results include flagged start/end, maximum observed level and time in the event, and largest rolling rise.

Events are constructed only within each station's **common target-time range across horizons**, with preceding observations used to compute the first rolling value. A flag close to the coverage edge or to missing observations is marked censored. The rule is conservative and may not identify all uncertain boundaries. A 12h rolling rise >200 mm is a research event, not automatically a flood, and predicted +6h delta >200 mm and a 12h observed event are different quantities.

## Warnings and matching

For each station, forecast rows with predicted delta above `--warning-delta-mm` form warning episodes; issue times up to one hour apart are grouped by default, across horizons in `combined` mode. A warning episode is eligible for an event only when at least one of its forecast **target times** lies within the observed event interval. Eligible episode/event pairs are greedily matched one-to-one by earliest eligible issue time, then event start. This is deterministic but not necessarily maximum-cardinality and may differ from another operational matching policy. One warning episode cannot claim multiple events; one event cannot claim multiple episodes.

Warning statuses: `matched`, `unmatched_competing` (eligible for an event but not selected due to one-to-one matching), and `unmatched_no_overlap`. The last category is **not automatically a false alarm**: an event can be outside the observed/evaluated period, hidden by missing observations, not captured by the 12-hour rule, or a warning may have a real effect outside the permitted interval. Inspect censoring and coverage before using it operationally. Matching is recalculated separately for +6h, +12h and their combined warning stream; all modes use the *same observed event list and IDs*.

For a match, `lead_to_event_start_hours = event_start − issue_time`; negative values are late. `lead_to_peak_hours` uses the observed maximum level within the event. The selected issue is the earliest eligible forecast in the matched warning episode, which can differ from the episode's first issue. For peak error, use a forecast with target exactly equal to observed event peak time, choosing the latest available issue within the evaluated horizon(s); absence is reported missing. This is distinct from the peak of a predicted hydrograph or the peak change between anchor and target.

## Outputs and limitations

- `regression_detection.csv`, `regimes.csv`, `precision_recall.csv`.
- `events_{combined,h6,h12}.csv`, `warnings_*.csv`, `matches_*.csv`, `event_summary_*.csv`. Station IDs with leading zeros should be loaded as strings (`dtype={'station_id':str}`).
- `config.json` records policy, coverage, artifact paths and SHA-256 hashes. Store it with the outputs.

Censored events are marked, with both all-event and uncensored counts. Missing observations prevent rolling-event construction; no reliable denominator exists for intervals without data. The evaluator does not supply flood-level thresholds, independent hydrological flood delineation, spatial event consolidation, calibrated thresholds or prospective operational performance. Validate policy and sensitivity, then freeze it before comparing on untouched future events. An observation series with retrospectively reconstructed or questionable readings needs independent quality review.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve


@dataclass(frozen=True)
class EvaluationConfig:
    rise_threshold_mm: float = 200.0
    event_window_hours: int = 12
    event_merge_gap_hours: int = 6
    warning_merge_gap_hours: int = 1
    tolerance_mm: float = 20.0
    warning_delta_mm: float = 200.0
    def __post_init__(self):
        if (not np.isfinite([self.rise_threshold_mm, self.tolerance_mm, self.warning_delta_mm]).all()
                or self.rise_threshold_mm <= 0 or self.tolerance_mm < 0
                or self.event_window_hours < 1 or self.event_merge_gap_hours < 0
                or self.warning_merge_gap_hours < 0):
            raise ValueError('Invalid thresholds or event timing parameters')


def _check_predictions(df):
    required = {'station_id', 'anchor_utc', 'target_utc', 'horizon_hours', 'actual_mm', 'current_mm', 'predicted_mm'}
    missing = required - set(df.columns)
    if missing: raise ValueError(f'Missing prediction columns: {sorted(missing)}')
    x = df.copy(); x['station_id'] = x.station_id.astype(str)
    for col in ('anchor_utc', 'target_utc'): x[col] = pd.to_datetime(x[col], utc=True, errors='raise')
    cols = ['actual_mm', 'current_mm', 'predicted_mm', 'horizon_hours']
    if x.empty or not np.isfinite(x[cols].to_numpy(dtype=float)).all(): raise ValueError('Empty or nonfinite predictions')
    if x[['anchor_utc','target_utc']].isna().any().any(): raise ValueError('Missing prediction timestamps')
    if x.duplicated(['station_id', 'anchor_utc', 'horizon_hours']).any(): raise ValueError('Duplicate prediction keys')
    if not (x.horizon_hours == np.floor(x.horizon_hours)).all(): raise ValueError('Horizons must be whole hours')
    if not (x.horizon_hours > 0).all() or not ((x.target_utc - x.anchor_utc) == pd.to_timedelta(x.horizon_hours, unit='h')).all():
        raise ValueError('Target times must equal issue time plus positive horizon')
    x['actual_delta'] = x.actual_mm - x.current_mm
    x['predicted_delta'] = x.predicted_mm - x.current_mm
    x['error_mm'] = x.predicted_mm - x.actual_mm
    return x


def _summary(group, cfg):
    y = group.actual_delta > cfg.rise_threshold_mm; alarm = group.predicted_delta > cfg.warning_delta_mm
    e = group.error_mm; tp = int((y & alarm).sum()); fp = int((~y & alarm).sum()); fn = int((y & ~alarm).sum())
    actual = group.actual_delta; f = group[~y & alarm]; tol = cfg.tolerance_mm
    safe = lambda n,d: n/d if d else np.nan
    eligible = (actual < cfg.rise_threshold_mm-tol) | (actual > cfg.rise_threshold_mm+tol) if tol else actual.notna()
    clearpos = actual > cfg.rise_threshold_mm+tol
    clearneg = actual < cfg.rise_threshold_mm-tol if tol else actual <= cfg.rise_threshold_mm
    gtp = int((eligible & clearpos & alarm).sum()); gfp = int((eligible & clearneg & alarm).sum()); gfn = int((eligible & clearpos & ~alarm).sum())
    return dict(n=len(group), mae=e.abs().mean(), rmse=np.sqrt((e**2).mean()), bias=e.mean(),
        abs_error_p90=e.abs().quantile(.9), errors_gt_200=int((e.abs()>200).sum()),
        tp=tp, fp=fp, fn=fn, precision=safe(tp,tp+fp), recall=safe(tp,tp+fn), f1=safe(2*tp,2*tp+fp+fn),
        average_precision=float(average_precision_score(y, group.predicted_delta)) if y.any() else np.nan,
        fp_actual_near=int(((f.actual_delta>=cfg.rise_threshold_mm-20)&(f.actual_delta<=cfg.rise_threshold_mm)).sum()),
        fp_actual_moderate=int(((f.actual_delta>=cfg.rise_threshold_mm-100)&(f.actual_delta<cfg.rise_threshold_mm-20)).sum()),
        fp_actual_low=int((f.actual_delta<cfg.rise_threshold_mm-100).sum()), fp_error_mean=f.error_mm.mean(), fp_error_median=f.error_mm.median(),
        fp_error_p90=f.error_mm.quantile(.9), fp_error_gt_200=int((f.error_mm>200).sum()),
        fp_threshold_shortfall_mean=(cfg.rise_threshold_mm-f.actual_delta).clip(lower=0).mean(),
        gray_ignored=int((~eligible).sum()), gray_tp=gtp, gray_fp=gfp, gray_fn=gfn,
        gray_precision=safe(gtp,gtp+gfp), gray_recall=safe(gtp,gtp+gfn))


def regression_detection(predictions, cfg):
    x = _check_predictions(predictions); rows=[]; regimes=[]
    groups=[('pooled','all',0,x)]
    groups += [('horizon','all',int(h),g) for h,g in x.groupby('horizon_hours')]
    groups += [('station_horizon',s,int(h),g) for (s,h),g in x.groupby(['station_id','horizon_hours'])]
    for level,s,h,g in groups:
        key=dict(level=level,station_id=s,horizon_hours=h)
        rows.append(dict(**key,**_summary(g,cfg)))
        b=pd.cut(g.actual_delta,[-np.inf,-200,-50,0,200,500,1000,np.inf],
                 labels=['fall_le_-200','fall_-200_-50','stable_-50_0','rise_0_200','rise_200_500','rise_500_1000','rise_gt_1000'])
        for name,sub in g.groupby(b,observed=True):
            ratio = sub.predicted_delta/sub.actual_delta if name in ('rise_200_500','rise_500_1000','rise_gt_1000') else pd.Series(dtype=float)
            e=sub.error_mm
            regimes.append(dict(**key,regime=str(name),n=len(sub),bias=e.mean(),mae=e.abs().mean(),rmse=np.sqrt((e**2).mean()),
                                actual_delta_mean=sub.actual_delta.mean(),predicted_delta_mean=sub.predicted_delta.mean(),
                                ratio_mean=ratio.mean(),ratio_median=ratio.median()))
    return pd.DataFrame(rows),pd.DataFrame(regimes)


def observation_events(observations, coverage, cfg):
    """12h exact observed rise crossing; never bridges missing hours or coverage gaps.

    `coverage` maps station to forecast target coverage bounds. Boundaries are
    censored and must not be treated as independently complete events.
    """
    required={'station_id','timestamp_utc','water_level_mm'}
    if not required.issubset(observations.columns): raise ValueError('Missing observation columns')
    o=observations.copy();o.station_id=o.station_id.astype(str);o.timestamp_utc=pd.to_datetime(o.timestamp_utc,utc=True,errors='raise')
    if o.duplicated(['station_id','timestamp_utc']).any(): raise ValueError('Duplicate observation keys')
    if not o.timestamp_utc.eq(o.timestamp_utc.dt.floor('h')).all(): raise ValueError('Observations must align to UTC hour')
    rows=[]
    for sid,g in o.groupby('station_id',sort=True):
        if sid not in coverage: continue
        start,end=coverage[sid];start=pd.Timestamp(start);end=pd.Timestamp(end)
        if start.tz is None or end.tz is None: raise ValueError('Coverage must be timezone aware')
        if start>end: raise ValueError('Invalid coverage')
        # Include history to calculate first rolling change, but never events outside coverage.
        series=g.set_index('timestamp_utc').water_level_mm.sort_index().reindex(pd.date_range(start-pd.Timedelta(hours=cfg.event_window_hours),end,freq='h'))
        values=pd.to_numeric(series,errors='coerce').replace([np.inf,-np.inf],np.nan)
        complete=values.notna().rolling(cfg.event_window_hours+1,min_periods=cfg.event_window_hours+1).sum().eq(cfg.event_window_hours+1)
        rise=values-values.shift(cfg.event_window_hours)
        flagged=(rise>cfg.rise_threshold_mm)&complete&(values.index>=start)
        # A missing hour, including an unobservable rolling window, always splits events.
        flag_times=values.index[flagged];clusters=[]
        for t in flag_times:
            if not clusters or (t-clusters[-1][-1]>pd.Timedelta(hours=cfg.event_merge_gap_hours+1)) or not complete.loc[clusters[-1][-1]:t].all(): clusters.append([t])
            else: clusters[-1].append(t)
        for i,ts in enumerate(clusters):
            s,e=ts[0],ts[-1];z=values.loc[s:e];peak=z.idxmax()
            rows.append(dict(station_id=sid,event_id=f'{sid}:{i+1}',start_utc=s,end_utc=e,
                peak_time_utc=peak,peak_level_mm=float(z.loc[peak]),peak_rolling_rise_mm=float(rise.loc[ts].max()),
                n_flagged_hours=len(ts), left_censored=((s-start)<=pd.Timedelta(hours=cfg.event_merge_gap_hours) or (s-pd.Timedelta(hours=1) in complete.index and not complete.loc[s-pd.Timedelta(hours=1)])),
                right_censored=((end-e)<=pd.Timedelta(hours=cfg.event_merge_gap_hours) or (e+pd.Timedelta(hours=1) in complete.index and not complete.loc[e+pd.Timedelta(hours=1)]))))
    return pd.DataFrame(rows,columns=['station_id','event_id','start_utc','end_utc','peak_time_utc','peak_level_mm','peak_rolling_rise_mm','n_flagged_hours','left_censored','right_censored'])


def event_matching(predictions, events, cfg, horizons=None):
    """Group issue-time warnings; greedy one-to-one overlap with event target times.

    Only alerts whose forecast target falls inside an event are eligible. Greedy
    matching by earliest issue time, then event start, is explicit, not optimal.
    """
    x=_check_predictions(predictions)
    if horizons is not None: x=x[x.horizon_hours.isin(horizons)].copy()
    w=x[x.predicted_delta>cfg.warning_delta_mm].sort_values(['station_id','anchor_utc','target_utc']).copy()
    episodes=[]; members=[]
    for sid,g in w.groupby('station_id',sort=True):
        times=g.anchor_utc.drop_duplicates().sort_values();c=(times.diff().isna() | (times.diff()>pd.Timedelta(hours=cfg.warning_merge_gap_hours))).cumsum()
        for j,ts in times.groupby(c):
            idx=g.anchor_utc.isin(ts);z=g[idx];eid=f'{sid}:w{j}'
            episodes.append(dict(station_id=sid,warning_id=eid,start_utc=ts.min(),end_utc=ts.max(),n_predictions=len(z)))
            for row in z.itertuples(): members.append(dict(station_id=sid,warning_id=eid,anchor_utc=row.anchor_utc,target_utc=row.target_utc,horizon_hours=row.horizon_hours))
    episodes=pd.DataFrame(episodes,columns=['station_id','warning_id','start_utc','end_utc','n_predictions'])
    members=pd.DataFrame(members,columns=['station_id','warning_id','anchor_utc','target_utc','horizon_hours'])
    candidates=[]
    for ev in events.itertuples():
        if members.empty: break
        eligible=members[(members.station_id==ev.station_id)&(members.target_utc>=ev.start_utc)&(members.target_utc<=ev.end_utc)]
        for wid,g in eligible.groupby('warning_id'):
            first=g.sort_values(['anchor_utc','target_utc']).iloc[0]
            candidates.append((first.anchor_utc,ev.start_utc,ev.event_id,wid,first.target_utc,int(first.horizon_hours)))
    used_e=set();used_w=set(); matches=[]
    for issue,_,eid,wid,target,h in sorted(candidates):
        if eid in used_e or wid in used_w: continue
        ev=events.loc[events.event_id==eid].iloc[0]; used_e.add(eid);used_w.add(wid)
        matches.append(dict(event_id=eid,warning_id=wid,first_issue_utc=issue,matched_target_utc=target,horizon_hours=h,
                            lead_to_event_start_hours=(ev.start_utc-issue).total_seconds()/3600,
                            lead_to_peak_hours=(ev.peak_time_utc-issue).total_seconds()/3600))
    matches=pd.DataFrame(matches,columns=['event_id','warning_id','first_issue_utc','matched_target_utc','horizon_hours','lead_to_event_start_hours','lead_to_peak_hours'])
    events=events.copy();events['detected']=events.event_id.isin(used_e)
    # Same valid time as the observed peak; latest issue avoids look-ahead.
    peak_rows=[]
    for ev in events.itertuples():
        available=x[(x.station_id==ev.station_id)&(x.target_utc==ev.peak_time_utc)&(x.anchor_utc<ev.peak_time_utc)]
        if not available.empty:
            row=available.sort_values('anchor_utc').iloc[-1]
            peak_rows.append(dict(event_id=ev.event_id,peak_forecast_issue_utc=row.anchor_utc,
                                  peak_forecast_horizon=int(row.horizon_hours),peak_predicted_level_mm=row.predicted_mm,
                                  peak_error_mm=row.predicted_mm-ev.peak_level_mm))
    cols=['event_id','peak_forecast_issue_utc','peak_forecast_horizon','peak_predicted_level_mm','peak_error_mm']
    events=events.merge(pd.DataFrame(peak_rows,columns=cols),on='event_id',how='left',validate='one_to_one')
    episodes['matched']=episodes.warning_id.isin(used_w)
    candidate_warning_ids={item[3] for item in candidates}
    episodes['has_observed_event_overlap']=episodes.warning_id.isin(candidate_warning_ids)
    episodes['status']=np.where(episodes.matched,'matched',
        np.where(episodes.has_observed_event_overlap,'unmatched_competing','unmatched_no_overlap'))
    return events,episodes,matches


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions',type=Path,required=True);parser.add_argument('--observations',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--event-window-hours',type=int,default=12)
    parser.add_argument('--event-merge-gap-hours',type=int,default=6);parser.add_argument('--warning-merge-gap-hours',type=int,default=1)
    parser.add_argument('--rise-threshold-mm',type=float,default=200);parser.add_argument('--warning-delta-mm',type=float,default=200)
    parser.add_argument('--tolerance-mm',type=float,default=20)
    args=parser.parse_args();cfg=EvaluationConfig(args.rise_threshold_mm,args.event_window_hours,args.event_merge_gap_hours,args.warning_merge_gap_hours,args.tolerance_mm,args.warning_delta_mm)
    pred=_check_predictions(pd.read_parquet(args.predictions));obs=pd.read_parquet(args.observations)
    # Wide regional parquet or explicit long format.
    if {'timestamp_utc','station_id','water_level_mm'}-set(obs.columns):
        if 'timestamp_utc' not in obs: raise ValueError('Observations require timestamp_utc')
        obs=obs.melt(id_vars=['timestamp_utc'],var_name='station_id',value_name='water_level_mm')
    # Common target coverage excludes horizons lacking complete overlap, not station-specific starts.
    cover={}
    for sid,g in pred.groupby('station_id'):
        spans=g.groupby('horizon_hours').target_utc.agg(['min','max']);start=spans['min'].max();end=spans['max'].min()
        if start>end: raise ValueError(f'No common horizon coverage: {sid}')
        for h,part in g.groupby('horizon_hours'):
            target=part.target_utc.sort_values()
            if target.duplicated().any() or (target.diff().dropna()!=pd.Timedelta(hours=1)).any():
                raise ValueError(f'Incomplete hourly coverage for station {sid}, horizon {h}')
        cover[sid]=(start,end)
    # Validate saved labels against the exact dataset observations, to catch incompatible artifacts.
    obs['timestamp_utc']=pd.to_datetime(obs.timestamp_utc,utc=True);obs['station_id']=obs.station_id.astype(str)
    pair=pred.merge(obs.rename(columns={'timestamp_utc':'target_utc','water_level_mm':'observed_target'}),on=['station_id','target_utc'],how='left',validate='many_to_one')
    if pair.observed_target.isna().any() or not np.allclose(pair.actual_mm,pair.observed_target,rtol=0,atol=.01):
        raise ValueError('Prediction targets do not match observed series; select compatible inputs')
    current=pred.merge(obs.rename(columns={'timestamp_utc':'anchor_utc','water_level_mm':'observed_current'}),on=['station_id','anchor_utc'],how='left',validate='many_to_one')
    if current.observed_current.isna().any() or not np.allclose(current.current_mm,current.observed_current,rtol=0,atol=.01):
        raise ValueError('Prediction current levels do not match observed series')
    summary,regimes=regression_detection(pred,cfg);events=observation_events(obs,cover,cfg)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    summary.to_csv(args.output_dir/'regression_detection.csv',index=False);regimes.to_csv(args.output_dir/'regimes.csv',index=False)
    curves=[]
    for (sid,h),g in pred.groupby(['station_id','horizon_hours']):
        y=g.actual_delta>cfg.rise_threshold_mm
        if not y.any(): continue
        precision,recall,threshold=precision_recall_curve(y,g.predicted_delta)
        curves.extend(dict(station_id=sid,horizon_hours=h,threshold_mm=t,precision=precision[i],recall=recall[i])
                      for i,t in enumerate(threshold))
    pd.DataFrame(curves,columns=['station_id','horizon_hours','threshold_mm','precision','recall']).to_csv(args.output_dir/'precision_recall.csv',index=False)
    for label,hs in [('combined',None)]+[(f'h{int(h)}',[h]) for h in sorted(pred.horizon_hours.unique())]:
        subset=pred if hs is None else pred[pred.horizon_hours.isin(hs)]
        # Only fully covered prediction targets, to avoid matching censored horizons.
        subset=subset[subset.apply(lambda r: cover[r.station_id][0]<=r.target_utc<=cover[r.station_id][1],axis=1)]
        ev,wa,ma=event_matching(subset,events,cfg,hs)
        ev.to_csv(args.output_dir/f'events_{label}.csv',index=False);wa.to_csv(args.output_dir/f'warnings_{label}.csv',index=False);ma.to_csv(args.output_dir/f'matches_{label}.csv',index=False)
        rows=[]
        for sid in sorted(cover):
            e=ev[ev.station_id==sid];w=wa[wa.station_id==sid];complete=e[~(e.left_censored|e.right_censored)]
            rows.append(dict(station_id=sid,events=len(e),detected=int(e.detected.sum()),
                uncensored_events=len(complete),uncensored_detected=int(complete.detected.sum()),
                warning_episodes=len(w),warning_matched=int(w.matched.sum()),
                warning_unmatched_competing=int((w.status=='unmatched_competing').sum()),
                warning_unmatched_no_overlap=int((w.status=='unmatched_no_overlap').sum()),
                peak_mae_mm=e.peak_error_mm.abs().mean()))
        pd.DataFrame(rows).to_csv(args.output_dir/f'event_summary_{label}.csv',index=False)
    import hashlib
    def digest(path):
        h=hashlib.sha256()
        with path.open('rb') as source:
            for chunk in iter(lambda: source.read(1024*1024),b''): h.update(chunk)
        return h.hexdigest()
    (args.output_dir/'config.json').write_text(json.dumps(dict(config=asdict(cfg),event_type='rolling_observed_rise',flood_warning_thresholds='not provided',
        predictions_path=str(args.predictions),predictions_sha256=digest(args.predictions),observations_path=str(args.observations),observations_sha256=digest(args.observations),
        coverage={s:[str(a),str(b)] for s,(a,b) in cover.items()}),indent=2)+'\n')
    print(f'Wrote evaluation to {args.output_dir}')

if __name__=='__main__': main()
