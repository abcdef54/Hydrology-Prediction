"""Describe contiguous large-error runs and antecedent rain; no causal attribution."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

OUT=Path('reports/large_error_events_2026-10-15')
WEST=['750062','750007','750003','750011']
SOUTH=['750010','228005','750066']
TZ='Asia/Ho_Chi_Minh'


def runs(series):
    """Return strictly adjacent hourly True sequences; missing/False breaks a run."""
    s=series.sort_index().fillna(False)
    active=s[s.astype(bool)]
    if active.empty:return []
    ids=active.index.to_series().diff().ne(pd.Timedelta(hours=1)).cumsum()
    return [part.index for _,part in active.groupby(ids)]


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    root=Path('/tmp/hydrology-rain-regimes')
    obs=pd.read_csv(root/'observations.csv',dtype={'station_id':str})
    obs['observed_at']=pd.to_datetime(obs.observed_at,utc=True)
    f=pd.read_csv(root/'forecasts.csv')
    for col in ['forecast_at','anchor_observed_at']:f[col]=pd.to_datetime(f[col],utc=True)
    water=obs[obs.station_id.eq('0074647403')].set_index('observed_at').depth
    b=water.resample('1h',closed='right',label='right')
    truth=b.mean().where(b.count().eq(6))
    f['actual']=f.forecast_at.map(truth);f['anchor_level']=f.anchor_observed_at.map(truth)
    f['error']=f.water_level_mm-f.actual;f['ae']=f.error.abs();f['bad']=f.ae.gt(200)&f.actual.notna()
    f['local']=f.forecast_at.dt.tz_convert(TZ)
    f['rainy']=f.local.dt.month.isin([9,10,11,12])
    f['persistence_ae']=(f.anchor_level-f.actual).abs()
    grid=truth.index
    rain=pd.DataFrame(index=grid);complete=pd.DataFrame(index=grid)
    for sid in WEST+SOUTH:
        r=obs[obs.station_id.eq(sid)].set_index('observed_at').depth.resample('1h',closed='right',label='right')
        rain[sid]=r.sum(min_count=1).reindex(grid).where(r.count().reindex(grid).eq(6))
    for name,ids in [('west',WEST),('south',SOUTH)]:
        rain[name]=rain[ids].mean(axis=1).where(rain[ids].notna().all(axis=1))
        rain[name+'_24']=rain[name].rolling(24,min_periods=24).sum()
        rain[name+'_72']=rain[name].rolling(72,min_periods=72).sum()
        for phase,col in [('target','forecast_at'),('known','anchor_observed_at')]:
            f[f'{phase}_{name}_24']=f[col].map(rain[name+'_24'])
    f['change3']=f.forecast_at.map(truth-truth.shift(3))
    runrows=[];runsummary=[]
    for h,g in f.groupby('horizon_minutes'):
        g=g.set_index('forecast_at').sort_index()
        for seq in runs(g.bad):
            sub=g.loc[seq]
            runrows.append(dict(horizon_hours=h//60,start_local=str(seq[0].tz_convert(TZ)),end_local=str(seq[-1].tz_convert(TZ)),consecutive_hourly_targets=len(seq),max_ae=sub.ae.max(),mean_ae=sub.ae.mean(),mean_bias=sub.error.mean(),under_count=sub.error.lt(0).sum(),over_count=sub.error.gt(0).sum(),rainy=bool(sub.rainy.all())))
        for season,part in [('all',g),('rainy_months',g[g.rainy])]:
            rr=runs(part.bad);total=sum(len(x) for x in rr);clustered=sum(len(x) for x in rr if len(x)>=2)
            runsummary.append(dict(horizon_hours=h//60,scope=season,valid=int(part.actual.notna().sum()),bad=total,runs=len(rr),longest_run=max([len(x) for x in rr],default=0),bad_in_runs_ge2=clustered,clustered_pct=100*clustered/total if total else 0))
    rdf=pd.DataFrame(runrows);rdf.to_csv(OUT/'all_error_runs.csv',index=False)
    pd.DataFrame(runsummary).to_csv(OUT/'run_summary.csv',index=False)
    f.to_csv(OUT/'forecast_context.csv',index=False)
    rain.to_csv(OUT/'hourly_rain_context.csv',index_label='timestamp_utc')
    # Compare rainy-season target states, retaining only complete rain context.
    conditions=[]
    v=f[f.rainy & f.actual.notna() & f.anchor_level.notna()].copy()
    for h,g in v.groupby('horizon_minutes'):
        for phase in ['target','known']:
            valid=g.dropna(subset=[f'{phase}_west_24',f'{phase}_south_24'])
            wet=(valid[f'{phase}_west_24']>=50)|(valid[f'{phase}_south_24']>=50)
            dry=(valid[f'{phase}_west_24']<=10)&(valid[f'{phase}_south_24']<=10)
            for label,mask in [('either_group_24h_ge50',wet),('both_groups_24h_le10',dry)]:
                p=valid[mask]
                conditions.append(dict(horizon_hours=h//60,phase=phase,state=label,n=len(p),bad=int(p.bad.sum()),bad_pct=p.bad.mean()*100,mae=p.ae.mean(),bias=p.error.mean(),persistence_mae=p.persistence_ae.mean()))
        for label,mask in [('rise3h_ge100',g.change3>=100),('fall3h_le_minus100',g.change3<=-100),('abschange3h_le25',g.change3.abs()<=25)]:
            p=g[mask];conditions.append(dict(horizon_hours=h//60,phase='water_change',state=label,n=len(p),bad=int(p.bad.sum()),bad_pct=p.bad.mean()*100,mae=p.ae.mean(),bias=p.error.mean(),persistence_mae=p.persistence_ae.mean()))
    pd.DataFrame(conditions).to_csv(OUT/'conditional_metrics.csv',index=False)
    # Any-horizon consecutive-error runs are distinct from horizon-specific runs.
    anybad=f.groupby('forecast_at').bad.any()
    pd.DataFrame([dict(start=str(x[0].tz_convert(TZ)),end=str(x[-1].tz_convert(TZ)),hours=len(x)) for x in runs(anybad)]).to_csv(OUT/'any_horizon_runs.csv',index=False)
    events=[('nov2025','2025-11-22','2025-11-25'),('sep2026','2026-09-11','2026-09-15'),('oct7','2026-10-07','2026-10-11'),('oct12','2026-10-12','2026-10-15'),('sep30','2026-09-29','2026-10-03')]
    eventrows=[];eventrain=[];eventops=[];wetrows=[]
    rawrain=pd.concat([pd.read_csv(p,dtype={'station_id':str}) for p in Path('/tmp/hydrology-error-events').glob('rain_*.csv')],ignore_index=True).drop_duplicates(['station_id','observed_at'])
    rawrain['observed_at']=pd.to_datetime(rawrain.observed_at,utc=True)
    stations=pd.read_csv('/tmp/hydrology-error-events/stations.csv',dtype={'id':str}).set_index('id')
    ops=pd.concat([pd.read_csv(p) for p in Path('/tmp/hydrology-error-events').glob('operations_*.csv')],ignore_index=True).drop_duplicates()
    ops['observed_at']=pd.to_datetime(ops.observed_at,utc=True)
    for name,aa,bb in events:
        a=pd.Timestamp(aa,tz=TZ);z=pd.Timestamp(bb,tz=TZ)
        sel=f[(f.forecast_at>=a)&(f.forecast_at<z)]
        w=truth[(truth.index>=a)&(truth.index<z)]
        seqs=runs(sel.groupby('forecast_at').bad.any())
        maxchange=(w-w.shift(3));max24=(w-w.shift(24))
        eventrows.append(dict(event=name,start=str(a),end_exclusive=str(z),forecast_points=int(sel.actual.notna().sum()),bad=int(sel.bad.sum()),bad_pct=sel.bad.mean()*100,unique_bad_hours=sel[sel.bad].forecast_at.nunique(),longest_any_run=max([len(x) for x in seqs],default=0),water_min=w.min(),water_max=w.max(),water_peak_local=str(w.idxmax().tz_convert(TZ)),max_rise3h=maxchange.max(),max_fall3h=maxchange.min(),max_rise24h=max24.max()))
        for group in ['west','south']:
            r=rain.loc[(rain.index>=a)&(rain.index<z),group]
            for threshold in [.2,1,5]:
                rr=runs(r>=threshold)
                longest=max(rr,key=len) if rr else []
                wetrows.append(dict(event=name,group=group,threshold=threshold,valid_hours=r.notna().sum(),wet_hours=r.ge(threshold).sum(),longest_consecutive_wet_hours=len(longest),longest_start=str(longest[0].tz_convert(TZ)) if len(longest) else '',longest_end=str(longest[-1].tz_convert(TZ)) if len(longest) else '',max_hour=r.max(),max24=rain.loc[(rain.index>=a)&(rain.index<z),group+'_24'].max(),max72=rain.loc[(rain.index>=a)&(rain.index<z),group+'_72'].max()))
        # Fixed antecedent/event window, same endpoints for all stations.
        ra=a-pd.Timedelta(hours=24);rz=z-pd.Timedelta(hours=1)
        raw=rawrain[(rawrain.observed_at>ra)&(rawrain.observed_at<=rz)]
        expected=int((rz-ra).total_seconds()/600)
        for sid,g in raw.groupby('station_id'):
            eventrain.append(dict(event=name,station_id=sid,name=stations.loc[sid,'name'],start_exclusive=str(ra),end_inclusive=str(rz),valid=g.depth.notna().sum(),expected=expected,coverage_pct=g.depth.notna().sum()/expected*100,sum_source_readings=g.depth.sum(min_count=1)))
        oo=ops[(ops.observed_at>=ra)&(ops.observed_at<z)&ops.name.isin(['Hồ Tả Trạch','Thủy điện Bình Điền','Thủy điện Hương Điền'])]
        for nm,g in oo.groupby('name'):
            eventops.append(dict(event=name,name=nm,n=len(g),first=str(g.observed_at.min().tz_convert(TZ)),last=str(g.observed_at.max().tz_convert(TZ)),qdi_min=g.qdi.min(),qdi_max=g.qdi.max(),qden_min=g.qden.min(),qden_max=g.qden.max(),qdi_max_time=str(g.loc[g.qdi.idxmax(),'observed_at'].tz_convert(TZ)) if g.qdi.notna().any() else ''))
        plotstart=a-pd.Timedelta(hours=24)
        plotrain=rain.loc[(rain.index>=plotstart)&(rain.index<z)]
        fig,axes=plt.subplots(4,1,figsize=(15,12),sharex=True,gridspec_kw={'height_ratios':[1,1,1.6,1.2]})
        for group,color in [('west','#487c25'),('south','#8a4cad')]:
            axes[0].plot(plotrain.index.tz_convert(TZ),plotrain[group],label=group,color=color)
            axes[1].plot(plotrain.index.tz_convert(TZ),plotrain[group+'_24'],label=group,color=color)
        axes[0].set_ylabel('Hourly rain source sum\n(station mean)');axes[1].set_ylabel('Rolling 24h source sum')
        tw=truth.loc[(truth.index>=plotstart)&(truth.index<z)]
        axes[2].plot(tw.index.tz_convert(TZ),tw,color='#ce4525',label='Observed hourly mean',linewidth=2)
        for h,col in [(60,'#1775b5'),(360,'#888888'),(1440,'#159080')]:
            ss=f[(f.horizon_minutes==h)&(f.forecast_at>=plotstart)&(f.forecast_at<z)].set_index('forecast_at').sort_index().reindex(tw.index)
            axes[2].plot(ss.index.tz_convert(TZ),ss.water_level_mm,label=f'+{h//60}h',color=col,alpha=.85)
            axes[3].plot(ss.index.tz_convert(TZ),ss.error,label=f'+{h//60}h',color=col)
        axes[2].set_ylabel('Water level (stored mm)');axes[3].set_ylabel('Signed error (mm)')
        axes[3].axhline(200,color='red',ls='--');axes[3].axhline(-200,color='red',ls='--');axes[3].axhline(0,color='black',lw=.5)
        for ax in axes:
            ax.axvline(a,color='black',linestyle=':',alpha=.4);ax.legend(loc='upper left',ncol=4,fontsize=8);ax.grid(alpha=.2)
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %Hh',tz=a.tzinfo));axes[-1].set_xlabel('Vietnam local time; dotted line = analysis window start')
        fig.suptitle(f'Da Vien error event: {name} | observational association, not flood attribution')
        fig.tight_layout();fig.savefig(OUT/f'event_{name}.png',dpi=160);fig.savefig(OUT/f'event_{name}.pdf');plt.close(fig)
    for name,data in [('event_summary',eventrows),('event_station_rain',eventrain),('event_operations',eventops),('wet_spells',wetrows)]:pd.DataFrame(data).to_csv(OUT/f'{name}.csv',index=False)
    summary=pd.DataFrame(eventrows)
    print('EVENTS\n',summary.round(2).to_string(index=False))
    print('RUNS\n',pd.DataFrame(runsummary).round(2).to_string(index=False))
    print('LONGEST\n',rdf.sort_values('consecutive_hourly_targets',ascending=False).head(12).round(2).to_string(index=False))
    print('WET\n',pd.DataFrame(wetrows).query('threshold == 1').round(2).to_string(index=False))
    print('CONDITIONS24\n',pd.DataFrame(conditions).query('horizon_hours == 24').round(2).to_string(index=False))
    meta=dict(threshold=200,rainy_months=[9,10,11,12],runs='strict consecutive hourly target times with absolute error >200; gaps break',rain='right-closed hourly source sums; six valid readings/hour per station; groups mean of fixed stations; missing retained',caveats=['No Da Vien alarm thresholds in database','Station level datum not reconciled to official gauges','Rain source interval not verified','Saved predictions may be historical backfill; train/test mixed','No causal attribution from temporal coincidence'])
    (OUT/'manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
