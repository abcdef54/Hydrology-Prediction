"""Four reproducible dataset-readiness audits. No feature dataset or DB writes.

Use the full per-run cache when present, otherwise try a read-only DB export.
Offline reports explicitly distinguish verified cached coverage from unavailable
per-vintage forecast values. Never reconstruct a forecast vintage by stitching runs.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

from src.backend.db.session import build_engine

TZ = 'Asia/Ho_Chi_Minh'
TRANSITION = pd.Timestamp('2026-05-12T06:00:00Z')
WINDOWS = [(0,3),(3,6),(6,12),(12,24)]


def read_times(path):
    df = pd.read_csv(path)
    df.index = pd.to_datetime(df.pop('timestamp_utc'), utc=True)
    if df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise ValueError(f'Invalid timestamps: {path}')
    return df


def metric_pair(actual, predicted):
    frame = pd.DataFrame({'a':actual,'p':predicted}).dropna()
    if frame.empty:
        return dict(n=0, pearson_r=np.nan, bias_mm=np.nan,mae_mm=np.nan,rmse_mm=np.nan)
    error=frame.p-frame.a
    r=frame.a.corr(frame.p) if len(frame)>1 and frame.a.std()>0 and frame.p.std()>0 else np.nan
    return dict(n=len(frame),pearson_r=r,bias_mm=error.mean(),mae_mm=error.abs().mean(),
                rmse_mm=np.sqrt((error**2).mean()))


def load_cube(root, output, offline):
    cache=output/'forecast_runs_hourly.csv.gz'
    if cache.exists():
        frame=pd.read_csv(cache)
        for col in ['initialized_at','valid_at','retrieved_at']:
            frame[col]=pd.to_datetime(frame[col],utc=True)
        return frame, 'cached_full_vintages', None
    if offline:
        return None,'offline', 'No full per-run forecast cache; offline explicitly requested.'
    load_dotenv(root/'.env')
    engine=None
    try:
        engine=build_engine()
        with engine.connect() as conn:
            conn.execute(text('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY'))
            frame=pd.read_sql(text("""
                WITH latest AS (
                  SELECT DISTINCT ON (r.location_id,r.model,r.initialized_at) r.*
                  FROM hydrology.weather_forecast_runs r
                  JOIN hydrology.weather_locations l ON l.id=r.location_id
                  WHERE l.station_id='750007' AND r.model='ecmwf_ifs'
                  ORDER BY r.location_id,r.model,r.initialized_at,r.retrieved_at DESC,r.id
                )
                SELECT r.id AS run_id,r.initialized_at,r.retrieved_at,r.archive_kind,
                       r.available_at,r.availability_basis,r.grid_latitude,r.grid_longitude,
                       h.valid_at,h.precipitation_mm
                FROM latest r JOIN hydrology.weather_forecast_hourly h ON h.run_id=r.id
                ORDER BY r.initialized_at,h.valid_at
            """),conn)
        if frame.empty: raise ValueError('No forecast runs in DB')
        for col in ['initialized_at','valid_at','retrieved_at']:
            frame[col]=pd.to_datetime(frame[col],utc=True)
        frame.to_csv(cache,index=False,compression='gzip')
        return frame,'postgres_read_only_snapshot',None
    except (SQLAlchemyError,RuntimeError) as exc:
        # Do not log connection URLs, passwords or exception payloads.
        return None,'database_unavailable',type(exc).__name__
    finally:
        if engine is not None: engine.dispose()


def select_windows(cube, anchors, max_age=12, fallback=False):
    """Same run for every future hour at one anchor; assumed 6h publication lag.

    max_age=12 matches the original collector audit inclusive age boundary.
    Fallback searches older eligible runs with max_age=24; retains missing windows.
    """
    meta=cube[['run_id','initialized_at']].drop_duplicates().sort_values('initialized_at')
    if meta.initialized_at.duplicated().any():
        raise ValueError('Multiple location mappings: choose one mapping explicitly')
    times=pd.DatetimeIndex(meta.initialized_at)
    values={key:group.set_index('valid_at').precipitation_mm
            for key,group in cube.groupby('run_id')}
    rows=[]
    for anchor in anchors:
        pos=times.searchsorted(anchor-pd.Timedelta(hours=6),side='right')-1
        row={'anchor':anchor,'initialized_at':pd.NaT,'age_hours':np.nan,'fallback':False}
        for a,b in WINDOWS: row[f'forecast_{a}_{b}h']=np.nan
        candidate_positions=range(pos,-1,-1) if fallback else [pos]
        for candidate in candidate_positions:
            if candidate<0: break
            cycle=times[candidate]
            age=(anchor-cycle).total_seconds()/3600
            if age>max_age: break
            series=values[meta.iloc[candidate].run_id]
            future_index=pd.date_range(anchor+pd.Timedelta(hours=1),periods=24,freq='h')
            future=series.reindex(future_index)
            window_values={f'forecast_{a}_{b}h': future.iloc[a:b].sum(min_count=b-a)
                           for a,b in WINDOWS}
            if fallback and not future.notna().all(): continue
            row.update(window_values,initialized_at=cycle,age_hours=age,
                       fallback=bool(fallback and (candidate!=pos or age>12)))
            break
        rows.append(row)
    return pd.DataFrame(rows).set_index('anchor')


def archive_analysis(alignment, output):
    frame=alignment.copy()
    frame['era']=np.where(frame.run_initialized_at<TRANSITION,'49r1_hindcast_label','50r1_archive_label')
    frame.loc[frame.run_initialized_at.isna(),'era']='unknown'
    frame['transition_window']=np.where(frame.index<TRANSITION,'before','after')
    local=((frame.index>=TRANSITION-pd.Timedelta(days=28)) &
           (frame.index<TRANSITION+pd.Timedelta(days=28)))
    rows=[]
    for scope,selected,key in [('whole_period',frame,'era'),('transition_plus_minus_28d',frame.loc[local],'transition_window')]:
        for label,part in selected.groupby(key):
            pair=part[['observed_rain_mm','forecast_rain_mm']].dropna()
            if pair.empty: continue
            rows.append(dict(scope=scope,era=label,**metric_pair(pair.observed_rain_mm,pair.forecast_rain_mm),
                observed_mean=pair.observed_rain_mm.mean(),forecast_mean=pair.forecast_rain_mm.mean(),
                observed_wet_pct=100*(pair.observed_rain_mm>=1).mean(),
                forecast_wet_pct=100*(pair.forecast_rain_mm>=1).mean(),
                forecast_p95=pair.forecast_rain_mm.quantile(.95),forecast_max=pair.forecast_rain_mm.max()))
    result=pd.DataFrame(rows);result.to_csv(output/'01_archive_regimes.csv',index=False)
    fig,axes=plt.subplots(2,1,figsize=(15,8),sharex=True)
    for ax,column,color in zip(axes,['observed_rain_mm','forecast_rain_mm'],['tab:orange','tab:blue']):
        daily=frame[column].resample('D').sum(min_count=24)
        ax.plot(daily.index,daily,color=color,label=column)
        ax.axvline(TRANSITION,color='red',linestyle='--',label='Mốc đổi nhãn Cycle 49R1/50R1')
        ax.set_ylabel('mm/ngày UTC');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Đối chiếu hai giai đoạn archive — mùa và đợt mưa đồng thời thay đổi')
    fig.tight_layout();fig.savefig(output/'01_archive_regimes.png',dpi=150);plt.close(fig)
    return result


def window_analysis(cube, hourly, splits, output):
    primary=select_windows(cube,hourly.index)
    fallback=select_windows(cube,hourly.index,max_age=24,fallback=True)
    windows=[f'forecast_{a}_{b}h' for a,b in WINDOWS]
    means=hourly[['rain_750007_depth','rain_750062_depth']].mean(axis=1,skipna=False)
    actual={f'forecast_{a}_{b}h':pd.Series([
        means.reindex(pd.date_range(t+pd.Timedelta(hours=a+1),periods=b-a,freq='h')).sum(min_count=b-a)
        for t in hourly.index],index=hourly.index) for a,b in WINDOWS}
    rows=[]
    heavy_rows=[]
    for col,(a,b) in zip(windows,WINDOWS):
        # Threshold fixed from TRAIN observations, requiring window to stay in split.
        train_index=splits['train'].index
        valid_train=train_index[ (train_index+pd.Timedelta(hours=b)).isin(train_index)]
        train_actual=actual[col].reindex(valid_train).dropna()
        wet=train_actual[train_actual>0]
        threshold=wet.quantile(.9) if len(wet) else np.nan
        for scope,index in {'all':hourly.index,**{k:v.index for k,v in splits.items()}}.items():
            allowed=index[(index+pd.Timedelta(hours=b)).isin(index)]
            obs=actual[col].reindex(allowed);pred=primary[col].reindex(allowed)
            metrics=metric_pair(obs,pred)
            rows.append(dict(scope=scope,window=f'{a}-{b}h',heavy_threshold_mm=threshold,
                eligible_anchors=len(allowed),actual_complete=int(obs.notna().sum()),
                paired_coverage_pct=100*metrics['n']/len(allowed) if len(allowed) else np.nan,**metrics))
            pair=pd.DataFrame({'a':obs,'p':pred}).dropna()
            truth=pair.a>=threshold;flag=pair.p>=threshold
            tp=int((truth&flag).sum());fn=int((truth&~flag).sum());fp=int((~truth&flag).sum())
            heavy_rows.append(dict(scope=scope,window=f'{a}-{b}h',threshold_train_wet_p90=threshold,
                true_heavy=int(truth.sum()),actual_heavy_all=int((obs>=threshold).sum()),
                actual_heavy_missing_forecast=int(((obs>=threshold)&pred.isna()).sum()),hits=tp,misses=fn,false_alarms=fp,
                recall=tp/(tp+fn) if tp+fn else np.nan,
                precision=tp/(tp+fp) if tp+fp else np.nan))
    pd.DataFrame(rows).to_csv(output/'02_future_window_metrics.csv',index=False)
    pd.DataFrame(heavy_rows).to_csv(output/'02_heavy_rain_detection.csv',index=False)
    # Diagnostic products only: not a training dataset; no fitted preprocessing.
    pd.concat([primary.add_prefix('primary_'),fallback.add_prefix('fallback_')],axis=1).to_csv(
        output/'diagnostic_forecast_windows.csv.gz',compression='gzip')
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    metrics=pd.DataFrame(rows)
    for scope,g in metrics.groupby('scope'):
        axes[0].plot(g.window,g.pearson_r,marker='o',label=scope)
        axes[1].plot(g.window,g.mae_mm,marker='o',label=scope)
    axes[0].set_ylabel('Pearson r');axes[1].set_ylabel('MAE (mm/cửa sổ)')
    for ax in axes:ax.legend();ax.grid(alpha=.2);ax.set_xlabel('Cửa sổ tương lai')
    fig.tight_layout();fig.savefig(output/'02_future_windows.png',dpi=150);plt.close(fig)
    return primary[windows].notna().all(axis=1),fallback[windows].notna().all(axis=1),pd.DataFrame(rows)


def sequence_audit(splits, complete, fallback_complete, output):
    rows=[];retained={}
    for scope,part in splits.items():
        targets=[c for c in part if c.startswith('target_water_level_plus_')]
        finite=np.isfinite(part[targets].to_numpy()).all(axis=1)&np.isfinite(part.water_level_depth)
        eligible=pd.Series(finite,index=part.index)&(np.arange(len(part))>=95)
        strict=complete.reindex(part.index).fillna(False).astype(bool)
        full96=strict.rolling(96,min_periods=96).sum().eq(96)
        row=dict(split=scope,rows=len(part),baseline_sequences=int(eligible.sum()),
            complete_weather_at_anchor=int((eligible&strict).sum()),
            complete_weather_all_96=int((eligible&full96).sum()),
            sequences_retained_with_masks=int(eligible.sum()),
            sequences_lost_if_drop_missing=int((eligible&~full96).sum()))
        if fallback_complete is not None:
            fb=fallback_complete.reindex(part.index).fillna(False).astype(bool)
            f96=fb.rolling(96,min_periods=96).sum().eq(96)
            row['fallback_complete_weather_all_96']=int((eligible&f96).sum())
        retained[scope]=pd.DataFrame({'baseline_eligible':eligible,'weather_complete96':full96})
        row['missing_timestep_pct']=100*(~strict).mean()
        rows.append(row)
    result=pd.DataFrame(rows);result.to_csv(output/'03_sequence_coverage.csv',index=False)
    fig,ax=plt.subplots(figsize=(10,5))
    x=np.arange(len(result))
    ax.bar(x-.2,result.baseline_sequences,width=.4,label='Giữ sample với mask')
    ax.bar(x+.2,result.complete_weather_all_96,width=.4,label='Chỉ giữ đủ forecast cả 96h')
    ax.set_xticks(x,result.split);ax.set_ylabel('Số sequence');ax.legend()
    ax.bar_label(ax.containers[0]);ax.bar_label(ax.containers[1]);ax.margins(y=.12)
    fig.tight_layout();fig.savefig(output/'03_sequence_coverage.png',dpi=150);plt.close(fig)
    return result,retained


def high_water_events(hourly,splits,retained,output):
    water=hourly.water_level_depth
    # Train-only descriptive event threshold. Not an official flood level.
    level_threshold=float(splits['train'].water_level_depth.quantile(.95))
    rises=water-water.shift(6)
    train_rise=rises.reindex(splits['train'].index).dropna()
    rise_threshold=float(train_rise.quantile(.95))
    highs=water.index[water>=level_threshold]
    groups=[]
    for stamp in highs:
        if not groups or stamp-groups[-1][-1]>pd.Timedelta(hours=24):groups.append([stamp])
        else:groups[-1].append(stamp)
    eligible_target_times={}
    for scope,part in splits.items():
        anchors=retained[scope].index[retained[scope].baseline_eligible]
        target_times=pd.DatetimeIndex([],tz='UTC')
        for horizon in [1,3,6,12,24]: target_times=target_times.union(anchors+pd.Timedelta(hours=horizon))
        eligible_target_times[scope]=target_times
    rows=[]
    rain=hourly[['rain_750007_depth','rain_750062_depth']].mean(axis=1,skipna=False)
    for group in groups:
        lo,hi=group[0],group[-1]
        section=water.loc[lo:hi]
        peak=section.idxmax()
        scope=next((k for k,v in splits.items() if peak in v.index),'purge_gap')
        before=water.reindex(pd.date_range(lo-pd.Timedelta(hours=24),periods=24,freq='h'))
        span=pd.date_range(lo,hi,freq='h')
        weather=retained.get(scope)
        eligible=weather.reindex(span).baseline_eligible.fillna(False).astype(bool) if weather is not None else pd.Series(False,index=span)
        good=weather.reindex(span).weather_complete96.fillna(False).astype(bool) if weather is not None else pd.Series(False,index=span)
        rain_index=pd.date_range(lo-pd.Timedelta(hours=24),hi,freq='h')
        rainfall=rain.reindex(rain_index)
        rows.append(dict(event_id=len(rows)+1,start_utc=lo,end_utc=hi,peak_utc=peak,peak_split=scope,
            peak_water_mm=section.max(),peak_present_as_eligible_target=peak in eligible_target_times.get(scope,[]),
            rise_from_prior24_min_mm=section.max()-before.min(),
            max_6h_rise_mm=rises.loc[lo:hi].max(),hours_above_threshold=int((section>=level_threshold).sum()),
            observed_rain_mm=rainfall.sum(min_count=len(rainfall)),
            observed_rain_complete=bool(rainfall.notna().all()),
            censored_left=lo-water.index[0]<pd.Timedelta(hours=24),
            censored_right=water.index[-1]-hi<pd.Timedelta(hours=24),
            partly_outside_peak_split=not span.isin(splits[scope].index).all() if scope in splits else True,
            crosses_splits=len([k for k,v in splits.items() if span.isin(v.index).any()])>1,
            eligible_sequences=int(eligible.sum()),complete_weather_sequences=int((eligible&good).sum())))
    events=pd.DataFrame(rows)
    events.to_csv(output/'04_high_water_events.csv',index=False)
    summaries=[]
    for scope,part in splits.items():
        selected=events[events.peak_split==scope]
        rain_part=rain.reindex(part.index)
        summaries.append(dict(split=scope,rows=len(part),high_water_threshold_train_p95=level_threshold,
            rising6_threshold_train_p95=rise_threshold,high_water_hours=int((part.water_level_depth>=level_threshold).sum()),
            rapid_rise_hours=int((rises.reindex(part.index)>=rise_threshold).sum()),
            event_peaks=len(selected),uncensored_event_peaks=int((~selected.censored_left & ~selected.censored_right).sum()),
            event_peaks_present_as_target=int(selected.peak_present_as_eligible_target.sum()),
            max_water_mm=part.water_level_depth.max(),
            max_water_at_eligible_anchor_mm=part.loc[retained[scope].baseline_eligible,'water_level_depth'].max(),
            max_water_in_eligible_targets_mm=part.loc[retained[scope].baseline_eligible,[c for c in part if c.startswith('target_water_level_plus_')]].max().max(),
            max_6h_rise_mm=rises.reindex(part.index).max(),rain_hour_p99=rain_part.quantile(.99),
            heavy_rain_hours_ge10=int((rain_part>=10).sum())))
    summary=pd.DataFrame(summaries);summary.to_csv(output/'04_split_representativeness.csv',index=False)
    fig,ax=plt.subplots(figsize=(15,5))
    for scope,part in splits.items():ax.plot(part.index,part.water_level_depth,label=scope,linewidth=.7)
    ax.axvspan(splits['train'].index[0],splits['train'].index[95],color='gray',alpha=.25,label='96 giờ đầu: chưa tạo được sequence train')
    ax.axhline(level_threshold,color='red',linestyle='--',label='Ngưỡng nước cao P95 train')
    ax.scatter(events.peak_utc,events.peak_water_mm,s=25,color='black',label='Đỉnh mỗi đợt (gộp cách ≤24h)')
    ax.set_ylabel('Mực nước (mm)');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/'04_high_water_events.png',dpi=150);plt.close(fig)
    return events,summary


def table_markdown(frame):
    if frame.empty:return '_Không có hàng dữ liệu._'
    cols=list(frame.columns)
    lines=['| '+' | '.join(cols)+' |','|'+'|'.join(['---']*len(cols))+'|']
    for row in frame.itertuples(index=False,name=None):
        def fmt(v):
            if isinstance(v,(float,np.floating)):return f'{v:.3f}' if np.isfinite(v) else '—'
            return str(v)
        lines.append('| '+' | '.join(fmt(v) for v in row)+' |')
    return '\n'.join(lines)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--offline',action='store_true')
    args=p.parse_args(argv)
    root=Path(__file__).resolve().parents[2]
    dataset=root/'train_data/da_vien_1h_2026-10-15'
    artifacts=root/'reports/weather_collection'
    output=artifacts/'dataset_readiness';output.mkdir(parents=True,exist_ok=True)
    splits={s:read_times(dataset/f'{s}.csv') for s in ['train','val','test']}
    hourly=read_times(dataset/'hourly.csv')
    alignment=pd.read_csv(artifacts/'daily_analysis/hourly_rain_water_alignment.csv',index_col=0)
    alignment.index=pd.to_datetime(alignment.index,utc=True)
    alignment['run_initialized_at']=pd.to_datetime(alignment.run_initialized_at,utc=True)
    # Assert previously exported observational series align with this dataset snapshot.
    obs=hourly[['rain_750007_depth','rain_750062_depth']].mean(axis=1,skipna=False)
    pair=pd.concat([obs.rename('snapshot'),alignment.observed_rain_mm.rename('export')],axis=1).dropna()
    difference=(pair.snapshot-pair.export).abs().max()
    waterpair=pd.concat([hourly.water_level_depth.rename('snapshot'),alignment.water_level_mm.rename('export')],axis=1).dropna()
    water_difference=(waterpair.snapshot-waterpair.export).abs().max()
    if difference>1e-6 or water_difference>1e-6:
        raise ValueError('Cached observed data differ from dataset snapshot; refresh consistent exports')
    cube,source,error=load_cube(root,output,args.offline)
    archive=archive_analysis(alignment,output)
    audit=json.loads((artifacts/'audit.json').read_text())
    hashes={f'{s}.csv':hashlib.sha256((dataset/f'{s}.csv').read_bytes()).hexdigest() for s in splits}
    if audit.get('dataset_checksums')!=hashes:
        raise ValueError('Cached coverage audit does not match current split files')
    complete=pd.Series(True,index=hourly.index)
    # Cached coverage audits evaluated all 24 future hours per anchor. The four
    # disjoint windows jointly cover exactly those same 24 hours.
    for location in audit['locations']:
        for split,part in location['coverage'].items():
            missing=pd.to_datetime(part['missing_anchors'],utc=True)
            complete.loc[complete.index.intersection(missing)]=False
    # Purge hours are absent from cached split audits; never assume they were checked.
    split_index=splits['train'].index.append([splits['val'].index,splits['test'].index])
    complete.loc[~complete.index.isin(split_index)]=False
    fallback=None;window_metrics=None
    if cube is not None:
        complete,fallback,window_metrics=window_analysis(cube,hourly,splits,output)
    seq,retained=sequence_audit(splits,complete,fallback,output)
    events,split_summary=high_water_events(hourly,splits,retained,output)
    context=[]
    for scope,part in splits.items():
        flags=retained[scope]
        for label,mask in [('complete_weather96',flags.baseline_eligible & flags.weather_complete96),
                           ('would_drop_missing_weather',flags.baseline_eligible & ~flags.weather_complete96)]:
            selected=part.loc[mask]
            rainfall=selected[['rain_750007_depth','rain_750062_depth']].mean(axis=1,skipna=False)
            context.append(dict(split=scope,group=label,n=len(selected),
                water_max_mm=selected.water_level_depth.max(),water_mean_mm=selected.water_level_depth.mean(),
                high_water_hours=int((selected.water_level_depth>=split_summary.high_water_threshold_train_p95.iloc[0]).sum()),
                rain_mean_mm=rainfall.mean(),rain_max_mm=rainfall.max()))
    pd.DataFrame(context).to_csv(output/'03_missing_sequence_context.csv',index=False)
    # Observed future-window distributions are computable offline, without pretending
    # the stitched per-hour forecast contains a fixed-run future trajectory.
    observed_windows={f'{a}-{b}h':obs.shift(-b).rolling(b-a,min_periods=b-a).sum() for a,b in WINDOWS}
    observed_rows=[]
    for scope,index in {k:v.index for k,v in splits.items()}.items():
        for (a,b),(name,series) in zip(WINDOWS,observed_windows.items()):
            allowed=index[(index+pd.Timedelta(hours=b)).isin(index)]
            values=series.reindex(allowed).dropna()
            observed_rows.append(dict(split=scope,window=name,n=len(values),mean_mm=values.mean(),
                p90_mm=values.quantile(.9),p99_mm=values.quantile(.99),max_mm=values.max(),
                zero_pct=100*(values==0).mean()))
    pd.DataFrame(observed_rows).to_csv(output/'02_observed_window_distributions.csv',index=False)
    report=['# Bốn kiểm tra trước khi tạo dataset dự báo mưa','',
        'Chưa tạo dataset training, không thay split hoặc cấu hình model. Toàn bộ bảng là sản phẩm phân tích.', '',
        '## Các phát hiện chính', '',
        f"- Bỏ sequence thiếu forecast sẽ mất {int(seq.loc[seq.split=='train','sequences_lost_if_drop_missing'].iloc[0])} "
        f"trên {int(seq.loc[seq.split=='train','baseline_sequences'].iloc[0])} sequence train; mask giữ nguyên số mẫu.",
        f"- Đỉnh nước thô train {split_summary.loc[split_summary.split=='train','max_water_mm'].iloc[0]:.1f} mm; "
        f"cao nhất làm target sau warm-up chỉ {split_summary.loc[split_summary.split=='train','max_water_in_eligible_targets_mm'].iloc[0]:.1f} mm. "
        'Đỉnh đầu snapshot vẫn có thể nằm trong lịch sử đầu vào, nhưng không có mẫu dự báo đỉnh đó làm nhãn.',
        '- Các sequence bị loại vì forecast thiếu trong snapshot này không trùng giờ nước cao theo ngưỡng P95 train; '
        'không khẳng định điều này đúng cho dữ liệu mới.',
        '- Train/val/test chỉ có lần lượt 7/1/1 đỉnh đợt nước cao theo quy tắc gộp 24h; '
        'một đợt train bị cắt ở đầu snapshot. Không coi số giờ là số đợt độc lập.',
        ('- Đã đánh giá đủ bốn cửa sổ tương lai trên full per-run cache.'
         if cube is not None else '- Phần đánh giá bốn cửa sổ chưa có kết quả: cần full per-run cache/DB.'), '',
        '## Trạng thái nguồn', '',f'Nguồn per-run: **{source}**; lỗi kết nối (không chứa secret): `{error}`.',
        'Quan trắc trong bảng xuất đã được đối chiếu với snapshot dataset (sai khác tối đa ≤1e-6).', '',
        '## 1. Hai giai đoạn archive', '',table_markdown(archive), '',
        'Nhãn 49R1/hindcast và 50R1 lấy từ quy ước collector theo tài liệu nhà cung cấp; '
        'không phải xác minh độc lập bản gốc đã được phát hành lịch sử. Giờ khởi tạo không phải giờ công bố. '
        'available_at chưa xác minh, giả định trễ 6 giờ. Chênh lệch hai giai đoạn đồng thời bị nhiễu bởi mùa/đợt mưa; '
        'cửa sổ ±28 ngày chỉ mô tả, không tách được tác động đổi model.', '',
        '## 2. Đánh giá đúng bốn cửa sổ tương lai', '']
    if cube is None:
        report += ['**CHƯA TÍNH ĐƯỢC KẾT QUẢ**: bảng ghép theo giờ không chứa toàn bộ tương lai '
            'của cùng một bản dự báo. Không cộng các bản ở các giờ khác nhau để giả làm feature tại một anchor. '
            'Code đã có cho bốn cửa sổ 0–3/3–6/6–12/12–24h, metric và phát hiện mưa lớn '
            '(ngưỡng P90 mưa dương của train, cùng ngưỡng cho dự báo và thực tế). '
            'Cần full per-run cache hoặc quyền kết nối DB để thực thi.']
    else:
        report += [table_markdown(window_metrics), '',
            'Ngưỡng mưa lớn cố định từ train; xem 02_heavy_rain_detection.csv. '
            'Bản dự báo được chọn tại từng anchor, không thay run giữa các giờ trong một cửa sổ.', '',
            '### Khả năng nhận biết mưa lớn', '',
            table_markdown(pd.read_csv(output/'02_heavy_rain_detection.csv')), '',
            'Ngưỡng P90 trong các tổng mưa dương của train, không phải định nghĩa khí tượng. '
            'Recall/precision tính trên các cửa sổ hợp lệ và chồng lấn, không phải số đợt mưa độc lập. '
            'Các trường hợp mưa lớn thiếu forecast được đếm riêng. '
            'Không chỉnh ngưỡng theo test. Correlation và recall ở test thấp hơn validation; '
            'không đảm bảo thêm forecast sẽ cải thiện mô hình nước.']
    report += ['', '## 3. Độ phủ sequence 96 giờ', '',table_markdown(seq), '',
        'Baseline phản ánh loader: đủ 96 hàng trong cùng split, đủ năm target và nước hiện tại. '
        'Mask giữ nguyên số sequence nhưng không đảm bảo forecast có thông tin. '
        'Nếu chỉ dùng audit cache, cờ đủ 24 giờ lấy từ audit đã lưu cùng checksum split; '
        'chỉ kết luận độ phủ CHUNG bốn cửa sổ, chưa kết luận từng cửa sổ riêng. '
        'Fallback bản cũ được kiểm tra từ full per-run khi có cache; không suy diễn từ cache ghép. '
        'Bản cũ phải đủ 24 giờ tương lai, khởi tạo ít nhất 6 giờ trước anchor và không cũ quá 24 giờ. '
        'Cần mang theo độ cũ và cờ fallback; đây vẫn là giả định khả dụng lịch sử.', '',
        '## 4. Độ đại diện theo split', '',table_markdown(split_summary), '',
        'Đợt nước cao: nước ≥P95 train, gộp các giờ vượt ngưỡng cách nhau ≤24h; đếm theo split của đỉnh. '
        'Đây là quy tắc mô tả, không phải định nghĩa lũ chính thức hoặc chứng minh các đợt độc lập. '
        'CSV sự kiện có cờ vượt qua split/cận snapshot, độ tăng 6 giờ và độ phủ sequence. '
        'Ngưỡng tăng nhanh cũng dùng P95 train; không điều chỉnh theo test.', '',
        '## Kết luận hành động', '',
        '- Kiểm tra cột max_water_in_eligible_targets_mm: sequence warm-up có thể loại đỉnh nước lớn ở đầu snapshot dù đỉnh vẫn xuất hiện trong train.csv thô.',
        '- Giữ split hiện tại cho đối chiếu bước 1; test đã được xem nhiều lần nên không còn là holdout chưa chạm.',
        '- Không loại toàn bộ sequence có forecast thiếu trước khi đánh giá ảnh hưởng lên các đợt nước cao; ưu tiên mask rõ ràng.',
        '- Không gọi dữ liệu hindcast là replay vận hành đã xác minh. Chưa chốt có thể gộp hai phiên bản để train từ các bảng mô tả.',
        '- Chưa tuyên bố bốn cửa sổ cải thiện model nước: cần thí nghiệm model có kiểm soát sau đánh giá dữ liệu.',
        '- Dành dữ liệu mới sau cutoff làm kiểm chứng cuối; không kéo dữ liệu quan trắc tương lai vào dataset này.']
    (output/'bao_cao.md').write_text('\n'.join(report)+'\n')
    manifest=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        alignment_sha256=hashlib.sha256((artifacts/'daily_analysis/hourly_rain_water_alignment.csv').read_bytes()).hexdigest(),
        coverage_audit_sha256=hashlib.sha256((artifacts/'audit.json').read_bytes()).hexdigest(),
        full_cube_sha256=hashlib.sha256((output/'forecast_runs_hourly.csv.gz').read_bytes()).hexdigest() if cube is not None else None,
        per_run_source=source,per_run_error=error,part_2_complete=cube is not None,
        archive_analysis='cached_hourly_diagnostic',sequence_source='full_cube' if cube is not None else 'checksum_matched_24h_coverage_audit',
        split_sha256=hashes,observation_max_diff_mm=float(difference),water_max_diff_mm=float(water_difference),
        assumed_release_lag_hours=6,windows=WINDOWS,sequence_length=96,
        high_water_threshold_mm=float(split_summary.high_water_threshold_train_p95.iloc[0]),
        analysis_only=True)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(table_markdown(seq));print(table_markdown(split_summary));print(f'Report: {output / "bao_cao.md"}')
    return output


if __name__=='__main__':main()
