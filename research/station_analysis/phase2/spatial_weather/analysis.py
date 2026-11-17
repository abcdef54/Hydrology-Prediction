"""Spatial diagnostics and train-only candidate design; no predictive training."""
from itertools import combinations
import numpy as np
import pandas as pd

from ..weather_metrics import correlations,partial_correlations,rain_metrics
from ..weather_quality import haversine_km

WINDOWS=['rain_0_3h_mm','rain_3_6h_mm','rain_6_12h_mm','rain_12_24h_mm']
LENGTHS=[3,3,6,12]


def intergrid_statistics(synchronized,grids,cutoff):
    frame=synchronized.set_index('anchor_utc')
    idx=frame.index;boundary=pd.Timestamp(cutoff)
    rows=[];spread=[]
    for col,length in zip(WINDOWS,LENGTHS):
        matrix=frame[[f'{g}|{col}' for g in grids.grid_id]]
        spread.append(pd.DataFrame({'anchor_utc':idx,'feature':col,
                                    'spatial_std_mm':matrix.std(axis=1).to_numpy(),
                                    'spatial_range_mm':(matrix.max(axis=1)-matrix.min(axis=1)).to_numpy(),
                                    'heavy_grid_count':matrix.ge(5*length).sum(axis=1).to_numpy()}))
        for scope,base in [('train',idx+pd.Timedelta(hours=12)<=boundary),('evaluation',idx>boundary)]:
            for season,season_mask in [('all',np.ones(len(idx),dtype=bool)),('flood_months',idx.tz_convert('Asia/Ho_Chi_Minh').month.isin([9,10,11,12])),('other_months',~idx.tz_convert('Asia/Ho_Chi_Minh').month.isin([9,10,11,12]))]:
                for a,b in combinations(grids.grid_id,2):
                    x=frame[f'{a}|{col}'];y=frame[f'{b}|{col}'];mask=base&season_mask
                    n,r,s=correlations(x.loc[mask],y.loc[mask])
                    ha=x.loc[mask].ge(5*length);hb=y.loc[mask].ge(5*length)
                    union=int((ha|hb).sum())
                    rows.append({'grid_a':a,'grid_b':b,'scope':scope,'season':season,'feature':col,
                                 'provenance':'assumed_release_6h; archive/hindcast mixture by calendar',
                                 'n':n,'pearson_r':r,'spearman_r':s,'heavy_both':int((ha&hb).sum()),
                                 'heavy_only_a':int((ha&~hb).sum()),'heavy_only_b':int((~ha&hb).sum()),
                                 'heavy_jaccard':int((ha&hb).sum())/union if union else np.nan,
                                 'mean_abs_difference_mm':float((x.loc[mask]-y.loc[mask]).abs().mean())})
    return pd.DataFrame(rows),pd.concat(spread,ignore_index=True)


def spatial_water_relationships(synchronized,grids,targets,rain,candidates,stations,cutoff,baseline_grid):
    weather=synchronized.set_index('anchor_utc')
    raw=[];partial=[];controls_meta=[];event=[]
    boundary=pd.Timestamp(cutoff)
    names=stations.set_index('station_id').name.to_dict()
    for wid,target in targets.items():
        idx=target.index
        train=idx+pd.Timedelta(hours=12)<=boundary;evaluation=idx>boundary
        rank=candidates[candidates.water_station_id.astype(str).eq(wid)&candidates.best_corr_dh12.notna()].sort_values('best_corr_dh12',ascending=False)
        ids=rank.rain_station_id.astype(str).head(2).tolist()
        controls=pd.DataFrame({'water_now':target.water_level_mm,'trend3':target.dh_m3h,'trend6':target.dh_m6h},index=idx)
        for sid in ids:
            for hours in [6,12]:controls[f'rain_{sid}_{hours}h']=rain[sid].rolling(hours,min_periods=hours).sum()
        # Additional-to-old-grid association is evaluated on the SAME synchronized anchors.
        baseline=controls.copy()
        for col in WINDOWS:baseline[f'old_grid_{col}']=weather[f'{baseline_grid}|{col}'].reindex(idx)
        controls_meta.append({'water_station_id':wid,'rain_control_ids':';'.join(ids),'controls':';'.join(controls.columns),
                              'baseline_grid':baseline_grid,'train_controls_n':int(controls.loc[train].notna().all(axis=1).sum())})
        for gid in grids.grid_id:
            for col in WINDOWS:
                x=weather[f'{gid}|{col}'].reindex(idx)
                for target_col,horizon in [('water_level_mm',0),('dh_p6h',6),('dh_p12h',12)]:
                    y=target[target_col]
                    for scope,mask in [('train',train),('evaluation',evaluation)]:
                        for condition,cond in [('all',pd.Series(True,index=idx)),
                                               ('flood_months',pd.Series(idx.tz_convert('Asia/Ho_Chi_Minh').month.isin([9,10,11,12]),index=idx)),
                                               ('past_rising',target.dh_m3h.gt(0)),
                                               ('future_abs_change_ge200_retrospective',y.abs().ge(200) if horizon else pd.Series(False,index=idx))]:
                            n,r,s=correlations(x.loc[mask&cond],y.loc[mask&cond])
                            raw.append({'water_station_id':wid,'water_station_name':names.get(wid,wid),
                                        'provenance':'assumed_release_6h; same initialization across grids',
                                        'grid_id':gid,'feature':col,'target':target_col,'horizon_h':horizon,
                                        'scope':scope,'condition':condition,'n':n,'pearson_r':r,'spearman_r':s})
                    if horizon and len(ids)==2:
                        for label,design in [('water_and_observed_rain',controls),('plus_existing_a_luoi',baseline)]:
                            if label=='plus_existing_a_luoi' and gid==baseline_grid:continue
                            for result in partial_correlations(x,y,design,train,evaluation):
                                partial.append({'water_station_id':wid,'grid_id':gid,'feature':col,'horizon_h':horizon,
                                                'provenance':'assumed_release_6h; mixed archive periods',
                                                'control_set':label,'rain_control_ids':';'.join(ids),**result})
    return pd.DataFrame(raw),pd.DataFrame(partial),pd.DataFrame(controls_meta)


def event_rain_skill(pairs,episodes,wet_mm=.1,heavy_rate=5):
    rows=[]
    source=pairs[pairs.observation_policy.eq('standard_5of6')].copy()
    endpoints={'0_3h':3,'3_6h':6,'6_12h':12,'12_24h':24}
    source['_interval_end']=source.anchor_utc+pd.to_timedelta(source.window.map(endpoints),unit='h')
    for ep in episodes.itertuples():
        start=pd.Timestamp(ep.start_utc);end=pd.Timestamp(ep.end_utc)
        use=(source.anchor_utc>=start)&(source['_interval_end']<=end)
        for keys,part in source.loc[use].groupby(['location_id','gauge_id','window','scope','archive_kind']):
            gid,sid,window,scope,kind=keys
            if scope=='purge':continue
            n_hours=int(window.split('_')[1][:-1])-int(window.split('_')[0])
            stats=rain_metrics(part.forecast_mm,part.observed_mm,wet_mm=wet_mm,heavy_mm=heavy_rate*n_hours)
            if stats['n']<6:continue
            rows.append({'episode':int(ep.episode),'start_utc':str(start),'end_utc':str(end),
                         'grid_id':gid,'gauge_id':sid,'window':window,'scope':scope,'archive_kind':kind,**stats})
    return pd.DataFrame(rows)


def recommendations(grids,stations,candidates,partial,coverage,cutoff):
    water=stations[stations.station_type.eq('water_level')]
    rain=stations[stations.station_type.eq('rainfall')].set_index('station_id')
    baseline=grids.loc[(grids.grid_latitude-16.274164).abs().lt(1e-5)&(grids.grid_longitude-107.264960).abs().lt(1e-5),'grid_id'].iloc[0]
    recommendations=[]
    scores=partial[partial.scope.eq('train')&partial.control_set.eq('plus_existing_a_luoi')].copy() if len(partial) else pd.DataFrame()
    for station in water.itertuples():
        distances=haversine_km(station.latitude,station.longitude,grids.grid_latitude,grids.grid_longitude)
        local=grids.iloc[int(np.argmin(distances))].grid_id
        selected=candidates[candidates.water_station_id.astype(str).eq(station.station_id)&candidates.best_corr_dh12.notna()].sort_values('best_corr_dh12',ascending=False).head(3)
        upstream=[]
        for sid in selected.rain_station_id.astype(str):
            d=haversine_km(rain.loc[sid,'latitude'],rain.loc[sid,'longitude'],grids.grid_latitude,grids.grid_longitude)
            gid=grids.iloc[int(np.argmin(d))].grid_id
            if gid not in upstream:upstream.append(gid)
        # Complete a second spatial candidate only using train associations, never evaluation.
        if len(scores):
            rank=scores[scores.water_station_id.eq(station.station_id)].assign(score=lambda d:d.partial_pearson_r.abs()).groupby('grid_id').score.mean().dropna().sort_values(ascending=False)
            for gid in rank.index:
                if len(upstream)>=2:break
                if gid not in upstream:upstream.append(gid)
        upstream=upstream[:3]
        for arm,ids,reason in [('W0',[],'Không dùng dự báo thời tiết'),('W1',[baseline],'Ô A Lưới hiện tại; đối chứng nguồn cũ'),
                               ('W2',[local],'Ô gần trạm nước nhất trong tập đã thu thập; chưa mặc định là đúng ô API trả cho tọa độ trạm'),
                               ('W3',upstream,'Ánh xạ các trạm mưa tuyển chọn trên train sang ô dự báo; bổ sung ứng viên theo liên hệ train nếu cần'),
                               ('W4',list(dict.fromkeys([local]+upstream)),'Kết hợp ô gần trạm và các ô liên hệ mưa trên train')]:
            recommendations.append({'water_station_id':station.station_id,'water_station_name':station.name,'arm':arm,
                                    'grid_ids':';'.join(ids),'grid_count':len(ids),'continuous_forecast_features':4*len(ids),
                                    'missing_mask_features':4*len(ids),'age_features':1 if ids else 0,
                                    'provenance_metadata_features':len(ids),'local_proxy_distance_km':float(np.min(distances)),
                                    'reason':reason,'confidence':'Tentative; Stage 4 chưa thực hiện',
                                    'selection_scope':'train only','cutoff':str(cutoff)})
    return pd.DataFrame(recommendations)


def same_gauge_comparisons(synchronized,comparators,rain,cutoff):
    """Compare nearby cells against identical gauge/window/anchors and initialization."""
    from ..weather_metrics import future_amount
    forecast=synchronized.set_index('anchor_utc');idx=forecast.index
    boundary=pd.Timestamp(cutoff);rows=[]
    memberships={gid:set(part.gauge_id) for gid,part in comparators.groupby('location_id')}
    for a,b in combinations(memberships,2):
        for sid in sorted(memberships[a]&memberships[b]):
            for col,(left,right) in zip(WINDOWS,[(0,3),(3,6),(6,12),(12,24)]):
                truth=future_amount(rain[sid],left,right).reindex(idx)
                p=forecast[f'{a}|{col}'];q=forecast[f'{b}|{col}']
                for scope,mask in [('train',idx+pd.Timedelta(hours=24)<=boundary),('evaluation',idx>boundary)]:
                    for version,cohort in [('49r1',forecast.initialized_at<pd.Timestamp('2026-05-12T06:00Z')),
                                           ('50r1',forecast.initialized_at>=pd.Timestamp('2026-05-12T06:00Z'))]:
                        use=mask&cohort&truth.notna()
                        x=rain_metrics(p.loc[use],truth.loc[use],heavy_mm=5*(right-left))
                        y=rain_metrics(q.loc[use],truth.loc[use],heavy_mm=5*(right-left))
                        if not x['n']:continue
                        rows.append({'grid_a':a,'grid_b':b,'gauge_id':sid,'feature':col,'scope':scope,
                                     'model_version':version,'n':x['n'],'rmse_a_mm':x['rmse_mm'],
                                     'rmse_b_mm':y['rmse_mm'],'delta_rmse_b_minus_a_mm':y['rmse_mm']-x['rmse_mm'],
                                     'mae_a_mm':x['mae_mm'],'mae_b_mm':y['mae_mm'],
                                     'availability':'assumed 6h; not verified operational history'})
    return pd.DataFrame(rows)
