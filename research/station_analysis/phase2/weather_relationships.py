"""Descriptive forecast-water relationships; no predictive feature ablation."""
import numpy as np
import pandas as pd
from .weather_metrics import correlations,partial_correlations


def assess_relationships(selected, targets, rainfall, candidates, stations, cutoff):
    rows=[];partial=[];control_rows=[];missing=[]
    boundary=pd.Timestamp(cutoff)
    names=stations.set_index('station_id').name.to_dict()
    # One location remains one source view; never pool duplicate-grid requests as independent samples.
    legacy=selected[selected.policy.eq('legacy_assumed_release')]
    columns=['rain_0_3h_mm','rain_3_6h_mm','rain_6_12h_mm','rain_12_24h_mm']
    for wid,target in targets.items():
        idx=target.index
        train=idx+pd.Timedelta(hours=12)<=boundary
        evaluation=idx>boundary
        rank=candidates[candidates.water_station_id.astype(str).eq(wid)].sort_values('best_corr_dh12',ascending=False)
        rain_ids=[str(s) for s in rank.rain_station_id if str(s) in rainfall.columns][:2]
        controls=pd.DataFrame({'water_now':target.water_level_mm,'trend3':target.dh_m3h,
                               'trend6':target.dh_m6h},index=idx)
        for sid in rain_ids:
            controls[f'rain_{sid}_6h']=rainfall[sid].rolling(6,min_periods=6).sum()
            controls[f'rain_{sid}_12h']=rainfall[sid].rolling(12,min_periods=12).sum()
        control_rows.append({'water_station_id':wid,'water_station_name':names.get(wid,wid),
                             'rain_ids_from_train_candidates':';'.join(rain_ids),
                             'controls':';'.join(controls.columns), 'train_valid_controls':int(controls.loc[train].notna().all(axis=1).sum())})
        for location,part in legacy.groupby('location_id'):
            feature=part.set_index('anchor_utc').reindex(idx)
            # Confirm forecast is one row per anchor before reindexing in the caller.
            age=feature.age_hours
            for horizon in [6,12]:
                y=target[f'dh_p{horizon}h']
                for split,mask in [('train',train),('evaluation',evaluation)]:
                    for available in [True,False]:
                        use=mask & feature[columns].notna().all(axis=1).eq(available)
                        valid=y.loc[use].dropna()
                        missing.append({'water_station_id':wid,'location_id':location,'horizon_h':horizon,
                                        'scope':split,'complete_weather':available,'n':len(valid),
                                        'mean_abs_actual_change_mm':float(valid.abs().mean()),
                                        'abs_actual_ge200_n':int(valid.abs().ge(200).sum())})
                for col in columns+['age_hours']:
                    x=feature[col]
                    for split,mask in [('train',train),('evaluation',evaluation)]:
                        conditions={'all':np.ones(len(idx),dtype=bool),
                                    'flood_months':idx.tz_convert('Asia/Ho_Chi_Minh').month.isin([9,10,11,12]),
                                    'past_rising':target.dh_m3h.gt(0),
                                    'future_abs_change_ge200_retrospective':y.abs().ge(200)}
                        for kind in sorted(feature.selected_archive_kind.dropna().unique()):
                            conditions[f'archive_kind:{kind}']=feature.selected_archive_kind.eq(kind)
                        for condition,cond in conditions.items():
                            n,r,s=correlations(x.loc[mask&cond],y.loc[mask&cond])
                            rows.append({'water_station_id':wid,'water_station_name':names.get(wid,wid),
                                         'station_subtype':stations.set_index('station_id').loc[wid,'station_subtype'],
                                         'location_id':location,'horizon_h':horizon,'scope':split,
                                         'condition':condition,'feature':col,'n':n,'pearson_r':r,'spearman_r':s})
                    if len(rain_ids)==2:
                        cohorts={'all_assumed_release':np.ones(len(idx),dtype=bool)}
                        cohorts.update({str(kind):feature.selected_archive_kind.eq(kind).to_numpy()
                                        for kind in feature.selected_archive_kind.dropna().unique()})
                        for cohort,cohort_mask in cohorts.items():
                            for result in partial_correlations(x,y,controls,train&cohort_mask,evaluation&cohort_mask):
                                partial.append({'water_station_id':wid,'water_station_name':names.get(wid,wid),
                                                'location_id':location,'horizon_h':horizon,'feature':col,
                                                'cohort':cohort,'rain_control_ids':';'.join(rain_ids),**result})
    return {'weather_water_correlations':pd.DataFrame(rows),
            'weather_partial_correlations':pd.DataFrame(partial,columns=['water_station_id','water_station_name','location_id','horizon_h','feature','cohort','rain_control_ids','scope','train_fit_n','n','partial_pearson_r','partial_spearman_r']),
            'weather_controls':pd.DataFrame(control_rows),
            'weather_missingness_water':pd.DataFrame(missing)}
