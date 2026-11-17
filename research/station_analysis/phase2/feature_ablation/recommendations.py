"""Regional input recommendations derived from paired validation, not a test-selected winner."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def recommendation_matrix(work,seed,incremental):
    rows=[]
    for sd in sorted((Path(work)/'stations').glob('*')):
        if not (sd/'complete.json').exists():continue
        sid=sd.name;discovery=json.loads((sd/'discovery.json').read_text())
        finalists=json.loads((sd/'finalists.json').read_text())['arms']
        metrics=pd.read_csv(sd/'metrics.csv')
        fold_count=metrics[metrics.scope.eq('validation')].fold.nunique()
        for horizon in [6,12]:
            for group,label in [('observed_rain','observed_rain_added'),('weather','weather_added_to_observed_rain'),('selected_observed_rain','train_selected_observed_rain'),('challenger_weather','weather_added_to_observed_rain')]:
                pool=incremental[(incremental.station_id==sid)&incremental.scope.eq('validation')&incremental.horizon.eq(horizon)&incremental.diagnostic.eq(label)]
                if group=='weather':pool=pool[pool.arm.isin(finalists)&pool.arm.ne('W0')]
                if group=='challenger_weather':pool=pool[pool.arm.isin(['C1','C2'])]
                if pool.empty:
                    rows.append(dict(station_id=sid,horizon=horizon,feature_group=group,status='Chưa có ứng viên forecast được validation chọn',
                        selected_arm='W0',rain_ids=';'.join(discovery['fixed_rain']),mean_delta_rmse_mm=np.nan,mean_delta_n200=np.nan,
                        seed_win_fraction=np.nan,fold_count=fold_count,confidence='Không suy ra feature vô ích cho LSTM'))
                    continue
                rank=pool.groupby('arm').agg(n200=('delta_n200','mean'),rmse=('delta_rmse_mm','mean')).sort_values(['n200','rmse'])
                arm=rank.index[0];part=pool[pool.arm.eq(arm)]
                drm=part.delta_rmse_mm.mean();dn=part.delta_n200.mean();wins=part.delta_rmse_mm.lt(0).mean()
                q=metrics[metrics.scope.eq('validation')&metrics.horizon.eq(horizon)&metrics['mode'].eq('normal')&metrics.lag.eq(6)]
                if 'version' in q:q=q[q.version.isna()]
                after_e={'observed_rain':'E1','weather':'E3','selected_observed_rain':'E5','challenger_weather':'E3'}[group]
                before_e={'observed_rain':'E0','weather':'E1','selected_observed_rain':'E3','challenger_weather':'E1'}[group]
                after=q[q.experiment.eq(after_e)&q.arm.eq(arm)][['fold','seed','rmse_mm']]
                before=q[q.experiment.eq(before_e)&q.arm.eq(arm if group=='selected_observed_rain' else 'W0')][['fold','seed','rmse_mm']]
                paired=after.merge(before,on=['fold','seed'],suffixes=('_a','_b'),validate='one_to_one')
                fold_wins=float(paired.assign(delta=paired.rmse_mm_a-paired.rmse_mm_b).groupby('fold').delta.mean().lt(0).mean()) if len(paired) else np.nan
                if group=='selected_observed_rain' and sorted(discovery['selected_rain'])==sorted(discovery['fixed_rain']):
                    status='Schema trùng đối chứng; tái sử dụng cùng fit';confidence='Không có thay đổi feature để diễn giải'
                elif drm<0 and dn<=0 and wins>=.8 and fold_wins>=2/3:

                    status='Ứng viên có lợi trên validation' if fold_count>=2 else 'Có lợi trên một fold; chưa đánh giá ổn định thời gian'
                    confidence='Cần kiểm chứng qua thêm sự kiện và dữ liệu mới' if fold_count>=2 else 'Chỉ có một fold; chưa xác nhận ổn định thời gian'
                elif drm>=0 and dn>=0 and wins<=.2:
                    status='Chưa khuyến nghị trong schema chung từ bằng chứng XGBoost này';confidence='Không phải bằng chứng LSTM không thể khai thác'
                else:status='Kết quả hỗn hợp; cần giữ nhánh ablation';confidence='Có đánh đổi RMSE/N200 hoặc khác biệt giữa seed'
                rows.append(dict(station_id=sid,horizon=horizon,feature_group=group,status=status,selected_arm=arm,
                    rain_ids=';'.join(discovery['selected_rain'] if group=='selected_observed_rain' else discovery['fixed_rain']),mean_delta_rmse_mm=drm,mean_delta_n200=dn,
                    seed_win_fraction=wins,fold_win_fraction=fold_wins,fold_count=fold_count,confidence=confidence))
    result=pd.DataFrame(rows)
    grid_mapping={}
    for sd in sorted((Path(work)/'stations').glob('*')):
        if not (sd/'complete.json').exists():continue
        for r in json.loads((sd/'ledger.json').read_text()):
            if r.get('status')=='completed' and r.get('mode')=='normal' and r.get('lag',6)==6 and r.get('version') is None:
                grid_mapping[(sd.name,r['arm'])]=';'.join(r['grids'])
    result['grid_ids']=[grid_mapping.get((r.station_id,r.selected_arm),'') for r in result.itertuples()]
    return result


def schema_candidates(matrix):
    return {
        'status':'recommendations_only_no_regional_dataset_or_training',
        'common_history':{'timesteps':96,'water_level':'station-specific observed level, past only',
                          'water_missingness':'retain missingness; do not treat missing as zero',
                          'normalization':'station-specific train-only scaling must preserve residual persistence coefficients'},
        'candidates':[
            {'id':'R0','features':['water_history','station_identity'],'role':'regional baseline'},
            {'id':'R1','features':['water_history','observed_rain_slot1','observed_rain_slot2','rain_missing_masks','station_identity'],
             'role':'compare fixed/local versus train-selected rain mapping, not one fixed city-wide pair'},
            {'id':'R2','features':['R1','forecast_slot1','forecast_missing_masks','shared_forecast_age'],
             'role':'single-grid candidate; grid selection remains station-specific'},
            {'id':'R3','features':['R1','up_to_three_forecast_slots','slot_presence_masks','forecast_missing_masks','shared_forecast_age'],
             'role':'station-specific multi-grid candidate only where validation supports additional information'},
        ],
        'rain_history_windows_h':[1,3,6,12,24],
        'forecast_windows_h':[[0,3],[3,6],[6,12],[12,24]],
        'targets_h':[6,12],
        'weather_history_policy':'anchor_only in this tree study; keep explicit in regional design and validate before switching to all_timesteps',
        'spatial_metadata':'grid/station coordinates and DEM are metadata; local XGBoost cannot establish predictive value of station-constant static features',
        'not_proven':['regional weight sharing','station embeddings','causal runoff paths','operational publication chronology','LSTM performance'],
        'recommendation_matrix':'regional_feature_recommendation_matrix.csv',
        'evidence_policy':'paired pre-cutoff chronological validation; post-cutoff is descriptive only',
    }
