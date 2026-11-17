"""Stage 4 evidence pack, Vietnamese interpretation, regional schemas, and full manifests."""
from pathlib import Path
import json,logging,os
import numpy as np
import pandas as pd
from .data import Source,sha
from .training import write_json
from .evaluation import gather,aggregate_seeds,paired_incremental,evaluate_strata,factorial_interactions,KEYS,persistence_baselines
from .explanations import grouped_gain,explain_finalists
from .uncertainty import event_bootstrap
from .rain_events import evaluate as evaluate_rain_episodes
from .recommendations import recommendation_matrix,schema_candidates
from .patterns import diagnostic_effects,paired_strata,feature_windows,predicted_change_summary,target_variability
LOG=logging.getLogger(__name__)


def table(frame):
    headers=frame.columns.tolist();rows=[]
    def fmt(v):
        if isinstance(v,(float,np.floating)):return '—' if not np.isfinite(v) else f'{v:.3f}'
        return str(v).replace('|','/').replace('\n',' ')
    for r in frame.itertuples(index=False,name=None):rows.append('| '+' | '.join(fmt(v) for v in r)+' |')
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',*rows])


def diagnostic_coverage(ledger,eligible,completed_stations=None):
    rows=[]
    mapping={'D1':ledger['mode'].eq('availability'),'D2':ledger['mode'].eq('duplicate'),'D3':ledger.experiment.eq('E4'),
        'D4':ledger.experiment.isin(['E0','E1','E2','E3']),'D5':ledger.experiment.eq('E5'),
        'D6':ledger['mode'].eq('strict'),'D7':ledger.lag.isin([9,12]),'D8':ledger.diagnostic.eq('D8'),
        'D9':ledger.diagnostic.eq('D9')}
    for name,mask in mapping.items():
        for sid in eligible:
            part=ledger[mask&ledger.station_id.eq(sid)]
            completed=part[part.status.eq('completed')]
            if completed_stations is not None and sid not in completed_stations:status='Đang chạy; chưa kết luận phạm vi'
            elif len(completed) and part.status.eq('not_estimable').any():status='Đã thực hiện phần đủ dữ liệu; còn nhánh không ước lượng được'
            elif len(completed):status='Đã thực hiện'
            elif len(part) and part.status.eq('not_estimable').any():status='Không đủ điều kiện dữ liệu'
            else:status='Không kích hoạt theo điều kiện đã chốt'
            rows.append(dict(station_id=sid,diagnostic=name,status=status,completed_seed_aliases=len(completed),
                unique_fits=completed.fit_key.nunique(),not_estimable_branches=int(part.status.eq('not_estimable').sum()),reason='; '.join(part.get('reason',pd.Series(dtype=str)).dropna().astype(str).unique())))
    return pd.DataFrame(rows)


def plots(output,summary,incremental,matrix,interactions,permutation):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=output/'figures';figures.mkdir(exist_ok=True)
    station_names=summary.drop_duplicates('station_id').set_index('station_id').station_name.to_dict()
    base=summary[(summary.station_id=='0074647403')&summary.experiment.eq('E3')&summary['mode'].eq('normal')&summary.lag.eq(6)&summary.version.eq('all')&summary.arm.str.startswith('W')]
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for row,scope in enumerate(['validation','exploratory']):
        for h in [6,12]:
            p=base[base.scope.eq(scope)&base.horizon.eq(h)].sort_values('arm')
            axes[row,0].plot(p.arm,p.rmse_mean,marker='o',label=f'+{h}h')
            axes[row,1].plot(p.arm,p.n200_mean,marker='o',label=f'+{h}h')
        axes[row,0].set_ylabel('RMSE trung bình 5 seed (mm)');axes[row,1].set_ylabel('Số lỗi >200 mm trung bình 5 seed')
        for ax in axes[row]:ax.legend();ax.set_title('Validation trước cutoff' if scope=='validation' else 'Đánh giá thăm dò sau cutoff');ax.grid(alpha=.2)
    fig.suptitle('Dã Viên: đầy đủ W0–W7, cùng mốc dự báo trong mỗi phạm vi');fig.tight_layout();fig.savefig(figures/'01_da_vien_factorial.png',dpi=150);plt.close(fig)
    p=incremental[incremental.scope.eq('validation')&incremental.diagnostic.eq('observed_rain_added')]
    pivot=p.groupby(['station_id','horizon']).delta_rmse_mm.mean().unstack('horizon')
    fig,ax=plt.subplots(figsize=(9,7));im=ax.imshow(pivot,vmin=-10,vmax=10,cmap='RdBu_r',aspect='auto')
    ax.set_yticks(range(len(pivot)),[station_names.get(s,s) for s in pivot.index],fontsize=7);ax.set_xticks(range(len(pivot.columns)),[f'+{h}h' for h in pivot.columns]);fig.colorbar(im,ax=ax,label='ΔRMSE khi thêm mưa quan trắc (mm); âm là tốt hơn')
    ax.set_title('Giá trị bổ sung của mưa quan trắc trên validation');fig.tight_layout();fig.savefig(figures/'02_observed_rain_increment.png',dpi=150);plt.close(fig)
    p=matrix[matrix.feature_group.eq('weather')].pivot(index='station_id',columns='horizon',values='mean_delta_rmse_mm')
    fig,ax=plt.subplots(figsize=(10,7));cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#cccccc');im=ax.imshow(np.ma.masked_invalid(p.to_numpy()),vmin=-5,vmax=5,cmap=cmap,aspect='auto')
    ax.set_yticks(range(len(p)),[station_names.get(s,s) for s in p.index],fontsize=7);ax.set_xticks(range(len(p.columns)),[f'+{h}h' for h in p.columns]);fig.colorbar(im,ax=ax,label='ΔRMSE so với nước+mưa quan trắc (mm)')
    ax.set_title('Forecast được validation chọn; ô xám: chưa có ứng viên');fig.tight_layout();fig.savefig(figures/'03_weather_increment.png',dpi=150);plt.close(fig)
    p=summary[(summary.station_id=='0074647403')&summary.scope.eq('validation')&summary.experiment.eq('E3')&summary['mode'].eq('normal')&summary.version.eq('all')&summary.lag.eq(6)&summary.arm.str.startswith('W')]
    fig,ax=plt.subplots(figsize=(10,4))
    for h,part in p.groupby('horizon'):ax.errorbar(part.arm,part.rmse_mean,yerr=part.rmse_sd,marker='o',capsize=3,label=f'+{h}h')
    ax.set_ylabel('RMSE trung bình ± độ lệch chuẩn giữa seed (mm)');ax.set_title('Độ nhạy với seed tại Dã Viên; không phải khoảng tin cậy theo giờ');ax.legend();fig.tight_layout();fig.savefig(figures/'04_seed_stability.png',dpi=150);plt.close(fig)
    if len(permutation):
        p=permutation[(permutation.station_id=='0074647403')&permutation.scope.eq('validation')].groupby('group').delta_rmse_mm.mean().sort_values()
        p.index=p.index.str.replace('water_history','Lịch sử nước').str.replace('observed_rain:','Mưa trạm:').str.replace('forecast:','Ô dự báo:').str.replace('forecast_age','Tuổi dự báo').str.replace('missing_masks','Cờ thiếu')
        fig,ax=plt.subplots(figsize=(10,5));ax.barh(p.index,p.values);ax.set_xlabel('ΔRMSE sau dịch khối feature (mm)');ax.set_title('Độ nhạy feature tại Dã Viên; đây là phép can thiệp chẩn đoán hồi cứu');fig.tight_layout();fig.savefig(figures/'05_grouped_permutation.png',dpi=150);plt.close(fig)


def finalize(work,output,c):
    work=Path(work);output=Path(output)
    if os.environ.get('HYDROLOGY_STAGE4_COORDINATED_WORKER')=='1':
        marker=work/'worker_fit_complete.json'
        write_json(marker,{'status':'fitting_complete','reporting':'deferred_to_coordinator'})
        return marker
    output.mkdir(parents=True,exist_ok=True);td=output/'tables';td.mkdir(exist_ok=True)
    source=Source(c,work/'shared');metrics,ledger=gather(work)
    if metrics.empty:raise ValueError('No completed fits available')
    for col,default in [('mode','normal'),('lag',6),('diagnostic','principal'),('version','all')]:
        if col not in ledger:ledger[col]=default
        else:ledger[col]=ledger[col].fillna(default)
    eligibility=pd.DataFrame(json.loads((work/'eligibility.json').read_text()))
    eligible=eligibility[eligibility.eligible].station_id.tolist()
    complete=[p.parent.name for p in (work/'stations').glob('*/complete.json')]
    stage_complete=set(eligible)==set(complete) and len(eligibility)==len(source.water)
    seed,summary=aggregate_seeds(metrics);incremental=paired_incremental(seed)
    persistence=persistence_baselines(work)
    LOG.info('Stage 4 analysis: hydrological strata, episodes, and peaks')
    strata,peaks,catalogue=evaluate_strata(work,source)
    rain_event_metrics,rain_event_effects,rain_event_summary=evaluate_rain_episodes(work,source)
    gains=grouped_gain(work,ledger)
    LOG.info('Stage 4 analysis: fixed-seed explanations and grouped interactions')
    shap,shap_inter,permutation=explain_finalists(work,source,c)
    bootstrap=event_bootstrap(work,strata,c['bootstrap_replicates'])
    factorial=factorial_interactions(seed);matrix=recommendation_matrix(work,seed,incremental)
    coverage=diagnostic_coverage(ledger,eligible,complete)
    diagnostic_results,diagnostic_pairs=diagnostic_effects(incremental)
    stratum_pairs=paired_strata(work,strata)
    window_gains=feature_windows(work,ledger)
    change_summary=predicted_change_summary(ledger,strata)
    variability=target_variability(work,c['cutoff'])
    constant_targets=variability[variability.fold.eq('full_pre_cutoff')&variability.std_mm.lt(1e-6)]
    train_variability=variability[variability.fold.eq('full_pre_cutoff')][['station_id','horizon','std_mm','abs_change_ge200_pct']].rename(columns={'std_mm':'train_dh_std_mm','abs_change_ge200_pct':'train_abs_dh_ge200_pct'})
    matrix=matrix.merge(train_variability,on=['station_id','horizon'],how='left',validate='many_to_one')
    small_dynamics=matrix.train_dh_std_mm.lt(20)&matrix.train_dh_std_mm.ge(1e-6)&matrix.status.eq('Ứng viên có lợi trên validation')
    matrix.loc[small_dynamics,'status']='Có lợi cho biến động nhỏ; chưa có bằng chứng riêng về lũ'
    matrix.loc[small_dynamics,'confidence']='Độ lệch chuẩn target train <20 mm; đọc RMSE cùng thang biến động và số trường hợp nước dâng lớn'
    for r in constant_targets.itertuples():
        mask=matrix.station_id.eq(r.station_id)&matrix.horizon.eq(r.horizon)
        matrix.loc[mask,'status']='Không đủ biến thiên target trong train để đánh giá feature'
        matrix.loc[mask,'confidence']='Không suy ra feature vô ích khi target train hầu như cố định'
    precision_interpretation=json.loads((work/'checkpoint_precision/interpretation.json').read_text()) if (work/'checkpoint_precision/interpretation.json').exists() else None
    precision_effects=pd.read_csv(work/'checkpoint_precision/feature_effect_sensitivity.csv',dtype={'station_id':str}) if (work/'checkpoint_precision/feature_effect_sensitivity.csv').exists() else pd.DataFrame()
    precision_file=work/'checkpoint_precision/fold_comparison.csv'
    precision=pd.read_csv(precision_file,dtype={'station_id':str}) if precision_file.exists() else pd.DataFrame()
    tables={'persistence_baselines':persistence,'all_fold_metrics':metrics,'seed_metrics':seed,'configuration_summary':summary,'incremental_pairs':incremental,
            'eligibility':eligibility,'experiment_ledger':ledger,'hydrological_strata':strata,'peak_errors':peaks,
            'episode_catalogue':catalogue,'grouped_gain':gains,'grouped_shap':shap,'shap_interactions':shap_inter,
            'blocked_permutation':permutation,'event_bootstrap':bootstrap,'factorial_interactions':factorial,
            'regional_feature_recommendation_matrix':matrix,'diagnostic_coverage':coverage,
            'diagnostic_effect_summary':diagnostic_results,'diagnostic_seed_averages':diagnostic_pairs,
            'paired_stratum_effects':stratum_pairs,'feature_window_gain':window_gains,'predicted_change_summary':change_summary,'target_variability':variability,'rain_event_metrics':rain_event_metrics,'rain_event_feature_effects':rain_event_effects,'rain_event_stability':rain_event_summary}
    if len(precision):tables['checkpoint_precision_sensitivity']=precision
    if len(precision_effects):tables['checkpoint_feature_effect_sensitivity']=precision_effects
    for name,frame in tables.items():
        frame.to_csv(td/f'{name}.csv',index=False)
        if len(frame)>10000:frame.to_parquet(td/f'{name}.parquet',index=False)
    write_json(output/'regional_candidate_schemas.json',schema_candidates(matrix))
    names=source.water.set_index('station_id').name.to_dict()
    display_summary=summary.copy();display_summary['station_name']=display_summary.station_id.map(names)
    primary=display_summary[display_summary['mode'].eq('normal')&display_summary.lag.eq(6)&display_summary.version.eq('all')]
    dv=primary[primary.station_id.eq('0074647403')&primary.experiment.eq('E3')&primary.arm.str.startswith('W')]
    metric_columns=['arm','scope','horizon','n','rmse_mean','rmse_sd','mae_mean','n200_mean','n200_min','n200_max']
    column_names={'arm':'Phương án','scope':'Phạm vi','horizon':'Horizon','n':'Số dự báo mỗi seed','rmse_mean':'RMSE TB (mm)',
                  'rmse_sd':'SD RMSE giữa seed','mae_mean':'MAE TB (mm)','n200_mean':'N200 TB','n200_min':'N200 nhỏ nhất','n200_max':'N200 lớn nhất',
                  'station_name':'Trạm nước','experiment':'Nhóm feature','station_id':'Mã trạm','feature_group':'Nhóm feature',
                  'mean_delta_rmse_mm':'ΔRMSE TB (mm)','mean_delta_n200':'ΔN200 TB','seed_win_fraction':'Tỷ lệ seed giảm RMSE',
                  'train_dh_std_mm':'SD biến thiên train (mm)','train_abs_dh_ge200_pct':'Train |ΔH|≥200 (%)','fold_win_fraction':'Tỷ lệ fold giảm RMSE','fold_count':'Số fold','selected_arm':'Phương án','status':'Kết luận','rain_ids':'Mã trạm mưa','grid_ids':'Mã ô dự báo'}
    def show(frame):
        f=frame.copy()
        if 'feature_group' in f:f['feature_group']=f.feature_group.map({'observed_rain':'Mưa quan trắc','weather':'Dự báo mưa','selected_observed_rain':'Trạm mưa tuyển chọn','challenger_weather':'Ô ngoài bộ chính (D9)'}).fillna(f.feature_group)
        if 'scope' in f:f['scope']=f.scope.map({'validation':'Validation trước cutoff','exploratory':'Đánh giá thăm dò'}).fillna(f.scope)
        return table(f.rename(columns=column_names))
    diag_summary=coverage.groupby(['diagnostic','status']).agg(stations=('station_id','nunique'),seed_aliases=('completed_seed_aliases','sum')).reset_index()
    e_summary=ledger[ledger.status.eq('completed')].groupby('experiment').agg(stations=('station_id','nunique'),seed_aliases=('fit_key','size'),unique_fits=('fit_key','nunique')).reset_index()
    stats={}
    for label in ['observed_rain_added','weather_added_to_observed_rain','remove_12_24h_forecast','train_selected_observed_rain']:
        p=incremental[incremental.scope.eq('validation')&incremental.diagnostic.eq(label)]
        stats[label]=p.groupby(['station_id','horizon','arm']).agg(delta_rmse=('delta_rmse_mm','mean'),delta_n200=('delta_n200','mean')).reset_index()
    rain_stats=stats['observed_rain_added'];weather_stats=matrix[matrix.feature_group.eq('weather')]
    n_rain_help=int((rain_stats.delta_rmse.lt(0)&rain_stats.delta_n200.le(0)).sum())
    n_weather_candidate=int((weather_stats.mean_delta_rmse_mm.lt(0)&weather_stats.mean_delta_n200.le(0)&weather_stats.seed_win_fraction.ge(.8)&weather_stats.fold_win_fraction.ge(2/3)&weather_stats.fold_count.ge(2)).sum())
    used=sorted(ledger.fit_key.dropna().unique());model_count=sum(json.loads((work/'fits'/k/'done.json').read_text())['trained_models'] for k in used)
    fresh_model_count=sum(json.loads((work/'fits'/k/'done.json').read_text()).get('newly_fitted_models',json.loads((work/'fits'/k/'done.json').read_text())['trained_models']) for k in used)
    reused_fold_count=model_count-fresh_model_count
    reuse_count=int(ledger.status.eq('completed').sum())-len(used)
    dv_rows=[]
    for h in [6,12]:
        for scope in ['validation','exploratory']:
            p=primary[(primary.station_id=='0074647403')&primary.horizon.eq(h)&primary.scope.eq(scope)&
                      ((primary.experiment.isin(['E0','E1']))|(primary.experiment.eq('E3')&primary.arm.eq('W1')))]
            dv_rows.append(p)
    dv_baselines=pd.concat(dv_rows)
    selected_rows=[]
    for sid in complete:
        finalists=json.loads((work/'stations'/sid/'finalists.json').read_text())['arms']
        p=primary[(primary.station_id==sid)&((primary.experiment.isin(['E0','E1']))|(primary.experiment.eq('E3')&primary.arm.isin(finalists)))]
        selected_rows.append(p)
    station_table=pd.concat(selected_rows)
    thermal_parts=[pd.read_json(p,lines=True) for p in [work/'thermal.jsonl',work/'checkpoint_precision/thermal.jsonl'] if p.exists()]
    thermal=pd.concat(thermal_parts,ignore_index=True) if thermal_parts else pd.DataFrame()
    plots(output,summary,incremental,matrix,factorial,permutation)
    diagnostic_table=diagnostic_results[diagnostic_results.scope.eq('validation')&diagnostic_results.diagnostic.isin(['D1','D2','D6','D7','remove_12_24h_forecast','train_selected_observed_rain'])]
    diagnostic_table=diagnostic_table.copy()
    diagnostic_table['diagnostic']=diagnostic_table.diagnostic.replace({'remove_12_24h_forecast':'D3/E4','train_selected_observed_rain':'D5/E5'})
    regional_table=primary[primary.station_id.eq('751117')&primary.arm.str.startswith('R')]
    rain_focus=rain_event_summary[rain_event_summary.scope.eq('validation')&rain_event_summary.station_id.isin(['0074647402','0074647403','0074647404','0074647606'])&rain_event_summary.arm.eq('W1')].copy()
    rain_focus['station_name']=rain_focus.station_id.map(names)
    rain_focus=rain_focus[['station_name','horizon','evaluated_episodes','episodes_lower_rmse','episodes_higher_rmse']].rename(columns={'station_name':'Trạm nước','horizon':'Horizon','evaluated_episodes':'Đợt có đủ mẫu','episodes_lower_rmse':'Đợt giảm RMSE','episodes_higher_rmse':'Đợt tăng RMSE'})
    regional_validation=regional_table[regional_table.scope.eq('validation')&regional_table.horizon.eq(12)]
    regional_note=('Tại Trung Làng, RMSE validation +12h nằm trong khoảng '+f'{regional_validation.rmse_mean.min():.3f}–{regional_validation.rmse_mean.max():.3f} mm'+', với N200 trung bình từ '+f'{regional_validation.n200_mean.min():.1f} đến {regional_validation.n200_mean.max():.1f}'+'. Mức cải thiện nhỏ này chưa đủ xác nhận khả năng dự báo lũ.' if len(regional_validation) else 'Không có đối chứng Trung Làng trong phạm vi báo cáo này.')
    weather_examples=matrix[matrix.feature_group.eq('weather')&matrix.mean_delta_rmse_mm.notna()].sort_values('mean_delta_rmse_mm').head(3).copy()
    weather_examples['station_name']=weather_examples.station_id.map(names)
    pattern_table=show(weather_examples[['station_name','horizon','selected_arm','mean_delta_rmse_mm','mean_delta_n200','seed_win_fraction','fold_win_fraction','status']])
    constant_station_ids=sorted(constant_targets.station_id.unique())
    constant_station_names='; '.join(names[s] for s in constant_station_ids)
    dv_selected=json.loads((work/'stations/0074647403/finalists.json').read_text())['arms']
    precision_note=(f"Đã chạy {precision_interpretation['models']} model validation bổ sung với seed 42. Trên {precision_interpretation['paired_feature_cases']} phép đối chiếu feature có forecast, không có trường hợp lợi ích RMSE ban đầu dưới −0,05 mm chuyển thành tác hại trên +0,05 mm; có {precision_interpretation['old_harm_to_new_benefit']} trường hợp đổi theo chiều ngược lại. Mức thay đổi RMSE gộp lớn nhất của một model là {precision_interpretation['largest_pooled_rmse_reduction_mm']:.3f} mm, nên không nên khẳng định những ưu thế nhỏ ở mức dưới vài mm chỉ từ một thứ hạng. Đây chỉ là độ nhạy ở seed 42, chưa thay được kiểm chứng đủ năm seed." if precision_interpretation else 'Chưa có kết quả kiểm tra độ nhạy số học; không coi phần này là đã đạt.')
    dv_change=change_summary[(change_summary.station_id=='0074647403')&change_summary.stratum.eq('rapid_rise')&((change_summary.experiment.isin(['E0','E1']))|(change_summary.experiment.eq('E3')&change_summary.arm.eq('W1')))]
    record=json.loads((work/'execution_manifest.json').read_text())
    summary_status='HOÀN THÀNH' if stage_complete else 'TẠM THỜI; CÒN TRẠM CHƯA HOÀN THÀNH'
    text=f'''# Stage 4 — Feature ablation đa trạm bằng XGBoost

**Trạng thái: {summary_status}.** Phân tích dùng mô hình XGBoost riêng cho từng trạm, dự báo mức thay đổi mực nước +6h/+12h rồi cộng với mực nước hiện tại. Không huấn luyện Regional LSTM, không chuẩn bị Regional Dataset và không sửa Model B hoặc pipeline production.

## 1. Kết quả chính và mức độ bằng chứng

Đã hoàn thành {len(complete)}/{len(eligible)} trạm đủ điều kiện trong tổng số {len(source.water)} trạm nước. Có **{len(used)} cấu hình/seed thực sự khác nhau**, tương ứng **{model_count} mô hình theo fold/horizon**. Trong đó {fresh_model_count} model được fit mới và {reused_fold_count} model theo fold được tái sử dụng sau khi kiểm tra cùng source, schema, seed, cấu hình và anchor; hash model nguồn được lưu trong từng completion marker. Tổng số nhãn thí nghiệm/seed đã hoàn thành là {int(ledger.status.eq('completed').sum())}; {reuse_count} nhãn được gộp với fit tương đương. Việc gộp dựa trên schema, trạm, nguồn mưa, giả định khả dụng và seed, không dựa trên việc chỉ số tình cờ giống nhau.

**Một giới hạn dữ liệu quan trọng:** {len(constant_station_ids)}/{len(complete)} trạm có ΔH +6/+12 giờ hoàn toàn không biến động trong phần train trước cutoff: {constant_station_names}. Các trạm này đủ số mẫu để chạy đối chứng nhưng không có đủ biến thiên target để chứng minh feature học được nước dâng. Đây là tính chất của snapshot, chưa kết luận cảm biến hỏng hay thực tế luôn khô. Khi thiết kế Regional Model, đưa nhiều mẫu từ các trạm chỉ có trạng thái ổn định vào loss chung có thể làm tăng xu hướng học ΔH nhỏ; cần tách vai trò dữ liệu nền và dữ liệu động, không tự tạo biến động hoặc lấy target sau cutoff để bù train.

Khi thêm mưa quan trắc vào lịch sử nước, **{n_rain_help}/{len(rain_stats)} cặp trạm–horizon** vừa giảm RMSE trung bình vừa không tăng N200 trên validation. Với forecast trong các phương án được validation chọn, **{n_weather_candidate}/{len(weather_stats)} cặp trạm–horizon** đạt quy tắc ứng viên có lợi: giảm RMSE, không tăng N200, ít nhất 80% seed giảm RMSE và ít nhất 2/3 fold giảm RMSE. Đây là quy tắc diễn giải được áp dụng thống nhất cho ma trận, không phải kiểm định ý nghĩa thống kê hoặc chứng minh hiệu quả cho Regional LSTM. Các trạm chỉ có biến động target rất nhỏ được đánh dấu riêng, dù vẫn đạt quy tắc giảm sai số.

Những cặp trạm–horizon có mức giảm RMSE lớn nhất trong các ứng viên forecast được validation chọn được trình bày dưới đây. Chỉ tiêu N200 và độ đồng thuận seed/fold phải đọc cùng RMSE; danh sách này không được dùng chọn lại cấu hình sau khi xem phần đánh giá.

{pattern_table}

Điểm đáng chú ý cần kiểm tra là tính phụ thuộc trạm và horizon: một feature có thể giảm lỗi tại trạm này nhưng làm tăng lỗi tại trạm khác, hoặc giảm RMSE nhưng tăng N200. Không suy từ lợi ích tại một trạm sang cả vùng. D1/D2 và phân tích sự kiện bên dưới giúp kiểm tra nguồn của mức cải thiện, thay vì gán ngay cho quan hệ thủy văn.

Các kết quả sau 01/08/2026 đã được xem trong các giai đoạn trước. Chúng được trình bày để kiểm tra sự thay đổi phân phối và khả năng chuyển tiếp theo thời gian, **không phải test độc lập cuối cùng**. Các ô W0–W4 cũng từng được xem trong EDA trước cutoff, nên validation nội bộ vẫn chịu ảnh hưởng của lịch sử tuyển chọn feature; cần dữ liệu mới để xác nhận cuối cùng.

Các thuật ngữ trong báo cáo: **anchor t** là mốc phát dự báo, **horizon** là khoảng cách từ t tới thời điểm cần dự báo, **seed** là giá trị khởi tạo ngẫu nhiên, **fold** là một lần chia train/validation theo thời gian. **Persistence** dự báo giữ nguyên H(t). MAE là sai số tuyệt đối trung bình, RMSE phạt nặng hơn các lỗi lớn; cả hai tính bằng mm. **N200** đếm từng dự báo có |sai số|>200 mm, không đếm số trận.

## 2. Dữ liệu, feature và cách kiểm soát leakage

Nguồn là các artifact đóng băng Stage 1–3B, không truy vấn lại PostgreSQL và không thay quan trắc gốc. Mỗi mẫu dùng mực nước của chính trạm tại t, hai nhãn đúng timestamp t+6/t+12 và lịch sử 96 giờ. Giữ tối thiểu 80% giờ quan trắc trong lịch sử, nhưng bắt buộc có mực tại t và cả hai nhãn. Các giờ thiếu trong input giữ NaN để XGBoost xử lý; cờ thiếu của mưa được giữ riêng. Đây là schema cây phục vụ ablation, không mặc định tương đương hoàn toàn với đầu vào LSTM production.

Feature nước gồm 96 lag theo giờ, biến thiên quá khứ 1/3/6/12/24/48/95h, trung bình/độ lệch chuẩn quá khứ và tỷ lệ có dữ liệu. Mưa quan trắc gồm tổng 1/3/6/12/24h và lag 0/3/6/12/24/48/72/95h cho hai trạm mưa, kèm cờ thiếu. Forecast chỉ đặt tại anchor (`anchor_only`): bốn khoảng 0–3/3–6/6–12/12–24h cho từng ô, mask và một tuổi dự báo chung. Không ghép forecast khác chu kỳ trong cùng một vector.

E1/E3 dùng hai trạm mưa cố định theo vị trí; riêng Dã Viên giữ Hồng Thái+Nhâm để làm đối chứng đã biết. E5 tuyển hai trạm bằng tương quan mưa quá khứ 6h với ΔH +12h **chỉ trên train của fold hợp lệ sớm nhất** tại từng trạm, rồi đóng băng cho mọi fold sau. Ô thách thức D9 cũng được chọn trên train sớm nhất, bằng trung bình độ lớn liên hệ riêng phần ngoài nước, mưa quan trắc và A Lưới, với ngưỡng tối thiểu 0,05; không chọn bằng kết quả sau cutoff.

Các fold validation cố định theo giờ Việt Nam: **15/12/2025–01/01/2026**, 01/01–01/03/2026, 01/04–01/06/2026 và 01/07–01/08/2026. Fold tháng 12 được bổ sung sau kiểm tra phạm vi thời gian vì ba fold ban đầu không nằm trong tháng 9–12. Nó có 601 mẫu train và 384 mẫu validation tại Dã Viên, gồm các mốc nước tăng nhanh. Bản kết luận chỉ dùng protocol bốn fold này; các model cũ chỉ tái sử dụng khi phép tính thực sự tương đương, không gộp kết quả của hai protocol để chọn cấu hình. Mỗi fold chỉ train trên quá khứ trước phần validation và loại 24 giờ ở ranh giới. Cần ít nhất 512 mẫu train và 168 mẫu validation. Một trạm được giữ nếu có ít nhất một fold đủ điều kiện; trạm có ít fold phải được diễn giải với độ tin cậy thấp hơn.

Mọi phương án tại cùng trạm dùng giao anchor chung của các giả định phát hành +6/+9/+12h. Tuổi dự báo tối đa 24h. Đây vẫn là giả định nghiên cứu, không chứng minh dự báo đã được công bố trong quá khứ. Receipt time của mọi quan trắc cũng chưa được chứng minh vận hành tại từng anchor. Source trước 12/05/2026 06Z là hindcast 49r1; source sau đó là archive 50r1 chưa xác minh giờ công bố, được tách khi đánh giá.

XGBoost dùng cấu hình cố định: depth={c['xgboost']['max_depth']}, learning rate={c['xgboost']['eta']}, tối đa {c['num_boost_round']} cây, early stopping {c['early_stopping_rounds']}, subsample/colsample=0,8. Checkpoint ưu tiên số lỗi nghiêm ngặt >200 mm, với thành phần phụ RMSE theo công thức cố định **N200 + 0,001×RMSE/(RMSE+1)**. API custom metric của phiên bản XGBoost đang dùng biểu diễn kết quả ở sáu chữ số thập phân, nên thành phần phụ có độ phân giải hữu hạn; không diễn giải đây là phép chọn RMSE chính xác vô hạn trong mọi trường hợp hòa N200. Dùng đủ seed {c['seeds']}. Sau validation, số cây của mỗi seed/horizon là trung vị số cây tốt nhất qua các fold; refit trên toàn bộ phần trước cutoff đã purge rồi mới dự báo phần đánh giá. Không dùng phần đánh giá để chọn checkpoint hay số cây.

[Điều kiện từng trạm](tables/eligibility.csv) · [Toàn bộ chỉ số từng fold/seed](tables/all_fold_metrics.csv) · [Ledger thí nghiệm](tables/experiment_ledger.csv).

## 3. Kiểm kê đầy đủ E0–E5, W0–W7 và D1–D9

E0 chỉ nước; E1 nước+mưa quan trắc; E2 nước+forecast; E3 cả ba nhóm; E4 như E3 nhưng bỏ forecast 12–24h và mask tương ứng; E5 như E3 nhưng dùng rain selection trên train. Mưa quá khứ 24h vẫn giữ trong E4. W0 của E3 chính là E1, và E5 có thể trùng E3 nếu chọn cùng trạm; ledger giữ tên thí nghiệm nhưng tái sử dụng cùng model.

{table(e_summary.rename(columns={'experiment':'Nhóm','stations':'Số trạm','seed_aliases':'Nhãn thí nghiệm/seed','unique_fits':'Fit khác nhau'}))}

Dã Viên chạy đủ tám tập con của A Lưới, Dã Viên và Thuận An: W0 rỗng, W1 A Lưới, W2 Dã Viên, W3 A Lưới+Thuận An, W4 cả ba, W5 Thuận An, W6 A Lưới+Dã Viên, W7 Dã Viên+Thuận An. Mỗi phương án có đủ năm seed. Các trạm khác thực hiện W0–W4 và các đối chứng có điều kiện theo specification, không tự áp tọa độ Dã Viên cho mọi trạm.

D1 giữ mask/age nhưng đặt mưa forecast bằng 0. D2 sao chép A Lưới vào các kênh ô bổ sung. D3 là E4; D4 là đối chiếu E0–E3; D5 là E5; D6 đổi mưa standard 5/6 sang strict 6/6; D7 giả định khả dụng sau 9/12h; D8 phân tầng phiên bản và chỉ retrain riêng phiên bản khi fold đủ mẫu và có ít nhất hai đợt mưa train; D9 là ô thách thức được train chọn. Các phạm vi có điều kiện được giữ như thiết kế, không nhân tất cả với mọi cấu hình.

{table(diag_summary.rename(columns={'diagnostic':'Đối chứng','status':'Trạng thái','stations':'Số trạm','seed_aliases':'Nhãn thí nghiệm/seed đã chạy'}))}

[Bảng D1–D9 theo từng trạm](tables/diagnostic_coverage.csv) ghi các nhánh hoàn thành, không đủ dữ liệu hoặc không kích hoạt. Không kích hoạt D2 khi không có finalist nhiều ô là đúng điều kiện đã chốt, không phải bỏ bớt thí nghiệm. Các kết quả không đủ điều kiện không được thay bằng số 0.

### 3.1. Kết quả định lượng của các đối chứng

Bảng sau lấy trung bình năm seed trong từng cặp trạm–phương án–horizon, rồi tóm tắt các cặp trên validation. Một trạm có thể xuất hiện nhiều lần nếu có nhiều phương án đủ điều kiện; vì vậy cột số phép đối chiếu không phải số trạm độc lập.

{table(diagnostic_table[['diagnostic','horizon','comparisons','stations','median_delta_rmse_mm','comparisons_reducing_rmse','comparisons_increasing_rmse','mean_delta_n200']].rename(columns={'diagnostic':'Đối chứng','horizon':'Horizon','comparisons':'Số phép đối chiếu','stations':'Số trạm','median_delta_rmse_mm':'ΔRMSE trung vị (mm)','comparisons_reducing_rmse':'Số cặp giảm RMSE','comparisons_increasing_rmse':'Số cặp tăng RMSE','mean_delta_n200':'ΔN200 TB'}))}

Với D1, ΔRMSE = mô hình chỉ mask/age trừ mô hình có lượng mưa thật: số dương ủng hộ việc giá trị mưa có thông tin ngoài tuổi dự báo, còn số âm không ủng hộ lợi ích ấy. Trong tập anchor forecast đầy đủ này, forecast mask hầu như không biến thiên, nên D1 chủ yếu kiểm tra tín hiệu tuổi dự báo cùng các đầu vào còn lại, không chứng minh mô hình chịu được missing forecast trong vận hành.

Với D2, số dương cho biết các kênh mưa mới tốt hơn việc lặp lại A Lưới trong đúng cấu hình đó. Số lượng kênh được giữ bằng nhau; số nút/cây mà XGBoost thực sự học vẫn có thể khác. Đây không phải phép kiểm tra tham số tương đương của một mạng LSTM. Với D3/E4, số âm nghĩa bỏ forecast 12–24h có lợi; với D5/E5, số âm nghĩa thay trạm mưa có lợi. Với D6/D7, số âm nghĩa chính sách strict hoặc giả định trễ hơn tốt hơn trong đúng phép đối chiếu, không có nghĩa chúng đúng hơn về mặt công bố dự báo.

Kết quả D8 được giữ riêng phiên bản trong bảng đầy đủ. Phần đánh giá sau cutoff chỉ có phiên bản mới; vì vậy không tạo phép so sánh giả giữa một test 49r1 không tồn tại và test 50r1. D9 chỉ dùng ô được train chọn; ledger và discovery.json giữ mã ô, điểm và lý do kích hoạt.

## 4. Dã Viên: đối chiếu đầy đủ và các baseline

Persistence giữ nguyên H(t), còn E0 là XGBoost học từ lịch sử nước; đây là hai baseline khác nhau. Persistence không cần seed hoặc training. Bảng sau dùng đúng cùng anchor của các phương án E/W, và gộp các fold validation theo số mẫu.

{show(persistence[persistence.station_id.eq('0074647403')][['scope','horizon','n','rmse_mm','mae_mm','n200']].rename(columns={'rmse_mm':'RMSE (mm)','mae_mm':'MAE (mm)','n200':'N200'}))}

[Bảng persistence cho toàn bộ trạm](tables/persistence_baselines.csv).

{show(dv_baselines[['experiment',*metric_columns]])}

Theo quy tắc tuyển chọn dùng chung hai horizon, N200 và RMSE trung bình đủ năm seed trên validation, danh sách tiếp tục kiểm chứng tại Dã Viên là **{", ".join(dv_selected)}**. Không đổi sang một tổ hợp khác chỉ vì nó có chỉ số sau cutoff tốt hơn. Thêm nhiều ô không mặc định tốt hơn: cần đọc đồng thời RMSE, N200 và độ nhạy seed/checkpoint.

Bảng sau giữ tất cả W0–W7. N200 là số dự báo có sai số tuyệt đối >200 mm, không phải số trận lũ. Các giá trị TB là trung bình năm seed; N là số dự báo của một seed trên cùng phạm vi. Chỉ số validation đã gộp các fold theo số mẫu, không lấy trung bình RMSE của những fold khác kích thước.

{show(dv[metric_columns])}

![Dã Viên đầy đủ tổ hợp](figures/01_da_vien_factorial.png)

![Độ nhạy seed](figures/04_seed_stability.png)

## 5. So sánh đa trạm và giá trị bổ sung của feature

Các phương án trong bảng được chọn bằng validation ở giai đoạn trước cutoff, với N200 và RMSE trung bình năm seed. Việc liệt kê kết quả đánh giá của phương án đó không có nghĩa chọn lại phương án bằng dữ liệu đánh giá. Những trạm chỉ có một fold chưa đủ để khẳng định ổn định theo thời gian.

{show(station_table[['station_name','experiment','arm','scope','horizon','rmse_mean','mae_mean','n200_mean','rmse_sd']])}

![Đóng góp mưa quan trắc](figures/02_observed_rain_increment.png)

![Đóng góp forecast](figures/03_weather_increment.png)

[Bảng chênh lệch ghép cặp](tables/incremental_pairs.csv) giữ từng seed và horizon: thêm mưa quan trắc, thêm forecast khi đã có mưa quan trắc, bỏ forecast 12–24h, thay trạm mưa, và các đối chứng. Chênh lệch âm là giảm sai số hoặc giảm số lỗi. Không tính “cải thiện” bằng cách so các tập anchor khác nhau.

### 5.1. Đối chứng riêng Trung Làng, Quảng Thái

R0 không forecast; R1 chỉ A Lưới; R2 chỉ Bạch Mã thấp; R3 A Lưới+Bạch Mã thấp. Cả bốn dùng cùng nền nước+mưa quan trắc và cùng năm seed. Có thể kiểm tra trực tiếp liệu tín hiệu riêng phần từ Stage 3B chuyển thành predictive value hay chỉ là liên hệ thống kê.

{show(regional_table[['arm','scope','horizon','n','rmse_mean','rmse_sd','n200_mean','n200_min','n200_max']])}

{regional_note} Tín hiệu riêng phần từ Stage 3B chỉ gợi ý một ứng viên, không tự động trở thành lợi ích dự báo lớn có ý nghĩa thực tế.

## 6. Điều kiện thủy văn, sự kiện, lỗi lớn và độ bất định

Mực cao được xác định bằng quantile 95% của mực nước trong train sớm nhất, riêng từng trạm. Rapid rise dùng ΔH +6h≥100 mm hoặc ΔH +12h≥200 mm. Phân biệt nước tăng/giảm, tháng 9–12, giờ mực cao và giờ thuộc đợt. Đợt được phát hiện hồi cứu từ mực cao hoặc biến thiên quá khứ lớn, gộp các mốc cách nhau không quá 12h; đây là đợt thuật toán, không phải danh mục lũ độc lập hay ngưỡng báo động chính thức.

[Bảng theo điều kiện và từng đợt](tables/hydrological_strata.csv) bao gồm số lỗi >200 được sửa và phát sinh mới so với persistence. [Chênh lệch feature theo từng điều kiện](tables/paired_stratum_effects.csv) còn so với nền nước hoặc nước+mưa quan trắc trên cùng nhóm và cùng seed. [Sai số đỉnh](tables/peak_errors.csv) tách chênh lệch đỉnh lớn nhất, sai số tại thời điểm đỉnh thật và độ lệch giờ đỉnh trong các đợt có ít nhất sáu dự báo. Không dùng thời điểm đỉnh thật để chuyển model lúc inference. Ở các trạm có độ lệch chuẩn ΔH trong train dưới 20 mm, N200 có thể bằng 0 cho hầu hết phương án; mức giảm RMSE nhỏ ở đó chỉ là bằng chứng về biến động nhỏ, chưa phải bằng chứng dự báo lũ. Mức 20 mm chỉ là cờ mô tả thang biến động trong báo cáo, không phải ngưỡng lũ hoặc báo động. Cần biết độ phân giải và sai số cảm biến trước khi gán ý nghĩa vật lý cho mức cải thiện rất nhỏ.

### 6.1. Mô hình có còn dự báo biên độ thay đổi quá nhỏ không?

Bảng sau chỉ xét các mốc nước thực tế tăng nhanh tại Dã Viên. Cùng một nhóm mốc được dùng cho tất cả model. Độ lớn thay đổi dự báo là |H dự báo(t+h)−H(t)|, khác với sai số dự báo. Tỷ lệ biên độ là trung bình độ lớn dự báo chia trung bình độ lớn thực tế; tỷ lệ thấp cho thấy mô hình vẫn co dự báo về persistence trong nhóm này. Chỉ tiêu này không thay RMSE hoặc độ chính xác chiều thay đổi.

{show(dv_change[['experiment','arm','scope','horizon','n','actual_abs_change_mm','predicted_abs_change_mm','amplitude_ratio_pct']].rename(columns={'actual_abs_change_mm':'TB độ lớn thay đổi thực tế (mm)','predicted_abs_change_mm':'TB độ lớn thay đổi dự báo (mm)','amplitude_ratio_pct':'Tỷ lệ biên độ (%)'}))}

[Bảng biên độ cho toàn bộ trạm](tables/predicted_change_summary.csv) còn tách nước giảm nhanh, mực cao, giờ sự kiện và toàn phần dữ liệu. Không dùng nhãn “tăng nhanh trong tương lai” để chọn model lúc dự báo.

[Bootstrap theo seed và đợt](tables/event_bootstrap.csv) sử dụng {c['bootstrap_replicates']} lần lấy mẫu ghép cặp cho W1 và các finalist. Chỉ xuất khoảng khi có ít nhất ba đợt thuật toán; các hàng thiếu sự kiện được ghi không đủ điều kiện. Các khoảng này không chứng minh các đợt có độc lập vật lý và không thay được một mùa lũ mới chưa từng xem. Nếu khoảng chênh lệch chứa 0 thì không khẳng định hiệu quả ổn định. Không bootstrap từng giờ chồng lấn như những mẫu độc lập.

### 6.2. Đánh giá riêng theo các đợt mưa của Stage 2

Các đợt biến động nước ở trạm ven biển có thể gộp rất dài vì thủy triều lặp lại, nên không được tự coi chúng là những trận lũ độc lập. Bổ sung [sai số theo đợt mưa](tables/rain_event_metrics.csv), [chênh lệch feature trong cùng đợt](tables/rain_event_feature_effects.csv) và [độ ổn định qua các đợt](tables/rain_event_stability.csv) cho E0/E1/W1 và các phương án được validation chọn, đủ năm seed. Mỗi cặp chỉ dùng các target timestamp nằm trong cùng cửa sổ đợt mưa và đủ ít nhất sáu dự báo. Mốc thời gian nằm ngoài cửa sổ không bị tự kéo vào đợt bằng nội suy.

Các cửa sổ lấy nguyên từ thuật toán Stage 2 trên mạng trạm mưa. Chúng chưa phải catalog lũ độc lập tại từng lưu vực và có thể bỏ sót phản ứng nước trễ sau khi đợt mưa kết thúc. Bảng ghi riêng số đợt giảm/tăng RMSE thay vì chỉ một giá trị gộp; không suy ra tín hiệu ổn định qua lũ nếu phần lớn lợi ích chỉ xuất hiện trong một đợt. Đây là đánh giá theo sự kiện của các dự báo chia theo thời gian, không phải leave-entire-event-out retraining. Nếu một đợt đi qua ranh giới fold thì một phần quá khứ của đợt có thể đã nằm trong train của fold sau; không gọi toàn bộ đợt đó là hoàn toàn chưa từng được thấy. Nhãn sự kiện chỉ dùng kiểm chứng hồi cứu, không làm feature hoặc cổng chọn model lúc inference.

Bảng minh họa dưới giữ ô A Lưới W1 và đối chiếu với nước+mưa quan trắc trên cùng đợt, lấy trung bình đủ năm seed. Số đợt giảm RMSE không đồng nghĩa với số trận lũ độc lập.

{table(rain_focus)}

Lợi ích không xuất hiện đều trong mọi đợt. Chẳng hạn, cần đọc kết quả +12h của Huỳnh Liên cùng số đợt tăng sai số, thay vì chỉ dựa vào mức giảm RMSE gộp lớn. Tương tự, một tín hiệu có lợi tại Dã Viên +6h chưa chắc ổn định ở +12h. Đây là lý do giữ feature theo trạm/horizon và tiếp tục kiểm chứng ở một mùa mới, thay vì tuyên bố có một bộ forecast tốt nhất cho cả vùng.

## 7. Feature importance, interactions và giải thích có giới hạn

[Bảng biến thiên target trong từng fold train](tables/target_variability.csv) giúp phân biệt thiếu tín hiệu đầu vào với thiếu biến động cần học. Nếu target hầu như cố định, XGBoost có thể chỉ học một mức hiệu chỉnh chung và không tạo split. Các trường hợp này vẫn có đầy đủ dự báo/metric và được ghi `constant_prediction_no_splits` trong bảng gain, không phải file importance bị lỗi hoặc thiếu thí nghiệm. Có {len(constant_targets)} cặp trạm–horizon có độ lệch chuẩn ΔH trên toàn phần trước cutoff dưới 0,000001 mm. Chưa đủ cơ sở từ những trường hợp đó để nói mưa hoặc forecast không có giá trị trong một trận nước dâng chưa xuất hiện ở train.


[Bảng gain theo nhóm](tables/grouped_gain.csv) giữ đủ model/seed. [Gain theo độ dài lịch sử/cửa sổ mưa](tables/feature_window_gain.csv), [TreeSHAP theo nhóm](tables/grouped_shap.csv), [tương tác TreeSHAP](tables/shap_interactions.csv) và [dịch khối feature](tables/blocked_permutation.csv) dùng seed 42 được cố định trước, gồm baseline và các phương án được chọn. TreeSHAP lấy tối đa 48 mốc trải đều trong từng phạm vi, không đủ để tự đại diện mọi cực trị; kết luận sự kiện phải đọc cùng bảng metric trên toàn bộ mốc. Các mô hình validation giải thích bằng model của chính fold đó; phần đánh giá dùng model refit trước cutoff, tránh giải thích validation bằng một model đã train trên validation.

Phép dịch 24/72/168 hàng chỉ phục vụ chẩn đoán độ nhạy trong từng phạm vi, không phải forecast có thể dùng vận hành. Nó có thể phá vỡ liên hệ giữa các feature tương quan; gain cũng có thể phân bổ đóng góp không ổn định giữa feature gần trùng. Vì vậy, importance lớn hoặc khác 0 không chứng minh nhân quả, và importance nhỏ không đủ để loại feature khỏi LSTM.

[Bảng tương tác factorial tại Dã Viên](tables/factorial_interactions.csv) tính lợi ích kết hợp của từng cặp ô so với tổng lợi ích từng ô riêng sau refit. Với MSE, synergy = MSE(A)+MSE(B)−MSE(A+B)−MSE(không forecast); số dương nghĩa lợi ích kết hợp vượt tổng lợi ích riêng trong đúng thiết kế này. Đây là tương tác của mô hình và dữ liệu, không khẳng định tương tác thủy văn nhân quả.

![Độ nhạy feature tại Dã Viên](figures/05_grouped_permutation.png)

### 7.1. Kiểm tra độ nhạy của tie-break số học

Quy tắc checkpoint chính được giữ nguyên cho mọi phương án và mọi seed để so sánh công bằng. Khi kiểm tra implementation của API, phát hiện phần RMSE bị nén bởi hệ số 0,001 trước khi làm tròn sáu chữ số thập phân. Vì thế không nên coi một chênh lệch rất nhỏ giữa các cấu hình là kết luận chắc chắn chỉ từ thứ hạng model.

[Bảng kiểm tra độ nhạy checkpoint](tables/checkpoint_precision_sensitivity.csv) nếu đã tạo, dùng seed 42 cố định, baseline E1 và W1/các finalist tại từng trạm, chỉ trên validation. Nó thay phần phụ thành RMSE/(RMSE+1), vẫn luôn nhỏ hơn 1 nên không thay thứ tự ưu tiên N200 trong số học đầy đủ. Kiểm tra này không dùng dữ liệu sau cutoff để chọn model, không thay kết quả chính năm seed và không tự cập nhật danh sách feature. Cần đọc độ nhạy này cùng seed/fold và khoảng theo sự kiện trước khi khẳng định lợi ích nhỏ. Nếu chưa có artifact tương ứng thì phần kiểm tra này chưa được thực hiện, không được ghi là đạt.

{precision_note}

[Bảng độ nhạy của chính chênh lệch feature](tables/checkpoint_feature_effect_sensitivity.csv) đối chiếu forecast với cùng baseline sau khi thay độ phân giải tie-break. Các cấu hình có lợi ích nhỏ cần được giữ nhãn thăm dò thay vì gọi là tối ưu ổn định.

## 8. Regional Feature Recommendation Matrix

Các kết luận trong ma trận dựa trên validation ghép cặp, không dựa trên ô có correlation cao nhất hoặc model có test tốt nhất. Một feature được đánh dấu ứng viên có lợi chỉ khi giảm RMSE, không tăng N200 và có độ đồng thuận đủ giữa seed/fold; đây vẫn là bằng chứng từ XGBoost local, chưa phải từ Regional ResidualLSTM.

{show(matrix[['station_id','horizon','feature_group','selected_arm','rain_ids','grid_ids','train_dh_std_mm','mean_delta_rmse_mm','mean_delta_n200','seed_win_fraction','fold_win_fraction','fold_count','status']])}

[Ma trận đầy đủ](tables/regional_feature_recommendation_matrix.csv) · [Candidate schemas](regional_candidate_schemas.json).

Nhóm chung cần giữ làm baseline là lịch sử nước của chính trạm và cách biểu diễn missingness. Mưa quan trắc được ánh xạ theo hai slot có ý nghĩa tương đương nhưng station ID nguồn có thể khác nhau. Forecast nên có nhánh một ô và nhánh nhiều ô theo trạm; không ép tất cả trạm dùng cả bảy ô. Nếu một trạm không có bằng chứng forecast bổ sung, vẫn cần giữ W0 làm đối chứng trong nghiên cứu regional. Ma trận tách riêng ô thách thức D9 và bộ ô chính: một ô ngoài bộ chính chỉ được xem là ứng viên khi đã có bằng chứng validation ghép cặp, không tự đưa vào schema vì correlation cao hoặc kết quả sau cutoff đẹp.

DEM, tọa độ và station identity là ứng viên conditioning cho Regional Model, nhưng trong một model XGBoost local chúng là hằng số theo trạm nên Stage 4 này không đo được giá trị dự báo riêng của chúng. Việc chuẩn hóa theo trạm và persistence coefficients phải được thiết kế ở bước Regional Dataset Preparation; không tự lấy scaler của Dã Viên dùng cho các trạm khác. Cũng chưa chứng minh chuyển từ forecast anchor_only sang all_timesteps sẽ giữ nguyên kết quả.

## 9. Chạy lại, artifact và điểm dừng

Cấu hình đầy đủ nằm trong `research/station_analysis/phase2/feature_ablation/config.yaml`; code được chia theo đọc dữ liệu/feature, lập thí nghiệm, training có resume, đánh giá sự kiện, giải thích và báo cáo. Mỗi fit lưu identity, checkpoint theo fold/horizon, dự báo từng mốc, metrics và file hoàn thành. [Manifest kết quả](manifest.json) liên kết manifest thực thi và hash nguồn. Lỗi hoặc gián đoạn không khiến model cũ bị ghi đè; đổi đầu vào/code thực thi phải đổi run_name.

Các trạm chưa xong được chạy bằng hai worker có thư mục riêng; chỉ ghép các trạm đã có completion marker và model đầy đủ. Coordinator kiểm tra cấu hình khoa học, hash nguồn và code giữa các worker trước khi ghép; chỉ namespace output và nguồn cache được khác. Báo cáo cuối được tạo một lần ở coordinator, không phải ghép chỉ số tính khác phương pháp.

GPU được theo dõi liên tục; nghỉ từ {c['thermal_pause_c']}°C và tiếp tục khi không quá {c['thermal_resume_c']}°C. Nhiệt độ lớn nhất ghi nhận trong log là {thermal.c.max() if len(thermal) else 'không có'}°C. Log nhiệt được giữ cùng artifact nghiên cứu.

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/hydrology-stage4-mpl .venv/bin/python -m research.station_analysis.phase2.feature_ablation.pipeline
# Chỉ tái tạo phân tích/báo cáo từ fit đã lưu:
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/hydrology-stage4-mpl .venv/bin/python -m research.station_analysis.phase2.feature_ablation.pipeline --report-only
```

Chưa chuẩn bị Regional Dataset hoặc huấn luyện Regional LSTM. Các giới hạn còn lại gồm số mùa lũ ít, quan trắc receipt time chưa đầy đủ, forecast availability lịch sử chưa được chứng minh, feature discovery trước đây đã xem một phần validation, và chuyển tiếp phiên bản trùng với thay đổi theo mùa. Không dùng kết quả này để tự thay Model B hoặc đưa model mới vào production. Tài liệu phương pháp: [XGBoost Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html), [GPU và TreeSHAP](https://xgboost.readthedocs.io/en/stable/gpu/index.html).
'''
    if not len(precision):text=text.replace('[Bảng kiểm tra độ nhạy checkpoint](tables/checkpoint_precision_sensitivity.csv)', 'Bảng kiểm tra độ nhạy checkpoint (chưa có artifact)')
    if not len(precision_effects):text=text.replace('[Bảng độ nhạy của chính chênh lệch feature](tables/checkpoint_feature_effect_sensitivity.csv)','Bảng độ nhạy của chính chênh lệch feature (chưa có artifact)')
    (output/'report.md').write_text(text)
    import sys,importlib.metadata
    environment={'python':sys.version,'packages':{name:importlib.metadata.version(name) for name in ['xgboost','numpy','pandas','pyarrow','PyYAML']}}
    write_json(output/'environment.json',environment)
    audit_record=json.loads((work/'audit.json').read_text()) if (work/'audit.json').exists() else None
    if audit_record is not None:write_json(output/'audit.json',audit_record)
    hashes={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')}
    parallel_record=json.loads((work/'parallel_manifest.json').read_text()) if (work/'parallel_manifest.json').exists() else None
    if parallel_record is not None:write_json(output/'parallel_execution.json',parallel_record)
    write_json(output/'manifest.json',{'independent_audit':audit_record,'environment':environment,'checkpoint_precision_sensitivity':precision_interpretation,'parallel_execution':parallel_record,'stage':'Stage 4','complete':stage_complete,'execution_manifest':str(work/'execution_manifest.json'),
        'execution_signature':record['signature'],'completed_stations':complete,'eligible_stations':eligible,'used_unique_seed_fits':len(used),
        'trained_models':model_count,'newly_fitted_models':fresh_model_count,'reused_fold_models':reused_fold_count,'code_sha256':hashes,'source_inputs_sha256':record['inputs'],
        'not_executed':['regional_dataset_preparation','regional_training','production_changes'],
        'report_path':str(output/'report.md'),'created_at':pd.Timestamp.now(tz='UTC').isoformat()})
    write_json(Path(c['output_root'])/'latest.json',{'report':str(output/'report.md'),'complete':stage_complete})
    LOG.info('Stage 4 report complete=%s: %s',stage_complete,output/'report.md')
    return output/'report.md'
