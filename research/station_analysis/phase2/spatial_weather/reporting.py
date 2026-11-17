"""Vietnamese spatial forecast evidence and an explicit Stage 4 input specification."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..weather_reporting import vietnamese_table
from research.station_analysis.config import ROOT
from .water_comparison_reporting import water_comparison_section
from .experiment_plan import stage4_design, design_section


def report_spatial_weather(tables,synced,stations,config,output,work,progress):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import rasterio
    output=Path(output);td=output/'tables';fd=output/'figures';td.mkdir();fd.mkdir()
    for name,frame in tables.items():frame.to_csv(td/f'{name}.csv',index=False)
    grids=tables['grid_registry'];names=grids.set_index('grid_id').grid_name.to_dict()
    def show(frame):
        extra={'grid_id':'Ô lưới','grid_name':'Tên khu vực','grid_a':'Ô thứ nhất','grid_b':'Ô thứ hai',
               'expected_cycles':'Chu kỳ cần có','actual_cycles':'Chu kỳ thu được','missing_cycles':'Chu kỳ thiếu',
               'coverage_pct':'Độ đầy đủ (%)','first_run':'Khởi tạo đầu (UTC)','last_run':'Khởi tạo cuối (UTC)',
               'run_count':'Số chu kỳ','heavy_only_a':'Chỉ ô thứ nhất có mưa mạnh','heavy_only_b':'Chỉ ô thứ hai có mưa mạnh',
               'heavy_both':'Cả hai ô có mưa mạnh','heavy_jaccard':'Tỷ lệ giao/hợp mưa mạnh',
               'season':'Mùa','provenance':'Giới hạn nguồn dữ liệu','control_set':'Các yếu tố đã xét',
               'arm':'Phương án','grid_ids':'Các ô lưới','grid_count':'Số ô','continuous_forecast_features':'Số giá trị mưa',
               'missing_mask_features':'Số cờ thiếu','age_features':'Số đặc trưng tuổi dự báo',
               'water_station_name':'Trạm nước','water_station_id':'Mã trạm nước',
               'local_proxy_distance_km':'Khoảng cách tới ô gần nhất (km)',
               'reason':'Lý do','confidence':'Độ tin cậy','elevation_m':'Cao độ điểm trạm mưa (m)',
               'sample_n':'Số cặp hợp lệ','max_abs_difference_mm':'Chênh lệch lớn nhất (mm)',
               'mismatch_n':'Số giá trị không khớp','common_n':'Số giờ chung','initialized_at':'Khởi tạo (UTC)',
               'episode':'Mã đợt mưa','window':'Khoảng dự báo (giờ)',
               'location_id':'Ô lưới','request_ids':'Mã vị trí đã truy vấn',
               'pair_n':'Số cặp ô','min_pearson':'Pearson thấp nhất','max_pearson':'Pearson cao nhất',
               'median_pearson':'Pearson trung vị','only_one_heavy_total':'Tổng lượt chỉ ô thứ nhất có mưa mạnh',
               'mean_abs_r':'Trung bình độ lớn r','min_r':'r nhỏ nhất','max_r':'r lớn nhất',
               'event_n':'Số đợt có cặp hợp lệ','skill_defined_events':'Số đợt tính được đối chứng',
               'positive_skill_fraction':'Tỷ lệ đợt tốt hơn dự báo không mưa',
               'event_rmse_min_mm':'RMSE nhỏ nhất giữa các đợt (mm)',
               'event_rmse_max_mm':'RMSE lớn nhất giữa các đợt (mm)'}
        def translate(v):
            if not isinstance(v,str):return v
            for gid,name in names.items():v=v.replace(gid,name)
            return {'water_and_observed_rain':'Lịch sử nước và mưa quan trắc',
                    'plus_existing_a_luoi':'Thêm bốn khoảng dự báo từ ô A Lưới',
                    'Tentative; Stage 4 chưa thực hiện':'Ứng viên; chưa kiểm chứng bằng mô hình dự báo',
                    'all':'Toàn phần dữ liệu','flood_months':'Tháng 9–12','other_months':'Tháng 1–8'}.get(v,v)
        display=frame.copy()
        for c in display.columns:display[c]=display[c].map(translate)
        return vietnamese_table(display.rename(columns=extra))
    # 1. Raster-backed map, showing centers rather than invented square cell boundaries.
    fig,ax=plt.subplots(figsize=(10,8))
    dem=ROOT/'data/processed/station_analysis/dem/hue_fabdem.tif'
    with rasterio.open(dem) as src:
        raster=src.read(1,out_shape=(450,450),masked=True)
        ax.imshow(raster,extent=[src.bounds.left,src.bounds.right,src.bounds.bottom,src.bounds.top],
                  cmap='terrain',vmin=0,vmax=1500,alpha=.6,origin='upper')
    rain=stations[stations.station_type.eq('rainfall')];water=stations[stations.station_type.eq('water_level')]
    ax.scatter(rain.longitude,rain.latitude,s=12,c='royalblue',alpha=.5,label='55 trạm mưa')
    ax.scatter(water.longitude,water.latitude,s=22,c='red',marker='s',label='19 trạm nước')
    ax.scatter(grids.grid_longitude,grids.grid_latitude,s=100,c='black',marker='*',label='Tâm ô dự báo')
    for r in grids.itertuples():
        ax.annotate(r.grid_name,(r.grid_longitude,r.grid_latitude),xytext=(5,7),textcoords='offset points',fontsize=8,
                    bbox={'facecolor':'white','alpha':.7,'edgecolor':'none'})
    ax.set_xlim(min(rain.longitude.min(),water.longitude.min())-.05,max(rain.longitude.max(),water.longitude.max())+.05)
    ax.set_ylim(min(rain.latitude.min(),water.latitude.min())-.05,max(rain.latitude.max(),water.latitude.max())+.05)
    ax.set_xlabel('Kinh độ');ax.set_ylabel('Vĩ độ');ax.set_title('Các tâm ô dự báo, mạng quan trắc và địa hình FABDEM')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(fd/'01_grid_map_dem.png',dpi=160);plt.close(fig)
    # 2. Monthly forecast availability by grid; each anchor is counted once per grid.
    selected=tables['forecast_anchor_availability']
    legacy=selected[selected.policy.eq('legacy_assumed_release')].copy()
    legacy['month']=legacy.anchor_utc.dt.tz_convert('Asia/Ho_Chi_Minh').dt.strftime('%Y-%m')
    monthly=legacy.groupby(['location_id','month']).complete_24h.mean().unstack('month')*100
    monthly.to_csv(td/'monthly_grid_anchor_coverage.csv')
    fig,ax=plt.subplots(figsize=(12,4));im=ax.imshow(monthly,vmin=0,vmax=100,cmap='Blues',aspect='auto')
    ax.set_yticks(range(len(monthly)),[names.get(s,s) for s in monthly.index]);ax.set_xticks(range(len(monthly.columns)),monthly.columns,rotation=45,ha='right')
    fig.colorbar(im,ax=ax,label='Mốc đủ dự báo 24 giờ (%)');ax.set_title('Độ đầy đủ theo ô và tháng; giả định sẵn có sau 6 giờ')
    fig.tight_layout();fig.savefig(fd/'02_grid_monthly_coverage.png',dpi=150);plt.close(fig)
    # 3. One predefined closest-gauge comparator per cell; never pick the best skill gauge.
    comparators=tables['gauge_comparators']
    primary=comparators.sort_values('grid_distance_km').drop_duplicates('location_id')[['location_id','gauge_id','gauge_name','grid_distance_km','elevation_m']]
    quality=tables['spatial_rain_skill']
    q=quality.merge(primary[['location_id','gauge_id']],on=['location_id','gauge_id'])
    q=q[q.observation_policy.eq('standard_5of6')&q.breakdown.eq('all')&q.n.gt(0)]
    q_display=q[q.scope.eq('evaluation')]
    mountain_comparison=q_display[q_display.window.eq('6_12h') & q_display.location_id.map(names).isin(['A Lưới','Bạch Mã thấp','Đỉnh Bạch Mã'])]
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,window in zip(axes,['0_3h','6_12h']):
        group=q[q.scope.eq('evaluation')&q.window.eq(window)].set_index('location_id').reindex(grids.grid_id)
        ax.bar(range(len(group)),group.rmse_mm)
        ax.set_xticks(range(len(group)),grids.grid_name,rotation=45,ha='right',fontsize=8)
        ax.set_ylabel('RMSE lượng mưa (mm)');ax.set_title('Khoảng '+window.replace('_','–').replace('h',' giờ'))
    fig.suptitle('Đánh giá thăm dò tại trạm mưa gần nhất; không chọn trạm bằng độ chính xác')
    fig.tight_layout();fig.savefig(fd/'03_spatial_rain_skill.png',dpi=150);plt.close(fig)
    # 4. Pairwise grid correlations on train, one selected vintage common across all cells.
    inter=tables['intergrid_correlations']
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,feature in zip(axes,['rain_0_3h_mm','rain_6_12h_mm']):
        part=inter[inter.scope.eq('train')&inter.season.eq('all')&inter.feature.eq(feature)]
        matrix=pd.DataFrame(np.eye(len(grids)),index=grids.grid_id,columns=grids.grid_id)
        for r in part.itertuples():matrix.loc[r.grid_a,r.grid_b]=r.pearson_r;matrix.loc[r.grid_b,r.grid_a]=r.pearson_r
        im=ax.imshow(matrix,vmin=-1,vmax=1,cmap='RdBu_r')
        ax.set_xticks(range(len(grids)),grids.grid_name,rotation=50,ha='right',fontsize=8)
        ax.set_yticks(range(len(grids)),grids.grid_name,fontsize=8);ax.set_title(feature.replace('rain_','Mưa ').replace('_','–').replace('h–mm',' giờ'))
    fig.subplots_adjust(left=.12,right=.84,bottom=.25,wspace=.6)
    cax=fig.add_axes([.88,.30,.015,.5])
    fig.colorbar(im,cax=cax,label='Tương quan Pearson trên phần huấn luyện')
    fig.savefig(fd/'04_intergrid_train_correlations.png',dpi=150);plt.close(fig)
    # 5. Train-only spatial spread: all seven known-at-anchor predictions.
    spread=tables['spatial_heterogeneity']
    part=spread[spread.feature.eq('rain_6_12h_mm')].set_index('anchor_utc')
    fig,ax=plt.subplots(figsize=(12,3.5));part.spatial_range_mm.plot(ax=ax,lw=.7)
    ax.set_ylabel('Khoảng chênh lệch lớn nhất (mm)');ax.set_xlabel('Mốc dự báo (UTC)')
    ax.set_title('Chênh lệch dự báo mưa 6–12 giờ giữa các ô, không phải sai số quan trắc')
    fig.tight_layout();fig.savefig(fd/'05_spatial_heterogeneity.png',dpi=150);plt.close(fig)
    # 6. Incremental linear associations after past observations AND old grid, by station.
    partial=tables['spatial_partial_associations']
    fig,axes=plt.subplots(1,2,figsize=(13,8),sharey=True);im=None
    for ax,horizon in zip(axes,[6,12]):
        part=partial[partial.scope.eq('train')&partial.control_set.eq('plus_existing_a_luoi')&partial.horizon_h.eq(horizon)]
        matrix=part.groupby(['water_station_id','grid_id']).partial_pearson_r.apply(lambda s:s.abs().mean()).unstack('grid_id')
        matrix=matrix.reindex(index=water.station_id,columns=grids.grid_id)
        im=ax.imshow(matrix.to_numpy(),vmin=0,vmax=.5,cmap='Blues',aspect='auto')
        ax.set_yticks(range(len(water)),water.name,fontsize=7)
        ax.set_xticks(range(len(grids)),grids.grid_name,rotation=55,ha='right',fontsize=8)
        ax.set_title(f'+{horizon} giờ: trung bình độ lớn tương quan đã điều chỉnh')
    axes[1].tick_params(axis='y',labelleft=False)
    fig.subplots_adjust(left=.28,right=.9,bottom=.20,wspace=.08)
    cax=fig.add_axes([.93,.28,.015,.5]);fig.colorbar(im,cax=cax,label='Trung bình |r| trên bốn khoảng mưa')
    fig.savefig(fd/'06_water_additional_associations.png',dpi=150);plt.close(fig)

    events=tables['rainfall_event_skill'].merge(primary[['location_id','gauge_id']],left_on=['grid_id','gauge_id'],right_on=['location_id','gauge_id'])
    event_summary=events[events.window.eq('0_3h')].groupby(['grid_id','scope','archive_kind']).agg(
        event_n=('episode','nunique'),skill_defined_events=('mse_skill_vs_zero',lambda s:int(s.notna().sum())),
        positive_skill_fraction=('mse_skill_vs_zero',lambda s:s.dropna().gt(0).mean()),
        event_rmse_min_mm=('rmse_mm','min'),event_rmse_max_mm=('rmse_mm','max')).reset_index()
    event_summary.to_csv(td/'event_skill_stability_summary.csv',index=False)
    arms=tables['candidate_arms']
    design=stage4_design(grids)
    expanded_design=design_section(design,td)
    specifications={
        'stage':'Stage 4 preparation only; not executed',
        'model_unchanged':'Model B and existing training datasets are protected',
        'forecast_model':'ecmwf_ifs','cell_selection':'land',
        'horizons_h':[6,12],'rain_windows_h':[[0,3],[3,6],[6,12],[12,24]],
        'cutoff':config['future_analysis']['grouping_end'],'target_purge_hours':24,
        'sequence_length_candidate':96,
        'synchronized_forecast_path':str(work/'synchronized_features.parquet'),
        'run_metadata_path':str(work/'runs.parquet'),
        'hourly_forecast_path':str(work/'points.parquet'),
        'forecast_history_policy':'Reuse existing all_timesteps/anchor_only contract explicitly; do not change it silently',
        'experiment_groups':{
            'E0':'Water history only, plus persistence evaluation',
            'E1':'Same water history + fixed train-selected observed rainfall',
            'E2':'Same water history + spatial forecast arm',
            'E3':'E1 + spatial forecast arm',
            'E4':'E3 removing only 12–24h forecast and corresponding missing mask',
            'E5':'E3 using Stage 2 train-selected observed rain gauges'},
        'arm_candidates':json.loads(arms.to_json(orient='records')),
        'extended_design':design,
        'rules':[
            'Freeze water and observed-rain features across W0–W7 comparisons; W5–W7 are explicitly scoped to Da Vien.',
            'Use common station-specific anchor intersection for all compared arms and both horizons.',
            'Choose latest eligible initialization common to every selected grid; never mix cycles.',
            'Use observed future rain only as verification target, never an input.',
            'Keep standard and strict rain policies separate; fit all scaling/imputation on train only.',
            'Keep forecast values, missing masks and a shared initialization-age scalar explicit.',
            'Keep provenance as audit/stratification metadata; do not encode future download dates as inputs.',
            'Separate hypothetical 6h availability from evidenced operational branch; hindcasts never become verified operational forecasts.',
            'Already explored evaluation period is not an independent final test.',
            'Assess RMSE/MAE/N200, high-water episodes, rapid rises, each station and horizon.',
            'Treat no-train stations as ineligible for local fitting; do not create train observations.',
        ],
    }
    (output/'stage4_spatial_ablation_spec.json').write_text(json.dumps(specifications,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    cutoff=pd.Timestamp(config['future_analysis']['grouping_end'])
    common_train=int((pd.to_datetime(synced.anchor_utc,utc=True)+pd.Timedelta(hours=12)<=cutoff).sum())
    common_eval=int((pd.to_datetime(synced.anchor_utc,utc=True)>cutoff).sum())
    coverage=tables['grid_run_coverage']
    inter_summary=inter[inter.scope.eq('train')&inter.season.eq('all')].groupby('feature').agg(
        pair_n=('pearson_r','size'),min_pearson=('pearson_r','min'),max_pearson=('pearson_r','max'),
        median_pearson=('pearson_r','median'),only_one_heavy_total=('heavy_only_a','sum')).reset_index()
    inter_summary.to_csv(td/'train_intergrid_summary.csv',index=False)
    first_interval=q[q.window.eq('0_3h')&q.scope.eq('evaluation')]
    signal_train=partial[partial.scope.eq('train')&partial.control_set.eq('plus_existing_a_luoi')].groupby('grid_id').partial_pearson_r.apply(lambda s:s.abs().mean()).sort_values(ascending=False)
    facts=f'Các ô có độ đầy đủ chu kỳ từ {coverage.coverage_pct.min():.2f}% đến {coverage.coverage_pct.max():.2f}%.'
    if len(first_interval):
        facts+=f' Khi đối chiếu mưa 0–3 giờ với trạm gần nhất, RMSE trên phần đánh giá dao động {first_interval.rmse_mm.min():.2f}–{first_interval.rmse_mm.max():.2f} mm; đây là các trạm đối chiếu khác nhau nên không phải bảng xếp hạng công bằng giữa vùng.'

    mapping=tables['request_mapping']
    mapping_summary=mapping.groupby('request_id').agg(unique_grids=('grid_id','nunique'),runs=('initialized_at','nunique')).reset_index()
    mapping_summary.to_csv(td/'mapping_stability.csv',index=False)
    dv=partial[partial.water_station_id.eq('0074647403')&partial.control_set.eq('plus_existing_a_luoi')]
    dv_summary=dv.groupby(['grid_id','scope','horizon_h']).agg(mean_abs_r=('partial_pearson_r',lambda s:s.abs().mean()),
             min_r=('partial_pearson_r','min'),max_r=('partial_pearson_r','max'),n=('n','min')).reset_index()
    dv_summary.to_csv(td/'da_vien_additional_signal.csv',index=False)
    dv_train12=dv_summary[dv_summary.scope.eq('train')&dv_summary.horizon_h.eq(12)]
    water_takeaway=''
    if len(dv_train12):
        water_takeaway=(f'Tại Dã Viên, sau khi đã xét lịch sử mực nước, mưa quan trắc và cả bốn khoảng dự báo của ô A Lưới, '
                       f'trung bình độ lớn tương quan riêng phần với biến động +12 giờ của các ô mới chỉ còn '
                       f'{dv_train12.mean_abs_r.min():.3f}–{dv_train12.mean_abs_r.max():.3f} trên phần huấn luyện. '
                       'Đây là tín hiệu bổ sung nhỏ trong phép kiểm tra tuyến tính này, chưa phải bằng chứng cải thiện mô hình. '
                       'Dấu của các hệ số tại một số ô thay đổi trên phần đánh giá, nên chưa thể coi quan hệ này ổn định theo thời gian.')
    empty_cycles=tables['forecast_run_integrity'].loc[tables['forecast_run_integrity'].valid_forecast_hours.eq(0),'initialized_at'].drop_duplicates()
    raw=tables['spatial_water_relationships']
    comparison_rows=[]
    for wid,part in raw[raw.condition.eq('all')&raw.target.isin(['dh_p6h','dh_p12h'])].groupby('water_station_id'):
        for gid,group in part.groupby('grid_id'):
            comparison_rows.append({'water_station_id':wid,'grid_id':gid,
                                    'train_mean_abs_r':group[group.scope.eq('train')].pearson_r.abs().mean(),
                                    'evaluation_mean_abs_r':group[group.scope.eq('evaluation')].pearson_r.abs().mean()})
    pd.DataFrame(comparison_rows).to_csv(td/'water_grid_scope_summary.csv',index=False)
    water_comparisons=water_comparison_section(raw,partial,grids,stations,arms,td)
    text=f'''# Giai đoạn 3B — Dự báo mưa theo không gian và định hướng mô hình vùng

## 1. Mục tiêu, phạm vi và điều chưa được kết luận

Phân tích này mở rộng nguồn dự báo từ một ô gần Hồng Thái ở A Lưới sang {len(grids)} ô tại các khu vực đã xác định trước ở Huế. Mục tiêu là kiểm tra độ đầy đủ, đối chiếu lượng mưa, mô tả khác biệt giữa các ô và tìm ứng viên cho thí nghiệm dự báo nước +6/+12 giờ. Không huấn luyện XGBoost, LSTM hoặc thay đổi mô hình B trong giai đoạn này.

Mô hình khí tượng được giữ đúng `ecmwf_ifs`; cách chọn ô là `land`. Tọa độ yêu cầu, tâm ô trả về và trạm quan trắc là ba đối tượng riêng. Hai yêu cầu trả cùng ô và cùng lần chạy chỉ đóng góp một chuỗi dự báo; nếu giá trị xung đột, chương trình dừng để kiểm tra thay vì lấy trung bình.

Khoảng mốc dự báo được phân tích bắt đầu từ {pd.Timestamp(config['start']).tz_convert('Asia/Ho_Chi_Minh')}; giới hạn cuối theo bản dữ liệu quan trắc cố định và cấu hình kết thúc. Với cấu hình mặc định, phạm vi giữ như các phần trước: 15/11/2025 đến 21/10/2026 theo giờ Việt Nam. Các chu kỳ được chuẩn hóa sang UTC trước khi tạo yêu cầu. Mỗi lần chạy có bản dự báo theo giờ riêng, không thay bằng chuỗi thời tiết lịch sử đã ghép hay tái phân tích khí tượng.

Các kết quả trong báo cáo là **nghiên cứu hồi cứu**, chủ yếu dùng giả định dự báo sẵn có sau khởi tạo 6 giờ. Dự báo tái dựng trước chuyển tiếp phiên bản và dữ liệu lưu trữ sau chuyển tiếp được giữ nhãn nguồn riêng. Thời điểm tải thêm dữ liệu hiện nay không chứng minh chúng đã sẵn có tại các mốc dự báo trong quá khứ. Bảng kiểm tra khả dụng thực tế tiếp tục được giữ riêng, không được diễn giải như có độ chính xác bằng 0 khi thiếu mẫu.

Kết quả kiểm tra thực tế: **{facts}** Các ô có chung một mô hình khí tượng nhưng mức độ đồng biến không giống nhau; bảng dưới ghi định lượng trên phần huấn luyện, không dùng kết quả đánh giá để chọn ô.

{show(inter_summary)}

{water_takeaway}

## 2. Kiểm tra nhỏ trước khi thu thập lịch sử

Sáu lần chạy đại diện đã được truy vấn thành công, gồm đầu khoảng nghiên cứu, một đợt mưa năm 2025, hai phía chuyển tiếp 12/05/2026 và hai lần chạy tháng 10/2026. Ba lần chạy tại ô A Lưới được đối chiếu với bản dữ liệu cố định Giai đoạn 3: tất cả giá trị chung và trạng thái thiếu đều khớp. Bảng sau kiểm tra thêm các lần chạy trùng giữa dữ liệu mới và nguồn cũ.

{show(tables['reference_replay'].head(12))}

Tài liệu nhà cung cấp xác định trước ngày 12/05/2026 06:00 UTC là dữ liệu tái dựng phiên bản 49r1, sau đó dùng phiên bản 50r1. Nhãn phiên bản ở đây dựa trên chuyển tiếp chính thức, không phải trường phiên bản do từng phản hồi JSON tự trả về. Mốc khởi tạo không được coi là mốc công bố.

Bộ tải dùng yêu cầu nhiều tọa độ cho cùng một lần chạy, lưu nguyên phản hồi, có thử lại và lưu tiến độ. Việc giới hạn tốc độ tính bảo thủ mỗi tọa độ là một lượt sử dụng; có theo dõi giới hạn theo phút, giờ và ngày. Dữ liệu nguồn không bị ghi đè. Mỗi lần xử lý tạo một thư mục phiên bản mới để kiểm tra lại.

## 3. Các ô lưới và độ ổn định của ánh xạ tọa độ

{show(grids[['grid_id','grid_name','grid_latitude','grid_longitude','request_ids','first_run','last_run']])}

![Tâm ô, trạm quan trắc và địa hình](figures/01_grid_map_dem.png)

Bản đồ vẽ tâm các ô, không tự tạo các hình vuông 9 km rồi coi đó là ranh giới ô thực. Địa hình lấy lại từ FABDEM đã xử lý; không thay đổi DEM gốc. Tọa độ trạm mưa và trạm nước lấy từ cùng danh mục đã đóng băng.

[Bảng ánh xạ đầy đủ](tables/request_mapping.csv) lưu từng yêu cầu, tọa độ gửi, tọa độ nhận, thời điểm khởi tạo, mã lần chạy, tham số truy vấn mới và mã kiểm tra nội dung. Nguồn A Lưới tái sử dụng được truy ngược tới mã lần chạy trong bản dữ liệu cố định; bản cố định không chứa đầy đủ tham số yêu cầu gốc, nên trường đó được để thiếu thay vì tự tạo lại như một bằng chứng đã ghi nhận. [Bảng độ ổn định](tables/mapping_stability.csv) cho biết mỗi yêu cầu đã trả bao nhiêu tâm ô khác nhau. Vì ánh xạ được kiểm tra theo lịch sử, chương trình không tự coi một tọa độ phát hiện hôm nay là đúng cho mọi năm.

Cách chọn `nearest` chỉ được xem là kiểm tra độ nhạy nhỏ qua phản hồi đã lưu ở lần xác minh ô lưới trước. [Bảng kiểm tra cách chọn ô](tables/nearest_sensitivity.csv) ghi riêng từng tọa độ và hai tâm trả về. Cách này có thể chọn ô khác ở Nhâm/Hồng Thái, nhưng chưa có đủ lịch sử và đối chứng để khẳng định tốt hơn. Tập dữ liệu và mọi phân tích chính ở đây tiếp tục dùng `land`; không nhân đôi toàn bộ thí nghiệm theo cách chọn ô.

## 4. Độ đầy đủ và nguồn gốc dự báo

{show(coverage)}

{show(tables['missing_grid_runs'])}

Những chu kỳ có phản hồi nhưng toàn bộ giá trị mưa bị thiếu là: {', '.join(map(str,empty_cycles)) or 'không có'}. Đây là khoảng thiếu chung cho các ô, được giữ trong bảng kiểm tra thay vì thay bằng 0. Nhờ còn chu kỳ trước đủ tuổi và đủ giờ dự báo, bảng đồng bộ vẫn bao phủ toàn bộ {len(synced)} mốc nghiên cứu; điều đó không có nghĩa bản lưu trữ có đầy đủ mọi chu kỳ.

[Bảng lần chạy còn thiếu](tables/missing_grid_runs.csv) chỉ rõ từng ô và mốc khởi tạo bị thiếu. Bộ tải ghi nhận {len(progress['failed_batches'])} lô yêu cầu không hoàn thành sau các lần thử lại. Khoảng thiếu không được thay bằng quan trắc hoặc một loại sản phẩm khác. Khi đủ điều kiện về tuổi dự báo, lựa chọn có thể lùi về một lần chạy cũ hơn, nhưng cùng một vector nhiều ô phải cùng lùi về một mốc khởi tạo.

![Độ đầy đủ theo ô và tháng](figures/02_grid_monthly_coverage.png)

[Bảng tổng hợp dữ liệu thiếu và sai lệch](tables/grid_integrity_summary.csv) phân biệt một lần chạy có bản ghi nhưng không có giá trị mưa hợp lệ với một chu kỳ hoàn toàn không truy xuất được. [Kiểm tra từng lần chạy](tables/forecast_run_integrity.csv) bao gồm đơn vị, giờ hiệu lực, bản ghi trùng, giá trị âm, giờ thiếu và nguồn gốc. [Độ bao phủ theo chính sách khả dụng](tables/forecast_coverage.csv) tách cách có bằng chứng thực tế khỏi cách giả định sau 6 giờ. [Chu kỳ và khoảng gián đoạn](tables/forecast_cycle_inventory.csv) giữ đúng thông tin từng nguồn.

Các giá trị dự báo theo giờ biểu diễn mưa trong giờ kết thúc tại thời điểm hiệu lực. Bốn khoảng lần lượt dùng t+1..3, t+4..6, t+7..12 và t+13..24. Mỗi ô dùng một lần chạy cho tất cả khoảng; các phép so sánh giữa ô và tổ hợp ứng viên còn yêu cầu cùng thời điểm khởi tạo trên tất cả ô. Sau khi xét cả độ đầy đủ và tuổi không quá {config['max_forecast_age_hours']} giờ, bảng đồng bộ có **{len(synced)} mốc**, gồm **{common_train} mốc thuộc phần huấn luyện sau loại nhãn vượt ranh giới** và **{common_eval} mốc đánh giá thăm dò**. Những số này là kích thước bảng dự báo; số cặp với mực nước còn tùy dữ liệu thiếu ở từng trạm.

Nguồn cũ tại A Lưới được tái sử dụng khi có thể. Khi thu thập thêm các chu kỳ trước đây chưa có, số mẫu có thể tăng so với Giai đoạn 3; vì vậy không lấy thay đổi chỉ số giữa hai giai đoạn làm bằng chứng riêng về lợi ích của thêm ô lưới.

## 5. Đối chiếu dự báo với mưa đo tại các khu vực

Mỗi ô được so với các trạm mưa gần nó theo khoảng cách địa lý. Bảng đối chiếu lưu cao độ tại điểm trạm và khoảng cách tới tâm ô. Cao độ điểm trạm được so với cao độ DEM tại tâm ô trong bảng đầy đủ; cao độ tại tâm vẫn không phải cao độ trung bình địa hình của cả ô hoặc cao độ mà mô hình khí tượng sử dụng. Đây chưa phải bằng chứng trạm đại diện hoàn toàn cho ô hoặc cùng lưu vực; không tính trung bình các trạm rồi gọi đó là lượng mưa thực của ô.

{show(primary)}

Dữ liệu quan trắc giữ hai cách xử lý thiếu như Giai đoạn 2–3: cho phép ước tính giờ có 5/6 bản ghi và đối chứng chỉ dùng giờ đủ 6/6. Mỗi tổng mưa tương lai phải đủ mọi giờ trong đúng khoảng hiệu lực. Không nội suy hoặc thay giờ thiếu bằng 0. Phần huấn luyện cho đánh giá mưa kết thúc sớm đủ 24 giờ trước mốc chia, để không dùng quan trắc thuộc phần đánh giá khi xác định dữ liệu huấn luyện.

Bảng dưới chỉ lấy **trạm gần nhất được chọn trước bằng khoảng cách** cho mỗi ô nhằm trình bày ngắn gọn. Bảng đầy đủ vẫn giữ mọi trạm đối chiếu riêng. Không chọn trạm có sai số nhỏ nhất để làm đẹp kết quả.

{show(q_display[['location_id','gauge_id','archive_kind','scope','window','n','rmse_mm','mae_mm','bias_mm','wet_f1','heavy_recall','mse_skill_vs_zero']])}

Bảng chính trình bày phần đánh giá thăm dò; kết quả huấn luyện 49r1 và 50r1 được giữ riêng trong bảng đầy đủ, không gộp thành một ước lượng chất lượng duy nhất.

![Sai số tại các khu vực](figures/03_spatial_rain_skill.png)

MAE là sai số tuyệt đối trung bình; RMSE là căn bậc hai của trung bình bình phương sai số. Độ lệch trung bình là lượng dự báo trừ lượng quan trắc. Các chỉ số này có đơn vị mm của tổng mưa mỗi khoảng; không so tổng 3 giờ với tổng 12 giờ như cùng một bài toán. F1 đánh giá nhận diện có mưa với ngưỡng {config['weather_analysis']['wet_threshold_mm']} mm. Mưa mạnh dùng cường độ trung bình khoảng ít nhất {config['weather_analysis']['heavy_mean_rate_mm_h']} mm/giờ. Tỷ lệ phát hiện để trống khi không có trường hợp thực tế để xác định mẫu số. Đối chứng không mưa được tính trên đúng những cặp đó.

[Đối chiếu hai ô với cùng trạm mưa](tables/same_gauge_grid_comparisons.csv) chỉ tính các trạm xuất hiện trong danh sách lân cận của cả hai ô. Phép đối chiếu này giữ cùng khoảng mưa, cùng chu kỳ và đúng cùng các mốc giờ, giúp tránh việc chênh lệch số mẫu hoặc dùng hai trạm khác nhau tạo ra khác biệt chỉ số. Chênh lệch RMSE được tính là ô thứ hai trừ ô thứ nhất; số âm nghĩa ô thứ hai có sai số nhỏ hơn trong đúng phép so sánh đó.

Riêng câu hỏi Bạch Mã khác A Lưới thế nào, bảng sau giữ cùng khoảng mưa 6–12 giờ và cùng giai đoạn đánh giá. Mỗi hàng vẫn đối chiếu với trạm gần ô tương ứng, nên nó mô tả sự khác biệt quan sát được chứ chưa cô lập tác động của vùng địa lý.

{show(mountain_comparison[['location_id','gauge_id','n','rmse_mm','bias_mm','heavy_recall']])}

Các khác biệt giữa miền núi, đồng bằng và ven biển mới là mô tả của các ô và trạm đang so. Chúng đồng thời chịu ảnh hưởng địa hình, khoảng cách, phân bố mưa và loại nguồn lưu trữ, nên không khẳng định có khác biệt có ý nghĩa thống kê giữa toàn bộ các vùng chỉ từ một năm dữ liệu.

[Bảng theo mùa và tuổi dự báo](tables/spatial_rain_skill.csv) và [bảng từng đợt mưa](tables/rainfall_event_skill.csv) cho phép kiểm tra sự ổn định theo thời gian. Các đợt mưa được lấy từ thuật toán của Giai đoạn 2 trên trung bình mạng trạm, không phải lượng mưa trung bình theo diện tích lưu vực hoặc danh mục trận lũ độc lập đã xác minh. Các cửa sổ kiểm chứng được giữ trong phạm vi đợt; đây là phân tích hồi cứu, không dùng để tối ưu ứng viên ngoài phần huấn luyện.

Bảng sau tóm tắt riêng khoảng 0–3 giờ tại trạm đối chiếu gần nhất. Tỷ lệ đợt tốt hơn dự báo không mưa chỉ tính trên những đợt có mẫu số đối chứng xác định; không coi một đợt không có mưa tại trạm là đối chứng hoàn hảo. Khoảng RMSE cho thấy mức biến động giữa các đợt, không phải khoảng tin cậy.

{show(event_summary)}

Các cửa sổ dự báo của những mốc giờ liền nhau có thể chồng lấn và các trạm cùng chịu một hình thế thời tiết. Số cặp lớn vì thế không đồng nghĩa với nhiều sự kiện độc lập. Báo cáo trình bày thống kê mô tả, không tuyên bố khác biệt có ý nghĩa thống kê dựa trên giả định các cặp này độc lập.

## 6. Các ô khác nhau đến mức nào?

![Tương quan giữa các ô trên phần huấn luyện](figures/04_intergrid_train_correlations.png)

[Bảng tương quan và mưa mạnh](tables/intergrid_correlations.csv) ghi riêng phạm vi thời gian, mùa, cửa sổ mưa, số cặp, Pearson, Spearman, số giờ chỉ một ô dự báo mưa mạnh và tỷ lệ giao/hợp giữa hai tập giờ mưa mạnh. Những phép tính này dùng cùng chu kỳ và cùng mốc giờ, không trộn các lần chạy khác nhau giữa hai ô.

Tương quan cao cho thấy hai ô thường biến đổi cùng nhau, nhưng không chứng minh có thể bỏ một ô mà không ảnh hưởng dự báo nước. Tương quan thấp và những giờ chỉ một ô dự báo mưa mạnh cho thấy sự khác nhau trong tín hiệu khí tượng; chúng chưa chứng minh giá trị dự báo bổ sung cho mực nước.

![Mức phân hóa không gian của mưa dự báo](figures/05_spatial_heterogeneity.png)

Đường trên biểu diễn khoảng chênh lệch lớn nhất giữa các ô trong cùng khoảng dự báo 6–12 giờ. Đây là chênh lệch giữa dự báo, không phải sai số đối với mưa thực tế hoặc mức nước dâng đã đo.

## 7. Quan hệ với mực nước tại từng trạm

[Bảng quan hệ đầy đủ](tables/spatial_water_relationships.csv) gồm mực nước hiện tại, biến thiên +6h/+12h, từng ô và bốn khoảng mưa; được tách theo phần huấn luyện, đánh giá, tháng mùa lũ, nước đang dâng theo quá khứ và biến động mạnh trong tương lai. Nhóm cuối chỉ dùng mô tả sau khi sự kiện đã xảy ra, không làm cổng chọn mô hình lúc dự báo.

Tương quan có điều chỉnh xét H(t), thay đổi nước 3h/6h, tổng mưa 6h/12h tại hai trạm mưa đã tuyển chọn bằng tập huấn luyện. Một phép kiểm tra bổ sung còn xét cả bốn khoảng dự báo của ô A Lưới cũ trước khi tính phần liên hệ còn lại của các ô mới. Mọi hệ số điều chỉnh và chuẩn hóa đều chỉ ước lượng trên tập huấn luyện, rồi giữ cố định cho phần đánh giá. Không chạy mô hình dự báo nước hoặc điều chỉnh siêu tham số để tạo các bảng này.

![Liên hệ còn lại ngoài dữ liệu quá khứ và ô A Lưới](figures/06_water_additional_associations.png)

Bảng dưới minh họa Dã Viên. Trung bình độ lớn r là tóm tắt bốn khoảng mưa, không phải điểm chất lượng mô hình và không thể so trực tiếp với RMSE. Các dấu và trị số từng khoảng vẫn được giữ trong bảng đầy đủ.

{show(dv_summary)}

Hai trạm nước không có đủ dữ liệu huấn luyện không được ước lượng phép điều chỉnh bằng dữ liệu đánh giá. Đối với những trạm khác, r riêng phần khác 0 mới chỉ là liên hệ sau bộ yếu tố tuyến tính đã xét, chưa chứng minh thông tin nhân quả hoặc giảm lỗi của Regional ResidualLSTM. Việc chuyển nguồn 49r1/50r1 cùng thời gian có thể gây ảnh hưởng; kết quả hỗn hợp nguồn không được xem là độc lập với chuyển tiếp đó.

{water_comparisons}

## 8. Các phương án ô dự báo cho Giai đoạn 4

[Danh sách ứng viên ban đầu của 19 trạm](tables/candidate_arms.csv) giữ nguyên W0–W4, được xác định từ vị trí và kết quả trên phần huấn luyện. W2 là ô gần trạm nhất trong tập đã thu thập; không mặc định đúng ô API trả khi truy vấn chính xác tọa độ từng trạm. W3 là tổ hợp ứng viên tuyển chọn trên train, không mặc định các ô đó thuộc vùng đầu nguồn. Các phương án dưới mở rộng thiết kế thí nghiệm, không sửa dữ liệu, kết quả EDA hoặc mô hình hiện có.

Mỗi ô thêm bốn giá trị mưa và bốn cờ thiếu dữ liệu. Khi dùng chung chu kỳ, tuổi dự báo là một giá trị chung cho cả tổ hợp. Nguồn gốc được giữ để kiểm toán và phân tầng; không đưa thời điểm tải dữ liệu hiện nay vào đầu vào như thông tin đã biết trong quá khứ.

{expanded_design}

## 9. Đặc tả thí nghiệm tiếp theo và giới hạn kết luận

[Đặc tả máy đọc được](stage4_spatial_ablation_spec.json) giữ E0–E5 theo master plan, W0–W4 cho các trạm và ma trận W0–W7 riêng cho Dã Viên cùng các đối chứng D1–D9. Phần `extended_design` ghi đầy đủ phạm vi, seed và quy tắc tuyển chọn; `arm_candidates` tiếp tục là danh sách W0–W4 ban đầu của 19 trạm. Mỗi cặp phương án phải giữ cùng lịch sử nước, cùng mưa quan trắc, cùng mốc dự báo và cùng nhãn +6/+12 giờ. Cần có đối chứng giữ nguyên mực nước hiện tại, đối chứng chỉ dùng lịch sử nước và đánh giá riêng các đợt nước tăng nhanh. Các cửa sổ 96 giờ là ứng viên cấu hình để kiểm chứng, chưa được sử dụng để huấn luyện ở Giai đoạn 3B.

Mốc chia là **{config['future_analysis']['grouping_end']}**. Với nhãn nước, tập huấn luyện phải thỏa t+12h≤mốc chia; phần sau đã được xem nhiều lần nên chỉ là đánh giá thăm dò, không còn kiểm thử độc lập cuối cùng. Các trạm thiếu dữ liệu huấn luyện không được tự tạo quan trắc hoặc dùng phần đánh giá để bổ sung train.

Những kết luận được dữ liệu hỗ trợ gồm: khả năng truy xuất các ô và lần chạy đã lưu, độ đầy đủ thực tế, mức chênh lệch dự báo giữa ô và các thống kê đối chiếu theo phạm vi cụ thể. Việc ô nào giúp mô hình nước tốt hơn, tổ hợp nào tối ưu, hay nearest có tốt hơn land vẫn là giả thuyết cần kiểm chứng ở Giai đoạn 4. Không suy ra hiệu quả vận hành thực tế từ tương quan hoặc sai số mưa hồi cứu.

## 10. Nguồn, dữ liệu và cách chạy lại

Dữ liệu nghiên cứu đã xử lý nằm trong `{work}`. Phản hồi gốc, tham số yêu cầu, thời điểm tải và tiến độ nằm trong thư mục thu thập riêng `collection_v1`. [Hồ sơ thực thi](manifest.json) ghi cấu hình, phiên bản mã nguồn và mã kiểm tra nội dung các đầu vào. Bản dữ liệu cũ, PostgreSQL quan trắc, DEM gốc, các bộ trọng số mô hình và tập huấn luyện đều được giữ nguyên.

Tài liệu chính thức: [ECMWF API](https://open-meteo.com/en/docs/ecmwf-api), [truy xuất từng lần chạy](https://open-meteo.com/en/docs/single-runs-api), [thời điểm cập nhật](https://open-meteo.com/en/docs/model-updates), [hạn mức dịch vụ](https://open-meteo.com/en/terms). Tài liệu giúp xác định ý nghĩa trường và sản phẩm; không thay bằng chứng về thời điểm công bố của từng bản ghi lịch sử. Quy ước mưa quan trắc 10 phút tiếp tục là hợp đồng dữ liệu của dự án như Giai đoạn 2, chưa được xác minh độc lập bằng hồ sơ thiết bị trong lượt này.
'''
    report=output/'report.md';report.write_text(text,encoding='utf-8')
    return report
