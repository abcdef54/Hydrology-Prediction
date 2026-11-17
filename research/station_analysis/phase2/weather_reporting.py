"""Readable Stage 3 findings with computed tables and clearly separated evidence."""
from pathlib import Path
import numpy as np
import pandas as pd
from .stage2_report import markdown_table


def vietnamese_table(frame):
    """Translate report labels without changing machine-readable analysis tables."""
    labels = {
        'policy':'Cách xác định thời điểm sẵn có', 'anchor_count':'Số mốc giờ',
        'selected_n':'Mốc có dự báo', 'complete24_n':'Mốc đủ dự báo 24 giờ',
        'hindcast_selected_n':'Mốc dùng dự báo tái dựng',
        'retrieved_after_anchor_n':'Mốc tải dữ liệu sau thời điểm dự báo',
        'age_median_h':'Tuổi dự báo trung vị (giờ)', 'age_p95_h':'Phân vị 95% tuổi dự báo (giờ)',
        'window':'Khoảng dự báo (giờ)', 'gauges':'Mã trạm mưa đối chiếu',
        'rmse_min_mm':'RMSE thấp nhất (mm)', 'rmse_max_mm':'RMSE cao nhất (mm)',
        'bias_min_mm':'Độ lệch trung bình thấp nhất (mm)', 'bias_max_mm':'Độ lệch trung bình cao nhất (mm)',
        'horizon_h':'Dự báo trước (giờ)', 'feature':'Đặc trưng', 'n':'Số cặp hợp lệ',
        'partial_pearson_r':'Tương quan Pearson sau khi xét các yếu tố quá khứ',
        'partial_spearman_r':'Tương quan Spearman sau khi xét các yếu tố quá khứ',
        'pearson_r':'Tương quan Pearson', 'spearman_r':'Tương quan Spearman',
        'scope':'Phần dữ liệu', 'train_fit_n':'Số mẫu dùng ước lượng trên tập huấn luyện',
        'location_id':'Mã yêu cầu tải', 'model':'Mô hình khí tượng', 'archive_kind':'Loại dữ liệu lưu trữ',
        'availability_basis':'Căn cứ xác định thời điểm sẵn có', 'run_count':'Số bản ghi lần chạy',
        'point_rows':'Số bản ghi theo giờ', 'valid_forecast_hours':'Số bản ghi hợp lệ',
        'first_initialized_at':'Khởi tạo đầu tiên (UTC)', 'last_initialized_at':'Khởi tạo cuối cùng (UTC)',
        'strict_eligible_runs':'Lần chạy có bằng chứng khả dụng', 'missing_available_at_runs':'Lần chạy thiếu thời điểm sẵn có',
        'hindcast_runs':'Lần chạy dự báo tái dựng', 'grid_coordinate_pairs':'Số tọa độ lưới trả về',
        'requested_latitude':'Vĩ độ yêu cầu', 'requested_longitude':'Kinh độ yêu cầu',
        'grid_latitude':'Vĩ độ lưới trả về', 'grid_longitude':'Kinh độ lưới trả về',
        'request_to_grid_distance_km':'Khoảng cách yêu cầu–lưới (km)',
        'locations_sharing_returned_grid':'Số yêu cầu chung điểm lưới', 'shared_location_ids':'Các yêu cầu chung điểm lưới',
        'interpretation':'Ý nghĩa', 'location_a':'Yêu cầu thứ nhất', 'location_b':'Yêu cầu thứ hai',
        'common_n':'Số mốc chung', 'equal_within_1e_9':'Số mốc có giá trị trùng nhau',
        'max_abs_difference_mm':'Chênh lệch tuyệt đối lớn nhất (mm)',
        'gauge_id':'Mã trạm mưa', 'gauge_name':'Tên trạm mưa',
        'primary_reference_gauge':'Trạm đối chiếu chính', 'grid_distance_km':'Khoảng cách tới điểm lưới (km)',
        'comparison_support':'Phạm vi đối chiếu', 'mae_mm':'MAE (mm)', 'rmse_mm':'RMSE (mm)',
        'bias_mm':'Độ lệch trung bình (mm)', 'mse_skill_vs_zero':'Cải thiện MSE so với dự báo không mưa',
        'wet_f1':'F1 nhận diện có mưa', 'heavy_recall':'Tỷ lệ phát hiện mưa mạnh',
        'status':'Đánh giá', 'reason':'Căn cứ', 'next_step':'Việc cần kiểm tra tiếp',
    }
    values={
        'legacy_assumed_release':'Giả định sẵn có sau khởi tạo 6 giờ',
        'strict_operational':'Có bằng chứng sẵn có thực tế', 'train':'Huấn luyện', 'evaluation':'Đánh giá thăm dò',
        'unknown_archive':'Chưa xác minh thời điểm công bố', 'observed_retrieval':'Có ghi nhận thời điểm tải thực tế',
        'hindcast_49r1':'Dự báo tái dựng, phiên bản 49r1',
        'archive_50r1_unverified':'Dữ liệu lưu trữ 50r1, chưa xác minh thời điểm công bố',
        'operational_single_run':'Dự báo của một lần chạy vận hành',
        'rain_0_3h_mm':'Mưa dự báo 0–3 giờ', 'rain_3_6h_mm':'Mưa dự báo 3–6 giờ',
        'rain_6_12h_mm':'Mưa dự báo 6–12 giờ', 'rain_12_24h_mm':'Mưa dự báo 12–24 giờ',
        '0_3h':'0–3', '3_6h':'3–6', '6_12h':'6–12', '12_24h':'12–24',
        'point gauge; grid-area representativeness unverified':'Trạm đo tại điểm; chưa xác minh tính đại diện cho ô lưới',
        'grid-model precipitation; not a gauge measurement or citywide spatial mean':'Mưa từ lưới mô hình khí tượng; không phải số đo tại trạm hay trung bình toàn thành phố',
    }
    shown=frame.copy()
    for column in shown.columns:
        shown[column]=shown[column].map(lambda v: values.get(v,v) if isinstance(v,str) else ('Có' if v is True else 'Không' if v is False else v))
    return markdown_table(shown.rename(columns=labels))


def write_weather_report(tables,pairs,frames,config,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);table_dir=output/'tables';figure_dir=output/'figures'
    table_dir.mkdir(exist_ok=True);figure_dir.mkdir(exist_ok=True)
    for name,frame in tables.items():frame.to_csv(table_dir/f'{name}.csv',index=False)
    cov=tables['weather_city_coverage']
    primary=config['weather_analysis']['reference_gauge_ids']
    skill=tables['weather_skill']
    primary_skill=skill[skill.gauge_id.isin(primary)&skill.observation_policy.eq('standard_5of6')&skill.breakdown.eq('all')&skill.n.gt(0)]
    raw=tables['weather_water_correlations']
    partial=tables['weather_partial_correlations']
    dv_raw=raw[raw.water_station_id.eq('0074647403')&raw.condition.eq('all')&raw.feature.ne('age_hours')]
    dv_partial=partial[partial.water_station_id.eq('0074647403')&partial.feature.ne('age_hours')&partial.cohort.eq('all_assumed_release')]
    names=frames['stations'].set_index('station_id').name.to_dict()
    chosen=tables['weather_city_selection']
    legacy=chosen[chosen.policy.eq('legacy_assumed_release')].set_index('anchor_utc')
    fig,ax=plt.subplots(figsize=(11,3.5))
    for column in ['rain_0_3h_mm','rain_3_6h_mm','rain_6_12h_mm','rain_12_24h_mm']:
        legacy[column].resample('ME').apply(lambda s:s.notna().mean()*100).plot(ax=ax,label={'rain_0_3h_mm':'0–3 giờ','rain_3_6h_mm':'3–6 giờ','rain_6_12h_mm':'6–12 giờ','rain_12_24h_mm':'12–24 giờ'}[column])
    ax.set_ylabel('Mốc giờ có dự báo hợp lệ (%)');ax.set_title('Tỷ lệ có dự báo chung theo tháng, theo giả định sẵn có sau 6 giờ')
    ax.set_xlabel('Tháng')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(figure_dir/'01_city_availability.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
    for ax,scope in zip(axes,['train','evaluation']):
        part=primary_skill[primary_skill.scope.eq(scope)]
        for (gauge,archive),group in part.groupby(['gauge_id','archive_kind']):
            group=group.set_index('window').reindex(['0_3h','3_6h','6_12h','12_24h'])
            ax.plot(range(4),group.rmse_mm,'o-',label=f"{names.get(gauge,gauge)} / {'49r1 tái dựng' if 'hindcast' in archive else '50r1 lưu trữ'}")
        ax.set_xticks(range(4),['0–3','3–6','6–12','12–24']);ax.set_title('Huấn luyện' if scope=='train' else 'Đánh giá thăm dò');ax.set_xlabel('Khoảng dự báo (giờ)')
        ax.legend(fontsize=7)
    axes[0].set_ylabel('RMSE lượng mưa (mm)');fig.suptitle('Đối chiếu hồi cứu dự báo lưới với trạm mưa, tách theo nguồn')
    fig.tight_layout();fig.savefig(figure_dir/'02_rainfall_skill.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,10),sharey=True)
    image=None
    for i,horizon in enumerate([6,12]):
        for j,scope in enumerate(['train','evaluation']):
            sub=partial[partial.horizon_h.eq(horizon)&partial.scope.eq(scope)&partial.feature.ne('age_hours')&partial.cohort.eq('all_assumed_release')]
            matrix=sub.pivot(index='water_station_name',columns='feature',values='partial_pearson_r').reindex(columns=['rain_0_3h_mm','rain_3_6h_mm','rain_6_12h_mm','rain_12_24h_mm'])
            if matrix.empty:
                axes[i,j].text(.5,.5,'Không đủ quan trắc huấn luyện',ha='center')
                continue
            image=axes[i,j].imshow(matrix.to_numpy(),vmin=-1,vmax=1,cmap='RdBu_r',aspect='auto')
            axes[i,j].set_yticks(range(len(matrix)),matrix.index,fontsize=7)
            axes[i,j].tick_params(axis='y',labelleft=(j==0))
            axes[i,j].set_xticks(range(len(matrix.columns)),['0–3 giờ','3–6 giờ','6–12 giờ','12–24 giờ'],rotation=25,ha='right',fontsize=7)
            axes[i,j].set_title(f"Tương quan sau điều chỉnh: +{horizon} giờ / {'huấn luyện' if scope=='train' else 'đánh giá'}")
    if image is not None:
        color_axis=fig.add_axes([.92,.20,.015,.60])
        fig.colorbar(image,cax=color_axis,label='Tương quan Pearson sau điều chỉnh')
    fig.subplots_adjust(left=.25,right=.89,hspace=.45,wspace=.1,bottom=.12)
    fig.savefig(figure_dir/'03_partial_associations.png',dpi=150);plt.close(fig)
    example=pairs[pairs.gauge_id.eq(primary[0])&pairs.observation_policy.eq('standard_5of6')&pairs.window.eq('0_3h')&pairs.scope.eq('evaluation')].copy()
    example['absolute_error']=(example.forecast_mm-example.observed_mm).abs()
    fig,ax=plt.subplots(figsize=(8,3.5))
    group=example.groupby(['archive_kind','age_bin']).absolute_error.agg(['mean','count'])
    if len(group):
        group['mean'].plot.bar(ax=ax)
    else:
        ax.text(.5,.5,'Không có cặp quan trắc hợp lệ',ha='center')
    ax.set_xlabel('Loại nguồn và tuổi dự báo (giờ)')
    ax.set_ylabel('Sai số tuyệt đối trung bình lượng mưa (mm)');ax.set_title('Tuổi dự báo và sai số mưa: kết quả mô tả')
    fig.tight_layout();fig.savefig(figure_dir/'04_age_error.png',dpi=150);plt.close(fig)

    integrity=tables['forecast_inventory']
    coverage_table=markdown_table(cov)
    quality_columns=['gauge_id','archive_kind','scope','window','n','mae_mm','rmse_mm','bias_mm',
                     'mse_skill_vs_zero','wet_f1','heavy_recall']
    recommendations=pd.DataFrame([
        {'feature':'Mưa dự báo 0–3, 3–6 và 6–12 giờ','status':'Tiếp tục thử nghiệm',
         'reason':'Đã đo được chất lượng hồi cứu và tương quan; chưa xác nhận chất lượng vận hành lịch sử.',
         'next_step':'Giữ làm ứng viên cho thí nghiệm so sánh nhóm đặc trưng ở Giai đoạn 4.'},
        {'feature':'Mưa dự báo 12–24 giờ','status':'Chưa đủ bằng chứng để giữ hoặc loại',
         'reason':'Có thể phản ánh một hình thế thời tiết kéo dài; tương quan chưa chứng minh giảm sai số dự báo nước.',
         'next_step':'So sánh hai cấu hình E3/E4 trên cùng mốc giờ ở Giai đoạn 4.'},
        {'feature':'Tuổi dự báo và cờ thiếu dữ liệu','status':'Giữ để mô tả chất lượng đầu vào',
         'reason':'Có thể phản ánh chuyển tiếp nguồn dữ liệu và khoảng thời gian mất dữ liệu.',
         'next_step':'Kiểm tra độ ổn định theo thời gian; chưa xem đây là tín hiệu thủy văn.'},
        {'feature':'Nguồn dự báo chung cho thành phố','status':'Một nguồn dùng chung cho mọi trạm nước',
         'reason':'Hai yêu cầu Hồng Thái/Nhâm trả cùng điểm lưới và cùng giá trị tại các mốc chung.',
         'next_step':'Mỗi mốc giờ chỉ lưu một bộ dự báo chung khi chuẩn bị dữ liệu vùng.'},
    ])
    recommendations.to_csv(table_dir/'weather_feature_assessment.csv',index=False)
    strict=cov[cov.policy.eq('strict_operational')].iloc[0]
    assumed=cov[cov.policy.eq('legacy_assumed_release')].iloc[0]
    unmatched=skill[skill.policy.eq('strict_operational')].n.sum()
    evaluation_skill=primary_skill[primary_skill.scope.eq('evaluation')]
    summary_rows=[]
    for window,part in evaluation_skill.groupby('window'):
        summary_rows.append({'window':window,'gauges':';'.join(sorted(part.gauge_id.unique())),
                             'rmse_min_mm':part.rmse_mm.min(),'rmse_max_mm':part.rmse_mm.max(),
                             'bias_min_mm':part.bias_mm.min(),'bias_max_mm':part.bias_mm.max()})
    rain_summary=pd.DataFrame(summary_rows)
    if len(rain_summary):
        order={window:i for i,window in enumerate(['0_3h','3_6h','6_12h','12_24h'])}
        rain_summary=rain_summary.sort_values('window',key=lambda s:s.map(order))
    operational=tables['forecast_run_integrity']
    operational=operational[operational.strict_eligible_run]
    first_available=operational.available_at.min()
    display_eval=dv_partial[dv_partial.scope.eq('evaluation')][['horizon_h','feature','n','partial_pearson_r']]
    text=f'''# Giai đoạn 3 — Đánh giá dự báo mưa và mối liên hệ với mực nước

## 1. Kết quả chính và phạm vi nghiên cứu

Phân tích này sử dụng lại bản dữ liệu cố định và các quy ước xử lý đã chốt ở Giai đoạn 1–2. Dự báo thời tiết được xem là **một bộ thông tin chung của thành phố Huế**, áp dụng cho mọi trạm nước. Hồng Thái và Nhâm trong danh mục nguồn là hai yêu cầu tải dữ liệu; chúng không được tính thành hai nguồn dự báo độc lập.

Trong {int(strict.anchor_count)} mốc giờ được khảo sát, **không có mốc nào đủ bằng chứng cho thấy bản dự báo đã thực sự sẵn có vào thời điểm cần dự báo mực nước**. Nếu giả định dự báo có thể được sử dụng sau khi mô hình khí tượng khởi tạo 6 giờ, có **{int(assumed.selected_n)} mốc** sử dụng được. Vì vậy, những kết quả về độ chính xác dưới đây là đánh giá hồi cứu có điều kiện, chưa xác nhận độ chính xác của hệ thống khi vận hành trong quá khứ.

{vietnamese_table(cov)}

Các bản dự báo có ghi nhận thời điểm tải thực tế chỉ bắt đầu sẵn có từ **{first_available}**, sau khoảng quan trắc đang phân tích. Do đó, chưa thể tính độ chính xác lịch sử cho nhóm có bằng chứng khả dụng thực tế.

Khi đối chiếu với lượng mưa đo riêng tại Hồng Thái và Nhâm trong giai đoạn đánh giá, kết quả như sau:

{vietnamese_table(rain_summary)}

Đơn vị là mm của tổng mưa trong từng khoảng. Độ lệch trung bình âm cho biết lượng mưa dự báo thấp hơn lượng đo trên các cặp dữ liệu hợp lệ. Phép đối chiếu này chưa tách được sai số mô hình khí tượng khỏi khác biệt giữa vị trí trạm đo và điểm lưới dự báo. Không so trực tiếp RMSE của tổng mưa 3 giờ với RMSE của tổng mưa 12 giờ như thể chúng là cùng một bài toán.

Tại Dã Viên, sau khi xét thêm lịch sử mực nước và mưa quan trắc, tương quan còn lại giữa dự báo mưa và biến thiên mực nước trong giai đoạn đánh giá là:

{vietnamese_table(display_eval)}

Những con số này chưa đủ để kết luận thêm hay bỏ một đặc trưng sẽ giúp mô hình dự báo nước chính xác hơn. Đặc biệt, tương quan thấp chưa phải lý do để loại lượng mưa dự báo 12–24 giờ. Cần thí nghiệm so sánh các nhóm đặc trưng trên cùng dữ liệu ở Giai đoạn 4 để trả lời câu hỏi đó.

Phân tích bao gồm {raw.water_station_id.nunique()} trạm nước tại hai mốc dự báo trước 6 giờ và 12 giờ. Với trạm không đủ dữ liệu huấn luyện, chỉ tính các thống kê có đủ số cặp quan trắc; không ước lượng phép điều chỉnh thống kê từ dữ liệu đánh giá. Mô hình B được giữ nguyên và không huấn luyện mô hình dự báo mới trong giai đoạn này.

## 2. Kiểm tra thời điểm khởi tạo, hiệu lực và sẵn có của dự báo

Thời điểm khởi tạo mô hình khí tượng, lưu trong trường `initialized_at`, khác với thời điểm bản dự báo được công bố và có thể tải về. Để coi một bản dự báo là thực sự sẵn có tại thời điểm dự báo mực nước, phân tích yêu cầu bản ghi thuộc một lần chạy vận hành và có bằng chứng tải dữ liệu thực tế. Thứ tự thời gian phải là: **khởi tạo mô hình → tải được dữ liệu → dữ liệu sẵn sàng sử dụng → thời điểm dự báo mực nước**.

Một phần dữ liệu mang nhãn dự báo tái dựng, tức dự báo được chạy lại cho thời gian đã qua. Phần khác là dữ liệu lưu trữ nhưng chưa xác minh được thời điểm công bố. Hai nhóm này được phân tích hồi cứu riêng, không coi là bằng chứng dữ liệu đã có trong quá khứ chỉ vì cộng thêm 6 giờ vào thời điểm khởi tạo.

{vietnamese_table(integrity)}

Số bản ghi lần chạy trong bảng có thể gồm hai yêu cầu tải cho cùng một chu kỳ, nên không phải số dự báo độc lập. Các [chu kỳ cập nhật](tables/forecast_cycle_inventory.csv), [khoảng thiếu](tables/forecast_cycle_gaps.csv) và [kiểm tra từng lần chạy](tables/forecast_run_integrity.csv) được lưu để tra cứu. Kết quả độ chính xác được tách theo loại dữ liệu lưu trữ. Khác biệt giữa các phiên bản chưa chứng minh tác động riêng của phiên bản, vì thời tiết và thời gian quan trắc cũng thay đổi.

Lượng mưa theo giờ của nguồn dự báo là tổng mưa trong giờ kết thúc tại thời điểm ghi trong bản tin. Tại thời điểm dự báo nước t, bốn khoảng 0–3, 3–6, 6–12 và 12–24 giờ lần lượt cộng các bản ghi tại t+1 đến t+3, t+4 đến t+6, t+7 đến t+12 và t+13 đến t+24. Các khoảng không chồng lấn.

Cả bốn khoảng tại một mốc giờ đều lấy từ **cùng một lần chạy mô hình khí tượng**. Phân tích không ghép các lần chạy khác nhau để bù dữ liệu thiếu. Sau khi kiểm tra thời điểm sẵn có của từng yêu cầu tải, hệ thống ưu tiên bản đủ 24 giờ dự báo, rồi ưu tiên thời điểm khởi tạo và thời điểm tải mới nhất. Nếu vẫn bằng nhau, mã yêu cầu tải được dùng để chọn theo một thứ tự cố định.

Tuổi dự báo là số giờ từ lúc mô hình khí tượng khởi tạo đến thời điểm dự báo nước, không phải thời gian kể từ lúc tải dữ liệu. Phân tích chỉ chọn bản có tuổi không quá {config['max_forecast_age_hours']} giờ, nên không có kết luận về chất lượng của các bản cũ hơn giới hạn này.

![Tỷ lệ mốc giờ có dự báo chung theo tháng](figures/01_city_availability.png)

## 3. Một nguồn dự báo chung được đối chiếu với các trạm mưa

{vietnamese_table(tables['forecast_location_grids'])}

Việc ứng dụng dùng dự báo cho toàn thành phố không làm giá trị tại điểm lưới trở thành lượng mưa trung bình đo trên toàn thành phố. Dự báo là giá trị của mô hình tại một điểm lưới; quan trắc là lượng mưa đo tại một trạm cụ thể. Khác biệt vị trí và địa hình phải được giữ trong giới hạn diễn giải.

Đối chiếu hai yêu cầu tải cho thấy:

{vietnamese_table(tables['weather_duplicate_grid_comparison'])}

Hai bản ghi trùng nhau tại các mốc chung chỉ được tính là một bộ dự báo. Phân tích không lấy trung bình hoặc cộng dồn số mẫu của hai yêu cầu tải. [Bảng nguồn dự báo chung theo giờ](tables/weather_city_selection.csv) ghi mã yêu cầu tải đã được chọn (`request_location_id`) và mã lần chạy mô hình (`selected_run_id`), giúp tra lại nguồn của từng giá trị.

Các trạm đối chiếu chính được khai báo trước trong cấu hình: {', '.join(primary)}. Những trạm gần điểm lưới hơn được bổ sung theo khoảng cách địa lý, không chọn dựa trên kết quả độ chính xác. Mỗi trạm có kết quả riêng; không tự coi trung bình các trạm là lượng mưa thực tế đại diện cho cả ô lưới hay lưu vực.

{vietnamese_table(tables['weather_comparators'].drop(columns='policy').drop_duplicates())}

## 4. Độ chính xác dự báo mưa trong từng khoảng thời gian

Lượng mưa thực tế được cộng trong đúng khoảng thời gian tương ứng với dự báo. Nếu thiếu bất kỳ giờ nào, cặp dữ liệu đó không tham gia tính sai số. Hai cách xử lý quan trắc được giữ riêng: cách chuẩn cho phép ước tính giờ có 5/6 bản ghi và cách đối chứng chỉ dùng giờ đủ 6/6 bản ghi. Không nội suy giờ mất hoàn toàn hoặc thay dữ liệu thiếu bằng 0. Vì số cặp hợp lệ có thể khác nhau, chênh lệch chỉ số giữa hai cách không chỉ phản ánh tác động của phép ước tính.

Mốc chia thời gian là **{config['future_analysis']['grouping_end']}**. Đối với đánh giá mưa, tập huấn luyện chỉ nhận các mốc có toàn bộ 24 giờ quan trắc tương lai nằm trước mốc chia. Các mốc sau đó thuộc giai đoạn đánh giá thăm dò, đã được xem nhiều lần và không còn là tập kiểm thử độc lập cuối cùng. Mưa thực tế trong tương lai chỉ dùng để kiểm chứng dự báo, không được đưa vào đầu vào dự báo mực nước.

Ngưỡng nhận diện có mưa là {config['weather_analysis']['wet_threshold_mm']} mm mỗi khoảng. Mưa mạnh được quy ước trước là cường độ trung bình khoảng ít nhất {config['weather_analysis']['heavy_mean_rate_mm_h']} mm/giờ; ngưỡng tổng mưa tương ứng bằng cường độ này nhân số giờ của khoảng. Đây là ngưỡng nghiên cứu, không phải cấp cảnh báo chính thức.

F1 tóm tắt khả năng nhận diện có mưa. Tỷ lệ phát hiện mưa mạnh cho biết bao nhiêu trường hợp mưa mạnh thực tế được dự báo đúng. Nếu không có trường hợp để tính tỷ lệ, bảng để trống; không coi đó là kết quả hoàn hảo.

{vietnamese_table(primary_skill[quality_columns])}

MAE là sai số tuyệt đối trung bình; RMSE là căn bậc hai của trung bình bình phương sai số. Cả hai có đơn vị mm. Độ lệch trung bình được tính bằng lượng dự báo trừ lượng quan trắc. Chỉ số cải thiện MSE so sánh với cách luôn dự báo không mưa trên đúng các cặp dữ liệu đó: số âm nghĩa dự báo đang xét có sai số bình phương lớn hơn cách đối chứng này. Đây chỉ là đối chứng đơn giản, chưa phải mô hình dự báo khí tượng mạnh và không phản ánh trực tiếp sai số dự báo mực nước.

![Sai số lượng mưa theo khoảng dự báo và loại nguồn](figures/02_rainfall_skill.png)

[Bảng độ chính xác đầy đủ](tables/weather_skill.csv) tách kết quả theo mùa, tuổi dự báo, nguồn dữ liệu, trạm mưa và cách xử lý quan trắc thiếu. Do các khoảng mưa có thể chồng nhau giữa những mốc giờ liên tiếp, số cặp kiểm chứng không đồng nghĩa với số sự kiện độc lập. Báo cáo không tính khoảng tin cậy hay mức ý nghĩa thống kê dựa trên giả định các cặp này độc lập.

![Tuổi bản dự báo và sai số lượng mưa](figures/04_age_error.png)

Quan hệ giữa tuổi dự báo và sai số mới là mô tả. Tuổi dự báo, giờ trong ngày, chu kỳ mô hình, loại dữ liệu lưu trữ và chế độ mưa có thể thay đổi cùng nhau. Vì vậy chưa thể khẳng định tuổi dự báo là nguyên nhân riêng gây thay đổi chất lượng. Tương tự, cờ thiếu dữ liệu có thể phản ánh khoảng mất dữ liệu hoặc thay đổi nguồn, chưa tự động là tín hiệu thủy văn.

## 5. Quan hệ giữa dự báo mưa và biến thiên mực nước

Bảng dưới trình bày Dã Viên tại hai mốc +6 giờ và +12 giờ. Tương quan Pearson mô tả quan hệ tuyến tính; tương quan Spearman mô tả mức độ đồng biến theo thứ hạng. Kết quả được tính riêng trên tập huấn luyện và giai đoạn đánh giá. Tập huấn luyện đã loại 12 giờ cuối trước mốc chia để nhãn mực nước tương lai không kéo sang giai đoạn đánh giá.

{vietnamese_table(dv_raw[['horizon_h','scope','feature','n','pearson_r','spearman_r']])}

Kết quả cho mọi trạm nằm trong [bảng tương quan đầy đủ](tables/weather_water_correlations.csv). Mỗi đặc trưng chỉ dùng những cặp hợp lệ của chính nó, nên số mẫu có thể khác nhau. Không xếp hạng hiệu quả dự báo chỉ từ các hệ số tương quan.

Phân tích còn xét riêng tháng mùa lũ, thời điểm nước đang dâng theo quan trắc quá khứ và nhóm có biến thiên mực nước tương lai từ 200 mm trở lên về độ lớn. Nhóm cuối chỉ phục vụ phân tích sau sự kiện; không thể dùng để chọn đặc trưng khi đang dự báo. Loại trạm được giữ theo danh mục hiện có, không tự suy ra trạm chịu triều hay thuộc sông nào khi chưa có bằng chứng xác minh.

## 6. Tín hiệu còn lại sau khi xét lịch sử mực nước và mưa

Để kiểm tra liên hệ có điều kiện, phân tích trước hết ước lượng phần biến thiên của mực nước và dự báo mưa có thể được giải thích bằng các yếu tố đã biết: mực nước hiện tại, thay đổi mực nước trong 3 giờ và 6 giờ trước đó, cùng tổng mưa 6 giờ và 12 giờ tại hai trạm mưa được tuyển chọn trên tập huấn luyện ở Giai đoạn 2. Sau đó tính tương quan giữa hai phần còn lại. Đây là tương quan riêng phần theo một phép điều chỉnh tuyến tính.

Các hệ số điều chỉnh và chuẩn hóa chỉ được ước lượng trên tập huấn luyện rồi giữ cố định khi áp dụng vào giai đoạn đánh giá. Bảng dưới mô tả chung dữ liệu dùng giả định sẵn có sau 6 giờ. Bảng đầy đủ còn tách riêng dự báo tái dựng và dữ liệu lưu trữ, mỗi nhóm chỉ ước lượng bằng phần huấn luyện của chính nhóm đó. Dù vậy, sự thay đổi nguồn cùng thời gian vẫn có thể ảnh hưởng kết quả; chưa thể xem các hệ số này là độc lập với phiên bản khí tượng.

{vietnamese_table(dv_partial[['horizon_h','scope','feature','train_fit_n','n','partial_pearson_r','partial_spearman_r']])}

![Tương quan sau điều chỉnh tại các trạm đủ dữ liệu huấn luyện](figures/03_partial_associations.png)

Tương quan còn lại khác 0 cho biết hai đại lượng vẫn có liên hệ sau những yếu tố tuyến tính đã xét. Nó chưa chứng minh quan hệ nhân quả, thông tin hoàn toàn mới hay khả năng giảm sai số của mô hình dự báo. Với mưa dự báo 12–24 giờ, tín hiệu có thể cùng phản ánh một hình thế thời tiết kéo dài với biến thiên mực nước +6/+12 giờ. Không dùng mưa thực tế 12–24 giờ làm đầu vào để giải thích tín hiệu đó.

[Bảng các yếu tố đã xét](tables/weather_controls.csv) ghi rõ mã trạm mưa cho từng trạm nước; các trạm này không được chọn lại bằng kết quả đánh giá. [Bảng dữ liệu thiếu và biến động mực nước](tables/weather_missingness_water.csv) so sánh những mốc đủ hoặc thiếu dự báo với biến động thực tế. Bảng này chưa đo sai số của mô hình B. Muốn biết dữ liệu thiếu ảnh hưởng thế nào đến lỗi dự báo nước, cần thí nghiệm chạy mô hình đối chứng được thiết kế riêng.

## 7. Đề xuất trước Giai đoạn 4

{vietnamese_table(recommendations)}

Dữ liệu hiện tại cho phép nghiên cứu hồi cứu với nguồn gốc được ghi rõ, nhưng chưa đủ để xác nhận chất lượng dự báo đã thực sự sẵn có khi vận hành trong quá khứ. Cần tiếp tục lưu các bản dự báo vận hành kèm thời điểm tải thực tế, rồi đánh giá trên một giai đoạn mới chưa được dùng để lựa chọn mô hình.

Sau khi báo cáo này được xem xét, Giai đoạn 4 có thể so sánh các nhóm đặc trưng trên cùng mốc dự báo, cùng cách chia thời gian và chỉ ước lượng các tham số xử lý bằng tập huấn luyện. Những kết quả phụ thuộc giả định thời điểm sẵn có của dự báo cần được tách riêng.

Giai đoạn này chưa triển khai thí nghiệm so sánh đó, chưa loại lượng mưa dự báo 12–24 giờ, chưa thay mô hình B và chưa thay đổi tập dữ liệu huấn luyện.

## 8. Nguồn dữ liệu và cách kiểm tra lại

Nguồn và cấu hình của lần phân tích được ghi trong [hồ sơ thực thi](run_manifest.json). Các [bảng CSV](tables/) giữ nguyên tên trường kỹ thuật để đối chiếu bằng code; các bảng trong báo cáo đã được chuyển sang tiếng Việt. Dữ liệu đã xử lý được lưu tại `data/processed/station_analysis/phase2/weather/`. Dữ liệu dự báo gốc và quan trắc trong cơ sở dữ liệu được giữ nguyên.

Quy ước mưa tích lũy theo khoảng 10 phút được giữ như Giai đoạn 2; lần phân tích này không bổ sung bằng chứng thiết bị để xác minh độc lập quy ước đó.

[Tài liệu dự báo của Open-Meteo](https://open-meteo.com/en/docs) mô tả lượng mưa theo giờ kết thúc tại thời điểm ghi trong bản tin. [Tài liệu dữ liệu dự báo lịch sử](https://open-meteo.com/en/docs/historical-forecast-api) mô tả chuỗi ghép từ phần đầu của các lần chạy mô hình. Mô tả này không thay thế bằng chứng về thời điểm sẵn có của các bản ghi đang nghiên cứu. [Tài liệu các lần dự báo trước](https://open-meteo.com/en/docs/previous-runs-api) mô tả một sản phẩm theo độ dài dự báo trước, khác với chuỗi dự báo lịch sử đã ghép. Các trang được đối chiếu ngày 26/10/2026; việc phân loại nguồn vẫn dựa trên thông tin trong bản dữ liệu cố định của dự án.
'''
    path=output/'report.md';path.write_text(text,encoding='utf-8')
    return path
