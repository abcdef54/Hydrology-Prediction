"""Readable Stage 1 report and coverage figures; no correlations or skill scores."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from research.station_analysis.config import write_json
from research.station_analysis.reporting import _number, _local_time, _display, markdown


def _figures(tables, stations, output, work):
    os.environ.setdefault('MPLCONFIGDIR',str(work/'matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    monthly=tables['observation_monthly_coverage']
    names=stations.set_index('station_id').name.to_dict()
    for number,kind,label in [(1,'water_level','mực nước'),(2,'rainfall','mưa')]:
        part=monthly.loc[monthly.station_type.eq(kind)]
        grid=part.pivot(index='station_id',columns='month',values='missing_rate')
        if grid.empty:
            continue
        fig,ax=plt.subplots(figsize=(13,max(6,len(grid)*.26)))
        im=ax.imshow(1-grid.to_numpy(),aspect='auto',cmap='YlGnBu',vmin=0,vmax=1)
        ax.set_yticks(range(len(grid)),[f'{sid}: {names.get(sid,sid)}' for sid in grid.index],fontsize=8)
        ax.set_xticks(range(len(grid.columns)),grid.columns,rotation=45,ha='right')
        ax.set_title(f'Độ phủ quan trắc {label} theo tháng — có số đo hợp lệ không đồng nghĩa đủ cả giờ')
        fig.colorbar(im,ax=ax,label='Tỷ lệ giờ có ít nhất một số đo hợp lệ')
        fig.tight_layout()
        fig.savefig(output/'figures'/f'0{number}_{kind}_coverage.png',dpi=140)
        plt.close(fig)
    coverage=tables['forecast_coverage']
    if not coverage.empty:
        location_names={row.location_id:names.get(row.station_id,row.station_id)
                        for row in tables['forecast_locations'].itertuples()}
        rows=[]
        for row in coverage.itertuples():
            policy='Có bằng chứng khả dụng' if row.policy=='strict_operational' else 'Giả định phát hành sau 6h'
            rows.append({'label':f'{location_names.get(row.location_id,row.location_id)}\n{policy}', '6h':row.complete_6h_fraction,
                         '12h':row.complete_12h_fraction,'24h':row.complete_24h_fraction})
        frame=pd.DataFrame(rows).set_index('label')
        fig,ax=plt.subplots(figsize=(12,5))
        frame.plot.barh(ax=ax)
        ax.set_xlim(0,1)
        ax.set_ylabel('Địa điểm yêu cầu và điều kiện khả dụng')
        ax.set_xlabel('Tỷ lệ anchor có đủ số giờ forecast; chưa đánh giá độ chính xác')
        ax.set_title('Độ phủ forecast phụ thuộc vào bằng chứng khả dụng theo thời gian')
        fig.tight_layout()
        fig.savefig(output/'figures/03_forecast_availability.png',dpi=140)
        plt.close(fig)


def create_inventory_report(tables,frames,anchors,config,work,output,snapshot_folder):
    (output/'tables').mkdir(exist_ok=True)
    (output/'figures').mkdir(exist_ok=True)
    for name,frame in tables.items():
        if name in {'observation_availability','forecast_anchor_availability','dataset_forecast_rows'}:
            frame.to_csv(output/'tables'/f'{name}.csv.gz',index=False,compression='gzip')
        else:
            frame.to_csv(output/'tables'/f'{name}.csv',index=False)
    obs=tables['observation_availability']
    for value,name in [('has_valid_value','data_availability_matrix'),
                       ('complete_expected_grid','complete_observation_grid_matrix')]:
        obs.pivot(index='timestamp_utc',columns='station_id',values=value).astype('int8').to_csv(
            output/'tables'/f'{name}.csv.gz',compression='gzip')
    forecast=tables['forecast_anchor_availability']
    strict=forecast.loc[forecast.policy.eq('strict_operational')]
    strict.pivot_table(index='anchor_utc',columns=['location_id','model'],values='complete_12h',aggfunc='max').to_csv(
        output/'tables/strict_forecast_availability_matrix.csv.gz',compression='gzip')
    write_json(output/'configuration.json',config)
    stations=frames['stations']
    _figures(tables,stations,output,work)
    inventory=tables['observation_inventory']
    rain=inventory.loc[inventory.station_type.eq('rainfall')].copy()
    water=inventory.loc[inventory.station_type.eq('water_level')].copy()
    for part in [water,rain]:
        part['missing_percent']=part.missing_rate*100
        part['active_missing_percent']=part.active_span_missing_rate*100
        part['complete_percent']=part.complete_grid_hours/part.requested_hours*100
    source_rows=[]
    for label,part in [('Mực nước',water),('Mưa quan trắc',rain)]:
        source_rows.append({'Nguồn':label,'Số trạm':len(part),'Bản ghi gốc':int(part.raw_records.sum()),
                            'Thời gian bắt đầu (UTC+7)':_local_time(part.first_observed_utc.min()),
                            'Thời gian cuối (UTC+7)':_local_time(part.last_observed_utc.max()),
                            'Trạm chưa có bản ghi':int(part.raw_records.eq(0).sum())})
    source_table=markdown(pd.DataFrame(source_rows))
    coverage_columns={'name':'Trạm','raw_records':'Bản ghi gốc','hours_with_valid_values':'Giờ có số đo',
                      'missing_percent':'Thiếu toàn kỳ (%)','active_missing_percent':'Thiếu trong thời gian hoạt động (%)',
                      'complete_percent':'Giờ đủ lưới lấy mẫu (%)','longest_missing_hours':'Khoảng thiếu dài nhất (h)'}
    water_table=_display(water,coverage_columns)
    # Show every rain station; no ranking or station selection is performed.
    rain_table=_display(rain,coverage_columns)
    semantics=tables['rainfall_semantics_audit']
    confirmed=int(semantics.semantics_status.eq('confirmed_by_supplied_contract').sum())
    run_inventory=tables['forecast_inventory'].copy()
    run_inventory['Địa điểm yêu cầu']=run_inventory.location_id
    forecast_table=_display(run_inventory,{
        'Địa điểm yêu cầu':'Địa điểm yêu cầu','archive_kind':'Loại bản dự báo được lưu',
        'availability_basis':'Căn cứ thời điểm khả dụng','run_count':'Số bản dự báo',
        'missing_available_at_runs':'Bản chưa biết thời điểm khả dụng'})
    cov=tables['forecast_coverage'].copy()
    cov['policy_vi']=cov.policy.map({'legacy_assumed_release':'Giả định cũ: khởi tạo + 6h',
                                   'strict_operational':'Có bằng chứng đã nhận trước anchor'})
    forecast_coverage_table=_display(cov,{'location_id':'Địa điểm yêu cầu','policy_vi':'Điều kiện khả dụng',
                                        'anchor_count':'Số mốc được kiểm tra',
                                        'complete_6h_anchors':'Mốc đủ forecast +6h',
                                        'complete_12h_anchors':'Mốc đủ forecast +12h',
                                        'complete_24h_anchors':'Mốc đủ forecast +24h'})
    provenance=tables['dataset_forecast_integrity']
    all_provenance=provenance.loc[provenance.split.eq('all_hourly')].copy()
    all_provenance['dataset_short']=all_provenance.dataset.str.replace('da_vien_1h_2026-10-15_','',regex=False)
    grid=tables['forecast_location_grids']
    grid=grid.copy()
    location_names=frames['weather_locations'].set_index('location_id').station_id.map(
        stations.set_index('station_id').name)
    grid['request_name']=grid.location_id.map(location_names)
    for column in ['requested_latitude','requested_longitude','grid_latitude','grid_longitude']:
        grid[column+'_display']=grid[column].map(lambda value:f'{value:.6f}'.replace('.',','))
    grid_table=_display(grid,{'request_name':'Trạm dùng tọa độ yêu cầu',
                             'requested_latitude_display':'Vĩ độ yêu cầu',
                             'requested_longitude_display':'Kinh độ yêu cầu',
                             'grid_latitude_display':'Vĩ độ ô lưới trả về',
                             'grid_longitude_display':'Kinh độ ô lưới trả về',
                             'locations_sharing_returned_grid':'Số yêu cầu dùng chung ô'})
    grid_summary=(f"Trong dữ liệu hiện tại, {len(frames['weather_locations'])} địa điểm yêu cầu "
                  f"được ánh xạ tới {len(grid[['grid_latitude','grid_longitude']].drop_duplicates())} "
                  'cặp tọa độ ô lưới trả về. Bảng dưới ghi riêng hai loại tọa độ để tránh coi '
                  'các yêu cầu khác tên là các nguồn dự báo không gian độc lập.')
    operational=frames['forecast_runs'].loc[frames['forecast_runs'].availability_basis.eq('observed_retrieval')]
    operational_note=(
        f"Có {len(operational)} bản có bằng chứng tải thực tế, nhưng lần tải sớm nhất là "
        f"{_local_time(operational.retrieved_at.min())}. Thời điểm này sau mốc quan trắc cuối "
        'của khoảng phân tích, nên các bản đó chưa cung cấp cặp forecast–quan trắc khả dụng '
        'để replay vận hành trên giai đoạn đang xét.'
        if len(operational) and pd.to_datetime(operational.retrieved_at,utc=True).min()>anchors.max()
        else 'Các bản có bằng chứng tải thực tế được đối chiếu theo từng anchor; sự tồn tại của một bản trong database không tự động làm nó hợp lệ cho mọi thời điểm lịch sử.')
    targets=tables['dataset_target_alignment']
    target_mismatches=int(targets.value_mismatches.sum()) if len(targets) else 0
    alignment=tables['hourly_alignment_audit']
    forward_count=int(alignment.phase1_left_label_readings_after_label.sum())
    raw_issues={key:int(inventory[key].sum()) for key in [
        'duplicate_timestamp_extra_rows','invalid_timestamps','off_expected_grid_records',
        'invalid_or_missing_values','receipt_after_anchor_records','updates_after_anchor_records']}
    verified_anchors=int(strict.complete_12h.sum())
    forecast_quality=tables['forecast_run_integrity']
    point_missing=int(forecast_quality.missing_value_rows.sum())
    point_duplicates=int(forecast_quality.duplicated_hour_rows.sum())
    point_negative=int(forecast_quality.negative_value_rows.sum())
    cycle_gap_count=len(tables['forecast_cycle_gaps'])
    findings=[
        {'Mã':'I01','Vấn đề':'Ý nghĩa tích lũy mưa chưa được xác minh',
         'Bằng chứng':f'{confirmed}/{len(rain)} trạm có hợp đồng phép đo được xác nhận',
         'Ảnh hưởng':'Chưa được coi tổng các giá trị nguồn là lượng mưa giờ; chặn phần định lượng mưa của Stage 2–4 cho trạm chưa xác minh'},
        {'Mã':'I02','Vấn đề':'Forecast lịch sử chưa có bằng chứng khả dụng vận hành',
         'Bằng chứng':f'{verified_anchors} cặp địa điểm–anchor có forecast +12h thỏa điều kiện nghiêm ngặt',
         'Ảnh hưởng':'Stage 3 và weather ablation ở Stage 4 phải tách nhánh giả định/hindcast; chưa được dùng làm bằng chứng vận hành'},
        {'Mã':'I03','Vấn đề':'Quy ước nhãn giờ của Phase 1 khác builder train/serving',
         'Bằng chứng':f'{forward_count} bản ghi hợp lệ nằm sau nhãn giờ nếu dùng cách gắn nhãn Phase 1 trên các nguồn đang kiểm kê',
         'Ảnh hưởng':'Phải tính lại đặc trưng biến động/phân nhóm từ raw theo cùng quy ước trước Stage 2; không đổi nhãn rồi tái sử dụng giá trị cũ'},
        {'Mã':'I04','Vấn đề':'Lịch sử nhận và cập nhật quan trắc không tái tạo được upstream as-of',
         'Bằng chứng':f"{raw_issues['receipt_after_anchor_records']} bản ghi trong khoảng khảo sát được chèn local sau anchor đo",
         'Ảnh hưởng':'Backtest theo thời điểm quan trắc phải công bố giả định độ trễ và sửa số liệu; chưa đủ điều kiện gọi là replay vận hành chính xác'},
        {'Mã':'I05','Vấn đề':'Địa điểm yêu cầu forecast không nhất thiết là ô lưới độc lập',
         'Bằng chứng':f"{len(frames['weather_locations'])} yêu cầu, {len(grid[['grid_latitude','grid_longitude']].drop_duplicates())} cặp tọa độ grid trả về",
         'Ảnh hưởng':'Không nhân đôi tín hiệu forecast hoặc diễn giải là dự báo mưa riêng từng gauge khi cùng ô lưới'},
        {'Mã':'I06','Vấn đề':'Các tập test cũ đã dùng nhiều lần; tập mở rộng dùng test cũ làm validation',
         'Bằng chứng':'Manifest và lịch sử thí nghiệm hiện có ghi rõ đây là đánh giá khám phá',
         'Ảnh hưởng':'Stage 4 cần protocol thời gian đóng băng mới; không gọi tập cũ là independent final test'},
    ]
    findings_table=markdown(pd.DataFrame(findings))
    tables['integrity_findings']=pd.DataFrame(findings)
    tables['integrity_findings'].to_parquet(work/'tables/integrity_findings.parquet',index=False)
    tables['integrity_findings'].to_csv(output/'tables/integrity_findings.csv',index=False)
    source_manifest=json.loads((snapshot_folder/'manifest.json').read_text())
    # Column names are controlled by the provenance module; retain full CSV and select readable metrics.
    provenance_labels={
        'dataset_short':'Dataset','anchor_count':'Số dòng được kiểm tra',
        'hindcast_label_rows':'Dòng hindcast','assumed_availability_rows':'Dòng giả định thời điểm khả dụng',
        'historical_availability_verified_rows':'Dòng có bằng chứng khả dụng',
        'source_retrieved_after_anchor_rows':'Dòng tải sau anchor',
        'mismatched_forecast_value_count':'Số giá trị lệch so với vintage trong DB',
    }
    available_labels={k:v for k,v in provenance_labels.items() if k in all_provenance}
    provenance_table=_display(all_provenance,available_labels)
    report=f'''# Stage 1: Kiểm kê và kiểm tra tính toàn vẹn dữ liệu thủy văn Huế

## 1. Mục tiêu và phạm vi của báo cáo

Stage 1 được thực hiện theo master plan trong `EDA2.md` để xác định các nguồn dữ liệu hiện có có thể được dùng như thế nào trong nghiên cứu Regional ResidualLSTM. Đây là bước kiểm tra điều kiện đầu vào, trước khi phân tích quan hệ mưa–mực nước hoặc đo giá trị dự báo bổ sung của các nhóm đặc trưng.

Báo cáo kiểm kê {len(water)} trạm nước, {len(rain)} trạm mưa, {len(frames['forecast_runs'])} bản dự báo thời tiết và {len(frames['forecast_points'])} giá trị dự báo theo giờ. Khoảng thời gian dùng để so độ phủ giữa các nguồn gồm {_number(len(anchors),0)} mốc, từ **{_local_time(anchors.min())} đến {_local_time(anchors.max())}**. Các số đếm bản ghi gốc bên dưới bao gồm toàn bộ lịch sử đang lưu; chúng được tách khỏi độ phủ trên khoảng so sánh chung. Toàn bộ thời gian giải thích trong báo cáo dùng giờ Việt Nam, còn bảng dữ liệu máy đọc giữ UTC.

**Kết quả quan trọng nhất là các nguồn đang có số liệu nhưng chưa đồng nghĩa với việc chúng đã đủ điều kiện để tạo feature vận hành.** Ý nghĩa phép đo mưa vẫn cần xác minh, forecast lịch sử chủ yếu phụ thuộc giả định thời điểm phát hành, và cách gắn nhãn giờ trong Phase 1 khác quy ước của dataset huấn luyện. Những điểm này cần được giải quyết hoặc tách thành giả định nghiên cứu trước khi tiếp tục các bước định lượng.

Lượt thực hiện này chỉ bao gồm **Stage 1 — Data Inventory & Integrity**. Không có phân tích tương quan mưa–mực nước, đánh giá độ chính xác forecast, thí nghiệm ablation hoặc training model nào được chạy.

## 2. Các nguồn dữ liệu thực tế

{source_table}

Nguồn quan trắc và forecast được trích trong cùng một transaction chỉ đọc, với thời điểm chụp dữ liệu là {_local_time(source_manifest['snapshot_utc'])}. Bản chụp được lưu riêng dưới dạng Parquet và có mã kiểm tra nội dung. Điều này giúp các stage sau dùng đúng cùng trạng thái dữ liệu thay vì vô tình kết hợp những lần tải khác nhau.

Danh mục trạm, tọa độ và đơn vị được lấy trực tiếp từ database. Đơn vị `mm` phản ánh contract đang được dự án sử dụng, nhưng chưa giải thích một bản ghi mưa là lượng tích lũy trong khoảng nào, tốc độ mưa hay giá trị bộ đếm. Metadata về cao độ và ý nghĩa mực nước vẫn giữ các giới hạn đã nêu trong Phase 1; không có ngưỡng báo động hoặc mốc cao độ nào được suy đoán thêm ở Stage 1.

Các GeoTIFF, bảng địa hình quanh trạm và kết quả EDA trước được kiểm kê và đọc lại. Pipeline không xử lý lại DEM hoặc chạy lại clustering. Bảng [station_dem_coverage.csv](tables/station_dem_coverage.csv) cho biết tọa độ của từng trạm có nằm trên phần raster đã xử lý còn số liệu hay không; đây chưa phải thống kê địa hình mới quanh trạm mưa hoặc xác định lưu vực.

Trong lần kiểm kê hiện tại, FABDEM đã xử lý có số liệu tại tọa độ của toàn bộ trạm nước và trạm mưa. Copernicus đã xử lý bao phủ toàn bộ trạm nước nhưng thiếu tại tọa độ của một số trạm mưa. Thống kê địa hình lân cận cho các trạm mưa chưa được trích trong Phase 1 và cần được thực hiện khi Stage 2 đối chiếu không gian giữa nguồn mưa và trạm nước.

## 3. Độ phủ, tần suất lấy mẫu và các khoảng thiếu

Một giờ ở đây được gắn nhãn bằng **thời điểm kết thúc giờ**. Tại anchor `t`, chỉ những quan trắc có thời gian đo thuộc `(t−1h,t]` được đưa vào bản tổng hợp đó. Quy ước này khớp builder dataset và serving hiện tại, đồng thời không đưa số đo có timestamp sau `t` vào đầu vào tại `t`.

Bảng phân biệt ba mức độ đầy đủ: có ít nhất một bản ghi, có ít nhất một số đo hợp lệ và có đủ các lần lấy mẫu trên lưới 10 phút dự kiến. Một giờ chỉ có một số đo không được coi là đã có đủ sáu số đo. “Thiếu toàn kỳ” dùng cùng lịch thời gian cho mọi trạm; “thiếu trong thời gian hoạt động” chỉ tính từ giờ có số đo đầu tiên đến giờ có số đo cuối cùng của chính trạm.

### 3.1. Các trạm mực nước

{water_table}

### 3.2. Các trạm mưa

{rain_table}

Các khoảng thiếu liên tục, bao gồm phần trước khi trạm bắt đầu ghi nhận, phần gián đoạn bên trong và phần cuối thiếu số liệu, được lưu trong [observation_missing_gaps.csv](tables/observation_missing_gaps.csv). Tần suất lấy mẫu, số timestamp trùng và giá trị không hợp lệ nằm trong [observation_inventory.csv](tables/observation_inventory.csv).

![Độ phủ dữ liệu nước theo tháng](figures/01_water_level_coverage.png)

*Hình 1 thể hiện tỷ lệ giờ có ít nhất một số đo mực nước hợp lệ. Màu đậm không có nghĩa là mọi giờ đều đủ số lần lấy mẫu.*

![Độ phủ dữ liệu mưa theo tháng](figures/02_rainfall_coverage.png)

*Hình 2 dùng cùng cách tính cho toàn bộ trạm mưa. Tỷ lệ cao cho biết số liệu xuất hiện đều, nhưng chưa xác nhận cách cộng chúng thành lượng mưa theo giờ.*

## 4. Tính toàn vẹn của quan trắc và ý nghĩa lượng mưa

Trong toàn bộ nguồn quan trắc đã kiểm kê, quy trình ghi nhận {_number(raw_issues['duplicate_timestamp_extra_rows'],0)} bản ghi trùng thêm theo khóa trạm–timestamp, {_number(raw_issues['invalid_timestamps'],0)} timestamp không hợp lệ, {_number(raw_issues['off_expected_grid_records'],0)} bản ghi không nằm đúng lưới 10 phút dự kiến và {_number(raw_issues['invalid_or_missing_values'],0)} giá trị thiếu hoặc không hữu hạn. Các số này có phạm vi toàn bộ bản ghi nguồn; tỷ lệ giờ thiếu trong mục 3 có phạm vi lịch so sánh chung.

Có **{confirmed}/{len(rain)} trạm mưa** được cung cấp bằng chứng xác nhận ý nghĩa tích lũy trong cấu hình nghiên cứu. Vì vậy, ở những trạm chưa được xác minh, tổng các giá trị gốc trong giờ chỉ được lưu dưới tên “tổng chẩn đoán theo cách cũ”, chưa được công nhận là lượng mưa giờ đúng về vật lý. Dữ liệu thiếu cũng không được thay bằng 0 mm.

Sự phân biệt này quan trọng vì ba cách đo đòi hỏi ba phép tổng hợp khác nhau. Nếu từng bản ghi là lượng mưa của một khoảng 10 phút không chồng lấn, có thể cộng các lượng đó khi đủ toàn bộ khoảng đo. Nếu là tốc độ mưa theo mm/giờ, cần tích phân theo thời gian. Nếu là bộ đếm tích lũy, cần lấy chênh lệch giữa các lần đọc và xử lý những lần bộ đếm reset. Việc nhìn thấy chuỗi không giảm hoặc nhiều giá trị lặp lại chưa đủ để xác định thiết bị thuộc loại nào.

Stage 1 đã bổ sung và kiểm thử các hàm xử lý ba trường hợp trên, nhưng không tự lựa chọn một trường hợp cho các trạm hiện tại. Những khoảng reset hoặc thiếu mốc để lấy chênh lệch được giữ là thiếu, thay vì suy ra lượng mưa không có bằng chứng. Các cửa sổ tổng mưa tương lai của Stage 2 chỉ được tạo khi có contract được xác nhận và đủ các giờ quá khứ cần thiết.

Các giá trị lớn được đánh dấu để xem lại theo ngưỡng cấu hình, nhưng chưa được gọi là “mưa cực đoan” hoặc quy thành cường độ mm/giờ khi chưa biết khoảng tích lũy. Với mực nước, giá trị âm được giữ nguyên vì có thể phù hợp với số 0 riêng của trạm. Bảng [rainfall_semantics_audit.csv](tables/rainfall_semantics_audit.csv) lưu đầy đủ mức xác minh, các chẩn đoán và những câu hỏi còn mở của từng trạm mưa.

## 5. Quy ước thời gian và giới hạn của lịch sử dữ liệu

### 5.1. Sự khác biệt với Phase 1

Phase 1 gắn trung bình của khoảng `[t,t+1h)` vào nhãn `t`. Cách này có thể dùng để mô tả dữ liệu, nhưng không được tái sử dụng trực tiếp như feature đã biết tại thời điểm phát dự báo `t`. Trong nguồn hiện tại, có {_number(forward_count,0)} bản ghi hợp lệ sẽ nằm sau nhãn giờ nếu áp dụng cách gắn nhãn đó trên khoảng đang khảo sát.

Stage 1 giữ nguyên các kết quả Phase 1 và đồng thời tạo bảng kiểm kê theo `(t−1h,t]`. Trước khi dùng lại các đặc trưng biến động hoặc kết luận phân nhóm của Phase 1 để thiết kế dataset, cần tính lại từ quan trắc gốc với quy ước giờ đã thống nhất. Chỉ đổi tên timestamp mà giữ nguyên giá trị sẽ không giải quyết được sự khác biệt về các bản ghi được đưa vào từng giờ.

Phát hiện về nhãn giờ này thuộc phần EDA của Phase 1. Builder huấn luyện và serving đang dùng quy ước cuối giờ, nên chưa có căn cứ từ phát hiện này để kết luận rằng Model B đã được train bằng cùng cách gắn nhãn của Phase 1.

### 5.2. Thời gian đo khác với thời gian thực sự nhận được số liệu

Các trường `received_at` và `updated_at` lần lượt cho biết thời gian bản ghi được đưa vào database local lần đầu và thời gian cập nhật gần nhất. Có {_number(raw_issues['receipt_after_anchor_records'],0)} bản ghi hợp lệ trong khoảng khảo sát được chèn local sau anchor tương ứng; {_number(raw_issues['updates_after_anchor_records'],0)} bản ghi có lần cập nhật local sau anchor.

Đối với dữ liệu được backfill, các thời gian này không chứng minh rằng upstream chỉ công bố số liệu vào lúc import local. Ngược lại, chúng cũng không đủ để tái tạo chính xác giá trị mà hệ thống đã nhìn thấy tại một thời điểm trong quá khứ. Vì vậy, backtest dựa trên thời gian quan trắc phải ghi rõ giả định rằng giá trị lịch sử sau hiệu chỉnh được dùng để mô phỏng đầu vào. Muốn thực hiện replay vận hành chặt chẽ hơn thì cần archive theo thời gian nhận và phiên bản sửa số liệu.

## 6. Forecast hiện có khả dụng theo nghĩa nào?

### 6.1. Thời điểm khởi tạo, thời điểm có hiệu lực và thời điểm khả dụng

Mỗi bản forecast có thời điểm mô hình thời tiết được khởi tạo, các mốc dự báo có hiệu lực và thời điểm dự án tải bản đó. Thời điểm khởi tạo không đồng nghĩa với thời điểm forecast đã được phát hành để sử dụng. Tài liệu Single Runs của Open-Meteo mô tả dữ liệu cũ của ECMWF có phần là hindcast, tức là được tạo lại cho thời điểm lịch sử; dữ liệu đó cần được phân biệt với forecast đã lưu nguyên bản khi vận hành.

{forecast_table}

Các giá trị forecast đang lưu có đơn vị mm được xác nhận qua metadata của response. Trong dữ liệu đã kiểm kê có {_number(point_missing,0)} giá trị giờ bị thiếu, {_number(point_duplicates,0)} dòng giờ bị trùng trong cùng run và {_number(point_negative,0)} giá trị mưa âm. Kiểm tra chu kỳ khởi tạo dự kiến mỗi 6 giờ còn ghi nhận {cycle_gap_count} khoảng gián đoạn theo từng địa điểm yêu cầu; các mốc cụ thể nằm trong [forecast_cycle_gaps.csv](tables/forecast_cycle_gaps.csv). Các chỉ tiêu này mô tả độ đầy đủ và tính toàn vẹn, chưa đánh giá forecast có dự báo đúng lượng mưa thực tế hay không.

Các nhãn `hindcast_49r1` và `archive_50r1_unverified` được mã thu thập của dự án gán theo mốc chuyển phiên bản mà tài liệu provider mô tả. Chúng không phải là bằng chứng độc lập, được provider ký xác nhận, cho từng bản forecast. Trong báo cáo này, nhãn và căn cứ thời điểm khả dụng luôn được giữ riêng để tránh biến một giả định thành dữ kiện đã xác minh.

### 6.2. Độ phủ theo hai điều kiện khả dụng

Nhánh chẩn đoán tái hiện giả định cũ: forecast được xét khi thời điểm khởi tạo cộng 6 giờ không vượt anchor, tuổi bản không quá 24 giờ và các điều kiện dữ liệu đáp ứng cửa sổ cần dùng. Nhánh nghiêm ngặt chỉ chấp nhận bản vận hành có bằng chứng đã được tải trước anchor, với thứ tự thời gian hợp lệ; bản hindcast hoặc archive chưa rõ thời điểm phát hành không được chuyển sang nhánh này chỉ nhờ giả định cộng 6 giờ.

{forecast_coverage_table}

{operational_note}

Các con số ở đây chỉ cho biết có đủ dữ liệu dưới từng điều kiện, chưa đo độ chính xác dự báo mưa. Giá trị coverage bằng 0 trong nhánh nghiêm ngặt có nghĩa là archive hiện tại chưa cung cấp bằng chứng cho replay vận hành trên khoảng đang khảo sát; nó không chứng minh rằng trong thực tế đã không tồn tại forecast ở thời điểm đó.

![Độ phủ forecast theo chính sách khả dụng](figures/03_forecast_availability.png)

*Hình 3 đối chiếu cùng tập anchor dưới giả định cũ và điều kiện có bằng chứng khả dụng. Sự khác biệt này phải được giữ lại trong mọi đánh giá giá trị của weather features ở các stage sau.*

Các khoảng mưa forecast được kiểm tra theo `(anchor,anchor+3h]`, `(anchor+3h,anchor+6h]`, `(anchor+6h,anchor+12h]` và `(anchor+12h,anchor+24h]`. Theo mô tả provider, giá trị tại một mốc giờ là lượng mưa của giờ liền trước; do đó cách chọn các mốc như trên không dùng trùng một giờ giữa hai khoảng. Toàn bộ cửa sổ tại một anchor lấy từ cùng một vintage, nghĩa là cùng một phiên bản bản dự báo. Một bản sửa được tải trong tương lai không được thay bản đã có trong quá khứ ở nhánh nghiêm ngặt.

### 6.3. Phạm vi không gian của forecast

{grid_summary}

{grid_table}

Các tọa độ và khoảng cách chi tiết được lưu trong [forecast_location_grids.csv](tables/forecast_location_grids.csv). Khi dùng chung ô lưới, không thể coi các yêu cầu là những nguồn thông tin mưa không gian độc lập, hoặc gọi forecast đó là phép dự báo đã được kiểm chứng tại từng gauge.

## 7. Kiểm tra các dataset đã dùng cho thí nghiệm trước

Quy trình đã kiểm tra {_number(len(tables['dataset_inventory']),0)} dataset đang còn trong `train_data/`. Các kiểm tra bao gồm timestamp, ranh giới split, ghép target đúng mốc tương lai và provenance của weather features. Có {_number(target_mismatches,0)} cặp target hữu hạn không khớp giá trị nước tại timestamp tương ứng trong chính bản hourly đóng băng của dataset. Phép so sánh này không lấy dữ liệu DB đã được sửa sau đó để kết luận dataset cũ đúng hay sai.

{provenance_table}

Việc giá trị weather feature khớp tổng từ vintage trong database chỉ xác nhận khả năng truy nguyên cách dựng feature. Nó không xác nhận forecast đó đã sẵn có tại anchor lịch sử. Ngược lại, việc tải archive sau anchor không tự động chứng minh giá trị bên trong có sử dụng quan trắc tương lai; loại archive và quá trình tạo bản dự báo vẫn phải được xem xét riêng.

Các bản ghi provenance ở phần thời gian dùng chung giữa các dataset có thể xuất hiện nhiều lần vì các dataset là các biến thể cùng nguồn. Không được cộng các dòng giữa các dataset để gọi đó là số mốc forecast hoặc số ví dụ độc lập. Dataset train mở rộng cũng đã dùng test cũ làm validation; kết quả của nó chưa thể được dùng như một final test độc lập khi Stage 4 lựa chọn feature.

## 8. Những vấn đề ảnh hưởng đến các stage sau

{findings_table}

Các kiểm tra cũng đã được lưu thành bảng [integrity_findings.csv](tables/integrity_findings.csv) để các stage sau có thể kiểm tra điều kiện đầu vào trước khi chạy, thay vì chỉ dựa vào lời nhắc trong báo cáo. Những phát hiện không yêu cầu sửa dữ liệu quan trắc gốc; các quyết định xử lý phải tạo sản phẩm dẫn xuất và giữ được thông tin nguồn.

## 9. Điều chỉnh đề xuất cho master plan

**Trước phần định lượng của Stage 2**, nên bổ sung một bước xác nhận contract mưa cho từng loại thiết bị: đơn vị, khoảng tích lũy, cách gắn timestamp và quy tắc reset. Có thể tiếp tục xem độ phủ và chẩn đoán nguồn trước khi xác minh xong, nhưng chưa nên xếp hạng trạm bằng tổng mưa vật lý chưa được xác định. Các đặc trưng nước và nhóm trạm cần được tính lại theo quy ước cuối giờ đã thống nhất.

**Đối với Stage 3**, nên tách hai nhánh kết quả: nhánh có bằng chứng khả dụng vận hành và nhánh nghiên cứu phụ thuộc giả định hoặc hindcast. Nếu nhánh nghiêm ngặt chưa có đủ anchor trùng quan trắc, không được chuyển sang dùng giả định rồi giữ tên đánh giá vận hành. Việc đối chiếu forecast với mưa đo cũng cần xác nhận phạm vi ô lưới và contract mưa quan trắc trước khi tính skill.

**Đối với Stage 4**, nên đóng băng protocol chia thời gian và anchors trước khi xem kết quả feature ablation. Các phương án E0–E5 phải được so sánh trên cùng các mốc hợp lệ hoặc ghi rõ số mốc bị mất do nguồn đầu vào. Những tập đã dùng để chọn checkpoint hoặc chọn feature vẫn là kết quả khám phá. Chưa có cơ sở từ Stage 1 để tuyên bố một feature giúp dự báo tốt hơn, nên báo cáo này không đưa ra thứ hạng trạm mưa hoặc feature set chiến thắng.

Kiến trúc chung và các điều kiện chuyển stage được ghi tại [README của Phase 2](../../../../research/station_analysis/phase2/README.md). Stage 2, Stage 3 và Stage 4 chưa được triển khai trong lượt này và CLI sẽ từ chối chạy chúng. Công việc dừng tại Stage 1 để review các nguồn dữ liệu và những giả định nêu trên.

## 10. Các kết quả có thể sử dụng lại và nguồn tham khảo

- [Ma trận có dữ liệu theo trạm và giờ](tables/data_availability_matrix.csv.gz) cho biết giờ nào có ít nhất một số đo hợp lệ; [ma trận đủ lưới lấy mẫu](tables/complete_observation_grid_matrix.csv.gz) cho biết giờ nào đủ các lần đo dự kiến.
- [Bảng khả dụng forecast theo từng anchor](tables/forecast_anchor_availability.csv.gz) giữ riêng địa điểm, mô hình, vintage và chính sách khả dụng. [Ma trận forecast nghiêm ngặt](tables/strict_forecast_availability_matrix.csv.gz) cho biết các mốc đáp ứng điều kiện có bằng chứng.
- [Bảng provenance từng dòng dataset](tables/dataset_forecast_rows.csv.gz) và [kiểm tra target](tables/dataset_target_alignment.csv) cho phép kiểm tra lại kết luận trong báo cáo.
- [Bảng kiểm kê artifact Phase 1](tables/phase1_artifacts.csv) ghi các sản phẩm đã có được dùng lại, trong khi cấu hình và manifest giữ mã kiểm tra nguồn, code và master plan.

Bản chụp nguồn được lưu tại `{snapshot_folder}`. Các bảng Parquet trung gian và SQLite phân tích nằm tại `{work}`; đây là nơi các stage tiếp theo sẽ đọc lại dữ liệu. Dữ liệu PostgreSQL, DEM gốc, dataset và checkpoint mô hình cũ không bị sửa.

Tài liệu về ý nghĩa dữ liệu forecast được đối chiếu với [Open-Meteo Single Runs API](https://open-meteo.com/en/docs/single-runs-api) và [Open-Meteo ECMWF API](https://open-meteo.com/en/docs/ecmwf-api). Các mô tả này được dùng để diễn giải hạn chế của archive, không thay thế bằng chứng thời điểm phát hành của từng run. Phần đơn vị quan trắc dựa trên metadata và quyết định đã ghi trong repository; chưa có tài liệu độc lập xác nhận khoảng tích lũy mưa cho từng thiết bị.
'''
    (output/'report.md').write_text(report,encoding='utf-8')
