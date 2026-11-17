"""Generate both Stage 2 reports from one set of computed results."""
from pathlib import Path

import pandas as pd


def markdown_table(frame):
    """Render without the optional tabulate dependency, preserving station IDs."""
    def cell(value):
        if pd.isna(value):
            return '—'
        if isinstance(value, float):
            return f'{value:.4f}'
        return str(value).replace('|', '/').replace('\n', ' ')
    return '\n'.join([
        '| ' + ' | '.join(map(str, frame.columns)) + ' |',
        '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |',
        *['| ' + ' | '.join(cell(v) for v in row) + ' |'
          for row in frame.itertuples(index=False, name=None)],
    ])


def write_stage2_report(summary_df, seasonal_df, decay_df, clusters_df, events_df,
                        eval_df, cond_df, loo_df, sens_df, multi_eval_df, excl_df,
                        figures, output_dir, *, anchors, cutoff, rainfall, targets,
                        episodes, snapshot, stations):
    """Write reproducible tables, master report and clarification from live results."""
    output = Path(output_dir)
    tables = output / 'tables'
    tables.mkdir(parents=True, exist_ok=True)
    frames = {
        '01_station_rainfall_summary': summary_df,
        '02_seasonal_rainfall_summary': seasonal_df,
        '03_spatial_distance_decay': decay_df,
        '04_rainfall_clusters': clusters_df,
        '05_independent_rainfall_events': events_df,
        '06_single_station_candidates': eval_df,
        '07_conditional_relationships': cond_df,
        '08_leave_one_out_stability': loo_df,
        '09_policy_sensitivity': sens_df,
        '10_multistation_combinations': multi_eval_df,
        '11_station_exclusion_list': excl_df,
    }
    boundary = pd.Timestamp(cutoff).tz_convert('UTC')
    train = anchors + pd.Timedelta(hours=12) <= boundary
    after = anchors > boundary
    purged = (anchors <= boundary) & ~train
    names = summary_df.set_index('station_id')['name'].to_dict()
    period_rows = []
    for label, mask in [('Train sau purge', train), ('Vùng purge', purged),
                        ('Giai đoạn đánh giá sau cutoff', after)]:
        for sid in ['750062', '750066']:
            rain = rainfall.loc[mask, sid]
            period_rows.append({'period': label, 'station_id': sid, 'name': names.get(sid, sid),
                                'anchor_hours': int(mask.sum()), 'rain_valid_hours': int(rain.notna().sum()),
                                'rain_total_mm': float(rain.sum(min_count=1))})
    period_df = pd.DataFrame(period_rows)
    frames['12_period_rainfall_audit'] = period_df
    lag_counts = eval_df.groupby('best_lag_dh12').size().rename('pair_count').reset_index()
    frames['13_optimal_lag_counts'] = lag_counts
    # Event descriptions are computed; an event is a flagged interval, not a named historical flood.
    dv = targets['0074647403']
    r6 = rainfall['750062'].rolling(6, min_periods=6).sum()
    episode_rows = []
    for number, (start, end) in enumerate(episodes, 1):
        h = dv.loc[start:end, 'water_level_mm']
        episode_rows.append({'episode': number, 'start_utc': str(start), 'end_utc': str(end),
                             'span_hours': int((end-start)/pd.Timedelta(hours=1)) + 1,
                             'valid_water_hours': int(h.notna().sum()),
                             'max_water_mm': h.max(),
                             'max_dh12_mm': dv.loc[start:end, 'dh_p12h'].max(),
                             'hong_thai_max_rain6_mm': r6.loc[start:end].max(),
                             'hong_thai_rain_total_mm': rainfall.loc[start:end, '750062'].sum(min_count=1)})
    ep_df = pd.DataFrame(episode_rows)
    frames['14_water_episodes'] = ep_df
    scope = pd.DataFrame([
        {'period': label, 'first_anchor_utc': str(anchors[mask].min()),
         'last_anchor_utc': str(anchors[mask].max()), 'anchor_count': int(mask.sum()),
         'valid_dh12': int(dv.loc[mask, 'dh_p12h'].notna().sum()),
         'abs_dh12_ge200': int(dv.loc[mask, 'dh_p12h'].abs().ge(200).sum())}
        for label, mask in [('train', train), ('purge', purged), ('evaluation', after)]
    ])
    frames['15_split_audit'] = scope
    for name, frame in frames.items():
        frame.to_csv(tables / f'{name}.csv', index=False)

    candidates = eval_df[eval_df.water_station_id.astype(str) == '0074647403'].sort_values('best_corr_dh12', ascending=False)
    candidate_cols = ['rain_station_id', 'rain_station_name', 'distance_km', 'elevation_m',
                      'best_corr_dh12', 'best_window_dh12', 'best_lag_dh12',
                      'best_corr_dh6', 'best_window_dh6', 'best_lag_dh6',
                      'best_corr_h', 'best_window_h', 'best_lag_h']
    candidate_md = markdown_table(candidates[candidate_cols].head(10))
    combo_cols = ['subset_name', 'train_n', 'evaluation_n', 'max_collinearity',
                  'train_pearson_r', 'test_pearson_r', 'test_r2']
    combo_md = markdown_table(multi_eval_df[combo_cols])
    sensitivity_md = markdown_table(sens_df[['rain_station_id', 'sample_gain', 'r_standard',
                                             'r_strict', 'r_difference', 'common_n',
                                             'common_anchor_r_difference']])
    lag02 = int(eval_df.best_lag_dh12.isin([0, 2]).sum())
    ht = candidates[candidates.rain_station_id.astype(str) == '750062'].iloc[0]
    cluster_md = markdown_table(clusters_df.groupby(['cluster_id', 'cluster_name']).size().rename('station_count').reset_index())
    # Plain Vietnamese paragraphs explain what the estimates can and cannot support.
    master = f'''# Stage 2 — Lượng mưa và quan hệ với mực nước tại Huế

## 1. Mục tiêu và phạm vi

Phân tích này khảo sát lượng mưa quan trắc, mối liên hệ với mực nước và các cấu hình trạm mưa đáng thử ở bước ablation. Kết quả tương quan giúp sàng lọc giả thuyết; kết quả này chưa chứng minh một cấu hình sẽ cải thiện ResidualLSTM và không thay thế Model B.

Dữ liệu được lấy từ snapshot `{Path(snapshot).name}`, gồm {len(summary_df)} trạm mưa và {len(targets)} trạm mực nước. Chuỗi phân tích có {len(anchors):,} mốc giờ, từ {anchors.min().tz_convert('Asia/Ho_Chi_Minh')} đến {anchors.max().tz_convert('Asia/Ho_Chi_Minh')}. Đây là khoảng quan trắc chưa đủ một năm, vì vậy tổng mưa và tỷ trọng mùa dưới đây không được hiểu là thống kê khí hậu năm.

## 2. Cách xử lý và ranh giới nghiên cứu

Mưa được xử lý theo quy ước của dự án: mỗi bản ghi là lượng mưa trong 10 phút kết thúc tại timestamp, đơn vị mm. Quy ước này là hợp đồng xử lý dùng trong nghiên cứu, không phải kết quả xác minh độc lập bằng thiết bị hoặc tài liệu gốc của nhà cung cấp. Tổng giờ tại t sử dụng khoảng (t−1h, t]. Có đủ sáu bản ghi thì tính tổng; có năm bản ghi thì ước tính tổng bằng cách nhân 6/5; có ít hơn năm thì giữ thiếu. Ước tính 5/6 giả định cường độ trong phần thiếu tương tự phần quan trắc, nên không được xem như số đo thật.

Mực nước giờ và target +6h/+12h dùng đúng timestamp trên lưới giờ, không nội suy nhãn. Timestamp quan trắc không chứng minh bản ghi đã sẵn có ở thời điểm dự báo; dữ liệu đã backfill cần được phân biệt với bằng chứng vận hành as-of. Các cửa sổ mưa được tìm trong 1, 3, 6, 12, 24 và 48 giờ; lag được khảo sát từ 0 đến 72 giờ với bước 2 giờ. Input luôn kết thúc tại hoặc trước anchor.

Cutoff duy nhất lấy từ config là **{pd.Timestamp(cutoff)}**, tương ứng **{boundary}**. Để target không vượt cutoff, quá trình tuyển chọn chỉ dùng anchor có t+12h ≤ cutoff. Các anchor sau cutoff được dùng cho đánh giá thăm dò. Vùng giữa hai tập được loại khỏi việc fit và tuyển chọn.

{markdown_table(scope)}

Số giờ trong bảng là kích thước lưới. Số cặp hợp lệ còn phụ thuộc dữ liệu thiếu. Hai horizon dùng cùng giới hạn train 12 giờ để bảo đảm phạm vi tuyển chọn nhất quán. Cutoff này khác bản báo cáo cũ dùng 00:00 UTC; mọi số liệu tuyển chọn đã được tính lại.

## 3. Lượng mưa và cấu trúc không gian

Bảng sau liệt kê mười trạm có tổng mưa quan trắc lớn nhất trong khoảng phân tích. Tổng mưa là tổng các giờ hợp lệ, có thể bao gồm giờ ước tính 5/6; không bù cho các giờ hoàn toàn thiếu.

{markdown_table(summary_df[['station_id', 'name', 'elevation_m', 'valid_hours', 'total_rainfall_mm', 'max_1h_rainfall_mm']].sort_values('total_rainfall_mm', ascending=False).head(10))}

![Vị trí trạm và cao độ điểm DEM](figures/01_dem_station_network_map.png)

Cao độ ở bảng này là giá trị pixel tại tọa độ trạm mưa, chưa phải thống kê địa hình lân cận và không phải cao độ mốc đo mực nước. DEM không đủ để xác nhận trạm nằm trong lưu vực nào. Bạch Mã 750041, VQG Bạch Mã 750030 và Đỉnh Bạch Mã 750066 là ba trạm riêng; kết quả của trạm này không được gán cho trạm khác.

Phân tích theo mùa dùng tháng địa phương UTC+7: tháng 9–12 được quy ước là mùa lũ và tháng 1–8 là mùa còn lại. [Bảng theo mùa](tables/02_seasonal_rainfall_summary.csv) phản ánh phần thời gian được quan trắc, không đại diện một chu kỳ năm đầy đủ.

![Tương quan mưa giữa các trạm](figures/02_spatial_rain_correlation_matrix.png)

{markdown_table(decay_df)}

Các cụm được tạo từ tương quan toàn kỳ, tọa độ và cao độ. Chúng phục vụ mô tả không gian, chưa được kiểm định như ranh giới lưu vực hoặc nhóm tối ưu cho regional training. Nếu dùng cụm để chọn feature ở Stage 4, phải fit lại trên train và kiểm tra độ nhạy với cách chuẩn hóa các nhóm đặc trưng.

{cluster_md}

## 4. Tương quan với mực nước

Bảng dưới là mười ứng viên đứng đầu tại Dã Viên theo tương quan với ΔH+12h trên train đã purge. Mỗi cột tương quan có cửa sổ và lag riêng; không được ghép giá trị tương quan của một cấu hình với lag của cấu hình khác.

{candidate_md}

Riêng Hồng Thái, tương quan cực đại với H(t) trong lưới train là **{ht.best_corr_h:.4f}**, ứng với **cửa sổ {int(ht.best_window_h)} giờ và lag {int(ht.best_lag_h)} giờ**. Đây là độ trễ thống kê của một cấu hình cụ thể, không phải thời gian truyền lũ vật lý.

Trong {len(eval_df)} cặp trạm được khảo sát, có **{lag02} cặp** có lag tối ưu cho ΔH+12h bằng 0 hoặc 2 giờ. Các cặp còn lại có lag khác. Vì vậy không có cơ sở áp dụng kết luận “lag 0–2h tối ưu” cho toàn bộ mạng lưới. Việc tìm cực đại qua nhiều cấu hình cũng có thể làm tương quan trong train lạc quan; Stage 4 cần kiểm tra giá trị dự báo ngoài train.

![Tương quan theo lag trên toàn kỳ, cửa sổ mưa 6 giờ](figures/03_lag_cross_correlation_profiles.png)

![Đáp ứng theo cửa sổ và lag trên toàn kỳ](figures/04_lag_heatmaps_dh6_dh12.png)

Hai hình trên mô tả toàn kỳ và có thể dùng phạm vi cấu hình khác bảng tuyển chọn. Chúng không được dùng làm bằng chứng tuyển chọn train-only. Các kết quả điều kiện cũng tính trên toàn kỳ; nhóm được xác định bằng ΔH tương lai chỉ có ý nghĩa phân tích hồi cứu, không thể dùng để chuyển mô hình lúc dự báo.

![Tương quan theo các điều kiện hồi cứu](figures/05_conditional_relationship_analysis.png)

## 5. Độ nhạy với sự kiện và chính sách thiếu

Thuật toán nhận diện sự kiện đánh dấu H(t) ≥ phân vị 90% toàn kỳ hoặc |ΔH+12h| ≥ 200 mm, sau đó tách hai đợt khi khoảng gián đoạn đạt ít nhất 12 giờ. Thuật toán tìm được **{len(episodes)} khoảng sự kiện**. Đây là các khoảng biến động do thuật toán tạo ra, không phải danh mục trận lũ lịch sử đã được xác minh độc lập.

{markdown_table(loo_df[['rain_station_id', 'baseline_r', 'loo_mean_r', 'loo_std_r', 'loo_min_r', 'loo_max_r', 'stability_score', 'n_episodes']])}

LOO loại các anchor nằm trong từng khoảng rồi tính lại tương quan trên phần còn lại của toàn kỳ. Cửa sổ mưa và nhãn ở sát mép sự kiện vẫn có thể chạm sự kiện; đây là kiểm tra độ nhạy, chưa phải kiểm thử trên các trận lũ độc lập. Điểm stability_score chỉ tóm tắt độ phân tán của các lần loại, có thể che giấu một sự kiện chi phối. Cần đọc cùng giá trị nhỏ nhất và bảng sự kiện, không diễn giải điểm cao là đã tổng quát hóa tốt.

{markdown_table(ep_df.head(3))}

Không áp ngưỡng Kim Long hay trạm khác để kết luận Dã Viên vượt báo động. Dữ liệu này cũng chưa chứng minh ảnh hưởng riêng của triều, xả hồ hay hướng đi của một cơn bão.

![Độ nhạy khi lần lượt loại sự kiện](figures/06_event_stability_loo.png)

Độ nhạy chính sách 5/6 được trình bày riêng cho từng trạm:

{sensitivity_md}

r_difference so sánh hai chính sách trên các tập anchor hợp lệ riêng, vì vậy phản ánh đồng thời thay đổi số mẫu và giá trị. common_anchor_r_difference dùng đúng giao hai tập anchor để tách tác động do giá trị. Khi một giờ đạt 6/6 và mọi giờ trong cửa sổ đều đầy đủ, hai chính sách cho cùng tổng mưa. Không có một mức sai lệch chung dưới 0,0007 áp dụng cho tất cả các trạm.

## 6. So sánh các tổ hợp trạm

Các hàng dưới dùng Ridge(alpha=1) với duy nhất tổng mưa 6 giờ tại lag 0 để dự báo ΔH+12h. Mọi tổ hợp dùng cùng tập anchor hợp lệ của hợp tất cả trạm tham gia, cả train lẫn giai đoạn đánh giá. Đa cộng tuyến được tính trên cùng các anchor train đã purge. Bảng được sắp theo tương quan train, không dùng kết quả ngoài train để chọn tổ hợp.

{combo_md}

Tên “Model B” trong bảng chỉ biểu thị cặp trạm đầu vào; đây không phải dự báo của checkpoint Model B. Ridge không có lịch sử mực nước và chưa đo lợi ích bổ sung so với baseline water-only. Tương quan mưa giữa hai trạm thấp hơn không tự động chứng minh thông tin độc lập hoặc dự báo tốt hơn. R² âm ngoài train có nghĩa mô hình này kém hơn dự báo bằng trung bình target của chính giai đoạn đánh giá theo định nghĩa R²; không thể suy rộng thành ResidualLSTM sẽ thất bại.

![So sánh các cấu hình Ridge trên cùng anchor](figures/07_station_selection_collinearity.png)

Tổng mưa theo từng phần thời gian được tính trực tiếp để đối soát:

{markdown_table(period_df)}

Tổng mưa của cả một giai đoạn không đủ để kết luận tâm mưa của từng sự kiện đã chuyển hướng hoặc xác nhận nguyên nhân tương quan giảm. Các giải thích này cần bản đồ sự kiện, cùng cửa sổ thời gian và thông tin lưu vực đã kiểm chứng. Báo cáo này không giữ các khẳng định về một “trận bão tháng 9” khi chưa có chuỗi bằng chứng tương ứng.

## 7. Khuyến nghị và chuyển tiếp Stage 3

Model B tiếp tục là baseline hiện có. Hồng Thái + Nhâm, Hồng Thái + Bạch Mã 750041 và Hồng Thái + Bạch Mã 750041 + Phú Bài là các cấu hình ứng viên để kiểm tra ở Stage 4; không có cấu hình nào được Stage 2 xác nhận tối ưu. Nếu muốn dùng VQG Bạch Mã 750030 hoặc Đỉnh Bạch Mã 750066, phải tạo nhánh thí nghiệm riêng thay vì coi các mã này thay thế tương đương.

Danh sách dưới là các trạm không đủ độ bao phủ cho cửa sổ train đang chọn, không phải kết luận thiết bị không đáng tin cậy. Khoảng trước quan trắc đầu tiên cũng được tính vào thiếu, nên cần tách nguyên nhân bắt đầu hoạt động muộn với mất dữ liệu trong thời gian hoạt động.

{markdown_table(excl_df)}

Stage 3 phải dùng lại snapshot này và đối chiếu forecast với tổng mưa đúng khoảng hiệu lực. Kết quả Stage 1 cho biết các forecast lịch sử có hạn chế về bằng chứng thời điểm sẵn có; hindcast, archive có thời điểm phát giả định và forecast có bằng chứng truy xuất thực tế phải được báo cáo riêng. Không được gọi kết quả hồi cứu là khả năng vận hành thời gian thực đã được xác nhận.

Giai đoạn sau cutoff đã được xem nhiều lần, nên là tập đánh giá thăm dò, không còn là test độc lập để chốt mô hình. Trước Stage 4 cần đóng băng thiết kế split, purge theo horizon lớn nhất, fit scaler/selection trên train và báo cáo trên cùng anchor. Không training hoặc thay đổi pipeline mô hình trong lần chốt EDA này.

[Báo cáo làm rõ tám nhóm câu hỏi](stage2_clarification_report.md) · [Phạm vi split](tables/15_split_audit.csv) · [Danh mục sự kiện](tables/14_water_episodes.csv) · [Manifest và dấu vết tái lập](run_manifest.json)
'''
    (output / 'report.md').write_text(master, encoding='utf-8')
    from .stage2_clarification import audit_clarification
    clarification, _, _ = audit_clarification(
        output=output, rainfall=rainfall, targets=targets, stations=stations,
        candidates=eval_df, combinations=multi_eval_df, clusters=clusters_df,
        episodes=episodes, cutoff=cutoff, snapshot=snapshot, table=markdown_table)
    (output / 'stage2_clarification_report.md').write_text(clarification, encoding='utf-8')
    diagnostic = pd.read_csv(output/'tables/clarification/07_conditional_diagnostics.csv')
    evaluated = diagnostic[diagnostic.scope.eq('Đánh giá thăm dò')]
    def measured(signal, condition):
        return float(evaluated.loc[evaluated.signal.eq(signal) & evaluated.condition.eq(condition), 'r'].iloc[0])
    strong_hours = int(evaluated.loc[evaluated.signal.eq('Rain6 750066') & evaluated.condition.eq('Retrospective abs future dh12 >=200'),'n'].iloc[0])
    first_water = targets['750048'].water_level_mm.dropna().index.min()
    master += f'''
## 8. Đối soát bổ sung theo tám nhóm câu hỏi

Audit bổ sung đã tái tính độc lập 55 hàng ứng viên Dã Viên tại cấu hình lưu trong CSV, xác định sự kiện chi phối LOO và đối chiếu mưa cùng biến thiên mực nước giữa hai giai đoạn. Bảng nháp và protocol cũ được giữ riêng để truy nguyên trong báo cáo clarification; các artifact trước lượt audit nằm trong thư mục audit_history.

Trong giai đoạn đánh giá, tương quan R6 Hồng Thái với ΔH+12h là {measured('Rain6 750062','All'):.4f} trên toàn giai đoạn và {measured('Rain6 750062','Flood months Sep-Dec'):.4f} khi chỉ xét các tháng 9–12. Tín hiệu yếu này không biến mất chỉ vì gộp mùa kiệt. Đỉnh Bạch Mã 750066 đạt {measured('Rain6 750066','All'):.4f} trên toàn giai đoạn và {measured('Rain6 750066','Retrospective abs future dh12 >=200'):.4f} trong {strong_hours} giờ biến động mạnh. Đây là khác biệt thống kê có thể kiểm tra, chưa chứng minh hướng đi của một cơn bão hoặc hiệu quả bổ sung của Bạch Mã 750041.

Cầu Rào Bại có quan trắc giờ hợp lệ đầu tiên trong snapshot tại {first_water}; tương tự Cầu Máng cũng bắt đầu trong tháng 10/2026. Nhận định cũ hai trạm có dữ liệu từ tháng 8 không khớp artifact. Audit lưu timestamps thực tế thay vì suy ra ngày lắp đặt.

[Báo cáo trả lời đủ tám nhóm câu hỏi](stage2_clarification_report.md) · [Các bảng kiểm chứng bổ sung](tables/clarification/)
'''
    (output/'report.md').write_text(master, encoding='utf-8')
    # Keep report generation reproducible even when called by the full rainfall entrypoint.
    from datetime import datetime, timezone
    import hashlib
    import json
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    artifacts = [output/'report.md', output/'stage2_clarification_report.md']
    artifacts += list((output/'tables/clarification').glob('*.csv'))
    artifacts += [output/'figures/08_clarification_influential_episode.png']
    record = {
        'stage': 'stage2_targeted_clarification_audit',
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'snapshot_directory': str(snapshot),
        'snapshot_manifest_sha256': sha(Path(snapshot)/'manifest.json'),
        'cutoff': str(cutoff),
        'code_sha256': {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')},
        'output_sha256': {str(p.relative_to(output)): sha(p) for p in artifacts},
    }
    (output/'clarification_manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    return output / 'report.md'
