"""Stage 4 design only: complete three-cell factorial plus bounded diagnostics."""
import pandas as pd

from .water_comparison_reporting import BASELINE

DA_VIEN = 'ecmwf_ifs:16.485060:107.571434'
THUAN_AN = 'ecmwf_ifs:16.555359:107.588180'
BACH_MA = 'ecmwf_ifs:16.274164:107.863250'
SEEDS = [17, 29, 42, 71, 103]


def stage4_design(grids):
    """Explicit IDs avoid silently changing the existing W0-W4 station candidates."""
    names = grids.set_index('grid_id').grid_name.to_dict()
    definitions = [
        ('W0', [], 'Đối chứng không có dự báo mưa'),
        ('W1', [BASELINE], 'Đối chứng ô A Lưới cũ'),
        ('W2', [DA_VIEN], 'Chỉ dùng ô gần Dã Viên trong tập đã thu thập'),
        ('W3', [BASELINE, THUAN_AN], 'Tổ hợp ứng viên từ train; chưa xác minh là vùng đầu nguồn'),
        ('W4', [DA_VIEN, BASELINE, THUAN_AN], 'Kiểm tra đồng thời cả ba ô'),
        ('W5', [THUAN_AN], 'Tách riêng giá trị của ô Thuận An'),
        ('W6', [BASELINE, DA_VIEN], 'Kiểm tra thêm ô Dã Viên vào ô A Lưới'),
        ('W7', [DA_VIEN, THUAN_AN], 'Kiểm tra hai ô mới khi bỏ ô A Lưới'),
    ]
    for _, ids, _ in definitions:
        if not set(ids).issubset(names):
            raise ValueError('The planned three-cell design is not covered by this archive')
    rows = []
    for arm, ids, reason in definitions:
        rows.append(dict(water_station_id='0074647403', arm=arm, grid_ids=ids,
                         grid_names=[names[g] for g in ids], grid_count=len(ids),
                         rain_features=4*len(ids), missing_masks=4*len(ids),
                         shared_age_features=int(bool(ids)), reason=reason))
    diagnostics = [
        dict(id='D1', question='Is the effect due to weather values or availability?',
             design='Weather value slots fixed at zero; keep the same masks and age as the real-weather arm.',
             scope='W1 and up to two train/chronological-validation finalists'),
        dict(id='D2', question='Is the effect due to more input slots/parameters?',
             design='Replace all added grids by repeated A Luoi values/masks; retain the same slot count and architecture.',
             scope='Two- and three-grid finalists; report trainable parameter count'),
        dict(id='D3', question='Does 12-24h forecast add useful predictive information?',
             design='E4: remove only 12-24h forecast values and corresponding masks; retain observed rain_24h.',
             scope='All seven nonempty Da Vien arms, same anchors and seeds'),
        dict(id='D4', question='Does weather add information beyond observed rain?',
             design='Compare E0/E1/E2/E3; keep a fixed observed-rain schema within each spatial comparison.',
             scope='E2 for seven nonempty Da Vien arms; E3 main factorial; collapse identical baselines'),
        dict(id='D5', question='Does train-selected observed rainfall change the spatial result?',
             design='E5 compared with E3 on W1 and finalists; if selected rain schema is identical, declare equivalence and do not retrain.',
             scope='At most two spatial finalists plus W1'),
        dict(id='D6', question='Does a result depend on permissive missing-data handling?',
             design='Compare standard 5/6 and strict 6/6 observed-rain policies on their common anchor intersection; report coverage separately.',
             scope='W1 and at most two finalists'),
        dict(id='D7', question='Does a result depend on assumed release time?',
             design='Research sensitivity at initialization+6h/+9h/+12h, max age 24h; reselect one common initialization per vector.',
             scope='W1 and at most two finalists, same anchor intersection; not evidence of operational publication'),
        dict(id='D8', question='Does a result depend on old hindcast versus newer archive?',
             design='Report 49r1/50r1 and season/event strata separately, including sample counts; calendar and version effects remain confounded.',
             scope='All completed arms; retrain within a version only if chronological train/validation and enough events exist'),
        dict(id='D9', question='Is useful information being missed outside the three-cell design?',
             design='Freeze at most one excluded-grid challenger per station using training-only additional associations and metadata; compare W1, challenger alone, W1+challenger.',
             scope='Conditional after primary analysis; choose without post-cutoff scores and publish the rule before fitting'),
    ]
    regional = [dict(id='R1', water_station_id='751117', grids=[BASELINE, BACH_MA],
                     baseline_grids=[BASELINE],
                     arm_grid_sets=[[], [BASELINE], [BACH_MA], [BASELINE, BACH_MA]],
                     reason='Train partial association at Trung Lang merits a separate station-specific check; rank association is not stable.')]
    return {
        'status': 'specification_only_not_executed',
        'design_scope': 'Complete factorial for these three Da Vien cells, not all seven cells or all possible research questions',
        'da_vien_factorial_arms': rows,
        'paired_seeds': SEEDS,
        'primary_training_jobs_da_vien': len(rows)*len(SEEDS),
        'primary_feature_group': 'E3; W0 equals E1 and must not be fitted twice',
        'diagnostics': diagnostics,
        'station_specific_checks': regional,
        'selection': {
            'discovery': 'Training data only; freeze cell/rain candidates before fitting',
            'validation': 'Chronological validation inside the pre-cutoff period; never use post-cutoff exploration to select finalists',
            'finalists': 'At most two Pareto-nondominated arms by validation N200 and RMSE averaged over all five paired seeds; choose lowest N200 then lowest RMSE if more than two remain',
            'evaluation': 'Post-cutoff results are exploratory; a future frozen untouched holdout is required for final operational claims',
        },
        'evaluation': {
            'metrics': ['RMSE_mm','MAE_mm','N200_strict_gt','N200_pct','peak_amplitude_error','peak_timing_error'],
            'strata': ['station','horizon_6_12','rising_falling','high_water','season','event','model_version'],
            'uncertainty': 'Report all seeds and paired differences; block/event resampling where enough independent events exist, otherwise state insufficient event replication',
            'no_synthetic_thresholds': 'Use verified station thresholds or train-derived quantiles frozen before validation; never substitute another station alarm level',
        },
        'safeguards': [
            'Freeze backbone, hyperparameters, epoch budget, early stopping and checkpoint rule across paired feature comparisons.',
            'Fit every scaler/imputer and train feature selection inside the chronological training fold only.',
            'Use station-specific identical anchors and both targets for every compared arm; report exclusions and purge.',
            'Training targets may not cross validation boundaries; use a 24-hour temporal purge for consistency and the overlapping forecast verification windows.',
            'Forecast source metadata must be available no later than anchor; retrospective +6h/+9h/+12h assumptions stay labelled hypothetical.',
            'No-train stations are excluded from local fitting, not filled with evaluation data.',
            'Keep forecast_history_policy fixed; masks, age and deduplication follow the existing contract.',
            'Do not assume nearest beats land; a nearest sensitivity requires a separate matched archive validation before any fitting.',
            'Do not silently change Model B, production checkpoints, datasets, or regional architecture.',
        ],
    }


def design_section(design, table_dir):
    frame = pd.DataFrame(design['da_vien_factorial_arms'])
    csv = frame.copy()
    csv['grid_ids'] = csv.grid_ids.map(';'.join)
    csv['grid_names'] = csv.grid_names.map(';'.join)
    csv.to_csv(table_dir/'da_vien_factorial_arms.csv', index=False)
    lines = ['| Phương án | Các ô lưới | Số ô | Giá trị mưa | Cờ thiếu | Tuổi dự báo | Câu hỏi |',
             '|---|---|---:|---:|---:|---:|---|']
    for r in frame.itertuples():
        lines.append(f'| {r.arm} | {", ".join(r.grid_names) or "Không dùng"} | {r.grid_count} | {r.rain_features} | {r.missing_masks} | {r.shared_age_features} | {r.reason} |')
    table='\n'.join(lines)
    return f'''### 8.1. Bộ đối chứng đầy đủ cho ba ô ứng viên tại Dã Viên

Giữ W0–W4 và bổ sung W5, W6, W7. Với ba ô A Lưới, Dã Viên và Thuận An, bảng này bao phủ toàn bộ tám tập con, gồm không dùng dự báo mưa, từng ô riêng, từng cặp và cả ba ô. Vì thế có thể kiểm tra giá trị bổ sung của từng ô trong các bối cảnh khác nhau, thay vì chỉ so một tổ hợp ba ô với một ô cũ. Đây là thiết kế đầy đủ trong phạm vi ba ô này; nó không chứng minh đã tìm ra tổ hợp tốt nhất của cả bảy ô.

{table}

[Bảng phương án bổ sung cho Dã Viên](tables/da_vien_factorial_arms.csv) ghi mã ô cụ thể. W5–W7 ở đây chỉ được định nghĩa cho Dã Viên, không tự áp cùng tên và cùng tọa độ cho mọi trạm. Danh sách W0–W4 của 19 trạm vẫn được giữ trong bảng ứng viên ban đầu. W3 là tổ hợp từ tuyển chọn trên train, chưa phải bằng chứng Thuận An là vùng đầu nguồn của Dã Viên.

Các phép đối chiếu cần báo cáo gồm W1→W3 và W6→W4 để xét phần bổ sung của Thuận An; W1→W6 và W3→W4 để xét phần bổ sung của Dã Viên; W5→W3 và W7→W4 để xét phần bổ sung của A Lưới. So W1/W2/W5 giữ cùng số ô; so W3/W6/W7 giữ cùng số ô. Các thay đổi phải được diễn giải bằng chênh lệch sai số trên cùng mốc giờ, không bằng chênh lệch correlation.

### 8.2. Những đối chứng bổ sung để giải thích kết quả

| Mã | Thắc mắc cần giải quyết | Thí nghiệm và phạm vi |
|---|---|---|
| D1 | Mô hình được lợi từ lượng mưa hay chỉ từ tình trạng có/thiếu dự báo? | Giữ cờ thiếu, tuổi dự báo và số kênh như phương án thật, nhưng đặt toàn bộ kênh lượng mưa dự báo bằng 0. Đối chiếu W1 và tối đa hai phương án được chọn bằng train/validation. |
| D2 | Thêm ô tốt hơn vì có thông tin mới hay chỉ vì tăng số đầu vào/tham số? | Thay các ô bổ sung bằng bản sao dữ liệu A Lưới, giữ cùng số kênh và cùng backbone như tổ hợp hai/ba ô. Công bố số tham số và so với tổ hợp thật; đây là đối chứng thông tin trùng lặp, không phải nguồn dự báo mới. |
| D3 | Dự báo mưa 12–24h có giúp dự báo nước +6/+12h không? | Thực hiện E4 cho bảy phương án có forecast: chỉ bỏ khoảng 12–24h và mask tương ứng. Giữ nguyên mưa quan trắc quá khứ 24h, lịch sử nước và anchor. |
| D4 | Forecast có ích khi đã có mưa quan trắc không? | Giữ E0–E3 theo master plan: chỉ nước, nước+mưa quan trắc, nước+forecast, và cả ba nhóm. E2 xét bảy phương án có forecast; E3 là ma trận chính. W0 của E3 chính là E1, không train trùng. |
| D5 | Kết luận có phụ thuộc cách chọn trạm mưa quan trắc không? | So E5 với E3 trên W1 và tối đa hai phương án. Đóng băng hai schema mưa trước khi chạy; nếu schema giống nhau thì ghi hai cấu hình tương đương, không train lại cùng một thí nghiệm. |
| D6 | Kết quả có do cách xử lý giờ thiếu tạo ra không? | So chính sách mưa quan trắc 5/6 và 6/6 trên phần giao anchor cho W1 và các phương án được chọn. Báo cáo riêng mức mất dữ liệu; không biến giờ thiếu thành không mưa. |
| D7 | Kết quả có phụ thuộc giả định dự báo sẵn có sau 6 giờ không? | Kiểm tra thêm giả định sau 9h và 12h trên W1 và các phương án được chọn; giữ tuổi tối đa 24h và chọn lại cùng một chu kỳ cho các ô. Mỗi phép đối chiếu dùng giao anchor; cả ba giả định đều chưa chứng minh thời điểm công bố thực tế. |
| D8 | Tín hiệu có chỉ xuất hiện ở một mùa, một trận hoặc một phiên bản ECMWF không? | Tách kết quả theo 49r1/50r1, mùa và sự kiện. Đây trước hết là phân tầng kết quả, không phải chạy lại toàn bộ ma trận. Chỉ train riêng từng phiên bản nếu còn đủ dữ liệu train/validation theo thời gian và đủ sự kiện; nếu không, ghi rõ chưa thể tách ảnh hưởng phiên bản khỏi thời gian. |
| D9 | Có bỏ sót một ô ngoài ba ô chính không? | Cho phép tối đa một ô thách thức cho mỗi trạm, được cố định bằng liên hệ bổ sung và metadata trên train trước khi fit. So ô đó riêng và A Lưới+ô đó với A Lưới. Không chọn bằng correlation hoặc sai số sau cutoff; nếu không đủ bằng chứng train thì không mở nhánh này. |

Các đối chứng D1/D2/D5/D6/D7 chỉ chạy cho W1 và tối đa hai phương án được chọn bằng validation theo thời gian trong giai đoạn trước cutoff. Quy tắc là xét các phương án không bị phương án khác vượt ở cả N200 và RMSE, lấy trung bình năm seed; nếu còn quá hai phương án thì ưu tiên N200 thấp hơn rồi RMSE thấp hơn. Không chọn một seed đẹp nhất. Với D2 chỉ các phương án hai/ba ô mới cần đối chứng sao chép kênh.

D9 không tự động thêm ô Ô Lâu chỉ vì kết quả đánh giá của Dã Viên cao hơn. Nếu một ô ngoài danh sách được chọn thì phải công bố trước mã ô, tiêu chí trên train và phạm vi thử. Kiểm tra `nearest` so với `land` cũng chưa đưa vào ma trận chính: cần bộ lịch sử tương ứng, cùng chu kỳ và cùng anchor trước khi đưa ra phép so sánh công bằng.

### 8.3. Kiểm chứng theo từng trạm và khả năng tổng quát hóa

Tại **Trung Làng, Quảng Thái (751117)**, bổ sung bốn đối chứng riêng: **không forecast, chỉ A Lưới, chỉ Bạch Mã thấp và A Lưới+Bạch Mã thấp**. Cách này phân biệt được ô Bạch Mã có ích khi đứng riêng hay chỉ khi kết hợp với A Lưới. Cơ sở là tín hiệu riêng phần đã có trên train, nhưng Spearman chưa ổn định; đây là giả thuyết cần kiểm chứng tại trạm đó, không phải khuyến nghị mặc định cho Dã Viên hoặc cả vùng.

Với các trạm còn lại có đủ dữ liệu, bắt đầu từ W0–W4 đã cố định bằng train và vị trí. Nếu cần phân giải một tổ hợp, dùng các tập con của tối đa ba ô ứng viên theo cùng nguyên tắc của Dã Viên, công bố trước khi fit. Không mở toàn bộ 2^7 tổ hợp cho mọi trạm. Hai trạm chưa có dữ liệu train không được lấp bằng phần đánh giá để tạo thí nghiệm local. Việc Regional Model chia sẻ trọng số hoặc fine-tune từng trạm vẫn là bước nghiên cứu sau khi chốt feature ablation, không được coi là đã triển khai ở đây.

### 8.4. Cách chạy, đánh giá và giới hạn của kế hoạch

Ma trận E3 chính cho Dã Viên gồm **8 phương án × 5 seed = 40 lượt train**, với seed cố định **17, 29, 42, 71, 103**. Số này chưa bao gồm E0/E2/E4/E5, các đối chứng có điều kiện và các trạm khác. Chỉ nước và persistence phải có mặt; persistence không cần train. Các thí nghiệm trùng schema phải được gộp. Không nhân toàn bộ D1–D9 với tất cả E0–E5, W0–W7 và mọi trạm khi chưa có lý do.

Trước khi chạy cần cố định backbone, learning rate, loss, ngân sách epoch, early stopping, tiêu chí checkpoint và cách đưa forecast vào chuỗi. Khi so feature, không đồng thời đổi kiến trúc hoặc tối ưu riêng từng nhánh. Với cùng một phép đối chiếu, giữ cùng hai nhãn +6/+12h, cùng anchor, cùng split và loại 24 giờ ở ranh giới; fit scaler, imputer và tuyển chọn trạm chỉ trong phần train của từng fold. Việc thay số input do thêm ô được ghi rõ và được kiểm tra bằng D2.

Dùng validation theo thời gian bên trong giai đoạn trước **01/08/2026 00:00 UTC+7** để chọn phương án và checkpoint. Phần sau cutoff đã được xem nhiều lần chỉ dùng đánh giá thăm dò, không được dùng chọn phương án rồi gọi là final test độc lập. Các fold và sự kiện đủ điều kiện phải được chốt trước khi chạy; nếu không đủ nhiều sự kiện độc lập thì kết luận về tính ổn định phải để ngỏ. Một khoảng dữ liệu tương lai được đóng băng trước khi xem kết quả vẫn cần thiết cho tuyên bố cuối cùng về vận hành.

Báo cáo MAE, RMSE, số lỗi nghiêm ngặt >200 mm và tỷ lệ lỗi theo từng trạm, từng horizon và từng seed; bổ sung chênh lệch ghép cặp giữa các phương án. Tách nước tăng/giảm, mực cao, mùa và sự kiện; đánh giá biên độ đỉnh và thời điểm đỉnh khi có sự kiện đủ dữ liệu. Ngưỡng mực cao phải có căn cứ riêng cho trạm hoặc được xác định bằng quantile trên train, không mượn ngưỡng báo động của trạm khác. Nếu đủ số đợt độc lập, ước lượng độ bất định bằng lấy mẫu lại theo khối hoặc sự kiện; không lấy từng giờ chồng lấn làm mẫu độc lập.

Kế hoạch này bao phủ các thắc mắc chính về nguồn tín hiệu, giá trị bổ sung của từng ô, số lượng input, cửa sổ mưa, missing data và tính ổn định. Nó không bảo đảm giải quyết mọi câu hỏi có thể phát sinh: hiệu quả vận hành, khả năng tổng quát sang một mùa lũ mới và tác động tách biệt của phiên bản ECMWF có thể vẫn chưa xác định được với dữ liệu hiện tại. **Toàn bộ nội dung trên mới là đặc tả, chưa thực hiện Stage 4 hoặc huấn luyện mô hình.**
'''
