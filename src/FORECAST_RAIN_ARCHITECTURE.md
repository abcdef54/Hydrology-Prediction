# Bổ sung dự báo mưa vào mô hình dự báo mực nước

**Architecture & Implementation Plan — v0.1**
**Ngày lập:** 18/10/2026
**Trạng thái:** Kế hoạch triển khai; chưa tích hợp nguồn dự báo mưa vào training hoặc backend.

**Thứ tự thí nghiệm đã thống nhất, cập nhật 18/10/2026:** chọn số trạm mưa đầu
vào trước → thử feature dự báo thời tiết → cuối cùng so sánh 5 và 24 output.
Tài liệu bao gồm các bước đối chứng này để không trộn tác động của nhiều thay đổi.

Tài liệu mở rộng [ARCHITECTURE.md](ARCHITECTURE.md), tập trung vào dự báo mưa
ECMWF qua Open-Meteo cho trạm Dã Viên trước, sau đó có thể cấu hình cho trạm khác.

Quy ước:

- **[ĐÃ THỐNG NHẤT]:** mục tiêu và ý nghĩa tín hiệu đã trao đổi với chủ dự án.
- **[ĐỀ XUẤT]:** thiết kế triển khai, được điều chỉnh khi có kết quả kiểm thử.
- **[CẦN XÁC MINH]:** điều kiện dữ liệu/chất lượng phải kiểm tra trước khi sử dụng.
- **[ĐÃ KHẢO SÁT]:** có artifact đối chiếu; không đồng nghĩa đã triển khai production.

## 1. Công việc này là gì?

**[ĐÃ THỐNG NHẤT]** Xây dựng pipeline thu thập, lưu lịch sử từng bản dự báo mưa,
tạo feature và train lại mô hình nước với thông tin có sẵn tại thời điểm dự báo.

Hiện mô hình dùng mưa đã quan trắc. Đầu vào mới bổ sung mưa có thể xảy ra trong
tương lai, giúp mô hình có cơ hội nhận biết sớm một đợt nước tăng. Không bắt buộc
phải xây dựng mô hình dự báo mưa riêng ngay từ đầu.

Lượng mưa dự báo mới nhất có thể là tín hiệu sớm về mưa lớn sắp xảy ra, dù
chưa khớp hoàn toàn với lượng mưa đo được sau đó. Mô hình phải học cách sử dụng
thông tin này từ dữ liệu; không viết quy tắc cứng “mưa tăng thì cộng thêm nước”.
Tín hiệu mưa lớn không tự xác nhận có bão hoặc lũ tại Dã Viên.

### 1.1. Hai hướng thực hiện

| Hướng | Đầu vào → đầu ra | Thứ tự |
|---|---|---|
| Bổ sung feature cho mô hình nước | Nước/mưa quá khứ + dự báo mưa → nước tương lai | Sau khi thử số trạm mưa |
| Hiệu chỉnh dự báo mưa riêng | Dự báo ECMWF + quan trắc quá khứ → mưa tương lai đã hiệu chỉnh | Chỉ thử sau khi đánh giá hướng đầu |

### 1.2. Phạm vi và thứ tự đã thống nhất

- Dã Viên; ở hai bước đầu giữ 5 horizon +1/+3/+6/+12/+24 giờ và lịch sử đầu vào 96 giờ.
- Giữ cách tổng hợp quan trắc hiện tại để so sánh có kiểm soát.
- **Bước 1:** thử 4 trạm mưa hiện tại so với 1 và 2 trạm có correlation cao nhất;
  chưa thêm feature thời tiết mới.
- **Bước 2:** cố định cấu hình trạm đã chọn, thử thêm lượng mưa dự báo mới nhất đã có tại thời điểm dự báo nước.
- **Bước 3, cuối cùng:** cố định nguồn/feature, thử 24 output cho +1 đến +24 giờ
  so với 5 output hiện tại.
- Tách thử nghiệm feature khỏi thay đổi số output, loss, correction penalty,
  kiến trúc lớn hoặc thay đồng loạt các nguồn mưa.
- Đánh giá cả pha nước lên, nước rút, mưa ít và các đợt mưa lớn.

### 1.3. Bước 1 — ablation số trạm mưa quan trắc

**[ĐÃ THỐNG NHẤT]** Bốn nguồn đang nói tới là bốn **trạm mưa** đầu vào của mô
hình nước Dã Viên, không phải bốn trạm nước cần dự báo. Cần kiểm tra liệu các
trạm gần nhau có thông tin dư thừa và một/hai trạm đã đủ hiệu quả hay chưa.

| Nhóm | Nguồn mưa | Feature thời tiết mới | Output |
|---|---|---|---|
| S4 | Cả bốn trạm hiện tại | Không | 5 |
| S1 | Một trạm đứng đầu theo train | Không | 5 |
| S2 | Hai trạm đứng đầu theo train | Không | 5 |

Các việc phải làm:

1. Chốt snapshot, split, target, quy tắc tổng hợp và các anchor có thể chấm chung.
2. Tính lại correlation **chỉ trên train** cho bốn trạm; lưu số cặp hợp lệ, độ
   phủ, cách tính, lag và cửa sổ mưa. Không tái sử dụng thứ hạng đã tính trên toàn
   lịch sử như thể đó là thứ hạng không leakage.
3. Đánh giá mưa quá khứ với nước hoặc thay đổi nước tương lai theo horizon;
   mọi feature mưa phải kết thúc tại anchor. Kiểm tra cả tương quan giữa các trạm
   để mô tả dư thừa. Chọn trước quy tắc gộp điểm để có một thứ hạng; không chọn
   lag/thứ hạng riêng theo test hoặc theo đợt lỗi đang quan tâm.
4. Tạo schema S1/S2/S4 từ cùng nguồn; mỗi trạm giữ cùng bộ mưa hiện tại và mask.
   Bỏ toàn bộ feature/mask của trạm bị loại, không chỉ che giá trị bằng 0. Giữ
   feature nước và mùa như nhau; fit scaler riêng trên cùng train của từng nhóm.
5. Train mới cả ba nhóm, cùng seed, cấu hình LSTM/loss/penalty và ngân sách.
   Input size tự đổi theo schema; không ép còn 29 hoặc dùng checkpoint cũ để
   đại diện cho S4 train trên dataset/split mới.
6. Chọn cấu hình bằng validation và độ ổn định qua seed/đợt; báo cáo test chung
   sau khi chọn. Correlation cao là cách đề xuất nguồn, không phải bằng chứng
   một/hai trạm đủ cho dự báo hoặc bốn trạm là tối ưu.
7. So sánh MAE/RMSE/bias theo horizon, lỗi >200 mm, chuỗi thiếu/thừa, pha tăng/
   giảm và thời gian train/inference. Báo cả độ phủ; bỏ trạm có thể thay đổi số
   mẫu nhưng không được làm tập so sánh dễ hơn một cách âm thầm.

**Đầu ra:** bảng ranking và tương quan giữa trạm; manifest S1/S2/S4; bảng metric
trên cùng anchor; figure theo đợt; kết luận cấu hình quan trắc dùng làm đối chứng
ở bước 2. Nếu kết quả gần tương đương, báo mức chênh lệch và độ biến động, ưu
tiên cấu hình đơn giản khi đáp ứng tiêu chí đã chốt; không tuyên bố tối ưu tuyệt đối.

**[ĐỀ XUẤT]** Khởi đầu với top-1/top-2 thay vì quét mọi tập con. Nếu muốn thử
cặp trạm bổ sung thông tin thay vì top-2 correlation, đó là nhánh bổ sung chọn
trên train/val, có tên riêng. Chưa thay nguồn phía nam hoặc tăng số trạm trong
thí nghiệm đầu tiên.

### 1.4. Bước 2 — bổ sung feature dự báo thời tiết

**[ĐÃ THỐNG NHẤT]** Sau khi chốt cấu hình trạm mưa từ bước 1, thử bổ sung
lượng mưa dự báo mới nhất từ ECMWF đã có tại thời điểm dự báo nước. Giữ nguyên 5 output
để tách hiệu quả nguồn thông tin mới khỏi thay đổi số horizon.

Các việc phải làm:

1. **Giữ số trạm đã chọn ở bước 1.** Ví dụ bước 1 cho thấy dùng hai trạm tốt
   nhất thì các model ở bước 2 đều dùng đúng hai trạm đó. Train một model chỉ
   nhận nước và mưa quá khứ để làm mốc so sánh.
2. **Tải dự báo mưa lịch sử từ ECMWF.** Với từng ngày trong dataset, lấy các
   bản dự báo mưa cho khu vực liên quan. Kiểm tra ngày nào có/thiếu dữ liệu,
   bản dự báo được công bố lúc nào và áp dụng cho vị trí nào. Lưu cả dữ liệu
   lẫn thông tin nguồn để sau này biết chính xác model đã dùng bản nào.
3. **Thêm lượng mưa dự kiến trong các giờ tới vào dataset.** Ví dụ dự báo
   nước lúc 06:00 ngày 12/09: thêm lượng mưa ECMWF dự kiến từ 06:00 đến
   09:00, 12:00, 18:00 và 06:00 hôm sau, tương ứng 3/6/12/24 giờ tới.
   Chỉ dùng bản đã công bố trước hoặc đúng 06:00; không lấy mưa đo được sau
   06:00 rồi đưa vào như thể model đã biết trước.
4. **Train hai phiên bản để biết dự báo mưa có giúp ích không.** Phiên bản
   thứ nhất chỉ dùng nước/mưa quá khứ; phiên bản thứ hai thêm lượng mưa dự báo
   mới nhất đã có. Giữ cùng các khoảng train/val/test, cùng cách train, loss,
   penalty và 5 output. Chỉ thay đầu vào cần thử.
5. **Cho hai model dự báo đúng cùng các thời điểm rồi so sai số.** Kiểm tra
   từng mốc +1/+3/+6/+12/+24 giờ, lúc nước lên/rút và các đợt mưa lớn. Đếm
   cả lỗi >200 mm và các chuỗi lỗi kéo dài. Ghi rõ khi nguồn thời tiết bị thiếu
   và cách xử lý. Dùng tập validation để chọn phiên bản; tập test để đánh giá
   sau khi chọn, không liên tục sửa model để khớp các ví dụ test đã xem.
6. **Chọn phiên bản hiệu quả nhất để sang bước 3.** Nếu thêm dự báo mưa giúp
   ích thì giữ; nếu chưa giúp thì giữ model chỉ dùng quan trắc. Sau đó mới
   thử tăng từ 5 lên 24 output.

Tên dùng ở các mục kỹ thuật bên dưới: **A** là phiên bản chỉ có quan trắc,
**B** thêm lượng mưa dự báo mới nhất đã có tại thời điểm dự báo nước.
`Mask` là cờ đánh dấu thiếu dữ liệu; `anchor` là thời điểm đầu vào cuối cùng
của một lần dự báo nước.

**Đầu ra:** dữ liệu dự báo mưa đã lưu và kiểm tra; hai bộ đầu vào A/B;
hai phiên bản model; bảng sai số và biểu đồ; kết luận nên giữ đầu vào nào.
Các task tương ứng là FR-01 đến FR-09. Mô hình hiệu chỉnh mưa riêng là nhánh
tùy chọn ở mục 10, không phải điều kiện để hoàn thành bước 2.

### 1.5. Bước 3 — thử 24 output sau các thí nghiệm nguồn đầu vào

**[ĐÃ THỐNG NHẤT]** Cuối cùng thử dự báo trực tiếp **+1,+2,…,+24 giờ**, mỗi
output là mực nước trung bình giờ tại horizon tương ứng. Không nội suy 5 output
để tạo 24 nhãn. Dataset cần lấy thực tế cho từng giờ tương lai.

Các việc phải làm:

1. Chốt cấu hình nguồn/feature sau bước 2; nếu thời tiết không cải thiện thì giữ
   cấu hình chỉ quan trắc. Đóng băng cấu hình này ở cả hai nhóm O5 và O24.
2. Tạo dataset 24 target từ cùng snapshot; giữ nhãn thiếu, lưới giờ và provenance.
   Kiểm tra số mẫu hợp lệ cho đủ 24 target; 24 output có thể mất thêm anchor có
   giờ trung gian thiếu. So sánh O5/O24 trên cùng anchor và báo riêng độ phủ đầy đủ.
3. Giữ purge ít nhất 24 giờ vì horizon dài nhất không đổi; fit target scalers
   trên train, có thứ tự cột +1 đến +24 rõ ràng. Đổi output head, không giữ
   checkpoint/scaler 5 output cho model 24 output.
4. Cập nhật builder target, loader, CLI và trọng số loss để không cố định năm
   cột/horizon; chạy kiểm tra save/load và inverse transform đủ 24 giá trị.
5. Train O5/O24 với cùng backbone, input, seed, lịch tối ưu và quy tắc chọn
   checkpoint. Log số tham số mới do output head và số mẫu thực sự dùng.
6. Chốt trước cách phân bổ trọng số loss theo giờ. Khi dùng loss không đồng
   nhất, giữ tỷ lệ trọng số trên năm horizon chung và thử kiểm soát tổng trọng
   số dành cho chúng; log trọng số chuẩn hóa thực tế. Không mặc định thêm 19
   output sẽ làm +24 giờ được phạt nặng hơn: với MSE trung bình đều, tỷ trọng
   riêng của +24 giờ giảm từ 1/5 xuống 1/24. Mọi khác biệt phân bổ loss phải
   được ghi là một phần của thí nghiệm, không quy toàn bộ hiệu quả cho số output.
7. Đánh giá ngang hàng tại +1/+3/+6/+12/+24 giờ, đặc biệt +24; báo thêm đủ 24
   horizon, đường dự báo, đỉnh/pha nước, chuỗi lỗi và độ ổn định. Không so trực
   tiếp loss chuẩn hóa tổng của hai bài toán để kết luận model nào tốt hơn.
8. Chỉ khi O24 được chọn mới cập nhật cấu hình serving, metadata/API/frontend
   và test để biểu diễn đúng 24 mốc thật. Mỗi forecast ghi horizon list/schema;
   không làm kết quả cũ 5 mốc trông như kết quả mới 24 mốc.

**Đầu ra:** dataset/manifest O5 và O24; cấu hình trọng số; checkpoint/scaler
riêng; bảng so sánh năm horizon chung và bảng đủ 24 giờ; figure cùng anchor;
kết luận có cải thiện +24 giờ và có đáng đổi model serving hay không.

Thử 24 output không bảo đảm cải thiện +24 giờ. Mục tiêu là đo hiệu quả của
giám sát thêm các giờ trung gian, tách khỏi các thay đổi nguồn mưa trước đó.

## 2. Hiện trạng và bằng chứng đã có

### 2.1. Các thành phần hiện tại

| Thành phần | Vị trí | Liên quan công việc mới |
|---|---|---|
| Tạo dataset Dã Viên | `research/build_training_dataset_1h_top4.py` | Mở rộng qua phiên bản dataset riêng |
| Refresh PostgreSQL / chia thời gian | `research/refresh_da_vien_dataset.py`, `research/resplit_hourly_dataset.py` | Tái sử dụng cách snapshot, manifest, purge |
| Dataset và chọn feature | `src/ai/dataset.py`, `src/ai/utils.py` | Thêm nhóm feature, cờ thiếu, thứ tự cột |
| Train và loss | `src/ai/train.py`, `src/ai/trainers.py`, `src/ai/losses.py` | Log cấu hình và chạy ablation |
| Mô hình | `src/ai/lstm.py` | Input mở rộng cần checkpoint mới |
| Tạo đầu vào backend | `src/backend/prediction/features.py`, `inputs.py`, `preprocessing.py` | Phải khớp preprocessing training |
| Registry và lưu dự báo | `src/backend/prediction/config.py`, `model_configs.json`, `persistence.py` | Nhận diện schema và bản dự báo mưa đã dùng |
| PostgreSQL / ingestion | `src/backend/db/`, `src/backend/ingestion/` | Thêm dữ liệu thời tiết riêng với quan trắc |

Registry đang phục vụ Dã Viên với 29 feature, 96 timestep, 5 output và 4 trạm
mưa: Hồng Thái `750062`, Nhâm `750007`, thị trấn A Lưới `750003`, Hồ A Lá `750011`.
Các checkpoint nghiên cứu mới không mặc nhiên là model đang phục vụ ứng dụng.

### 2.2. Đối chiếu ECMWF đã thực hiện

**[ĐÃ KHẢO SÁT]** API trả dữ liệu tại Huế. Tại tọa độ bốn trạm trên, cả bốn được
ánh xạ đến cùng ô lưới `16.274164, 107.26496`; dự báo giống nhau.

| Cửa sổ được dự báo, giờ Việt Nam | ECMWF tổng 24 giờ | Tổng số đo nguồn ở từng trạm |
|---|---:|---:|
| 12/09/2026 06:00 → 13/09/2026 06:00 | 109,9 mm | 403,8–543,2 mm |
| 13/09/2026 06:00 → 14/09/2026 06:00 | 11,2 mm | 69,2–123,8 mm |

Nguồn: [báo cáo](../reports/openmeteo_september_rain_check/phan_tich.md),
[bảng số liệu](../reports/openmeteo_september_rain_check/summary.csv),
[figure](../reports/openmeteo_september_rain_check/ecmwf_vs_observed.png).
Các file `reports/` là artifact local, có thể không có khi clone repository.

Hai cửa sổ cho thấy dự báo thấp hơn quan trắc và có phân biệt pha mưa giảm.
Các bản chạy khảo sát được chọn sớm hơn anchor, chưa chứng minh là bản mới nhất
đã có tại anchor hoặc dữ liệu đã tồn tại trên hệ thống cung cấp vào đúng lúc đó.

**[CẦN XÁC MINH]** Đơn vị mưa mm đã được dự án chấp nhận; khoảng tích lũy của
mỗi bản ghi quan trắc nguồn chưa xác định. Các tổng quan trắc trên dùng phép cộng
của pipeline hiện tại. Đối chiếu Nhâm với bản tin là bằng chứng hỗ trợ, chưa thay
thế tài liệu mô tả sensor; không tự đổi cách cộng hoặc coi đó là cường suất mm/h.

## 3. Luồng dữ liệu dự kiến

```mermaid
flowchart TD
    S[Train đối chứng 4 trạm, top 1 và top 2 trạm mưa] --> T[Chọn cấu hình quan trắc bằng validation]
    T --> A[Cấu hình vị trí, mô hình thời tiết, khoảng thời gian]
    A --> B[Tải từng lần chạy dự báo mưa]
    B --> C[Lưu response gốc và metadata nguồn]
    C --> D[Kiểm tra giờ, đơn vị, độ phủ và ô lưới]
    D --> E[Lưu PostgreSQL theo lần chạy và giờ được dự báo]
    O[Quan trắc nước và mưa trong PostgreSQL] --> F[Chọn dữ liệu đã có tại từng mốc dự báo]
    E --> F
    F --> G[Tạo feature từ bản dự báo mưa mới nhất đã có]
    G --> H[Dataset có phiên bản và kiểm tra leakage]
    H --> I[Train và so sánh mô hình trên cùng các mẫu]
    I --> U[Chốt feature và đối chiếu 5 với 24 output]
    U --> J{Đạt tiêu chí đánh giá?}
    J -->|Có| K[Đóng gói model, scaler, schema và chính sách nguồn]
    J -->|Chưa| L[Phân tích theo đợt và độ tin cậy nguồn]
    L --> G
    K --> M[Backend dùng cùng feature builder]
    E --> M
    M --> N[Dự báo nước kèm định danh nguồn mưa]
```

## 4. Hợp đồng thời gian: phần bắt buộc làm đúng

### 4.1. Các thời điểm phải phân biệt

| Trường | Ý nghĩa |
|---|---|
| `anchor_at` | Mốc quan trắc cuối của mẫu dự báo nước |
| `run_initialized_at` | Thời điểm khởi tạo mô hình thời tiết |
| `available_at` | Thời điểm bản dự báo có thể được sử dụng, kèm bằng chứng hoặc giả định |
| `first_seen_at` | Lần đầu collector thực sự nhìn thấy response; không sửa lùi khi tải lại |
| `retrieved_at` | Thời điểm tải bản lưu cụ thể về hệ thống |
| `valid_at` | Giờ tương lai mà giá trị dự báo mô tả |
| `availability_basis` | `observed_live`, `provider_documented`, hoặc `assumed` |

Khởi tạo mô hình không đồng nghĩa bản dự báo đã công bố. Tài liệu nguồn nêu độ
trễ thường 4–6 giờ cho mô hình toàn cầu; cần kiểm tra lịch của model cụ thể.
Xem [Single Runs API](https://open-meteo.com/en/docs/single-runs-api) và
[Model Updates](https://open-meteo.com/en/docs/model-updates).

### 4.2. Quy tắc chọn bản dự báo

1. Với mỗi anchor `a`, chỉ xét bản có `available_at <= a`; chọn bản mới nhất
   trong các bản hợp lệ theo chính sách có phiên bản.
2. Các giá trị dự báo có `valid_at > a` được phép làm feature nếu bản dự báo
   đã có trước `a`. Quan trắc sau `a` chỉ dùng làm target hoặc đánh giá.
3. Với archive tải hôm nay, `retrieved_at` không phải giờ công bố trong quá khứ.
   Nếu chưa có bằng chứng, dùng giả định trễ bảo thủ và đánh dấu kết quả là mô
   phỏng lịch sử, không tuyên bố replay chính xác hoạt động thực tế.
4. Kiểm tra dữ liệu archive là dự báo vận hành gốc hay hindcast/tái chạy. Tài
   liệu có nhắc hindcast cho một phần lịch sử ECMWF; ngày có dữ liệu không tự
   chứng minh bản đó đã được công bố vào ngày ấy. Tách hai loại trong manifest.
5. Mỗi giá trị mưa giờ phải có khoảng tích lũy rõ, quy về `(t−1h,t]` sau khi
   xác nhận quy ước API. Không cộng giá trị mưa đầu tiên null thành 0.
6. Bản live bị công bố trễ phải đợi tải được mới có thể dùng. Chỉ dùng giờ
   khởi tạo cộng độ trễ cố định khi mô phỏng, không để thay cho thực tế live.

Ví dụ: dự báo nước tại 06:00 ngày 12/09 cho đến 06:00 ngày 13/09 cần mưa trong
`(12/09 06:00, 13/09 06:00]` của bản đã có trước anchor. Không được chọn bản
khởi tạo 07:00 ngày 12/09 hoặc bản 01:00 nhưng đến 07:00 mới có.

## 5. Thu thập và lưu dữ liệu

### 5.1. Khảo sát trước khi tải toàn bộ

- Xác nhận mã model cụ thể, chu kỳ chạy, độ trễ, đơn vị, bước thời gian gốc và
  nội suy. Tránh `best_match` tự thay model giữa train và inference.
- Thử mẫu đầu/cuối khoảng 11/2025 đến cutoff mới, cùng các đợt 09–10/2026;
  phân loại ngày thiếu, hindcast, đổi phiên bản model và response lỗi.
- Tọa độ lấy từ catalog có nguồn. Lưu riêng tọa độ yêu cầu và tọa độ ô lưới trả về.
- Bắt đầu từ ô lưới tương ứng bốn trạm hiện tại. Các vị trí phía nam, trung tâm,
  phía bắc hoặc vùng khác là nhánh thí nghiệm riêng; chọn theo địa hình/lưu vực,
  độ phủ và train, không chọn vì riêng một đợt test nổi bật.
- Kiểm tra chia sẻ ô lưới trước khi nhân số lượt tải; bốn tọa độ trùng ô không
  phải bốn nguồn thông tin độc lập. Không cộng chúng thành tổng mưa khu vực.

### 5.2. Schema PostgreSQL đề xuất

Tên dưới đây là thiết kế, chưa có migration hoặc bảng thực tế.

| Bảng | Dữ liệu chính |
|---|---|
| `weather_locations` | Vị trí yêu cầu, vùng, ô lưới trả về, phiên bản mapping |
| `weather_forecast_runs` | Provider/model/cycle, khởi tạo, khả dụng, loại archive, trạng thái đầy đủ |
| `weather_forecast_hourly` | Run, location/grid, variable, valid_at, value, unit, QC |
| `weather_fetches` | Tham số không chứa secret, lần tải, HTTP status, checksum, đường dẫn response gốc |

Yêu cầu: UTC có timezone; unique key theo provider/model/run/location/variable/
valid_at, có phiên bản nếu provider sửa response; index phục vụ chọn run tại
anchor và truy vấn khoảng valid time. Lưu provenance ngay cả khi hai tọa độ trùng
ô; deduplicate giá trị khi đã xác nhận cùng model, ô, cách chọn và nội suy.

Không ghi dự báo mưa vào bảng quan trắc. Tải lại không tạo hàng trùng, không ghi
đè mất bản dự báo cũ; nội dung thay đổi phải giữ checksum/phiên bản để replay.
Một run không đầy đủ phải được đánh dấu, không để người dùng đọc nhầm là đầy đủ.

### 5.3. Collector có thể chạy lại

- Cấu hình theo vị trí/model/date range, không hardcode Dã Viên vào client API.
- Backfill chia lô có checkpoint/resume; cache request đã tải và kiểm tra checksum.
- Live poll theo lịch cập nhật, tải một lần rồi tái sử dụng từ DB cho nhiều dự báo nước.
- Retry có giới hạn, backoff cho timeout/429/5xx; lỗi tham số không retry vô hạn.
- Tính quota theo số model/vị trí/biến và khoảng yêu cầu, không chỉ số HTTP request.
- Lưu raw response, provenance và bảng chuẩn hóa; không phụ thuộc lâu dài vào `/tmp`.
- Kiểm tra model mới công bố đủ dữ liệu; giữ bản hợp lệ cũ trong giới hạn tuổi cho phép.

Gói miễn phí hiện dành cho phi thương mại và có giới hạn theo phút/giờ/ngày/tháng;
Single Runs có trong free, còn quyền thương mại cho nguồn này cần gói phù hợp.
Kiểm tra lại [bảng giá và cách tính lượt](https://open-meteo.com/en/pricing)
trước khi backfill hoặc mở rộng triển khai. Ghi nguồn dữ liệu theo điều kiện attribution.

## 6. Feature cần xây dựng

### 6.1. Lượng mưa từ bản dự báo mới nhất đã có

Với ô/vùng `g`, anchor `a` và bản chạy hợp lệ `r`:

```text
forecast_rain_sum_Hh(a, g) = tổng P(r, g, t), với a < t <= a + H giờ
H thuộc {3, 6, 12, 24}
```

Mỗi tổng cần đủ H giá trị giờ hợp lệ. Có thể thử thêm max một giờ trong 24 giờ
tới, giờ bắt đầu mưa mạnh, tuổi bản dự báo và số giờ còn thiếu. Ngưỡng “mưa mạnh”
phải cấu hình/chọn trên train và ghi đơn vị, không chọn từ sự kiện test.

### 6.2. Thiếu dữ liệu và chuẩn hóa

- `0` là dự báo không mưa hợp lệ; thiếu bản/run/giờ phải là null kèm mask.
- Thiếu một phần cửa sổ không được tổng thành lượng mưa 24 giờ đầy đủ.
- Không dùng quan trắc tương lai điền vào chỗ dự báo thiếu; không bỏ toàn bộ
  hàng vì một nguồn mưa phụ bị thiếu.
- Scaler/imputer và ngưỡng feature chỉ fit trên train. Thứ tự cột và preprocessing
  được đóng gói cùng checkpoint; không fit lại lúc chạy app.

## 7. Tạo dataset và cập nhật mã AI

### 7.1. Cách tích hợp tối thiểu đề xuất

Thêm một nhóm feature riêng, ví dụ `forecast_rain`, vào từng hàng thời gian `t`.
Hàng `t` chỉ chứa bản dự báo đã có tại chính `t`. LSTM nhận chuỗi 96 hàng mở rộng;
hàng cuối chứa dự báo mưa tương lai so với anchor hiện tại.

Không đưa một bản tải tại anchor vào tất cả 96 hàng quá khứ rồi mô tả các hàng
đó là dữ liệu đã biết tại từng thời điểm. Nếu sau này dùng nhánh nhận vector mưa
tương lai riêng tại anchor thì phải thiết kế dataset/forward riêng và so sánh như
một thay đổi kiến trúc độc lập.

### 7.2. Việc cần thực hiện

1. Snapshot quan trắc, metadata trạm và forecast archive có cutoff cố định.
2. Dùng một feature builder chung cho nghiên cứu và backend; kiểm tra khả dụng
   trước khi tính tổng mưa dự báo, lưu run ID đã chọn cho mỗi hàng.
3. Xuất dataset mới; giữ nguyên dataset/checkpoint cũ và target trung bình nước giờ.
4. Giữ mốc train/val/test đã thống nhất để chạy đối chứng, purge ít nhất horizon
   dài nhất. Mốc trước Aug/val Aug–giữa Sep/test sau giữa Sep chỉ là cấu hình hiện
   có; tính ngày chính xác từ manifest, không diễn giải ngày đã purge là mẫu val.
5. Đóng băng các mẫu chung có thể chấm giữa các nhóm để so sánh công bằng; báo
   riêng độ phủ toàn bộ và số mẫu bị mất vì forecast thiếu.
6. Cập nhật chọn nhóm feature, input size, cờ thiếu, schema và CLI trong `src/ai/`.
   Không chỉ thêm cột CSV rồi giả định loader đang chọn chúng.
7. Log phiên bản dataset, cutoff, model thời tiết, chính sách chọn bản dự báo đã có,
   scaler, feature order, split, seed, checkpoint và mã nguồn.
8. Lưu model/scaler vào thư mục theo experiment/run ID; tránh đường dẫn training
   hiện tại ghi đè checkpoint giữa các thử nghiệm. Kiểm tra reload artifact thành công.

Checkpoint 29 feature cũ không nhận trực tiếp input mới. Mô hình mở rộng phải
train lại, có scaler/schema riêng, và kiểm chứng trước khi đưa vào registry.

## 8. Thiết kế thử nghiệm và đánh giá

### 8.1. Các nhóm so sánh

| Nhóm | Mục đích |
|---|---|
| Persistence | Baseline: giữ nguyên nước tại anchor |
| Model đang phục vụ | Mốc tham chiếu thực tế, provenance riêng |
| A: train lại chỉ với quan trắc | Dùng cấu hình chọn từ S1/S2/S4, cùng dataset/split/seed mới |
| B: A + tổng mưa dự báo | Kiểm tra giá trị thông tin tương lai |
| Mở rộng B bằng vùng/ô lưới khác | Tùy chọn; kiểm tra đại diện không gian |

Giữ kiến trúc cơ bản, loss, correction penalty, số output và ngân sách train
giống nhau trong A/B; chạy nhiều seed sau smoke test. Model cũ có thể khác
train/split nên không thay thế nhóm A trong việc quy cải thiện cho feature mới.

Sau khi chốt A/B và các nhánh tùy chọn, mới chạy O5/O24 theo mục 1.5.
Việc chọn số output không được trộn vào việc kết luận dự báo mưa có ích.

Các tập test đã xem nhiều lần chỉ dùng phân tích khám phá. Cần giữ một khoảng
tương lai chưa dùng chọn model và/hoặc đánh giá rolling theo thời gian, purge
đúng horizon. Đợt 11–14/09 hiện thuộc khoảng val của dataset đã trao đổi; không
gọi việc fit tốt đợt đó là bằng chứng trên test độc lập.

### 8.2. Chỉ số cần báo cáo

- Mỗi horizon: MAE, RMSE, bias, median, phân vị 90/95/99, số và tỷ lệ >200 mm.
- Thiếu/thừa: số mốc, độ lớn, chiều dài chuỗi lỗi; ngắt chuỗi khi mất giờ hoặc
  không vượt ngưỡng; tính riêng từng horizon.
- Theo pha: nước lên, nước rút, mưa ít/nhiều, lượng mưa dự báo cao/thấp.
- Theo đợt: độ cao/độ lệch giờ đỉnh, onset sai số; figure có bối cảnh trước/sau.
- Giá trị forecast: có báo trước được mưa tăng không, với lead time bao nhiêu;
  sai lệch ô lưới–trạm và tính ổn định qua các đợt/mùa/model cycle.
- Độ phủ, số lần fallback, tuổi forecast, số hàng thiếu; không để mất mẫu khó
  tạo ảo giác metric tốt hơn.

Mốc ngưỡng và phân nhóm chọn trên train hoặc được định nghĩa trước đánh giá.
Các giờ liên tiếp có tương quan; khi ước lượng độ không chắc chắn cần xét theo
block thời gian/đợt, không coi mỗi giờ là một sự kiện độc lập.

### 8.3. Tiêu chí trước khi đưa vào app

**[ĐỀ XUẤT]** Chốt trước thí nghiệm mức cải thiện yêu cầu và mức suy giảm cho phép:
ưu tiên +12/+24 giờ, không đánh đổi quá mức +1/+3 giờ hoặc giai đoạn bình thường.
Không chọn chỉ vì một figure đẹp. Cần vượt/so sánh rõ A và persistence trên cùng
mẫu, có kiểm tra nhiều đợt, reload artifact được và online/offline feature khớp.
Nếu chưa cải thiện, giữ kết quả nghiên cứu và chưa đổi model phục vụ.

## 9. Tích hợp backend và cập nhật sau này

- Worker thời tiết tải theo lịch model, ghi DB; request dự báo nước đọc dữ liệu
  đã lưu, không phụ thuộc vào một HTTP call thời tiết cho mỗi lượt người dùng.
- Registry mới có vị trí/ô lưới/model, feature schema, policy chọn run, giới hạn
  tuổi forecast và quy tắc thiếu nguồn. Không hardcode 4 trạm = 4 ô khác nhau.
- Online và historical replay dùng chung as-of selector và feature builder.
  Mỗi hàng chuỗi tuân thủ cutoff của chính nó như lúc tạo dataset.
- Cache/input hash phải gồm forecast values, masks, run IDs, response/version
  và policy; bản forecast mới không được lấy lại kết quả của bản cũ.
- Lưu lineage của dự báo nước: checkpoint/scaler/schema, anchor, weather run,
  valid window, thời điểm khả dụng, trạng thái forecast và input hash.
- Thiếu/già forecast: ưu tiên bản cũ còn hạn; nếu không có, dùng cơ chế missing
  đã train và kiểm thử, hoặc model chỉ quan trắc đã đăng ký riêng. Phải ghi model
  thực sự dùng; không âm thầm thay toàn bộ mưa dự báo bằng 0.
- Bản giả lập người dùng nhập để kiểm tra độ nhạy phải có nhãn scenario; không
  lưu thành dự báo thời tiết thật hoặc trộn vào quan trắc/training.
- Giao diện chỉ cần cho biết nguồn, thời điểm cập nhật và tình trạng dữ liệu
  nếu ảnh hưởng việc đọc kết quả; thiết kế UI là công việc sau pipeline và model.

## 10. Nếu cần mô hình hiệu chỉnh dự báo mưa riêng

Chỉ thực hiện sau khi đo sai lệch có tính hệ thống và có đủ dữ liệu theo mùa/đợt.
Đầu vào là forecast có sẵn, vị trí, lead time và quan trắc trước anchor; target
là mưa thực tế tương lai với khoảng tích lũy đã xác minh.

- Bắt đầu từ hiệu chỉnh đơn giản; so với ECMWF gốc trước khi dùng model phức tạp.
- Fit trên train; để train model nước cần dự báo mưa hiệu chỉnh ngoài mẫu theo
  rolling/forward chaining. Không dùng dự đoán fitted trên cùng target mưa làm
  đầu vào rồi báo hiệu quả như ngoài mẫu.
- Tính cả chi phí, độ trễ, failure mode và drift của hai mô hình nối tiếp.
- Mô hình nước vẫn nhận forecast gốc/mask/độ tin cậy nếu thí nghiệm chứng minh hữu ích.

## 11. Danh sách task và đầu ra nghiệm thu

Tất cả task sau là việc cần làm; khảo sát nhỏ đã có không đánh dấu hoàn thành cả task.

| ID | Công việc | Đầu ra để kiểm tra |
|---|---|---|
| RS-01 | Xếp hạng lại 4 trạm trên train | Ranking, coverage, lag/window, tương quan giữa trạm |
| RS-02 | Dataset và train S1/S2/S4 | Schema/scaler riêng, cùng split/anchor/config/seed |
| RS-03 | Đánh giá và chốt số trạm | Metric theo horizon/đợt, baseline quan trắc cho bước thời tiết |
| FR-01 | Audit nguồn và khả dụng lịch sử | Báo cáo coverage, operational/hindcast, units, cadence, độ trễ |
| FR-02 | Catalog vị trí và ô lưới | Cấu hình có nguồn tọa độ, mapping và phiên bản |
| FR-03 | Client API + lưu raw + quota/resume | Tải lại không trùng, retry có giới hạn, manifest/checksum |
| FR-04 | Migration và repository dữ liệu thời tiết | Bảng riêng, unique/index, lineage, trạng thái run |
| FR-05 | As-of selector | Chỉ chọn bản đã có tại cutoff, theo policy có test |
| FR-06 | Builder feature tổng mưa dự báo | Cùng cửa sổ, mask đúng, dùng chung online/offline |
| FR-07 | Dataset mới và audit leakage | Train/val/test, feature schema, scalers, manifest, audit |
| FR-08 | Cập nhật mã AI và artifact | Loader chọn đúng feature; train/reload với checkpoint riêng |
| FR-09 | Ablation A/B và báo cáo | Bảng theo horizon/đợt, persistence, chuỗi lỗi, figures |
| OH-01 | Tạo target +1 đến +24 giờ | Dataset và audit nhãn/độ phủ/split/anchor chung |
| OH-02 | Train và so sánh O5/O24 | Loss có trọng số rõ, artifact riêng, metric tại năm horizon chung và đủ 24 |
| OH-03 | Chốt số output và hợp đồng serving | Kết luận +24 giờ, schema/horizon list mới nếu được chọn |
| FR-10 | Worker live + backend + cache | Replay có provenance, fallback rõ, parity với dataset |
| FR-11 | Chạy thử song song, đánh giá khoảng mới | So sánh model cũ/mới trước khi đổi registry phục vụ |
| FR-12 | Hiệu chỉnh mưa riêng nếu cần | Baseline và dự đoán ngoài mẫu cho model nước |

Thứ tự thí nghiệm: **RS-01/02/03 → FR-01…09 → OH-01/02/03**.
Sau khi chốt model cuối mới làm FR-10/11 để đưa vào app. Nếu cần triển khai bản
5 output trước, đó là mốc triển khai riêng, không thay đổi thứ tự nghiên cứu.
FR-12 là nhánh tùy chọn sau khi đánh giá nguồn thời tiết; chưa nằm trong ba bước
bắt buộc. Tải/lưu archive sớm không đồng nghĩa đã bắt đầu train feature thời tiết;
ưu tiên hiện tại là hoàn thành RS-01…03 trước khi mở rộng feature/model.

### 11.1. Kiểm thử bắt buộc khi triển khai

- Timestamp UTC/VN, giờ kết thúc và 24 giá trị tạo tổng 24 giờ.
- Run khởi tạo trước nhưng công bố sau cutoff phải bị loại; null không thành 0.
- Bốn vị trí cùng ô không nhân tổng hoặc đếm thành bốn dự báo độc lập.
- Pagination/lô tải, retry, resume, idempotency, nội dung provider sửa đổi.
- Không leak target, scaler, nguồn chọn trên val/test hoặc hindcast không rõ nguồn.
- Loader lấy đúng thứ tự/input size; save/load model/scaler và inference hoạt động.
- Thay tập trạm phải bỏ đúng mọi feature/mask của trạm bị loại; S1/S2/S4 chấm
  cùng anchor. Ranking/lag không dùng val/test để fit.
- Với O24, đủ 24 target đúng timestamp, mask thiếu đúng, trọng số đủ chiều,
  scaler/forward/inverse transform và metadata output khớp; đối chiếu O5 cùng mẫu.
- Feature offline so với backend trên cùng anchor phải khớp cả value/mask/run ID.
- Forecast mới làm thay input hash; forecast thiếu/già đi đúng fallback đã cấu hình.

### 11.2. Cấu trúc file đề xuất, chưa được tạo

```text
src/backend/weather/                # Client, selector, feature builder dùng chung
src/backend/ingestion/weather_import.py
src/backend/db/alembic/versions/...  # Migration mới sau khi duyệt schema thực tế
research/build_forecast_rain_dataset.py
research/evaluate_forecast_rain.py
reports/forecast_rain/<run_id>/      # Coverage, audit, manifest, metrics, figures
train_data/<dataset_version>/       # Snapshot và split có phiên bản
model/<experiment>/<run_id>/        # Checkpoint, scaler, feature/policy metadata
```

Mỗi pha hoàn thành phải cập nhật tài liệu với trạng thái thực tế, đường dẫn
artifact và cách chạy lại. Không ghi “đã tích hợp” chỉ vì tải được JSON.

## 12. Những điều cần nhớ khi thực hiện

1. Mục tiêu chính là kiểm tra dự báo mưa có giúp dự báo nước tốt hơn không; độ
   chính xác tuyệt đối của dự báo mưa là một phần đánh giá.
2. Mỗi mẫu dùng bản dự báo mưa mới nhất đã có tại thời điểm dự báo nước.
3. Dự báo tương lai hợp lệ khác với quan trắc tương lai bị lọt vào feature.
4. Không coi mọi archive là bản dự báo đã công bố thực tế; giữ provenance và giới hạn.
5. Giữ mưa quan trắc và mưa dự báo tách biệt; giữ model/scaler/dataset cũ để đối chứng.
6. Thứ tự: số trạm trước, dự báo mưa sau, 24 output cuối. Không đổi loss
   hoặc correction penalty đồng thời với chọn trạm hay thêm feature; khi thử
   24 output phải công bố rõ phân bổ loss và giới hạn so sánh.
7. Mưa, hồ và triều có thể cùng liên quan; thêm forecast không tự giải hết lỗi.
8. Các số đã phân tích đến 15/10 không mặc nhiên bao gồm dữ liệu mới đến 18/10.
   Muốn cập nhật phải tạo cutoff/snapshot mới và ghi lại phạm vi đánh giá.

## 13. Tài liệu tham chiếu

- [Kiến trúc ứng dụng](ARCHITECTURE.md)
- [Hợp đồng đầu vào dự báo hiện tại](backend/prediction/README.md)
- [Phân tích chuỗi lỗi, mưa và vận hành hồ](../reports/large_error_events_2026-10-15/phan_tich_mua_lu_va_sai_so.md)
- [Đối chiếu ECMWF đã tải](../reports/openmeteo_september_rain_check/phan_tich.md)
- [Single Runs API](https://open-meteo.com/en/docs/single-runs-api)
- [Lịch cập nhật model](https://open-meteo.com/en/docs/model-updates)
- [Giá và điều kiện sử dụng](https://open-meteo.com/en/pricing)

Tài liệu API đã được đọc lại ngày 18/10/2026. Thông tin nguồn có thể thay đổi;
phải kiểm tra lại khi triển khai hoặc đổi model/cycle.
