# Tự so sánh model vừa train với model trong app

## Cách chạy

Giữ nguyên lệnh `python -m src.ai.train ...`. Với model sequence, sau khi train,
khôi phục checkpoint tốt nhất và chấm test, chương trình tự chạy so sánh.
Không cần thêm lệnh hoặc bật tính năng. Mặc định so với trạm Dã Viên
`0074647403` trong `src/backend/prediction/model_configs.json`.

Nếu cần trạm/registry khác, có thể truyền `--comparison-station-id` và
`--comparison-registry`. Registry phải đúng với trạm nước của dataset. Hiện phần
so sánh đọc dữ liệu giờ và model Residual LSTM đang được cấu hình trong app;
không gọi PostgreSQL hoặc API thời tiết.

## Bảng kết quả

Sau mỗi lần train, terminal hiện bảng theo +1/+3/+6/+12/+24 giờ, hoặc các horizon
chung nếu số output hai model khác nhau:

- RMSE của model app và model mới, đơn vị mm.
- Số dự báo có sai số tuyệt đối **>200 mm**, tỷ lệ trên số dự báo hợp lệ.
- MAE và số mẫu của mỗi model để đối chiếu.

Báo cáo tổng hợp có **trung bình RMSE của các horizon** và **RMSE gộp toàn bộ
dự báo**. Hai số này khác nhau: số đầu là trung bình cộng các RMSE, số sau là
căn của trung bình tất cả bình phương sai số. Tỷ lệ lỗi lớn tổng thể có mẫu số
là số anchor chung × số horizon chung, không phải số sự kiện mưa/lũ độc lập.

Model mới có thể tốt hơn ở một horizon, kém hơn ở horizon khác. Chương trình
chỉ đưa nhận xét mô tả; bạn quyết định giữ model. Nó không thay registry hay model app.

## Dữ liệu dùng để so sánh

- Cùng `test.csv` của dataset đang train, cùng actual chưa chuẩn hóa.
- Mỗi model dùng đúng các feature, thứ tự cột, scaler đã fit và độ dài chuỗi
  riêng. Không fit lại scaler của model app trên dữ liệu mới.
- Chỉ lấy giao các thời điểm đủ điều kiện của cả hai model: qua warmup, có đủ
  target và có nước tại anchor. Thiếu predictor vẫn được xử lý bằng mask/quy
  tắc chuẩn hóa hiện có, không loại cả hàng vì một feature phụ bị thiếu.
- Không nối dự báo của các thời điểm khác nhau để so metric. Các giờ thiếu làm
  đứt lưới thời gian phải được biểu diễn thành hàng thiếu, không dịch sequence.
- So sánh mặc định trên phần test của file, chưa có lựa chọn khoảng ngày riêng.
  Bảng coverage ghi số hàng/mẫu bị loại và khoảng anchor thực tế.

Đây là đánh giá trên cùng mẫu, không chứng minh riêng việc giảm số trạm gây cải
thiện, vì model app có thể khác dataset train/loss. Để kết luận số trạm, vẫn cần
train các nhóm 1/2/4 trên cùng cấu hình. Test đã xem nhiều lần là kết quả khám phá.

## File được lưu

```text
reports/model_comparisons/<mlflow_run_id>_<unique_suffix>/
  comparison.md                     # Bảng tiếng Việt và giải thích chỉ số
  metrics.csv                       # Chỉ số của từng model, từng horizon
  summary.json                      # Các chỉ số tổng hợp
  paired_predictions.csv            # Actual và dự báo cùng anchor/horizon
  manifest.json                     # Model version, checksum, coverage, feature order
  candidate.pt                      # Bản sao checkpoint được đánh giá
  candidate.preprocessing.joblib    # Scaler và schema đi cùng checkpoint
```

Mỗi lần đánh giá có thư mục riêng. Bản sao candidate giúp giữ lại model đã đánh
giá dù lần train cùng nhóm sau đó ghi đè đường dẫn checkpoint ban đầu.
Không upload model lên MLflow. MLflow chỉ ghi settings/metrics, trạng thái so
sánh `comparison/status` và đường dẫn báo cáo `comparison/report_dir`.

Model app được đọc theo registry local tại lúc so sánh. Model version và checksum
định danh artifact thực sự đã dùng; không kiểm tra tiến trình API đang giữ model
nào trong bộ nhớ. Không sửa checkpoint/scaler/registry phục vụ ứng dụng.

Nếu lỗi ở phần so sánh, train đã hoàn tất vẫn được giữ. Terminal báo lỗi, tag
MLflow là `failed` cho bước so sánh và thư mục chứa `comparison_error.json`.
Không diễn giải thiếu báo cáo là model tốt hơn hoặc tự chọn mẫu khác để tiếp tục.

## Chạy riêng cho model đã train trước khi có pipeline

```bash
.venv/bin/python -m src.ai.comparison \
  --checkpoint model/da_vien_1h_2026-10-15/rain_750007/1h/ResidualLSTM_water_level+rain+flood_season_best.pt \
  --dataset-dir train_data/da_vien_1h_2026-10-15 \
  --output-dir reports/model_comparisons/manual_trial_01
```

Thư mục output phải chưa tồn tại; đổi tên cho lần sau. Checkpoint cần có file
`.preprocessing.joblib` cạnh nó. Chế độ chạy riêng hỗ trợ checkpoint LSTM và
Residual LSTM, tái dựng cấu trúc từ trọng số, chạy eval và không train lại.
Chạy riêng không sửa run MLflow cũ.

Đã kiểm tra với checkpoint một trạm của run `a5fbc616204547ff9258500d9c306011`:
592 anchor chung, 2.960 dự báo trên năm horizon. Báo cáo local:
[so sánh model một trạm](../reports/model_comparisons/recovered_a5fbc616_2026-10-18/comparison.md).

## Độ ổn định theo thời gian của các tổ hợp trạm mưa — 21/10/2026

**Kết luận:** chưa có tổ hợp thay thế B tốt hơn một cách toàn diện. Trong 17 tổ hợp
có dự báo từng mẫu lưu tại đây, HT+BM+PB giảm N200 gộp +12h/+24h từ 43 xuống 40
nhưng RMSE gộp tăng 90,00 → 91,39 mm. Lợi ích +12h tập trung ở 13–14/10;
NH giảm N200 +24h từ 27 xuống 23, tập trung ở 07–08/10. Chưa phải bằng chứng
lợi ích ổn định qua nhiều đợt mưa độc lập hoặc ảnh hưởng nhân quả của một trạm.

### Phạm vi và cách chấm

- Dùng 17 `paired_predictions.csv` của các tổ hợp có đủ năm horizons và 592 anchor.
  Không tính bốn model đổi cadence/đổi target; không suy diễn các run chỉ còn metric
  tổng hợp là đã có dự báo từng mẫu. Đây là phân tích các snapshot báo cáo hiện có,
  không phải thống kê mọi lần train lịch sử.
- Nạp Model B từ `model/experiments/model_b_retrain_20261019/…/1h/` và inference CPU
  bằng preprocessing đi kèm, trên `train_data/da_vien_1h_2026-10-15_forecast_mask_o24/test.csv`.
  Không train/refit, không đọc prediction production làm baseline B.
- Ghép một-một theo `(anchor_utc, horizon_minutes)`; kiểm tra nhãn thực tế khớp
  và mỗi candidate có đúng 2.960 dự báo trên cùng các khóa với B. B tái hiện 48
  lỗi cả năm horizons, 16 lỗi +12h và 27 lỗi +24h. Sai số dùng đơn vị mm, N200 là `abs(error) > 200`.
- Candidate dùng dự báo và sai số đã lưu trong báo cáo; không nạp checkpoint đường dẫn
  train có thể đã bị ghi đè. Ví dụ HT+BM+PB ở đây là snapshot run 50 epochs,
  chọn epoch 33, không phải lần train 27 epochs sau đó. Đối chiếu MLflow cho 17
  snapshot: cùng seed 42, giới hạn 50 epochs. Sai khác lưu float32 trong CSV
  dưới 0,0001 mm không làm thay đổi phân loại ngưỡng: mẫu gần ngưỡng nhất cách
  200 mm khoảng 0,0128 mm ở candidate và 0,1266 mm ở B.
- Anchor: 19/09/2026 23:00–14/10/2026 14:00 UTC+7. Phần tập trung +12h/+24h
  có 1.184 dự báo, 592 mỗi horizon. Chia khoảng theo **thời điểm được dự đoán**
  `target_utc = anchor_utc + horizon`, để hai horizon được gắn cùng đợt nước thực tế.
  Target chung trải từ 20/09 11:00 đến 15/10 14:00 UTC+7; độ phủ đầu/cuối khác nhau giữa horizons.
- Chia lịch cố định: [20/09,27/09), [27/09,04/10), [04/10,11/10), [11/10,16/10),
  tất cả bắt đầu lúc 00:00 UTC+7. Số mẫu +12h/+24h lần lượt 157/145, 168/168,
  168/168, 99/111. Trong mỗi ô so sánh, candidate và B có đúng cùng mẫu.
- Chia đợt từ **quan trắc**, không nhìn lỗi model: đánh dấu giờ u có
  `abs(H(u)-H(u-12h)) > 200` hoặc `abs(H(u)-H(u-24h)) > 200`; gộp giờ đánh dấu
  liên tiếp nếu cách nhau không quá 24h, lấy toàn đoạn từ giờ đầu tới giờ cuối.
  Không thêm padding. Đây là các đợt biến động mô tả, không khẳng định là các
  trận mưa/lũ độc lập. Phần ngoài các đợt vẫn được tính và báo cáo.

HT = Hồng Thái; NH = Nhâm; BM = Bạch Mã; PB = Hồ chứa nước Phú Bài;
BD = Thủy điện Bình Điền; VP = VP QLKTCTTL TP Huế.

### Toàn bộ 17 tổ hợp trên +12h/+24h

RMSE là RMSE gộp hai horizons. Sửa được = B sai >200, candidate không sai >200;
phát sinh = B không sai >200 nhưng candidate sai >200. Không cộng lỗi các model
để coi là số mẫu độc lập. Cột tổng 5 horizons gồm +1/+3/+6/+12/+24h.

| Tổ hợp | RMSE +12 | RMSE +24 | RMSE gộp | N200 +12 | N200 +24 | N200 gộp | Sửa được | Phát sinh | N200 tổng 5 horizons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **B** | 82,15 | 97,23 | **90,00** | 16 | 27 | 43 | — | — | **48** |
| HT+BM+PB | 84.25 | 98.01 | 91.39 | 11 | 29 | 40 | 9 | 6 | 50 |
| BM+PB | 84.07 | 98.97 | 91.82 | 14 | 27 | 41 | 13 | 11 | 63 |
| HT+NH+BM+PB | 83.78 | 102.06 | 93.37 | 13 | 31 | 44 | 11 | 12 | 62 |
| HT+VP | 90.60 | 99.78 | 95.30 | 18 | 26 | 44 | 9 | 10 | 74 |
| NH | 86.16 | 94.91 | 90.64 | 22 | 23 | 45 | 6 | 8 | 58 |
| HT+NH+PB | 88.26 | 101.54 | 95.13 | 15 | 30 | 45 | 6 | 8 | 60 |
| HT+PB | 84.24 | 98.21 | 91.49 | 17 | 29 | 46 | 4 | 7 | 54 |
| HT | 88.44 | 94.79 | 91.67 | 21 | 26 | 47 | 3 | 7 | 58 |
| HT+BD | 93.05 | 99.29 | 96.22 | 18 | 29 | 47 | 10 | 14 | 71 |
| VP | 91.54 | 101.16 | 96.47 | 18 | 30 | 48 | 8 | 13 | 77 |
| HT+BM+PB+VP | 80.15 | 98.73 | 89.92 | 13 | 36 | 49 | 8 | 14 | 72 |
| HT+BM+BD+VP | 90.51 | 96.01 | 93.30 | 19 | 32 | 51 | 11 | 19 | 90 |
| HT+NH+BM | 89.75 | 103.63 | 96.94 | 20 | 34 | 54 | 12 | 23 | 77 |
| BM+PB+VP | 88.41 | 105.21 | 97.17 | 18 | 36 | 54 | 12 | 23 | 96 |
| PB | 90.40 | 96.29 | 93.39 | 27 | 29 | 56 | 8 | 21 | 81 |
| BD | 90.44 | 99.36 | 95.00 | 27 | 30 | 57 | 4 | 18 | 78 |
| BM | 87.33 | 102.65 | 95.30 | 21 | 37 | 58 | 8 | 23 | 88 |

### Các khoảng lịch cố định

Mỗi ô là **N200 +12h / N200 +24h**. Hai khoảng đầu không có lỗi lớn của B,
nên việc các model bằng 0 ở đó không chứng minh đã cải thiện khả năng dự báo nước lớn.

| Tổ hợp | 20–26/09 | 27/09–03/10 | 04–10/10 | 11–15/10 |
| --- | --- | --- | --- | --- |
| B | 0 / 0 | 0 / 0 | 7 / 17 | 9 / 10 |
| HT+BM+PB | 0 / 0 | 0 / 0 | 7 / 16 | 4 / 13 |
| BM+PB | 0 / 0 | 0 / 0 | 8 / 12 | 6 / 15 |
| HT+NH+BM+PB | 0 / 0 | 0 / 0 | 7 / 14 | 6 / 17 |
| HT+VP | 3 / 0 | 0 / 0 | 7 / 13 | 8 / 13 |
| NH | 0 / 0 | 0 / 0 | 12 / 13 | 10 / 10 |
| HT+NH+PB | 0 / 0 | 0 / 0 | 8 / 17 | 7 / 13 |
| HT+PB | 0 / 0 | 0 / 0 | 11 / 16 | 6 / 13 |
| HT | 0 / 0 | 0 / 0 | 10 / 15 | 11 / 11 |
| HT+BD | 3 / 0 | 0 / 0 | 8 / 20 | 7 / 9 |
| VP | 2 / 0 | 0 / 0 | 10 / 16 | 6 / 14 |
| HT+BM+PB+VP | 0 / 0 | 0 / 0 | 6 / 16 | 7 / 20 |
| HT+BM+BD+VP | 0 / 0 | 0 / 0 | 11 / 15 | 8 / 17 |
| HT+NH+BM | 0 / 0 | 0 / 0 | 15 / 24 | 5 / 10 |
| BM+PB+VP | 3 / 0 | 0 / 0 | 7 / 21 | 8 / 15 |
| PB | 7 / 0 | 0 / 0 | 16 / 19 | 4 / 10 |
| BD | 4 / 0 | 1 / 0 | 15 / 21 | 7 / 9 |
| BM | 0 / 0 | 0 / 0 | 14 / 24 | 7 / 13 |

![Chênh lệch số lỗi theo khoảng và horizon](../reports/model_comparisons/rain_combination_stability_blocks.png)

### Các đợt biến động quan trắc

Thời gian dưới đây là **target time UTC+7**. n là số dự báo cho mỗi horizon,
không phải số sự kiện độc lập. E1/E5 rất ngắn nên không dùng để tuyên bố độ ổn định.

| Đợt | Bắt đầu | Kết thúc | n / horizon | B N200 +12 | B N200 +24 |
| --- | --- | --- | --- | --- | --- |
| E1 | 21/09 11:00 | 21/09 13:00 | 3 | 0 | 0 |
| E2 | 28/09 11:00 | 29/09 14:00 | 28 | 0 | 0 |
| E3 | 06/10 09:00 | 08/10 06:00 | 46 | 7 | 16 |
| E4 | 09/10 08:00 | 10/10 00:00 | 17 | 0 | 0 |
| E5 | 11/10 11:00 | 11/10 14:00 | 4 | 0 | 0 |
| E6 | 13/10 06:00 | 14/10 16:00 | 35 | 9 | 10 |
| Ngoài đợt | — | — | 459 | 0 | 1 |

**42/43 lỗi lớn của B (97,7%) nằm trong E3 và E6.** Hai đoạn này chứa tổng
81 mốc giờ, tương ứng 162/1.184 dự báo (13,7%). Sáu đợt nói trên gộp lại chiếm
266/1.184 dự báo; bốn đợt còn lại không có lỗi lớn của B.

![Quan trắc và vị trí lỗi theo ngày](../reports/model_comparisons/rain_combination_stability_events.png)

### Tổ hợp có lợi ích thực sự ở đâu?

**HT+BM+PB, +12h:** tổng 16 → 11 lỗi; sửa 8, phát sinh 3. Ngày 07/10 sửa
3 lỗi nhưng tạo 1 lỗi mới ngày 07/10 và 2 lỗi mới ngày 08/10, nên E3 vẫn
7 → 7 lỗi. Toàn bộ mức giảm ròng 5 lỗi đến từ E6: 9 → 4 lỗi, RMSE
143,60 → 117,91 mm. Trên toàn test, RMSE +12h vẫn tăng 82,15 → 84,25 mm.

**HT+BM+PB, +24h:** E3 giữ nguyên 16 lỗi, E6 tăng 10 → 13 lỗi
(RMSE 179,74 → 202,30 mm); sửa được 1 lỗi ngoài các đợt. Vì vậy tổng
+24h tăng 27 → 29. Xét hai horizons chung, giảm lỗi ở cả hai khoảng lịch
04–10/10 (24 → 23) và 11–15/10 (19 → 17), nhưng đó là đánh đổi giữa
horizons, không phải cùng cải thiện cả +12h lẫn +24h.

**NH, +24h:** sửa đúng 4 lỗi B và không phát sinh lỗi >200 mới; cả 4
target rơi vào 07–08/10. E3 giảm 16 → 13 lỗi; lỗi được sửa thứ tư ở
08/10 sau cửa sổ E3. E6 giữ nguyên 10 lỗi nhưng RMSE tăng
179,74 → 186,32 mm. Trên toàn test RMSE giảm 97,23 → 94,91 mm;
chưa thấy lợi ích giảm N200 ở một đợt khó thứ hai. +12h lại tăng 16 → 22 lỗi.

**BM+PB:** giảm N200 hai horizons 43 → 41 nhưng sửa 13 lỗi và phát sinh
11 lỗi. Khoảng 04–10/10 cải thiện 24 → 20, khoảng 11–15/10 xấu đi
19 → 21. Riêng +24h sửa 6 nhưng phát sinh 6 lỗi mới, tất cả lỗi mới
rơi vào 13/10: tổng 27 → 27 che giấu việc đổi vị trí lỗi.

### Nước tăng nhanh có được khắc phục không?

Phân nhóm từng dự báo theo `actual(t+h)-H(t)` (chỉ dùng hậu kiểm, không dùng
làm feature hay luật chọn model lúc dự báo). Đây là thay đổi ròng sau
horizon; nhóm ≤200 vẫn có thể dao động mạnh ở giữa nên không gọi là nước ổn định.

| Horizon | Nhóm quan trắc | n | B N200 | HT+BM+PB N200 | NH N200 |
| --- | --- | --- | --- | --- | --- |
| +12h | Tăng >200 mm | 11 | 6 | 5 | 6 |
| +12h | Giảm >200 mm | 19 | 2 | 0 | 4 |
| +12h | Thay đổi ròng trong ±200 mm | 562 | 8 | 6 | 12 |
| +24h | Tăng >200 mm | 32 | 21 | 22 | 18 |
| +24h | Giảm >200 mm | 11 | 4 | 4 | 4 |
| +24h | Thay đổi ròng trong ±200 mm | 549 | 2 | 3 | 1 |

Điểm khó nhất vẫn là **+24h khi nước tăng >200 mm**: B sai 21/32,
HT+BM+PB sai 22/32, NH sai 18/32. Toàn bộ các lỗi lớn này đều là dự báo
thấp hơn thực tế trên 200 mm, không phải dự báo quá cao. Tổ hợp ba trạm không khắc phục
được nhóm này dù có tổng N200 +12h/+24h thấp hơn. Số mẫu tăng nhanh chỉ
11/592 ở +12h và 32/592 ở +24h, có tương quan theo giờ và không tương
đương 43 đợt nước tăng độc lập.

### Kiểm tra nhạy cảm và giới hạn kết luận

- Đổi khoảng gộp giờ đánh dấu từ 24h sang 12h/6h tạo 6/9/12 đợt;
  vị trí và số đợt được gọi là tốt hơn thay đổi. Tổng lỗi, ngày sửa/phát sinh
  và kết luận lợi ích +12h HT+BM+PB tập trung 13–14/10, +24h NH tập trung
  07–08/10 không thay đổi. Không chọn khoảng gộp để tối đa số đợt thắng.
- Chia theo anchor thay vì target với các mốc lịch 27/09, 04/10, 11/10
  vẫn cho HT+BM+PB 24 → 23 và 19 → 17, NH 24 → 25 và 19 → 20,
  BM+PB 24 → 20 và 19 → 21 trên hai khoảng có lỗi; kết luận không đảo chiều.
- Chỉ phân đoạn test cũ, không train lại, không rolling-origin validation,
  không thử nhiều seed. Việc xem nhiều tổ hợp trên test đã dùng nhiều lần
  có thiên lệch lựa chọn. Không suy ra trạm nào gây ra biến động mực nước,
  hoặc thiếu dữ liệu là nguyên nhân duy nhất. Số giờ không phải cỡ mẫu độc lập.
- B vẫn tốt nhất nếu xét đồng thời RMSE gộp và N200 cả năm horizons.
  Nếu chỉ xét mục tiêu +12h/+24h, HT+BM+PB là đánh đổi ít N200 hơn nhưng
  RMSE cao hơn; NH có tín hiệu riêng +24h. Chưa có bằng chứng đủ để thay B.
- Không dùng nhãn đợt hoặc thay đổi nước tương lai để chuyển model khi chạy
  thực tế. Các kết quả này không chứng minh một bộ chọn model online sẽ tốt hơn.

### Nguồn kết quả từng mẫu

Mỗi đường dẫn dưới đây chứa `paired_predictions.csv`, `manifest.json` và
snapshot model/preprocessing tại thời điểm đánh giá. Những bảng trong phần
này so với **Model B**, khác baseline production của các báo cáo nguồn.

- [BM](../reports/model_comparisons/bach_ma/comparison.md) — run `49dfc9dd11d54cff87f46bc6d9b2de6b`.
- [BM+PB](../reports/model_comparisons/bach_ma_phu_bai/comparison.md) — run `85190dbd301e4597bbebdbd95a001675`.
- [BM+PB+VP](../reports/model_comparisons/bach_ma_phu_bai_vp/comparison.md) — run `7ea67afb174743b99432ccfcee2cf3e7`.
- [BD](../reports/model_comparisons/binh_dien/comparison.md) — run `1a7eeb04fc334028939fb84e58472d1e`.
- [HT](../reports/model_comparisons/hong_thai/comparison.md) — run `c4cc3bb3aac4496982977ece6eed7923`.
- [HT+BM+BD+VP](../reports/model_comparisons/hong_thai_bach_ma_binh_dien_vp/comparison.md) — run `171afc54fe624fb4bd3059f32ff00d8c`.
- [HT+BM+PB](../reports/model_comparisons/hong_thai_bach_ma_phu_bai/comparison.md) — run `d3ec10dbb3c74759b123e6fa462f6f70`.
- [HT+BM+PB+VP](../reports/model_comparisons/hong_thai_bach_ma_phu_bai_vp/comparison.md) — run `7ee5cf7add2742269508e125e5e6276a`.
- [HT+BD](../reports/model_comparisons/hong_thai_binh_dien/comparison.md) — run `e179167ef9ef45bca1afd7e9bab3e318`.
- [HT+NH+BM+PB](../reports/model_comparisons/hong_thai_nham_Bach_ma_phu_bai/comparison.md) — run `343de6430748403c83732725adc9f79c`.
- [HT+NH+BM](../reports/model_comparisons/hong_thai_nham_bach_ma/comparison.md) — run `5537b146aa24451aa3b46e3e07d66ecd`.
- [HT+NH+PB](../reports/model_comparisons/hong_thai_nham_phu_bai/comparison.md) — run `e939ece23a3342088b6cabb143918b56`.
- [HT+PB](../reports/model_comparisons/hong_thai_phu_bai/comparison.md) — run `eb60326b1c1f40fa9fb2d4600509b3bb`.
- [HT+VP](../reports/model_comparisons/hong_thai_vp/comparison.md) — run `e389c8cdf271414690d6cea75e0842b5`.
- [NH](../reports/model_comparisons/nham/comparison.md) — run `4362bbbc1dab468393c741fb21ec582b`.
- [PB](../reports/model_comparisons/phu_bai/comparison.md) — run `8a47c424279041769243839ae7071cd2`.
- [VP](../reports/model_comparisons/vp/comparison.md) — run `646d51d345e94c4eae6457370dd41066`.

## Kiểm tra độ ổn định qua seed — 21/10/2026

Thực hiện theo yêu cầu: giữ nguyên cấu hình B và cách chia dữ liệu, thay seed
để kiểm tra các khác biệt giữa tổ hợp trạm có lặp lại qua khởi tạo và shuffle hay không.
Đây không phải hyperparameter search, không thêm dữ liệu và không thay production/Model B.

### Thiết kế cố định trước khi chạy

- Ba cấu hình: B = `hong_thai+nham`; `hong_thai+bach_ma+phu_bai`; `nham`.
- Năm seed cố định cho từng cấu hình: **42, 123, 2026, 7, 99**. Tổng **15 run** mới;
  chạy lại cả seed 42 để kiểm tra tái lập, không trộn run cũ vào thống kê.
- Dataset `train_data/da_vien_1h_2026-10-15_rain_pool_o24`; chọn 5 targets
  +1/+3/+6/+12/+24h. Đã kiểm tra cả train/val/test: timestamp, 26 feature B
  và năm nhãn khớp chính xác với dataset B `da_vien_1h_2026-10-15_forecast_mask`.
- Cả 15 run: ResidualLSTM, hourly sequence 96; hidden 32, 3 layers;
  LSTM/input/head dropout 0,7/0,2/0,2; AdamW lr 0,0009, weight decay 0,001;
  correction penalty 0,1; horizon weights đều 1; tail penalty 0,5, threshold
  200 mm, transition 20 mm; batch 128, gradient accumulation 1, num-worker 0;
  tối đa 50 epochs, patience 25. Scheduler hiện có StepLR mỗi 10 epochs, gamma 0,9.
- Checkpoint được chọn bằng **N200 trên validation của cả năm horizons**, phân định
  hòa bằng RMSE; không chọn checkpoint hoặc seed theo test. Input B/ba trạm/NH
  lần lượt 26/32/20, giữ hidden và các tham số còn lại; số tham số đầu vào thay đổi theo số trạm.
- CUDA deterministic theo `src.ai.utils.set_seed`, CUBLAS_WORKSPACE_CONFIG hiện có;
  GPU RTX 4050 Laptop, PyTorch 2.14.1+cu130. OMP/MKL threads = 2 cho mọi run.
- Codex chạy B và NH, Gemini chạy HT+BM+PB song song, không sửa chung file nguồn.
- Mỗi seed lưu riêng dưới `model/experiments/rain_seed_stability_20261021/<combination>/seed_<seed>/`;
  không ghi đè checkpoint thí nghiệm cũ. MLflow chỉ có metadata/settings/metrics, không upload model.
- Experiment [da_vien_rain_seed_stability_20261021](http://localhost:5000/#/experiments/18).

### Kiểm tra kết quả trước tổng hợp

- 15/15 run FINISHED; đúng mọi tham số cố định, có checkpoint/preprocessing và
  post-training comparison hoàn tất. Không tính run thiếu hoặc thay run thất bại bằng seed khác.
- Mỗi run chấm 592 anchor chung, 2.960 dự báo; khóa `(anchor_utc, horizon_minutes)`
  và actual khớp trên cả 15 run. Focus +12h/+24h: 1.184 dự báo/run.
- B seed 42 tái hiện checkpoint epoch 24, 48 lỗi cả năm horizons, 16/27 lỗi
  +12h/+24h, RMSE cả năm horizons xấp xỉ 75,78 mm. Hai candidate seed 42 cũng
  tái hiện số lỗi +12h/+24h của các snapshot báo cáo cũ (11/29 và 22/23).
- Dùng `candidate_error_mm` trong snapshot từng run. Cột `serving_*` vẫn là
  production, không dùng làm B. So sánh cặp luôn là candidate seed s với **B seed s**,
  không phải tất cả candidate với B seed 42.

### Trung bình và độ phân tán của năm seed

Mỗi ô = **trung bình ± độ lệch chuẩn mẫu** (`ddof=1`, n=5). N200 là số
dự báo có sai số tuyệt đối >200 mm trên cùng mẫu; giá trị trung bình có thể
không nguyên. Độ lệch chuẩn qua seed không phải khoảng tin cậy qua các trận lũ.

| Cấu hình | N200 +12h | RMSE +12h | N200 +24h | RMSE +24h | N200 +12/+24 | RMSE +12/+24 |
| --- | --- | --- | --- | --- | --- | --- |
| B: Hồng Thái + Nhâm | 20.80 ± 5.54 | 87.99 ± 3.67 | 28.80 ± 2.77 | 101.65 ± 3.70 | 49.60 ± 8.11 | 95.07 ± 3.61 |
| Hồng Thái + Bạch Mã + Phú Bài | 14.60 ± 3.36 | 85.10 ± 3.27 | 33.00 ± 6.60 | 101.86 ± 5.55 | 47.60 ± 8.53 | 93.86 ± 4.33 |
| Nhâm | 21.80 ± 5.40 | 90.19 ± 3.43 | 26.00 ± 2.83 | 101.45 ± 3.88 | 47.80 ± 7.60 | 96.00 ± 3.23 |

| Cấu hình | N200 cả 5 horizons | Min–max N200 | RMSE cả 5 horizons |
| --- | --- | --- | --- |
| B: Hồng Thái + Nhâm | 67.80 ± 15.50 | 48–84 | 80.40 ± 3.05 |
| Hồng Thái + Bạch Mã + Phú Bài | 65.20 ± 12.91 | 50–84 | 79.43 ± 2.52 |
| Nhâm | 65.20 ± 9.26 | 55–77 | 81.33 ± 2.30 |

![Kết quả từng seed](../reports/model_comparisons/rain_seed_stability_20261021.png)

### Diễn giải kết quả nhiều seed

**Có tín hiệu lựa chọn trạm theo horizon, rõ hơn so với chỉ xem seed 42.**
HT+BM+PB có N200 +12h thấp hơn cấu hình B được train lại cùng seed ở **5/5 cặp**:
16→11, 20→17, 15→11, 26→18, 27→16. Trung bình giảm 20,8→14,6 lỗi
(giảm 29,8%), RMSE trung bình giảm 87,99→85,10 mm nhưng RMSE chỉ giảm ở 3/5 cặp.
Ở +24h, HT+BM+PB lại tăng N200 trung bình 28,8→33,0; chỉ thấp hơn B ở
1/5 seed, bằng ở 1/5 và cao hơn ở 3/5. Cấu hình này chưa tốt đồng thời ở cả hai horizon.

**NH có lợi ích riêng ở +24h:** N200 thấp hơn B cùng seed ở **4/5 cặp**,
trung bình 28,8→26,0 (giảm 9,7%). RMSE trung bình chỉ giảm khoảng 0,21 mm,
101,65→101,45, nên lợi ích chủ yếu ở ngưỡng lỗi lớn. +12h lại tăng N200
trung bình 20,8→21,8 và RMSE 87,99→90,19 mm.

**Seed ảnh hưởng đáng kể:** cấu hình B có N200 cả năm horizons từ 48 tới 84,
trung bình 67,8 ±15,5. B seed 42 đang được giữ có 48 lỗi, RMSE 75,78 mm;
đây vẫn là checkpoint tốt nhất xét cả năm horizons trong 15 run này. Không có
run mới nào giảm được tổng lỗi và RMSE so với checkpoint B hiện tại. Việc cấu
hình ba trạm có trung bình cả năm horizons tốt hơn B (65,2 so với 67,8 lỗi;
RMSE 79,43 so với 80,40 mm) là lợi ích nhỏ, không lặp lại ở mọi seed:
N200 chỉ giảm 2/5 cặp, đồng thời N200/RMSE tốt hơn ở 2/5 cặp.

**Phân biệt hai kết luận:** ổn định qua seed trên test cố định không đồng nghĩa
ổn định qua các đợt nước/mùa khác nhau. Phân tích trước vẫn cho thấy lợi ích
tập trung ở một vài đoạn test. Kết quả mới củng cố tín hiệu giảm N200 +12h của
HT+BM+PB và +24h của NH trước nhiễu khởi tạo, không chứng minh ảnh hưởng trực
tiếp của trạm hay khả năng tổng quát. Tiếp tục giữ checkpoint B, không tự chọn
seed/ghép model theo kết quả test để thay production.

### So sánh cặp với B cùng seed

N200 thấp hơn/bằng/cao hơn đếm trên năm cặp. Cột cả hai chỉ số tốt hơn
nghĩa là N200 và RMSE đều không tăng, ít nhất một chỉ số giảm. Δ = candidate − B.

| Cấu hình | Phạm vi | N200 thấp/bằng/cao | RMSE thấp hơn | Cả hai chỉ số tốt hơn | ΔN200 TB | ΔRMSE TB |
| --- | --- | --- | --- | --- | --- | --- |
| HT+BM+PB | +12h | 5/0/0 | 3/5 | 3/5 | -6.20 | -2.88 |
| HT+BM+PB | +24h | 1/1/3 | 2/5 | 2/5 | +4.20 | +0.20 |
| HT+BM+PB | +12h/+24h gộp | 3/0/2 | 3/5 | 2/5 | -2.00 | -1.20 |
| HT+BM+PB | Cả 5 horizons | 2/0/3 | 3/5 | 2/5 | -2.60 | -0.97 |
| NH | +12h | 2/0/3 | 2/5 | 1/5 | +1.00 | +2.20 |
| NH | +24h | 4/0/1 | 3/5 | 3/5 | -2.80 | -0.21 |
| NH | +12h/+24h gộp | 2/1/2 | 2/5 | 2/5 | -1.80 | +0.93 |
| NH | Cả 5 horizons | 3/0/2 | 2/5 | 2/5 | -2.60 | +0.93 |

### Chi tiết toàn bộ 15 run

Val N200 và best epoch dùng cả năm horizons, đúng quy tắc chọn checkpoint B.

| Seed | Cấu hình | Best epoch | Val N200 | N200 +12 | N200 +24 | N200 +12/+24 | N200 cả 5 | RMSE +12/+24 | RMSE cả 5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | B | 24 | 92 | 16 | 27 | 43 | 48 | 90.00 | 75.78 |
| 42 | HT+BM+PB | 33 | 92 | 11 | 29 | 40 | 50 | 91.39 | 77.04 |
| 42 | NH | 39 | 88 | 22 | 23 | 45 | 58 | 90.64 | 77.78 |
| 123 | B | 48 | 89 | 20 | 28 | 48 | 64 | 93.95 | 79.53 |
| 123 | HT+BM+PB | 48 | 87 | 17 | 44 | 61 | 84 | 100.70 | 83.27 |
| 123 | NH | 40 | 78 | 15 | 23 | 38 | 55 | 98.23 | 82.68 |
| 2026 | B | 35 | 99 | 15 | 26 | 41 | 60 | 94.46 | 80.51 |
| 2026 | HT+BM+PB | 45 | 94 | 11 | 32 | 43 | 71 | 91.63 | 79.96 |
| 2026 | NH | 43 | 82 | 22 | 27 | 49 | 72 | 98.48 | 83.86 |
| 7 | B | 50 | 86 | 26 | 33 | 59 | 83 | 99.37 | 82.49 |
| 7 | HT+BM+PB | 36 | 93 | 18 | 33 | 51 | 62 | 95.53 | 79.63 |
| 7 | NH | 44 | 88 | 30 | 29 | 59 | 77 | 97.27 | 81.56 |
| 99 | B | 30 | 98 | 27 | 30 | 57 | 84 | 97.56 | 83.69 |
| 99 | HT+BM+PB | 36 | 87 | 16 | 27 | 43 | 59 | 90.08 | 77.25 |
| 99 | NH | 27 | 88 | 20 | 28 | 48 | 64 | 95.35 | 80.79 |

### Đường dẫn run và snapshot

- B seed 42: [MLflow](http://localhost:5000/#/experiments/18/runs/651c48c983464e528ddb4d17c6596862), [snapshot so với production](../reports/model_comparisons/651c48c983464e528ddb4d17c6596862_44b268bd/comparison.md).
- HT+BM+PB seed 42: [MLflow](http://localhost:5000/#/experiments/18/runs/db9e226afb484e19a5ca66092a4fc303), [snapshot so với production](../reports/model_comparisons/db9e226afb484e19a5ca66092a4fc303_e9f5a7c7/comparison.md).
- NH seed 42: [MLflow](http://localhost:5000/#/experiments/18/runs/675267a99093494190922eed8e3d4ec8), [snapshot so với production](../reports/model_comparisons/675267a99093494190922eed8e3d4ec8_c0588c92/comparison.md).
- B seed 123: [MLflow](http://localhost:5000/#/experiments/18/runs/794b2d9037124968b9971f0dc1dc25f3), [snapshot so với production](../reports/model_comparisons/794b2d9037124968b9971f0dc1dc25f3_c9053fd8/comparison.md).
- HT+BM+PB seed 123: [MLflow](http://localhost:5000/#/experiments/18/runs/aa43797c18fc4c048479deab3094e219), [snapshot so với production](../reports/model_comparisons/aa43797c18fc4c048479deab3094e219_190ed24f/comparison.md).
- NH seed 123: [MLflow](http://localhost:5000/#/experiments/18/runs/5ad45b2334fe41f1883fce3ab717e856), [snapshot so với production](../reports/model_comparisons/5ad45b2334fe41f1883fce3ab717e856_100fbf15/comparison.md).
- B seed 2026: [MLflow](http://localhost:5000/#/experiments/18/runs/e692cac26c57463b8bc4a10d9a398c5c), [snapshot so với production](../reports/model_comparisons/e692cac26c57463b8bc4a10d9a398c5c_91a5f195/comparison.md).
- HT+BM+PB seed 2026: [MLflow](http://localhost:5000/#/experiments/18/runs/dc38376e00514c9f9c8a0159889cbcbc), [snapshot so với production](../reports/model_comparisons/dc38376e00514c9f9c8a0159889cbcbc_297b4077/comparison.md).
- NH seed 2026: [MLflow](http://localhost:5000/#/experiments/18/runs/5084e6d77ece46e596ea4c53e01751b5), [snapshot so với production](../reports/model_comparisons/5084e6d77ece46e596ea4c53e01751b5_feb6bf64/comparison.md).
- B seed 7: [MLflow](http://localhost:5000/#/experiments/18/runs/8b8c80449b1144eb831f82df07ceab64), [snapshot so với production](../reports/model_comparisons/8b8c80449b1144eb831f82df07ceab64_c819ee49/comparison.md).
- HT+BM+PB seed 7: [MLflow](http://localhost:5000/#/experiments/18/runs/ddd4c549be464b72ac948e26ff538ae2), [snapshot so với production](../reports/model_comparisons/ddd4c549be464b72ac948e26ff538ae2_4c4e98a3/comparison.md).
- NH seed 7: [MLflow](http://localhost:5000/#/experiments/18/runs/db32e4b9122d47ff96088bdbaf19cae2), [snapshot so với production](../reports/model_comparisons/db32e4b9122d47ff96088bdbaf19cae2_46d24263/comparison.md).
- B seed 99: [MLflow](http://localhost:5000/#/experiments/18/runs/b53a440fadc94df99e9ebc2804da0022), [snapshot so với production](../reports/model_comparisons/b53a440fadc94df99e9ebc2804da0022_b704323f/comparison.md).
- HT+BM+PB seed 99: [MLflow](http://localhost:5000/#/experiments/18/runs/58f3a3c85c534b25a96364f27fe3dc90), [snapshot so với production](../reports/model_comparisons/58f3a3c85c534b25a96364f27fe3dc90_1d191c5b/comparison.md).
- NH seed 99: [MLflow](http://localhost:5000/#/experiments/18/runs/d1f32934835d46c0aa84313c5d92fa88), [snapshot so với production](../reports/model_comparisons/d1f32934835d46c0aa84313c5d92fa88_33ca1f61/comparison.md).

Lưu ý: thống kê seed đo độ nhạy với khởi tạo/shuffle trên **cùng test cũ**.
Không tạo thêm đợt mưa độc lập, không chứng minh tổng quát sang mùa khác và
không khẳng định nguyên nhân thủy văn của một trạm. Test đã xem nhiều lần;
không tự thay Model B hoặc chọn seed có test tốt nhất để tuyên bố cải thiện.
