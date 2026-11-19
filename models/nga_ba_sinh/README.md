# Trạm Ngã Ba Sình (0074647705): model `combined_v1`, dự báo +3/+6/+12 h

Trạm trên sông Hương, tại nơi sông Bồ nhập vào và gần phá Tam Giang (16,528°N 107,574°E). Hướng dẫn tái tạo từng bước: [REPRODUCE.md](REPRODUCE.md); phần quy trình dùng chung: [../PIPELINE.md](../PIPELINE.md). Cấu hình dataset: `dataset_config_onshore_tide7_sea7.json` và `dataset_config_onshore_tide7_sea.json`.

Chọn ngày 16/11/2026.

## Model

`combined_v1/` **ghép theo mốc dự báo**: mỗi mốc lấy đầu ra của đúng một model, không lấy trung bình (`combined.json`).

| Mốc | Thư mục | Kiến trúc | Cửa sổ | Đầu vào | Tham số | Cách train |
|---|---|---|---|---|---:|---|
| +3 h, +6 h | `combined_v1/short_3h_6h/` | GRU 64×3 | 12 giờ | 39 kênh | 74.955 | lịch học `step`, seed 23, epoch 59 |
| +12 h | `combined_v1/long_12h/` | GRU 128×3 | 25 giờ | 35 kênh | 279.435 | trọng số loss mốc +12 h gấp 2,5, seed 23, epoch 110 |

Tổng 354.390 tham số. Hai model đều được train để dự báo cả ba mốc, nhưng chỉ dùng đầu ra của mốc ghi ở trên. Lý do tách: khi dùng chung một GRU, cải thiện +6 h làm xấu +12 h và ngược lại.

## Hiệu năng 01/09–31/10/2026

1.451 thời điểm phát dự báo (đã loại 13 mẫu ở lúc sensor đổi mốc đo), mực nước đã lọc gai:

| Mốc | MAE (mm) | RMSE (mm) | Sai số lớn nhất (mm) | Lỗi > 200 mm | MAE persistence (mm) |
|---|---:|---:|---:|---:|---:|
| +3 h | 38,0 | 46,6 | 152,7 | **0** | 90,2 |
| +6 h | 55,7 | 68,8 | 187,1 | **0** | 130,6 |
| +12 h | 52,0 | 64,9 | 197,6 | **0** | 71,5 |

Theo từng giai đoạn, số lỗi > 200 mm đều bằng 0 ở cả ba mốc: validation (1.080 mẫu), test (710 mẫu) và giai đoạn mới 15/10–12/11 (668 mẫu). Sai số lớn nhất ở +12 h là 197,6 mm (đợt bão 13/09, lúc nước rút), sát ngưỡng.

## Đầu vào

| Nhóm | Nội dung | `short_3h_6h` | `long_12h` |
|---|---|---:|---:|
| Cơ bản | mực nước đã lọc gai, cờ thiếu, cờ mùa lũ | 3 | 3 |
| `fc_ecmwf` | mưa dự báo ECMWF 0–3, 3–6, 6–12, 12–24 giờ tới ở A Lưới, Thuận An, Bạch Mã và trung bình 7 ô (log1p); tuổi dự báo; tỉ lệ thiếu | 18 | 18 |
| `wind` | gió vào bờ ECMWF (Previous Runs, dự báo từ 24 giờ trước): trung bình 24 giờ qua, 6 và 12 giờ tới, thay đổi 12 giờ; tỉ lệ thiếu | 5 | 5 |
| `tide` | hồi quy hài hòa 7 thành phần trên 7 ngày: triều hiện tại, thay đổi sau 3/6/12 giờ; tỉ lệ thiếu | 5 | 5 |
| `sea` | mực nước biển ngoài khơi dự báo (Open-Meteo Marine `previous_day1`): thay đổi nước dâng, độ lệch so với trung bình 7 ngày, độ lệch 12 giờ tới; tỉ lệ thiếu | 4 | 4 |
| `sea` (trực tiếp) | giá trị dự báo tại +3, +6, +12 giờ trừ giá trị hiện tại | 4 | – |
| **Tổng** | | **39** | **35** |

Dataset: `nbs_onshore_tide7_fc_sea7_v3` (39 kênh) và `nbs_onshore_tide7_fc_sea_v3` (35 kênh). Thứ tự kênh có trong `model.preprocessing.joblib`.

## Xử lý dữ liệu riêng cho trạm này

- **Lọc gai** (`despike`): bỏ số đo 10 phút lệch hơn 150 mm so với trung vị cửa sổ 121 phút. Bỏ 24 số đo trên giai đoạn train/validation/test.
- **Sensor đổi mốc đo** lúc 15:20 UTC ngày 10/10/2026: mực nước tụt 157 mm trong 10 phút, chênh lệch với Thuận An và Dã Viên đổi khoảng −172 mm và giữ nguyên. Số đo thô được giữ, 13 mẫu quanh thời điểm đổi mốc bị loại khỏi dataset. Không hiệu chỉnh số đo, vì không biết mốc nào đúng.

## Cách dùng

```python
from src.ai.sequence_checkpoint import SequenceRuntime

short = SequenceRuntime('models/nga_ba_sinh/combined_v1/short_3h_6h/model.pt')
long_ = SequenceRuntime('models/nga_ba_sinh/combined_v1/long_12h/model.pt')
p_short = short.predict_mm(x_short, station_ids)   # x_short: [1, 12, 39], đã chuẩn hóa
p_long  = long_.predict_mm(x_long,  station_ids)   # x_long:  [1, 25, 35]
forecast = {3: p_short[0, 0], 6: p_short[0, 1], 12: p_long[0, 2]}   # mực nước dự báo, mm
```

Chấm trên dataset và khoảng thời gian: lệnh `evaluate combined` ở [../README.md](../README.md), mục 5 và [REPRODUCE.md](REPRODUCE.md), mục 4.6.