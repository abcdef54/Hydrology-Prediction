# Các model dự báo mực nước

Thư mục này chứa các model dự báo mực nước đã được chọn cho ba trạm ở Huế, cùng cấu hình dataset, thông tin chọn model và hướng dẫn sử dụng. Đây là phần người dùng repo cần để dùng hoặc đánh giá lại các model; toàn bộ quá trình nghiên cứu dẫn tới chúng được kể ở [`../documentations/`](../documentations/README.md).

Mỗi model dự báo **mực nước tuyệt đối (mm) tại một trạm** ở các mốc **+3, +6 và +12 giờ** sau thời điểm phát dự báo, dựa trên lịch sử mực nước của trạm và một số nguồn dữ liệu phụ (dự báo mưa, dự báo gió, mực nước biển dự báo, triều).

## 1. Chọn model nào

| Trạm | Model nên dùng | Mốc | Lỗi > 200 mm | Tài liệu |
|---|---|---|---|---|
| **Cảng Cá Thuận An** (`0074647402`) | `thuan_an/ensemble_34k.json` (trung bình hai model) | +3, +6, +12 h | 0 ở validation, test, giai đoạn mới | [thuan_an/README.md](thuan_an/README.md) |
| **Trạm Ngã Ba Sình** (`0074647705`) | `nga_ba_sinh/combined_v1/` (ghép theo mốc) | +3, +6, +12 h | 0 ở validation, test, giai đoạn mới | [nga_ba_sinh/README.md](nga_ba_sinh/README.md) |
| **Dã Viên** (`0074647403`) | không đóng gói checkpoint; ghi lại dataset và lệnh train | +3 h | 0 | [da_vien/README.md](da_vien/README.md) |
| | như trên | +6 h | 6 (một epoch) hoặc 8 (ghép) | |
| | chưa có | +12 h | 37–72 | |

Thuận An và Ngã Ba Sình đạt mục tiêu ở cả ba mốc. Dã Viên mới đạt ở mốc +3 h; nghiên cứu cho trạm này đang tạm dừng. Thư mục này chỉ giữ các model cuối cùng của mỗi trạm; các model thử nghiệm khác nằm trong lịch sử git (commit `783f894`).

**Lưu ý về độ tin cậy.** Các model được chọn sau khi đã xem validation, test và giai đoạn dữ liệu mới (15/10–12/11/2026), và hầu hết chỉ huấn luyện với một seed, nên "0 lỗi" là con số lạc quan. Sai số lớn nhất của các model 0 lỗi nằm ở 190–198 mm, rất sát ngưỡng. Cần chấm lại trên dữ liệu sau 12/11/2026 trước khi dựa vào chúng.

## 2. Cấu trúc thư mục

```text
models/
├── README.md                  tài liệu này
├── PIPELINE.md                quy trình chung: dữ liệu, dataset, kiến trúc, loss, huấn luyện, chấm điểm
├── NOTES.md                   ghi chú nghiên cứu Thuận An và Ngã Ba Sình
├── thuan_an/
│   ├── README.md              mô tả model, đầu vào, hiệu năng
│   ├── REPRODUCE.md           tái tạo từ đầu (dữ liệu, dataset, train, chọn epoch)
│   ├── dataset_config.json    cấu hình dựng dataset
│   ├── ensemble_34k.json      quy tắc ensemble và hiệu năng
│   ├── wind_tide30_fc_gru32x3_seq12_cosine_ep77/   model đơn
│   └── wind_gru32x2_seq25_ep33/                    thành phần thứ hai của ensemble
├── nga_ba_sinh/
│   ├── README.md
│   ├── REPRODUCE.md
│   ├── dataset_config_onshore_tide7_sea7.json, ..._sea.json
│   ├── eval_config_ext_20261112_*.json             cấu hình dataset đánh giá đến 12/11
│   └── combined_v1/           combined.json + short_3h_6h/ (+3, +6 h) + long_12h/ (+12 h)
└── da_vien/
    ├── README.md              kết quả tốt nhất, dataset, lệnh train (không có checkpoint)
    └── dataset_config_h36_all.json   cấu hình dựng dataset
```

**Một thư mục model** (mỗi checkpoint) gồm:

| File | Nội dung |
|---|---|
| `model.pt` | trọng số (PyTorch `state_dict`) |
| `model.preprocessing.joblib` | bộ chuẩn hóa đầu vào và đầu ra (fit chỉ trên train), thứ tự cột đầu vào, độ dài cửa sổ, ánh xạ trạm |
| `model.architecture.json` | kiến trúc (kích thước, số lớp, số đầu vào, đầu ra) |
| `model.loss.json` | cấu hình hàm loss lúc train |
| `selection.json`, `summary.json` | lý do chọn epoch, lệnh train, chỉ số, mã sha256 (một số model) |
| `validation_predictions.parquet`, `*_eval_metrics.csv` | dự báo và chỉ số validation đã lưu |

Checkpoint từng epoch của các lần train đã bị xóa; chỉ giữ các epoch được chọn. Các đường dẫn `model/experiments/...` trong `selection.json` hay `package.json` chỉ là ghi chép nguồn gốc.

## 3. Kiến trúc chung

Tất cả đều là **ResidualGRU**: GRU nhiều lớp đọc cửa sổ lịch sử ngắn (12–25 giờ), rồi một đầu ra tuyến tính cho từng mốc dự báo. Dự báo của mỗi mốc là

```
mực nước dự báo(t + h) = mực nước hiện tại (đã đưa về thang chuẩn hóa của mốc h) + phần hiệu chỉnh học được
```

tức mô hình học **mức thay đổi so với giữ nguyên mực nước** (persistence), và khởi tạo cho đầu ra đúng bằng persistence. Mô hình nhỏ: 12.747 đến 279.435 tham số mỗi thành phần. Chi tiết (số tham số từng phần, loss, scheduler) ở [PIPELINE.md](PIPELINE.md), mục 4–6.

Hàm loss `combined` là bình phương sai số chuẩn hóa cộng một thành phần phạt riêng cho lỗi lớn quanh ngưỡng 200 mm; hệ số `under-weight` quyết định mức phạt cho dự báo thiếu.

## 4. Đầu vào

Mọi đầu vào tại thời điểm phát t chỉ dùng thông tin đã có tại t. Các nhóm đầu vào (kênh) và trạm dùng chúng:

| Nhóm | Nội dung | Thuận An | Ngã Ba Sình | Dã Viên |
|---|---|:---:|:---:|:---:|
| Cơ bản | mực nước trạm (trung bình giờ, đã lọc gai), cờ thiếu, cờ mùa lũ | ✓ | ✓ | ✓ |
| `fc_ecmwf` (18 kênh) | mưa dự báo ECMWF 0–3, 3–6, 6–12, 12–24 giờ tới ở các ô A Lưới, Thuận An, Bạch Mã và trung bình 7 ô (log1p) | ✓ | ✓ | ✓ |
| `wind` | gió, áp suất ngoài khơi (ECMWF Previous Runs, bản dự báo khoảng 24 giờ trước): 15 kênh ở Thuận An; 4 kênh gió vào bờ ở Ngã Ba Sình và Dã Viên | ✓ | ✓ | ✓ |
| `tide` | triều thiên văn: hồi quy hài hòa trên 30 ngày (Thuận An) hoặc 7 ngày (hai trạm kia) của mực nước trạm | ✓ | ✓ | ✓ |
| `sea` | mực nước biển ngoài khơi dự báo (Open-Meteo Marine, `previous_day1`) | | ✓ | ✓ |
| `station_obs` | mực nước quan trắc Ngã Ba Sình (đã lọc gai) | | | ✓ |
| `spring_neap` | pha nước cường/nước kém | | | (ứng viên +1…+6 h) |

Cách tính từng nhóm, nguồn dữ liệu và quy tắc chọn lượt chạy dự báo nằm ở [PIPELINE.md](PIPELINE.md), mục 3. Thứ tự kênh đầu vào của từng model nằm trong `model.preprocessing.joblib` (khóa `feature_columns`).

Nguồn dữ liệu cần có khi chạy thật:

1. **Quan trắc mực nước** từ API DCU Huế (nhịp 10 phút), tổng hợp thành trung bình giờ trên khoảng (t − 1 h, t], loại số đo hỏng và lọc gai.
2. **ECMWF Single Runs** (mưa dự báo) qua Open-Meteo: lượt chạy mới nhất khởi tạo trong khoảng 6–24 giờ trước t.
3. **ECMWF Previous Runs** (gió, áp suất, bản `previous_day1`).
4. **Open-Meteo Marine** (mực nước biển `previous_day1`; cho Ngã Ba Sình và Dã Viên).

## 5. Cách dùng

Cần môi trường của repo (`requirements.txt`) và chạy từ thư mục gốc.

**a) Chấm một model trên dataset đã dựng.** Dataset dựng từ file cấu hình của trạm (`dataset_config*.json`) bằng `src/ai/station_dataset/build.py`; xem các lệnh ở `REPRODUCE.md` của từng trạm.

```python
from src.ai.station_dataset.evaluate import predict

df = predict('models/thuan_an/wind_tide30_fc_gru32x3_seq12_cosine_ep77/model.pt',
             '<thư mục dataset>', 'test_exploratory')
# cột: anchor_utc, horizon_hours, predicted_mm, actual_mm
```

Tập: `validation`, `purged` (các giờ ranh giới), `test_exploratory`, `all`.

**b) Ensemble hoặc ghép theo mốc.**

```bash
# Thuận An: trung bình đều hai model
python -m src.ai.station_dataset.evaluate ensemble \
  --member train_data/thuan_an/ta_water_only_wind_tide30_fc models/thuan_an/wind_tide30_fc_gru32x3_seq12_cosine_ep77 \
  --member train_data/thuan_an/ta_water_only_wind models/thuan_an/wind_gru32x2_seq25_ep33

# Ngã Ba Sình: mỗi mốc lấy từ đúng một model, trong một khoảng thời gian
python -m src.ai.station_dataset.evaluate combined \
  --member train_data/nga_ba_sinh/nbs_onshore_tide7_fc_sea7_v3 models/nga_ba_sinh/combined_v1/short_3h_6h 3,6 \
  --member train_data/nga_ba_sinh/nbs_onshore_tide7_fc_sea_v3 models/nga_ba_sinh/combined_v1/long_12h 12 \
  --start 2026-09-01T00:00:00+07:00 --end 2026-10-31T23:00:00+07:00

```

**c) Chạy thật, từng thời điểm.**

```python
from src.ai.sequence_checkpoint import SequenceRuntime

rt = SequenceRuntime('models/nga_ba_sinh/combined_v1/short_3h_6h/model.pt')   # tự đọc model.preprocessing.joblib cùng thư mục
mm = rt.predict_mm(x, station_ids)   # x: tensor [1, L, F] đã chuẩn hóa; trả mực nước dự báo (mm), [1, số mốc]
```

`x` có kích thước `[1, độ dài cửa sổ, số kênh]` (12 × 39 hoặc 25 × 35 ở Ngã Ba Sình; 12 × 41 ở Thuận An), các kênh theo đúng thứ tự `feature_columns` và đã chuẩn hóa bằng bộ chuẩn hóa trong `model.preprocessing.joblib`; giá trị thiếu đặt bằng 0 kèm cờ thiếu. Cách dễ nhất để dựng `x` đúng là dựng dataset cho khoảng thời gian cần dự báo bằng `build.py` với lưới mở rộng (`--grid-end-utc`) và bộ chuẩn hóa cũ (`--normalizers-from`), như hướng dẫn trong `REPRODUCE.md`.

Muốn chạy độc lập khỏi repo, mã nguồn `src/ai/station_dataset/deploy.py` và `research/scripts/make_deploy_folders.py` dựng gói dự báo tự chứa (model dưới dạng dict và mảng numpy, kèm `predict.py`).

## 6. Quy ước đánh giá

- **Lỗi lớn:** |dự báo − thực tế| > 200 mm, chặt (đúng 200 mm không tính). Số lỗi lớn là tiêu chí xếp hạng chính; MAE xếp sau.
- **Nhãn:** mực nước trung bình giờ trong khoảng (t − 1 h, t]. Nếu nhãn là số đo tức thời lúc hh:00 thì kết quả đổi (với Dã Viên `v1`: +3 h từ 0 thành 4 lỗi, +6 h từ 12 thành 19 lỗi).
- **Chia tập:** train đến 29/07/2026; validation 31/07–14/09; test 15/09–14/10; giai đoạn mới 15/10–12/11/2026. Các giờ ranh giới giữa các tập (`purged`, 24 giờ mỗi phía) được chấm riêng.
- **Chấm mọi epoch trên mọi tập**, kể cả giờ ranh giới và giai đoạn mới, rồi chọn.
- Kết quả là **hồi cứu**: dự báo thời tiết trong quá khứ được giả định có sẵn sau khi khởi tạo 6 giờ; thời điểm thực sự có sẵn chưa được kiểm chứng.

## 7. Điều cần biết khi dùng

- **Sensor Ngã Ba Sình đổi mốc đo** lúc 15:20 UTC ngày 10/10/2026 (mực nước lệch khoảng −172 mm so với trước đó). Các model dự báo mức thay đổi nên dùng được với số đo thô, nhưng khi dựng đầu vào cần giữ nhất quán theo mốc mới. Model Dã Viên dùng mực nước Ngã Ba Sình nên cũng phụ thuộc vào điều này.
- **Dữ liệu đầu vào phải khớp quy tắc dựng:** lọc gai, loại đoạn sensor hỏng, chọn đúng lượt chạy dự báo. Sai quy tắc cho kết quả khác đáng kể.
- **Dự báo mưa ECMWF** trong giai đoạn train phần lớn là phiên bản 49r1, còn validation và test là 50r1 (ECMWF đổi phiên bản ngày 12/05/2026).
- **Đợt nước dâng không do mưa hoặc gió** (ví dụ 23/10/2026 ở Thuận An) không được giải thích bởi đầu vào nào; các model dự báo thiếu khoảng 150–220 mm.
- **Kết quả nhạy với seed.** Ngã Ba Sình và Dã Viên chỉ đạt kết quả tốt ở seed 23.
- **Dữ liệu chỉ từ 11/2025** nên số đợt bão lũ để học rất ít.

## 8. Tài liệu liên quan

| Tài liệu | Nội dung |
|---|---|
| [PIPELINE.md](PIPELINE.md) | Quy trình chung từ API đến model: đầu vào, dataset, kiến trúc, loss, huấn luyện, chấm điểm |
| [thuan_an/REPRODUCE.md](thuan_an/REPRODUCE.md), [nga_ba_sinh/REPRODUCE.md](nga_ba_sinh/REPRODUCE.md) | Tái tạo từng bước hai trạm |
| [NOTES.md](NOTES.md) | Ghi chú nghiên cứu Thuận An và Ngã Ba Sình |
| [da_vien/README.md](da_vien/README.md) | Dã Viên: kết quả tốt nhất, dataset, lệnh train |
| [../documentations/](../documentations/README.md) | Toàn bộ quá trình nghiên cứu, theo giai đoạn |
