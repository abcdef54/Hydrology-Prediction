# Cảng Cá Thuận An (0074647402): tái tạo 2 model đã chọn

Tài liệu này hướng dẫn tái tạo hai model của trạm Thuận An, từ API quan trắc đến model đã đóng gói. Các khái niệm dùng chung (API, cách lấy dữ liệu Open-Meteo, quy tắc dựng dataset, kiến trúc model, loss, vòng lặp train) được giải thích trong `models/PIPELINE.md`; nên đọc file đó trước. Người đọc chỉ cần thư mục `src/`, 2 file Postman, và thư mục này.

| Thư mục | Vai trò | Dataset | Tham số | Epoch |
|---|---|---|---|---|
| `wind_tide30_fc_gru32x3_seq12_cosine_ep77/` | **Model đơn** | `ta_water_only_wind_tide30_fc` (41 đầu vào) | 21.291 | 77 |
| `wind_gru32x2_seq25_ep33/` | Thành phần thứ hai của ensemble 34k (không dùng riêng) | `ta_water_only_wind` (18 đầu vào) | 12.747 | 33 |

**Ensemble 34k** (`ensemble_34k.json`): trung bình đều mực nước dự báo (mm) của hai model trên, theo từng thời điểm phát và từng mốc giờ.

**Đã kiểm chứng ngày 12/11/2026 trên máy gốc:**
- Lấy toàn bộ quan trắc Thuận An trực tiếp từ API DCU (48.143 bản ghi), tổng hợp theo giờ bằng `src/ai/station_dataset/build.py`: kết quả **giống hệt** dữ liệu train trên cả 8.031 giờ.
- Dataset dựng bằng `src/ai/station_dataset/build.py` **giống từng byte** dataset đã dùng để train (water, samples, rainfall, calendar parquet; bộ chuẩn hóa giống hệt).
- Loader của `train.py` trả về tensor giống hệt (kể cả thứ tự xáo trộn với seed 17).
- Train là tất định: hai lần train cùng cấu hình cho dự báo trùng 0,0 mm qua 100 epoch.
- Các lượt chạy ECMWF cũ tải lại từ Open-Meteo trùng tuyệt đối với bản đã dùng.

---

## 1. Thông tin trạm

| | |
|---|---|
| Mã trạm (API và database) | `0074647402`, collection `water_level_depth`, đơn vị mm |
| Tọa độ | 16,554968N 107,635270E (cửa biển, phá Tam Giang) |
| Dữ liệu quan trắc | Từ 14/11/2025 18:50 UTC, nhịp 10 phút |
| Lưới giờ của dataset | 2025-11-14T17:00Z → 2026-10-15T07:00Z (8.031 giờ) |
| Dự báo | +3, +6, +12 h |
| Chia tập | train: t < 2026-07-29T17:00Z; validation: 2026-07-30T17:00Z → 2026-09-13T16:00Z; test thăm dò: t ≥ 2026-09-14T17:00Z (cutoff 2026-09-15 00:00 giờ VN); còn lại bị loại (24 h trước mỗi ranh giới) |

Toàn bộ tham số dataset nằm trong `models/thuan_an/dataset_config.json`: lưới, các mốc chia tập, điểm gió, 7 điểm dự báo mưa, 3 ô có tên, độ dài cửa sổ fit triều.

---

## 2. Lấy quan trắc vào PostgreSQL

Chuẩn bị `.env` và chạy migration như mục 1.2 của `PIPELINE.md`. Sau đó:
```bash
.venv/bin/python -m src.backend.ingestion.station_catalog --apply           # tạo hydrology.stations (có 0074647402)
.venv/bin/python -m src.backend.ingestion.api_backfill --station 0074647402 \
    --start 2025-11-14T00:00:00+00:00 --end 2026-10-16T00:00:00+00:00 --apply
```
Kết quả mong đợi: khoảng 48.000 quan trắc, từ 2025-11-14 18:50 UTC. Muốn chấm thêm tháng 10–11 thì đặt `--end` muộn hơn, hoặc chạy worker `python -m src.backend.ingestion --once`.

## 3. Tải dữ liệu Open-Meteo

```bash
# Gió và áp suất (dự báo từ 24 h trước), điểm yêu cầu 16,85N 107,85E -> ô lưới 16,75N 107,75E, ngoài khơi khoảng 25 km về phía đông bắc trạm
.venv/bin/python -m src.ai.station_dataset.open_meteo previous-runs --lat 16.85 --lon 107.85 \
    --start 2025-11-01 --end 2026-10-21 --out data/external/open_meteo/prev_offshore.json

# Mưa dự báo ECMWF, mỗi lượt chạy một file (khoảng 1.350 request), 7 điểm yêu cầu
.venv/bin/python -m src.ai.station_dataset.open_meteo single-runs --start 2025-11-13T00:00 --end 2026-10-15T06:00 \
    --out-dir data/external/open_meteo/ecmwf_single_runs --points \
    16.458944,107.571361 16.554968,107.63527 16.251017,107.873289 16.199675,107.857476 \
    16.122778,107.694444 16.576556,107.347889 16.221111,107.179444
```
- sha256 của `prev_offshore.json` đã dùng nằm trong `reproduce_sha256.json`. File tải lại có thể khác ở metadata (`generationtime_ms`), nhưng các biến `*_previous_day1` giống hệt.
- 7 điểm yêu cầu được Open-Meteo đưa về 7 ô đất liền (xem `all_cells` trong config), lần lượt ở gần Dã Viên, **Thuận An** (16,555/107,588, cách trạm khoảng 5 km), Bạch Mã thấp, Đỉnh Bạch Mã, Thượng Nhật, Sông Ô Lâu và A Lưới.
- Ba ô có tên dùng làm kênh riêng: `a_luoi` (16,274/107,265), `thuan_an` (16,555/107,588), `bach_ma` (16,274/107,863). Kênh `fc_basin_*` là trung bình cả 7 ô.
- Lượt chạy 2026-06-10T12:00 không có trên API; đây là điều bình thường (lúc train cũng không có).

## 4. Dựng 2 dataset

```bash
.venv/bin/python -m src.ai.station_dataset.build --config models/thuan_an/dataset_config.json \
    --name ta_water_only_wind --groups wind --output-dir train_data/thuan_an
.venv/bin/python -m src.ai.station_dataset.build --config models/thuan_an/dataset_config.json \
    --name ta_water_only_wind_tide30_fc --groups fc_ecmwf wind tide --output-dir train_data/thuan_an
```
Kết quả mong đợi (cả hai dataset giống nhau ở các phần chung):
- Số mẫu: `{'train': 6073, 'validation': 1080, 'test_exploratory': 723, 'purged': 48}`.
- Chuẩn hóa mực nước: mean 147,0908, scale 217,3541. Nhãn (mean, scale): +3 h (138,1685, 206,1978); +6 h (137,7039, 205,3832); +12 h (136,8736, 204,0262).
- sha256 của `water.parquet`, `samples.parquet`, `rainfall.parquet` và `calendar.parquet` trùng với mục `datasets` trong `reproduce_sha256.json`.

**Thứ tự cột đầu vào:**
- `ta_water_only_wind` (18 cột): `water_level_depth, water_level_missing, flood_season`, rồi 15 kênh `wind_*`: `onshore`, `along`, `speed` × (`past24`, `next6`, `next12`, `chg12`), cùng `wind_p_now`, `wind_p_chg12`, `wind_missing_frac`.
- `ta_water_only_wind_tide30_fc` (41 cột):
  - `water_level_depth, water_level_missing, flood_season`
  - `fc_{a_luoi,thuan_an,bach_ma,basin}_rain_{0_3h,3_6h,6_12h,12_24h}` (16 cột), `fc_age_hours`, `fc_missing_frac`
  - 15 kênh `wind_*` như trên
  - `tide_now, tide_d3, tide_d6, tide_d12, tide_missing_frac` (fit 30 ngày, 10 thành phần)

## 5. Train 2 model

Chạy được song song (máy gốc chỉ chạy tối đa 2 run cùng lúc). Nếu không có MLflow server thì thêm `--mlflow-tracking-uri sqlite:///mlflow_local.db`.

**5a. Model đơn**, khoảng 8 phút trên RTX 4050:
```bash
.venv/bin/python -m src.ai.train --mlflow-ex-name thuan_an --horizon 1h \
  --dataset-dir train_data/thuan_an/ta_water_only_wind_tide30_fc --target-horizons 3 6 12 \
  --rain-stations all --use-weather-forecast false --feature-set water_level+rain+flood_season \
  --train-method gru --head-mode station --lstm-seq-len 12 --lstm-hidden-size 32 --lstm-num-layers 3 \
  --lstm-drop-out 0.2 --lstm-input-dropout 0.1 --lstm-head-dropout 0.1 \
  --lstm-lr 0.0001 --lstm-weight-decay 0.001 --lr-scheduler warmup-cosine --lr-warmup-epochs 3 \
  --batch-size 64 --num-epochs 200 --early-stopping 200 \
  --change-loss combined --change-under-weight 0.3 --change-event-weight 0 --change-margin-mm 0 \
  --lstm-checkpoint-metric val_loss --skip-test --seed 17 \
  --checkpoint-dir model/experiments/thuan_an/ta_water_only_wind_tide30_fc/b64_lr0.0001_wd0.001_gru_h32x3_do0.2-0.1-0.1_seq12_e200_combinedu0.3_warmup-cosine_s17_cos \
  --diagnose-change-dynamics
```
Log mong đợi:
- `Trainable parameters: 21,291`, `Train/validation/test sequences: 6073/1080/723`
- `Validation persistence loss (ChangeRiskLoss): 0.7167`
- Val loss ở các epoch 1 / 10 / 33 / 50 / 77 / 100 / 200 lần lượt là 0,7155 / 0,1540 / 0,0874 / 0,0817 / **0,0771** / 0,0807 / 0,0822
- `Best checkpoint: epoch 77.`

**5b. Model gió**, khoảng 2 phút:
```bash
.venv/bin/python -m src.ai.train --mlflow-ex-name thuan_an --horizon 1h \
  --dataset-dir train_data/thuan_an/ta_water_only_wind --target-horizons 3 6 12 \
  --rain-stations all --use-weather-forecast false --feature-set water_level+rain+flood_season \
  --train-method gru --head-mode station --lstm-seq-len 25 --lstm-hidden-size 32 --lstm-num-layers 2 \
  --lstm-drop-out 0.2 --lstm-input-dropout 0.1 --lstm-head-dropout 0.1 \
  --lstm-lr 0.0005 --lstm-weight-decay 0.001 --lr-scheduler step \
  --batch-size 64 --num-epochs 50 --early-stopping 30 \
  --change-loss combined --change-under-weight 0.3 --change-event-weight 0 --change-margin-mm 0 \
  --lstm-checkpoint-metric val_loss --skip-test --seed 17 \
  --checkpoint-dir model/experiments/thuan_an/ta_water_only_wind/b64_lr0.0005_wd0.001_gru_h32x2_do0.2-0.1-0.1_seq25_e50_combinedu0.3_s17 \
  --diagnose-change-dynamics
```
Log mong đợi:
- `Trainable parameters: 12,747`
- Val loss ở các epoch 1 / 10 / 33 / 50 lần lượt là 0,6762 / 0,1433 / **0,0780** / 0,0806
- Chạy đủ 50 epoch; `Best checkpoint: epoch 33.`

Ý nghĩa các tham số: xem mục 4–6 của `PIPELINE.md`. Hai thay đổi đã thử và **không** tốt hơn: `--change-under-weight 0.0` (kém hơn ở mọi chỉ số), và dropout nhóm `--group-dropout fc_:0.3 wind_:0.3` (validation không còn epoch nào 0 lỗi).

## 6. Chấm điểm và chọn epoch

```bash
.venv/bin/python -m src.ai.station_dataset.evaluate epochs --dataset train_data/thuan_an/ta_water_only_wind_tide30_fc \
  --run-dir model/experiments/thuan_an/ta_water_only_wind_tide30_fc/b64_lr0.0001_wd0.001_gru_h32x3_do0.2-0.1-0.1_seq12_e200_combinedu0.3_warmup-cosine_s17_cos
```
- **Model đơn:** các epoch có 0 lỗi >200 mm trên validation là 44, 47, 50, 51, 52, 54, 57, 58, 59, 61, 64, 67, 70, 71, 72, 74, 76, 77, 78, 81, 84, 87, 91. Tất cả các epoch này cũng 0 lỗi trên test. Chọn **epoch 77**, epoch có val loss thấp nhất, do trainer tự chọn.
- **Model gió:** không có epoch nào đạt 0 lỗi validation. Chọn **epoch 33** (trainer chọn). Model này chỉ dùng trong ensemble.

Metric mong đợi tại epoch đã chọn (số lỗi >200 mm / MAE / sai số lớn nhất, đơn vị mm):

| Model | Tập | +3 h | +6 h | +12 h | Tất cả |
|---|---|---|---|---|---|
| Model đơn, ep77 | validation | 0 / 32,16 / 138,2 | 0 / 45,35 / 185,1 | 0 / 52,98 / 194,1 | **0** / 43,50 / 194,1 |
| | test | 0 / 30,23 / 119,7 | 0 / 40,44 / 137,9 | 0 / 41,36 / 151,7 | **0** / 37,34 / 151,7 |
| Model gió, ep33 | validation | 0 / 32,89 / 154,5 | 3 / 44,44 / 216,6 | 3 / 51,70 / 209,6 | 6 / 43,01 / 216,6 |
| | test | 0 / 31,87 / 133,8 | 0 / 42,31 / 184,2 | 4 / 47,18 / 212,9 | 4 / 40,45 / 212,9 |
| **Ensemble 34k** | validation | 0 / 30,23 / 135,5 | 0 / 41,40 / 185,2 | 0 / 49,02 / 187,1 | **0** / 40,22 / 187,1 |
| | test | 0 / 28,90 / 115,6 | 0 / 37,55 / 140,0 | 0 / 40,67 / 172,1 | **0** / 35,71 / 172,1 |

## 7. Đóng gói và ensemble

```bash
cp -r $(find model/experiments/thuan_an/ta_water_only_wind_tide30_fc/b64_lr0.0001_wd0.001_gru_h32x3_do0.2-0.1-0.1_seq12_e200_combinedu0.3_warmup-cosine_s17_cos -type d -name epoch_077) \
      models/thuan_an/wind_tide30_fc_gru32x3_seq12_cosine_ep77
cp -r $(find model/experiments/thuan_an/ta_water_only_wind/b64_lr0.0005_wd0.001_gru_h32x2_do0.2-0.1-0.1_seq25_e50_combinedu0.3_s17 -type d -name epoch_033) \
      models/thuan_an/wind_gru32x2_seq25_ep33
.venv/bin/python -m src.ai.station_dataset.evaluate ensemble \
  --member train_data/thuan_an/ta_water_only_wind_tide30_fc models/thuan_an/wind_tide30_fc_gru32x3_seq12_cosine_ep77 \
  --member train_data/thuan_an/ta_water_only_wind models/thuan_an/wind_gru32x2_seq25_ep33
```
Trên cùng GPU và cùng phiên bản thư viện, sha256 của `model.pt` sẽ trùng với mục `sha256` trong `selection.json` của từng thư mục. Lệnh `ensemble` in ra đúng các dòng "Ensemble 34k" ở bảng mục 6.

## 8. Đánh giá tròn tháng 9 và 10/2026 (tùy chọn)

1. Nạp quan trắc đến ít nhất 01/11/2026 04:00 UTC (mục 2).
2. Tải gió Previous Runs đến 2026-11-02 (`--end 2026-11-02`, ghi vào file khác) và mưa Single Runs đến `2026-11-01T00:00`.
3. Dựng dataset với lưới kéo dài **và bộ chuẩn hóa của dataset train**. Đặt `previous_runs_json` và `single_runs_dir` trong một bản sao của config trỏ tới dữ liệu mới:
```bash
.venv/bin/python -m src.ai.station_dataset.build --config <config_bản_sao.json> --name ta_water_only_wind_tide30_fc \
  --groups fc_ecmwf wind tide --output-dir train_data/thuan_an_eval --grid-end-utc 2026-11-01T04:00:00+00:00 \
  --normalizers-from train_data/thuan_an/ta_water_only_wind_tide30_fc
```
4. Lấy dự báo bằng `evaluate.predict(model, dataset, split)` cho `validation`, `purged` và `test_exploratory`. Giữ các mốc phát có tháng (giờ VN) là 9 hoặc 10, được 1.464 mốc cho mỗi mốc giờ.

Kết quả mong đợi:

| | MAE +3 / +6 / +12 h | Tất cả | Lỗi >200 mm | Sai số lớn nhất |
|---|---|---|---|---|
| Model đơn ep77 | 30,99 / 40,93 / 43,81 | 38,58 | 1 (phát lúc 2026-10-23 03:00 UTC, +12 h, −215) | 215,4 |
| Ensemble 34k | 30,54 / 40,07 / 44,20 | 38,27 | 0 | 190,7 |
| *Persistence* | *114,3 / 162,1 / 74,9* | *117,1* | *743* | |

## 9. Dùng khi chạy thật

Tại thời điểm phát t, dựng đúng các cột của mục 4 theo cửa sổ 12 h (model đơn) hoặc 25 h (model gió) tính đến t, theo đúng quy tắc ở mục 3 của `PIPELINE.md`:
- gió: bản dự báo từ 24 h trước;
- triều: fit 30 ngày từ mực nước đến t;
- mưa dự báo: lượt chạy ECMWF mới nhất có init ≤ t − 6 h, trong vòng 24 h, đủ cả 7 ô.

Chuẩn hóa bằng `normalizers` chứa trong `model.preprocessing.joblib`, rồi gọi `SequenceRuntime(...).predict_mm(x, [0])`. Ensemble là trung bình mm của hai model.

Hiện worker chỉ thu dự báo mưa cho ô A Lưới và chưa thu gió ngoài khơi; cần bổ sung trước khi đưa vào ứng dụng.

## 10. Lưu ý

- Mới train với một seed (17). Hai model và ensemble được chọn sau nhiều lần thử, có xem cả validation, test và đoạn 16–31/10, nên kết quả "0 lỗi" có phần lạc quan.
- Biên an toàn mỏng: sai số lớn nhất trên validation là 194 mm (model đơn) và 191 mm (ensemble, tháng 9–10).
- Đợt nước dâng 23/10/2026 (khoảng +200 mm so với triều, không mưa, gió dự báo yếu) không được giải thích bởi đầu vào nào; mọi model đều dự báo thiếu 150–220 mm.
- Dự báo ECMWF trong giai đoạn train phần lớn là bản 49r1; validation và test là bản 50r1 (ECMWF đổi phiên bản ngày 12/05/2026).
