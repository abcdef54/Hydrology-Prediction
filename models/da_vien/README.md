# Trạm Dã Viên (0074647403): các kết quả tốt nhất và cách tái tạo

Thư mục này **không còn chứa checkpoint**. Nó ghi lại, cho từng mốc dự báo, những lần train đạt số lỗi > 200 mm thấp nhất, kèm dataset và lệnh train để tái tạo. Quá trình nghiên cứu được kể ở [`../../documentations/09_da_vien_hien_tai.md`](../../documentations/09_da_vien_hien_tai.md).

```text
da_vien/
├── README.md                     tài liệu này
└── dataset_config_h36_all.json   cấu hình dựng dataset
```

Các model đã từng được đóng gói (`v1/`, `best_h1to6_sn_aug/`, `h3_v1/`) đã bị xóa khỏi thư mục; chúng nằm trong lịch sử git (commit `783f894`).

## 1. Kết quả tốt nhất theo từng mốc

Số lỗi > 200 mm **ngoài tập train**, gồm validation, giờ ranh giới, test và giai đoạn mới 15/10–12/11/2026 (2.519 giờ). Mỗi con số là một epoch của một lần train, chọn sau khi chấm mọi epoch trên mọi tập.

| Mốc | Lần train (mục 3) | Epoch | Lỗi ngoài train | Theo tập (validation / ranh giới / test / mới) |
|---|---|---:|---:|---|
| **+3 h** | lệnh C | nhiều epoch | **0** | 43 epoch đạt 0 lỗi |
| **+6 h** | lệnh A (`aug 0.8–1.3, p 1.0`) | 67 | **6** | 2 / 0 / 1 / 3 |
| +6 h | lệnh B (`aug 0.8–1.3, p 0.5`) | — | 7 | 1 / 0 / 3 / 3 |
| +6 h | lệnh D (không augmentation) | 61 | 10 | 4 / 1 / 4 / 1 |
| +12 h | run +3/+6/+12 h, seed 23 | — | 37 | chưa có ứng viên dùng được |

Các cách ghép nhiều checkpoint (trung bình dự báo của 10 epoch cuối, epoch 91–100) cho kết quả ổn định hơn nhưng nhiều lỗi hơn một epoch tốt nhất:

| Mốc | Cách ghép | Lỗi ngoài train |
|---|---|---:|
| +3 h | trung bình epoch 91–100 của lệnh C | 0 (sai số lớn nhất 197 mm) |
| +6 h | trung bình epoch 91–100 của lệnh A và lệnh B (20 checkpoint) | 8 (đều 200–217 mm) |
| +6 h | trung bình epoch 91–100 của lệnh D | 12 (lớn nhất 256 mm) |

Với lệnh A và B, mô hình dự báo cả quỹ đạo +1…+6 h; cùng cách ghép 20 checkpoint cho +1, +2 h: 0 lỗi; +3 h: 1; +4 h: 2; +5 h: 5; +6 h: 8.

## 2. Dataset

Hai dataset đã được dựng sẵn trong `train_data/da_vien/`:

| Dataset | Nội dung | Dùng cho |
|---|---|---|
| `dv_h36_all` | mốc +3, +6 h; cửa sổ dựng 12 giờ (tối đa 96 giờ); nhóm `tide`, `rain_obs`, `station_obs`, `sea`, `wind`, `fc_ecmwf`, `lin_fc` | lệnh C, D |
| `dv_h36_sn` | như `dv_h36_all` thêm nhóm `spring_neap` (pha nước cường/nước kém) | lệnh A, B |

Cả hai dựng từ cấu hình `models/da_vien/dataset_config_h36_all.json`. Chia tập: train đến 29/07/2026, validation 31/07–14/09, test từ 15/09; giai đoạn mới 15/10–12/11 cần dựng thêm (mục 4).

**Mô hình thực sự đọc** các nhóm `wind` (4 kênh gió vào bờ), `tide` (triều 7 ngày), `fc_ecmwf` (mưa dự báo 7 ô, log1p), `sea` (mực nước biển dự báo Marine `previous_day1`), `station_obs` (mực nước quan trắc Ngã Ba Sình đã lọc gai, đưa về cùng mốc đo sau 10/10/2026), và (lệnh A, B) `spring_neap`. Hai nhóm `lin_fc`, `rain_obs` có trong dataset nhưng bị bỏ qua bởi tham số `--rain-stations`.

Dựng lại dataset (nếu `train_data/da_vien/` chưa có), từ thư mục gốc repo; cần các file Open-Meteo trong `data/external/open_meteo/` đúng như `dataset_config_h36_all.json` chỉ ra và cơ sở dữ liệu quan trắc đã đồng bộ:

```bash
.venv/bin/python -m src.ai.station_dataset.build --config models/da_vien/dataset_config_h36_all.json \
  --name dv_h36_all --groups tide rain_obs station_obs sea wind fc_ecmwf lin_fc --output-dir train_data/da_vien

.venv/bin/python -m src.ai.station_dataset.build --config models/da_vien/dataset_config_h36_all.json \
  --name dv_h36_sn --groups tide rain_obs station_obs sea wind fc_ecmwf lin_fc spring_neap --output-dir train_data/da_vien
```

## 3. Lệnh train

Mọi lần train dùng cùng công thức: ResidualGRU 64×3 trên nền persistence, cửa sổ 25 giờ, đầu ra chung (không station embedding), hệ số phạt thiếu 0, loss `combined`, lr 1e-4, weight decay 1e-3, lịch `step`, batch 64, 100 epoch, **seed 23**. Chỉ khác dataset, các mốc dự báo, augmentation. Việc train là tất định: cùng GPU và cùng phiên bản thư viện cho đúng các epoch trên. Cần MLflow server (hoặc thêm `--mlflow-tracking-uri sqlite:///mlflow_local.db`). Mỗi lệnh dưới đây chạy riêng, tối đa hai lệnh cùng lúc.

**Lệnh A: +1…+6 h, augmentation xác suất 1,0 (+6 h tốt nhất, epoch 67)**

```bash
.venv/bin/python -m src.ai.train --mlflow-ex-name da_vien --horizon 1h --dataset-dir train_data/da_vien/dv_h36_sn --target-horizons 1 2 3 4 5 6 --rain-stations wind+tide+fc_ecmwf+sea+station_obs+spring_neap --use-weather-forecast false --feature-set water_level+rain+flood_season --train-method gru --head-mode shared --station-embedding-size 0 --lstm-seq-len 25 --lstm-hidden-size 64 --lstm-num-layers 3 --lstm-drop-out 0.2 --lstm-input-dropout 0.1 --lstm-head-dropout 0.1 --lstm-lr 0.0001 --lstm-weight-decay 0.001 --lr-scheduler step --batch-size 64 --num-epochs 100 --early-stopping 1000 --change-loss combined --change-under-weight 0 --change-event-weight 0 --change-margin-mm 0 --lstm-checkpoint-metric val_loss --skip-test --seed 23 --diagnose-change-dynamics --augment-surge-scale 0.8:1.3:1.0 --checkpoint-dir model/experiments/da_vien/dv_h123456_sn/h123456_pers_seq25_u0_s23_aug0.8-1.3p1.0
```

**Lệnh B: +1…+6 h, augmentation xác suất 0,5**

```bash
.venv/bin/python -m src.ai.train --mlflow-ex-name da_vien --horizon 1h --dataset-dir train_data/da_vien/dv_h36_sn --target-horizons 1 2 3 4 5 6 --rain-stations wind+tide+fc_ecmwf+sea+station_obs+spring_neap --use-weather-forecast false --feature-set water_level+rain+flood_season --train-method gru --head-mode shared --station-embedding-size 0 --lstm-seq-len 25 --lstm-hidden-size 64 --lstm-num-layers 3 --lstm-drop-out 0.2 --lstm-input-dropout 0.1 --lstm-head-dropout 0.1 --lstm-lr 0.0001 --lstm-weight-decay 0.001 --lr-scheduler step --batch-size 64 --num-epochs 100 --early-stopping 1000 --change-loss combined --change-under-weight 0 --change-event-weight 0 --change-margin-mm 0 --lstm-checkpoint-metric val_loss --skip-test --seed 23 --diagnose-change-dynamics --augment-surge-scale 0.8:1.3:0.5 --checkpoint-dir model/experiments/da_vien/dv_h123456_sn/h123456_pers_seq25_u0_s23_aug0.8-1.3p0.5
```

**Lệnh C: +3/+6 h, augmentation xác suất 0,5 (+3 h đạt 0 lỗi)**

```bash
.venv/bin/python -m src.ai.train --mlflow-ex-name da_vien --horizon 1h --dataset-dir train_data/da_vien/dv_h36_all --target-horizons 3 6 --rain-stations wind+tide+fc_ecmwf+sea+station_obs --use-weather-forecast false --feature-set water_level+rain+flood_season --train-method gru --head-mode shared --station-embedding-size 0 --lstm-seq-len 25 --lstm-hidden-size 64 --lstm-num-layers 3 --lstm-drop-out 0.2 --lstm-input-dropout 0.1 --lstm-head-dropout 0.1 --lstm-lr 0.0001 --lstm-weight-decay 0.001 --lr-scheduler step --batch-size 64 --num-epochs 100 --early-stopping 1000 --change-loss combined --change-under-weight 0 --change-event-weight 0 --change-margin-mm 0 --lstm-checkpoint-metric val_loss --skip-test --seed 23 --diagnose-change-dynamics --augment-surge-scale 0.8:1.3:0.5 --checkpoint-dir model/experiments/da_vien/dv_h36_all/h36_pers_seq25_u0_s23_aug0.8-1.3p0.5
```

**Lệnh D: +3/+6 h, không augmentation (+6 h epoch 61)**

```bash
.venv/bin/python -m src.ai.train --mlflow-ex-name da_vien --horizon 1h --dataset-dir train_data/da_vien/dv_h36_all --target-horizons 3 6 --rain-stations wind+tide+fc_ecmwf+sea+station_obs --use-weather-forecast false --feature-set water_level+rain+flood_season --train-method gru --head-mode shared --station-embedding-size 0 --lstm-seq-len 25 --lstm-hidden-size 64 --lstm-num-layers 3 --lstm-drop-out 0.2 --lstm-input-dropout 0.1 --lstm-head-dropout 0.1 --lstm-lr 0.0001 --lstm-weight-decay 0.001 --lr-scheduler step --batch-size 64 --num-epochs 100 --early-stopping 1000 --change-loss combined --change-under-weight 0 --change-event-weight 0 --change-margin-mm 0 --lstm-checkpoint-metric val_loss --skip-test --seed 23 --diagnose-change-dynamics --checkpoint-dir model/experiments/da_vien/dv_h36_all/h36_pers_seq25_u0_s23
```

Lệnh D giống lệnh C, chỉ bỏ `--augment-surge-scale` và đổi thư mục lưu.

## 4. Chấm mọi epoch và chọn epoch

Sau khi train, chấm mọi epoch trên mọi tập (train, validation, ranh giới, test, giai đoạn mới) bằng `research/scripts/score_runs_all_splits.py`. Script cần một dataset **dựng lại trên lưới dài hơn** để có giai đoạn mới, giữ bộ chuẩn hóa cũ:

```bash
.venv/bin/python -m src.ai.station_dataset.build --config models/da_vien/dataset_config_h36_all.json \
  --name dv_h36_sn --groups tide rain_obs station_obs sea wind fc_ecmwf lin_fc spring_neap \
  --output-dir train_data/da_vien_eval --grid-end-utc <thời điểm quan trắc cuối, 12/11/2026> \
  --normalizers-from train_data/da_vien/dv_h36_sn

PYTHONPATH=. .venv/bin/python research/scripts/score_runs_all_splits.py train_data/da_vien_eval/dv_h36_sn \
  model/experiments/da_vien/dv_h123456_sn/h123456_pers_seq25_u0_s23_aug0.8-1.3p1.0
```

Kết quả ghi vào `reports/scores/` (số lỗi, MAE, sai số lớn nhất theo epoch, tập và mốc). Chọn epoch có số lỗi ngoài train ít nhất (theo tiêu chí xếp hạng chính); dùng GPU để chấm nhanh. Cần tải thêm dữ liệu Open-Meteo đến hết giai đoạn mới và đồng bộ cơ sở dữ liệu quan trắc trước khi dựng.

## 5. Hạn chế

- **+6 h chưa đạt 0 lỗi** (6 lỗi với epoch tốt nhất, 8 với bản ghép); **+12 h chưa có model dùng được** (model chung +3/+6/+12 h có 37–72 lỗi).
- **Kết quả nhạy seed.** Chỉ seed 23 đạt các con số trên. Cùng công thức lệnh B với seed 42 cho 18 lỗi và seed 17 cho 24 lỗi ở +6 h; ghép seed không giúp.
- Các epoch được chọn sau khi xem validation, test và giai đoạn mới (không còn tập chưa dùng), nên số liệu lạc quan. Sai số lớn nhất của các model +3 h 0 lỗi là 197 mm.
- Nếu nhãn là số đo tức thời lúc hh:00 thay vì trung bình giờ, model +3 h có 4 lỗi và model +6 h có 19 lỗi (chấm trên bản ghép `v1` cũ).
- Cần chấm lại trên dữ liệu sau 12/11/2026.
- Đầu vào lúc chạy thật: mực nước Dã Viên (lọc gai ngưỡng 85 mm, loại các đoạn sensor đứng 01–12/02 và 20–25/06/2026), mực nước Ngã Ba Sình (lệch khoảng +172 mm từ 15:20 UTC ngày 10/10/2026), mưa dự báo ECMWF, gió và mực nước biển `previous_day1`, triều ước lượng trên 7 ngày.
