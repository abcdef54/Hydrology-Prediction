# XGBoost / LightGBM trên dataset forecast mới

Trạng thái 20/10/2026: đã chuẩn bị code, kiểm tra schema, lag, native save/load
và so sánh trên tập mẫu thật; chưa chạy training tree trên dataset thật.

Entry point riêng: `research.tree_horizon_experiment`. Các wrapper tree cũ
trong `src.ai.models` và command train cũ giữ nguyên.

- Mặc định dataset `da_vien_1h_2026-10-15_forecast_mask_o24`, chọn 5 mốc chung.
  Cũng nhận dataset `_forecast_mask` khi `--outputs 5`. Chỉ hỗ trợ biến thể mask.
- Toàn bộ 26 feature, gồm forecast ở từng timestep quá khứ, trải theo thứ tự
  lag0 (mới nhất), lag1,…,lag95. 96 ×26 =2496 cột. Không đưa target/timestamp
  vào input; không thay missing bằng 0, không fit scaler/imputer cho tree.
- Giữ split/purge24h và warmup riêng của từng split: **6095/985/592** mẫu.
  Không reset/sửa dataset nguồn. Mỗi model dùng preprocessing riêng của nó.
- `--target-mode residual` mặc định: học `H(t+h)−H(t)` rồi cộng lại `H(t)`.
  `direct` học mực nước tương lai trực tiếp; đây là thí nghiệm riêng.
- Mỗi horizon là một booster độc lập, dùng squared error đơn vị mm. Không
  áp dụng tail loss, correction penalty, aux-weight của LSTM. Với 24 output,
  không có backbone chung hoặc tác dụng regularization qua head phụ.
- `--checkpoint-metric large_error_rate`: chọn round mỗi horizon bằng
  `(số |error|>200mm, RMSE)`, lexicographic; `rmse` chọn RMSE thuần. Chỉ dùng
  validation. Không dùng test để chọn round. Nếu không round nào tốt hơn
  persistence validation, horizon đó giữ persistence và ghi rõ trong artifact.
- Thư mục riêng tự tạo có timestamp; `--output-dir` nếu dùng phải chưa tồn tại.
  Checkpoint JSON/text native, không upload model lên MLflow, không thay app.

## Command XGBoost

```bash
.venv/bin/python -m research.tree_horizon_experiment \
  --mlflow-ex-name da_vien_tree_horizons \
  --dataset-dir train_data/da_vien_1h_2026-10-15_forecast_mask_o24 \
  --train-method xgb \
  --outputs 5 \
  --sequence-length 96 \
  --target-mode residual \
  --seed 42 \
  --n-estimators 500 \
  --early-stopping 50 \
  --learning-rate 0.05 \
  --max-depth 4 \
  --n-jobs 4 \
  --device cpu \
  --checkpoint-metric large_error_rate
```

XGBoost có thể chọn `--device cuda` khi CUDA thực sự khả dụng; runner từ chối
silent CPU fallback theo kiểm tra PyTorch. Native artifact được reload CPU để
đánh giá, không cần GPU chỉ để đọc checkpoint. LightGBM runner dùng CPU vì
không giả định package cài sẵn được build với GPU.

## Command LightGBM

```bash
.venv/bin/python -m research.tree_horizon_experiment \
  --mlflow-ex-name da_vien_tree_horizons \
  --dataset-dir train_data/da_vien_1h_2026-10-15_forecast_mask_o24 \
  --train-method lgbm \
  --outputs 5 \
  --sequence-length 96 \
  --target-mode residual \
  --seed 42 \
  --n-estimators 500 \
  --early-stopping 50 \
  --learning-rate 0.05 \
  --max-depth 4 \
  --num-leaves 15 \
  --n-jobs 4 \
  --device cpu \
  --checkpoint-metric large_error_rate
```

Đổi `--outputs 24` để train đủ 24 booster; đánh giá với B/production vẫn chỉ
tính năm mốc chung. Metric đủ24 nằm riêng. Không cộng19 mốc tree vào mẫu số
khi so với5 mốc LSTM. Không coi cấu hình mặc định là đã được chứng minh tốt.

MLflow local localhost5000 chỉ lưu settings/metrics; thêm `--no-mlflow` nếu
không chạy server. Báo cáo và checkpoint local vẫn được lưu.

## Kiểm tra không train

```bash
.venv/bin/python -m research.tree_horizon_experiment \
  --train-method xgb --outputs 5 --check-only
```

## Đầu ra

Trong thư mục được in cuối lệnh:

- `tree_model.json`: thứ tự feature/target, sequence length, status, round và
  fallback từng horizon. Chỉ status complete mới load như model đầy đủ.
- `horizon_XXh.json` (XGBoost) hoặc `.txt` (LightGBM): booster tại đúng round đã
  chọn; horizon fallback không tạo booster giả.
- `settings.json`, `lagged_features.json`, `validation_hXX.csv`.
- `REPORT.md`, `comparison_B_metrics.csv`, `comparison_production_metrics.csv`:
  RMSE, MAE gộp trong summary JSON, N200 và tỷ lệ từng mốc trên giao anchor.
- `comparison_*_sequences.csv`: thiếu/thừa, chuỗi dài nhất, số điểm lỗi liên tục;
  report tập trung +12/+24h. Không coi nhiều horizon là nhiều đợt độc lập.
- `all_horizon_metrics.csv`, `all_predictions.csv.gz`: toàn bộ horizon của tree.

Tree 96h và B96h hiện dùng592 anchor test. Model production đọc registry local
lúc so sánh, không phải kiểm tra tiến trình API đang nằm trong RAM. Actual và
timestamp được kiểm tra khớp; model reference dùng đúng scaler đã lưu.
Test đã xem nhiều lần là đánh giá khám phá; không tự chọn/thay model phục vụ.

Native load và dự báo trên frame có nhãn để đánh giá lại:

```python
import pandas as pd
from src.ai.tree_sequence import TreeSequenceForecast
model = TreeSequenceForecast("model/experiments/<run>")
data, prediction_mm = model.predict_frame(pd.read_csv("<dataset>/test.csv"))
# Hàng dự báo khớp data["times"]; cột khớp model.spec["target_columns"].
```

Đây là loader đánh giá có lọc nhãn; chưa phải API inference live không có target.

API đã đối chiếu package local XGBoost3.4.1/LightGBM4.7.0 và tài liệu chính thức:
[XGBoost callbacks](https://xgboost.readthedocs.io/en/latest/python/python_api.html),
[LightGBM callbacks](https://lightgbm.readthedocs.io/en/latest/_modules/lightgbm/callback.html).
