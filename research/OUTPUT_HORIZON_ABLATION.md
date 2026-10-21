# O5/O24 — chuẩn bị dataset và tự train 24 mốc giờ

**Trạng thái 19/10/2026:** code và dataset đã chuẩn bị; đã kiểm tra dữ liệu,
forward/backward và save/load, **chưa chạy training O5/O24**.

## Dataset đã có

`train_data/da_vien_1h_2026-10-15_forecast_mask_o24/`

- Input giữ đúng 26 feature của B, forecast ở cả 96 timestep; không dùng loss
  hướng/biên độ mới hoặc policy forecast chỉ tại anchor.
- Target là mực nước trung bình giờ quan trắc thật tại +1,+2,…,+24 giờ; lấy
  đúng timestamp từ hourly.csv, không nội suy từ năm nhãn cũ.
- Giữ nguyên split/purge 24 giờ, NaN nhãn và dataset nguồn. Các nhãn +1/+3/
  +6/+12/+24 khớp nhãn nguồn. Không gọi API, không đọc thêm PostgreSQL.
- Cùng **6095/985/592** sequence train/validation/test cho cả O5 và O24.
  Builder và loader kiểm tra hai nhóm có cùng quần thể nhãn đầy đủ. Nếu
  snapshot khác có lỗ nhãn trung gian làm hai quần thể khác nhau, dừng rõ
  trước khi train thay vì âm thầm so trên các mẫu khác nhau hoặc sửa nhãn.
- Cả hai model fit scaler chỉ trên train, cùng 26 input; mean/scale năm
  target chung khớp nhau. Đổi head/scaler target theo số output, không nạp
  checkpoint B vào O24. Test đã xem nhiều lần nên chỉ đánh giá khám phá.

Builder để chạy trên checkout chưa có dataset (dùng thư mục mới nếu đã tồn tại):

```bash
.venv/bin/python -m research.build_output24_dataset
```

## Loss: giữ nguyên hệ số năm mốc cũ

Đặt C = {1,3,6,12,24}, A = 19 giờ còn lại. Với mỗi nhóm, loss B là mean MSE
chuẩn hóa + `0.5 * mean(softplus((abs(error_mm)-200)/20))`.

```text
L_O5  = B_loss(C) + 0.1 * mean(correction_C^2)
L_O24 = B_loss(C) + 0.25 * B_loss(A)
        + 0.1 * [mean(correction_C^2) + 0.25 * mean(correction_A^2)]
```

`--lstm-correction-penalty` hỗ trợ số âm, ví dụ `-0.1`, cho cả O5/O24.
Giá trị phải hữu hạn và **lớn hơn -1**: MSE và correction dùng cùng trọng số
theo nhóm nên hệ số bậc hai còn dương khi `1 + penalty > 0`.
Giá trị âm thưởng khoảng cách với persistence, không bảo đảm đúng hướng hoặc
giảm lỗi lớn. Validation/checkpoint vẫn dùng data loss hoặc metric lỗi thực tế,
không cộng phần thưởng correction vào validation loss.

`--aux-weight 0.25` là lựa chọn ban đầu, không phải giá trị đã chứng minh tối
ưu. Không normalize lại tổng 24 output: hệ số MSE mỗi mốc chính vẫn 1/5,
mỗi mốc phụ là 0.25/19. Hệ số tail và correction được chia nhóm tương tự.
Như vậy không giảm hệ số riêng của +24 giờ từ 1/5 xuống 1/24. Gradient chung
vẫn có thể thay đổi do supervision phụ và gradient clipping; đây là một
thử nghiệm multi-task với cấu hình trọng số được công bố, không chứng minh
hiệu quả chỉ từ số output. Không so trực tiếp combined val_loss O5 với O24.

Checkpoint/early stopping mặc định dựa trên **N200 của năm horizon chung**,
hòa thì RMSE gộp trên đúng năm mốc này. Các metric đủ 24 giờ vẫn được log,
không dùng tổng 24 horizon để so với tổng năm horizon. `--checkpoint-metric
val_loss` cho phép chọn bằng combined val_loss nếu cần. Cả checkpoint chính
và checkpoint combined val_loss thấp nhất đều được lưu riêng.

## Kiểm tra mà không train

```bash
.venv/bin/python -m research.output_horizon_experiment \
  --outputs 24 --device cpu --check-only
```

Chỉ một forward/backward, không optimizer.step, không epoch training,
không checkpoint và không kết nối MLflow. Cũng hỗ trợ `--outputs 5`.

## Train O24 — cấu hình B, thêm 19 target phụ

Pipeline hỗ trợ `--train-method residual_lstm` (mặc định) và `--train-method lstm`.
LSTM thường dự đoán trực tiếp target chuẩn hóa, không cộng persistence.
Để thử LSTM thường, giữ các tham số khác và thêm:

```bash
--train-method lstm --lstm-correction-penalty 0.0
```

Nếu bỏ correction penalty, mặc định là `0.1` với ResidualLSTM và `0.0` với
LSTM thường; LSTM thường từ chối correction penalty khác 0. Cả hai hỗ trợ
5/24 output, aux/tail loss, chọn checkpoint và báo cáo so với B/production.
Checkpoint epoch 0 của LSTM thường là model ngẫu nhiên chưa train, không phải
persistence. Thư mục tự tạo và tên MLflow run có tên kiến trúc để phân biệt.

Chạy từ thư mục gốc repository. Mặc định log settings/metrics vào MLflow local
`http://localhost:5000`, không upload model. Có thể thêm `--no-mlflow` để train
hoàn toàn offline. Thư mục output phải mới; không ghi đè B hay lượt train trước.

```bash
.venv/bin/python -m research.output_horizon_experiment \
  --mlflow-ex-name da_vien_output_horizons \
  --dataset-dir train_data/da_vien_1h_2026-10-15_forecast_mask_o24 \
  --outputs 24 \
  --output-dir model/experiments/da_vien_o24_seed42_aux025 \
  --seed 42 \
  --batch-size 128 \
  --num-epochs 50 \
  --early-stopping 25 \
  --device cuda \
  --lstm-seq-len 96 \
  --lstm-hidden-size 32 \
  --lstm-num-layers 3 \
  --lstm-drop-out 0.7 \
  --lstm-input-dropout 0.2 \
  --lstm-head-dropout 0.2 \
  --lstm-lr 0.0009 \
  --lstm-weight-decay 0.001 \
  --lstm-correction-penalty 0.1 \
  --lstm-tail-penalty 0.5 \
  --lstm-tail-threshold-mm 200 \
  --lstm-tail-transition-mm 20 \
  --aux-weight 0.25 \
  --checkpoint-metric large_error_rate
```

## Train lại đối chứng O5 trên cùng dataset

Command dưới dùng các mặc định đúng cấu hình B như command O24 ở trên:

```bash
.venv/bin/python -m research.output_horizon_experiment \
  --mlflow-ex-name da_vien_output_horizons \
  --dataset-dir train_data/da_vien_1h_2026-10-15_forecast_mask_o24 \
  --outputs 5 \
  --output-dir model/experiments/da_vien_o5_seed42 \
  --seed 42 --device cuda
```

Nếu thay hyperparameter O24, đặt cùng giá trị ở O5 để có đối chứng tương ứng.
Không cần tạo dataset năm nhãn riêng: loader chỉ chọn năm target C cho O5,
luôn kiểm tra quần thể mẫu vẫn trùng O24. Auxiliary weight hiệu lực là 0 ở O5.

O24 mặc định tự so với B gốc đã bảo toàn. Muốn so trực tiếp với O5 vừa train,
thêm vào command O24 (sau khi O5 đã chạy xong):

```text
--reference-checkpoint model/experiments/da_vien_o5_seed42/model_best.pt
```

## Artifact sau mỗi lượt

Trong `--output-dir`:

- `model_best.pt` và `.preprocessing.joblib`: checkpoint chọn bằng metric đã
  cấu hình; cùng thứ tự target, feature, scaler và metadata loss.
- `model_lowest_loss.pt` và preprocessing: checkpoint combined loss thấp nhất.
- `settings.json`, `result.json`, `validation_history.csv`: cấu hình, số
  tham số, các thành phần loss, metric chung/đủ horizon và epoch đã chọn.
- `REPORT.md`, `comparison_summary.json`: tự so B/reference5 và production
  trên năm horizon chung, cùng 592 anchor (2960 dự báo mỗi model).
- `comparison_*_metrics.csv`, `comparison_*_predictions.csv.gz`: RMSE, MAE,
  >200 mm theo horizon; summary có thêm >400 mm và số lỗi sửa/phát sinh.
- `all_horizon_metrics.csv`, `all_horizon_predictions.csv.gz`: riêng đủ 24
  mốc cho O24 (592×24 = 14208 dự báo), không trộn với tổng năm horizon.
- `common_error_runs.csv`, `common_extrema_summary.csv`: hướng/chuỗi lỗi và
  ghép đỉnh/đáy, đọc cả số đỉnh không ghép được cùng với lag.
- `common_horizon_predictions.png`: năm plot đối chiếu thực tế/reference/mới.

Không cập nhật backend/registry/frontend sang 24 output ở bước nghiên cứu này.
Mã chạy: `research/output_horizon_experiment.py`; loss:
`CommonHorizonAuxiliaryLoss` trong `src/ai/losses.py`.
