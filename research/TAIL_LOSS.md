# Phạt sai số mực nước quanh ngưỡng 200 mm

Mục tiêu: giảm tỷ lệ dự báo có |sai số| >200 mm, đồng thời giữ thành phần MSE
để vẫn học độ chính xác chung. Chưa khẳng định sẽ tốt hơn model bước 1.

## Loss

Với `z_pred`, `z_true` là target chuẩn hóa và `scale[h]` được fit trên train:

```text
error_mm[h] = abs(z_pred[h] - z_true[h]) * target_scale[h]
alpha = horizon_weights / sum(horizon_weights)
base = mean_batch(sum_h(alpha[h] * (z_pred[h] - z_true[h])**2))
tail = mean_batch(sum_h(alpha[h] * softplus((error_mm[h] - threshold_mm) / transition_mm)))
data_loss = base + tail_penalty * tail
training_objective = data_loss + correction_penalty * mean(correction**2)
```

Giữ horizon weights `1 1 1.5 1.5 2` và correction penalty `0.1`. Mean của target
scaler triệt tiêu khi lấy hiệu, chỉ cần scale để đổi sai số về mm. Không so
trực tiếp sai số chuẩn hóa với ngưỡng 200. Softplus tăng trơn quanh ngưỡng;
transition 20 mm nghĩa là có áp lực giảm lỗi cả khi đang tiến gần 200 mm,
không chỉ khi đã vượt. Đây là surrogate, không bảo đảm giảm số lỗi trên test.

## Tham số mới

| CLI | Mặc định | Ý nghĩa |
|---|---:|---|
| `--lstm-tail-penalty` | 0 | Hệ số lambda; 0 giữ loss cũ |
| `--lstm-tail-threshold-mm` | 200 | Ngưỡng vật lý |
| `--lstm-tail-transition-mm` | 20 | Độ rộng chuyển tiếp |
| `--lstm-checkpoint-metric` | `val_loss` | `large_error_rate` ưu tiên ít lỗi hơn, hòa chọn pooled RMSE thấp hơn |

Số mẫu/horizon validation cố định nên tối thiểu số lỗi tương đương tối thiểu
phần trăm. Checkpoint criterion đếm không trọng số trên mọi horizon; loss vẫn
có trọng số horizon. Ngưỡng kiểm tra là **strict >**; đúng 200 không tính lỗi lớn.
Early stopping đi theo criterion đã chọn. Với `large_error_rate`, `best_val_loss`
là loss của checkpoint được chọn, **không nhất thiết là loss nhỏ nhất lịch sử**.

Persistence epoch 0 được đánh giá bằng cùng data loss đầy đủ và cùng tiêu chí,
không tính correction penalty vào validation. Điều này tránh chọn sai baseline
khi loss mới có thêm tail. Test không tham gia chọn checkpoint.

MLflow ghi settings, validation count/rate/RMSE từng horizon và tổng, base MSE,
tail penalty, cùng các metric của checkpoint được chọn. Không upload model.
Checkpoint của cấu hình mới tự tách thư mục theo lambda/ngưỡng/transition/criterion,
không ghi đè đường dẫn mặc định cũ; lặp đúng cùng cấu hình vẫn dùng cùng đường dẫn.

## Chạy thử

```bash
.venv/bin/python -m src.ai.train \
  --mlflow-ex-name da_vien_tail_loss \
  --dataset-dir train_data/da_vien_1h_2026-10-15_forecast_mask \
  --train-method residual_lstm \
  --horizon 1h \
  --feature-set water_level+rain+flood_season+weather_forecast \
  --num-rain-station 2 \
  --seed 42 \
  --batch-size 128 \
  --num-epochs 150 \
  --early-stopping 25 \
  --gradient-accumulation 1 \
  --device cuda \
  --lstm-seq-len 96 \
  --lstm-hidden-size 32 \
  --lstm-num-layers 3 \
  --lstm-drop-out 0.5 \
  --lstm-input-dropout 0.2 \
  --lstm-head-dropout 0.2 \
  --lstm-lr 0.01 \
  --lstm-weight-decay 0.001 \
  --lstm-correction-penalty 0.1 \
  --lstm-horizon-weights 1 1 1.5 1.5 2 \
  --lstm-tail-penalty 0.1 \
  --lstm-tail-threshold-mm 200 \
  --lstm-tail-transition-mm 20 \
  --lstm-checkpoint-metric large_error_rate
```

Để phân biệt tác động của loss với tác động chọn checkpoint, dùng cùng dataset
và thử: (A) lambda 0 + val_loss; (B) lambda 0 + large_error_rate; (C) lambda
0.03/0.1/0.3 + large_error_rate. Chọn trên validation. Test hiện tại đã xem nhiều
lần; xác nhận cuối bằng giai đoạn mới và theo các đợt, không chỉnh để khớp riêng test.
Auto comparison vẫn đối chiếu model app. Cần đối chiếu thêm checkpoint bước 1
trên cùng anchor để đếm lỗi cũ được sửa và lỗi mới phát sinh.
