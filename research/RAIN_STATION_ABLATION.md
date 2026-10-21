# Thử train với 1, 2 hoặc 4 trạm mưa

Tùy chọn `--num-rain-station N` nhận số nguyên từ 1 đến 4. Code tự tính lại
correlation trên train, xếp hạng rồi chỉ giữ feature của N trạm đứng đầu.
Toàn bộ cột mưa giờ, tổng 3/6/12/24 giờ và cờ thiếu của trạm bị loại sẽ bị bỏ
khỏi tensor, không thay bằng 0. CSV nguồn không đổi; model tự nhận input size mới.
Hiện tùy chọn áp dụng cho các model sequence, gồm Residual LSTM, không áp dụng
cho XGBoost/LightGBM. Không truyền tùy chọn thì vẫn lấy mọi trạm như trước.

Với `water_level+rain+flood_season`, seq_len = 96 và dataset hiện tại:

| Thử nghiệm | Trạm giữ lại | Tùy chọn | Tensor mỗi mẫu |
|---|---|---|---|
| Top 1 theo train | Nhâm | `--num-rain-station 1` | 96 × 11 |
| Top 2 theo train | Nhâm + Hồng Thái | `--num-rain-station 2` | 96 × 17 |
| Top 3 theo train | Nhâm + Hồng Thái + Thị trấn A Lưới | `--num-rain-station 3` | 96 × 23 |
| Đối chứng 4 trạm | Cả bốn | `--num-rain-station 4` | 96 × 29 |

Feature nước, mùa và 5 target giữ nguyên. Train/val/test dùng cùng tập cột,
scaler fit trên train của từng cấu hình; số mẫu và target không đổi do việc chọn
trạm. N ngoài 1–4, feature-set không có rain hoặc không đủ trạm có correlation hợp lệ sẽ bị từ chối.

Ranking dùng Pearson giữa mưa 6 giờ quá khứ và thay đổi nước 12 giờ tiếp theo,
trên cùng các mốc train hợp lệ ở các trạm. Không dùng validation/test để xếp hạng.
Các trạm trong bảng là kết quả của dataset hiện tại; đổi dataset thì code tính lại.
Ranking và ID thực sự chọn được in ra và lưu trong MLflow.

## Lệnh ví dụ: chỉ dùng Nhâm

Chạy từ thư mục gốc repository, MLflow server hiện tại cần sẵn sàng tại
`http://localhost:5000`. Lệnh dưới giữ một cấu hình cố định để so sánh, không
khẳng định đây là hyperparameter tốt nhất.

```bash
.venv/bin/python -m src.ai.train \
  --mlflow-ex-name da_vien_rain_ablation \
  --dataset-dir train_data/da_vien_1h_2026-10-15 \
  --train-method residual_lstm \
  --horizon 1h \
  --feature-set water_level+rain+flood_season \
  --num-rain-station 1 \
  --seed 42 \
  --batch-size 128 \
  --num-epochs 150 \
  --early-stopping 50 \
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
  --lstm-correction-penalty 0.0 \
  --lstm-horizon-weights 1 1 1.5 2 5
```

Để train top 2 hoặc cả 4, chỉ thay giá trị `--num-rain-station` theo bảng trên.
Giữ các tùy chọn còn lại giống nhau. Cần train lại cả nhóm 4 trạm cho đối chứng,
không dùng checkpoint cũ khác cấu hình loss/seed/dataset để kết luận số trạm tốt hơn.
Correlation đề xuất trạm, không bảo đảm nhóm top 1/top 2 tốt hơn.

## Checkpoint và thông tin lần train

Khi truyền `--num-rain-station`, checkpoint có thư mục con theo ID trạm, ví dụ:

```text
model/da_vien_1h_2026-10-15/rain_750007/1h/...
model/da_vien_1h_2026-10-15/rain_750007-750062/1h/...
model/da_vien_1h_2026-10-15/rain_750003-750007-750011-750062/1h/...
```

ID trong tên thư mục được sắp xếp để cùng tập trạm luôn có cùng định danh;
thứ tự feature vẫn theo CSV. Các tập trạm khác nhau không ghi đè checkpoint của nhau. Lặp lại
cùng một nhóm vẫn dùng lại đường dẫn của nhóm đó: muốn giữ từng lần thử, thêm
`--checkpoint-dir model/da_vien_1h_trial_01` và đổi tên cho các lần tiếp theo.

MLflow ghi settings, metrics, các ID thực sự được chọn, input size, feature order,
đường dẫn checkpoint và tên run có tập trạm; không upload model lên MLflow.
Checkpoint giữ trên máy. Scaler và schema lưu cạnh checkpoint với đuôi
`.preprocessing.joblib`; không dùng scaler 29 feature cho model 11/17 feature.

Chỉ đánh giá hiệu quả sau khi train: so từng horizon, số lỗi >200 mm và các đợt
nước lên/rút trên cùng thời điểm. Không cập nhật model phục vụ app trong bước này.

Sau mỗi lần train, [pipeline so sánh với model app](MODEL_COMPARISON.md) tự chạy,
in RMSE và tỷ lệ lỗi >200 mm từng horizon, lưu report cùng bản sao model/scaler.
