# Model phục vụ dự báo

Đặt checkpoint và bộ preprocessing tương ứng ngay trong thư mục `model/`.
Tên file gồm tên trạm, mã trạm và loại model để dễ phân biệt khi thêm trạm mới.

```text
model/
├── README.md
├── da_vien_0074647403_residual_lstm.pt
├── da_vien_0074647403_residual_lstm_preprocessing.joblib
├── da_vien_0074647403_model_b_seed42.pt
├── da_vien_0074647403_model_b_seed42_preprocessing.joblib
├── da_vien_0074647403_three_station_seed42.pt
└── da_vien_0074647403_three_station_seed42_preprocessing.joblib
```

File `.pt` chứa trọng số model. File `_preprocessing.joblib` chứa scaler đầu vào,
scaler đầu ra và metadata của đúng model đó. Cần lưu cả hai file lên GitHub để
khôi phục phần dự báo sau khi clone; backend không cần thư mục MLflow để tải chúng.

## Thêm model cho trạm khác

1. Đặt cặp file riêng theo mẫu `<ten_tram>_<ma_tram>_<loai_model>.pt` và
   `<ten_tram>_<ma_tram>_<loai_model>_preprocessing.joblib`.
2. Nếu giữ nhiều phiên bản, thêm hậu tố phiên bản giống nhau vào tên của cả cặp;
   dùng tên riêng để không ghi đè model đang phục vụ.
3. Thêm cấu hình của trạm vào
   [model_configs.json](../src/backend/prediction/model_configs.json), gồm đường
   dẫn hai file, model version, sequence length, features/sources và các horizon.
4. Kiểm chứng cặp model/scaler rồi khởi động lại API.

Registry hỗ trợ nhiều model cho mỗi trạm, phân biệt bởi `model_id`, cùng một
`is_default` cho lựa chọn mặc định. Dã Viên giữ `legacy` làm mặc định và có thêm
`model_b` (Hồng Thái + Nhâm, seed 42), `three_station` (Hồng Thái + Bạch Mã +
Phú Bài, seed 42). Các cặp mới được sao chép riêng; bản cũ và checkpoint thử nghiệm
giữ nguyên. Không cần đưa `model/experiments/` lên Git để phục vụ dự báo.

Mỗi model có scaler riêng;
trạm chưa được cấu hình sẽ không dùng model Dã Viên thay thế. Thêm kiến trúc model
mới hoặc nhóm feature mới cần có runtime/preprocessing hỗ trợ tương ứng.
