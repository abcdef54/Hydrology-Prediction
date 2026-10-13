# Nghiên cứu và phân tích mô hình

Thư mục này lưu công cụ tạo dữ liệu và các notebook dùng trong quá trình phát
triển mô hình. Backend của ứng dụng nằm trong `src/backend/`; mã train và đánh
giá mô hình nằm trong `src/ai/`.

| File | Mục đích |
|---|---|
| `build_training_dataset_1h_top4.py` | Tạo dataset theo giờ với bốn trạm mưa đã chọn, chia Train/Val/Test và kiểm tra dữ liệu |
| `seasonal_analysis.ipynb` | Phân tích mùa mưa, tương quan mưa–mực nước và các đợt mưa |
| `mlflow_metrics.ipynb` | Tổng hợp, đối chiếu metric từ MLflow và đánh giá theo mùa |

## Sử dụng

Chạy lệnh từ thư mục gốc dự án:

```bash
.venv/bin/python research/build_training_dataset_1h_top4.py --help
```

Mở các notebook tại đây bằng Jupyter hoặc VS Code. Các đường dẫn dữ liệu và
artifacts vẫn tính từ thư mục gốc dự án. Cần có dữ liệu/model/MLflow tương ứng
trên máy để chạy; việc chuyển thư mục không tạo lại dataset hoặc chạy model.
Kết quả xuất vào `reports/`, được bỏ qua trong Git.
