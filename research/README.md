# Nghiên cứu và phân tích mô hình

Thư mục này lưu công cụ tạo dữ liệu và các notebook dùng trong quá trình phát
triển mô hình. Backend của ứng dụng nằm trong `src/backend/`; mã train và đánh
giá mô hình nằm trong `src/ai/`.

| File | Mục đích |
|---|---|
| `build_training_dataset_1h_top4.py` | Tạo dataset theo giờ với bốn trạm mưa đã chọn, chia Train/Val/Test và kiểm tra dữ liệu |
| `seasonal_analysis.ipynb` | Phân tích mùa mưa, tương quan mưa–mực nước và các đợt mưa |
| `mlflow_metrics.ipynb` | Tổng hợp, đối chiếu metric từ MLflow và đánh giá theo mùa |
| `refresh_da_vien_dataset.py` | Xuất snapshot PostgreSQL chỉ đọc, tạo dataset có phiên bản và audit |
| `resplit_hourly_dataset.py` | Chia lại snapshot theo mốc thời gian, purge horizon và giữ dataset nguồn |
| `export_rain_regime_snapshot.py` | Xuất dự báo đã lưu và quan trắc bằng transaction chỉ đọc |
| `analyze_rain_regime_errors.py`, `rain_regime_report.py` | Đối chiếu sai số theo nhóm mưa phía tây/phía nam |
| `analyze_large_error_events.py` | Phân tích chuỗi lỗi >200 mm, mưa kéo dài và dữ liệu vận hành hồ |
| `plot_forecast_error_histograms.py`, `summarize_error_histograms.py` | Histogram theo horizon và báo cáo thống kê theo khoảng ngày |
| `plot_monthly_rain_water.py` | Tổng hợp mưa và mực nước theo tháng để khảo sát cách chia dữ liệu |
| `plot_september_error_windows.py`, `plot_error_event_comparison.py` | Figure tiếng Việt cho các chuỗi dự báo thiếu/thừa |
| `compare_residual_models.py` | So sánh checkpoint mới/cũ và persistence trên cùng các mẫu |

## Sử dụng

Chạy lệnh từ thư mục gốc dự án:

```bash
.venv/bin/python research/build_training_dataset_1h_top4.py --help
```

Mở các notebook tại đây bằng Jupyter hoặc VS Code. Các đường dẫn dữ liệu và
artifacts vẫn tính từ thư mục gốc dự án. Cần có dữ liệu/model/MLflow tương ứng
trên máy để chạy; việc chuyển thư mục không tạo lại dataset hoặc chạy model.
Kết quả xuất vào `reports/`, được bỏ qua trong Git.

## Giới hạn của các công cụ phân tích hiện tại

Chạy từ thư mục gốc repository. Công cụ đọc PostgreSQL dùng `DATABASE_URL` trong
môi trường hoặc `.env`; không đưa thông tin đăng nhập vào mã nguồn. Việc export
không train model hoặc sửa quan trắc.

Một số script ghi rõ snapshot `/tmp/hydrology-*`, dataset ngày 15/10/2026 và
thư mục report cụ thể trong mã. Đây là công cụ tái lập phân tích đã thực hiện,
chưa phải pipeline nhiều trạm tổng quát. Snapshot mưa toàn mạng/vận hành hồ và
tổng hợp tháng phải có sẵn; chúng không được đóng gói trong repository. Kiểm tra
đường dẫn và sao lưu report trước khi chạy lại vì một số script ghi đè đầu ra.

`compare_residual_models.py` đọc run mới nhất của experiment `da_vien_weighted_loss`
và checkpoint local tương ứng, kiểm tra scaler, validation loss và test MAE trước
khi xuất kết quả. Phải giữ đúng cặp run/checkpoint/dataset; không dùng script như
công cụ so sánh mọi experiment hoặc như bằng chứng triển khai model.

Dataset PostgreSQL mới và checkpoint thử nghiệm giữ local; model/scaler đang
phục vụ và các dataset đã được track trước đó không đổi. Dữ liệu mưa tổng hợp
giữ phép cộng số đo nguồn của pipeline; khoảng tích lũy của từng bản ghi chưa
được xác minh. Các tập test đã phân tích nhiều lần là kết quả khám phá.

Kế hoạch tiếp theo nằm tại
[FORECAST_RAIN_ARCHITECTURE.md](../src/FORECAST_RAIN_ARCHITECTURE.md): thử số trạm
mưa, thêm dự báo mưa mới nhất đã có, cuối cùng so sánh 5 với 24 output.
