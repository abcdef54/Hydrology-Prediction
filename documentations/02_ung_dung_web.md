# Giai đoạn 2: Ứng dụng web

Phần này không phải nghiên cứu mô hình nên viết ngắn. Nó cho thấy các model của giai đoạn 1 đã được đưa vào một hệ thống chạy được, và hệ thống đó hiện chưa được cập nhật.

## 1. Thời gian

12/10/2026 đến khoảng 23/10/2026. Phần lớn backend và frontend được làm trong ba ngày 12–14/10. Chọn model trong giao diện (21/10), Docker Compose và worker thu thập dự báo mưa (22/10) làm sau đó. Từ đó đến nay chưa cập nhật.

## 2. Trọng tâm

Tích hợp model Dã Viên (ResidualLSTM tốt nhất lúc đó, sau đó là model B và model ba trạm) vào một ứng dụng để xem quan trắc, dự báo và so sánh dự báo với thực tế.

## 3. Thành phần

- **Backend** (`src/backend/`): FastAPI, PostgreSQL (SQLAlchemy, Alembic), Redis (cache và giới hạn request), CORS và rate limit. Các API chính: danh sách trạm, quan trắc mới nhất, lịch sử theo giờ, dự báo một mốc hoặc theo khoảng ngày, kiểm tra đủ dữ liệu đầu vào, lưu và so sánh dự báo đã lưu với thực tế.
- **Thu thập dữ liệu** (`src/backend/ingestion/`): đồng bộ quan trắc từ API DCU mỗi 11 phút, thu dự báo mưa ECMWF cho các model dùng nó.
- **Model registry** (`src/backend/prediction/model_configs.json`): ba model độc lập cho Dã Viên, chọn được từ giao diện (không phải ensemble): `legacy` (mặc định), `model_b`, `three_station`. Mỗi model có checkpoint, scaler và thứ tự feature riêng trong `model/`.
- **Frontend** (`src/frontend/`): React, Vite, MapLibre. Bản đồ trạm, thanh bên với số liệu mới nhất, bảng quan trắc theo giờ, tự điền đầu vào dự báo và cho sửa để so sánh kịch bản, biểu đồ so sánh dự báo với thực tế.
- **Triển khai:** Docker Compose (profile `ingestion` bật worker), nginx, có tùy chọn GPU.
- **Công cụ phục vụ nghiên cứu:** `src/ai/comparison.py` tự so model vừa train với model đang chạy trong app trên cùng mốc test; sinh ra các báo cáo trong `reports/model_comparisons/` mà giai đoạn 1 dùng.

## 4–6. Kết quả, so sánh, lỗi > 200 mm

Không áp dụng cho giai đoạn này. Chất lượng dự báo là của các model giai đoạn 1 (xem [01](01_da_vien_local.md)).

## 7. Điều mới rút ra

Không có phát hiện nghiên cứu. Chỉ có điểm thực dụng: viết sẵn công cụ so sánh trong quy trình train (`comparison.py`) giúp mọi model mới đều được đặt cạnh model đang phục vụ trên cùng mẫu, trên cùng scaler.

## 8. Kết luận và tình trạng hiện tại

- Dự báo trong app chỉ bật cho Dã Viên. Các trạm khác chỉ hiển thị quan trắc.
- **Chưa cập nhật các model mới:** Thuận An, Ngã Ba Sình và Dã Viên `v1` (giai đoạn 7–9) chưa được deploy trong repo này (chưa tích hợp vào app). Chúng chỉ được đóng gói riêng ở `deploy/thuan_an/` và `deploy/nga_ba_sinh/` (bundle model, `predict.py`, `worker.py` chạy độc lập) và bàn giao cho mentor.
- Nếu muốn cập nhật app: thêm các model mới vào registry và hỗ trợ đầu vào gió, triều, mực biển (hiện chỉ có mưa quan trắc và dự báo mưa).

## Nguồn

`src/ARCHITECTURE.md`, `src/backend/README.md`, `src/backend/prediction/README.md`, `src/backend/ingestion/README.md`, `src/frontend/README.md`, `reports/FORECAST_RAIN_ARCHITECTURE.md`, `model/README.md`, `README.md`, `docker/`; log git 12–23/10
