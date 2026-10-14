# Dự báo mực nước

Dự án phân tích dữ liệu thủy văn, huấn luyện mô hình và cung cấp API dự báo mực
nước tại trạm Dã Viên, Huế, cho các mốc 1, 3, 6, 12 và 24 giờ.
Các mô hình đã thử nghiệm gồm XGBoost, LightGBM và các biến thể LSTM.

## Cấu trúc dự án

| Thư mục | Nội dung |
|---|---|
| `src/backend/` | API, chạy dự báo, lưu dữ liệu và tự động cập nhật quan trắc |
| [src/frontend/](src/frontend/README.md) | React + Vite, bản đồ tương tác khu vực Huế |
| `src/ai/` | Huấn luyện và đánh giá mô hình |
| `research/` | Tạo dataset và các notebook phân tích |
| `train_data/` | Dữ liệu Train/Val/Test |
| [model/](model/README.md) | Trọng số và scaler riêng cho từng trạm |

## Chạy backend

Backend cần PostgreSQL đã có schema và dữ liệu, cùng model và scaler tương ứng.
Đặt cấu hình kết nối trong `.env`; Redis được dùng cho cache và giới hạn request.

Chạy từ thư mục gốc dự án:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r src/backend/requirements.txt
.venv/bin/python -m uvicorn src.backend.main:app --env-file .env \
  --host 127.0.0.1 --port 8000 --workers 1 --no-proxy-headers
```

Mở Swagger tại http://127.0.0.1:8000/docs để thử API.
Xem [hướng dẫn backend](src/backend/README.md) để cấu hình và chạy worker cập nhật dữ liệu.

## Chạy frontend

Mở terminal riêng:

```bash
cd src/frontend
npm ci
npm run dev
```

Mở http://localhost:5173 để xem bản đồ Huế và sidebar số đo mới nhất; lọc loại trạm
hoặc chọn một trạm để bản đồ di chuyển tới vị trí tương ứng.
Click marker hoặc item sidebar để mở box thông tin, số đo và các ngưỡng báo động.
Bản đồ nền cần Internet; marker trạm lấy từ backend đang chạy ở cổng 8000.
Xem [hướng dẫn frontend](src/frontend/README.md) cho yêu cầu Node.js và cách build.

## Huấn luyện và phân tích

Xem các tùy chọn huấn luyện:

```bash
.venv/bin/python src/ai/train.py --help
```

Khởi động MLflow sau khi đã cài các dependencies phục vụ huấn luyện:

```bash
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Mở http://localhost:5000 để xem kết quả. Các notebook tổng hợp metric và phân
tích mưa–mực nước nằm trong [research/](research/README.md).

Dữ liệu gốc, thông tin đăng nhập và một số artifacts chỉ được lưu local, nên
cần chuẩn bị chúng để chạy đầy đủ dự án. Các báo cáo xuất ra nằm trong `reports/`.
