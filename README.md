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
| `research/scripts/`, `research/notebooks/` | Script tạo dataset/thí nghiệm và các notebook phân tích |
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

## Cấu hình Docker

[`Dockerfile`](Dockerfile) có hai target `backend` và `frontend`;
[`docker-compose.yaml`](docker-compose.yaml) cấu hình API một worker, giao diện
static qua Nginx và Redis nội bộ. Cổng giao diện mặc định là
`http://localhost:5173`, chỉnh bằng `HTTP_BIND_ADDRESS` và `HTTP_PORT`.
Nginx chuyển `/api/` tới backend, giữ timeout 190 giây cho dự báo theo khoảng.
Backend không mở cổng trực tiếp ra host.

Compose dùng **PostgreSQL hiện có**, đã migrate và có dữ liệu. `.env` cung cấp
`DATABASE_URL`; có thể đặt `DOCKER_DATABASE_URL` để ghi đè riêng cho container.
Địa chỉ database phải truy cập được từ container, không dùng `localhost` hoặc
`127.0.0.1` để trỏ tới host. Với PostgreSQL trong WSL, dùng IP của WSL trong
`DOCKER_DATABASE_URL`; `host.docker.internal` trên Docker Desktop có thể trỏ
tới Windows. PostgreSQL phải cho phép kết nối từ mạng Docker. IP WSL có thể
đổi sau khi khởi động lại, khi đó cần cập nhật `.env` và tạo lại container.
Model và scaler của ba model production lấy từ `./model`, mount chỉ đọc.
Image mặc định cài PyTorch CPU và dùng `PREDICTION_DEVICE=cpu`, kể cả khi `.env`
local đang đặt CUDA; cấu hình này chưa cấp GPU cho container.

Service `ingestion` thuộc profile `ingestion`, đồng bộ quan trắc và dự báo mưa
ECMWF mới cho các model production, dùng thông tin đăng nhập nguồn
trong `.env` và lưu trạng thái trong volume `ingestion-reports`. API tắt
`HYDROLOGY_SYNC_AUTOSTART` để tránh worker trùng. Service `migrate` thuộc profile
`tools`, dành cho migration chủ động; khởi động thông thường không chạy migration.
Các service này không tự nhập lịch sử hoặc khôi phục dữ liệu PostgreSQL.

Khi đổi địa chỉ/cổng giao diện, đặt `CORS_ORIGINS` trong `.env` theo đúng origin
trình duyệt, không có dấu `/` ở cuối. Secrets chỉ được truyền lúc chạy backend
và worker; `.dockerignore` loại `.env`, model, dataset và artifact khỏi build context.

Chạy cả ứng dụng và worker quan trắc:

```bash
docker compose --profile ingestion up --build -d
```

Nếu image đã có, bỏ `--build`. Không bật profile `ingestion` thì chỉ chạy
frontend/backend/Redis, không tự cập nhật quan trắc. Xem log worker bằng
`docker compose logs -f --tail 50 ingestion`.

Tắt toàn bộ ứng dụng và worker, giữ dữ liệu volume:

```bash
docker compose --profile ingestion down
```

## Huấn luyện và phân tích

Xem các tùy chọn huấn luyện:

```bash
.venv/bin/python src/ai/train.py --help
```

Pipeline train và đánh giá trên dataset gộp nằm trong `src/ai/`:

```bash
.venv/bin/python -m src.ai.expanded_train_experiment --help
```

Dataset `train_data/da_vien_1h_2026-10-15_expanded_train_o24/` được giữ trong Git.
Builder là `src.ai.build_expanded_train_dataset`; helper dataset và báo cáo nằm
trong `src/ai/hourly_dataset.py` và `src/ai/expanded_train_report.py`.
Dataset này gộp train+validation cũ để train, dùng test cũ làm validation;
không có test độc lập. Thay vị trí code không thay checkpoint production.

Khởi động MLflow sau khi đã cài các dependencies phục vụ huấn luyện:

```bash
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Mở http://localhost:5000 để xem kết quả. Các notebook tổng hợp metric và phân
tích mưa–mực nước nằm trong [research/](research/README.md).

Dữ liệu gốc, thông tin đăng nhập và một số artifacts chỉ được lưu local, nên
cần chuẩn bị chúng để chạy đầy đủ dự án. Các báo cáo xuất ra nằm trong `reports/`.
