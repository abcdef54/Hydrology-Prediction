# Backend thủy văn

Backend hiện có database PostgreSQL, chuẩn bị đầu vào, runtime Residual LSTM và
API lấy dữ liệu để tự điền form. API hiện tại chưa cung cấp endpoint chạy dự báo.
Mỗi trạm nước có model/scaler riêng, được chọn bằng `station_id`.

## Chạy tại máy local

Chạy từ thư mục gốc dự án:

```bash
.venv/bin/python -m pip install -r src/backend/requirements.txt
.venv/bin/python -m uvicorn src.backend.main:app --env-file .env --host 127.0.0.1 --port 8000
```

`.env` chứa `DATABASE_URL` của PostgreSQL đã migrate và có dữ liệu. Không đưa mật
khẩu vào Git. Ứng dụng không tự tạo bảng, nhập dữ liệu hoặc gọi API nguồn khi khởi động.

Mặc định các đường dẫn model/scaler trong registry được tính từ thư mục gốc dự án.
Có thể đặt `MODEL_ARTIFACT_ROOT` và `MODEL_CONFIG_PATH` khi triển khai ở vị trí khác.
Các file artifact phải là file tin cậy của mô hình tương ứng.

- Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI: `http://127.0.0.1:8000/openapi.json`

## API đầu vào dự báo

```text
GET /api/v1/stations/{station_id}/prediction-inputs
```

Dã Viên có mã `0074647403`. Bỏ `reference_time` để lấy đầu vào mới nhất, hoặc gửi
một mốc ISO 8601 có múi giờ. Ví dụ:

```bash
curl --get 'http://127.0.0.1:8000/api/v1/stations/0074647403/prediction-inputs' \
  --data-urlencode 'reference_time=2026-09-20T14:30:00+07:00'
```

Mốc 14:30 được làm tròn xuống 14:00. Backend lấy lịch sử trước hoặc tại mốc đó,
tạo feature theo đúng pipeline training và chọn đủ số bước theo cấu hình.

Response gồm:

- `features`: giá trị gốc tại bước cuối, theo đúng thứ tự feature. Dã Viên có 29
  feature; không gửi toàn bộ tensor 96 × 29 về form.
- Mỗi feature có `name`, `value`, `unit`, `source_station_id` và `editable`.
  Mực nước dùng mm; cờ dùng `dimensionless`. Đơn vị mưa lấy từ metadata database;
  hiện chưa xác nhận nên trả `null`, không tự gán mm hoặc chuyển đổi giá trị.
- Giá trị thiếu là `null`, không phải 0. Mưa/lag thiếu vẫn có thể `input_ready=true`
  vì preprocessing đã hỗ trợ xử lý thiếu giống lúc training.
- `editable_features`, `missing_features`, `sequence_length`, `feature_count` và
  `forecast_horizons_minutes` lấy từ cấu hình và chuỗi thực tế.
- `requested_reference_time`, `hourly_reference_time`, `anchor_observed_at` giúp
  phân biệt mốc yêu cầu, mốc làm tròn và bước cuối thực sự được dùng. Timestamp
  trả về UTC có múi giờ; frontend có thể đổi sang giờ Việt Nam khi hiển thị.
- `model_version` và `feature_schema_version` xác định cấu hình đang dùng.

Endpoint chỉ đọc dữ liệu và tải scaler để chuẩn bị input; không tải checkpoint,
chạy inference hoặc nhận override. Override sẽ được gửi qua endpoint dự báo sau này.

### Dữ liệu cũ

`INPUT_STALE_AFTER_MINUTES` mặc định **120**, theo quyết định của chủ dự án:

- Latest: `input_age_minutes` là khoảng cách từ anchor đến giờ hiện tại.
- Historical: khoảng cách từ anchor đến mốc lịch sử đã làm tròn xuống giờ.
- `is_stale=true` khi khoảng cách **lớn hơn** ngưỡng; đúng 120 phút vẫn là false.

Đây là độ trễ của anchor đầu vào, không khẳng định mọi trạm mưa đều có dữ liệu mới.
Dữ liệu cũ vẫn được trả cùng `is_stale`; không có refresh tự động trong task này.

### Lỗi

Lỗi có dạng `error: {code, message, details}` và `request_id`. Header
`X-Request-ID` có cùng mã để đối chiếu log.

| HTTP | Code | Ý nghĩa |
|---|---|---|
| 400 | `INSUFFICIENT_HISTORY` | Không đủ bước hợp lệ; trả `required`, `available`, `input_ready=false` |
| 404 | `STATION_NOT_FOUND` | Trạm không tồn tại trong database |
| 422 | `INVALID_REQUEST` / `INVALID_REFERENCE_TIME` | Mốc sai định dạng, thiếu múi giờ, ở tương lai hoặc tham số không hỗ trợ |
| 503 | `MODEL_UNAVAILABLE` | Trạm chưa bật dự báo, chưa có cấu hình hoặc scaler không dùng được |
| 503 | `DATA_SERVICE_UNAVAILABLE` / `INPUT_UNAVAILABLE` | Database hoặc dữ liệu đầu vào không khả dụng |

Chưa có Redis cache ở giai đoạn này. Khi thêm cache, dữ liệu gốc phục vụ form phải
được giữ riêng với các giá trị người dùng sửa.

## Kiểm thử

```bash
.venv/bin/python -m pip install -r src/backend/requirements-dev.txt
.venv/bin/python -m unittest discover -s src/backend/api/tests -v
```

Các test API dùng dữ liệu giả và không cần PostgreSQL. Kiểm tra trực tiếp trên
PostgreSQL local nằm trong `scripts/local/verify_prediction_inputs_api.py` và chỉ
đọc dữ liệu. Xem thêm `db/README.md` và `prediction/README.md` cho các phần bên dưới.
