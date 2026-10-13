# Backend thủy văn

Backend hiện có database PostgreSQL, chuẩn bị đầu vào, runtime Residual LSTM và
API lấy dữ liệu để tự điền form và API chạy dự báo.
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

## API danh sách trạm

```bash
curl 'http://127.0.0.1:8000/api/v1/stations'
```

Response có dạng `{"stations": [...]}`. API đọc các trạm có `enabled = true`
trong PostgreSQL và sắp xếp theo mã trạm. Mỗi trạm có `station_id`, `name`,
`station_type`, `river_name`, `location` (tọa độ và địa chỉ), `measurement_unit`,
`model_configured` và `prediction_enabled`. Thông tin chưa có được trả về null.

- `model_configured`: trạm có cấu hình mô hình riêng trong registry.
- `prediction_enabled`: trạm nước được bật quyền dự báo trong database và có
  cấu hình mô hình riêng. Cờ này không xác nhận đủ lịch sử tại một thời điểm cụ thể
  hoặc xác nhận file mô hình có thể tải được; API dự báo kiểm tra các điều kiện đó.
- Hiện tại Dã Viên được phép dự báo; 4 trạm mưa cung cấp dữ liệu đầu vào.

Endpoint không tải trọng số mô hình. Khi PostgreSQL không khả dụng, API trả 503
theo định dạng lỗi chung.

## API thông tin một trạm

```bash
curl 'http://127.0.0.1:8000/api/v1/stations/0074647403'
```

`GET /api/v1/stations/{station_id}` trả các trường thông tin như trong danh sách,
thêm `enabled` và `alarm_levels_m` (`level_1`, `level_2`, `level_3`, đơn vị mét).
Ngưỡng báo động được đọc đúng từ database; nếu chưa cấu hình thì trả null.
Endpoint cho phép xem cả trạm đã tắt, nhưng `prediction_enabled` luôn là false
đối với trạm đó. Mã trạm không tồn tại trả 404 `STATION_NOT_FOUND`;
lỗi database trả 503. Endpoint không tải mô hình hay chạy dự báo.

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
chạy inference hoặc nhận override. Override được gửi qua endpoint `POST /api/v1/predictions`.

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


## API chạy dự báo

```text
POST /api/v1/predictions
```

```bash
curl 'http://127.0.0.1:8000/api/v1/predictions' \
  -H 'Content-Type: application/json' \
  -d '{"station_id":"0074647403","reference_time":"2026-09-20T14:30:00+07:00","overrides":{"water_level_depth":700.0}}'
```

- Chỉ `station_id` là bắt buộc. Bỏ `reference_time` hoặc gửi null để dự báo từ
  đầu vào mới nhất. Bỏ `overrides` để giữ nguyên các feature quan trắc.
- Override dùng đúng tên feature và đơn vị gốc của form, ví dụ `water_level_depth`
  tính bằng mm. Chỉ sửa timestep cuối, qua đúng preprocessing đã lưu.
- Feature không được sửa giữ nguyên; không tự tính lại các feature liên quan khi
  người dùng sửa một feature. Mưa/lag có thể nhận null, cờ chỉ nhận số 0/1 và
  mực nước hiện tại không được thiếu. Cho phép mực nước âm như pipeline training.
- Không nhận tên feature lạ, chuỗi thay cho số, boolean, NaN/Infinity, số vượt
  float32 hoặc trường request ngoài schema. Override không hợp lệ trả 422
  `INVALID_OVERRIDE`; lỗi kiểu dữ liệu trả `INVALID_REQUEST`.
- Mỗi worker tải model/scaler một lần cho từng trạm khi có yêu cầu dự báo đầu tiên.
  Mô hình khác trạm không dùng chung trọng số. Khi thay artifact/config, khởi động
  lại backend và cập nhật model version. Không cache kết quả hay override ở bước này.
- `PREDICTION_DEVICE` mặc định `cpu`; có thể đặt `cuda` nếu máy hỗ trợ. Hiện đã
  kiểm chứng trên CPU. Model chạy eval/inference mode, dropout tắt và không có
  correction penalty, optimizer hoặc training.

Response giữ các mốc thời gian và model/schema version, có `unit="mm"`,
`overrides_applied` và danh sách `predictions`. Mỗi phần tử gồm
`horizon_minutes`, `forecast_at`, `water_level_mm`. Các horizon lấy từ cấu hình;
Dã Viên hiện là 60/180/360/720/1440 phút. Thời điểm dự báo tính từ anchor thực tế,
không từ mốc 14:30 người dùng nhập. Các trường `is_stale`, `input_age_minutes`,
`stale_after_minutes` theo cùng quy tắc 2 giờ của API đầu vào.

Không ghi override hoặc dự báo vào PostgreSQL. `MODEL_UNAVAILABLE` (503) báo lỗi
load model/scaler; `PREDICTION_UNAVAILABLE` (503) báo không tạo được dự báo cho
đầu vào. Các lỗi trạm, database và thiếu lịch sử giữ cùng định dạng ở trên.

Script kiểm chứng local: `scripts/local/verify_predictions_api.py`. Script đối
chiếu latest, historical và override với artifact mô hình đã lưu, kiểm tra form
không bị thay đổi sau override; không train hoặc ghi database.
