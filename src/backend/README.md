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

## API quan trắc mới nhất

```bash
curl 'http://127.0.0.1:8000/api/v1/stations/0074647403/latest'
```

`GET /api/v1/stations/{station_id}/latest` trả `station_id`, `station_type`,
`value`, `unit`, `observed_at`, `received_at`, `quality_status`, `is_stale`,
`data_age_minutes` và `stale_after_minutes`.

- Lấy lần đo hợp lệ mới nhất không nằm trong tương lai. Bỏ qua bản ghi thiếu;
  giá trị 0 hoặc mực nước âm vẫn hợp lệ.
- `value` là giá trị quan trắc gốc, chưa tổng hợp theo giờ hoặc chuẩn hóa.
  Đầu vào mô hình theo giờ vẫn lấy qua `prediction-inputs`.
- Thời gian trả về UTC; mực nước có đơn vị mm, đơn vị mưa chưa xác nhận trả null.
- Tuổi dữ liệu tính từ `observed_at` đến lúc xử lý request. Dữ liệu trễ hơn
  120 phút được trả kèm `is_stale = true`; đúng 120 phút vẫn chưa cũ.
  Ngưỡng dùng chung cấu hình `INPUT_STALE_AFTER_MINUTES` với API đầu vào.
- Trạm đã tắt vẫn đọc được quan trắc. Trạm không tồn tại trả 404
  `STATION_NOT_FOUND`; không có lần đo hợp lệ trả 404 `OBSERVATION_NOT_FOUND`.
  Lỗi database trả 503 theo định dạng lỗi chung.

## API lịch sử quan trắc

```bash
curl --get 'http://127.0.0.1:8000/api/v1/stations/0074647403/observations' \
  --data-urlencode 'start=2026-09-13T00:00:00+07:00' \
  --data-urlencode 'end=2026-09-14T00:00:00+07:00' \
  --data-urlencode 'limit=100'
```

- `start`, `end` bắt buộc có múi giờ; lấy khoảng `[start, end)` (có start,
  không có end). `start` phải trước `end`; `end` không được trong tương lai.
- Trả quan trắc **gốc từ mới đến cũ**, không tổng hợp theo giờ hoặc nội suy.
  Bản ghi thiếu giữ `value = null`, `quality_status = missing`; giờ không có
  bản ghi không được tự thêm vào. Giá trị 0 và mực nước âm được giữ nguyên.
- `limit` mặc định 100, tối đa 1000. Response có `observations`, `has_more`,
  `next_before`. Nếu còn dữ liệu, gửi `before=next_before` cùng khoảng start/end
  để đọc trang kế tiếp. Cursor loại trừ dòng cuối trang trước, tránh lặp dòng.
- Mỗi dòng có `observed_at`, `received_at`, `value`, `quality_status`.
  Thời gian trả UTC; đơn vị và loại trạm nằm ở cấp response (`unit`, `station_type`).
- Trạm tồn tại nhưng khoảng chọn không có dữ liệu trả 200 với danh sách rỗng.
  Trạm không tồn tại trả 404; tham số sai trả 422; lỗi database trả 503.
- Khi người dùng chọn dòng lịch sử, gọi `prediction-inputs` bằng timestamp đó
  để lấy đủ feature theo giờ. Giá trị gốc trên bảng có thể khác giá trị theo giờ
  trong form. API lịch sử không xác nhận thời điểm có đủ dữ liệu để dự báo.

## API kiểm tra thời điểm có thể dự báo

```bash
curl --get 'http://127.0.0.1:8000/api/v1/stations/0074647403/prediction-availability' \
  --data-urlencode 'start=2026-09-13T00:00:00+07:00' \
  --data-urlencode 'end=2026-09-14T00:00:00+07:00'
```

- `start`, `end` bắt buộc có múi giờ, `start < end`, không quá 7 ngày và
  `end` không trong tương lai. API kiểm tra các mốc đúng giờ trong `[start, end)`,
  trả theo thứ tự tăng dần. Ví dụ từ 09:30 đến 11:00 chỉ kiểm tra 10:00.
- Response có phiên bản model/schema và `sequence_length` lấy từ cấu hình riêng
  của trạm, được kiểm tra khớp metadata preprocessing. Không cố định 96 cho mọi trạm.
- Mỗi dòng trong `availability` có `reference_time`, `input_ready`,
  `available_timesteps` (tối đa bằng sequence length), `anchor_observed_at`, `reason`.
  Thiếu lịch sử trả `input_ready = false`, `reason = INSUFFICIENT_HISTORY`.
- Dùng đúng `PredictionInputService.prepare()` như API đầu vào, cho phép gaps
  và thiếu mưa/lag giống training. Không tải trọng số hoặc chạy dự báo.
  Có thể đủ lịch sử nhưng anchor cũ; `input_ready` không có nghĩa dữ liệu còn mới.
- Mã trạm không tồn tại trả 404; trạm chưa bật dự báo/model preprocessing không
  khả dụng trả 503; tham số sai trả 422. POST dự báo vẫn phải kiểm tra lại vì
  dữ liệu có thể thay đổi sau khi đọc availability.
- Giới hạn 7 ngày nhằm giữ chi phí kiểm tra từng giờ ở mức hữu hạn. Frontend
  có thể gọi từng khoảng khi người dùng chuyển thời gian xem.

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

API dùng `forecasts` và `forecast_points` trong PostgreSQL (migration `0003`)
để lưu và dùng lại dự báo từ quan trắc gốc. Cần migrate trước khi chạy API.
Chi tiết ràng buộc nằm trong `db/README.md`.

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
  Mô hình khác trạm không dùng chung trọng số. Request tiếp theo kiểm tra SHA-256
  checkpoint/scaler; nếu file bị thay thế thì tải lại runtime. Khi đổi code/config,
  khởi động lại backend và cập nhật model version. Redis chưa được triển khai.
- `PREDICTION_DEVICE` mặc định `cpu`; có thể đặt `cuda` nếu máy hỗ trợ. Hiện đã
  kiểm chứng trên CPU. Model chạy eval/inference mode, dropout tắt và không có
  correction penalty, optimizer hoặc training.

Response giữ các mốc thời gian và model/schema version, có `unit="mm"`,
`overrides_applied` và danh sách `predictions`. Mỗi phần tử gồm
`horizon_minutes`, `forecast_at`, `water_level_mm`. Các horizon lấy từ cấu hình;
Dã Viên hiện là 60/180/360/720/1440 phút. Thời điểm dự báo tính từ anchor thực tế,
không từ mốc 14:30 người dùng nhập. Các trường `is_stale`, `input_age_minutes`,
`stale_after_minutes` theo cùng quy tắc 2 giờ của API đầu vào.

### Lưu và dùng lại dự báo

- Request không có overrides: chuẩn bị đầy đủ đầu vào, tính hash, tìm bản khớp
  trong PostgreSQL. Nếu có thì dùng lại; nếu chưa có thì chạy model và lưu cha
  cùng tất cả horizon trong một transaction.
- Hash gồm toàn bộ giá trị gốc, mask thiếu, tensor đã chuẩn hóa, mọi timestamp,
  feature order, anchor và horizon. Không chỉ dùng thời điểm hoặc timestep cuối.
  Pipeline identity gồm checkpoint/scaler, cấu hình model, mã nguồn pipeline và
  device. Khi đổi đầu vào hoặc pipeline, bản cũ không được dùng thay cho bản mới.
- Bất kỳ overrides không rỗng nào cũng là kịch bản riêng: không đọc hoặc ghi bảng
  dự báo, kể cả khi người dùng nhập lại đúng giá trị quan trắc. Không sửa quan trắc.
- Response thêm `forecast_id`, `generated_at`, `pipeline_version`, `reused`.
  `reused = true` nghĩa là đọc lại dự báo đã lưu; `generated_at` giữ thời điểm lưu
  kết quả đầu tiên. Kịch bản overrides có ba trường metadata trên là null.
- Mốc yêu cầu, tuổi dữ liệu và `is_stale` luôn theo request hiện tại; không lấy
  trạng thái cũ từ bản đã lưu. Khác mốc yêu cầu nhưng chọn cùng chuỗi vẫn dùng lại được.
- Nếu hai request cùng lưu, unique constraint giữ một bản và request còn lại đọc
  bản thắng. Có thể cả hai vẫn chạy inference khi cùng miss; chưa có cơ chế gộp
  các inference đang chạy hoặc Redis. Lỗi ghi DB trả 503, không trả thành công giả.

`MODEL_UNAVAILABLE` (503) báo lỗi
load model/scaler; `PREDICTION_UNAVAILABLE` (503) báo không tạo được dự báo cho
đầu vào. Các lỗi trạm, database và thiếu lịch sử giữ cùng định dạng ở trên.

Script local `scripts/local/verify_persisted_predictions_api.py` kiểm tra dùng lại
historical/latest, không chạy inference khi đã lưu và không lưu overrides.
Script lưu dự báo gốc thật vào PostgreSQL, không thay đổi dữ liệu quan trắc.
