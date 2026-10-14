# Backend thủy văn

Backend hiện có database PostgreSQL, chuẩn bị đầu vào, runtime Residual LSTM và
API lấy dữ liệu để tự điền form và API chạy dự báo.
Mỗi trạm nước có model/scaler riêng, được chọn bằng `station_id`.

## Chạy tại máy local

Chạy từ thư mục gốc dự án:

```bash
.venv/bin/python -m pip install -r src/backend/requirements.txt
.venv/bin/python -m uvicorn src.backend.main:app --env-file .env --host 127.0.0.1 --port 8000 --workers 1 --no-proxy-headers
```

`.env` chứa `DATABASE_URL` của PostgreSQL đã migrate và có dữ liệu. Không đưa mật
khẩu vào Git. Ứng dụng không tự tạo bảng, nhập dữ liệu hoặc gọi API nguồn khi khởi động.

Mặc định các đường dẫn model/scaler trong registry được tính từ thư mục gốc dự án.
Có thể đặt `MODEL_ARTIFACT_ROOT` và `MODEL_CONFIG_PATH` khi triển khai ở vị trí khác.
Các file artifact phải là file tin cậy của mô hình tương ứng.

- Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI: `http://127.0.0.1:8000/openapi.json`

## Bảo vệ API

Đã có [CORS và giới hạn request/body/inference](security/README.md). Mặc định
120 request/phút/IP, riêng Predict 20/phút/IP; body tối đa 16 KiB, thời gian nhận
body 10 giây, và một request dự báo đồng thời trong mỗi process. Request bị chặn
nhận 403/408/413/415/429 hoặc 503 với mã lỗi rõ ràng; 429 và inference bận có
`Retry-After`. Redis lỗi vẫn giữ giới hạn local. Cấu hình chỉnh bằng environment.

Frontend local được phép ở localhost/127.0.0.1:5173; Swagger local ở port 8000.
Khi triển khai cần đặt `CORS_ORIGINS` đúng domain và cấu hình trusted proxy đúng.
Chạy một API process cho GPU hiện tại; không tự tăng số workers để tăng throughput.

## Worker cập nhật dữ liệu tự động

Pipeline nằm trong [`ingestion/`](ingestion/README.md), chạy như process riêng
với API để tránh nhân bản lịch đồng bộ theo số Uvicorn workers. Mặc định mỗi
**11 phút** worker lấy dữ liệu mới của các trạm nguồn được bật và UPSERT vào
PostgreSQL. API đọc dữ liệu đã lưu từ database.

```bash
# Kiểm tra nguồn, không ghi quan trắc
.venv/bin/python -m src.backend.ingestion --once --dry-run
# Chạy worker liên tục
.venv/bin/python -m src.backend.ingestion
```

Worker dùng `DATABASE_URL`, `HYDROLOGY_API_BASE_URL`, `HYDROLOGY_API_EMAIL` và
`HYDROLOGY_API_PASSWORD` từ environment hoặc `.env` private. Xem hướng dẫn trong
`ingestion/README.md` để cấu hình chu kỳ, report và systemd service. Mật khẩu,
báo cáo runtime và các công cụ nhập lịch sử local không đưa vào code chính.

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

Endpoint này đã có [Redis cache](cache/README.md) cho giá trị quan trắc, TTL mặc
định 600 giây. Tuổi dữ liệu được tính lại mỗi request; thông tin trạm vẫn đọc từ
PostgreSQL. Worker xóa cache sau khi thêm/sửa quan trắc. Redis lỗi thì đọc database.

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
- Thời gian trả về UTC; mực nước và bốn trạm mưa hiện tại dùng mm theo quyết định
  chủ dự án (migration `0004`). Đơn vị chưa biết của trạm khác vẫn trả null.
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

## API bảng lịch sử đúng mốc giờ

`GET /api/v1/observations/hourly` phục vụ bảng tổng nhiều trạm. Mỗi dòng là một
trạm trong một ngày Việt Nam, có đúng 24 ô từ 00:00 đến 23:00. Đây là **bản ghi
đúng mốc giờ**, không phải trung bình nước hoặc tổng mưa trong giờ.

```bash
# Mặc định hôm nay theo Asia/Ho_Chi_Minh, tất cả trạm được bật
curl 'http://127.0.0.1:8000/api/v1/observations/hourly'

# Một ngày và một trạm; giữ nguyên số 0 đầu trong mã Dã Viên
curl --get 'http://127.0.0.1:8000/api/v1/observations/hourly' \
  --data-urlencode 'start_date=2026-10-13' \
  --data-urlencode 'station_id=0074647403'

# Khoảng ngày tính cả ngày đầu và cuối, lọc trạm mưa theo tên
curl --get 'http://127.0.0.1:8000/api/v1/observations/hourly' \
  --data-urlencode 'start_date=2026-10-01' \
  --data-urlencode 'end_date=2026-10-13' \
  --data-urlencode 'station_type=rainfall' \
  --data-urlencode 'station_name=Hồng Thái' \
  --data-urlencode 'limit=50'
```

| Tham số | Ý nghĩa |
|---|---|
| `start_date`, `end_date` | Ngày `YYYY-MM-DD`, tính cả hai đầu, từ 1900-01-01 đến hôm nay. Bỏ cả hai chọn hôm nay; chỉ gửi một ngày chọn đúng ngày đó. |
| `station_type` | Tùy chọn `water_level` hoặc `rainfall`; bỏ để lấy tất cả loại hiện hỗ trợ. |
| `station_id` | Tùy chọn, khớp chính xác mã trạm; dùng khi chọn tên trạm từ danh sách. |
| `station_name` | Tùy chọn, tìm tên chứa chuỗi, không phân biệt hoa/thường; giữ dấu tiếng Việt. `%` và `_` là ký tự tìm kiếm thông thường. |
| `limit` | Số dòng trạm/ngày mỗi trang: mặc định 50, tối đa 100. |
| `cursor_date`, `cursor_station_id` | Luôn gửi cùng nhau để đọc tiếp sau dòng cuối của trang trước. |

- Response có `timezone`, `start_date`, `end_date`, `as_of`, `rows`, `has_more`,
  `next_cursor`. `as_of` và timestamp ô dùng UTC; ngày dòng/cột giờ dùng giờ Việt Nam.
- Mỗi row có `date`, `station_id`, `name`, `station_type`, `unit`, `cells`.
  Các ô sắp tăng theo giờ, mỗi ô gồm `hour` (0–23), `at` (mốc dự kiến, không khẳng
  định đã có quan trắc), `value` và `status`:
  - `observed`: có bản ghi hợp lệ đúng timestamp; giữ nguyên số 0 và mực nước âm.
  - `missing`: không có bản ghi đúng mốc hoặc bản ghi thiếu; `value = null`, UI hiện “—”.
  - `future`: giờ chưa tới tại `as_of`; `value = null`, UI có thể ghi “Chưa tới giờ”.
- Không lấy bản ghi gần nhất, làm tròn timestamp, nội suy, cộng mưa hoặc tính trung bình.
  Ví dụ có 01:10 nhưng không có 01:00 thì ô 01:00 vẫn thiếu. Đơn vị lấy từ metadata.
- Chỉ lấy trạm `enabled=true`; mọi bộ lọc kết hợp theo AND. Không khớp trạm trả
  200 với `rows=[]`. Trạm/ngày chưa có quan trắc vẫn có dòng với các ô thiếu.
- Sắp ngày giảm dần, mã trạm tăng dần theo thứ tự ký tự ổn định. Nếu `has_more=true`,
  dùng `next_cursor.date` và `next_cursor.station_id` cho hai tham số cursor, giữ
  nguyên bộ lọc và gửi lại `start_date`/`end_date` đã được resolve trong response.
  Điều này cũng giữ đúng khoảng ngày khi phân trang qua thời điểm nửa đêm.
- Sai ngày/cursor/limit hoặc tham số lạ trả 422; PostgreSQL lỗi trả 503 an toàn.
  API chỉ đọc, dùng giới hạn request hiện có, không tải model hoặc tạo dự báo.
- Backend chỉ tạo tối đa `limit + 1` dòng ứng viên và tra tối đa 24 × `limit`
  khóa `(station_id, observed_at)` của trang đó bằng primary key hiện có. Không tải
  cả khoảng dữ liệu 10 phút; không cần migration hay cache riêng.

Kiểm thử hợp đồng bằng dữ liệu giả:

```bash
.venv/bin/python -m unittest src.backend.api.tests.test_hourly_history -v
```

Kiểm thử SQL trên PostgreSQL dùng bảng TEMP riêng, rollback khi kết thúc:

```bash
# DATABASE_URL cần có trong environment; không đưa URL có mật khẩu vào Git.
.venv/bin/python -m unittest src.backend.db.tests.test_hourly_history -v
```

Khởi động lại API process sau khi cập nhật code để nạp route mới; worker ingestion
không cần khởi động lại. Frontend đã dùng route này cho bảng lịch sử tổng;
xem [README frontend](../frontend/README.md#bảng-lịch-sử-quan-trắc).

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
  bốn trạm mưa hiện tại dùng mm. Đơn vị chưa biết vẫn trả `null`; không chuyển đổi giá trị.
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


## API đọc dự báo đã lưu và thực tế

```bash
curl --get 'http://127.0.0.1:8000/api/v1/stations/0074647403/forecasts' \
  --data-urlencode 'start=2026-09-13T00:00:00+07:00' \
  --data-urlencode 'end=2026-09-15T00:00:00+07:00' \
  --data-urlencode 'horizon_minutes=60' \
  --data-urlencode 'limit=100'
```

`GET /api/v1/stations/{station_id}/forecasts` phục vụ một đường biểu đồ so sánh:

- `start`, `end` có múi giờ, lọc theo **forecast_at** trong `[start, end)`,
  không theo anchor hoặc ngày chạy replay. `horizon_minutes` là bắt buộc và dương.
  Frontend gọi riêng từng horizon, không nối các horizon thành cùng một đường.
- Mặc định chọn model trong registry và pipeline tính từ code/config/artifact/
  `PREDICTION_DEVICE` hiện tại. Frontend không truyền hoặc cho người dùng chọn
  device. Để đọc lượt backfill GPU, backend dùng `PREDICTION_DEVICE=cuda`.
- Có thể truyền `model_version` và `pipeline_version` để đọc một series đã lưu
  thuộc phiên bản cũ. Nếu model khác model hiện tại thì phải chỉ rõ pipeline.
  Cặp phiên bản tường minh không yêu cầu file model/scaler cũ còn tồn tại.
- Một response chỉ chứa một model, pipeline và horizon. Nếu cùng thời điểm có
  nhiều lần replay do input được sửa, lấy bản có `generated_at` mới nhất;
  bản cũ vẫn giữ trong DB. Nếu thời điểm tạo trùng nhau, ID dùng để chọn ổn định.
- `points` gồm `forecast_id`, `feature_schema_version`, `anchor_observed_at`,
  `forecast_at`, `generated_at`, `predicted_water_level_mm`,
  `actual_water_level_mm`, `actual_sample_count`.
- Thực tế là trung bình các lần đo hợp lệ tại **chính trạm nước đó** trong
  `(forecast_at - 1 giờ, forecast_at]`, cùng cách tổng hợp nước khi training.
  Không dùng dữ liệu sau forecast_at, không nội suy hoặc thay thiếu bằng 0.
  Có thể chỉ có một phần số lần đo trong giờ; `actual_sample_count` cho biết số
  giá trị thực sự tham gia trung bình. Đây không phải cờ xác nhận giờ đầy đủ.
- Giờ không có lần đo hợp lệ hoặc forecast_at còn trong tương lai trả
  `actual_water_level_mm = null`, `actual_sample_count = 0`. Cho phép end ở tương
  lai để hiển thị các điểm dự báo đã lưu nhưng chưa đến thời điểm quan trắc.
- Thời gian trả UTC, đơn vị mm. Điểm được sắp từ mới đến cũ; frontend có thể đảo
  thứ tự khi vẽ. `limit` mặc định 100, tối đa 1000. Khi `has_more = true`, truyền
  `before=next_before` cùng khoảng thời gian và cặp phiên bản trả về để đọc trang
  tiếp theo mà không đổi series giữa các request.
- Trạm/series tồn tại nhưng không có điểm khớp trả danh sách rỗng. Mã trạm không
  tồn tại trả 404; trạm mưa/params không hợp lệ trả 422; lỗi database, đơn vị chưa
  xác nhận mm hoặc không xác định được pipeline mặc định trả 503.

Endpoint chỉ đọc DB và đọc bytes artifact khi cần xác định pipeline mặc định;
không deserialize model/scaler, không chạy inference hoặc ghi thêm dự báo.
Quan trắc dùng để đối chiếu là dữ liệu hiện tại trong DB, có thể đã được nguồn
chỉnh sửa sau lúc tạo dự báo. `generated_at` giúp phân biệt replay hiện tại với
một dự báo thực sự đã phát trong quá khứ.

Kiểm chứng local: `scripts/local/verify_forecast_history_api.py` đọc 240 điểm tại
5 horizon, kiểm tra phân trang và đối chiếu với các trung bình giờ tính riêng.

## API Visualize một mốc

```bash
curl 'http://127.0.0.1:8000/api/v1/forecasts/3923638a-f637-41ad-afdd-2443b9ec726e'
```

`GET /api/v1/forecasts/{forecast_id}` đọc đúng bản dự báo được chọn. Lấy ID từ
response `POST /predictions` hoặc mỗi điểm trong API lịch sử `/forecasts`.
ID trong ví dụ thuộc lượt backfill local; hệ thống khác dùng ID của database đó.

- Response có `forecast_id`, trạm, model/schema/pipeline, `anchor_observed_at`,
  `generated_at`, `window_end`, `unit = mm`, `predictions` và `actuals`.
- `predictions` giữ nguyên các horizon của bản lưu, sắp tăng dần. Mỗi điểm có
  `horizon_minutes`, `forecast_at`, `water_level_mm`. Model Dã Viên hiện có 5 điểm
  +1/+3/+6/+12/+24 giờ; không tạo thêm dự báo cho các giờ ở giữa.
- `actuals` có `timestamp`, `actual_water_level_mm`, `actual_sample_count`.
  Tạo một mốc mỗi giờ từ anchor + 1 giờ đến horizon lớn nhất, tính trung bình trong
  `(timestamp - 1 giờ, timestamp]`. Dã Viên có đúng 24 mốc; không cố định con số này
  cho model khác. Nếu horizon cuối không tròn giờ, chỉ tạo các mốc giờ nằm trong cửa sổ.
- Giờ thiếu hoặc chưa tới thời điểm đo vẫn có trong danh sách, với giá trị null và
  count 0. Không thay thiếu bằng 0, không nội suy. Frontend không nối đường qua
  null như thể đã có quan trắc. Giá trị 0 hoặc mực nước âm thật được giữ nguyên.
- Đọc theo ID không tự chọn lại model/device hoặc bản replay mới hơn. Bản cũ vẫn
  đọc được nếu không còn trong registry hoặc artifact đã chuyển chỗ. Quan trắc
  thực tế dùng dữ liệu hiện tại trong DB, có thể đã được nguồn sửa sau lúc dự báo.
- Không chạy inference, không tải model/scaler và không ghi database. ID không có
  trả 404 `FORECAST_NOT_FOUND`; UUID sai trả 422. Bản lưu không có horizon hoặc
  có thời điểm horizon sai trả 503 `STORED_FORECAST_INVALID`; đơn vị chưa xác nhận
  mm trả 503 `UNIT_UNAVAILABLE`; lỗi database trả 503 theo định dạng lỗi chung.

Luồng frontend: chọn mốc và tự điền input → Visualize đọc ID đã có. Nếu chưa có
dự báo gốc phù hợp, gọi POST dự báo để tạo/lấy ID rồi gọi endpoint này. Kịch bản
overrides không có forecast_id; nếu vẽ thì dùng response Predict và gắn nhãn
kịch bản, không coi là bản dự báo gốc đã lưu.

Kiểm chứng local: `scripts/local/verify_forecast_detail_api.py` đối chiếu ba mốc
GPU với các điểm đã lưu và từng trung bình giờ từ quan trắc gốc, ở chế độ chỉ đọc.

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
