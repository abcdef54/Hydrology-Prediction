# Quy trình chung: từ API quan trắc đến model dự báo mực nước một trạm

Tài liệu này mô tả phần **dùng chung cho mọi trạm**: lấy dữ liệu, dựng dataset, kiến trúc model, hàm loss, vòng lặp train và cách đánh giá. Các con số và lệnh cụ thể của từng trạm nằm trong thư mục của trạm đó (ví dụ `models/thuan_an/REPRODUCE.md`).

Người đọc được cung cấp thư mục `src/` và 2 file Postman (`data/HueNgapLut.postman_collection.json`, `data/HueNgapLut.postman_environment.json`). Mật khẩu và token trong file environment là **giả**; cần xin tài khoản thật từ đơn vị quản lý API. Mọi dữ liệu khác người đọc phải tự lấy: quan trắc từ API DCU, dự báo thời tiết từ Open-Meteo.

```
API DCU (Postman) ──► PostgreSQL ──► mực nước theo giờ ─┐
Open-Meteo Previous Runs (gió ECMWF) ───────────────────┼─► dataset (src/ai/station_dataset/build.py) ──► train (src/ai/train.py)
Open-Meteo Single Runs (mưa dự báo ECMWF) ──────────────┘                                                    │
                                    chấm điểm từng epoch, chọn epoch, ensemble (src/ai/station_dataset/evaluate.py) ◄──┘
```

| Bước | Code trong `src/` |
|---|---|
| Lấy quan trắc từ API vào PostgreSQL | `src/backend/db` (schema, migration), `src/backend/ingestion/station_catalog.py`, `api_backfill.py`, `worker.py` |
| Tải dữ liệu Open-Meteo | `src/ai/station_dataset/open_meteo.py` |
| Đặc trưng gió, triều, mưa dự báo | `src/ai/station_dataset/features.py` |
| Dựng dataset | `src/ai/station_dataset/build.py` (dùng `src/ai/regional_data/preprocessing.py`) |
| Đọc dataset khi train | `src/ai/regional_data/reader.py`, `src/ai/regional_training_data.py`, `src/ai/regional_data/training_selection.py` |
| Model | `src/ai/residual_sequence.py` (`ResidualGRU`) |
| Loss | `src/ai/change_objective.py` (`ChangeRiskLoss`), `src/ai/losses.py` |
| Train | `src/ai/train.py`, `src/ai/trainers.py`, `src/ai/lr_schedules.py`, `src/ai/utils.py` |
| Chấm điểm và ensemble | `src/ai/station_dataset/evaluate.py`, `src/ai/sequence_checkpoint.py` |

---

## 0. Môi trường

- Python 3.14. Các gói đã dùng: torch 2.14.1 (CUDA 13.0), numpy 2.5.3, pandas 3.0.6, pyarrow 25.0.1, scikit-learn 1.9.1, joblib 1.6.0, tqdm 4.70.1, mlflow 3.16.1, SQLAlchemy 2.1.3, psycopg 3.3.6, alembic 1.20.0, python-dotenv 1.2.4, requests 2.34.2, redis 6.4.0 (không bắt buộc). Phần backend có sẵn `src/backend/requirements.txt`.
- PostgreSQL. Redis không bắt buộc: nếu không có `REDIS_URL` thì bỏ qua cache.
- GPU NVIDIA. Mọi lệnh chạy từ thư mục gốc repo (thư mục chứa `src/`).
- **Train tất định:** `train.py` đặt `CUBLAS_WORKSPACE_CONFIG=:4096:2` trước khi import torch. `set_seed` (`src/ai/utils.py`) seed `random`, `numpy` và `torch`, bật `torch.use_deterministic_algorithms(True)` và cuDNN deterministic. Cùng GPU và cùng phiên bản thư viện thì hai lần train cho kết quả giống từng bit. Khác GPU hoặc khác phiên bản CUDA/torch thì có thể lệch nhỏ.
- **MLflow:** mặc định `train.py` gửi log tới `http://localhost:5000`. Có thể chạy một server (`mlflow server --backend-store-uri sqlite:///mlflow_local.db --port 5000`), hoặc thêm `--mlflow-tracking-uri sqlite:///mlflow_local.db` vào lệnh train. MLflow chỉ dùng để ghi log, không ảnh hưởng kết quả. **Lưu ý:** các experiment `nga_ba_sinh` và `thuan_an` trong `mlflow_local.db` được tạo qua server, nên artifact nằm ở `mlflow-artifacts:/…`. Ghi vào chúng **bắt buộc** phải qua server (`http://localhost:5000`). Dùng `--mlflow-tracking-uri sqlite:///…` sẽ lỗi ở bước `log_artifact`.

File `.env` ở thư mục gốc, **không commit**:
```bash
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST:5432/DATABASE
HYDROLOGY_API_BASE_URL=https://dcu.huecity.vn      # = dcu_base_url trong Postman environment
HYDROLOGY_API_EMAIL=...                            # = dcu_email
HYDROLOGY_API_PASSWORD=...                         # mật khẩu thật (bản trong Postman là giả)
```

---

## 1. Quan trắc: từ API DCU vào PostgreSQL

### 1.1 API (xem collection Postman)
- **Đăng nhập:** `POST {{dcu_base_url}}/auth/login` với body `{"email": ..., "password": ...}`. Kết quả nằm ở `data.access_token` (thêm `refresh_token`). Mọi request sau gửi header `Authorization: Bearer <access_token>`. Khi gặp HTTP 401 thì đăng nhập lại.
- **Danh mục trạm:** `GET /items/water_level_station` (trạm mực nước), `GET /items/rain_water_stations` (trạm mưa).
- **Quan trắc mực nước:** `GET /items/water_level_depth`. Mỗi bản ghi có `id` (số nguyên tăng dần), `station_id` (mã trạm, giữ số 0 ở đầu), `time_point` (ISO 8601 có múi giờ) và `depth` (mm, có thể `null`). Mưa dùng `/items/rain_water_depth`, cùng cấu trúc.
- **Phân trang:** lọc `filter[station_id][_in]=<mã>`, `filter[time_point][_gte]=<từ>`, `filter[time_point][_lte]=<đến>`. Duyệt theo con trỏ id với `filter[id][_gt]=<id cuối>`, `sort[]=id`, `limit=500`, lặp cho đến khi trang trả về rỗng. Không phân trang theo `time_point`, vì có thể bỏ sót các bản ghi trùng thời điểm ở ranh giới trang.

Đoạn code tương đương, minh họa cách client `HueSourceClient` trong `src/backend/ingestion/worker.py` gọi API:
```python
s = requests.Session()
tok = s.post(f"{base}/auth/login", json={"email": email, "password": pw}).json()["data"]["access_token"]
s.headers["Authorization"] = f"Bearer {tok}"
cursor, rows = 0, []
while True:
    page = s.get(f"{base}/items/water_level_depth", params={
        "filter[station_id][_in]": "0074647402", "filter[time_point][_gte]": "2025-11-01T00:00:00+00:00",
        "filter[time_point][_lte]": "2025-11-08T00:00:00+00:00", "filter[id][_gt]": cursor, "sort[]": "id", "limit": 500}).json()["data"]
    if not page: break
    rows += page; cursor = page[-1]["id"]
```

### 1.2 Lưu vào PostgreSQL
```bash
.venv/bin/python -m alembic -c src/backend/db/alembic.ini upgrade head        # tạo schema hydrology
.venv/bin/python -m src.backend.ingestion.station_catalog                      # dry run: đối chiếu danh mục trạm từ API
.venv/bin/python -m src.backend.ingestion.station_catalog --apply              # ghi hydrology.stations
.venv/bin/python -m src.backend.ingestion.api_backfill --station <mã> --start <ISO> --end <ISO>           # dry run
.venv/bin/python -m src.backend.ingestion.api_backfill --station <mã> --start <ISO> --end <ISO> --apply   # ghi lịch sử
.venv/bin/python -m src.backend.ingestion --once                               # đồng bộ một lượt tới hiện tại (worker)
```
- `api_backfill` đọc từng cửa sổ 7 ngày, kiểm tra từng bản ghi, rồi ghi vào bảng `hydrology.observations` bằng UPSERT theo cặp (trạm, thời điểm). Mỗi dòng gồm `observed_at` (UTC), `depth` (đơn vị gốc), `quality_status` (`observed`, hoặc `missing` nếu `depth` null) và `source_record_id`. Chạy lại cùng một khoảng thời gian không tạo dòng trùng.
- Worker (`python -m src.backend.ingestion`) mặc định đồng bộ mỗi 11 phút. Mỗi lượt nó đọc lại 24 h trước bản ghi mới nhất của từng trạm, vì vậy chỉ dùng để cập nhật dữ liệu mới, không dùng để nạp lịch sử.

### 1.3 Mực nước theo giờ (quy tắc dùng khi train)
Tại mỗi mốc giờ t (UTC), mực nước là **trung bình các quan trắc có `observed_at` trong khoảng (t − 1 h, t]**, chỉ tính các bản ghi có `quality_status = observed` và `depth` hữu hạn. Nếu trong giờ đó có hai bản ghi trùng thời điểm thì để trống giờ đó. Đơn vị phải là mm. Nhịp quan trắc chuẩn là 10 phút (6 bản ghi mỗi giờ), nhưng chỉ cần có 1 bản ghi hợp lệ là giờ đó có giá trị. Hàm thực hiện: `src/ai/station_dataset/build.py: hourly_level_from_db()` và `hourly_level()`.

---

## 2. Dữ liệu dự báo thời tiết từ Open-Meteo (không cần API key)

Code: `src/ai/station_dataset/open_meteo.py`.

### 2.1 Gió và áp suất dự báo: Previous Runs API (https://open-meteo.com/en/docs/previous-runs-api)
```bash
.venv/bin/python -m src.ai.station_dataset.open_meteo previous-runs --lat <lat> --lon <lon> \
    --start <YYYY-MM-DD> --end <YYYY-MM-DD> --out data/external/open_meteo/<tên>.json
```
- Câu truy vấn: `models=ecmwf_ifs025`, `hourly=wind_speed_10m,wind_direction_10m,pressure_msl,wind_speed_10m_previous_day1,wind_direction_10m_previous_day1,pressure_msl_previous_day1`, `timezone=UTC`, `wind_speed_unit=ms`.
- **Chỉ dùng các biến `*_previous_day1`**, tức "giá trị đã được dự báo từ 24 h trước giờ hiệu lực". Như vậy khi phát dự báo lúc t, gió cho các giờ t+1…t+12 đến từ những lượt chạy đã khởi tạo trước t − 12 h, nên đã có sẵn tại t.
- Open-Meteo trả về ô lưới gần nhất với tọa độ yêu cầu (`latitude` và `longitude` trong kết quả).

### 2.2 Mưa dự báo ECMWF theo từng lượt chạy: Single Runs API (https://open-meteo.com/en/docs/single-runs-api)
```bash
.venv/bin/python -m src.ai.station_dataset.open_meteo single-runs --points <lat,lon> <lat,lon> ... \
    --start <YYYY-MM-DDTHH:MM> --end <YYYY-MM-DDTHH:MM> --out-dir data/external/open_meteo/ecmwf_single_runs
```
- Mỗi lượt chạy là **một request** cho tất cả các điểm, khởi tạo lúc 00/06/12/18 UTC: `models=ecmwf_ifs`, `run=<giờ khởi tạo>`, `hourly=precipitation`, `forecast_days=3`, `cell_selection=land`, `precipitation_unit=mm`, `timeformat=unixtime`, `timezone=UTC`.
- Mỗi lượt chạy được lưu thành 1 file JSON. Chạy lại lệnh thì bỏ qua các file đã có. Lượt chạy mà API không có (ví dụ `2026-06-10T12:00`) được lưu dưới dạng thông báo lỗi và bỏ qua khi đọc. Một năm dữ liệu gồm khoảng 1.400 request.
- Mã ô lưới: `ecmwf_ifs:<lat>:<lon>`, lấy theo tọa độ mà API trả về, làm tròn 6 chữ số thập phân.
- Đã kiểm tra: các lượt chạy cũ (12/2025, 03/2026, 06/2026, 10/2026) tải lại ngày 12/11/2026 trùng tuyệt đối với bản đã tải để train.

Lúc chạy thật (online), cần áp dụng **đúng quy tắc chọn lượt chạy ở mục 3.4** với dữ liệu thu theo thời gian thực.

---

## 3. Dataset

Lệnh dựng: `python -m src.ai.station_dataset.build --config <dataset_config.json> --name <tên> --groups [fc_ecmwf] [wind] [tide] [rain_obs] --output-dir <thư mục>`. Mọi tham số của trạm nằm trong file config. Các khóa tùy chọn trong config:
- `wind.channels`, `fc_ecmwf.channels`: chỉ giữ một phần kênh của nhóm.
- `wind.onshore_pos`: cắt thành phần gió vào bờ tại 0.
- `tide.window_days`: 7 hoặc 30.
- `rain_obs`: các cụm trạm mưa, cửa sổ cộng dồn, biến đổi; mưa giờ theo đúng quy tắc 6/6 hoặc 5/6 bản ghi ở mục 1.3.
- `sensor_steps`: các lần sensor đổi mốc đo (mục 3.1b).
- `invalid_ranges`: các khoảng dữ liệu hỏng bị xóa hoàn toàn (mục 3.1d).
- `max_sequence_length`: cửa sổ lịch sử dài nhất sẽ dùng khi train trên dataset này. Dùng cho quy tắc xóa mẫu của `invalid_ranges` (mục 3.1d).
- `lin_fc` (nhóm `--groups lin_fc`, đặt sau các nhóm mà nó dùng): dự báo mức thay đổi +3 / +6 / +12 h của một hồi quy tuyến tính (`features.linear_forecast_features`). Đầu vào của hồi quy là lịch sử `history_hours` giờ và các nhóm trong `inputs`. Với các giờ thuộc giai đoạn train, giá trị được tính chéo (`folds` đoạn thời gian liên tiếp, chừa khoảng đệm lịch sử + mốc xa nhất); với val và test, dùng hồi quy fit trên toàn bộ train. Mục đích: GRU bị nén biên độ khi gặp thay đổi lớn hơn lúc train, còn dự báo tuyến tính thì không.

- `soil` (nhóm `--groups soil`): độ ẩm đất ECMWF (`features.soil_moisture_features`) lấy từ thư mục Single Runs tải kèm biến độ ẩm đất (`open_meteo single-runs --hourly precipitation,soil_moisture_0_to_7cm,...`). Chọn lượt chạy theo đúng quy tắc của `fc_ecmwf`, nhưng lượt chạy thiếu độ ẩm đất thì lùi về lượt cũ hơn (ví dụ 18:00 ngày 10/06 và 06:00 ngày 11/06/2026 không có). Kênh: `soil_<tầng>_basin` (trung bình 7 ô), `soil_x_fcrain` (độ ẩm tầng đầu × log1p mưa dự báo 0–24 h của cùng lượt chạy), `soil_missing_frac`.

- `sea` (nhóm `--groups sea`): mực nước biển ngoài khơi do mô hình dự báo (Open-Meteo Marine, `sea_level_height_msl_previous_day1`, tức giá trị dự báo từ khoảng 24 h trước giờ hiệu lực; tải bằng `open_meteo marine --lat --lon --start --end --out`). `features.sea_level_features`: `sea_setup_chg` = trung bình t+1…t+12 trừ trung bình t−23…t; `sea_anom` = giá trị tại t trừ trung bình 7 ngày qua; `sea_next12_anom` = trung bình t+1…t+12 trừ trung bình 7 ngày qua; `sea_missing_frac`. Chỉ dùng tới t+12, cùng quy tắc với gió, để mọi giá trị đều đã được phát ít nhất 12 h trước thời điểm dự báo. Dữ liệu `previous_day1` có từ 18/11/2025.

Định dạng thư mục dataset được `src/ai/regional_data/reader.py` đọc.

### 3.1 Lưới thời gian và mẫu
- **Lưới:** mỗi giờ một mốc (UTC), từ `grid_start_utc` đến `grid_end_utc`. Mỗi mốc giờ t trên lưới có thể là một **thời điểm phát dự báo**.
- **Mẫu hợp lệ** tại t khi: mực nước tại t có giá trị; cửa sổ `sequence_length` giờ (t − L + 1 … t, mặc định L = 96) có **≥ 80% giờ có dữ liệu**; và các nhãn tại t + h (h trong `horizons_hours`) **đều có giá trị**. Nhãn là mực nước theo giờ tại đúng mốc t + h, không nội suy (hàm `make_samples` trong `src/ai/regional_data/preprocessing.py`).
- **Chia tập cố định theo thời gian** với `validation_start`, `validation_end`, `cutoff` và `purge_hours` = 24:
  - train: t < `validation_start` − 24 h
  - validation: `validation_start` ≤ t và t + 24 h < `validation_end`
  - test (thăm dò): t ≥ `cutoff`
  - các mốc còn lại bị loại (purged)
- **Khi train với cửa sổ ngắn hơn** (`--lstm-seq-len` nhỏ hơn L): giữ nguyên tập mẫu, nhưng tính lại điều kiện ≥ 80% giờ có dữ liệu trên cửa sổ mới (`training_selection.select_samples`).

### 3.1b Lỗi sensor đổi mốc đo (`sensor_steps` trong config)
Đôi khi sensor của trạm bị đổi mốc đo, ví dụ khi lắp lại hoặc chỉnh lại thiết bị. Khi đó mực nước nhảy bậc đột ngột rồi giữ nguyên mức mới, dù không có nguyên nhân thủy văn nào.
- **Cách nhận biết:** dữ liệu 10 phút có một bước nhảy lớn, trong khi các trạm lân cận cùng lúc không có biến động tương ứng. Sau đó, chênh lệch giữa trạm này và các trạm lân cận đổi hẳn sang mức mới và **giữ nguyên** nhiều ngày. Khác với gai đơn lẻ: một số đo vọt lên hoặc tụt xuống rồi quay lại ngay ở lần đo kế tiếp.
- **Cách xử lý:** ghi mỗi lần đổi mốc vào `sensor_steps` của config, gồm `time_utc` (thời điểm số đo đầu tiên theo mốc mới) và `note` (bằng chứng).
  - `build.py` loại các mẫu có **giờ hiện tại hoặc một giờ nhãn nằm ở phía bên kia lúc đổi mốc**: thời điểm phát dự báo a thỏa a − 1 h < bước nhảy ≤ a + (mốc dự báo xa nhất). Các mẫu này bị gán `split = excluded_sensor_step`, không thuộc train, val hay test.
  - Lý do loại: nhãn của các mẫu đó chứa một bước nhảy giả, không model nào dự báo được.
- **Không hiệu chỉnh số đo** (không cộng hay trừ mức lệch), vì không biết mốc cũ hay mốc mới mới đúng với thực tế. Model dự báo mức thay đổi so với mực nước hiện tại, nên khi cả cửa sổ lịch sử và nhãn cùng nằm sau lúc đổi mốc, model vẫn dùng được với số đo thô. Đã kiểm chứng ở Ngã Ba Sình: loại thêm các mẫu có cửa sổ lịch sử 12–48 h chứa lúc đổi mốc cũng không thay đổi kết quả.
- Lúc chạy thật, dùng số đo thô như bình thường. Riêng trong 12 h sau lúc đổi mốc, không nên tin vào dự báo.
- **Khi chấm điểm:** `evaluate.predict(..., 'all')` vẫn tính cả các mẫu `excluded_sensor_step`, nên chấm theo `validation` và `test_exploratory`. Dataset dựng trước khi khai báo `sensor_steps` vẫn giữ các mẫu này trong test. Muốn so với run mới thì phải dựng lại dataset, hoặc loại cùng các mẫu đó khi chấm.
- Danh sách các lần đổi mốc đã ghi nhận được ghi trong `REPRODUCE.md` của từng trạm.

### 3.1c Gai số đo (`despike` trong config)
Gai là một hoặc vài số đo 10 phút liên tiếp (thường dưới 1 giờ) lệch hẳn rồi quay về mức cũ, trong khi trạm lân cận không biến động.
- **Quy tắc** (`build.despike_observations`): bỏ số đo lệch hơn `threshold_mm` (150 mm) so với trung vị các số đo hợp lệ trong `window_minutes` (121 phút, đặt giữa). Cửa sổ 2 giờ bắt được cả đoạn nền dài tới khoảng 50 phút.
- Mực nước giờ lấy trung bình các số đo **còn lại** trong giờ. Không dự đoán hay nội suy giá trị thay thế, vì mực nước giờ cũng là nhãn để train và chấm điểm. Database không bị sửa; danh sách số đo bị bỏ được lưu ở `despiked_readings.csv` trong dataset.
- `keep`: các khoảng thời gian đã xem lại và quyết định giữ nguyên (ví dụ có thể là hiện tượng thật), ghi kèm lý do.
- Không có khóa `despike` thì kết quả y như trước (đã kiểm chứng: dataset Thuận An dựng lại vẫn giống từng byte).
- Lúc chạy thật không có số đo của 1 giờ sau, nên giờ mới nhất chỉ so được với các số đo trước đó.

### 3.1d Khoảng dữ liệu hỏng (`invalid_ranges` trong config)
Ví dụ: sensor bị treo, vẫn gửi số đo đều mỗi 10 phút nhưng giá trị không đổi trong nhiều ngày, trong khi trạm lân cận vẫn dao động theo triều. Ví dụ thật: Dã Viên 01–12/02 và 20–25/06/2026, xem `models/da_vien/notes.md`.
- **Khai báo:** mỗi khoảng gồm `start_utc`, `end_utc` (tính cả hai đầu, theo thời điểm số đo) và `note` (bằng chứng).
- **Số đo:** `build.drop_invalid_ranges` bỏ mọi số đo trong khoảng **trước khi** lọc gai, nên chúng không ảnh hưởng tới trung vị của `despike` hay tới mực nước giờ. Giờ nào không còn số đo nào thì mực nước giờ là NaN. Không nội suy giá trị thay thế; database không bị sửa.
- **Mẫu:** mọi mẫu có cửa sổ lịch sử (t − W + 1 … t, với W = `max_sequence_length` của config, mặc định bằng `sequence_length`) hoặc giờ nhãn (t + 1 … t + mốc xa nhất) chạm vào một giờ thuộc khoảng hỏng đều bị gán `split = excluded_invalid_range`, nên không thuộc train, val hay test. Phần lớn các mẫu này vốn đã bị loại vì thiếu nhãn hoặc cửa sổ không đủ 80% giờ có dữ liệu; quy tắc này loại nốt các mẫu còn sót.
- **Một tập mẫu cho nhiều cửa sổ:**
  - Có thể dựng dataset với `sequence_length` nhỏ (ví dụ 12) rồi train với `--lstm-seq-len` lớn hơn (25 / 48 / 96). Khi đó `select_samples` tự lọc lại các mẫu theo quy tắc ≥ 80% trên cửa sổ mới.
  - Phải đặt `max_sequence_length` bằng cửa sổ dài nhất sẽ dùng. Nếu không, run cửa sổ dài vẫn giữ những mẫu có cửa sổ chạm vào giờ đã xóa (điền 0 kèm cờ thiếu).
  - Bộ chuẩn hóa được fit trên tập mẫu gốc (L = `sequence_length`) và dùng chung cho mọi cửa sổ.
- Fit triều chỉ dùng các giờ còn dữ liệu (mục 3.3).
- Không có khóa `invalid_ranges` thì kết quả y như trước (đã kiểm chứng 17/11/2026: dataset Ngã Ba Sình dựng lại vẫn giống từng byte).

### 3.2 Nhóm `wind` (15 kênh): `features.wind_features`
Từ chuỗi `*_previous_day1` theo giờ:
- Tách gió thành thành phần (phương hướng gió theo quy ước khí tượng, tức hướng gió **thổi tới từ đó**): u = −s·sin(d), v = −s·cos(d). Thành phần **vào bờ** = −(u + v)/√2 (gió từ hướng đông bắc vào); **dọc bờ** = (u − v)/√2; **tốc độ** = s.
- Với mỗi thành phần c ∈ {vào bờ, dọc bờ, tốc độ}, tại giờ t:
  - `past24` = trung bình các giờ t−23…t
  - `next6`, `next12` = trung bình các giờ t+1…t+6 và t+1…t+12
  - mỗi trung bình cần ít nhất nửa số giờ có dữ liệu
  - `chg12` = `next12` − `past24`
- `wind_p_now` = áp suất tại t; `wind_p_chg12` = trung bình áp suất t+1…t+12 trừ trung bình t−11…t.
- `wind_missing_frac` = 1 nếu thiếu bất kỳ kênh nào.

### 3.3 Nhóm `tide` (5 kênh): `features.tide_features`
Đặc trưng triều thiên văn, chỉ dùng dữ liệu đến t:
- **Fit 30 ngày** (`window_days=30`): tại mỗi giờ t, hồi quy bình phương tối thiểu mực nước quan trắc trên các giờ trong 30 ngày tính đến t (đầu chuỗi chưa đủ 30 ngày thì dùng toàn bộ lịch sử có sẵn, tối thiểu 7 ngày). Cửa sổ cần ≥ 70% số giờ có dữ liệu. Các biến hồi quy: [1, (t′ − t), cos/sin(2πt′/P)], với P là chu kỳ của 10 thành phần triều M2 12,4206012 h, S2 12 h, N2 12,65834751 h, K1 23,93447213 h, O1 25,81934166 h, Q1 26,868350 h, M4 6,210300601 h, MS4 6,103339275 h, MN4 6,269173 h, M6 4,140200 h. Thời gian t′ tính bằng giờ kể từ 2025-01-01 00:00 UTC.
- T(·) là **chỉ phần điều hòa** (bỏ hằng số và xu hướng). Kết quả: `tide_now` = T(t), `tide_dh` = T(t+h) − T(t) với h = 3, 6, 12.
- **Nếu không đủ dữ liệu** để fit: dùng một fit toàn cục cố định, chỉ trên các mốc **train**, của mực nước đã lọc dao động chậm (mực nước trừ trung bình trượt 25 h đặt giữa). Khi đó `tide_missing_frac` = 1, ngược lại = 0.
- Chọn `window_days=7` sẽ tái tạo bản cũ: 7 thành phần, cửa sổ cố định 7 ngày.

### 3.4 Nhóm `fc_ecmwf` (18 kênh): `features.synchronized_rain` và `forecast_rain_features`
- **Chọn lượt chạy tại t:** lấy **lượt chạy ECMWF mới nhất** có thời điểm khởi tạo trong khoảng t − 24 h ≤ init ≤ t − 6 h (giả định công bố trễ 6 h). Lượt chạy đó phải có đủ **mọi ô** trong `all_cells`, với lượng mưa hữu hạn và không âm cho các giờ t+1…t+24. Tất cả các ô dùng **cùng một lượt chạy**, không trộn nhiều lượt.
- Với mỗi ô, cộng lượng mưa từng giờ có giờ hiệu lực trong các cửa sổ t+1…t+3, t+4…t+6, t+7…t+12, t+13…t+24. Dùng phép cộng float của Python; từ Python 3.12, `sum()` có bù sai số.
- **Kênh:**
  - `fc_<tên>_rain_<0_3h|3_6h|6_12h|12_24h>` cho mỗi ô trong `named_cells`
  - `fc_basin_rain_<cửa sổ>` = trung bình **mọi** ô
  - tất cả đều biến đổi `log1p(max(mm, 0))`
  - `fc_age_hours` = t − init
  - `fc_missing_frac` = 1 nếu không tìm được lượt chạy nào

### 3.5 Chuẩn hóa (chỉ dùng train)
- Gọi R là tập **các giờ nằm trong ít nhất một cửa sổ L giờ của mẫu train** (`context_rows`).
- **Mực nước:** mean và độ lệch chuẩn (ddof = 0) trên R.
- **Mỗi nhãn h:** mean và độ lệch chuẩn của `target_h{h}_mm` trên các mẫu train.
- **Mỗi kênh phụ** (trừ `*_missing_frac`): mean và độ lệch chuẩn trên R.
- **Khi đọc dataset:** giá trị chuẩn hóa = (x − mean)/scale; giá trị thiếu (NaN) đặt bằng 0. Thêm kênh `water_level_missing` = 1 khi thiếu mực nước. `flood_season` (1 vào tháng 9–12 giờ Việt Nam) và các cột `*_missing_frac` giữ nguyên, không chuẩn hóa.

### 3.6 Các file của dataset
`water.parquet`, `calendar.parquet`, `rainfall.parquet` (chứa các nhóm phụ; tên file giữ theo reader), `weather.parquet` (rỗng), `samples.parquet`, `normalizers.json`, `feature_schema.json`, `station_mapping.json`, `config.json`, `manifest.json`.

**Thứ tự kênh đầu vào:** `water_level_depth`, `water_level_missing`, `flood_season`, rồi tới các nhóm theo thứ tự `--groups`. Danh sách đầy đủ nằm trong `feature_columns` của `selection.json` ở mỗi model.

---

## 4. Model: `ResidualGRU` (`src/ai/residual_sequence.py`, `--train-method gru --head-mode station`)

Đầu vào x có kích thước [B, L, F] (đã chuẩn hóa).
1. `Dropout1d(input_dropout)`: mỗi mẫu bỏ ngẫu nhiên **nguyên một kênh** trên toàn bộ cửa sổ.
2. `GRU(F → H, num_layers, batch_first, dropout giữa các lớp = encoder dropout)`. Lấy trạng thái ẩn ở bước cuối (t).
3. Ghép với embedding của trạm (`nn.Embedding(1, 8)` cho một trạm), qua `Linear(H + 8 → H)` + ReLU, rồi `Dropout(head_dropout)`.
4. **Head riêng cho trạm**, khởi tạo bằng 0: Δ_h = W·z + b, mỗi mốc h một đầu ra.
5. **Dự báo dạng residual**, trong không gian nhãn đã chuẩn hóa: ŷ_h = a_h · x_{t,mực nước} + b_h + Δ_h. Trong đó a_h = σ_w/σ_h và b_h = (μ_w − μ_h)/σ_h, nên a_h·x + b_h = (mực nước tại t − μ_h)/σ_h, chính là **persistence** (giữ nguyên mực nước hiện tại). Khi bắt đầu train, model đúng bằng persistence, rồi học dần phần chênh lệch.
6. Đổi về mm: ŷ_mm = ŷ_h · σ_h + μ_h.

**Không dùng embedding trạm** (`--head-mode shared --station-embedding-size 0`, dùng cho Dã Viên từ 17/11/2026): bỏ bước 3 và head riêng cho trạm. Với một trạm, embedding chỉ là một hằng số cộng thêm vào tầng ghép. Khi đó tầng ghép là `Linear(H → H)` + ReLU, head là `Linear(H → số mốc)` khởi tạo 0, và mã trạm chỉ còn dùng để lấy a_h, b_h. Số tham số giảm 8 + 8·H. Thuận An và Ngã Ba Sình vẫn giữ cấu hình cũ.

**Mỗi mốc một tầng ghép và một head riêng** (`--horizon-heads separate`, cần `--head-mode shared`; thêm ngày 17/11/2026):
- Thay tầng ghép dùng chung bằng 3 tầng `Linear(· → H)` + ReLU, mỗi mốc một tầng, cộng một head tuyến tính riêng cho từng mốc, khởi tạo 0.
- GRU vẫn dùng chung giữa các mốc. Mục đích: các mốc phản ứng khác nhau với cùng trạng thái GRU (xem phần "bập bênh +6 h / +12 h" trong `models/da_vien/notes.md`).
- Tăng thêm 2·(H·H + H) tham số (H = 128: 278.403 → 311.427).
- **Nhánh GRU riêng cho từng mốc** (`--horizon-gru-layers K`, cần `--train-method gru` và `--horizon-heads separate`): K lớp GRU trên cùng được nhân thành một nhánh cho mỗi mốc, các lớp dưới (`--lstm-num-layers` − K) dùng chung, giữa hai phần có `Dropout(--lstm-drop-out)`. Với H = 128, 3 lớp, 35 đầu vào: K = 1 → 509.571 tham số; K = 2 → 707.715.
- Chỉnh được kích thước mỗi head: `--horizon-head-size W` (số nơ-ron mỗi lớp ẩn, mặc định bằng `--lstm-hidden-size`) và `--horizon-head-layers K` (số lớp ẩn Linear + ReLU, mặc định 1). Với GRU 128×3 và 35 đầu vào, tổng tham số theo (W, K): (128, 1) 311.427; (64, 1) 286.467; (128, 2) 360.963; (64, 2) 298.947.

Số tham số: 3·(F·H + H·H + 2H) cho lớp GRU đầu, 3·(2H² + 2H) cho mỗi lớp GRU sau, cộng 8 (embedding), (H + 8)·H + H (fusion) và 3H + 3 (head). Ví dụ: F = 41, H = 32, 3 lớp → 21.291; F = 18, H = 32, 2 lớp → 12.747.

## 5. Loss: `ChangeRiskLoss`, chế độ `combined` (`--change-loss combined`)

Tính cho mỗi mẫu và mỗi mốc h, với σ_h là scale của nhãn (mm), y là nhãn và p là persistence (cả hai đã chuẩn hóa):
- sai số e = (ŷ − y)·σ_h (mm); thay đổi thực c = (y − p)·σ_h (mm); mẫu số d = max(σ_h, 200)
- base = (e/d)²
- under = (max(−e, 0)/d)² · 1[c > 200]: phạt **thêm** khi dự báo thiếu trong lúc mực nước thật sự lên hơn 200 mm
- tổng = base·(1 + event_weight·1[|c| > 200]) + under_weight·under; trong các model này `event_weight` = 0
- loss của batch = trung bình theo mẫu của Σ_h w_h·tổng, với w_h = 1/(số mốc)
- **Tùy chọn (thêm ngày 17/11/2026):**
  - `--change-margin-mm m`: thay e bằng dấu(e)·max(|e| − m, 0) trong base và under, tức các sai số nhỏ hơn m không bị phạt.
  - `--change-tail-weight λ --change-tail-start-mm s` (mặc định s = 150): cộng thêm λ·(max(|e| − s, 0)/d)² vào tổng, tính trên sai số **gốc** (trước margin). Khoản phạt bắt đầu trước ngưỡng 200 mm và tiếp tục tăng khi sai số lớn hơn. Mục đích: đẩy các sai số lớn nhất xuống dưới 200 mm, chấp nhận MAE tăng.
  - λ = 0 và m = 0 cho đúng loss gốc. Các model Thuận An và Ngã Ba Sình đã đóng gói đều dùng 0.

Với một trạm, hệ số cân bằng trạm bằng 1. Nếu σ_h > 200 thì base chính là MSE trong không gian đã chuẩn hóa. Val loss dùng đúng công thức này, lấy trung bình trên toàn tập validation.

## 6. Train (`src/ai/train.py`)

Mẫu lệnh, thay các giá trị trong `< >` bằng giá trị của từng model trong tài liệu trạm:
```bash
.venv/bin/python -m src.ai.train --mlflow-ex-name <exp> --horizon 1h --dataset-dir <dataset> --target-horizons 3 6 12 \
  --rain-stations all --use-weather-forecast false --feature-set water_level+rain+flood_season \
  --train-method gru --head-mode station --lstm-seq-len <L> --lstm-hidden-size <H> --lstm-num-layers <N> \
  --lstm-drop-out 0.2 --lstm-input-dropout 0.1 --lstm-head-dropout 0.1 --lstm-lr <lr> --lstm-weight-decay 0.001 \
  --lr-scheduler <step|warmup-cosine> [--lr-warmup-epochs 3] --batch-size 64 --num-epochs <E> --early-stopping <ES> \
  --change-loss combined --change-under-weight 0.3 --change-event-weight 0 --change-margin-mm 0 \
  --lstm-checkpoint-metric val_loss --skip-test --seed 17 --checkpoint-dir <thư mục> --diagnose-change-dynamics
```
- `--rain-stations all --feature-set water_level+rain+flood_season`: dùng mọi nhóm phụ có trong dataset. Chữ "rain" ở đây chỉ là tên chung của "nhóm đầu vào phụ". Dataset không có nhóm phụ thì bỏ `--rain-stations` và dùng `--feature-set water_level+flood_season` (`train.py` báo lỗi nếu có `--rain-stations` mà feature-set không chứa `rain`).
- **Optimizer:** AdamW (betas mặc định), lr và weight decay như trên. Cắt gradient theo chuẩn L2 = 1,0. Mỗi batch cập nhật một lần.
- **Thứ tự dữ liệu:** tập train được xáo bằng một `torch.Generator` khởi tạo với seed. Validation và test giữ nguyên thứ tự.
- **Scheduler** (bước một lần sau mỗi epoch):
  - `step`: lr·0,9^⌊(e − 1)/10⌋ cho epoch e
  - `warmup-cosine`: ở epoch thứ k (đếm từ 0), nếu k < 3 thì lr·(k + 1)/3; ngược lại lr·[r + (1 − r)·½·(1 + cos(π·min(1, (k − 3)/(E − 4))))], với r = 10⁻⁶/lr
- **Checkpoint:**
  - Epoch 0 là persistence (val loss của persistence).
  - Mỗi epoch, nếu val loss **nhỏ hơn hẳn** mức tốt nhất thì lưu `<thư mục>/.../*_best.pt`.
  - Early stopping khi val loss không cải thiện sau `--early-stopping` epoch.
  - `--diagnose-change-dynamics` lưu mọi epoch vào `…/*_best.dynamics/epoch_XXX/` (gồm `model.pt`, `model.preprocessing.joblib`, `validation_predictions.parquet`…).
  - `model.preprocessing.joblib` chứa sẵn bộ chuẩn hóa, cột đầu vào, độ dài cửa sổ và bảng ánh xạ trạm, nên mỗi thư mục epoch tự dùng được, không cần dataset.

## 7. Chấm điểm, chọn epoch, ensemble (`src/ai/station_dataset/evaluate.py`)

```bash
# mọi epoch của một run trên validation và test (in ra các epoch có 0 lỗi > 200 mm)
.venv/bin/python -m src.ai.station_dataset.evaluate epochs --dataset <dataset> --run-dir <thư mục checkpoint> --out <csv>
# từng model và ensemble (trung bình đều mực nước dự báo, mm)
.venv/bin/python -m src.ai.station_dataset.evaluate ensemble --member <dataset1> <model_dir1> --member <dataset2> <model_dir2>
# ghép theo mốc trong một khoảng thời gian (mỗi mốc lấy từ đúng một model), ví dụ models/nga_ba_sinh/REPRODUCE.md mục 4.6
.venv/bin/python -m src.ai.station_dataset.evaluate combined --member <dataset1> <model_dir1> 3,6 --member <dataset2> <model_dir2> 12 \
    --start 2026-09-01T00:00:00+07:00 --end 2026-10-31T23:00:00+07:00
```
- **Chỉ số theo từng mốc giờ:** n200 = số dự báo có |dự báo − quan trắc| > 200 mm; MAE (mm); sai số lớn nhất.
- **Chọn epoch** theo val loss (trainer tự chọn), hoặc theo số lỗi lớn trên validation. Test chỉ dùng để kiểm tra lại.
- **Ensemble:** lấy trung bình mm của các model, theo cùng thời điểm phát và cùng mốc giờ. Các dataset thành phần phải dựng từ cùng một config để có cùng tập mẫu.
- **Đóng gói:** chép thư mục `epoch_XXX` của epoch đã chọn vào `models/<trạm>/<tên>/`.
- **Đánh giá trên dữ liệu mới hơn dữ liệu train:**
  - Dựng lại dataset với `--grid-end-utc <mốc mới>` và `--normalizers-from <dataset train>`, để giữ đúng bộ chuẩn hóa lúc train. Tập train không đổi; các mốc mới sau `cutoff` tự động thuộc tập test.
  - Lấy dự báo bằng `predict(..., split)` với `split` ∈ {`validation`, `purged`, `test_exploratory`, `all`}. `purged` là các mốc bị loại giữa các tập; nên chấm cả nhóm này nếu muốn tính tròn theo tháng.

**Dùng một model trong code:**
```python
from src.ai.station_dataset.evaluate import predict          # trên dataset
df = predict('models/<trạm>/<model>/model.pt', '<dataset>', 'test_exploratory')   # anchor_utc, horizon_hours, predicted_mm, actual_mm
from src.ai.sequence_checkpoint import SequenceRuntime        # chạy thật: tự dựng x [1, L, F] theo feature_columns
mm = SequenceRuntime('models/<trạm>/<model>/model.pt').predict_mm(x, station_ids)      # [1, số mốc giờ] mm
```
