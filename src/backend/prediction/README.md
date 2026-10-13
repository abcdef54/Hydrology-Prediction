# Đầu vào dự báo theo từng trạm nước

Mỗi trạm nước dùng **một mô hình được train riêng**, đi cùng scaler, nguồn mưa và
cấu hình riêng. Thêm trạm không có nghĩa là đưa nhiều trạm nước vào mô hình Dã Viên.

## Cấu trúc

- `config.py`: `StationModelConfig` và `StationModelRegistry`, tra cứu theo `station_id`.
- `model_configs.json`: mapping trạm nước → model, preprocessing và hợp đồng đầu vào.
- `features.py`: tạo feature theo cấu hình được chọn.
- `preprocessing.py`: kiểm tra artifact đúng schema/sequence length, áp scaler riêng.
- `inputs.py`: query PostgreSQL theo nguồn của cấu hình; trả tensor và định danh model.
- `inference.py`: load checkpoint, chạy Residual LSTM ở chế độ eval và trả dự báo bằng mm.

Hiện registry chỉ cấu hình Dã Viên: Residual LSTM, 96 timestep, 29 feature và 4 trạm
mưa đã chọn. Logic dùng chung không cố định mã Dã Viên hoặc kích thước 96 × 29.
Trạm chưa có cấu hình báo `ModelNotConfiguredError`; không fallback sang Dã Viên.

## Cách sử dụng

`DATABASE_URL` nằm trong môi trường ứng dụng. Khởi tạo registry và preprocessor
một lần cho mỗi model được sử dụng, rồi tái sử dụng service đó:

```python
from datetime import datetime
from pathlib import Path

from src.backend.db.session import build_engine
from src.backend.prediction.config import StationModelRegistry
from src.backend.prediction.inputs import PredictionInputService
from src.backend.prediction.preprocessing import InputPreprocessor

project_root = Path.cwd()  # Ví dụ chạy từ thư mục gốc dự án.
registry = StationModelRegistry.from_json(
    project_root / "src/backend/prediction/model_configs.json",
    artifact_root=project_root,
)
config = registry.get("0074647403")
preprocessor = InputPreprocessor.from_file(config)
engine = build_engine()
service = PredictionInputService(engine, preprocessor)
try:
    latest = service.prepare()
    historical = service.prepare(datetime.fromisoformat("2026-09-20T14:30:00+07:00"))
    print(latest.station_id, latest.model_version, latest.tensor.shape)
finally:
    engine.dispose()
```

Service được gắn với đúng cấu hình của preprocessor, không truyền một mã trạm
khác vào service đã khởi tạo. API sau này sẽ lấy `station_id` người dùng chọn,
tra registry và lấy service/model tương ứng. `PredictionInputService` chỉ chuẩn bị
đầu vào; `ResidualLSTMRuntime` sử dụng `model_path` để load trọng số và chạy dự báo.

Các đường dẫn artifact được resolve tương đối với `artifact_root`. Preprocessing
phải là artifact tin cậy đi cùng checkpoint của trạm. Chưa đóng gói artifact cho Docker.

Cặp model/scaler Dã Viên phục vụ ứng dụng nằm ngay trong `model/`, theo quy ước
[đặt tên theo trạm và loại model](../../../model/README.md). Registry trỏ trực
tiếp vào cặp file này, không phụ thuộc thư mục MLflow. Khi thêm trạm hoặc đổi
phiên bản, đặt cặp file riêng rồi cập nhật registry và khởi động lại API.

## Quy tắc chuẩn bị dữ liệu

1. Latest dùng giờ hoàn tất của quan trắc nước mới nhất không nằm trong tương lai:
   14:30 → 14:00. Historical làm tròn xuống giờ. SQL và feature builder đều loại
   dữ liệu sau mốc đã chọn.
2. Nhãn t đại diện `(t−1h, t]`: nước lấy trung bình, mưa lấy tổng từng trạm.
   Lag và mưa tích lũy tính trên lưới giờ trước khi chọn chuỗi. Các khoảng lấy từ
   cấu hình của model, không shift theo vị trí hàng sau khi bỏ giờ thiếu.
3. Chọn số timestep theo `sequence_length`, cần có nước hiện tại, cho phép gaps.
   Mưa hoặc lag thiếu vẫn được giữ. Query mở rộng về trước khi thiếu timestep hoặc
   thiếu lịch sử để tính cửa sổ mưa. Không nội suy, thêm timestep giả hoặc điền mưa 0.
4. Cờ mùa lấy từ lịch và múi giờ trong cấu hình. Dã Viên dùng tháng 9–12, giờ Việt Nam.
5. Sắp đúng feature order, chuyển float32, áp scaler riêng rồi thay NaN đã chuẩn hóa
   bằng 0. Không fit lại scaler; không dùng các nhãn tương lai làm điều kiện chọn chuỗi.

Thiếu chuỗi trả `InsufficientHistoryError` với số hiện có và số cần thiết.
Đầu ra gồm tensor CPU, bảng feature, timestamps, `station_id`, `model_version`,
`feature_schema_version`, mốc yêu cầu, mốc làm tròn giờ và anchor thực tế.

`overrides` chỉ sửa feature được chỉ định ở timestep cuối trên một bản sao trước
khi áp scaler. Không ghi PostgreSQL hoặc sửa timestep trước. Flag nhận 0/1;
giá trị thiếu dùng None; nước hiện tại cần hữu hạn. Các feature không tự tính lại
lẫn nhau: sửa mưa 1h không tự đổi tổng 6h hoặc cờ thiếu do người dùng nhập.

## Khi thêm một trạm nước

Train và kiểm chứng model/scaler riêng, nhập quan trắc nước và các trạm mưa của
trạm đó vào PostgreSQL, rồi thêm một mục vào registry. Không sửa logic dùng chung
để thay mã Dã Viên. Pipeline hiện hỗ trợ feature nước, mưa, cờ thiếu và cờ mùa theo
lưới giờ; nếu model mới dùng loại feature khác thì cần bổ sung pipeline tương ứng
và kiểm chứng với dataset của model đó.

## Kiểm chứng

```bash
.venv/bin/python -m unittest discover -s src/backend/prediction/tests -v
```

Test gồm ranh giới giờ, loại dữ liệu tương lai, cờ mùa, dữ liệu thiếu, overrides,
thiếu lịch sử và gaps cần mở rộng query. Test hai trạm dùng nguồn riêng, scaler
riêng và kích thước 96 × 29 / 48 × 17; chưa có model thật cho trạm thứ hai.

Đối chiếu Dã Viên với Train/Val/Test bằng script local
`scripts/local/verify_prediction_inputs.py`; kết quả ở
`scripts/local/prediction_inputs_verification.json`. Không ghi DB hoặc chạy model.


## Chạy dự báo

Backend dùng lại hai lớp `LSTM`/`ResidualLSTM` trong `src/ai/lstm.py`. Module này
chỉ cần PyTorch; `src.ai.models` vẫn export các lớp đó để giữ tương thích training
và các model đã lưu. Không thay đổi công thức forward hoặc tên trọng số.

Sau khi đã lấy `config` và `engine` như ví dụ trên:

```python
from dataclasses import asdict
from src.backend.prediction.inference import PredictionService, ResidualLSTMRuntime

runtime = ResidualLSTMRuntime(config, device="cpu")
service = PredictionService(engine, runtime)
forecast = service.predict(
    reference_time=datetime.fromisoformat("2026-09-20T14:30:00+07:00"),
    overrides={"water_level_depth": 700.0},
)
print(asdict(forecast))
```

Khởi tạo runtime một lần cho mỗi mô hình rồi dùng lại. CPU là mặc định; có thể
truyền `device="cuda"` nếu máy có CUDA. Việc kiểm chứng hiện tại thực hiện trên CPU.

- `model_parameters` trong JSON lưu hidden size, số lớp và các dropout đúng cấu hình
  training. `eval()` vô hiệu hóa dropout; `torch.inference_mode()` tắt tính gradient.
- Không có optimizer, training loss hay correction penalty trong luồng dự báo.
- Kiểm tra thứ tự target, kích thước trọng số và hệ số persistence với scaler trước
  khi dùng. Chỉ load artifact tin cậy; scaler joblib thuộc cùng model đã xác minh.
- Input của trạm/model/schema khác bị từ chối. Model type chưa hỗ trợ cũng bị từ chối.
- Inverse transform bằng target scaler; không nhân/chia thêm đơn vị. Kết quả là mm,
  không làm tròn hoặc ép mực nước âm về 0.
- Mốc dự báo bằng `anchor_observed_at + horizon`, không dùng mốc người dùng yêu cầu
  nếu chuỗi thực tế kết thúc sớm hơn. Kết quả giữ station, model/schema version và
  cả mốc yêu cầu, mốc giờ, anchor để API có thể biểu diễn rõ.

Script local `scripts/local/verify_prediction_inference.py` đối chiếu toàn bộ chuỗi
Test với artifact mô hình đã lưu, kiểm tra RMSE với MLflow đã ghi nhận và chạy
latest/historical/override trên PostgreSQL ở chế độ chỉ đọc. Kết quả lưu tại
`scripts/local/prediction_inference_verification.json`. Không train hoặc ghi MLflow.

## Lưu và dùng lại dự báo

`PredictionService.predict()` chuẩn bị chuỗi trước, rồi tìm dự báo khớp trong
PostgreSQL nếu request không có overrides. Nếu đã lưu thì không gọi runtime.predict;
nếu chưa có thì inference và lưu tất cả horizon trong cùng transaction.

- `results.py`: các dataclass kết quả, được inference và persistence dùng chung.
  Import `ForecastPoint`/`ForecastResult` từ `inference.py` vẫn hoạt động.
- `persistence.py`: tạo SHA-256 cho pipeline và toàn bộ input, đọc/ghi bảng dự báo.
- Runtime tải và hash cùng một snapshot bytes của model/scaler để nhận diện đúng
  artifact đã dùng. API phát hiện file bị thay thế trước khi dùng lại runtime.
- Tất cả timestep, feature gốc, mask thiếu, tensor chuẩn hóa và timestamp tham gia
  hash. Cấu hình nguồn/cửa sổ/mùa/horizon, mã nguồn pipeline, scaler, checkpoint và
  device tham gia pipeline version. Thay code phải restart worker.
- Bản lưu thiếu horizon hoặc sai thời điểm bị từ chối. Trạng thái freshness và
  request time không được lưu cứng vào kết quả dùng lại. Overrides luôn bỏ qua DB.

Test trong `prediction/tests/test_persistence.py` không cần DB.
`db/tests/test_forecast_persistence.py` dùng PostgreSQL và rollback dữ liệu thử.
Script `scripts/local/verify_persisted_predictions_api.py` chạy model thật, lưu
dự báo gốc và kiểm tra request lặp không gọi model; không sửa quan trắc.
