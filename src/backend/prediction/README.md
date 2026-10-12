# Đầu vào dự báo theo từng trạm nước

Mỗi trạm nước dùng **một mô hình được train riêng**, đi cùng scaler, nguồn mưa và
cấu hình riêng. Thêm trạm không có nghĩa là đưa nhiều trạm nước vào mô hình Dã Viên.

## Cấu trúc

- `config.py`: `StationModelConfig` và `StationModelRegistry`, tra cứu theo `station_id`.
- `model_configs.json`: mapping trạm nước → model, preprocessing và hợp đồng đầu vào.
- `features.py`: tạo feature theo cấu hình được chọn.
- `preprocessing.py`: kiểm tra artifact đúng schema/sequence length, áp scaler riêng.
- `inputs.py`: query PostgreSQL theo nguồn của cấu hình; trả tensor và định danh model.

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
tra registry và lấy service/model tương ứng. `model_path` mới là metadata;
task hiện tại chưa load trọng số hoặc chạy inference.

Các đường dẫn artifact được resolve tương đối với `artifact_root`. Preprocessing
phải là artifact tin cậy đi cùng checkpoint của trạm. Chưa đóng gói artifact cho Docker.

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
