# PostgreSQL với SQLAlchemy và Alembic

## Cấu trúc

- `models.py`: ORM model `Station`, `Observation`, `Forecast`, `ForecastPoint`,
  gồm khóa và ràng buộc dữ liệu.
- `session.py`: tạo engine và session SQLAlchemy; lấy kết nối từ `DATABASE_URL`.
- `alembic/versions/`: migration Python tạo bảng và thêm metadata 5 trạm.
- `tests/test_schema.py`: kiểm tra trực tiếp trên PostgreSQL; rollback dữ liệu thử.

Alembic quản lý phiên bản schema; SQLAlchemy quản lý model và truy vấn bằng Python.
Không dùng `Base.metadata.create_all()` để thay migration. Migration là bản cố định,
không import ORM model hiện tại để tạo bảng, nên thay model sau này không làm đổi
lịch sử migration. Phần trigger PostgreSQL vẫn cần một đoạn SQL nhỏ trong migration
Python để cập nhật `updated_at` cả khi ghi qua ORM hoặc UPSERT.

## Khởi tạo

Cần một database PostgreSQL riêng đã được tạo. Từ thư mục gốc dự án:

```bash
.venv/bin/python -m pip install -r src/backend/requirements.txt
# Đặt DATABASE_URL trong môi trường server; không commit URL có mật khẩu.
# Cấu trúc: postgresql+psycopg://USER:PASSWORD@HOST:5432/DATABASE
.venv/bin/python -m alembic -c src/backend/db/alembic.ini upgrade head
```

Không có kết nối DB khi import module. Các kết nối dùng UTC. Alembic chạy trong
transaction và dùng advisory lock để tránh hai tiến trình migrate cùng lúc. Chạy
lại `upgrade head` không tạo bản ghi trùng hoặc ghi đè metadata đã sửa.

Database có schema `hydrology` cho bảng nghiệp vụ; bảng phiên bản
`public.hydrology_alembic_version` do Alembic quản lý. Tài khoản migrate cần quyền
CREATE schema và tạo bảng trong `public`. Thay đổi schema tiếp theo:

```bash
.venv/bin/python -m alembic -c src/backend/db/alembic.ini revision --autogenerate -m "describe change"
# Đọc và kiểm tra migration được sinh trước khi áp dụng.
.venv/bin/python -m alembic -c src/backend/db/alembic.ini upgrade head
```

Autogenerate chỉ xét schema `hydrology`, không đề xuất xóa bảng ứng dụng khác.
Trigger/function phải được cập nhật thủ công trong migration nếu cần thay đổi.

Bản SQL thuần trước đó chỉ chạy trên database tạm và đã được thay bằng các file
Python. Không tự `stamp head` lên một database SQL cũ: Alembic chưa xác minh schema
đó. Task này chưa triển khai database thường trực cho ứng dụng.

## Dữ liệu được lưu

| Model | Nội dung | Khóa chính |
|---|---|---|
| `Station` | Metadata, loại trạm, nguồn, vị trí và đơn vị | `id` dạng chuỗi |
| `Observation` | Giá trị `depth` gốc và thời điểm đo tại từng trạm | `(station_id, observed_at)` |
| `Forecast` | Danh tính mô hình/pipeline/đầu vào và thời điểm tạo dự báo | UUID `id` |
| `ForecastPoint` | Giá trị dự báo mm tại từng horizon | `(forecast_id, horizon_minutes)` |

Metadata khởi tạo gồm Dã Viên và bốn trạm mưa Hồng Thái, Nhâm, Thị trấn A Lưới,
Hồ A Lá, theo API khảo sát ngày 12/10/2026. Mã `0074647403` giữ số 0 đầu. Hồ A Lá
ở đây là **trạm mưa**, không phải dữ liệu vận hành hồ chứa.

- `depth` giữ nguyên giá trị API. Nước dùng mm theo xác nhận của chủ dự án; đơn vị
  mưa chưa xác nhận nên để `measurement_unit = NULL`. Chưa đổi đơn vị hoặc tổng hợp giờ.
- `observed_at` là `TIMESTAMPTZ`; truyền timestamp có múi giờ. Hiển thị giờ Việt Nam
  thuộc bước ứng dụng. Không tạo thêm hàng cho những thời điểm thiếu.
- Giá trị thiếu dùng `depth = None`, `quality_status = 'missing'`. Không thay bằng 0;
  số 0 là quan trắc thật. NaN và infinity bị từ chối. Mực nước âm vẫn được chấp nhận.
- Khóa nguồn ngăn trùng ID API trong cùng collection; foreign key kiểm tra đúng trạm
  và nguồn. Ngưỡng báo động có cột theo mét nhưng để trống khi chưa xác nhận cho Dã Viên.
- `received_at` giữ thời điểm lưu đầu tiên; trigger cập nhật `updated_at` khi hàng đổi.
  Ingestion sau này cần UPSERT và chỉ UPDATE khi dữ liệu nguồn thực sự thay đổi.

Primary key cũng hỗ trợ tìm latest và lịch sử bằng B-tree, không thêm index trùng.
Ví dụ truy vấn bằng Python:

```python
from sqlalchemy import select
from src.backend.db.models import Observation
from src.backend.db.session import build_engine, build_session_factory

engine = build_engine()
SessionFactory = build_session_factory(engine)
with SessionFactory() as session:
    latest = session.scalar(
        select(Observation)
        .where(Observation.station_id == "0074647403")
        .order_by(Observation.observed_at.desc())
        .limit(1)
    )
engine.dispose()
```

**96 quan trắc raw không phải 96 timestep model.** Tạo feature giờ `(t−1h, t]`,
chọn giờ đã hoàn tất, chuẩn bị 29 feature và áp scaler thuộc task sau. Không lưu
user overrides vào bảng quan trắc.

## Kiểm tra

### Lưu dự báo — migration `0003`

Migration thêm `hydrology.forecasts` và `hydrology.forecast_points`, không sửa
dữ liệu quan trắc hoặc tạo dự báo. UUID được SQLAlchemy sinh khi insert.

- `Forecast` lưu `station_id`, `model_version`, `feature_schema_version`,
  `pipeline_version`, `input_hash`, `anchor_observed_at`, `generated_at`.
  `input_hash` là SHA-256 gồm 64 ký tự hex thường. Unique constraint trên trạm,
  ba phiên bản và hash ngăn lưu cùng dự báo nhiều lần. Index trạm/model/anchor
  phục vụ đọc lịch sử. Mọi timestamp dùng TIMESTAMPTZ.
- `ForecastPoint` lưu `forecast_id`, `horizon_minutes`, `forecast_at`,
  `water_level_mm`. Horizon phải dương; mực nước phải hữu hạn, được phép âm hoặc 0.
  Foreign key không cho lưu điểm thiếu bản ghi cha hoặc xóa cha đang có điểm.
- Service sẽ phải tính hash từ toàn bộ chuỗi và timestamp, nhận diện scaler/
  preprocessing trong `pipeline_version`, tính `forecast_at = anchor + horizon`,
  và lưu cha cùng tất cả horizon trong một transaction. Schema không tự kiểm tra
  đủ horizon của mô hình hoặc quan hệ thời gian giữa hai bảng.
- Chỉ lưu dự báo từ quan trắc gốc; không lưu overrides. Cơ chế lưu/đọc, xử lý
  xung đột khi request đồng thời và Redis thuộc task tiếp theo, chưa nối vào API.

Đã áp dụng revision `0003` trên database ứng dụng: 14 test database đạt,
`alembic check` không thấy khác biệt ORM/schema. Giữ nguyên 5 trạm và 237954
quan trắc; sau rollback test có 0 dự báo và 0 điểm dự báo.

### Chạy test database

Dùng `DATABASE_URL` trỏ đến database thử nghiệm đã migrate:

```bash
.venv/bin/python -m unittest discover -s src/backend/db/tests -v
.venv/bin/python -m alembic -c src/backend/db/alembic.ini check
```

Các test dùng transaction và rollback mọi quan trắc giả. `alembic check` đối chiếu
model với schema đã migrate; không sửa database.

Đã kiểm chứng ngày 12/10/2026 với PostgreSQL 16.2:
6 integration test đạt; `alembic check` không thấy khác biệt giữa model và schema;
chạy lại migration không tạo trùng hoặc ghi đè metadata. Downgrade rồi upgrade
trên database tạm cũng đạt. Kết thúc có 5 trạm, 0 quan trắc giả, revision `0002`.
PostgreSQL thử được tắt sau kiểm tra; chưa triển khai database thường trực.
