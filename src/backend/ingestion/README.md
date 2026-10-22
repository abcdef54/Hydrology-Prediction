# Pipeline cập nhật dữ liệu quan trắc

## Thu thập dự báo lượng mưa ECMWF

Collector nằm trong `src/backend/ingestion/weather_forecast.py`; schema các bảng
nằm trong `src/backend/db/weather.py`, được tạo bởi migration `0009`.
Backend đọc trực tiếp các bảng này để chuẩn bị đầu vào cho Model B và model ba trạm.
Phần chạy ứng dụng không còn import từ `research/weather_forecast/`.

Chạy từ thư mục gốc, dùng `DATABASE_URL` từ environment hoặc `.env`:

```bash
.venv/bin/python -m src.backend.ingestion.weather_forecast \
  --dataset-dir train_data/da_vien_1h_2026-10-15_forecast_mask_o24 \
  --station-ids 750062 750007 \
  --report reports/weather_collection/report.json
```

Đây là collector **lịch sử theo các timestamp Train/Val/Test của dataset**:
lấy từng run ECMWF từ Open-Meteo Single Runs, bỏ qua run đã có trừ khi dùng
`--refresh`, giữ riêng các vintage thay đổi và phân biệt mưa 0 với thiếu dữ liệu.
`--plan-only` chỉ đọc dataset/catalog, không gọi nguồn hoặc ghi DB;
`--audit-only` kiểm tra độ phủ đã lưu. Các chế độ vẫn ghi report theo `--report`.
`available_at` chưa xác minh được giữ NULL; giả định trễ phát hành mặc định 6 giờ
không phải bằng chứng về thời điểm dự báo thực sự có sẵn trong lịch sử.

Collector lịch sử vẫn là lệnh riêng, không chạy trong luồng dự báo.
Notebook, phân tích độ phủ và công cụ tạo dataset vẫn nằm trong
`research/weather_forecast/`.

### Đồng bộ dự báo mới trong ứng dụng

`weather_live.py` được worker `src.backend.ingestion` gọi mỗi chu kỳ đồng bộ,
mặc định bật. Với Docker, lệnh chạy cả app và worker là:

```bash
docker compose --profile ingestion up --build -d
```

Module đọc các location của model production trong `model_configs.json`, gộp
location trùng, thu thập các run ECMWF sáu giờ gần nhất trong giới hạn tuổi của
model. Không cần dataset CSV. Giữ `ecmwf_ifs`, lưới và các cửa sổ feature hiện có;
không thay bằng dự báo đã ghép từ nhiều run. API chỉ đọc PostgreSQL, không gọi
Open-Meteo trong request dự báo.

Mỗi run đủ 48 giờ không thiếu được bỏ qua ở chu kỳ sau; run chưa có hoặc còn
thiếu được thử lại. Lỗi kết nối, HTTP 429/5xx retry tối đa ba lần, chờ có thể hủy;
run chưa công bố không ngăn thu thập run cũ hơn. Advisory lock riêng ngăn hai
worker cùng thu thập thời tiết. Lỗi thời tiết không rollback quan trắc đã commit,
và lỗi nguồn quan trắc không ngăn thử đồng bộ thời tiết.

Run mới ghi `available_at = retrieved_at`, `availability_basis=observed_retrieval`:
đây là lúc hệ thống thực sự nhận được dữ liệu, không suy ra giờ công bố từ giờ
khởi tạo. Không backdate dữ liệu để lấp đầu vào lịch sử. Các vintage archive đã
có vẫn giữ metadata khả dụng chưa xác minh; run/payload trùng không bị ghi đè.
Vì vậy, khi mới bật collector, lịch sử forecast 96 giờ có thể còn thiếu và dùng
mask của mô hình cho tới khi thu thập đủ. Không đảm bảo đầy đủ dữ liệu lịch sử
chỉ bằng việc bật worker hôm nay.

Xem mục `weather` trong log/report worker: `fetched`, `skipped`, `incomplete`,
`failures`, `latest_initialized_at`, `latest_age_hours`, `stale`. Tuổi này tính
từ giờ khởi tạo của run, không phải xác nhận mọi feature ở hiện tại đều đủ.
`WEATHER_SYNC_ENABLED=false` tắt riêng thu thập thời tiết. `--dry-run` chỉ liệt kê
location cho phần thời tiết, không gọi Open-Meteo hoặc ghi bảng thời tiết.
`WEATHER_SYNC_ENDPOINT` cho phép trỏ HTTP client vào simulator khi kiểm thử;
mặc định là Open-Meteo Single Runs API.

Kiểm thử HTTP bằng Postman CLI >=1.70 (terminal riêng cho simulator):

```bash
postman simulation run src/backend/ingestion/tests/fixtures/weather-simulation.sim.yaml \
  --port 4900 --no-history --output ndjson
WEATHER_SIMULATION_URL=http://127.0.0.1:4900 .venv/bin/python -m unittest \
  src.backend.ingestion.tests.test_weather_live \
  src.backend.ingestion.tests.test_weather_live_database -v
```

Fixture dùng contract [Open-Meteo Single Runs](https://open-meteo.com/en/docs/single-runs-api).
Test database dùng PostgreSQL local đã migrate, rollback toàn bộ dữ liệu thử.

## Hồ/đập SOHOA và báo cáo vận hành

Cần migration `0008`. Chạy từ thư mục gốc:

```bash
.venv/bin/python -m src.backend.ingestion.facility_import
.venv/bin/python -m src.backend.ingestion.facility_import --apply
```

Dry-run đọc/validate toàn bộ SQLite + API, chỉ ghi report local. Apply nhập 19
công trình, liên kết 4 điểm sông với trạm DCU đã có thay vì tạo trạm trùng.
Các hồ/đập/thủy điện còn lại nằm trong bảng facilities riêng, chưa thêm marker
trong frontend. Không sửa Station/quan trắc DCU, model/scaler hay dữ liệu training.
SQLite mở mode=ro/query_only, chỉ INSERT chưa có; API UPSERT theo ID nguồn,
kể cả trangthai=2 (Đã xóa). Giữ received_at; chạy lại idempotent, không skip bằng
checkpoint. Không hard-delete hàng mất khỏi API. Token không đưa vào report/log.

Số vận hành giữ nguyên, đơn vị chưa đủ bằng chứng để chuẩn hóa để NULL. Thông
số metadata giữ nguyên văn có đơn vị; không tự gán ngưỡng nước từ nguồn mới.
Giữ chuỗi số cửa mở (0/15, văn bản) và khoảng vận hành chưa đầy đủ thời gian.
Ngày đo + giờ H:M rõ ràng là giờ Việt Nam, chuyển UTC; ngày cập nhật và received_at
lưu riêng. Giờ sai/thiếu giữ nguyên văn, observed_at=NULL/time_status=invalid,
không dùng thời điểm cập nhật hoặc tự suy ra phút 00. Không lưu người nhập,
ghi chú tự do hoặc link tệp đính kèm. Các bản ghi xóa/giờ lỗi không lên latest.

Phân trang nguồn đối chiếu totalrow và ID mọi trang, không giả định trang ngắn
là hết. Nếu nguồn đổi totalrow/trùng ID trong lượt đọc thì dừng, chạy lại lượt;
không ghi snapshot API chưa đầy đủ. Đọc mọi chức năng/trạng thái với chucnang=0,
trangthai=0, khoảng ngày nguồn tính cả hai đầu.

Worker sau restart backend gọi SOHOA riêng sau khi DCU commit: đọc chồng ngày
đo ít nhất 24 giờ mỗi 11 phút; quét toàn bộ mỗi 24 giờ để bắt xóa/sửa lịch sử cũ.
last_full_sync_at tiến sau commit, dry-run/lỗi giữ nguyên. Lỗi SOHOA được báo trong
facilities.status, không làm mất dữ liệu DCU đã lưu; --once trả lỗi khi SOHOA lỗi.
Đặt sync_enabled=false để tạm dừng một công trình. Chưa có metadata thì không
gọi nguồn. Vẫn chỉ chạy trong phiên backend local, không bật service nền.

Cấu hình server: SOHOA_API_BASE_URL và SOHOA_API_TOKEN. Local chưa có token env
thì đọc sohoa_token từ file Postman environment private hiện có trong data/;
khi deploy cần cung cấp token env. Không copy token vào code/.env.example hay
gửi frontend. Token hết hạn: cập nhật config private rồi chạy lại/restart.
Report counts ở reports/facility_import.json, Git-ignore, không có payload gốc.
Đã nhập 39.894 hàng SQLite, thêm 719/cập nhật 2 từ API, tổng DB 40.613. Nguồn
API có 40.612 hàng, gồm 447 Đã xóa và 30 giờ lỗi; một hàng SQLite ngoài phạm vi
API hiện tại được giữ lại. Lượt đồng bộ code mới đọc lại 102 báo cáo gần nhất,
không tạo trùng/cập nhật không cần thiết. Metadata vẫn 19 công trình, 4 liên kết
sông; Station vẫn 78 trạm, quyền dự báo chỉ Dã Viên.

## Nhập toàn bộ lịch sử nước/mưa

Module `history_import` đọc SQLite `mode=ro` + `query_only`, không sửa file nguồn,
training data hay artifacts. Mặc định dry-run; dùng report riêng cho dry-run.

```bash
# Kiểm tra một trạm bằng report riêng, không ghi DB
.venv/bin/python -m src.backend.ingestion.history_import \
  --station 0074647404 --report reports/history_validation.json
# Nhập SQLite (mặc định); chạy lại cùng report để tiếp phần đã ngắt
.venv/bin/python -u -m src.backend.ingestion.history_import --apply --stage sqlite
# Bổ sung phần trước/sau snapshot SQLite từ API, cùng checkpoint
.venv/bin/python -u -m src.backend.ingestion.history_import --apply --stage api
# Refresh riêng 24 giờ cuối nếu đối chiếu phát hiện nguồn công bố trễ
.venv/bin/python -u -m src.backend.ingestion.history_import \
  --apply --stage api --station 750014 --refresh-api
```

Chỉ xử lý metadata nước/mưa đã có trong PostgreSQL, kể cả trạm chưa enabled.
SQLite chỉ INSERT hàng chưa có; mọi giá trị DB hiện có được giữ nguyên. Dữ liệu
API trực tiếp mới có quyền UPSERT sửa giá trị nguồn; `received_at` không đổi.
COPY staging + batch 5.000 hàng giới hạn RAM/transaction; mỗi batch dùng cùng
advisory lock writer với worker rồi invalidate cache sau commit. Worker có thể
tiếp tục giữa các batch, không bị dừng hoặc đổi cấu hình. Checkpoint chỉ tiến sau
commit; nguồn SQLite đổi fingerprint thì yêu cầu report mới. Không overwrite
report đã có bằng dry-run. Không tự bật trạm hoặc thêm quyền dự báo.

API lấy từ earliest tới snapshot nếu nguồn có lịch sử cũ hơn, và từ cuối snapshot
trừ 24 giờ tới cutoff cố định; trạm mới không có SQLite lấy toàn bộ khoảng nguồn.
Đọc từng cửa sổ 7 ngày, phân trang ID, làm mới token khi 401. Không khẳng định đã
kiểm tra mọi sửa đổi cũ trong nội bộ snapshot: chỉ tail 24 giờ được refresh.
Giữ số 0/âm/NULL, không tổng hợp theo giờ hoặc đổi đơn vị. Mã SQLite chưa có
metadata (hiện `750031`) chỉ ghi report `unmapped_sqlite_stations`, không tạo trạm
giả. Gió/hồ không thuộc importer này. Report/checkpoint nằm ở
`reports/full_history_import.json` (Git-ignore), không có credentials/payload gốc.


## Nhập và đồng bộ gió

Cần migration `0006`. Importer riêng tránh trộn gió với depth của nước/mưa:

```bash
.venv/bin/python -m src.backend.ingestion.wind_import
.venv/bin/python -m src.backend.ingestion.wind_import --apply
```

Lệnh mặc định chỉ đối chiếu metadata (không ghi DB); `--apply` nhập bốn trạm,
đọc SQLite ở chế độ read-only và bổ sung lịch sử API tới cutoff cố định. Giữ raw
ws/wd/wsg/wdg, 0 và NULL; đơn vị chưa xác minh để NULL. SQLite không ghi đè số đo
DB, API có thể cập nhật revision nhưng giữ received_at. Dùng writer lock chung;
chạy lại không tạo trùng, các batch đã commit không mất khi bị ngắt. Report ở
`reports/wind_import.json` (Git-ignore); chạy lại đọc lại nguồn, không skip bằng
checkpoint như importer nước/mưa. Không sửa nguồn SQLite/model/training/MLflow.

ID `wind:<code>` tránh trùng mã trạm mưa; source_station_id là UUID, query dùng
`sid`. Bốn trạm gió có sync_enabled=true nhưng enabled=false để chưa hiện trên
bản đồ khi nhập mới; bốn trạm hiện có đã được bật hiển thị bởi migration `0007`
sau khi frontend hỗ trợ. Không cho dự báo. Worker nhận enabled **hoặc** sync_enabled, đọc
watermark và ghi vào bảng đúng loại. Khởi động lại backend đang chạy để nạp
code mới; chu kỳ vẫn 11 phút, chỉ hoạt động cùng phiên backend local.

Đã nhập 95.974 hàng SQLite + 8.914 hàng API = 104.888 quan trắc tới 19:45:53 VN
ngày 14/10/2026. Đơn vị/thang đo DCU chưa được xác nhận; chi tiết nghiên cứu
WATEC/YOUNG và giả thuyết ADC nằm trong [ARCHITECTURE.md](../../ARCHITECTURE.md).
Đã so đúng 95.974 bản ghi SQLite (timestamp và bốn giá trị) với PostgreSQL;
count/mốc cuối khớp API cả bốn trạm sau lượt worker kiểm chứng 78 trạm. Lượt đó
thêm 6 hàng gió, tổng 104.894. Report `wind_verification.json` và
`wind_sync_verification.json` chỉ lưu local, không có credentials.

## Bổ sung danh mục trạm nước/mưa

Importer riêng chỉ đọc metadata nguồn và ghi bảng `hydrology.stations`:

```bash
# Đối chiếu kế hoạch, không ghi database
.venv/bin/python -m src.backend.ingestion.station_catalog
# Ghi metadata trong một transaction
.venv/bin/python -m src.backend.ingestion.station_catalog --apply
```

Cần migration `0005` trước khi chạy. API nguồn hiện có 19 trạm nước và 55 trạm mưa
(khảo sát 14/10/2026); phân trang tới trang rỗng, kiểm tra mã/tọa độ/loại và từ chối
mã trùng. Import lại không UPDATE hàng không đổi. Chỉ cập nhật địa chỉ/tọa độ và
subtype có dữ liệu nguồn; giữ tên đã cấu hình, đơn vị, tên sông, ngưỡng báo động và
quyền enabled/prediction_enabled của trạm hiện có. Không tự xóa trạm vắng ở nguồn.

**Importer tạo trạm mới chưa bật (`enabled=false`) và không được dự báo.** Sau
task nhập lịch sử, ngày 14/10/2026 đã bật 69 trạm mới theo yêu cầu chủ dự án:
worker hiện đồng bộ 19 trạm nước và 55 trạm mưa. Danh sách enabled được đọc lại
mỗi lượt nên không cần sửa worker hoặc khởi động lại backend. Chỉ Dã Viên có
prediction_enabled; thêm metadata sau này vẫn không tự bật trạm. Trạm mưa mới dùng mm
theo quyết định dự án; đơn vị nước mới cũng dùng mm theo xác nhận 14/10/2026.
Migration `0007` đã bổ sung mm cho metadata nước còn NULL. Importer
không ghi observations/forecasts, không đọc hay sửa model/scaler/training data.
Report chỉ chứa ID/count, lưu ở `reports/station_catalog_import.json`, không có
credentials hoặc payload gốc. Mật khẩu/token chỉ nằm trong môi trường/RAM.


Chạy các lệnh dưới đây từ thư mục gốc dự án. Worker là code chính của backend,
được quản lý cùng mã nguồn và chạy riêng với API. Import package không tự gọi
nguồn hoặc kết nối database.


`src/backend/ingestion/worker.py` chạy như process riêng, không chạy bên trong Uvicorn.
Worker local được API quản lý khi `HYDROLOGY_SYNC_AUTOSTART=true`; service
`hydrology-sync.service` đã tắt và bỏ enable để không chạy ngoài dự án. Chu kỳ
start-to-start là **11 phút**; đó là lựa chọn polling, không phải SLA của nguồn.

### Tự bật worker khi backend khởi động (local Linux/WSL)

Đặt `HYDROLOGY_SYNC_AUTOSTART=true` trong `.env` private và khởi động lại API.
API tạo process con bằng cùng Python interpreter, không cần systemd. Worker chạy
một lượt ngay, sau đó theo chu kỳ; tắt API thì gửi SIGTERM và đợi worker đóng an
toàn. Quá 40 giây sẽ kill/reap. Worker kiểm tra process cha mỗi giây để ngừng khi
API chết bất ngờ; request nguồn đang chạy có thể cần đợi timeout trước khi thoát.
File lock `reports/hydrology_sync_worker.lock` được giữ bởi cha/con để ngăn hai
API tạo worker trùng, tự nhả khi chúng thoát. File rỗng không chứa credentials.
Lỗi tạo worker chỉ warning, API vẫn đọc database. Injected engine trong tests mặc
định không tạo process. Docker/worker độc lập giữ flag false.

Worker dùng deadline UTC và kiểm tra lại tối đa mỗi 5 giây khi chờ; sau khi
máy/WSL ngủ, deadline đã qua sẽ được xử lý sớm. SIGCONT đánh thức lần chờ hoặc
xếp một lượt tiếp theo nếu đang đồng bộ, không chạy hai lượt song song. Shutdown
SIGINT/SIGTERM ngắt cả chờ. Khởi động lại backend sẽ nạp code worker mới.

### Cách chạy và quản lý

```bash
# Kiểm tra source, không ghi quan trắc
.venv/bin/python -m src.backend.ingestion --once --dry-run
# Đồng bộ ngay một lượt
.venv/bin/python -m src.backend.ingestion --once
# Xem trạng thái/log của worker tự động
systemctl --user status hydrology-sync.service
journalctl --user -u hydrology-sync.service -n 20
# Dừng hoặc chạy lại
systemctl --user stop hydrology-sync.service
systemctl --user restart hydrology-sync.service
```

Cài systemd **chỉ khi chủ động muốn worker chạy độc lập kể cả ngoài phiên API**
(đặt HYDROLOGY_SYNC_AUTOSTART=false; chỉnh đường dẫn nếu đổi nơi đặt project):

```bash
.venv/bin/python -m pip install -r src/backend/requirements.txt
systemctl --user enable --force --now /home/my_ubuntu/projects/hydrology/src/backend/ingestion/hydrology-sync.service
loginctl enable-linger my_ubuntu
```

Linger giữ user service hoạt động sau khi đăng xuất. Máy/WSL phải đang chạy thì
worker mới có thể gọi nguồn; service sẽ tiếp tục khi môi trường được khởi động lại.
Trong Docker sau này, chạy worker như container riêng bằng cùng lệnh module không có
`--once`, thay cho systemd; không khởi tạo một worker cho mỗi Uvicorn worker.

### Cấu hình private `.env`

- `HYDROLOGY_API_BASE_URL`, `HYDROLOGY_API_EMAIL`, `HYDROLOGY_API_PASSWORD`.
- `HYDROLOGY_SYNC_INTERVAL_MINUTES=11`.
- `HYDROLOGY_SYNC_OVERLAP_HOURS=24`.
- `HYDROLOGY_SYNC_REPORT_PATH`: đường dẫn JSON trạng thái có quyền ghi; mặc định
  `reports/hydrology_sync_status.json` tại thư mục gốc dự án. Có thể ghi đè bằng `--report`.
- `DATABASE_URL` dùng PostgreSQL của backend.
- `REDIS_URL` và `CACHE_TTL_HYDROLOGY_SECONDS=600`: dùng cùng cấu hình với API để
  [làm mới cache `/latest`](../cache/README.md). Không có URL thì tắt cache.

Đổi `.env` thì restart service. Credentials chỉ dùng trong bộ nhớ để xác thực,
không ghi vào code, unit, log hoặc báo cáo và không gửi xuống frontend. Token được
lấy lại mỗi lượt; nếu GET gặp 401 thì đăng nhập lại một lần và thử lại trang đó.

### Quy tắc đồng bộ

- Đọc trạm enabled hoặc sync_enabled có source `hue_dcu` và collection nước/mưa/gió
  được hỗ trợ: 19 trạm nước, 55 trạm mưa và 4 trạm gió; chỉ Dã Viên được dự báo.
- Bắt đầu ở timestamp mới nhất **riêng từng trạm** trừ 24 giờ, đến lúc lượt chạy
  bắt đầu. Lần đầu sau dữ liệu cũ sẽ tự bắt kịp khoảng còn thiếu, không chỉ lấy
  bản ghi cuối. Nếu một trạm chưa có dữ liệu thì chỉ lấy 24 giờ gần nhất; lịch sử
  cũ hơn cần backfill riêng.
- Phân trang theo `id` tăng dần và `id > cursor`; tiếp tục đến trang rỗng ngay cả
  khi source giới hạn kích thước trang. Đã kiểm tra live với cả nước và mưa.
- Giữ nguyên đơn vị, timestamp UTC và giá trị 0/âm/thiếu hợp lệ. Từ chối schema,
  station/time bounds, số không hữu hạn, ID không tăng hoặc timestamp trùng gây
  nhập nhằng; không tự bỏ qua hàng lỗi để báo thành công.
- Tải/validate cả nhóm trước khi ghi. UPSERT toàn bộ nhóm trong một transaction;
  lỗi một trạm rollback nhóm. Chỉ UPDATE khi source values đổi: giữ `received_at`
  và tránh đổi `updated_at` khi hàng không thay đổi.
- Khóa advisory `(20261012, 2)` dùng chung với importer SQLite, tránh hai process
  ghi quan trắc chồng nhau. Lượt đang bận thì lượt mới skip, không chạy trùng.
- HTTP timeout connect/read 10/30 giây; tối đa 3 attempts với lỗi mạng/429/5xx
  phù hợp. Retry-After được giới hạn 30 giây. Lượt thất bại giữ DB cũ và thử lại
  ở chu kỳ kế tiếp; không giả lập dữ liệu mới.
- Cửa sổ đọc chồng 24 giờ chỉ bắt các sửa đổi nằm trong cửa sổ đó. Khi cần đối
  soát sửa đổi cũ hơn, tăng overlap cho một lượt thủ công/backfill. Không tự xóa
  quan trắc nếu nguồn không còn trả về một hàng.
- Pipeline chỉ cập nhật quan trắc. Nó không train model, tạo dự báo mới hoặc sửa
  SQLite/Train/Val/Test/checkpoint/scaler/MLflow.
- Sau khi transaction commit, xóa cache `/latest` của các trạm đã thêm/sửa dữ
  liệu. Hàng không thay đổi, dry-run và rollback không xóa cache. Redis lỗi không
  làm thất bại việc lưu quan trắc; báo cáo ghi `latest_cache_invalidation=failed`
  và cache cũ hết hạn theo TTL. Cache form/dự báo chưa nằm trong task này.

### Báo cáo và kiểm tra

`reports/hydrology_sync_status.json`: lượt tự động gần nhất và thời điểm chạy tiếp theo.
Báo cáo cũ trước khi chuyển worker vẫn nằm trong `scripts/local/`. Các JSON trạng
thái không nằm trong package và không đưa lên Git. Khi triển khai, có thể đặt
`HYDROLOGY_SYNC_REPORT_PATH` vào thư mục/volume có quyền ghi.

```bash
.venv/bin/python -m unittest discover -s src/backend/ingestion/tests -p test_worker.py -v
```

14 test kiểm tra schema, phân trang, 401/retry, hủy, UPSERT, rollback nhóm, khóa
chống chồng và invalidation sau commit. Test database rollback dữ liệu thử. Lượt đầu thật thêm 9460 quan trắc,
không cập nhật 725 hàng đọc chồng; lượt tự động sau đó xác nhận không tạo trùng.
