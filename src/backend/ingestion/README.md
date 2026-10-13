# Pipeline cập nhật dữ liệu quan trắc

Chạy các lệnh dưới đây từ thư mục gốc dự án. Worker là code chính của backend,
được quản lý cùng mã nguồn và chạy riêng với API. Import package không tự gọi
nguồn hoặc kết nối database.


`src/backend/ingestion/worker.py` chạy như process riêng, không chạy bên trong Uvicorn.
Service local: `hydrology-sync.service`, đã enable trong systemd user. Chu kỳ
start-to-start là **11 phút**; đó là lựa chọn polling, không phải SLA của nguồn.

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

Cài trên môi trường có systemd user (điều chỉnh đường dẫn trong unit nếu đổi nơi đặt project):

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

- Đọc các trạm đang enabled có source `hue_dcu` và collection nước/mưa hỗ trợ từ
  PostgreSQL; hiện có Dã Viên và bốn trạm mưa của model.
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
