# Redis cache cho dữ liệu mới nhất

Cache áp dụng cho `/latest` và kết quả dự báo **input gốc đã commit**.
Form đầu vào và dữ liệu thực tế vẫn đọc theo luồng hiện có.

Kết quả dự báo dùng key `hydrology:v1:forecast:{station}:{pipeline}:{input_hash}`,
TTL dùng chung `CACHE_TTL_HYDROLOGY_SECONDS`. Chỉ chứa output/ID đã lưu; không có
feature, overrides hay trạng thái stale. Mỗi POST một mốc chuẩn bị input hiện tại
trước, kiểm tra Redis → DB → inference/save khi miss. Đổi dữ liệu/model/scaler
đổi hash nên không dùng nhầm entry; Redis hỏng fallback DB, không chặn lưu dự báo.
Dự báo có overrides bỏ qua cả cache và persistence.

Chế độ khoảng đọc hàng loạt bản dự báo lịch sử từ DB trước, tránh dựng lại 96×29
cho mỗi mốc đã có. Chỉ mốc còn thiếu đi qua luồng cache đầy đủ. Không tự replay
các bản lịch sử khi dữ liệu nguồn được sửa. Cache chỉ publish sau commit thành công.

## Cấu hình

Cài dependencies backend và đặt trong `.env` private của cả API và worker:

```bash
REDIS_URL=redis://127.0.0.1:6379/0
CACHE_TTL_HYDROLOGY_SECONDS=600
```

Không có `REDIS_URL` thì tắt cache. TTL phải là số nguyên dương. Client Redis
được tạo khi process khởi động; không bắt buộc Redis đang chạy để API khởi động.
Thời gian chờ kết nối/command mặc định 0.5 giây, không tự retry. Không đặt query
parameters ghi đè timeout trong `REDIS_URL` nếu muốn giữ giới hạn này.

Sau khi thay đổi code hoặc `.env`, khởi động lại API và worker:

```bash
# API: dừng process Uvicorn đang chạy bằng Ctrl+C rồi chạy lại
.venv/bin/python -m uvicorn src.backend.main:app --env-file .env --host 127.0.0.1 --port 8000
systemctl --user restart hydrology-sync.service
```

Redis local chỉ nên lắng nghe trên máy nội bộ. Khi dùng Docker sau này, API và
worker kết nối cùng service Redis trong mạng container; không public cổng Redis.

## Cách hoạt động

1. API đọc thông tin trạm từ PostgreSQL để xác nhận trạm tồn tại và lấy đúng đơn vị.
2. Nếu Redis có quan trắc hợp lệ, API bỏ qua truy vấn quan trắc mới nhất. Nếu chưa
   có, API đọc PostgreSQL rồi lưu giá trị vào Redis với TTL mặc định **600 giây**.
3. API tính lại `data_age_minutes` và `is_stale` theo giờ hiện tại ở mỗi request.
   Không lưu hai trường này vào cache. Response API giữ nguyên cấu trúc.
4. Worker chỉ xóa cache của trạm có dữ liệu được thêm hoặc sửa, **sau khi toàn bộ
   transaction đã commit**. Request kế tiếp sẽ đọc lại database và điền cache.
   Dry-run, rollback và hàng không thay đổi không xóa cache.

Key quan trắc: `hydrology:v1:latest:{station_id}`. Giá trị chứa station ID, giá trị
đo gốc, thời gian đo/nhập và trạng thái quan trắc. Thông tin trạm được đọc trực
tiếp nên thay đổi đơn vị hoặc xóa trạm không bị che bởi cache.

Mã phiên bản cache (`cache_version`) đổi khi xóa cache. Nó vẫn lưu tại key
`hydrology:v1:latest:{station_id}:generation` để tương thích với các process đang chạy. Hai lệnh
Lua nhỏ giúp kiểm tra và cập nhật Redis nguyên tử: request đã đọc DB trước lần
xóa cache không được ghi lại kết quả cũ sau đó. Key mã phiên bản cache không có TTL;
mỗi trạm đã được invalidation chỉ có một key nhỏ. Đây không phải khóa chống
request trùng; task này chưa thêm cơ chế đó hoặc Redis Cluster.

Redis lỗi thì API đọc PostgreSQL; worker vẫn hoàn thành việc lưu quan trắc.
Nếu xóa cache thất bại, báo cáo worker ghi `latest_cache_invalidation=failed`;
cache cũ còn tồn tại tối đa đến TTL. Worker dừng giữa commit và invalidation
cũng có giới hạn TTL này. Các công cụ ghi DB khác chưa nối invalidation nên
cần xóa cache tương ứng hoặc chờ TTL sau khi sửa dữ liệu thủ công.

PostgreSQL lỗi vẫn trả 503, vì API luôn đọc thông tin trạm từ database.
Entry sai schema/trạm, giá trị không hữu hạn hoặc timestamp ở tương lai được
coi là cache miss. Không cache 404 hoặc lỗi dịch vụ. TTL không chứng minh nguồn
công bố dữ liệu mỗi 10 phút; worker vẫn polling mỗi **11 phút**.

## Kiểm tra

```bash
.venv/bin/python -m unittest discover -s src/backend/api/tests -v
.venv/bin/python -m unittest discover -s src/backend/cache/tests -v
REDIS_TEST_URL=redis://127.0.0.1:6379/15 .venv/bin/python -m unittest discover -s src/backend/cache/tests -v
.venv/bin/python -m unittest discover -s src/backend/ingestion/tests -v
```

Kiểm tra Redis thật dùng key ngẫu nhiên và chỉ xóa key do test tạo; không dùng
`FLUSHDB`. Kiểm tra PostgreSQL của worker rollback dữ liệu thử; database cần có
schema và năm trạm seed của backend. Log DEBUG của `src.backend.cache.latest`
ghi hit/miss/invalidation; lỗi Redis chỉ ghi loại exception, không ghi URL/mật khẩu.
