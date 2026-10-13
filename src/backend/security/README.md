# Bảo vệ API công khai

Các giới hạn này áp dụng trước khi endpoint xử lý request. Không cần tài khoản
người dùng để sử dụng API. Mức mặc định là cấu hình khởi đầu, cần đo lại khi triển khai.

## Giới hạn hiện tại

| Nội dung | Mặc định | Khi vượt giới hạn |
|---|---|---|
| Request theo IP | 120 request / cửa sổ 60 giây | 429 `RATE_LIMITED` |
| POST Predict theo IP | 20 request / cửa sổ 60 giây, nằm trong quota chung | 429 `RATE_LIMITED` |
| Body request | 16 KiB (16.384 byte) | 413 `REQUEST_TOO_LARGE` |
| Thời gian nhận toàn bộ body | 10 giây | 408 `REQUEST_BODY_TIMEOUT` |
| Dự báo đồng thời | 1 request trong mỗi API process | 503 `INFERENCE_BUSY` |
| Số IP trong bộ đếm local | Tối đa 10.000 IP đang có cửa sổ hoạt động | IP mới nhận 429 đến khi có chỗ |

429 và `INFERENCE_BUSY` có header `Retry-After` tính bằng giây. Frontend nên thông
báo và cho phép thử lại sau thời gian đó, tránh tự gửi lại liên tục.

## Cấu hình environment / `.env`

```dotenv
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173,http://localhost:8000,http://127.0.0.1:8000
API_RATE_LIMIT_PER_MINUTE=120
PREDICTION_RATE_LIMIT_PER_MINUTE=20
MAX_REQUEST_BODY_BYTES=16384
REQUEST_BODY_TIMEOUT_SECONDS=10
MAX_CONCURRENT_PREDICTIONS=1
RATE_LIMIT_MAX_CLIENTS=10000
```

Đổi cấu hình thì khởi động lại API. Origins phải khớp scheme, host và port, không
có dấu `/` ở cuối, path hoặc wildcard `*`. Khi public, thay bằng domain frontend
thực tế; nếu dùng Swagger trên domain API, thêm origin đó. Chuỗi rỗng tắt quyền
cross-origin. Mặc định cho phép Vite local và Swagger local.

CORS cho phép GET/POST, header Accept/Content-Type, không dùng credentials. Origin
không được phép nhận 403 trước khi endpoint chạy; preflight không hợp lệ nhận
400 từ CORSMiddleware. Preflight được xử lý riêng và không tiêu thụ quota API.
Các lỗi API bình thường có request ID; CORS cho phép frontend đọc `X-Request-ID`
và `Retry-After`, kể cả 413/429/503 và lỗi route 500 đã được xử lý an toàn.
Request không có Origin (ví dụ curl) vẫn được phép và vẫn chịu các giới hạn.

## Rate limiting và IP

Redis dùng `REDIS_URL` của backend, nhưng bộ đếm tách namespace
`hydrology:rate:v1:` khỏi cache quan trắc. Lua cập nhật quota/TTL nguyên tử; IP
được băm khi tạo key và không chứa station ID hoặc URL do người dùng nhập.
Cửa sổ bắt đầu từ request đầu của IP, tồn tại 60 giây và không gia hạn theo hit.

Bộ đếm local luôn chạy cùng bộ đếm Redis. Nếu Redis lỗi hoặc không được cấu hình,
backend vẫn giữ quota local, không mở hết giới hạn. Bảng local được khóa khi có
nhiều thread, tự dọn cửa sổ hết hạn, và không xóa IP đang hoạt động để nhường chỗ
cho IP mới. Redis timeout 0.5 giây, không retry; không đặt query parameters trong
REDIS_URL ghi đè timeout này. Khi Redis phục hồi, cả hai quota lại được áp dụng;
quota local có thể chặt hơn quota Redis. Request sai dữ liệu cũng tiêu thụ quota.

Middleware lấy địa chỉ từ `scope.client`, không tự đọc X-Forwarded-For, Forwarded
hay X-Real-IP. Uvicorn có thể xử lý proxy headers trước middleware nên cấu hình
server phải đúng. Khi chạy trực tiếp local, dùng:

```bash
.venv/bin/python -m uvicorn src.backend.main:app --env-file .env \
  --host 127.0.0.1 --port 8000 --workers 1 --no-proxy-headers
```

Sau này sau reverse proxy, chỉ bật proxy headers cho đúng IP proxy tin cậy, không
dùng `--forwarded-allow-ips='*'`. Proxy phải thay các header IP do client gửi bằng
IP thực. Nếu nhiều người chung một IP NAT, họ dùng chung quota.

## Body và tài nguyên inference

Body được đếm theo số byte nhận thực tế, gồm trường hợp chunked hoặc Content-Length
khai nhỏ hơn. Bộ đệm không vượt giới hạn byte; body chậm có deadline trước khi
parse JSON. Body nén bị từ chối bằng 415 để tránh dữ liệu giải nén lớn vượt giới hạn.

Một semaphore dùng chung cho mọi trạm trong process bao quanh việc lấy/tải model,
chuẩn bị đầu vào, dự báo và lưu kết quả. Khi hết chỗ, request bị từ chối ngay, không
xếp hàng chờ GPU. Chỗ được trả lại trong `finally`, kể cả khi model/database báo lỗi.
Request dùng lại dự báo PostgreSQL cũng qua giới hạn này để giữ luồng đơn giản.

**Triển khai GPU hiện tại dùng một API process (`--workers 1`) và một replica.**
Quota Redis có thể chia sẻ giữa các process, còn semaphore và model đã tải thuộc
từng process. Thêm process/replica có thể tải thêm model và tăng tổng inference
đồng thời; cần thiết kế giới hạn chung trước khi mở rộng trên cùng GPU.

Deadline 10 giây chỉ áp dụng lúc nhận body. Không giả vờ hủy một thread/GPU đang
chạy rồi trả lại semaphore, vì như vậy request tiếp theo có thể chạy chồng lên
công việc cũ. Hard timeout cho model cần cơ chế cách ly/worker riêng nếu có nhu cầu.
HTTPS, giới hạn kết nối và timeout ở reverse proxy thuộc bước triển khai sau này.

## Kiểm tra

```bash
.venv/bin/python -m unittest discover -s src/backend/security/tests -v
REDIS_TEST_URL=redis://127.0.0.1:6379/15 .venv/bin/python -m unittest discover -s src/backend/security/tests -v
.venv/bin/python -m unittest discover -s src/backend/api/tests -v
```

Các test cover allowlist/preflight, lỗi có CORS/request ID, spoofed IP header,
body chunked/timeout, quota local/Redis, bảng IP có giới hạn, request dự báo chồng
nhau và trả lại semaphore sau lỗi. Test Redis dùng namespace riêng và chỉ xóa
key của test. Test inference dùng model giả, không train hoặc chiếm GPU.
