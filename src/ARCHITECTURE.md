# Water Level Forecasting System
**Software Architecture & System Design Plan — FINAL v1.0**
**Ngày cập nhật:** 12/10/2026  

**Trạng thái:** Final architecture baseline — các phần phụ thuộc dữ liệu thực sẽ được xác minh trong Phase 1  

**Mục tiêu:** Tài liệu độc lập và là nguồn yêu cầu chính để AI Coding Agent lập kế hoạch và triển khai; không phụ thuộc các bản kiến trúc trước. Agent phải đọc repository/Postman/model artifacts để xác minh giả định còn mở.

> **Phân biệt:** **[ĐÃ DUYỆT]** là quyết định đã được chủ dự án chấp thuận; **[ĐỀ XUẤT]** là hướng triển khai có thể điều chỉnh nếu nêu rõ lý do; **[CẦN KIỂM CHỨNG]** là nội dung Agent phải khảo sát từ repository, Postman collection, dataset hoặc model artifact. Không được biến giả định thành sự thật trong code.

---

## Project Resources & Model Configuration

### 1. Best Trained Model

Mô hình tốt nhất hiện tại là **Residual LSTM**, được sử dụng làm mô hình mặc định cho hệ thống dự báo mực nước.

**Model checkpoint:**

```text
models/1h_top4/1h/ResidualLSTM_water_level+rain+flood_season_best.pt
```

**Training Configuration:**

| Parameter             | Value                                                   |
| --------------------- | ------------------------------------------------------- |
| dataset_dir           | `/home/my_ubuntu/projects/hydrology/train_data/1h_top4` |
| train_method          | `residual_lstm`                                         |
| horizon               | `1h`                                                    |
| feature_set           | `water_level+rain+flood_season`                         |
| seed                  | `42`                                                    |
| batch_size            | `128`                                                   |
| num_epochs            | `150`                                                   |
| early_stopping        | `50`                                                    |
| gradient_accumulation | `1`                                                     |
| sequence_length       | `96`                                                    |
| input_size            | `29`                                                    |
| output_size           | `5`                                                     |
| learning_rate         | `0.01`                                                  |
| weight_decay          | `0.001`                                                 |
| device                | `cuda`                                                  |
| num_layers            | `3`                                                     |
| hidden_size           | `32`                                                    |
| lstm_dropout          | `0.5`                                                   |
| input_dropout         | `0.2`                                                   |

**Implementation Notes:**

- AI Agent phải kiểm tra kiến trúc `ResidualLSTM` trong repository trước khi load checkpoint.
- Sử dụng đúng `sequence_length = 96`, `input_size = 29` và `output_size = 5`.
- Xác định chính xác thứ tự 29 input features, ý nghĩa của 5 output values và các scaler/preprocessing artifacts từ training pipeline.
- Không tự ý thay đổi kiến trúc mô hình hoặc preprocessing pipeline khi tích hợp vào Backend.

### 2. Hydrology API Resources

Các file Postman Collection và Environment được lưu trong repository:

**Postman Collection:**

```text
data/HueNgapLut.postman_collection.json
```

**Postman Environment:**

```text
data/HueNgapLut.postman_environment.json
```

Mật khẩu: .env file

AI Agent phải đọc hai file này trong Phase 1 — Discovery để xác định:

- Các API endpoints cần thiết cho hệ thống.
- Request parameters và response schemas.
- Cách lấy dữ liệu mới nhất và dữ liệu lịch sử.
- Tần suất cập nhật dữ liệu thực tế.
- Authentication và các environment variables cần thiết.
- Những dữ liệu cần thu thập để phục vụ 29 input features, hiển thị thông tin trạm và các biểu đồ.


---

## 1. System Overview
### 1.1. Mục đích
Xây dựng ứng dụng web tra cứu, trực quan hóa dữ liệu thủy văn và dự báo mực nước bằng mô hình Machine Learning đã huấn luyện. Người dùng có thể tương tác với bản đồ, chọn trạm, xem mực nước mới nhất, xem dữ liệu lịch sử, điều chỉnh input và thực hiện dự báo.

Ứng dụng phục vụ quan sát và thử nghiệm dự báo, **không phải** hệ thống cảnh báo thiên tai chính thức, không khẳng định độ chính xác được bảo đảm.

### 1.2. Phạm vi MVP
- Hỗ trợ trạm Dã Viên (Sông Hương, Huế) như trạm mẫu ban đầu.

- Bản đồ tương tác: pan, zoom, markers, tooltip/popup; mặc định focus khu vực Huế, vẫn điều hướng được khu vực Việt Nam.

- Sidebar liệt kê các trạm được hỗ trợ; chọn trạm đồng bộ bản đồ và bảng chi tiết.

- Hiển thị metadata, mực nước mới nhất, thời điểm quan trắc, ngưỡng báo động 1/2/3 và tình trạng dữ liệu.

- Tra cứu dữ liệu quan trắc quá khứ, bao gồm những khoảng có dữ liệu và khoảng thiếu.

- Tự điền dữ liệu model input (29 features nếu metadata xác nhận) khi mở web/chọn trạm/chọn thời điểm; cho phép chỉnh sửa trước khi nhấn **Predict**.

- Dự báo mới nhất và Historical Replay cho timestamp bất kỳ có đủ **N bản ghi hợp lệ trước đó**.

- Biểu đồ mực nước quan trắc, dự báo, đối chiếu thực tế sau dự báo (nếu có), và biểu đồ mưa khi cần.

- Backend FastAPI, PyTorch model runtime, PostgreSQL lưu lịch sử, Redis cache, Docker Compose để triển khai.

### 1.3. Ngoài phạm vi
Không triển khai đăng nhập, đăng ký, tài khoản, lịch sử riêng theo người dùng, thông báo lũ tự động, thanh toán, Microservices, Kafka, Kubernetes hay Load Balancer ở MVP. Không đưa User Authentication vào kế hoạch tương lai nếu chưa phát sinh nhu cầu mới. API công khai vẫn cần các biện pháp bảo vệ phù hợp.

### 1.4. Technology Stack
| Thành phần | Công nghệ | Vai trò |
|---|---|---|
| Frontend | React + Vite | SPA và công cụ build |
| Styling | Tailwind CSS | Giao diện responsive |
| Interactive Map | Leaflet + React Leaflet | Map, station marker, zoom, fly-to |
| Charts | Recharts | Mực nước, dự báo, mưa |
| Backend | FastAPI + Pydantic | HTTP API, validation |
| ML | PyTorch | Inference theo model artifact đã train |
| Primary Data Storage | PostgreSQL | Dữ liệu trạm và quan trắc lịch sử |
| Cache | Redis | Hydrology Cache, Prediction Input Cache, Prediction Cache |
| Deployment | Docker + Docker Compose | Khởi chạy và quản lý containers |
| Production Web | Nginx | Serve React build, HTTPS/reverse proxy khi cấu hình |

**Chưa thêm công nghệ chỉ vì phổ biến.** Việc sử dụng một thư viện UI hay một số module phụ có thể điều chỉnh khi triển khai, với điều kiện ghi lại lý do.

---

## 2. Functional Requirements
### FR-01. Interactive Map & Station Selection
- Hiển thị bản đồ có thể kéo, phóng to/thu nhỏ, khởi tạo ở khu vực Huế.

- Hiển thị marker cho từng trạm `enabled = true`; biểu tượng đặc trưng cho trạm mực nước.

- Hover marker trên desktop hiển thị tooltip; click/tap mở popup hoặc Station Detail Panel.

- Sidebar hiển thị danh sách trạm; click một mục sẽ chọn trạm, tự động focus marker tương ứng, cập nhật Station Detail.

- Dùng chung state `selectedStationId` cho Map, Sidebar và Detail; không hardcode tên hoặc tọa độ ở UI.

- Phân biệt `enabled` (trạm được hỗ trợ), `data_status` (dữ liệu mới/thiếu/cũ) và `prediction_available` (có thể dự báo). Không đánh đồng trạm online với có dữ liệu hợp lệ.

### FR-02. Station Info / Hover / Click
**Nội dung ví dụ cho trạm Dã Viên (do chủ dự án cung cấp):**
> **Trạm đo mực nước tự động**  

> **Tên trạm:** Trạm Dã Viên (Sông Hương)  

> **Tọa độ:** 16.458944, 107.571361  

> **Vị trí:** phường Thuận Hóa, thành phố Huế  

> **Mực nước:** **0.51 m**  

> **Mức báo động 1:** 1.00 m  

> **Mức báo động 2:** 2.00 m  

> **Mức báo động 3:** 3.50 m

**Quy ước dữ liệu:** `0.51 m` chỉ là **giá trị minh họa người dùng cung cấp**, không phải số liệu quan trắc thực tế tại thời điểm hệ thống chạy. Khi có API/DB, mực nước phải được lấy động từ bản ghi mới nhất và hiển thị `observed_at`. Tọa độ, địa chỉ và ngưỡng cũng được nhập ban đầu theo thông tin người dùng; Agent xác minh metadata nguồn trước khi công bố Production. Đã chuẩn hóa lỗi lặp từ "thành phố Thành phố" trong ví dụ gốc.

- Tooltip trên hover ưu tiên ngắn gọn: tên, mực nước, trạng thái dữ liệu, thời gian quan trắc.

- Popup/Detail khi click hiển thị đầy đủ vị trí, tọa độ, mực nước, ba ngưỡng, biểu đồ và thao tác dự báo.

- Ngưỡng cảnh báo là **metadata từng trạm**, có thể khác nhau; mức trạng thái biểu diễn bằng màu và văn bản nhưng không thay thế dữ liệu nguồn.

- Nếu mực nước chưa có hoặc đã cũ, hiển thị "Chưa có dữ liệu" hoặc "Dữ liệu cũ" thay vì dùng số liệu giả.

### FR-03. Latest Hydrology Data
- Khi chọn trạm, Frontend lấy dữ liệu quan trắc mới nhất từ Backend và tự điền vào Input Form.

- Backend ưu tiên Cache Redis và PostgreSQL; nguồn bên ngoài được thu thập bởi Data Ingestion Job.

- Kèm `observed_at`, `received_at` (nếu có), đơn vị đo, trạng thái dữ liệu và thông tin thiếu dữ liệu.

- Chỉ lấy và lưu các trường cần cho biểu đồ, station detail và Model Input; không sao chép toàn bộ payload từ API khi không cần thiết.

- Nếu người dùng đã chỉnh sửa input, các lần refresh không tự ghi đè vào form; cung cấp thao tác "Reset to latest".

- **[CẦN KIỂM CHỨNG]** Agent đọc Postman collection trong repository, xác định endpoints, kiểu dữ liệu, quy tắc phân trang, thời điểm cập nhật thực tế, giới hạn API và khả năng backfill lịch sử.

### FR-04. Historical Observations
- Khi chọn một trạm nước, hiển thị bảng dữ liệu hiện tại và quá khứ của trạm. Bảng phân trang theo thời gian, ghi rõ thời điểm và đơn vị đo.

- Khi mở trạm, tự chọn thời điểm hợp lệ mới nhất để điền Input Form. Khi người dùng bấm một dòng lịch sử, lấy timestamp của dòng đó làm `reference_time` và gọi `prediction-inputs` để điền lại form; không dùng riêng giá trị mực nước của dòng để thay cho toàn bộ 29 feature.

- Bảng phục vụ dự báo sử dụng các mốc theo giờ phù hợp với mô hình. Phân biệt giá trị quan trắc gốc với giá trị tổng hợp theo giờ; không hiển thị chúng như cùng một loại số liệu.

- Sau khi input được điền, người dùng có thể giữ nguyên hoặc chỉnh sửa rồi nhấn Predict. Nếu thời điểm chọn chưa đủ lịch sử đầu vào, hiển thị lý do và không cho chạy dự báo. Request cũ trả về muộn không được ghi đè form của dòng vừa chọn.

- Cho chọn khoảng thời gian xem lịch sử (ví dụ 24 giờ/7 ngày/custom range).

- Hiển thị biểu đồ mực nước theo thời gian và đánh dấu vùng thiếu quan trắc khi có dữ liệu bị gián đoạn.

- Có thể chọn một `reference_time` quá khứ làm thời điểm dự báo tái lập.

- Không giả định từ mốc ngày đầu tiên của Database trở đi lúc nào cũng đủ dữ liệu.

### FR-05. Model Input & Overrides
- Chỉ hiển thị và chấp nhận feature mà model thực tế hỗ trợ; cấu hình feature và thứ tự feature lấy từ Model Metadata.

- Dữ liệu có thể gồm mực nước, lượng mưa và các feature khác **nếu mô hình đã được huấn luyện với chúng**.

- User Overrides ở MVP áp dụng **vào timestep cuối của chuỗi được chọn**; mọi feature không được sửa giữ nguyên theo quan trắc gốc. **[CẦN KIỂM CHỨNG]** Tính tương thích và ý nghĩa với pipeline thực tế trước khi chốt tính năng.

- Không ghi đè hay thay đổi dữ liệu gốc trong PostgreSQL bởi thao tác input của user; dữ liệu chỉnh sửa chỉ tồn tại ở Frontend/request và có thể xuất hiện trong Redis Prediction Cache.

### FR-06. Forecasting
- `Latest Forecast`: sử dụng thời điểm quan trắc có sẵn mới nhất.

- `Historical Replay`: chọn `reference_time` quá khứ, lấy **N bản ghi đầy đủ gần nhất với `observed_at <= reference_time`** theo timestamp tăng dần khi đưa vào mô hình.

- N được lấy từ `model_config.sequence_length` (ví dụ N=96), không hardcode trong ứng dụng.

- **KHÔNG yêu cầu N quan trắc liên tiếp từng giờ.** Có thể có khoảng thiếu giờ giữa các bản ghi được chọn; không tự resample hoặc điền thêm hàng để buộc liên tục.

- Chỉ cho inference khi thu thập đủ N bản ghi **hợp lệ về các feature bắt buộc**. Nếu ít hơn N bản ghi, trả lỗi rõ ràng.

- Mốc dự báo xuất ra phải dựa trên **thời điểm quan trắc cuối cùng thực sự được đưa vào mô hình** (`anchor_observed_at`), không tự động lấy `reference_time` nếu hai mốc khác nhau.

- **Không trả warning `IRREGULAR_OBSERVATION_INTERVALS` và không cảnh báo người dùng chỉ vì khoảng cách giữa các bản ghi không đều.** Có thể ghi nhận thống kê gap nội bộ để đánh giá chất lượng mô hình, không coi đó là lỗi dữ liệu.

- Dữ liệu training thực tế cũng có những khoảng thời gian không đều; Agent phải giữ cùng chính sách chọn chuỗi như pipeline huấn luyện. Việc khảo sát ảnh hưởng của khoảng thời gian giữa bản ghi chỉ là đánh giá ML offline, không tạo cảnh báo irregular interval cho người dùng.

- Inference trên dữ liệu tương lai không được vượt những horizon mà model hỗ trợ. Không dùng các quan trắc sau anchor để tạo feature đầu vào (tránh look-ahead leakage).

### FR-07. Charts
1. **Observed vs Predicted Water Level:** Line chart thể hiện đường quan trắc trước anchor và các điểm dự báo theo forecast horizon. Đánh dấu rõ đường phân cách quan trắc / dự báo; ở Historical Replay có thể overlay quan trắc thực tế sau anchor để so sánh, **chỉ trên biểu đồ đánh giá**, tuyệt đối không đưa vào input model.

2. **Historical Water Level:** Line hoặc scatter-line chart theo khoảng ngày do người dùng chọn. Trục tung, tooltip và sai số mực nước dùng **mm**; không có chế độ chuyển sang cm.

3. **Rainfall History (tùy chọn):** Bar chart khi nguồn dữ liệu/model sử dụng lượng mưa; rõ đơn vị mm và khoảng tích lũy (1h/3h/6h...).

Không thêm charts mang tính trang trí. Dữ liệu chart phải có schema nhất quán và phân biệt `observed`, `predicted`, `actual_after_anchor`.

### FR-07.1. Nút Visualize và chế độ xem dự báo

- Đặt nút **Visualize** cạnh **Predict**, dùng chung trạm và mốc thời gian đang chọn. Bấm dòng dữ liệu lịch sử vừa chọn mốc cho Visualize, vừa gọi `prediction-inputs` để điền vùng input như luồng hiện tại.
- Visualize hiển thị dự báo từ quan trắc gốc đã lưu. Nếu chưa có bản phù hợp tại mốc đang chọn, lấy/tạo dự báo qua luồng `POST /predictions` hiện có rồi hiển thị; API đọc biểu đồ không tự chạy inference. Không âm thầm dùng bản của model/pipeline khác.
- Biểu đồ dự báo có chỉnh sửa input, nếu được hiển thị sau Predict, phải ghi rõ là kịch bản chỉnh sửa; không trộn với đường dự báo từ quan trắc gốc hoặc lưu overrides lâu dài.
- Chỉ dùng **mm** cho giá trị mực nước, trục tung, tooltip và sai số. Không thêm nút đổi mm/cm.

**Chế độ 1 — chọn một mốc:**
- Lấy một dự báo xác định bằng `forecast_id`, gắn với đúng model/pipeline và `anchor_observed_at`. Cửa sổ hiển thị bắt đầu từ anchor thực tế, không từ mốc người dùng nhập nếu hai mốc khác nhau.
- Với model Dã Viên hiện tại, đường dự báo có **5 điểm** tại **+1, +3, +6, +12 và +24 giờ**. Đánh dấu rõ các điểm này; đường nối chỉ hỗ trợ nhìn xu hướng, không tạo hoặc mô tả thêm 19 dự báo theo giờ.
- Đường thực tế có **24 mốc theo giờ**, từ **anchor + 1 giờ đến anchor + 24 giờ**, khi có dữ liệu. Mỗi giá trị là trung bình quan trắc nước trong `(t - 1 giờ, t]`, đúng quy tắc tổng hợp khi training.
- Hai đường dùng chung trục thời gian và trục mực nước mm. Tại 5 thời điểm có dự báo, tooltip có thể hiển thị dự đoán, thực tế và độ lệch nếu có thực tế.
- Mốc thực tế còn trong tương lai hoặc không có lần đo hợp lệ để tính trung bình được giữ là `null`; không thay bằng 0 hoặc nội suy. Không nối qua phần thiếu theo cách khiến người xem hiểu đã có quan trắc.
- Khi chọn dữ liệu mới nhất, vẫn hiển thị 5 điểm dự báo đến +24 giờ; đường thực tế chỉ có những mốc đã quan trắc được sau anchor. Nếu chưa có thì chỉ hiện đường dự báo, kèm trạng thái chưa có dữ liệu thực tế.
- Khi mở rộng sang model khác, số điểm dự báo và horizon lấy từ metadata/bản lưu, không cố định 5 điểm hoặc 24 giờ cho mọi model.

**Chế độ 2 — chọn một khoảng bắt đầu/kết thúc:**
- Cho nhập start/end để vẽ toàn bộ các điểm đã lưu trong khoảng. Đọc hết các trang dữ liệu cần thiết; không chỉ vẽ trang đầu mặc định 100 điểm.
- Chọn một horizon để vẽ một đường dự báo liên tục theo các `forecast_at` trong khoảng, ví dụ chuỗi dự báo +1 giờ từ nhiều anchor khác nhau, cùng đường thực tế theo giờ.
- Phân biệt rõ chế độ này với dự báo 24 giờ từ một anchor: không ghép các điểm khác horizon/anchor thành một đường rồi gọi đó là dự báo từ một mốc duy nhất.
- Giữ cùng trạm, model và pipeline trong một series; ghi rõ horizon đang xem. Dữ liệu thực tế thiếu vẫn để trống, dữ liệu dự báo chưa được tạo không được tự điền giả.

### FR-08. Caching
- Hydrology Cache: phục vụ dữ liệu/latest window thường được đọc nhiều.

- Prediction Input Cache: cache dữ liệu đầu vào gốc đã chuẩn bị cho form theo trạm và thời điểm; không lưu overrides của user.

- Prediction Cache: trả kết quả inference đối với cùng model/version/chuỗi input/override.

- Dự báo từ dữ liệu quan trắc gốc được lưu lâu dài trong PostgreSQL để dùng lại và vẽ biểu đồ so sánh. Redis là cache tạm của các kết quả đó. Dự báo có overrides chỉ được cache tạm, không lưu lâu dài hoặc đưa vào đường dự báo từ dữ liệu gốc.

- Nếu phát hiện thêm cơ hội cache có lợi, Agent cần báo cáo chi phí-lợi ích và đề xuất thay đổi; **không tự ý bổ sung nhiều cache layer**.

### FR-09. Automatic Input Population
Hệ thống phải tự động điền các trường đầu vào dự báo từ dữ liệu quan trắc khi người dùng mở ứng dụng, chọn trạm hoặc thay đổi thời điểm dự báo.

Luồng xử lý:

1. Khi mở website, hệ thống tự động chọn trạm Dã Viên và thời điểm có dữ liệu hợp lệ mới nhất và điền vào biểu mẫu sẵn.

2. Khi người dùng chọn một thời điểm khác, Frontend gửi yêu cầu đến Backend để lấy dữ liệu input tại thời điểm đó.

3. Backend lấy `sequence_length` bản ghi hợp lệ gần nhất (`observed_at <= reference_time`), cho phép gián đoạn theo giờ; trích xuất **29 input features ở timestep cuối** để auto-fill form theo model metadata. Số 29 phải được xác minh bằng model artifact.

4. Frontend tự động điền các giá trị tương ứng vào biểu mẫu.

5. Người dùng có thể chỉnh sửa các field được cấu hình cho phép trước khi nhấn Predict. Chuỗi đầu vào inference đầy đủ có shape logic `N × feature_count`; form mặc định chỉ hiển thị timestep cuối, không yêu cầu người dùng nhập N × 29 giá trị.

- Khi đổi trạm/thời điểm, form được tự điền lại từ dữ liệu tương ứng. Refresh định kỳ không được ghi đè form đã được chỉnh sửa; có thao tác **Reset to observed values**. Request cũ không được ghi đè trạng thái mới nếu trả về muộn.

6. Backend kiểm tra Redis Prediction Input Cache trước khi truy vấn PostgreSQL/chuẩn bị dữ liệu. Cache chứa dữ liệu quan trắc gốc phục vụ form, không chứa giá trị người dùng tự chỉnh sửa; chi tiết ở mục 8.2.

---

## 3. Non-functional Requirements
| Nhóm | Yêu cầu / chỉ tiêu MVP |
|---|---|
| Latency | Hiện tại chưa có yêu cầu về latency, nhưng vẫn cố gắng không để hệ thống quá chậm |
| Performance | Không load model mỗi request; tránh query dữ liệu lịch sử toàn bộ trong một endpoint |
| Reliability | Timeout, retry có giới hạn cho external API; không sửa dữ liệu gốc khi user overrides; fallback khi Redis lỗi |
| Data Integrity | Giữ timezone chuẩn, đơn vị, timestamp, feature order, model/scaler version; không sử dụng dữ liệu tương lai |
| Security | HTTPS production, input validation, CORS allowlist, rate limiting công khai, secret ở server; Redis/Postgres không public Internet |
| Scalability | Một server + Docker Compose đủ cho MVP; Station Registry/Model Config theo `station_id` để mở rộng |
| Availability | Hiển thị dữ liệu stale/không có dữ liệu khi backend/upstream lỗi; không suy đoán mực nước |
| Maintainability | Modular Monolith; interface giữa service và infrastructure; có unit/integration/E2E tests |
| Observability | Structured logs có request ID, metrics cache hits/misses, upstream failures, DB sync lag, inference latency |
| Accessibility | Marker có tương tác bằng click/tap; Sidebar có thể thao tác bàn phím; trạng thái cảnh báo không chỉ bằng màu |

**Bảo vệ ứng dụng công khai không có login:** Rate limiting theo IP hoặc một chiến lược phù hợp, request body size limit, timeout, CORS, kiểm soát tài nguyên inference; không phụ thuộc authentication để chặn lạm dụng.

---

## 4. High-Level Architecture
### 4.1. Architectural Style
**[ĐÃ DUYỆT]** Modular Monolith + Layered Architecture. Frontend là React SPA, Backend FastAPI là một ứng dụng thống nhất nhưng chia module theo nghiệp vụ. Data Ingestion có thể chạy như **process/container riêng dùng chung mã nguồn Backend**, không có nghĩa đã chuyển sang Microservices.
```text

Browser / React SPA (Vite build + Tailwind)

       |  HTTPS / JSON REST API

       v

Nginx (production: static files + reverse proxy)

       |

       v

FastAPI Modular Monolith

  ├─ API Layer (routes / schemas / error handlers)

  ├─ Station Service (registry / metadata / thresholds)

  ├─ Hydrology Service (latest / history / input selection)

  ├─ Prediction Service (prepare / cache / inference / response)

  └─ Infrastructure Adapters

       ├─ PostgreSQL Repository <---- PostgreSQL

       ├─ Redis Cache Client   <---- Redis

       └─ Model Runtime        <---- PyTorch artifacts

Scheduled Ingestion Process (same backend codebase)

       |

       ├─> External Hydrology API (Postman collection)

       └─> Validate / normalize / PostgreSQL UPSERT

                        └─> Invalidate/refresh latest Redis cache

Leaflet map tiles: browser -> configured tile provider (separate from hydrology API)
```

Phân tách **Source of Truth**:

- Hydrology Provider là nguồn gốc upstream của quan trắc.

- PostgreSQL là nguồn quan trắc **nội bộ đã đồng bộ**, dùng cho lịch sử và inference.

- Redis chỉ là bản sao tạm của dữ liệu hoặc kết quả tính toán (Hydrology, Prediction Input, Prediction Result).

- Model artifacts và metadata là nguồn xác định cách preprocess/inference; không dựa vào tham số frontend.

### 4.2. Những quyết định về frontend map
- Dùng React Leaflet + Leaflet cho pan/zoom/marker/popup và `flyTo` khi đổi trạm.

- `MapContainer` tạo một instance map; thay `center` prop sau mount không đủ để điều khiển map. Dùng `MapController` + map instance để focus trạm.

- `selectedStationId` là state chung cho Sidebar, Map Marker và Station Detail Panel.

- Mặc định focus **Dã Viên/Huế** theo tọa độ do chủ dự án cung cấp; không cố tình giới hạn khả năng kéo map ra khỏi vùng Huế.

- Tile Provider phải là cấu hình thay được; giữ attribution và tuân thủ chính sách của nhà cung cấp. OpenStreetMap public tiles không có SLA, không bulk-prefetch tiles.

### 4.3. Database không đồng nghĩa bắt buộc có user account
PostgreSQL lưu station metadata và quan trắc lịch sử vì nhu cầu nghiệp vụ; không lưu user account, user-input history hay dữ liệu định danh. Redis cache dữ liệu input gốc theo trạm và thời gian như dữ liệu tạm thời, không định danh người dùng.

---

## 5. Component Responsibilities & Data Model
### 5.1. Frontend Components (đề xuất, có thể gộp/tách vừa đủ)
```text

App

├─ AppLayout

│  ├─ Header

│  ├─ StationSidebar

│  │  ├─ StationList

│  │  └─ StationListItem

│  └─ MapView

│     ├─ StationMarkers

│     ├─ StationTooltip / Popup

│     └─ MapController (focus/flyTo)

└─ StationDetailPanel

   ├─ StationOverview (metadata + water level + warning levels)

   ├─ ForecastModeSelector (latest / historical)

   ├─ DateTimeSelector + Availability Indicator

   ├─ PredictionForm (prefill + editable overrides + reset)

   ├─ PredictionResult

   └─ Charts

      ├─ WaterLevelForecastChart

      ├─ HistoricalWaterLevelChart

      └─ RainfallChart (optional)
```

**UI state rules:**
- `selectedStationId` dùng chung ở App level hoặc một context nhỏ.

- `observedInputs`, `editedInputs` và `predictionResult` có state riêng; chỉ `editedInputs` thay đổi khi người dùng sửa field. Khi người dùng đổi trạm hoặc thời điểm, khởi tạo lại form từ dữ liệu tương ứng. Không để background refresh ghi đè form dirty.

- Dữ liệu server (`stations`, `latest`, `history`, `predictions`) tách khỏi form state do user đang chỉnh sửa.

- Khi đổi trạm hoặc chuyển Latest/Historical: UI hiển thị loading rõ ràng; yêu cầu cũ không được ghi đè kết quả của trạm mới nếu response về muộn.

- Hover tooltip nhẹ, click/tap mở detail đủ thông tin; giao diện mobile có thể dùng bottom sheet thay Sidebar cố định.

- Không cần Redux cho MVP; sử dụng local state/custom hooks hoặc thư viện query nếu có lý do thực tế.

### 5.2. Backend Modules
| Module | Trách nhiệm |
|---|---|
| API / Schemas | HTTP endpoints, Pydantic validation, error response, request ID |
| Station Service | Danh sách trạm, tọa độ, alarm thresholds, station/model readiness |
| Hydrology Service | Latest/history, normalize dữ liệu theo trạm, lấy N bản ghi hợp lệ tại/trước T |
| Prediction Input Service | Chuẩn bị dữ liệu form tại thời điểm chọn, cache-aside dữ liệu quan trắc gốc, expose 29 giá trị timestep cuối theo model schema |
| Ingestion Service / Job | Khảo sát upstream, backfill, incremental sync, dedup/upsert, sync metadata |
| Prediction Service | Áp overrides, lập cache key, invoke model, trả forecast + trạng thái dữ liệu khi cần (không cảnh báo irregular intervals) |
| Model Runtime | Load model/scalers once, preprocess, inference, inverse-transform, model metadata |
| Cache Adapter | Redis operations/TTL/delete, timeout/fallback; không chứa nghiệp vụ dự báo |
| Persistence Adapter | PostgreSQL queries/transactions, schema migrations |

**Nguyên tắc dependency:** Service làm việc qua protocol/repository interface nơi mang lại ích lợi (Model Predictor, Hydrology Provider, Cache, Observation Repository); tránh các abstraction một lần dùng không cần thiết.

### 5.3. PostgreSQL Schema ban đầu
**Bảng `stations`:**
- `id` (stable ID/slug), `name`, `station_type`, `river_name`, `address`.

- `latitude`, `longitude`.

- `alarm_level_1_m`, `alarm_level_2_m`, `alarm_level_3_m` (nullable nếu trạm thiếu ngưỡng xác minh).

- `enabled`, `prediction_enabled`, `source_station_id` (nếu cần ánh xạ upstream).

- Khóa chính `id`.

**Bảng `observations`:**
- `station_id`, `observed_at` (TIMESTAMPTZ; lưu UTC, hiển thị Asia/Ho_Chi_Minh), `received_at`.

- `water_level_m` và các biến thực tế có trong nguồn/model, ví dụ `rainfall_1h_mm` (nullable khi upstream không cung cấp).

- Metadata cần thiết cho provenance/quality: `source_name`, `quality_status`, optional `source_record_id`.

- Unique constraint thích hợp để UPSERT không tạo dữ liệu trùng; thường `(station_id, observed_at)` nếu mỗi timestamp một observation tổng hợp.

- Index `(station_id, observed_at DESC)` để tìm N bản ghi gần nhất và history range.

**Lưu ý thiết kế model features:** Nếu rainfall/wind/reservoir hoặc feature khác đến từ nhiều station/source, phải xác định mô hình liên kết thời gian và nguồn trước khi cố nhét mọi feature vào cùng một hàng. Agent phải kiểm tra dữ liệu/model thực tế để chốt schema chi tiết; không tự giả định rainfall luôn thuộc trạm nước.

**Model Configuration** (được quản lý bằng file cấu hình/mapping ở MVP, có thể đưa vào DB sau): `station_id`, model path/version, scaler path/version, `sequence_length`, feature list/order, forecast horizons, preprocessing version, model readiness.

**Bảng `forecasts` — lưu dự báo từ dữ liệu quan trắc gốc:**
- `id`, `station_id`, `model_version`, `feature_schema_version`, `pipeline_version`, `input_hash`.
- `anchor_observed_at`: thời điểm cuối của chuỗi đầu vào; `generated_at`: lúc thực sự chạy mô hình. Lưu UTC và hiển thị giờ Việt Nam.
- `input_hash` nhận diện toàn bộ chuỗi đầu vào, timestamp, thứ tự feature và cấu hình horizon. Phiên bản pipeline phải nhận diện cả scaler và preprocessing; không chỉ dùng tên file.
- Unique constraint trên `(station_id, model_version, feature_schema_version, pipeline_version, input_hash)` để các request giống nhau không tạo kết quả trùng. Index theo trạm, phiên bản mô hình và anchor để đọc lịch sử.

**Bảng `forecast_points` — các horizon của một dự báo:**
- `forecast_id` liên kết `forecasts`, `horizon_minutes`, `forecast_at`, `water_level_mm`.
- Khóa duy nhất `(forecast_id, horizon_minutes)`. Lưu bản ghi dự báo và toàn bộ các điểm horizon trong cùng transaction.
- Mực nước thực tế lấy từ `observations` và tổng hợp theo cùng quy tắc của model khi so sánh; không chép vào bảng dự báo hoặc ghi đè giá trị dự báo bằng thực tế.

Các bảng trên là thiết kế bổ sung, cần triển khai bằng migration riêng. Không tạo sẵn dự báo cho toàn bộ lịch sử khi app khởi động. Dữ liệu chưa từng được yêu cầu sẽ chưa có dự báo đã lưu.

**Không có bảng `users` và không lưu `user_overrides` vĩnh viễn.**
---

## 6. API Design
### 6.1. Conventions
- REST over HTTPS (production); prefix `/api/v1`.

- JSON, timestamp ISO 8601 kèm timezone, sử dụng đơn vị rõ ràng.

- Danh sách các giá trị model-related phải lấy từ Model Configuration, không hardcode horizon hay số lượng features vào UI.

- API giữ tương thích cho việc mở rộng nhiều station.

### 6.2. Endpoint Inventory
Một số API endpoints gợi ý, nếu trong quá trình implement cần phải có thêm endpoint nào agent có thể thêm vào.

| Method | Endpoint | Mục đích |
|---|---|---|
| `GET` | `/api/v1/stations` | Danh sách trạm hỗ trợ để hiển thị Sidebar + map markers |
| `GET` | `/api/v1/stations/{station_id}` | Metadata, vị trí, thresholds, model support |
| `GET` | `/api/v1/stations/{station_id}/latest` | Dữ liệu mới nhất, timestamp, freshness |
| `GET` | `/api/v1/stations/{station_id}/observations?start=...&end=...` | Dữ liệu lịch sử theo khoảng thời gian |
| `GET` | `/api/v1/stations/{station_id}/prediction-availability?start=...&end=...` | Timestamp/range đủ N bản ghi hợp lệ cho model |
| `GET` | `/api/v1/stations/{station_id}/prediction-inputs?reference_time=...` | Dữ liệu auto-fill form ở Latest/Historical, qua Prediction Input Cache |
| `POST` | `/api/v1/predictions` | Latest hoặc Historical Replay + overrides |
| `GET` | `/api/v1/health` | Basic health/readiness status |

`reference_time` ở `prediction-inputs` là tùy chọn: bỏ trống sẽ lấy thời điểm hợp lệ mới nhất. Response phải trả `requested_reference_time`, `anchor_observed_at`, `model_version`, `feature_schema_version`, `features` (timestep cuối, đúng tên và thứ tự theo metadata), `editable_features`, `is_stale` và `input_ready`; không gửi N × features lên form nếu UI không cần. Endpoint không chạy inference. Nếu `N` dữ liệu không đủ, trả lỗi/availability rõ ràng. Dữ liệu source được cache, không bao giờ cache user overrides trong endpoint này.

`prediction-availability` ban đầu có thể trả về các khoảng có thể chọn, **nhưng logic cuối cùng tại `POST /predictions` vẫn phải kiểm tra N bản ghi thực tế**, vì dữ liệu có thể thay đổi trong thời gian chờ.

### 6.3. Station Detail — response mẫu
```json

{

  "station_id": "da-vien",

  "station_type": "automatic_water_level",

  "name": "Trạm Dã Viên (Sông Hương)",

  "river_name": "Sông Hương",

  "location": {

    "latitude": 16.458944,

    "longitude": 107.571361,

    "address": "phường Thuận Hóa, thành phố Huế"

  },

  "alarm_levels_m": { "level_1": 1.0, "level_2": 2.0, "level_3": 3.5 },

  "enabled": true,

  "prediction_enabled": true

}
```

**Các trường metadata của trạm cần Agent đối chiếu nguồn trước Production.** Không nhầm alarm thresholds với xác suất/dự báo chính thức.

### 6.4. Latest Observation — response mẫu
```json

{

  "station_id": "da-vien",

  "observed_at": "2026-10-12T09:00:00+07:00",

  "water_level_m": 0.51,

  "features": { "water_level_m": 0.51 },

  "data_status": "fresh",

  "is_stale": false,

  "missing_features": []

}
```

`0.51 m` và timestamp trong JSON là **ví dụ minh họa**. Dữ liệu thực lấy từ upstream/PostgreSQL. Nếu model còn yêu cầu rainfall hoặc feature khác, trường `features` mở rộng dựa trên model schema thực.

### 6.5. Prediction Request / Response — ví dụ
**Request:**
```json

{

  "station_id": "da-vien",

  "reference_time": "2026-10-10T09:00:00+07:00",

  "overrides": { "water_level_m": 0.60 }

}
```

- `reference_time = null`/không truyền: dùng bản ghi hợp lệ mới nhất của trạm.

- Có `reference_time`: chọn N bản ghi hợp lệ với timestamp `<= reference_time`.

- `overrides`: tùy chọn, áp dụng vào **bản ghi cuối** của chuỗi input được chọn, không sửa DB.

- Nếu thời điểm user thấy dữ liệu và thời điểm inference khác nhau, Backend trả `anchor_observed_at` thực tế để UI tránh gây hiểu nhầm.

**Response:**
```json

{

  "station_id": "da-vien",

  "requested_reference_time": "2026-10-10T09:00:00+07:00",

  "anchor_observed_at": "2026-10-10T08:00:00+07:00",

  "model_version": "lstm-v1",

  "input_observation_count": 96,

  "input_period": {

    "first_observed_at": "2026-10-07T01:00:00+07:00",

    "last_observed_at": "2026-10-10T08:00:00+07:00"

  },

  
  "predictions": [

    { "predicted_at": "2026-10-10T09:00:00+07:00", "water_level_m": 0.67 }

  ],

  "cached": false

}
```

**Response trên chỉ để mô tả schema; không phải dữ liệu đo hoặc output model thật.** Các horizon, sample timestamps và thời điểm đầu vào phải được xác định thực tế. Giờ trong `predicted_at` bắt đầu từ anchor, không từ thời gian user nhập nếu user chọn thời gian thiếu bản ghi.

### 6.6. Standard Errors
```json

{

  "error": {

    "code": "INSUFFICIENT_HISTORY",

    "message": "Not enough valid historical observations to form model input",

    "details": { "required": 96, "available": 68 }

  },

  "request_id": "..."

}
```

HTTP status: `400` yêu cầu nghiệp vụ không phù hợp, `404` station không tồn tại, `422` validation errors, `429` rate limit, `502` upstream lỗi, `503` model/data service không khả dụng, `500` lỗi nội bộ. Phân biệt rõ `INSUFFICIENT_HISTORY`, `MODEL_UNAVAILABLE`, `MISSING_REQUIRED_FEATURES`, `STALE_SOURCE_DATA`, `UPSTREAM_UNAVAILABLE`.

---

## 7. Data Flow & Historical Edge Cases
### 7.1. Scheduled Data Ingestion
1. Ingestion Job chạy theo lịch sau khi khảo sát API Postman.

2. Gọi các endpoint upstream tối thiểu để lấy station metadata, latest và historical incremental records.

3. Validate data type, timestamps, timezones, units, station mapping, feature availability.

4. Normalize và UPSERT PostgreSQL theo unique key; idempotent, xử lý records sửa đổi.

5. Cập nhật sync status, logging, invalidate/refresh các cache key latest bị ảnh hưởng.

6. Có backfill command để tải dữ liệu quá khứ ban đầu, theo khả năng nguồn API.

Không gọi upstream một cách lặp lại cho từng lượt truy cập nếu dữ liệu đã đồng bộ đủ mới. Nếu upstream hỏng, giữ dữ liệu quan trắc trong PostgreSQL và phản hồi đúng `is_stale` khi cần.

### 7.2. Latest UI Flow
```text

Select station / open page

  -> GET station detail + GET latest

  -> Backend reads Redis (optional)

      -> Hit: return cached value if fresh

      -> Miss: read latest record from PostgreSQL -> cache -> return

  -> React displays popup/detail and prefills form
```

### 7.3. Historical Replay / Last-N Selection
**Quy tắc đã thống nhất:** Lấy N bản ghi quá khứ hợp lệ gần nhất, **không bắt buộc các quan trắc phải cách nhau chính xác 1 giờ**.

**Historical Input Validation:** Hệ thống lấy `sequence_length` bản ghi hợp lệ gần nhất tính đến thời điểm dự báo, không yêu cầu các bản ghi phải cách đều nhau 1 giờ. Các bản ghi được sắp xếp theo thời gian trước khi đưa vào mô hình, nhất quán với pipeline huấn luyện. Không cần cảnh báo `IRREGULAR_OBSERVATION_INTERVALS`; chỉ từ chối dự báo khi không đủ dữ liệu hoặc thiếu các đặc trưng bắt buộc.

Giả sử `N = 96`. Thuật toán logic:

1. Lấy `reference_time` từ request (hoặc timestamp mới nhất khi chọn Latest).

2. Query observations với `station_id`, `observed_at <= reference_time`, sắp xếp `observed_at DESC`.

3. Lọc các bản ghi hợp lệ cho **các feature bắt buộc**; nếu feature đến từ nguồn khác phải căn theo quy tắc join đã được xác nhận.

4. Chọn **96 bản ghi hợp lệ gần nhất**, dù timestamps có khoảng trống. Nếu ít hơn 96: `INSUFFICIENT_HISTORY`.

5. Sắp xếp lại 96 bản ghi `ASC` trước khi đưa vào LSTM; **không thêm hàng thiếu giờ, không tự nội suy**.

6. Xác định `anchor_observed_at` từ bản ghi cuối; trả timestamp này để UI phân biệt với `reference_time`. Không tạo `IRREGULAR_OBSERVATION_INTERVALS` hoặc warning vì khoảng cách thời gian giữa các bản ghi không đều.

7. Apply overrides lên timestep cuối, theo feature đã được model cho phép.

8. Validate schema/shape, preprocess đúng training pipeline, kiểm tra Prediction Cache, inference nếu miss.

9. Map các horizon dự báo theo quy tắc model đã được kiểm chứng, inverse-transform về đơn vị gốc và trả kết quả. Không tự tạo warning vì các bản ghi cách nhau không đều.

**Ví dụ minh họa:** Trong 80 giờ gần nhất có 8 giờ không có bản ghi. Nếu tìm được 96 bản ghi hợp lệ trong khoảng lớn hơn 96 giờ, **vẫn cho dự báo**. Không từ chối chỉ vì thiếu các timestamp xen giữa.

**Lưu ý về ML:** Một LSTM được train trên các bước cách đều 1 giờ không tự hiểu thời gian bị nhảy; mô hình sẽ đọc 96 bản ghi như 96 bước liên tiếp trong chuỗi. Đây là rủi ro về chất lượng dự báo, **không phải lý do để thay đổi hành vi lấy dữ liệu mà chủ dự án đã quyết định**. Agent có thể đánh giá offline trên tập holdout có gaps (nếu dữ liệu cho phép) và đề xuất cải thiện khi có bằng chứng; không bổ sung cảnh báo gap vào response/UI.

**Nếu có đúng 96 bản ghi nhưng cách xa reference time quá lâu:** MVP vẫn dùng last-N theo chính sách được duyệt, nhưng phải báo rõ `anchor_observed_at` và độ mới dữ liệu (khác với cảnh báo về khoảng cách thời gian không đều); không ghi rằng dự báo bắt đầu chính xác ở thời gian người dùng chọn khi thực tế khác.

### 7.3.1. Việc cần làm sau: kiểm chứng ngưỡng stale bằng Train/Val

**Chưa thực hiện thống kê trong task hiện tại.** Trước khi chốt lại ngưỡng stale cho đầu vào dự báo, kiểm tra timestamp của các mẫu thực sự được dùng trong Train và Val, riêng từng tập:

- Đếm số cặp mẫu liên tiếp có khoảng cách lớn hơn 1 giờ, kèm tỷ lệ trên tổng số cặp trong tập.
- Ghi khoảng cách lớn nhất và hai timestamp tương ứng để kiểm tra lại.
- Không tính khoảng cách nối từ cuối Train sang đầu Val; dùng timestamp thực, không dùng số thứ tự dòng.
- Dùng kết quả để xem xét ngưỡng stale; không tự lấy khoảng cách lớn nhất làm ngưỡng. Khoảng cách giữa các mẫu và tuổi dữ liệu mới nhất là hai đại lượng khác nhau.

Trong thời gian chờ phân tích, giữ quy tắc hiện tại: dữ liệu trễ **hơn 2 giờ** mới được đánh dấu stale, cấu hình bằng `INPUT_STALE_AFTER_MINUTES=120`. Latest so thời điểm quan trắc với hiện tại; đầu vào dự báo lịch sử so anchor với mốc tham chiếu đã làm tròn theo giờ. Không thay đổi chính sách cho phép chuỗi có khoảng trống hoặc tự thêm cảnh báo khoảng cách không đều.

### 7.4. Historical Visualization and No Leakage
- Biểu đồ lịch sử có thể truy vấn dữ liệu trước và sau anchor để hiển thị/hậu kiểm, **nhưng Prediction Service chỉ sử dụng dữ liệu `<= anchor_observed_at`**.

- Khi có mực nước thực tế sau anchor, hiển thị thêm series `actual_after_anchor` để so sánh.

- API `GET /api/v1/stations/{station_id}/forecasts` đã phục vụ chế độ xem khoảng thời gian theo một horizon, lọc bằng `forecast_at` và phân trang. Frontend giữ cặp model/pipeline trả về khi đọc các trang tiếp theo.

- **Đã có API cho chế độ xem một mốc:** `GET /api/v1/forecasts/{forecast_id}` trả các horizon của đúng bản dự báo được chọn và series thực tế theo giờ đến horizon lớn nhất. Dã Viên hiện tại có 5 điểm dự báo và 24 mốc thực tế; mốc thiếu hoặc chưa có quan trắc giữ null. Không dùng truy vấn khoảng theo một horizon để thay cho nhóm điểm của một anchor. Endpoint chỉ đọc, không chạy inference hoặc tải model/scaler.

- Phân biệt thời điểm phát dự báo (`anchor_observed_at`) với thời điểm được dự báo (`forecast_at`). Khi vẽ một đường theo horizon, ví dụ 60 phút, căn dự báo và thực tế theo `forecast_at`.

- Dự báo chạy lại hôm nay cho một mốc quá khứ là Historical Replay, không phải bằng chứng hệ thống đã phát dự báo vào lúc đó. Hiển thị `generated_at` khi cần phân biệt; không dùng dữ liệu thực tế sau anchor để tạo đầu vào.

- Nếu timestamp nằm trong tập training, không đánh giá đây là bằng chứng generalization.

- Không lưu user overrides vào PostgreSQL.

---

## 8. Redis Caching Strategy
**Pattern:** Cache-Aside; PostgreSQL là nguồn dữ liệu quan trắc nội bộ, Redis là dữ liệu thứ cấp dễ làm mới.

### 8.1. Hydrology Cache
- Latest cache key: `hydrology:v1:latest:{station_id}`.

- Optional window cache (khi đo thấy cần): `hydrology:v1:window:{station_id}:{query_hash}`.

- Value: dữ liệu đã chuẩn hóa + timestamps + freshness metadata.

- **TTL ban đầu: 600 giây** (theo chỉnh sửa trong bản review), phải điều chỉnh sau khi Agent kiểm tra chu kỳ API.

- Invalidate khi ingestion UPSERT bản ghi mới hoặc cập nhật bản ghi đã thay đổi, đặc biệt với latest.

- `TTL=600` không chứng minh upstream cập nhật 10 phút/lần. Độ mới nguồn vẫn phải tính theo timestamp thật.

### 8.2. Prediction Input Cache

- Pattern: Cache-Aside; cache dữ liệu gốc đã chuẩn bị cho biểu mẫu, **không phải** toàn bộ kết quả dự báo hay giá trị user overrides.
- Key gợi ý: `prediction-inputs:v1:{station_id}:{reference_time_or_latest}:{model_version}:{feature_schema_version}:{data_revision}`. Với Latest cần phân biệt thời điểm/cửa sổ dữ liệu hiện hành để tránh dùng nhầm cache cũ. Nếu chưa có `data_revision`, dùng TTL kiểm soát và xóa các key bị ảnh hưởng khi ingestion cập nhật bản ghi.
- Value: 29 giá trị đầu vào (hoặc `feature_count` thực tế) của timestep cuối, `anchor_observed_at`, `requested_reference_time`, danh sách `editable_features`, `input_ready` và metadata liên quan.
- TTL: dữ liệu Latest bắt đầu thử nghiệm 600 giây hoặc theo lịch cập nhật upstream; dữ liệu Historical có thể dài hơn, **ví dụ 24 giờ**, nếu có cơ chế invalidation phù hợp. Chốt sau Discovery.
- Invalidation: khi ingestion làm thay đổi bất cứ bản ghi nào thuộc cửa sổ Last-N hoặc khi đổi model, mapping feature hay preprocessing. Nếu không thể xác định chính xác key bị ảnh hưởng, dùng revision/generation theo station để vô hiệu hóa một cách an toàn.
- Dữ liệu người dùng đang sửa nằm ở Frontend State và POST Prediction; tuyệt đối không ghi vào Input Cache.
- Cần đo hit ratio và độ trễ chuẩn bị input, tránh tạo thêm cache phức tạp nếu lợi ích không đáng kể.

### 8.3. Prediction Result Cache
**Luồng dùng lại và lưu kết quả:**
1. Chuẩn bị đầu vào, kiểm tra quyền dự báo và tạo dấu nhận diện chuỗi hiện tại. Latest/Historical đều áp dụng cùng quy tắc.
2. Tìm kết quả trong Redis. Nếu có kết quả khớp dấu nhận diện, trả lại mà không chạy mô hình.
3. Nếu Redis chưa có và request không có overrides, tìm dự báo tương ứng trong PostgreSQL. Nếu tìm thấy, đưa lại vào Redis và trả kết quả.
4. Nếu chưa có kết quả, chạy mô hình. Với request không có overrides, lưu `forecasts` và `forecast_points` trong một transaction rồi cache Redis. Với request có overrides, chỉ cache Redis.
5. Nếu nhiều request cùng tạo một dự báo, unique constraint bảo đảm chỉ lưu một bản; request gặp xung đột đọc lại bản đã lưu. Chưa cần thêm distributed locking.

- Kết quả lâu dài chỉ chứa giá trị dự báo và thông tin nguồn gốc. `requested_reference_time`, tuổi dữ liệu và `is_stale` phải được tính theo request hiện tại, không lấy nguyên các trạng thái động từ lần lưu/cache trước.
- Khi dữ liệu nguồn được sửa hoặc model/scaler/pipeline đổi, dấu nhận diện phải đổi. Giữ kết quả cũ để tham khảo nhưng không dùng thay cho dự báo từ cấu hình và đầu vào hiện tại.
- Redis lỗi thì tiếp tục đọc PostgreSQL hoặc chạy mô hình theo điều kiện an toàn. Nếu lưu PostgreSQL thất bại, trả lỗi dịch vụ rõ ràng và không báo rằng dự báo đã được lưu thành công.

- Key: `prediction:v1:{station_id}:{model_version}:{pipeline_version}:{input_hash}`.

- `input_hash` phải gồm representation ổn định của **toàn bộ N feature vectors sau chọn dữ liệu và apply overrides**, các timestamp cần thiết, model/horizon config. Không chỉ dùng station ID hoặc reference time.

- Value: forecast response dùng để trả cho UI; trường `cached` có thể thiết lập khi đọc thay vì lưu cứng.

- **TTL ban đầu: 3600 giây** (giá trị đề xuất thử nghiệm).

- Thay model, scaler, preprocessing, feature mapping hoặc dữ liệu đầu vào -> đổi key/hash và không tái sử dụng kết quả cũ.

- Cache không gắn user identity; không lưu lịch sử riêng của user.

### 8.3.1. Quyết định tạm thời: device trong danh tính pipeline

- **Tạm chấp nhận** việc `pipeline_version` phân biệt device chạy inference. Cùng model và đầu vào nhưng chạy CPU/GPU có thể thuộc các pipeline khác nhau, nên không tự dùng chung bản lưu/cache.
- Device là cấu hình nội bộ của backend qua `PREDICTION_DEVICE`, không phải input của người dùng. Frontend không hiển thị ô chọn CPU/GPU và không gửi device trong request dự báo.
- Hướng triển khai đã thống nhất là inference bằng GPU, kể cả khi public app; cấu hình backend dùng `PREDICTION_DEVICE=cuda`. Không tự chuyển sang CPU khi GPU không khả dụng; trả lỗi dịch vụ rõ ràng.
- API đọc dự báo và frontend phải chọn nhất quán pipeline phù hợp cho đường biểu đồ, không trộn các bản CPU/GPU thành một series. Người đọc biểu đồ không cần thao tác chọn device.

**Cần xử lý sau, chưa thay đổi trong task hiện tại:**
- Đánh giá lại việc đưa device vào danh tính lưu/cache; xác định có cần giữ phân biệt này hay cho phép dùng chung kết quả sau khi kiểm chứng sai khác số học.
- Nếu đổi quy tắc hash, cần kế hoạch tương thích/migration cho dự báo đã lưu, đặc biệt lượt backfill GPU 7.571 dự báo. Không âm thầm làm mất khả năng dùng lại hoặc yêu cầu chạy lại toàn bộ lịch sử.
- Giữ khả năng truy vết model/scaler/preprocessing và đầu vào dù cách xử lý device thay đổi. Không xóa bản cũ chỉ để đồng nhất device.

### 8.3.2. Việc xem xét sau: cache Visualize tại một mốc

- Người dùng có thể xem lặp lại cùng một mốc nhiều lần qua `GET /api/v1/forecasts/{forecast_id}`. Đây là vị trí tiềm năng để cache Redis nhằm giảm truy vấn các horizon và tính lại trung bình mực nước thực tế theo giờ.
- **Chưa cần triển khai hiện tại.** Đánh giá tần suất request, độ trễ và chi phí truy vấn trước khi quyết định bổ sung cache cho endpoint này.
- Nếu triển khai, key cần nhận diện đúng `forecast_id` và phiên bản schema response. Không dùng riêng station/time để thay cho một bản dự báo cụ thể.
- Dự báo đã lưu có thể dùng lại, nhưng series thực tế trong response sẽ thay đổi khi có quan trắc mới hoặc nguồn sửa dữ liệu. Cache toàn bộ response phải có TTL/invalidation theo các quan trắc của trạm trong cửa sổ hiển thị; không giữ mãi các giá trị null của những giờ trước đó chưa có dữ liệu.
- Không coi mọi dữ liệu quá khứ là bất biến. Thay đổi bản lưu hoặc quy tắc tổng hợp/response cũng phải làm mới cache tương ứng.
- Redis không khả dụng thì đọc PostgreSQL theo luồng hiện tại. Frontend không cần biết request có cache hay không để chọn mốc và vẽ biểu đồ.

### 8.4. Failure & Efficiency
- Redis timeout/unavailable: log, bypass cache và xử lý từ PostgreSQL/model khi an toàn.

- Cache Hit Ratio của cả ba loại cache, invalidations, memory usage, cache miss spikes là các metrics cần quan sát.

- Chưa cần Redis Cluster, distributed locking; chỉ nghiên cứu stampede/single-flight khi có bằng chứng request trùng gây tải.

---

## 9. Deployment
### 9.1. Local Development
- Vite React dev server.

- FastAPI chạy local/venv hoặc Docker, Redis + PostgreSQL trong Docker Compose.

- Model chạy trên CPU/GPU tùy máy; route API dùng Vite proxy để tránh lệ thuộc CORS dev không cần thiết.

### 9.2. Single-Server Production (MVP)
```text

One VPS / Physical Server

├─ nginx          -> React static build + /api reverse proxy

├─ api            -> FastAPI, modular monolith + PyTorch model

├─ ingestion      -> scheduled sync job (same codebase / shared module)

├─ postgres       -> durable observations + metadata (volume + backups)

└─ redis          -> ephemeral caches (private container network)
```

- Không public Postgres/Redis ports; chỉ API/Web Server mở những cổng cần thiết.

- Model load một lần khi mỗi API worker start. Bắt đầu 1 worker nếu model lớn/GPU memory giới hạn; benchmark trước khi thêm worker/replica.

- PostgreSQL dùng persistent Docker volume, có backup và migration (ví dụ Alembic). Redis được xem là cache có thể rebuild.

- Ingestion job tránh chạy trùng khi có nhiều API instances; thiết lập riêng process/cron với cơ chế chạy một job hoặc idempotent upsert.

- Nginx TLS và static file hosting trong production; không cần Load Balancer ở MVP.

- Log có request ID, không hiển thị internal stack trace ra frontend.

### 9.3. Environment & Secrets
Biến môi trường gợi ý: `DATABASE_URL`, `REDIS_URL`, `HYDROLOGY_API_BASE_URL`, `HYDROLOGY_API_KEY` (nếu upstream cần), `MODEL_CONFIG_PATH`, `MODEL_VERSION`, `CACHE_TTL_HYDROLOGY_SECONDS`, `CACHE_TTL_PREDICTION_SECONDS`, `CACHE_TTL_PREDICTION_INPUT_SECONDS`, `CORS_ORIGINS`, `MAP_TILE_URL` (frontend public config), `LOG_LEVEL`.

Không commit credentials. API keys đưa vào biến build client (`VITE_*`) **không phải secrets**, vì mã frontend có thể xem được.

---

## 10. Error Handling, Testing & Acceptance Criteria
### 10.1. Error Handling
| Tình huống | Hành vi mong muốn |
|---|---|
| Upstream hydrology API lỗi | Retry có giới hạn + timeout, giữ bản ghi DB cũ với nhãn stale, không bịa dữ liệu mới |
| Redis down | Bypass cache sang Postgres/model, log warning |
| PostgreSQL down | Trả 503 cho chức năng phụ thuộc DB, không giả sử Redis là nguồn truth |
| Station không tồn tại | 404 rõ ràng |
| Trạm chưa sẵn sàng dự báo | 503/MODEL_UNAVAILABLE hoặc response trạng thái phù hợp |
| Chỉ có < N bản ghi hợp lệ trước T | `INSUFFICIENT_HISTORY`; không tự điền giờ thiếu |
| Có N bản ghi nhưng timestamps gián đoạn | **Vẫn chạy inference**, không phát gap warning; trả anchor thực tế |
| Thiếu feature quan trọng | Bỏ qua bản ghi thiếu khi chọn last-N nếu hợp lệ theo policy; nếu không đủ N thì lỗi rõ |
| Client thay đổi input ngoài phạm vi | 422/INVALID_INPUT, không ghi DB |
| Model lỗi | Log request ID, trả lỗi có kiểm soát |
| Request cũ đến sau khi đổi trạm | Frontend bỏ qua response stale, không ghi đè trạm đang chọn |
| Historical look-ahead | Test ngăn dùng timestamp sau anchor làm model input |

### 10.2. Unit Tests
- Pydantic schemas, unit/timezone validation.

- Station alarm level display logic và trạng thái dữ liệu.

- Chuẩn hóa payload từ API, mapping source IDs, duplicate / revised records.

- Last-N selector: chọn đúng số lượng, thứ tự thời gian, không leakage.

- Overrides chỉ sửa timestep cuối.

- Cache key deterministic, đổi input/model/timestamps -> đổi hash; khi dữ liệu quan trắc thay đổi, Prediction Input Cache không phục vụ dữ liệu cũ không hợp lệ.

- Scaler/inverse transform giống pipeline offline; shape/feature order.

### 10.3. Integration Tests
- PostgreSQL migrations, indexed time-range queries, UPSERT idempotent.

- Lưu dự báo gốc và các horizon nguyên tử; request lặp không tạo bản trùng. Redis miss phải đọc được PostgreSQL mà không chạy lại mô hình. Overrides không tạo bản ghi dự báo lâu dài. Đổi model/scaler/pipeline hoặc sửa dữ liệu đầu vào không trả nhầm kết quả cũ.

- Redis integration: hit/miss/TTL/invalidation/fallback cho đủ ba cache; sửa giá trị user không làm thay đổi Input Cache gốc.

- Fake Hydrology Provider; test ingestion partial failure và lịch cập nhật.

- FastAPI -> Observation Repository -> Prediction Service -> Model Runtime.

### 10.4. End-to-End Tests
1. Mở website, map focus Huế, marker Dã Viên hiển thị.

2. Hover/click marker hiển thị metadata, ngưỡng 1/2/3 và mực nước từ latest record, không hardcode 0.51 m.

3. Sidebar chọn trạm -> map fly-to -> details cập nhật đồng bộ.

4. Latest hoặc Historical chọn T -> `prediction-inputs` auto-fill đủ 29 features (hoặc số thực của model), chỉnh sửa không bị refresh ghi đè; Reset khôi phục dữ liệu gốc.

5. Click Predict -> loading -> kết quả + chart, thời gian dự báo đúng horizon.

6. Dùng lại cùng input/version -> Cache Hit và không chạy inference thêm.

7. Historical mode chọn T -> lấy N bản ghi hợp lệ trước T, không cần hàng giờ liên tục.

8. Với mô hình cấu hình N=96, chuỗi có 96 bản ghi hợp lệ nhưng gián đoạn vẫn được chấp nhận **không có gap warning**; chỉ có 71 bản ghi thì bị từ chối.

9. Có quan trắc thật sau anchor -> overlay trên chart nhưng không tham gia input.

10. Nguồn upstream, Redis hoặc Model gặp lỗi -> UI hiển thị trạng thái phù hợp.

11. Trên thiết bị mobile, marker và Station Detail có thể truy cập bằng click/tap.

### 10.5. Performance & ML Validation
- Đo p50/p95 response time ở cache hit/miss, model inference latency, CPU/RAM/GPU memory.

- Đo hit ratio và giảm số lần gọi upstream.

- Kiểm tra kết quả API dự báo khớp offline inference trong sai số cho phép.

- So sánh hiệu năng mô hình trên input liên tục và input có gaps; thống kê độ dài gaps, mức độ trượt phân phối và giới hạn đáng tin cậy trước khi dùng cho quyết định rủi ro.

---

## 11. Open Items / Agent Investigation Checklist
**Ưu tiên P0 — phải xong trước khi xây Data Pipeline và chốt API payload:**
1. **Hydrology Postman collection:** Liệt kê endpoints và field cần; kiểm tra update frequency, historical lookback, pagination, rate limits, timestamps/timezone, units, auth, station IDs. Tài liệu kiến trúc **không** khẳng định lịch cập nhật khi chưa đọc API thực tế.

2. **Model artifacts:** Xác nhận checkpoint, framework, scaler, feature list/order, actual sequence length N, actual output horizons, preprocessing, missing-value policy, có hoặc không thể dùng overrides ở timestep cuối.

3. **Metadata:** Xác minh tên trạm, địa chỉ, tọa độ, ngưỡng báo động từ nguồn; số `0.51 m` trong yêu cầu chỉ để minh họa giao diện.

4. **Feature sources:** Một feature có đến từ trạm khác hay không; cách join theo thời gian và điều kiện valid row.

**Ưu tiên P1 — trước production:**
5. Quy tắc `data_status = fresh/stale/unavailable`, nguồn giờ cập nhật và ngưỡng freshness.

6. Ngưỡng rate limiting và tài nguyên inference; benchmark trên hạ tầng thực.

7. Thống nhất frontend UX khi `reference_time` khác `anchor_observed_at`.

8. Chọn map tile provider và kiểm tra attribution/usage policy.

9. Chốt API schemas và migration sau khi P0 được xác minh.

---

## 12. Implementation Plan (mỗi phase cần tests và kết quả review)
| Phase | Công việc | Deliverable / Exit Criteria |
|---|---|---|
| **1. Discovery** | Inspect repo, Postman collection, model, datasets, historical gaps, station metadata | Báo cáo API/Model Contracts, thông tin còn thiếu, quyết định được duyệt |
| **2. Persistence & Ingestion** | Schema PostgreSQL, migration, seed metadata, backfill/import, incremental sync | Idempotent ingestion; query latest/history/last-N hoạt động; log sync |
| **3. Backend API** | FastAPI routes, validation, Station Registry, error format | OpenAPI schemas, unit/integration tests và response có timestamps đúng |
| **4. Prediction Pipeline** | Load artifacts, last-N selector, overrides, preprocess, inference, inverse transform | Test 96 có gaps được phép; 71 bị từ chối; match offline inference |
| **5. Redis** | Hydrology + Prediction Input + Prediction Result cache, TTL, invalidation, fallback | Hit/Miss/TTL tests; khi Redis down hệ thống vẫn phục vụ khi an toàn |
| **6. Frontend** | Map + Sidebar + Details, forms, historical modes, Recharts | Chọn trạm đồng bộ map; click predict; show charts/data-status/mobile |
| **7. Deploy & QA** | Docker Compose, Nginx, observability, E2E, benchmark, security checks | Repeatable deployment; có metrics; đáp ứng acceptance criteria |

### Nguyên tắc cho AI Coding Agent
- Đọc `ARCHITECTURE.md` và Model/API investigation report trước khi triển khai.

- Không tự thay thế yêu cầu **last-N valid observations (allow gaps)** thành **N hourly continuous observations**.

- Không tự suy diễn data cadence, model features, output horizons hoặc dữ liệu trạm.

- Không thêm authentication, message broker, microservices hoặc thư viện dư thừa nếu chưa có approval.

- Tách từng Phase thành tasks nhỏ có acceptance tests; báo cáo trade-offs và blocked assumptions trước khi viết code cho mục chưa xác nhận.

- Review/QA sau mỗi phase. Mọi thay đổi kiến trúc quan trọng cần ghi quyết định và lý do.

---

## Appendix — References (implementation guidance)
- React Leaflet Map API: \<https\://react-leaflet.js.org/docs/api-map/>

- OpenStreetMap Tile Usage Policy: \<https\://operations.osmfoundation.org/policies/tiles/>

- FastAPI: \<https\://fastapi.tiangolo.com/>

- Redis: \<https\://redis.io/docs/latest/>

- PostgreSQL: \<https\://www.postgresql.org/docs/>
