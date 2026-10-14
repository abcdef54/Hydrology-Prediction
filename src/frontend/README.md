# Frontend thủy văn

React + Vite + Tailwind CSS. Bản đồ dùng MapLibre GL JS và vector tiles theo schema
OpenMapTiles, dữ liệu OpenStreetMap do OpenFreeMap cung cấp. Không cần API key.
Trang có bản đồ toàn màn hình, mở sẵn tại Huế, zoom 12; có thể kéo và zoom.
Marker trạm được lấy động từ backend, hiện gồm Dã Viên và 4 trạm mưa A Lưới.
Sidebar nổi bên trái hiển thị metadata và số đo mới nhất; chọn item để focus bản đồ.

## Chạy local

Cần Node.js 22.12 trở lên (đã kiểm tra với Node.js 24).

```bash
cd /home/my_ubuntu/projects/hydrology/src/frontend
npm ci
npm run dev
```

Mở <http://localhost:5173>. Bản đồ cần Internet và trình duyệt hỗ trợ WebGL.
Backend cần chạy tại cổng 8000, kết nối PostgreSQL đã có dữ liệu trạm.
Có trạng thái tải và thông báo khi không truy cập được bản đồ hoặc danh sách trạm;
lỗi API trạm không làm mất bản đồ nền.

## Marker trạm

Trang gọi `GET /api/v1/stations` qua proxy Vite, sử dụng `station_type` và
`location.longitude/latitude` từ API; không hardcode tọa độ hoặc danh sách trạm.

- Trạm nước: marker 40 px, xanh đậm, icon sóng nước; nổi bật và nằm trên marker phụ.
- Trạm mưa: marker 28 px, màu nhẹ hơn, icon mây/mưa.
- Loại khác: marker 24 px, màu trung tính; trạm gió dùng icon gió nếu API hỗ trợ.
- API hiện chỉ hỗ trợ `water_level` và `rainfall`; không tự tạo trạm gió.
- Thiếu/sai tọa độ: bỏ qua trạm đó và hiển thị thông báo, không dùng vị trí giả.
- Click marker hoặc nhấn Enter/Space khi marker được focus để chọn trạm, focus
  bản đồ và mở box thông tin. Không kéo marker; kéo bản đồ không coi là click marker.
- Trạm mưa A Lưới nằm về phía tây Huế; có thể cần kéo/thu nhỏ để xem tùy màn hình.

Request trạm được hủy và marker được gỡ khi component unmount, tránh marker lặp.

## Sidebar trạm

- Có khoảng trống với mép màn hình; phần danh sách cuộn độc lập bằng chuột/bàn phím.
- Mỗi item có icon cùng loại với marker, tên, loại, địa chỉ, số đo và giờ quan trắc.
  Trạm nước đứng đầu danh sách; số đo đọc từ `/api/v1/stations/{id}/latest`.
- Bộ lọc loại trạm áp dụng cho cả sidebar và marker. Loại được tạo từ danh sách
  API, có icon/nhãn mặc định cho loại mới; backend hiện vẫn chỉ hỗ trợ nước/mưa.
- Dòng “Dữ liệu mới nhất” ưu tiên `observed_at` mới nhất của nhóm trạm nước;
  nếu không có thì lấy mới nhất của loại khác. Hiển thị giờ Việt Nam, không đổi
  theo filter và không dùng `received_at` làm thời điểm quan trắc.
- Đơn vị đọc từ API: Dã Viên và bốn trạm mưa dùng mm sau migration `0004`, theo
  quyết định chủ dự án. Chưa xác định khoảng tích lũy bản ghi mưa; không ghi mm/h.
  Trạm chưa biết đơn vị hiển thị rõ, không tự gán mm. Số 0 và mực nước âm giữ nguyên.
- Trạng thái đang tải, chưa có số đo, lỗi từng trạm và dữ liệu cũ được phân biệt.
  Metadata vẫn hiển thị nếu request số đo thất bại; nút “Thử lại” tải lại dữ liệu.
- Tối đa 4 request số đo đồng thời, làm mới mỗi 10 phút khi tab đang hiển thị;
  không request lại chỉ vì đổi filter/chọn trạm. Không retry tự động ngay khi lỗi.
- Click item di chuyển bản đồ tới trạm trong khoảng 1,4 giây và mở box thông tin,
  chừa vùng cho các panel. Tôn trọng reduced motion; thiếu/sai tọa độ không cho focus.
- Trên màn hình hẹp, sidebar giới hạn chiều cao để giữ phần bản đồ phía dưới.

## Box thông tin trạm

- Box cố định trên màn hình: loại, tên, địa chỉ, tọa độ (vĩ độ, kinh độ), số đo và
  giờ quan trắc mới nhất. Dùng địa chỉ backend hiện có, không tự thêm phường/xã.
- Trạm nước có ba mức báo động bằng mm, chuyển từ `alarm_levels_m` bằng × 1.000.
  `null` hiện “Chưa cấu hình”; hiện Dã Viên chưa có ngưỡng. Trạm mưa không có mục này.
- Gọi detail và latest song song mỗi lần mở/chọn lại trạm, hoặc nhấn thử lại.
  Database bổ sung ngưỡng sẽ được đọc ở lần tải tiếp theo; box không polling riêng.
- Box giữ nguyên khi pan/wheel/zoom, kể cả double-click zoom. Click đơn nền map
  hoặc × đóng box; chọn trạm khác đổi nội dung. Click đóng được đợi 300 ms để phân
  biệt với double-click. Không coi các nút zoom là click nền bản đồ.
- Detail/latest có trạng thái lỗi riêng; giữ thông tin trạm cơ bản khi lỗi detail,
  không thay ngưỡng lỗi tải thành “Chưa cấu hình”. Đóng/chuyển trạm hủy request cũ.
- Box cuộn độc lập khi thiếu chỗ. Trên màn hình hẹp, thu chiều cao cả sidebar và
  box để giữ một khoảng bản đồ giữa hai panel; nút × luôn nằm trong header cố định.

## Bảng lịch sử quan trắc

- Nút “Lịch sử” trên header sidebar mở bảng tổng; “Xem lịch sử” của mỗi trạm mở
  bảng lọc sẵn loại/mã trạm đó. Mỗi lần mở mặc định hôm nay theo giờ Việt Nam.
- Lọc theo loại, tên trạm và một ngày hoặc khoảng ngày tính cả hai đầu. Chọn tên
  gửi mã trạm chính xác, giữ số 0 đầu. Nhấn “Áp dụng” để tải; bộ lọc này độc lập
  với bộ lọc bản đồ. Không tự chuyển về ngày cuối có dữ liệu.
- Đọc `GET /api/v1/observations/hourly`, mỗi dòng là một trạm/ngày, số đo đúng
  00:00, 01:00… Thiếu hiện “—”; giữ số 0/âm và đơn vị metadata (hiện tại mm).
  Ngày hôm nay đơn lẻ chỉ hiện đến giờ tại `as_of`; ngày cũ/khoảng ngày có 24 cột,
  giờ chưa tới có nền nhạt. Số đo mưa không được coi là tổng mưa trong giờ.
- Mỗi trang tối đa 50 dòng, nút trước/sau dùng cursor; áp dụng lọc trở về trang 1.
  Bảng cuộn ngang/dọc, ghim header và cột nhận diện trạm; hỗ trợ màn hình hẹp.
- Đóng bằng × hoặc Escape, trả focus về nút mở và giữ trạng thái bản đồ/box trạm.
  Có trạng thái tải, rỗng, lỗi và thử lại; hủy request khi đóng/đổi lọc/trang,
  không cho response cũ thay thế kết quả mới. Không tự tải mọi trang hoặc polling.

## Tự điền đầu vào dự báo

- Trong bảng lịch sử, ô giờ có thể chọn của trạm nước được đánh dấu bằng chữ xanh,
  gạch dưới nét chấm. Chỉ áp dụng khi metadata có `prediction_enabled` và
  `model_configured`; trạm mưa và các trạm chưa có model không có hành vi này.
- Click ô gọi `/stations/{id}/prediction-inputs?reference_time=...` với timestamp
  chính xác từ API bảng. Ô thiếu vẫn cho backend kiểm tra; ô tương lai không chọn.
  Nút “Đầu vào mới nhất” bỏ `reference_time`, có bộ chọn trạm nước riêng.
- Vùng đầu vào phía trên bảng hiển thị trạm, mốc yêu cầu/anchor thực tế, trạng thái
  đủ dữ liệu, dữ liệu cũ và lỗi/thử lại. Các giá trị tại bước cuối được tự điền vào
  ô có thể sửa nếu API cho phép, thu gọn/mở rộng được; số lượng/thứ tự/đơn vị đọc từ metadata API.
  Giữ số 0/âm, null hiện “Thiếu”; không tự gán đơn vị hoặc thay thiếu bằng 0.
- Nhãn dùng tiếng Việt dễ hiểu, ví dụ “Mực nước 1 giờ trước”, “Mưa đầu vào”; tên trạm
  nằm ở chú thích dưới ô, tra metadata và không hiển thị mã thay tên. Vùng nhãn
  dành chỗ cho hai dòng để các ô input thẳng hàng.
  Cờ có/thiếu dữ liệu và mùa lũ hiển thị bằng chữ. Khóa feature gốc giữ trong dữ
  liệu API; feature mới chưa có mô tả dùng nhãn “Thông tin bổ sung”.
- Chuỗi đầy đủ được backend chuẩn bị, không ghép từ các ô bảng trên frontend.
  Đổi mốc, bỏ chọn hoặc đóng bảng hủy request; áp dụng bộ lọc xóa lựa chọn,
  đổi trang giữ lựa chọn với trạm/mốc hiển thị rõ. Không tự polling hoặc inference.
- Chỉ sửa feature có `editable=true` và nằm trong `editable_features`. Cờ dùng
  dropdown có nhãn; số chấp nhận dấu chấm/phẩy thập phân, giữ 0/âm. Để trống
  mưa/lag gửi null; mực nước hiện tại và cờ không được thiếu. Kiểm tra số hữu hạn/
  phạm vi float32, khóa Dự báo nếu lỗi; backend vẫn kiểm tra lại.
- “Khôi phục giá trị gốc” bỏ chỉnh sửa. Đổi mốc/tải input mới bỏ chỉnh sửa;
  quay lại từ kết quả giữ form đang sửa. Không tự cập nhật feature liên quan.
  Chỉnh sửa chỉ ở React state/request, không dùng localStorage hoặc ghi database.
  Không chọn cả khoảng thời gian bằng click dòng.

## Dự báo và biểu đồ kết quả

- Nút “Dự báo” bật khi auto-fill thành công và `input_ready=true`. Gửi một POST
  `/api/v1/predictions`; chỉ gửi các giá trị khác gốc qua `overrides` nếu sửa.
  Với “Mới nhất”, cố định
  `reference_time` bằng `hourly_reference_time` của input đang xem để tránh tự
  chuyển sang mốc mới trong lúc chờ. Backend vẫn kiểm tra lại dữ liệu khi dự báo.
- Kết quả thay vùng lịch sử trong cùng modal; “Quay lại lịch sử” giữ bộ lọc,
  trang, ô giờ và input, trả focus về nút Dự báo. ×/Escape đóng toàn bộ modal.
- Đường dự báo lấy horizon từ API (Dã Viên hiện 5 điểm +1/+3/+6/+12/+24 giờ),
  tính từ `anchor_observed_at` thực tế. Không tạo điểm dự báo ở các giờ còn lại.
  Biểu đồ SVG có điểm/tooltip, hỗ trợ focus và cuộn ngang trên màn hình nhỏ.
- Với input gốc, đọc `/api/v1/forecasts/{forecast_id}` để lấy thực tế của đúng bản dự báo,
  xác thực trạm/model/pipeline/anchor/horizon. Thực tế là trung bình từng giờ
  sau anchor, tối đa 24 mốc với model hiện tại, khác số đo đúng giờ của bảng lịch sử.
  Null không nối đường hoặc thay bằng 0; chưa có thực tế chỉ vẽ dự báo.
- Bảng so sánh ghép theo timestamp chính xác tại các horizon, hiển thị dự báo,
  thực tế, sai lệch (dự báo − thực tế), sai số tuyệt đối và trung bình sai số tuyệt
  đối trên các mốc có thực tế. Tất cả bằng mm; hiển thị tối đa 2 chữ số thập phân.
- Khóa click liên tiếp, không tự retry POST. Timeout dự báo 60 giây, API đọc vẫn
  10 giây. Có trạng thái tải/lỗi/thử lại; lỗi tải thực tế vẫn giữ dự báo và chỉ
  retry GET thực tế. Quay lại/đóng hủy chờ, ngăn kết quả cũ thay thế lượt mới;
  không đảm bảo dừng inference hoặc lưu kết quả đã bắt đầu trên server.
- Đã kiểm chứng luồng quá khứ/hôm nay/mới nhất trên backend thật, desktop/mobile;
  các trường hợp thiếu/gãy dữ liệu, quota, sai bản dự báo và hủy bằng response giả.
  Chưa có dự báo theo khoảng ngày.
- Dự báo có chỉnh sửa ghi rõ “Dự báo thử”, không lưu input/kết quả vào database.
  Metadata lưu trữ (`forecast_id`, `generated_at`, `pipeline_version`) phải null;
  biểu đồ dùng response POST, không gọi API đọc bản lưu. Hiện chỉ vẽ đường dự báo
  thử và đọc `/api/v1/stations/{id}/forecast-actuals` để so sánh thực tế theo anchor/
  horizon POST trả về, không dùng mốc yêu cầu thay anchor. API chỉ nhận trạm/mốc/
  horizon, không nhận input tự nhập. Cùng quy tắc trung bình giờ, ngắt null và sai số
  như dự báo gốc. Retry thực tế chỉ GET; retry POST giữ nguyên overrides của lượt lỗi.

## Nguồn bản đồ và giới hạn hiển thị

Cấu hình nằm trong `src/mapConfig.js`. MapLibre dùng tọa độ `[kinh độ, vĩ độ]`.
Vùng điều hướng là khung `[101.5, 7.5]` đến `[118, 24]`, bao quanh Việt Nam và
vùng biển lân cận. Đây là khung UI, không phải đường biên hành chính.
Zoom cấu hình từ 6 đến 19; MapLibre tự ràng buộc khung nhìn theo `maxBounds`
và kích thước màn hình. Không lặp bản đồ thế giới.

Style Liberty được tải từ `https://tiles.openfreemap.org/styles/liberty` và
điều chỉnh **trước khi tạo map**, nên không hiển thị style gốc trong lúc chờ lọc.
`src/maps/mapStyle.js` áp các quy tắc:

- Mọi lớp `boundary` giữ bộ lọc gốc và chỉ vẽ feature có `maritime = 0`.
  Biên giới biển (`1`) và feature thiếu thuộc tính này không được vẽ.
- Trong `water_name`, nhãn `South China Sea` được hiển thị duy nhất là
  **Biển Đông**, `Gulf of Tonkin` là **Vịnh Bắc Bộ**. Không ghép thêm tên gốc
  hoặc ký tự Trung Quốc cho hai địa danh này.
- Trong `water_name`, ẩn thêm nhãn núi ngầm, sống núi và rãnh biển được nhận diện
  bằng thuật ngữ trong các trường tên gốc/Anh/Latin. Áp cho loại biển
  ocean/sea/bay/strait, kể cả ngoài các vùng đảo, vì nguồn có thể gắn địa hình
  dưới biển thành strait. Không dùng quy tắc này cho hồ, sông hoặc lớp nhãn đất liền.
- Ẩn toàn bộ text nhãn bên trong các vùng `hiddenOffshoreLabelAreas`, bao các
  cụm đảo nhỏ ngoài khơi Biển Đông. Áp theo vị trí vì nguồn có thể phân loại
  địa danh trên đảo thành city/town/village thay vì island. Biển Đông/Vịnh Bắc Bộ
  được ưu tiên giữ nhãn Việt. Không xóa hình dạng đảo, bờ biển hoặc dữ liệu nguồn.
- Vùng lọc chỉ phục vụ hiển thị, không phải dữ liệu chủ quyền/đường biên chính thức.
  Không phủ lên đất liền Trung Quốc, Hải Nam hoặc đảo ven bờ Việt Nam; nhãn ngoài
  các vùng này dùng biểu thức gốc, không loại chữ Trung Quốc trên toàn bản đồ.
- Nguồn vector chỉ yêu cầu tile giao với khung Việt Nam. Cache theo HTTP headers
  của nguồn; không tải trước toàn vùng hoặc thêm cache offline.

Attribution OpenMapTiles/OpenStreetMap lấy từ metadata nguồn và luôn hiển thị.
Tốc độ còn phụ thuộc mạng và nguồn tiles; chưa có benchmark để khẳng định vector
nhanh hơn nguồn raster trước đây. MapLibre và worker WebGL có bundle lớn hơn Leaflet.

Kiểm chứng tập trung: style hợp lệ theo validator MapLibre; bộ lọc trên một tile
thật giữ 9 đoạn biên giới đất liền, loại 5 đoạn biên giới biển. Dữ liệu nhãn thật
được kiểm tra cho hai tên biển, 16 nhãn ngoài khơi bị ẩn và 195 nhãn ngoài vùng
lọc giữ nguyên. Bổ sung kiểm tra các nhãn địa hình dưới biển trong ảnh phản hồi,
cả khi nằm ngoài vùng đảo; nhãn biển Việt, hồ/sông và địa danh đất liền Trung Quốc
được giữ đúng phạm vi. Các mẫu kiểm tra nằm trong `/tmp`, không đưa dữ liệu bản đồ vào Git.

## Kết nối backend

Vite giữ cố định cổng 5173, báo lỗi nếu cổng đã được dùng. Đường dẫn `/api`
được proxy nguyên dạng tới `http://127.0.0.1:8000`, cấu hình trong `vite.config.js`.
Không đọc `.env` private ở thư mục gốc hoặc đưa credentials backend vào frontend.
Xem [README backend](../backend/README.md) cho cách chạy API.

## Build và xem thử

```bash
npm run build
npm run preview
```

`npm test` kiểm tra quy tắc ưu tiên timestamp, giữ số 0/âm và đơn vị, cô lập lỗi,
giới hạn request đồng thời và hủy request; không cần backend hoặc dependency test.
`node tests/history.test.js` kiểm tra ngày Việt Nam, ngày hợp lệ, cột giờ, tham số
phân trang và xác thực response lịch sử. Kiểm chứng giao diện bằng trình duyệt
đã bao gồm bộ lọc, phân trang 50 + 20 dòng, desktop/mobile, lỗi/thử lại và hủy request.

Build ở `dist/`, không đưa vào Git. Preview tại <http://localhost:4173> dùng cùng
proxy local. Khi deploy static build cần reverse proxy `/api` riêng.

## Cấu trúc

- `src/App.jsx`: state chọn trạm/filter dùng chung giữa sidebar và map.
- `src/components/MapView.jsx`: vòng đời MapLibre, trạng thái tải/lỗi và WebGL worker.
- `src/components/StationSidebar.jsx`, `StationIcon.jsx`: danh sách, bộ lọc và icon.
- `src/components/StationDetail.jsx`, `StationReading.jsx`: box chi tiết và số đo dùng chung.
- `src/components/HistoryDialog.jsx`, `HourlyHistoryTable.jsx`: modal lọc và bảng theo giờ.
- `src/hooks/useHourlyHistory.js`, `src/api/history.js`: tải từng trang, xác thực và hủy request.
- `src/history.js`, `src/styles/history.css`: quy tắc lịch Việt Nam và giao diện lịch sử.
- `src/components/PredictionInputsPanel.jsx`, `src/hooks/usePredictionInputs.js`:
  chọn mốc, auto-fill, chỉnh sửa/khôi phục và hủy request khi đổi lựa chọn.
- `src/api/predictionInputs.js`, `src/styles/predictionInputs.css`: hợp đồng API và CSS input.
- `src/predictionInputs.js`: nhãn đầu vào tiếng Việt và cách trình bày cờ dữ liệu.
- `src/api/predictions.js`, `src/hooks/usePrediction.js`: POST dự báo và GET thực tế,
  khóa lượt chạy, hủy và thử lại riêng từng bước.
- `src/components/PredictionResult.jsx`, `ForecastChart.jsx`: kết quả, biểu đồ và bảng sai số.
- `src/predictions.js`, `src/styles/predictions.css`: ghép sai số, ngắt đoạn thiếu và giao diện kết quả.
- `src/hooks/useStations.js`: tải metadata, số đo và hủy/làm mới request.
- `src/stations.js`: nhãn/icon, tọa độ và quy tắc ưu tiên timestamp trạm nước.
- `src/mapConfig.js`: tâm nhìn, zoom, vùng điều hướng và các vùng ẩn nhãn ngoài khơi.
- `src/maps/mapStyle.js`: biên giới đất liền, tên biển tiếng Việt và lọc nhãn ngoài khơi.
- `src/maps/stationMarkers.js`: tạo marker theo loại trạm và kiểm tra tọa độ.
- `src/maps/focusStation.js`: focus trạm trong vùng map còn nhìn thấy.
- `src/api/client.js`, `src/api/stations.js`: đọc API danh sách/latest, timeout/hủy request.
- `src/main.jsx`, `src/styles.css`: khởi tạo React, CSS MapLibre và Tailwind.

Tham khảo: [OpenFreeMap](https://openfreemap.org/quick_start/),
[schema OpenMapTiles](https://openmaptiles.org/docs/schema/),
[biểu thức MapLibre](https://maplibre.org/maplibre-style-spec/expressions/).
