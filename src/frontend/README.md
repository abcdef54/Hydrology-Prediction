# Frontend thủy văn

React + Vite + Tailwind CSS. Bản đồ dùng MapLibre GL JS và vector tiles theo schema
OpenMapTiles, dữ liệu OpenStreetMap do OpenFreeMap cung cấp. Không cần API key.
Trang hiện chỉ có bản đồ toàn màn hình, mở sẵn tại Huế, zoom 12; có thể kéo và zoom.

## Chạy local

Cần Node.js 22.12 trở lên (đã kiểm tra với Node.js 24).

```bash
cd /home/my_ubuntu/projects/hydrology/src/frontend
npm ci
npm run dev
```

Mở <http://localhost:5173>. Bản đồ cần Internet và trình duyệt hỗ trợ WebGL.
Trang hiện không gọi API backend; backend có thể tiếp tục chạy ở terminal riêng.
Có trạng thái tải và thông báo khi không truy cập được nguồn bản đồ.

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

## Kết nối backend cho task sau

Vite giữ cố định cổng 5173, báo lỗi nếu cổng đã được dùng. Đường dẫn `/api`
được proxy nguyên dạng tới `http://127.0.0.1:8000`, cấu hình trong `vite.config.js`.
Không đọc `.env` private ở thư mục gốc hoặc đưa credentials backend vào frontend.
Xem [README backend](../backend/README.md) cho cách chạy API.

## Build và xem thử

```bash
npm run build
npm run preview
```

Build ở `dist/`, không đưa vào Git. Preview tại <http://localhost:4173> dùng cùng
proxy local. Khi deploy static build cần reverse proxy `/api` riêng.

## Cấu trúc

- `src/App.jsx`: vùng chứa bản đồ toàn màn hình.
- `src/components/MapView.jsx`: vòng đời MapLibre, trạng thái tải/lỗi và WebGL worker.
- `src/mapConfig.js`: tâm nhìn, zoom, vùng điều hướng và các vùng ẩn nhãn ngoài khơi.
- `src/maps/mapStyle.js`: biên giới đất liền, tên biển tiếng Việt và lọc nhãn ngoài khơi.
- `src/api/stations.js`: hàm gọi API đã có, hiện chưa dùng trên trang bản đồ.
- `src/main.jsx`, `src/styles.css`: khởi tạo React, CSS MapLibre và Tailwind.

Tham khảo: [OpenFreeMap](https://openfreemap.org/quick_start/),
[schema OpenMapTiles](https://openmaptiles.org/docs/schema/),
[biểu thức MapLibre](https://maplibre.org/maplibre-style-spec/expressions/).
