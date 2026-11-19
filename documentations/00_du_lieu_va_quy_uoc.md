# 0. Dữ liệu và quy ước chung

Phần này gom những gì mọi giai đoạn dùng chung, để các file sau khỏi phải nhắc lại.

## 0.1 Nguồn dữ liệu

| Nguồn | Nội dung | Dùng ở giai đoạn |
|---|---|---|
| **API DCU Huế** (Postman "Huế - Dữ liệu ngập lụt") | Quan trắc mưa, mực nước, gió, hồ/đập; danh mục trạm | Tất cả |
| **ECMWF qua Open-Meteo, Single Runs** | Dự báo mưa theo từng lần chạy (`ecmwf_ifs`) | 1, 3, 4, 7–9 |
| **ECMWF qua Open-Meteo, Previous Runs** | Dự báo gió (bản dự báo từ 24 h trước) | 7, 8 |
| **Open-Meteo Marine** | Mực nước biển ngoài khơi dự báo (`sea_level_height_msl_previous_day1`) | 8 (và thử ở 9) |
| **Triều thiên văn** | Không phải nguồn ngoài: hồi quy điều hòa trên chính mực nước quan trắc của trạm (cửa sổ 7 hoặc 30 ngày, chỉ dùng dữ liệu đến thời điểm dự báo) | 7, 8, 9 |
| GloFAS | Lưu lượng sông dự báo (theo ngày, trễ khoảng 1 ngày) | 9: chỉ sàng lọc bằng hồi quy tuyến tính (không giúp), chưa thử trên GRU |

### Quan trắc từ API

- Toàn bộ dữ liệu API trả về được tải vào `data/` (SQLite, CSV) và PostgreSQL: 2,6 triệu bản ghi mưa (56 mã trạm), 742 nghìn bản ghi mực nước (17 trạm), 96 nghìn bản ghi gió, 40 nghìn bản ghi hồ/đập. Số bản ghi đã đối soát khớp với số API báo.
- **Lịch sử chỉ bắt đầu từ cuối 2025**: mưa từ 31/10/2025, mực nước từ 14/11/2025, gió từ 05/11/2025. API không trả dữ liệu cũ hơn. Vì vậy cả quá trình chỉ có **khoảng một năm dữ liệu**, và đây là giới hạn gốc của nghiên cứu (rất ít đợt bão/lũ để model học).
- Đơn vị của mực nước (`depth`) và gió không được API mô tả; mực nước được xử lý theo mm.
- Dữ liệu được đồng bộ vào PostgreSQL bằng worker (mỗi 11 phút khi chạy ứng dụng, xem [02](02_ung_dung_web.md)); các dataset huấn luyện là snapshot trích từ đó.
- Một số sensor bị treo (gửi giá trị không đổi nhiều ngày), ví dụ Dã Viên 01–12/02 và 20–25/06/2026. Những giờ này bị loại khi dựng dataset.

### Dự báo mưa ECMWF

- Mỗi lần chạy ECMWF được lưu riêng (không ghép các lần chạy); feature là tổng mưa dự báo ở 0–3, 3–6, 6–12, 12–24 giờ tới, kèm cờ thiếu và tuổi của lần chạy.
- Cả 4 trạm mưa của Dã Viên rơi vào **cùng một ô lưới** (16,27°N, 107,26°E), nên dự báo giống nhau cho mọi trạm.
- **Hạn chế:** độ trễ phát hành được giả định 6 giờ, nhưng thời điểm dự báo thực sự có sẵn trong quá khứ chưa kiểm chứng được. Kết quả vì thế là "tái hiện hồi cứu" (retrospective), không phải bằng chứng vận hành thực tế.
- ECMWF dự báo thấp hơn quan trắc rất nhiều ở đợt mưa lớn: ngày 12/09/2026 dự báo 24 h là 109,9 mm trong khi trạm đo 404–543 mm. Nó nhận ra được xu hướng giảm sang ngày sau (11,2 mm so với 69–124 mm) nhưng thấp hơn rõ ở cả hai ngày.

### Dữ liệu chỉ tham khảo

Địa hình, DEM (Copernicus GLO-30, FABDEM) và tài liệu địa lý Huế chỉ được xem qua, không đưa vào model nào.

## 0.2 Chia dữ liệu theo giai đoạn

| Giai đoạn | Dataset | Chia train / val / test |
|---|---|---|
| 1 (Dã Viên local) | `da_vien_1h_2026-10-15` | 6.095 / 985 / 592 mẫu; test từ 19/09 23:00 đến 14/10 14:00 (mốc đầu của test cần đủ 96 giờ lịch sử nên bắt đầu muộn hơn 15/09) |
| 3–6 (regional, WSP, toán, hybrid) | `hue_regional_20261015_h6_h12_fixed_split_v1` (11 trạm) | train đến 29/07/2026; val 31/07 – 13/09; test từ 15/09. Dã Viên: 6.073 / 1.080 / 711 mẫu |
| 7–9 (từng trạm) | dựng theo cấu hình riêng mỗi trạm | train đến 29/07; val 31/07 – 14/09; test 15/09 – 14/10; **giai đoạn mới 15/10 – 12/11/2026** |

- Mỗi split cách nhau một khoảng 24 giờ (purge) để nhãn của split trước không lọt vào lịch sử của split sau.
- Các giờ ranh giới giữa các split cũng được chấm riêng.
- Con số 592 (giai đoạn 1) và 711 (giai đoạn 3–6) đều là test của Dã Viên nhưng đếm khác nhau: 592 đã loại các mốc chưa đủ 96 giờ lịch sử trong chính tập test, 711 thì không.

## 0.3 Quy ước đánh giá

- **Lỗi lớn:** `|dự báo − thực tế| > 200 mm`, chặt (đúng 200 mm không tính). Mentor yêu cầu xử lý hết mọi lỗi này.
- **Xếp hạng:** theo số lỗi > 200 mm trước, MAE hoặc RMSE sau. Chấp nhận MAE tăng nếu giảm được lỗi lớn.
- **Nhãn:** mực nước **trung bình giờ** trong khoảng (t − 1 h, t]. Câu hỏi còn mở với mentor: chấm theo trung bình giờ hay số đo tức thời lúc hh:00? Nếu tức thời thì kết quả Dã Viên `v1` đổi từ 0 / 12 lỗi thành 4 / 19 lỗi (+3 / +6 h).
- **Chọn model chỉ dựa vào validation.** Test được xem nhiều lần nên chỉ là kết quả khám phá (retrospective), không phải kiểm định độc lập.
- **Chấm mọi epoch trên mọi tập** (từ giai đoạn 7), kể cả giờ ranh giới và giai đoạn mới 15/10 – 12/11.
- **Mốc dự báo:** ba mốc chính của nghiên cứu là **+3, +6, +12 h**. Mốc +3 h là yêu cầu được thêm ở giai đoạn sau, nên các model trước đó có thể không có hoặc không tối ưu cho +3 h. Giai đoạn 1 dùng +1/+3/+6/+12/+24 h; giai đoạn 3–6 chủ yếu +6/+12 h (một số thí nghiệm có +1…+24 h hoặc +3 h); giai đoạn 7–9 dùng +3/+6/+12 h.
- Số lỗi là đếm các cặp (mốc dự báo, horizon) chồng lấn nhau, không phải số đợt lũ độc lập. Lỗi thường đến thành chuỗi liên tiếp của cùng một đợt.

## 0.4 Thuật ngữ

| Thuật ngữ | Nghĩa |
|---|---|
| Anchor | Thời điểm phát hành dự báo; mọi đầu vào chỉ dùng dữ liệu đến thời điểm này |
| +h | Mốc dự báo, h giờ sau anchor |
| Persistence | Dự báo "mực nước giữ nguyên như hiện tại"; là baseline |
| Residual / ResidualLSTM, ResidualGRU | Model học phần hiệu chỉnh so với persistence, rồi cộng lại với mực nước hiện tại |
| Sequence length | Số bước lịch sử đưa vào model (96 giờ ở giai đoạn 1–6; ngắn hơn, 12–25 giờ ở các trạm sau) |
| N200 | Số dự báo có lỗi > 200 mm |
| Seed | Hạt giống ngẫu nhiên khi train; kết quả có thể khác nhau nhiều giữa các seed |
| Ensemble | Lấy trung bình dự báo của nhiều checkpoint hoặc model |
| Regional | Một model dùng chung trọng số cho nhiều trạm |
| WSP | Weight-space programming: sửa trực tiếp trọng số, không train lại |
| Retrospective | Chấm trên dữ liệu đã có, không chứng minh được dự báo vận hành thực tế |
