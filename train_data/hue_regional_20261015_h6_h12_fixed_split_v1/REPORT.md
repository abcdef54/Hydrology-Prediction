# Regional dataset: chuẩn bị dữ liệu cho hai horizon +6h và +12h

Dataset này chưa được dùng huấn luyện model. Mỗi sample thuộc đúng một trạm nước, có lịch sử 96 giờ và hai nhãn theo timestamp chính xác. Mưa và forecast chỉ lưu một lần theo giờ, dùng chung cho mọi trạm.

## Các trạm và số mẫu

| Trạm | ID | Nhóm EDA1 | Train | Validation | Đánh giá thăm dò | Purged | Tổng | Thiếu nước (%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Cảng Cá Thuận An | 0074647402 | 1 | 6073 | 1080 | 723 | 48 | 7924 | 0.02 |
| Dã Viên | 0074647403 | 1 | 6073 | 1080 | 723 | 48 | 7924 | 0.02 |
| Trạm Tứ Phú (Sông Bồ) | 0074647404 | 1 | 6073 | 1080 | 723 | 48 | 7924 | 0.02 |
| Cầu Huỳnh Liên-Văn Trạch Hòa(Sông Ô Lâu) | 0074647606 | 0 | 6073 | 1080 | 723 | 48 | 7924 | 0.02 |
| Trạm Ngã Ba Sình (Sông Hương) | 0074647705 | 1 | 6073 | 1080 | 723 | 48 | 7924 | 0.02 |
| Cảng Tư Hiền | 0074648201 | 1 | 6073 | 1080 | 723 | 48 | 7924 | 0.02 |
| Cầu Hòa Xuân 1(Sông Ô Lâu) | 0097203841 | 2 | 1770 | 1080 | 723 | 48 | 3621 | 53.82 |
| Phú Xuân | 750043 | 0 | 6073 | 1080 | 723 | 48 | 7924 | 0.02 |
| An Cựu | 750044 | 2 | 6073 | 1080 | 723 | 48 | 7924 | 0.02 |
| Thôn Trung Làng, xã Quảng Thái | 751117 | 2 | 6073 | 1080 | 723 | 48 | 7924 | 0.02 |
| Thôn Phò Nam A, xã Quảng Thọ | 751118 | 0 | 6073 | 1080 | 723 | 48 | 7924 | 0.02 |

## Phạm vi và cách chia thời gian

Dữ liệu giờ: 2025-11-15 00:00:00+07:00 đến 2026-10-15 14:00:00+07:00. Tổng 82,861 sample. Mốc giờ trong bảng dưới là UTC+7; khoảng kết thúc là mốc cuối có sample.

| Split | Anchor đầu | Anchor cuối | Số sample |
|---|---|---|---:|
| purged | 2026-07-30 00:00:00+07:00 | 2026-09-14 23:00:00+07:00 | 528 |
| test_exploratory | 2026-09-15 00:00:00+07:00 | 2026-10-15 02:00:00+07:00 | 7953 |
| train | 2025-11-18 23:00:00+07:00 | 2026-07-29 23:00:00+07:00 | 62500 |
| validation | 2026-07-31 00:00:00+07:00 | 2026-09-13 23:00:00+07:00 | 11880 |

Dataset dùng duy nhất một bộ train/validation/test cố định. Train là phần trước 2026-07-31 00:00:00+07:00; validation từ 2026-07-31 00:00:00+07:00 đến trước 2026-09-15 00:00:00+07:00; test từ 2026-09-15 00:00:00+07:00 đến hết quan trắc 2026-10-15T14:00:00+07:00. Các mốc này là ranh giới khoảng dữ liệu, không phải mọi giờ đều đủ điều kiện làm anchor. Purge 24h loại anchor ở cuối train và cuối validation, không đẩy ngày bắt đầu validation/test sang ngày khác. Nhãn +6h/+12h đều phải nằm trước ranh giới tập tiếp theo. Anchor test cuối cách quan trắc cuối ít nhất 12 giờ. Lịch sử đầu vào được phép dùng quan trắc ở tập trước nếu chúng đã xảy ra trước anchor. Scaler được fit lại chỉ trên train; không có nhiều fold trong dataset này. Test vẫn là đánh giá thăm dò vì dữ liệu giai đoạn này đã từng được nghiên cứu.

## Input, target và lựa chọn feature

Mặc định x có shape [96, 68], y có shape [2], station_index là số nguyên riêng. Hai water channels là mực nước chuẩn hóa riêng từng trạm và cờ thiếu. Mỗi trạm mưa có 13 channels: tổng mưa quá khứ [1, 3, 6, 12, 24, 48] giờ, cờ thiếu cho từng tổng và cờ giờ mưa ước lượng 5/6 mẫu. Có thể chọn bất kỳ tập con nào trong 55 trạm mưa bằng alias hoặc ID, không thay đổi danh sách sample.

Forecast chỉ gồm ô ECMWF A Lưới 16.274164°N, 107.264960°E, bốn tổng 0–3/3–6/6–12/12–24h, cờ thiếu, tuổi forecast, cờ thiếu tuổi và ba cờ observed/assumed/hindcast. Các giá trị của một anchor đến từ cùng một run. Metadata chứa run ID, initialization, retrieval, available_at, grid, model version và provenance. Input chỉ dùng forecast tại timestep cuối (anchor_only).

Raw NaN được giữ trong Parquet. Reader chuẩn hóa bằng thống kê train, điền 0 trong không gian chuẩn hóa và giữ cờ thiếu. Mask và cờ provenance không được chuẩn hóa. Target là mực nước quan trắc giờ tại đúng t+6h/t+12h, kèm delta để kiểm tra. Target thiếu bị loại, không nội suy. Sample có H(t) hợp lệ, đủ 96 giờ trên lưới và ít nhất 80% lịch sử nước có số đo. Không loại thêm sample do thay tổ hợp mưa hoặc tắt forecast.

Persistence dùng hệ số riêng theo station và horizon: scale = water_std / target_std; offset = (water_mean − target_mean) / target_std. Không lấy cao độ DEM làm mốc nước và không trộn số đo các trạm thành một mốc tuyệt đối.

Mỗi timestep có thêm flood_season dùng chung cho mọi trạm: 1 trong tháng 9–12, 0 trong các tháng còn lại theo giờ Việt Nam. Quy ước này giống dataset Dã Viên hiện tại; đây là lịch mùa, không phải nhãn xác nhận lũ. Đặc trưng giữ nguyên 0/1 và không chuẩn hóa.

Calendar lưu đủ hour_sin/hour_cos theo chu kỳ 24 giờ và day_sin/day_cos theo chu kỳ năm. Công thức ngày dùng (dayofyear−1)/365 hoặc /366 theo năm địa phương, nên xử lý đúng ngày 29/02 và ranh giới năm. Tất cả lấy giờ Việt Nam; sin/cos giữ nguyên trong [-1,1], không fit scaler. Reader cho chọn time_features=none/hour/day_of_year/all mà không đổi sample hoặc rebuild: tương ứng mặc định 68/70/70/72 channels khi dùng bốn trạm mưa và forecast. flood_season luôn được giữ.

## Mưa quan trắc và tỷ lệ thiếu

Giữ đầy đủ 55 trạm trong pool để có thể đổi tổ hợp sau này; việc lưu một trạm không có nghĩa khuyến nghị dùng nó. Lựa chọn mặc định gồm Nhâm (A Lưới, phía tây), Hồng Thái 750062 (A Lưới), Bạch Mã 750041 (phía đông nam, khác VQG/Đỉnh Bạch Mã) và Hồ Hòa Mỹ 682816 (phía tây bắc). Các trạm Khe Mỏ Rang, Hương Sơn và Phú Bài được giữ để kiểm chứng các tín hiệu đã thấy trong EDA2. Một số trạm thiếu toàn bộ train; phải xem bảng dưới trước khi chọn.

Mưa giờ dùng đúng kết quả EDA2: tổng 6/6 mẫu 10 phút; 5/6 mẫu được nhân 6/5 và gắn cờ ước lượng; ít hơn 5 mẫu giữ thiếu. Tổng nhiều giờ yêu cầu đầy đủ mọi giờ trong cửa sổ. Không biến thiếu thành không mưa.

| Trạm mưa | ID | Thiếu giờ toàn kỳ (%) | Giờ ước lượng (%) | Thiếu tại train anchors (%) |
|---|---|---:|---:|---:|
| Hồ chứa nước Thủy Yên | 014690 | 0.22 | 0.25 | 0.02 |
| Hồ chứa nước Khe Ngang | 014696 | 0.00 | 0.02 | 0.00 |
| Hồ chứa nước Truồi | 014699 | 0.36 | 0.30 | 0.02 |
| Cảng Chân Mây | 228005 | 38.65 | 0.02 | 49.55 |
| VPTT PCTTHUE | 228006 | 20.66 | 0.02 | 25.75 |
| Quan Tượng Đài (Phú Xuân) | 567011 | 0.21 | 0.05 | 0.12 |
| TT Khe Tre (Phú Lộc) | 677493 | 0.00 | 0.02 | 0.00 |
| TT Sịa (Quảng Điền) | 677851 | 0.11 | 0.24 | 0.08 |
| Tà Lương (A Lưới) | 678628 | 0.00 | 0.02 | 0.00 |
| Hồ Thọ Sơn (Hương Trà) | 681524 | 0.00 | 0.10 | 0.00 |
| Hồ Hòa Mỹ (Phong Điền) | 682816 | 0.30 | 0.11 | 0.36 |
| TT Phú Đa (Phú Vang) | 750001 | 3.49 | 0.12 | 0.91 |
| Hồ Mỹ Xuyên (Phong Điền) | 750002 | 0.00 | 0.07 | 0.00 |
| Thị trấn A Lưới (A Lưới) | 750003 | 0.25 | 0.09 | 0.33 |
| Hồng Vân (A Lưới) | 750005 | 0.00 | 0.02 | 0.00 |
| Hương Nguyên (A Lưới) | 750006 | 1.54 | 0.31 | 1.10 |
| Nhâm (A Lưới) | 750007 | 0.00 | 0.02 | 0.00 |
| A Đớt (A Lưới) | 750008 | 0.00 | 0.02 | 0.00 |
| Lộc Tiến (Phú Lộc) | 750009 | 3.05 | 0.24 | 4.03 |
| Giang Hải (Phú Lộc) | 750010 | 0.00 | 0.05 | 0.00 |
| Hồ A Lá (A Lưới) | 750011 | 0.00 | 0.02 | 0.00 |
| Đập, Thủy điện Thượng Nhật (Phú Lộc) | 750012 | 4.57 | 0.07 | 6.04 |
| Lưu vực Thủy điện Thượng Nhật (Phú Lộc) | 750013 | 12.85 | 0.02 | 16.33 |
| Hồ chứa nước Phú Bài (Hương Thủy) | 750014 | 1.56 | 0.24 | 0.00 |
| Hồ Tà Rinh (Phú Lộc) | 750015 | 1.59 | 0.80 | 1.99 |
| Trạm bơm Sơn Bổn | 750016 | 0.07 | 0.06 | 0.10 |
| Đồn Biên phòng Lăng Cô (Phú Lộc) | 750021 | 0.00 | 0.02 | 0.00 |
| Đồn Biên phòng Cảng Chân Mây (Phú Lộc) | 750022 | 2.68 | 0.02 | 0.16 |
| Quảng Phú (Quảng Điền) | 750023 | 1.15 | 0.02 | 1.51 |
| An Tây (Thuận Hóa) | 750024 | 0.55 | 0.06 | 0.72 |
| Thị trấn Phong Điền (Phong Điền) | 750025 | 0.67 | 0.02 | 0.00 |
| Bình Tiến (Hương Trà) | 750026 | 0.00 | 0.04 | 0.00 |
| Thủy Thanh (Hương Thủy) | 750027 | 6.05 | 0.10 | 8.00 |
| Hương Sơn (Phú Lộc) | 750028 | 1.38 | 0.04 | 1.42 |
| Thượng Quảng (Phú Lộc) | 750029 | 0.52 | 0.04 | 0.68 |
| Vườn Quốc gia Bạch Mã (Phú Lộc) | 750030 | 0.00 | 0.04 | 0.00 |
| Trạm bơm Quảng Thành (Quảng Điền) | 750032 | 0.00 | 0.04 | 0.00 |
| VP QLKTCTTL TP Huế (TP Huế) | 750033 | 0.00 | 0.20 | 0.00 |
| Trạm bơm Vinh Hà (Phú Vang) | 750034 | 0.00 | 0.04 | 0.00 |
| Cảng Tư Hiền | 750035 | 0.17 | 0.02 | 0.00 |
| Hương Chữ (Hương Trà) | 750040 | 1.51 | 0.04 | 1.89 |
| Bạch Mã | 750041 | 0.00 | 0.07 | 0.00 |
| Trạm kiểm lâm Khe Mỏ Rang | 750042 | 0.25 | 0.07 | 0.03 |
| Đan Điền (Đập Cửa Lác) | 750045 | 0.00 | 0.02 | 0.00 |
| Vỹ Dạ | 750046 | 0.00 | 0.02 | 0.00 |
| Lộc An | 750047 | 1.58 | 0.05 | 2.09 |
| Cảng Thuận An | 750059 | 0.00 | 0.10 | 0.00 |
| Thủy điện Bình Điền (Hương Trà) | 750061 | 0.00 | 0.14 | 0.00 |
| Hồng Thái (A Lưới) | 750062 | 0.04 | 0.14 | 0.05 |
| Phú Sơn (Hương Thủy) | 750063 | 3.57 | 0.05 | 0.97 |
| Điền Hương (Phong Điền) | 750064 | 1.72 | 0.15 | 1.48 |
| Xuân Lộc (Phú Lộc) | 750065 | 9.64 | 0.21 | 11.18 |
| Đỉnh Bạch Mã (Phú Lộc) | 750066 | 0.00 | 0.02 | 0.00 |
| Đập Thủy điện Rào Trăng 3 (Phong Điền) | 958573 | 2.33 | 0.10 | 0.00 |
| Đập Thủy điện Rào Trăng 4 (Phong Điền) | 959428 | 2.52 | 0.04 | 3.03 |

## Độ đầy đủ và nguồn gốc forecast

Có 8031/8031 anchor giờ có đủ forecast. Số anchor có bằng chứng observed availability là 0; số dùng assumed availability là 8031. Số anchor gắn cờ hindcast là 4291.

| Loại archive | Số anchor giờ |
|---|---:|
| hindcast_49r1 | 4291 |
| archive_50r1_unverified | 3740 |

## Giới hạn nghiên cứu và leakage

Forecast có cả hindcast/tái dựng và archive chưa xác minh thời điểm phát hành. Chính sách +6h là giả định nghiên cứu; không coi initialization hoặc retrieval muộn là bằng chứng forecast từng có sẵn trong quá khứ. Các cờ và metadata bảo toàn sự phân biệt này. Quan trắc cũng chưa chứng minh đầy đủ receipt trước anchor. Dataset phù hợp nghiên cứu hồi cứu, chưa chứng minh chất lượng vận hành thời gian thực.

Nhóm EDA1 và danh sách trạm được xác định từ dữ liệu trước cutoff, nhưng đã xem một phần inner validation; đây là giới hạn selection-informed của nghiên cứu, không phải phép chọn độc lập ở mọi fold. Không tối ưu trạm mưa theo phần sau cutoff. Các nhóm chỉ là metadata nghiên cứu, không tự gán ý nghĩa vật lý hoặc ngưỡng báo động.

Không ghi đè dataset/model cũ hoặc sửa PostgreSQL. Các hash nguồn và cấu hình nằm trong manifest. Xem validation.json trong cùng thư mục để biết các phép kiểm tra đã thực sự chạy. README của src/ai/regional_data mô tả cách build và đọc dữ liệu.

## Kiểm chứng sau khi sửa mốc chia

Đã chạy 22 tests cho dataset và kiến trúc, tất cả pass. Audit thực tế đối chiếu 165.722 nhãn theo timestamp, tính lại forecast tại 8.031 giờ và kiểm tra các scaler chỉ fit từ tập train. Cả CLI và reader tự chọn split fixed khi không truyền regional-fold. Forward trên batch validation có shape [8,2]; không chạy training.

Các bảng nước, mưa, forecast và lịch được đối chiếu trực tiếp với bản v3 trên cùng khoảng thời gian và giống hệt nhau. Scaler được fit lại theo train mới. Bản mới không dùng các fold EDA2 hoặc dữ liệu quan trắc sau 15/10/2026 14:00 UTC+7.
