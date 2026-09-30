# Báo cáo hydrology_training_1h

## Tổng quan

- Số dòng: 7,667
- Số predictor features: 176
- Số target: 5
- Tổng số cột, gồm timestamp: 182
- Khoảng thời gian UTC: 2025-11-14 18:00:00+00:00 đến 2026-09-30 04:00:00+00:00
- Timeline: đều 60min, lấy theo coverage của mực nước Dã Viên
- Topology candidates: /home/my_ubuntu/projects/hydrology/reports/da_vien_candidate_input_stations.csv
- Full topology relations cross-check: /home/my_ubuntu/projects/hydrology/reports/station_hydrologic_relations.csv
- Trạm mưa được dùng: 14
- Trạm mực nước được dùng: 1
- Hồ chứa được dùng: 4
- Trạm gió được dùng: 2

Reservoir được forward-fill chỉ từ quá khứ và tối đa 60 phút; không nội suy. Rain, water level và wind không được forward-fill. Source-missing và forward-filled flags được giữ riêng cho reservoir.

- Wind network missing rate (chỉ missing khi cả hai trạm cùng thiếu): 54.6498%

## Missing rate theo nhóm feature

| feature_group | column_count | mean_missing_rate | min_missing_rate | max_missing_rate |
| --- | --- | --- | --- | --- |
| rain_current | 14 | 3.8905% | 0.0000% | 21.6121% |
| rain_rolling | 56 | 4.1940% | 0.0000% | 21.9121% |
| missing_flags | 37 | 0.0000% | 0.0000% | 0.0000% |
| water_level_current | 1 | 0.0000% | 0.0000% | 0.0000% |
| water_level_lags | 2 | 0.0261% | 0.0130% | 0.0391% |
| reservoir_current | 14 | 87.4136% | 79.9661% | 99.9739% |
| reservoir_lags | 28 | 87.4136% | 79.9661% | 99.9739% |
| reservoir_forward_fill_flags | 14 | 0.0000% | 0.0000% | 0.0000% |
| wind_current | 8 | 54.8846% | 54.6628% | 55.1063% |
| wind_network_indicators | 2 | 0.0000% | 0.0000% | 0.0000% |
| targets | 5 | 0.1200% | 0.0130% | 0.3130% |

## Coverage theo nguồn

| source | column_count | mean_coverage_rate | min_coverage_rate | max_coverage_rate |
| --- | --- | --- | --- | --- |
| rain | 14 | 96.1095% | 78.3879% | 100.0000% |
| water_level | 1 | 100.0000% | 100.0000% | 100.0000% |
| reservoir_after_bounded_fill | 14 | 12.5864% | 0.0261% | 20.0339% |
| wind_network_either_station | 8 | 45.3502% | 45.3502% | 45.3502% |

## Station thực sự được sử dụng

| type | station_id | station_name | relation_to_da_vien | raw_observations_selected | observed_bins_in_timeline | source_variables | predictor_columns |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rain | 228006 | VPTT PCTTHUE | upstream | 36052 | 6010 | depth | 6 |
| rain | 677493 | TT Khe Tre (Phú Lộc) | upstream | 48028 | 7667 | depth | 6 |
| rain | 750006 | Hương Nguyên (A Lưới) | upstream | 46780 | 7586 | depth | 6 |
| rain | 750012 | Đập, Thủy điện Thượng Nhật (Phú Lộc) | upstream | 45836 | 7306 | depth | 6 |
| rain | 750013 | Lưu vực Thủy điện Thượng Nhật (Phú Lộc) | upstream | 41836 | 6636 | depth | 6 |
| rain | 750015 | Hồ Tà Rinh (Phú Lộc) | upstream | 47278 | 7567 | depth | 6 |
| rain | 750024 | An Tây (Thuận Hóa) | same_catchment | 47761 | 7623 | depth | 6 |
| rain | 750027 | Thủy Thanh (Hương Thủy) | same_catchment | 45111 | 7184 | depth | 6 |
| rain | 750028 | Hương Sơn (Phú Lộc) | upstream | 47512 | 7582 | depth | 6 |
| rain | 750029 | Thượng Quảng (Phú Lộc) | upstream | 47779 | 7631 | depth | 6 |
| rain | 750042 | Trạm kiểm lâm Khe Mỏ Rang | upstream | 47894 | 7649 | depth | 6 |
| rain | 750046 | Vỹ Dạ | same_catchment | 48028 | 7667 | depth | 6 |
| rain | 750061 | Thủy điện Bình Điền (Hương Trà) | upstream | 48017 | 7667 | depth | 6 |
| rain | 750063 | Phú Sơn (Hương Thủy) | upstream | 46315 | 7387 | depth | 6 |
| reservoir | 2 | Thủy điện Bình Điền | upstream | 2534 | 672 | htl, qden, qdi | 15 |
| reservoir | 3 | Hồ Tả Trạch | upstream | 2257 | 768 | htl, hhl, qden, qdi, mucnuocsong | 25 |
| reservoir | 6 | Thủy điện Thượng Lộ | upstream | 3041 | 719 | htl, qden, qdi | 15 |
| reservoir | 7 | Thủy điện Thượng Nhật | upstream | 1136 | 184 | htl, qden, qdi | 15 |
| water_level | 74647403 | Trạm Dã Viên (Sông Hương) | same_catchment | 45982 | 7667 | depth | 4 |
| wind | 28f350b6-8149-45b0-98d2-49c3096cd9a3 | Cảng Thuận An | outside_catchment | 44205 | 3476 | ws, wsg, wd, wdg | 8 |
| wind | 2d9b1c98-16ec-4681-9d23-0769a49aedaa | Cảng Tư Hiền | outside_catchment | 43803 | 3442 | ws, wsg, wd, wdg | 8 |

## Biến hồ chứa được giữ

| station_id | station_name | raw_variables_kept |
| --- | --- | --- |
| 3 | Hồ Tả Trạch | htl, hhl, qden, qdi, mucnuocsong |
| 2 | Thủy điện Bình Điền | htl, qden, qdi |
| 6 | Thủy điện Thượng Lộ | htl, qden, qdi |
| 7 | Thủy điện Thượng Nhật | htl, qden, qdi |

Các tên htl, hhl, qden, qdi, mucnuocsong được giữ đúng theo raw data. Builder không suy đoán tên đầy đủ, unit hoặc đổi đơn vị. Các cột hồ chứa không có bất kỳ giá trị nào giao với timeline Dã Viên bị loại để tránh feature 100% missing.

Timestamp hồ chứa không có timezone trong raw data. Builder diễn giải theo timezone cấu hình Asia/Bangkok, sau đó đổi sang UTC. Có thể thay đổi bằng tham số --reservoir-timezone nếu metadata chính thức xác nhận khác.
 Forward-fill hồ chứa bị chặn sau 60 phút và có flag riêng; giới hạn có thể đổi bằng --reservoir-ffill-limit-minutes.

## Quy tắc tổng hợp và feature engineering

- Rain depth: cộng các bản ghi trong cùng bin 60 phút; rolling 1h, 3h, 6h, 24h gồm thời điểm t và chỉ dùng [t-window+1, t]. Cửa sổ thiếu bất kỳ bin nào sẽ là missing.
- Dã Viên water level: trung bình trong bin 60 phút; lag 60m, 180m.
- Reservoir: trung bình raw values trong bin 60 phút; forward-fill từ quá khứ tối đa 60 phút; lag 60m và 180m được tạo sau bước fill. Source-missing và forward-filled flags cho biết nguồn gốc từng giá trị.
- Wind: ws và wsg lấy trung bình trong bin; wd và wdg lấy quan sát cuối cùng trong bin. Dùng riêng Cảng Thuận An và Cảng Tư Hiền theo yêu cầu; wind_network_missing chỉ bằng 1 khi cả hai trạm cùng thiếu, và wind_network_available_station_count cho biết có 0, 1 hay 2 trạm.
- Hai trạm gió này được topology phân loại outside_catchment nhưng được dùng theo chỉ định mới của người dùng, thay cho VPTT PCTTHUE.
- Targets: water level Dã Viên tại +1h, +3h, +6h, +12h, +24h.

## Leakage audit

- Số phép kiểm tra: 119
- Tất cả kiểm tra đạt: True
- Kiểm tra lỗi: []

Dataset giữ các dòng có target missing ở cuối chuỗi và các khoảng mất dữ liệu. Khi huấn luyện từng horizon, chỉ loại các dòng target tương ứng bị missing sau khi đã chia train/validation theo thời gian.
