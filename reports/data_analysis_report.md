# Báo cáo phân tích khả năng dự báo thủy văn Huế

## 1. Executive summary

Dữ liệu hiện có hỗ trợ **thử nghiệm dự báo mực nước Dã Viên ngắn hạn**, không đủ bằng chứng để gọi là mô hình dự báo lũ vận hành và không plug-compatible với pipeline pretrained OpenHydroNet. Dữ liệu mới mở rộng mạnh theo không gian, nhưng chuỗi target vẫn chỉ dài 10.49 tháng; số trạm tăng không thay thế cho nhiều mùa mưa/lũ độc lập.

Các giá trị và sai số dưới đây giữ nguyên đơn vị nguồn. Dataset không mô tả đơn vị đáng tin cậy, nên báo cáo không chuyển đổi hoặc suy đoán cm/m, mm hay đơn vị gió.

## 2. Dataset inventory

| file | rows | columns | duplicate_rows | timestamp_column | start | end |
| --- | --- | --- | --- | --- | --- | --- |
| rain_observations.csv | 2603597 | 4 | 0 | time_point | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 |
| rain_stations.csv | 55 | 12 | 0 | NA | NA | NA |
| reservoir_observations.csv | 39894 | 26 | 0 | ngaylaysolieu | 2023-01-27 07:00:00+00:00 | 2026-10-19 13:00:00+00:00 |
| reservoirs.csv | 19 | 17 | 0 | NA | NA | NA |
| water_level_observations.csv | 742682 | 4 | 0 | time_point | 2025-11-14 18:50:00+00:00 | 2026-09-30 04:00:00+00:00 |
| water_level_stations.csv | 17 | 14 | 0 | date_created | 2026-09-29 23:00:00.890000+00:00 | 2026-09-29 23:00:00.906000+00:00 |
| wind_observations.csv | 95974 | 9 | 0 | time_point | 2025-11-05 20:44:00+00:00 | 2026-09-30 04:10:00+00:00 |
| wind_stations.csv | 4 | 11 | 0 | date_created | 2026-08-25 23:00:00.718000+00:00 | 2026-08-25 23:00:00.728000+00:00 |
| hydrology_training_10min.csv | 45992 | 25 | 0 | timestamp_utc | 2025-11-14 18:50:00+00:00 | 2026-09-30 04:00:00+00:00 |

### Coverage theo observation ID và station metadata

| station_type | total_stations | stations_with_observations | stations_without_observations | total_observations | minimum_observations | median_observations | maximum_observations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rain | 56 | 56 | 0 | 2603597 | 27382 | 4.792e+04 | 48028 |
| water_level | 17 | 17 | 0 | 742682 | 20072 | 4.598e+04 | 45990 |
| wind | 4 | 4 | 0 | 95974 | 3983 | 2.389e+04 | 44205 |
| reservoir | 19 | 17 | 2 | 39894 | 0 | 2221 | 4690 |

Manifest và raw files cùng hiện diện. Raw CSV/GZIP không bị thay đổi. Thống kê aggregate toàn bộ water-level observations có min=-6341, p95=1036, max=6650; toàn bộ rain observations có zero ratio=93.36%, max=40.4. Đây là distribution trong raw units, không phải xác nhận vật lý. Chi tiết từng trạm nằm trong `distribution_by_station.csv`.

## 3. Temporal coverage

| source | series_id | start | end | duration_days | actual_records | expected_records | availability_ratio | longest_gap_minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rain_observations.csv | 14690 | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 333.5 | 47902 | 48030 | 0.9973 | 1010 |
| rain_observations.csv | 14696 | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 333.5 | 48028 | 48030 | 1 | 20 |
| rain_observations.csv | 14699 | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 333.5 | 47839 | 48030 | 0.996 | 1660 |
| rain_observations.csv | 228005 | 2026-03-24 00:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 190.2 | 27382 | 27384 | 0.9999 | 20 |
| rain_observations.csv | 228006 | 2026-01-22 19:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 250.4 | 36052 | 36054 | 0.9999 | 20 |
| rain_observations.csv | 567011 | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 333.5 | 47932 | 48030 | 0.998 | 890 |
| rain_observations.csv | 677493 | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 333.5 | 48028 | 48030 | 1 | 20 |
| rain_observations.csv | 677851 | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 333.5 | 48000 | 48030 | 0.9994 | 60 |
| rain_observations.csv | 678628 | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 333.5 | 48028 | 48030 | 1 | 20 |
| rain_observations.csv | 681524 | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 333.5 | 48022 | 48030 | 0.9998 | 20 |
| rain_observations.csv | 682816 | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 333.5 | 47914 | 48030 | 0.9976 | 120 |
| rain_observations.csv | 750001 | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 333.5 | 46446 | 48030 | 0.967 | 1.053e+04 |
| rain_observations.csv | 750002 | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 333.5 | 48024 | 48030 | 0.9999 | 20 |
| rain_observations.csv | 750003 | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 333.5 | 47910 | 48030 | 0.9975 | 1110 |
| rain_observations.csv | 750005 | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 333.5 | 48028 | 48030 | 1 | 20 |
| rain_observations.csv | 750006 | 2025-11-04 06:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 329.9 | 46780 | 47508 | 0.9847 | 3180 |
| rain_observations.csv | 750007 | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 333.5 | 48028 | 48030 | 1 | 20 |
| rain_observations.csv | 750008 | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 333.5 | 48028 | 48030 | 1 | 20 |
| rain_observations.csv | 750009 | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 333.5 | 46559 | 48030 | 0.9694 | 1.264e+04 |
| rain_observations.csv | 750010 | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 333.5 | 48026 | 48030 | 0.9999 | 20 |
| rain_observations.csv | 750011 | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 333.5 | 48028 | 48030 | 1 | 20 |
| rain_observations.csv | 750012 | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 333.5 | 45836 | 48030 | 0.9543 | 1.94e+04 |
| rain_observations.csv | 750013 | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 333.5 | 41836 | 48030 | 0.871 | 6.193e+04 |
| rain_observations.csv | 750014 | 2025-10-31 15:10:00+00:00 | 2026-09-30 03:50:00+00:00 | 333.5 | 47263 | 48029 | 0.9841 | 7450 |
| rain_observations.csv | 750015 | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 333.5 | 47278 | 48030 | 0.9843 | 4240 |
| rain_observations.csv | 750016 | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 333.5 | 47992 | 48030 | 0.9992 | 300 |
| rain_observations.csv | 750021 | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 333.5 | 48028 | 48030 | 1 | 20 |
| rain_observations.csv | 750022 | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 333.5 | 46754 | 48030 | 0.9734 | 1.137e+04 |
| rain_observations.csv | 750023 | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 333.5 | 47479 | 48030 | 0.9885 | 5500 |
| rain_observations.csv | 750024 | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 333.5 | 47761 | 48030 | 0.9944 | 2660 |
| rain_observations.csv | 750025 | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 333.5 | 47705 | 48030 | 0.9932 | 3240 |
| rain_observations.csv | 750026 | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 333.5 | 48027 | 48030 | 0.9999 | 20 |
| rain_observations.csv | 750027 | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 333.5 | 45111 | 48030 | 0.9392 | 2.84e+04 |
| rain_observations.csv | 750028 | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 333.5 | 47512 | 48030 | 0.9892 | 4220 |
| rain_observations.csv | 750029 | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 333.5 | 47779 | 48030 | 0.9948 | 2120 |
| rain_observations.csv | 750030 | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 333.5 | 48028 | 48030 | 1 | 20 |
| rain_observations.csv | 750031 | 2025-10-31 15:10:00+00:00 | 2026-05-14 01:00:00+00:00 | 194.4 | 27994 | 27996 | 0.9999 | 20 |
| rain_observations.csv | 750032 | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 333.5 | 48027 | 48030 | 0.9999 | 20 |
| rain_observations.csv | 750033 | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 333.5 | 48011 | 48030 | 0.9996 | 20 |
| rain_observations.csv | 750034 | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 333.5 | 48027 | 48030 | 0.9999 | 20 |
| rain_observations.csv | 750035 | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 333.5 | 47951 | 48030 | 0.9984 | 720 |
| rain_observations.csv | 750040 | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 333.5 | 47202 | 48030 | 0.9828 | 5250 |
| rain_observations.csv | 750041 | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 333.5 | 48025 | 48030 | 0.9999 | 20 |
| rain_observations.csv | 750042 | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 333.5 | 47894 | 48030 | 0.9972 | 1090 |
| rain_observations.csv | 750045 | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 333.5 | 48028 | 48030 | 1 | 20 |
| rain_observations.csv | 750046 | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 333.5 | 48028 | 48030 | 1 | 20 |
| rain_observations.csv | 750047 | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 333.5 | 47267 | 48030 | 0.9841 | 7600 |
| rain_observations.csv | 750059 | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 333.5 | 48022 | 48030 | 0.9998 | 20 |
| rain_observations.csv | 750061 | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 333.5 | 48017 | 48030 | 0.9997 | 20 |
| rain_observations.csv | 750062 | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 333.5 | 48006 | 48030 | 0.9995 | 70 |
| rain_observations.csv | 750063 | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 333.5 | 46315 | 48030 | 0.9643 | 1.222e+04 |
| rain_observations.csv | 750064 | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 333.5 | 47191 | 48030 | 0.9825 | 5040 |
| rain_observations.csv | 750065 | 2025-12-16 22:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 287.2 | 41347 | 41364 | 0.9996 | 20 |
| rain_observations.csv | 750066 | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 333.5 | 48028 | 48030 | 1 | 20 |
| rain_observations.csv | 958573 | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 333.5 | 48024 | 48030 | 0.9999 | 20 |
| rain_observations.csv | 959428 | 2025-10-31 15:10:00+00:00 | 2026-09-30 04:00:00+00:00 | 333.5 | 46820 | 48030 | 0.9748 | 1.095e+04 |
| water_level_observations.csv | 11223344 | 2025-11-14 18:50:00+00:00 | 2026-09-30 04:00:00+00:00 | 319.4 | 45558 | 45992 | 0.9906 | 1371 |
| water_level_observations.csv | 55889944 | 2025-11-20 03:31:00+00:00 | 2026-09-30 04:00:00+00:00 | 314 | 44879 | 45219 | 0.9925 | 1840 |
| water_level_observations.csv | 74647402 | 2025-11-14 18:50:00+00:00 | 2026-09-30 04:00:00+00:00 | 319.4 | 45960 | 45992 | 0.9993 | 30 |
| water_level_observations.csv | 74647403 | 2025-11-14 18:50:00+00:00 | 2026-09-30 04:00:00+00:00 | 319.4 | 45982 | 45992 | 0.9998 | 30 |
| water_level_observations.csv | 74647404 | 2025-11-14 18:50:00+00:00 | 2026-09-30 04:00:00+00:00 | 319.4 | 45990 | 45992 | 1 | 30 |
| water_level_observations.csv | 74647606 | 2025-11-14 18:50:00+00:00 | 2026-09-30 04:00:00+00:00 | 319.4 | 45979 | 45992 | 0.9997 | 30 |
| water_level_observations.csv | 74647705 | 2025-11-14 18:50:00+00:00 | 2026-09-30 04:00:00+00:00 | 319.4 | 45981 | 45992 | 0.9998 | 30 |
| water_level_observations.csv | 74648201 | 2025-11-14 18:50:00+00:00 | 2026-09-30 04:00:00+00:00 | 319.4 | 45975 | 45992 | 0.9996 | 30 |
| water_level_observations.csv | 750043 | 2025-11-14 18:50:00+00:00 | 2026-09-30 04:00:00+00:00 | 319.4 | 45986 | 45992 | 0.9999 | 30 |
| water_level_observations.csv | 750044 | 2025-11-14 18:50:00+00:00 | 2026-09-30 04:00:00+00:00 | 319.4 | 45987 | 45992 | 0.9999 | 30 |
| water_level_observations.csv | 751115 | 2025-11-14 18:50:00+00:00 | 2026-09-30 04:00:00+00:00 | 319.4 | 43791 | 45992 | 0.9521 | 1.792e+04 |
| water_level_observations.csv | 751116 | 2025-11-14 18:50:00+00:00 | 2026-09-30 04:00:00+00:00 | 319.4 | 45988 | 45992 | 0.9999 | 30 |
| water_level_observations.csv | 751117 | 2025-11-14 18:50:00+00:00 | 2026-09-30 04:00:00+00:00 | 319.4 | 45988 | 45992 | 0.9999 | 30 |
| water_level_observations.csv | 751118 | 2025-11-14 18:50:00+00:00 | 2026-09-30 04:00:00+00:00 | 319.4 | 45989 | 45992 | 0.9999 | 30 |
| water_level_observations.csv | 75882244 | 2025-11-14 18:50:00+00:00 | 2026-09-30 04:00:00+00:00 | 319.4 | 45312 | 45992 | 0.9852 | 3317 |
| water_level_observations.csv | 75996644 | 2025-11-14 18:50:00+00:00 | 2026-07-31 14:30:00+00:00 | 258.8 | 37265 | 37271 | 0.9998 | 30 |
| water_level_observations.csv | 97203841 | 2026-05-13 19:00:00+00:00 | 2026-09-30 04:00:00+00:00 | 139.4 | 20072 | 20071 | 1 | 10 |
| wind_observations.csv | 28f350b6-8149-45b0-98d2-49c3096cd9a3 | 2025-11-05 20:45:00+00:00 | 2026-09-30 04:08:00+00:00 | 328.3 | 44205 | 94553 | 0.4675 | 2.373e+05 |
| wind_observations.csv | 2d9b1c98-16ec-4681-9d23-0769a49aedaa | 2025-11-05 20:44:00+00:00 | 2026-09-30 04:07:00+00:00 | 328.3 | 43803 | 94553 | 0.4633 | 2.395e+05 |
| wind_observations.csv | 63586eaa-f1cc-11f0-8f75-0692ec44dcd5 | 2026-08-21 05:00:00+00:00 | 2026-09-30 04:10:00+00:00 | 39.97 | 3983 | 5756 | 0.692 | 1.353e+04 |
| wind_observations.csv | 63604a34-f1cc-11f0-8f75-0692ec44dcd5 | 2026-08-21 05:00:00+00:00 | 2026-09-30 04:10:00+00:00 | 39.97 | 3983 | 5756 | 0.692 | 1.353e+04 |

Chuỗi Dã Viên chỉ dài khoảng 10.49 tháng. Không thể xác định số mùa lũ thực sự từ dữ liệu hiện có vì thiếu ngưỡng lũ chính thức và chuỗi chưa bao phủ nhiều năm độc lập.

## 4. Data quality

Wind Cảng Thuận An thiếu 54.7% trong bảng training 10 phút. Raw files không có duplicate rows; duplicate timestamp được đánh giá trong từng station series. Extreme được giữ nguyên vì không có units/QA flags đáng tin cậy. Có 1 observation ID không nối được metadata (rain:750031). Reservoir có 1 dòng mang ngày sau ngày xuất manifest 2026-09-30; cần kiểm tra timezone hoặc lỗi nhập liệu. Giá trị âm/cực lớn được xếp loại **unknown requiring domain verification**.

## 5. Spatial relationships

| type | id | name | longitude | latitude | coordinates_available |
| --- | --- | --- | --- | --- | --- |
| rain | 682816 | Hồ Hòa Mỹ (Phong Điền) | 107.3 | 16.5 | True |
| rain | 750059 | Cảng Thuận An | 107.6 | 16.55 | True |
| water_level | 74647403 | Trạm Dã Viên (Sông Hương) | 107.6 | 16.46 | True |
| wind | 759002 | Cảng Thuận An | 107.6 | 16.55 | True |
| reservoir | 16 | Trạm Dã Viên (Sông Hương) | NA | NA | False |

| name_a | name_b | distance_km | hydrologic_relationship |
| --- | --- | --- | --- |
| Cảng Thuận An | Cảng Thuận An | 0.2644 | UNKNOWN |
| Cảng Thuận An | Trạm Dã Viên (Sông Hương) | 12.03 | UNKNOWN |
| Trạm Dã Viên (Sông Hương) | Cảng Thuận An | 12.24 | UNKNOWN |
| Hồ Hòa Mỹ (Phong Điền) | Trạm Dã Viên (Sông Hương) | 27.67 | UNKNOWN |
| Hồ Hòa Mỹ (Phong Điền) | Cảng Thuận An | 34.56 | UNKNOWN |

Khoảng cách địa lý không chứng minh quan hệ thủy văn. Dataset không có catchment boundary, river network, upstream/downstream topology hoặc khóa nối hồ–trạm. Vì vậy quan hệ Hòa Mỹ → Dã Viên và vai trò Cảng Thuận An là **UNKNOWN**; chưa đủ căn cứ dùng rain Hòa Mỹ như forcing đại diện lưu vực Dã Viên.

## 6. Hydrologic signal analysis

### Persistence

| horizon_minutes | correlation | paired_observations |
| --- | --- | --- |
| 60 | 0.9904 | 45958 |
| 180 | 0.9416 | 45946 |
| 360 | 0.8649 | 45928 |

Correlation rất cao chỉ cho thấy target trơn/autocorrelated; nó làm persistence trở thành baseline khó vượt và không chứng minh khả năng dự báo lũ.

### Rainfall, wind, reservoir

Riêng rain Hòa Mỹ, correlation lớn nhất về trị tuyệt đối giữa rain lag và thay đổi mực nước 6 giờ xuất hiện ở lag 1h: r=0.053. Khi quét toàn bộ trạm mưa, trị tuyệt đối lớn nhất là `Không có trong station metadata` ở lag 0h: r=0.272. Đây chỉ là screening correlation; multiple comparisons và quan hệ lưu vực chưa biết nên không được diễn giải nhân quả. Kết quả đầy đủ nằm trong `all_station_lagged_signals.csv`.

Wind coverage yếu (45.3%); chỉ nên là feature ablation, không phải dependency bắt buộc. Reservoir correlations (86 series-variable pairs) không phải bằng chứng để merge vì hydrologic relationship vẫn UNKNOWN.

## 7. Event analysis

| percentile | threshold_raw_units | independent_events |
| --- | --- | --- |
| 0.9 | 575.2 | 126 |
| 0.95 | 675.9 | 24 |
| 0.99 | 1378 | 1 |

Các event là **high-water exceedance events theo percentile**, không được gọi là flood. Quy tắc gộp cho phép khoảng cách tối đa 6 giờ. Số event nhạy với threshold và không thay thế ngưỡng chính thức.

## 8. Leakage risks

Lag và target được đối chiếu bằng exact-shift. Kết quả chi tiết ở `leakage_audit.json`. Target columns không được đưa vào feature list; imputer/scaler/model chỉ fit trên train; split không shuffle. Rolling rain trong file hiện hữu được audit cả current-inclusive và strictly-past. Current rain tại issue time chỉ hợp lệ nếu thực sự sẵn có với đúng latency trong vận hành.

## 9. Baseline performance

| horizon_minutes | model | mae_best | rmse_best | nse_best | mae_persistence | rmse_persistence | nse_persistence | mae_improvement_vs_persistence_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 60 | ridge_selected_stations | 21.03 | 28.99 | 0.9589 | 26.27 | 37.11 | 0.9327 | 19.98 |
| 180 | ridge_selected_stations | 60.28 | 80.44 | 0.6841 | 66.53 | 95.02 | 0.5593 | 9.398 |
| 360 | ridge_selected_stations | 96.39 | 128.7 | 0.1928 | 100.7 | 140.1 | 0.0439 | 4.251 |

### High-water test metrics

| horizon_minutes | model | n | mae | rmse | nse |
| --- | --- | --- | --- | --- | --- |
| 60 | persistence | 480 | 51.52 | 62.25 | 0.8826 |
| 60 | seasonal_24h | 480 | 225.7 | 275.2 | -1.296 |
| 60 | ridge_selected_stations | 480 | 42.08 | 51.22 | 0.9205 |
| 60 | hist_gradient_boosting_selected_stations | 480 | 74.04 | 91.71 | 0.7451 |
| 60 | ridge_all_rain_stations | 480 | 51.03 | 64.89 | 0.8724 |
| 60 | hist_gradient_boosting_all_rain_stations | 480 | 77.58 | 102.8 | 0.6798 |
| 180 | persistence | 480 | 145.3 | 164.5 | 0.4708 |
| 180 | seasonal_24h | 480 | 264.4 | 321.1 | -1.018 |
| 180 | ridge_selected_stations | 480 | 134.4 | 151.9 | 0.5483 |
| 180 | hist_gradient_boosting_selected_stations | 480 | 274.3 | 369 | -1.664 |
| 180 | ridge_all_rain_stations | 480 | 143.1 | 174.9 | 0.4015 |
| 180 | hist_gradient_boosting_all_rain_stations | 480 | 272.7 | 358.8 | -1.519 |
| 360 | persistence | 480 | 241.8 | 270.9 | 0.02053 |
| 360 | seasonal_24h | 480 | 333 | 399.7 | -1.133 |
| 360 | ridge_selected_stations | 480 | 226.5 | 267 | 0.04845 |
| 360 | hist_gradient_boosting_selected_stations | 480 | 618.1 | 690.6 | -5.367 |
| 360 | ridge_all_rain_stations | 480 | 247.6 | 303.1 | -0.2265 |
| 360 | hist_gradient_boosting_all_rain_stations | 480 | 590.2 | 667.2 | -4.943 |

Ridge với các trạm được chọn là model tốt nhất ở cả ba horizon trong test hiện tại; HistGradientBoosting kém hơn rõ rệt, đặc biệt ở high-water periods. Mức cải thiện MAE tốt nhất so với persistence giảm từ 20.0% (+1h) xuống 4.3% (+6h), cho thấy lợi ích ML còn hạn chế và không ổn định theo horizon.

### Ablation: trạm được chọn so với toàn bộ trạm mưa

| horizon_minutes | model | mae | rmse | nse |
| --- | --- | --- | --- | --- |
| 60 | ridge_selected_stations | 21.03 | 28.99 | 0.9589 |
| 60 | ridge_all_rain_stations | 26.63 | 35.83 | 0.9373 |
| 180 | ridge_selected_stations | 60.28 | 80.44 | 0.6841 |
| 180 | ridge_all_rain_stations | 77.83 | 100.7 | 0.5051 |
| 360 | ridge_selected_stations | 96.39 | 128.7 | 0.1928 |
| 360 | ridge_all_rain_stations | 122.2 | 158.8 | -0.2291 |

Đưa đồng loạt tất cả trạm mưa vào Ridge làm test MAE xấu hơn ở cả ba horizon. Điều này không chứng minh các trạm khác vô ích; nó cho thấy blanket merge theo timestamp tạo thêm nhiễu/shift khi chưa có catchment mapping, feature selection và validation nhiều năm.

### Temporal split event support

| horizon_minutes | split | start | end | rows | high_water_events_using_train_p90 |
| --- | --- | --- | --- | --- | --- |
| 60 | train | 2025-11-14 18:50:00+00:00 | 2026-06-26 08:20:00+00:00 | 32194 | 71 |
| 60 | validation | 2026-06-26 08:30:00+00:00 | 2026-08-13 06:10:00+00:00 | 6899 | 7 |
| 60 | test | 2026-08-13 06:20:00+00:00 | 2026-09-30 04:00:00+00:00 | 6899 | 13 |
| 180 | train | 2025-11-14 18:50:00+00:00 | 2026-06-26 08:20:00+00:00 | 32194 | 71 |
| 180 | validation | 2026-06-26 08:30:00+00:00 | 2026-08-13 06:10:00+00:00 | 6899 | 7 |
| 180 | test | 2026-08-13 06:20:00+00:00 | 2026-09-30 04:00:00+00:00 | 6899 | 13 |
| 360 | train | 2025-11-14 18:50:00+00:00 | 2026-06-26 08:20:00+00:00 | 32194 | 71 |
| 360 | validation | 2026-06-26 08:30:00+00:00 | 2026-08-13 06:10:00+00:00 | 6899 | 7 |
| 360 | test | 2026-08-13 06:20:00+00:00 | 2026-09-30 04:00:00+00:00 | 6899 | 13 |

Split 70/15/15 chỉ là boundary minh bạch để benchmark, không được coi là tối ưu. Training rows ở sát biên được purge theo từng forecast horizon để nhãn không vượt sang validation; scaler/imputer chỉ fit trên phần train còn lại. Nếu validation/test thiếu event đáng kể, metric extreme không đủ tin cậy và cần event-aware rolling-origin evaluation khi có thêm năm dữ liệu.

## 10. OpenHydroNet compatibility

| OpenHydroNet requirement | Huế hiện có? | Source | Quality | Missing / Problem |
| --- | --- | --- | --- | --- |
| Historical meteorological forcing | Partial | 56 rain observation IDs and 4 wind stations | ~10-11 months; catchment relevance unverified; wind histories uneven | Long multi-year record and basin-verified forcings |
| Forecast meteorological forcing | No | None | Absent | Operational precipitation and other weather forecasts |
| Catchment/static attributes | No | Station points only | No basin descriptors | Area, elevation, soil, geology, land cover and compatible attributes |
| Catchment boundaries | No | None | Absent | Dã Viên contributing basin boundary and verified routing |
| Streamflow/discharge target | No | Water-level depth only | Different physical target; units unconfirmed | Discharge series or validated rating curve |
| Precipitation | Partial | 2.60M observations across 56 IDs | Units undocumented; one ID lacks metadata; catchment links UNKNOWN | Gauge/radar/reanalysis mapped to the verified contributing catchment |
| Temperature | No | None | Absent | Historical plus forecast temperature if matching pretrained inputs |
| Radiation/vapor pressure/other forcing | No | None | Absent | Exact variables/providers expected by selected pretrained run |
| Time resolution | Mismatch | 10-minute table | OpenHydroNet reference pipeline is not plug-compatible | Custom dataset/resampling and scientifically justified resolution |
| Normalizer/scaler compatibility | No | Local raw units | Feature names/units differ from pretrained scaler | Matching features, units, transformations and pretrained scaler |

OpenHydroNet hiện hỗ trợ `multimet`, yêu cầu hindcast/forecast inputs, static attributes và target variables; reference experiments dùng nhiều forcing meteorological, static catchment attributes và streamflow. Pretrained releases kèm scaler theo đúng feature/target của training run. Thay output head sang water level là **architecture adaptation/transfer learning experiment**, không phải fine-tune trực tiếp đúng pipeline; representation có thể còn hữu ích ở mức chưa biết, nhưng input semantics, scaler và target đã mismatch. Rating curve Q↔stage (và datum/backwater validity) là cần thiết nếu mục tiêu là discharge/compatibility vật lý. Nguồn: [repository](https://github.com/google-research/flood-forecasting), [configuration](https://openhydronet.readthedocs.io/en/latest/usage/config.html), [pretrained notes](https://github.com/google-research/flood-forecasting/blob/main/pretrained-models/Pretrained-Models-README.md), [HESS paper](https://hess.copernicus.org/articles/29/6221/2025/).

## 11. Missing data

| Objective | Priority | Data needed |
| --- | --- | --- |
| Baseline water-level | MUST HAVE | Confirmed units, datum, sensor QA/QC, and official station metadata |
| Baseline water-level | SHOULD HAVE | Several years of co-located/upstream rain and water-level history |
| Baseline water-level | SHOULD HAVE | Tide level and upstream gauges; Dã Viên may have downstream/backwater influences |
| OpenHydroNet fine-tuning | MUST HAVE | Discharge target or validated time-varying rating curve from stage to discharge |
| OpenHydroNet fine-tuning | MUST HAVE | Catchment boundary and pretrained-compatible static attributes |
| OpenHydroNet fine-tuning | MUST HAVE | Historical and forecast meteorological variables matching the pretrained config/scaler |
| OpenHydroNet fine-tuning | SHOULD HAVE | Long local record with multiple independent floods and regulated-flow metadata |
| Operational warning | MUST HAVE | Official alert/flood thresholds and reference datum |
| Operational warning | MUST HAVE | Real-time robust feeds, forecast rainfall, latency and outage monitoring |
| Operational warning | MUST HAVE | Reservoir releases/operations and verified upstream-downstream/catchment relationships |
| Operational warning | SHOULD HAVE | DEM, river network, land cover, soils and rainfall radar |
| Operational warning | OPTIONAL | Additional wind observations after demonstrating incremental skill |

## 12. Recommended next step

1. Xác nhận units/datum/QA, ID mưa thiếu metadata, record reservoir future-dated và ngưỡng báo động chính thức.
2. Giữ persistence làm chuẩn; Ridge với các trạm được chọn hiện thắng test. Không dùng blanket all-rain merge; trước hết xác định catchment/upstream rồi chọn hoặc aggregate mưa theo không gian.
3. Chạy ablation water-only vs catchment-verified rain vs wind/tide/upstream; dùng rolling-origin/event-based evaluation.
4. Thu thập nhiều năm và reservoir releases. Chỉ thử MEF-LSTM sau khi có signal ổn định; OpenHydroNet chỉ sau khi tạo dataset catchment/forcing/target tương thích.

## Trả lời trực tiếp

- **Có nên dùng Mean-Embedding-Forecast-LSTM ngay?** Không. Nhiều trạm hơn nhưng vẫn dưới một năm target và thiếu forecast forcing/static attributes.
- **Fine-tune pretrained OpenHydroNet trực tiếp?** Không. Target là water level thay vì discharge, feature/scaler/data adapter không tương thích, thiếu catchment statics và forecast meteorology.
- **Nếu vẫn muốn OpenHydroNet?** Bổ sung discharge hoặc rating curve hợp lệ, basin boundary/statics, forcing lịch sử+dự báo đúng schema/units và nhiều năm/event; sau đó đánh giá transfer trên thời gian post-2023/held-out hợp lệ.
- **Bài toán khả thi nhất?** Site-specific short-term water-level forecasting tại Dã Viên với autoregressive tabular baseline.
- **Target water level hay discharge?** Water level cho bài toán khả thi ngay; discharge cho pipeline OpenHydroNet/rainfall-runoff, chỉ khi có Q hoặc rating curve đáng tin cậy.
- **+1h/+3h/+6h?** Có thể benchmark cả ba; +1h có skill tốt nhất nhưng bị persistence chi phối, +6h chỉ cải thiện nhỏ.
- **Đủ high-water/extreme events?** Không đủ để khẳng định khả năng tổng quát hóa flood; chỉ có percentile events trong một phần năm và không có official threshold.
- **Baseline đầu tiên?** Persistence, sau đó Ridge với water-level lags và forcing đã xác minh; blanket all-rain model hiện kém hơn.
- **Experiment tiếp theo?** Xây catchment-aware rainfall aggregation/feature selection, rồi thêm tide/upstream/reservoir đã xác minh và dùng rolling-origin theo event.

## Giới hạn diễn giải

Bất kỳ upstream/downstream relation, physical units, official flood status, catchment membership hoặc rating-curve validity nào không có trong file đều được ghi là: **Không thể xác định từ dữ liệu hiện có**.
