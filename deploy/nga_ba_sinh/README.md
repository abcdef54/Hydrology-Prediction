# Model dự báo mực nước trạm Ngã Ba Sình 

Trạm **0074647705**, Ngã Ba Sình (sông Hương, nơi hợp lưu với sông Bồ), tọa độ 16,528167 N; 107,574000 E. Đơn vị mực nước: mm.

Model dự báo **mực nước tại t + 3 h, t + 6 h và t + 12 h** từ dữ liệu có tại giờ phát hành t. Có thể phát hành mỗi giờ.

## 1. Model là gì

`combined_v1` gồm **2 mạng GRU**. Mỗi mốc lấy kết quả từ đúng một mạng, không lấy trung bình:

| Mốc | Lấy từ | Lý do |
|---|---|---|
| +3 h, +6 h | `short_3h_6h` | mạng nhỏ, cửa sổ ngắn, tốt nhất cho mốc gần |
| +12 h | `long_12h` | mạng lớn hơn, cửa sổ dài, loss nhấn mạnh mốc +12 h |

Cả hai mạng được train để dự báo đủ 3 mốc; khi chạy thật chỉ dùng các mốc ghi trong bảng.

### Cách hoạt động
1. Với mỗi giờ trong cửa sổ L giờ gần nhất (t − L + 1 … t), dựng một vector đầu vào: mực nước, triều, gió, mưa dự báo, mực biển dự báo (mục 4). Mỗi kênh được chuẩn hóa bằng trung bình và độ lệch chuẩn tính trên tập train.
2. GRU 3 lớp đọc chuỗi L vector. Trạng thái ẩn ở bước cuối được ghép với embedding của trạm (8 chiều), qua một tầng `Linear + ReLU`, rồi tới tầng ra 3 giá trị.
3. **Dự báo dạng residual:** dự báo = persistence ("mực nước giữ nguyên như lúc t") + phần hiệu chỉnh do mạng học. Lúc chưa train, model đúng bằng persistence; mạng chỉ học phần thay đổi.
4. Kết quả được đổi ngược về mm bằng trung bình và độ lệch chuẩn của nhãn trên tập train.

### Cấu hình train

| | `short_3h_6h` | `long_12h` |
|---|---|---|
| Dataset | `nbs_onshore_tide7_fc_sea7_v3` | `nbs_onshore_tide7_fc_sea_v3` |
| Số đầu vào mỗi giờ | 39 | 35 |
| Cửa sổ (seq len) | 12 h | 25 h |
| GRU | hidden 64 × 3 lớp | hidden 128 × 3 lớp |
| Số tham số | 74.955 | 279.435 |
| Dropout (GRU / đầu vào / head) | 0,2 / 0,1 / 0,1 | 0,2 / 0,1 / 0,1 |
| Optimizer | Adam, lr 1e-4, weight decay 1e-3, batch 64 | như trên |
| Lịch lr | step | step |
| Số epoch train / epoch dùng | 100 / **59** | 130 / **110** |
| Trọng số loss các mốc +3 / +6 / +12 | 1 / 1 / 1 | 1 / 1 / 2,5 |
| Seed | 23 | 23 |

- **Loss `combined`:** với sai số e và mức thay đổi thật c (so với persistence), d = max(độ lệch chuẩn của mức thay đổi, 200 mm):
  - phần cơ bản (e/d)²;
  - phạt thêm 0,3 · (max(−e, 0)/d)² khi c > 200 mm, tức phạt dự báo thiếu lúc nước lên mạnh.
- **Hai mạng dùng cùng nguồn dữ liệu,** chỉ khác nhóm mực biển (7 kênh ở short, 3 kênh ở long) và cửa sổ.

### Giai đoạn dữ liệu
- Số đo của trạm có từ 14/11/2025.
- **Train:** giờ phát hành tới 29/07/2026 (6.073 mẫu). Bộ chuẩn hóa chỉ tính trên phần này.
- **Validation:** 31/07/2026 00:00 → 13/09/2026 23:00 giờ VN (1.080 mẫu). Bỏ 24 h ở ranh giới giữa các tập.

## 2. Kết quả tháng 9–10/2026

01/09/2026 00:00 → 31/10/2026 23:00 giờ VN, **mọi giờ** (1.451 giờ phát hành; đã bỏ các giờ vắt qua lần đổi mốc đo ngày 10/10, mục 4.1):

| Mốc | **MAE** | MAE của persistence |
|---|---|------|
| +3 h | **38,0 mm** | 90,2 mm |
| +6 h | **55,7 mm** |  130,6 mm |
| +12 h | **52,0 mm** |  71,5 mm |

Các số này được tính lại bằng chính `predict.py` trong folder này, mực nước đọc từ PostgreSQL.

## 3. Đầu vào và đầu ra

### 3.1 Đầu ra
Mực nước dự báo (mm) tại trạm Ngã Ba Sình ở **t + 3 h, t + 6 h và t + 12 h**, với t là giờ phát hành (giờ tròn). Mực nước ở đây là **mực nước trung bình của một giờ**: ví dụ "mực nước lúc 09:00" là trung bình các số đo từ 08:10 đến 09:00, cùng cách tính với đầu vào.

`predict.py` trả thêm mực nước lúc t, dự báo riêng của từng mạng và độ phủ dữ liệu đầu vào.

### 3.2 Đầu vào được đưa vào model như thế nào
- Mỗi lần dự báo, model nhận **một bảng L hàng × F cột**:
  - **mỗi hàng là một giờ** trong cửa sổ L giờ gần nhất, từ t − L + 1 tới t;
  - **mỗi cột là một đầu vào** (gọi là kênh).
  - `short_3h_6h`: L = 12 giờ, F = 39 kênh. `long_12h`: L = 25 giờ, F = 35 kênh.
- Tại mỗi hàng (giờ τ), các kênh chỉ dùng thông tin **đã có tại thời điểm τ**. Kể cả các kênh "dự báo cho vài giờ tới" (mưa, gió, mực biển) cũng là những bản dự báo đã phát hành trước τ (mục 4.3).
- **Chuẩn hóa:** trước khi đưa vào mạng, mỗi kênh được trừ trung bình rồi chia độ lệch chuẩn. Hai giá trị này tính trên tập train và lưu trong bundle (`normalizers`). Riêng các kênh cờ (tên chứa `missing`) giữ nguyên 0/1.
- **Khi thiếu dữ liệu:** kênh đó nhận giá trị 0 sau chuẩn hóa (tức bằng trung bình train), và kênh cờ tương ứng bằng 1 để model biết giá trị đó không có thật.

### 3.3 Danh sách đầy đủ các đầu vào

Thứ tự trong bảng là thứ tự cột trong bundle (`feature_columns`). Cột S / L: kênh có trong `short_3h_6h` / `long_12h`.

| # | Kênh | Ý nghĩa | Cách tính, đơn vị | S | L |
|---|---|---|---|---|---|
| 1 | `water_level_depth` | **Mực nước trung bình mỗi giờ tại trạm dự báo (Ngã Ba Sình)** | Trung bình các số đo 10 phút trong (τ − 1 h, τ], sau khi lọc gai (mục 4.1); mm | ✓ | ✓ |
| 2 | `water_level_missing` | Cờ: giờ đó không có số đo mực nước | 1 = thiếu, 0 = có | ✓ | ✓ |
| 3 | `flood_season` | **Cờ đánh dấu các tháng mùa mưa lũ ở Huế** | 1 nếu tháng 9, 10, 11 hoặc 12 (giờ VN), ngược lại 0 | ✓ | ✓ |
| 4–7 | `fc_a_luoi_rain_0_3h`, `_3_6h`, `_6_12h`, `_12_24h` | **Mưa dự báo ở vùng A Lưới** (thượng nguồn phía tây) cho các khoảng 0–3, 3–6, 6–12, 12–24 giờ tới | Tổng mưa ECMWF dự báo trong khoảng đó, ô lưới 16,27 N 107,26 E; log(1 + mm) | ✓ | ✓ |
| 8–11 | `fc_thuan_an_rain_0_3h` … `_12_24h` | **Mưa dự báo ở vùng Thuận An – Huế** (đồng bằng, gần cửa biển) | Như trên, ô 16,56 N 107,59 E | ✓ | ✓ |
| 12–15 | `fc_bach_ma_rain_0_3h` … `_12_24h` | **Mưa dự báo ở vùng Bạch Mã** (núi phía nam, thượng nguồn sông Hương) | Như trên, ô 16,27 N 107,86 E | ✓ | ✓ |
| 16–19 | `fc_basin_rain_0_3h` … `_12_24h` | **Mưa dự báo trung bình cả lưu vực** | Trung bình 7 ô lưới (mục 4.3), rồi log(1 + mm) | ✓ | ✓ |
| 20 | `fc_age_hours` | Bản dự báo mưa đang dùng đã "cũ" bao nhiêu giờ | Giờ τ trừ giờ khởi tạo lượt chạy ECMWF; từ 6 đến 24 h | ✓ | ✓ |
| 21 | `fc_missing_frac` | Cờ: không có bản dự báo mưa hợp lệ | 1 = thiếu | ✓ | ✓ |
| 22 | `wind_onshore_past24` | **Gió thổi từ biển vào bờ, trung bình 24 giờ qua** | Thành phần gió theo hướng đông bắc → tây nam (vuông góc bờ biển Thuận An), trung bình τ − 23 … τ; m/s, dương = thổi vào bờ | ✓ | ✓ |
| 23 | `wind_onshore_next6` | Gió thổi vào bờ, trung bình 6 giờ tới | Như trên, trung bình τ + 1 … τ + 6 (bản dự báo từ 24 h trước) | ✓ | ✓ |
| 24 | `wind_onshore_next12` | Gió thổi vào bờ, trung bình 12 giờ tới | Trung bình τ + 1 … τ + 12 | ✓ | ✓ |
| 25 | `wind_onshore_chg12` | Gió vào bờ sắp mạnh lên hay yếu đi | `next12` − `past24`; m/s | ✓ | ✓ |
| 26 | `wind_missing_frac` | Cờ: thiếu dữ liệu gió | 1 = thiếu | ✓ | ✓ |
| 27 | `tide_now` | **Mực nước triều thiên văn tại giờ τ** (mục 3.4) | Phần điều hòa của phép fit triều 7 ngày; mm, dao động quanh 0 | ✓ | ✓ |
| 28 | `tide_d3` | **Triều sẽ lên / xuống bao nhiêu trong 3 giờ tới** | Triều(τ + 3) − Triều(τ); mm | ✓ | ✓ |
| 29 | `tide_d6` | Triều sẽ lên / xuống bao nhiêu trong 6 giờ tới | Triều(τ + 6) − Triều(τ); mm | ✓ | ✓ |
| 30 | `tide_d12` | Triều sẽ lên / xuống bao nhiêu trong 12 giờ tới | Triều(τ + 12) − Triều(τ); mm | ✓ | ✓ |
| 31 | `tide_missing_frac` | Cờ: không đủ dữ liệu để fit triều tại giờ này | 1 = dùng fit dự phòng (mục 3.4) | ✓ | ✓ |
| 32 | `sea_setup_chg` | **Mực nước biển ngoài khơi sắp dâng hay rút** (bỏ qua dao động triều) | Trung bình 12 giờ tới trừ trung bình 24 giờ qua của mực biển dự báo; mm | ✓ | ✓ |
| 33 | `sea_anom` | **Biển hiện cao hơn hay thấp hơn bình thường** | Mực biển tại τ trừ trung bình 7 ngày qua; mm | ✓ | ✓ |
| 34 | `sea_next12_anom` | Biển 12 giờ tới cao hơn hay thấp hơn bình thường | Trung bình 12 giờ tới trừ trung bình 7 ngày qua; mm | ✓ | ✓ |
| 35 | `sea_p3` | Mực biển ngoài khơi thay đổi bao nhiêu trong 3 giờ tới | Mực biển(τ + 3) − mực biển(τ); mm | ✓ | |
| 36 | `sea_p6` | … trong 6 giờ tới | Mực biển(τ + 6) − mực biển(τ); mm | ✓ | |
| 37 | `sea_p12` | … trong 12 giờ tới | Mực biển(τ + 12) − mực biển(τ); mm | ✓ | |
| 38 | `sea_now` | Mực biển ngoài khơi tại giờ τ | Giá trị dự báo cho τ (so với mực nước biển trung bình); mm | ✓ | |
| 39 | `sea_missing_frac` | Cờ: thiếu dữ liệu mực biển | 1 = thiếu | ✓ | ✓ |

`long_12h` có 35 kênh: giống `short_3h_6h` nhưng không có 4 kênh `sea_p3`, `sea_p6`, `sea_p12`, `sea_now`.

### 3.4 Triều: là gì, vì sao cần, tính thế nào

**Triều là gì.** Mực nước biển lên xuống đều đặn do lực hút của Mặt Trăng và Mặt Trời, gọi là thủy triều. Con triều đi từ biển qua cửa Thuận An vào phá Tam Giang, rồi ngược lên sông Hương: tới Ngã Ba Sình chậm khoảng 40 phút so với Thuận An, tới Dã Viên chậm thêm khoảng 40 phút. Vì vậy, ngoài mùa lũ, phần lớn dao động hằng ngày của mực nước ở Ngã Ba Sình là do triều.

Triều ở đây là **triều hỗn hợp**: phần lớn các ngày có hai lần nước lên và hai lần nước xuống, với hai đỉnh cao thấp khác nhau. Sự lên xuống này là tổng của nhiều "thành phần triều", mỗi thành phần là một dao động hình sin với chu kỳ cố định do thiên văn quyết định:

| Thành phần | Chu kỳ | Nguồn gốc |
|---|---|---|
| M2 | 12,42 h | Mặt Trăng, lên xuống 2 lần mỗi ngày (thành phần chính) |
| S2 | 12,00 h | Mặt Trời, 2 lần mỗi ngày |
| N2 | 12,66 h | Mặt Trăng, do quỹ đạo elip |
| K1 | 23,93 h | Mặt Trăng và Mặt Trời, 1 lần mỗi ngày |
| O1 | 25,82 h | Mặt Trăng, 1 lần mỗi ngày |
| M4, MS4 | 6,21 h, 6,10 h | Méo dạng con triều khi đi vào vùng nước nông, sông, phá |

M2 và S2 cộng hưởng rồi triệt tiêu nhau theo chu kỳ khoảng 15 ngày, tạo ra **nước cường** (triều mạnh) và **nước kém** (triều yếu). Hai thành phần 1 lần/ngày (K1, O1) làm hai đỉnh trong ngày cao thấp khác nhau.

**Vì sao cần.** Chu kỳ của các thành phần triều là cố định, nên khi đã biết biên độ và pha của chúng thì **dự đoán trước được triều cho nhiều giờ tới**. Nhờ vậy model biết sẵn "theo triều, 6 giờ tới nước sẽ lên hay xuống bao nhiêu", và chỉ phải học phần còn lại (mưa, lũ, gió, nước dâng). Ví dụ: `tide_d6 = −150` nghĩa là chỉ riêng triều sẽ làm nước xuống khoảng 150 mm trong 6 giờ tới.

**Cách tính** (hàm `tide_features` trong `predict.py`). Tại mỗi giờ τ:
1. Lấy mực nước giờ đo được của **chính trạm Ngã Ba Sình** trong **7 ngày ngay trước τ** (169 giờ, tính cả τ). Cần ít nhất 70% số giờ có dữ liệu.
2. Fit bình phương tối thiểu:
   mực nước(s) = a + b·(s − τ) + Σₖ [cₖ·cos(2πs/Pₖ) + dₖ·sin(2πs/Pₖ)],
   với Pₖ là chu kỳ của 7 thành phần ở bảng trên. a là mực nước nền, b là xu hướng tăng/giảm chậm (ví dụ do lũ). Hai hệ số này giúp phép fit không nhầm lũ thành triều.
3. **Triều(s)** = chỉ phần tổng các hình sin (bỏ a và b). Vì chu kỳ đã biết, công thức này tính được cho cả các giờ tương lai.
4. Kết quả:
   - `tide_now` = Triều(τ);
   - `tide_d3` = Triều(τ + 3) − Triều(τ);
   - `tide_d6`, `tide_d12` tương tự.
5. Nếu 7 ngày trước τ thiếu quá nhiều dữ liệu, dùng một phép fit dự phòng trên toàn bộ chuỗi (đã bỏ phần biến đổi chậm), và đặt `tide_missing_frac = 1`. Khi chạy thật với đủ 32 ngày lịch sử thì hầu như không xảy ra.

**Vì sao chỉ dùng 7 ngày.** Biên độ triều ở Ngã Ba Sình đổi rất mạnh theo mùa: biên độ M2 khoảng 20 mm vào tháng 3, chỉ 6 mm vào tháng 7, nhưng tới khoảng 100 mm vào tháng 10–11. Cửa sổ 7 ngày theo kịp các thay đổi này nhanh. Đã thử cửa sổ 15 và 30 ngày (tách các thành phần tốt hơn về lý thuyết) nhưng kết quả không tốt hơn.

Triều không cần dữ liệu ngoài: chỉ tính từ mực nước đã đo của trạm.

### 3.5 Vì sao chọn các đầu vào còn lại
- **Mưa dự báo ECMWF:**
  - Mưa lớn ở thượng nguồn (A Lưới, Bạch Mã) sau vài giờ đến nửa ngày sẽ đổ về sông Hương và sông Bồ, làm nước ở Ngã Ba Sình lên. Mưa ở đồng bằng (ô Thuận An) làm nước lên nhanh hơn.
  - Model dùng mưa **dự báo** cho 24 giờ tới, của mô hình thời tiết ECMWF (châu Âu), vì mực nước 6–12 giờ tới phụ thuộc vào mưa chưa rơi.
  - `log(1 + mm)` giúp những trận mưa rất lớn không lấn át mọi thứ khác.
- **Gió thổi vào bờ:** gió mạnh từ biển (hướng đông bắc, thường gặp mùa đông và khi có bão) đẩy nước biển dồn vào cửa Thuận An và phá Tam Giang, làm nước sông dâng, đồng thời cản nước lũ thoát ra biển. Dùng bản dự báo của ECMWF cho điểm ngoài khơi 16,85 N 107,85 E.
- **Mực biển ngoài khơi:**
  - Mực nước sông gần cửa biển phụ thuộc vào mực nước biển. Ngoài triều, biển còn dâng hoặc rút do bão, gió mùa và áp thấp (gọi là nước dâng).
  - Các kênh `sea_*` so sánh mực biển dự báo với mức trung bình của 24 giờ hoặc 7 ngày qua, nên chủ yếu phản ánh phần dâng/rút này chứ không phải dao động triều thông thường.
  - Nguồn là mô hình đại dương của Open-Meteo Marine, điểm ngoài khơi Thuận An.
- **Mùa lũ (`flood_season`):** tháng 9–12 là mùa mưa lũ ở Huế. Trong mùa này, nước lên xuống mạnh và phản ứng với mưa khác hẳn mùa khô, nên cờ này giúp model phân biệt hai chế độ.

## 4. Nguồn dữ liệu

### 4.1 Mực nước trạm
- Đây là số đo 10 phút thu từ API quan trắc DCU Huế.
- **Mực nước giờ t** = trung bình các số đo trong (t − 1 h, t]. Trước đó **lọc gai**: bỏ số đo lệch quá 150 mm so với trung vị các số đo trong cửa sổ 121 phút lấy đối xứng quanh nó; giữ nguyên 11/04/2026 00:00–01:30 UTC.
- **Trạm đổi mốc đo lúc 2026-10-10 15:20 UTC** (mực nước tụt 157 mm trong một bước 10 phút). Model dùng mực nước thô; kết quả mục 2 đã gồm giai đoạn sau 10/10. Các giờ có mực nước hiện tại và mực nước dự báo nằm ở hai phía của lần đổi mốc bị bỏ khi chấm.
- **Cần ít nhất 32 ngày lịch sử** trước giờ phát hành, vì triều fit trên 7 ngày và cửa sổ dài 25 h, cộng thêm phần dự phòng.

### 4.2 Triều
Không cần nguồn ngoài. Mỗi giờ t, fit điều hòa trên 7 ngày mực nước của chính trạm ngay trước t, gồm 7 thành phần M2, S2, N2, K1, O1, M4, MS4, kèm hằng số và xu hướng. Kênh triều chỉ lấy phần điều hòa. Giải thích chi tiết ở mục 3.4.

### 4.3 Open-Meteo (API công khai, không cần key)

| File trong `data/open_meteo/` | API | Điểm | Biến dùng | Có dữ liệu |
|---|---|---|---|---|
| `wind_previous_runs.json` | Previous Runs API `previous-runs-api.open-meteo.com/v1/forecast`, `models=ecmwf_ifs025` | 16,85 N 107,85 E (ô 16,75 / 107,75) | `wind_speed_10m_previous_day1`, `wind_direction_10m_previous_day1`, `pressure_msl_previous_day1` | 01/11/2025 → 16/11/2026 |
| `ecmwf_single_runs/` (1.473 file, mỗi file một lượt chạy) | Single Runs API `single-runs-api.open-meteo.com/v1/forecast`, `models=ecmwf_ifs`, `cell_selection=land`, `forecast_days=3` | 7 điểm lưu vực (trong bundle: `feature_config['fc_ecmwf']['request_points']`) | `precipitation` theo giờ | lượt chạy 13/11/2025 00Z → 16/11/2026 00Z |
| `marine_sea_level.json` | Marine API `marine-api.open-meteo.com/v1/marine` | 16,7 N 107,75 E (ô 16,708 / 107,708) | `sea_level_height_msl_previous_day1` | 01/11/2025 → 16/11/2026 |

**Quy tắc thời gian (giữ đúng như lúc train):**
- **Gió và mực biển chỉ dùng bản `*_previous_day1`:** giá trị cho giờ H là bản dự báo phát hành khoảng 24 h trước H. Model dùng các giờ tới t + 12, nên mọi giá trị đã có ít nhất 12 h trước giờ phát hành.
- **Mưa ECMWF:** tại t, dùng lượt chạy mới nhất khởi tạo trong khoảng [t − 24 h, t − 6 h] (giả định cần 6 h để lượt chạy được phát hành). Lượt đó phải có đủ mưa cho mọi ô ở các giờ t + 1 … t + 24. Không trộn hai lượt chạy.

Dataset train được dựng từ các file cùng nguồn, tải trước đó. Trên giai đoạn train, giá trị giống hệt các file trong folder này (đã kiểm tra).

## 5. Các file trong folder

```
nga_ba_sinh/
├── README.md
├── predict.py                       # chạy dự báo (tự chứa)
├── worker.py                        # gọi Open-Meteo, bổ sung dữ liệu cho các lần dự báo sau
├── model/
│   └── nga_ba_sinh_combined_v1.joblib
└── data/
    ├── open_meteo/                  # dữ liệu Open-Meteo hiện có (mục 4.3)
    │   ├── wind_previous_runs.json
    │   ├── marine_sea_level.json
    │   └── ecmwf_single_runs/YYYYmmddTHHMM.json
    └── train_dataset/
        ├── nbs_onshore_tide7_fc_sea7_v3/    # dataset train của short_3h_6h
        └── nbs_onshore_tide7_fc_sea_v3/     # dataset train của long_12h
```

### `model/nga_ba_sinh_combined_v1.joblib`
`joblib.load()` trả về một dict, chỉ gồm kiểu Python cơ bản và mảng numpy (sha256 `1016d0178c8ed9dafc94ecde353eb24eab4346e748776fe13cd75af57284b9e8`):

| Khóa | Nội dung |
|---|---|
| `format`, `name`, `station_id`, `station_name` | Định danh (`hydrology_station_bundle_v2`) |
| `combination` | `{'rule': 'per_horizon', 'horizons': {3: ['short_3h_6h'], 6: ['short_3h_6h'], 12: ['long_12h']}}` |
| `sources` | Tên file và thư mục Open-Meteo trong `data/open_meteo/` |
| `members[tên].architecture` | `input_size`, `hidden_size`, `num_layers`, `output_size`, `num_stations`, `station_embedding_size` |
| `members[tên].state_dict` | Trọng số (mảng numpy), gồm cả hệ số persistence |
| `members[tên].feature_columns`, `sequence_length`, `horizons` | Thứ tự kênh đầu vào, độ dài cửa sổ, các mốc |
| `members[tên].normalizers` | `water` (mực nước đầu vào), `targets` (nhãn mỗi mốc), `shared` (từng kênh) |
| `members[tên].feature_config` | Tham số dựng đầu vào: lọc gai, triều, gió, ô ECMWF, mực biển |
| `members[tên].train_anchor_first/last`, `training_dataset`, `model_pt_sha256` | Nguồn gốc |

### `data/train_dataset/<dataset>/`

| File | Nội dung |
|---|---|
| `water.parquet` | Mực nước giờ của trạm (mm, đã lọc gai), 14/11/2025 → 15/10/2026 |
| `rainfall.parquet` | Các nhóm đầu vào ngoài mực nước, theo giờ, chưa chuẩn hóa. Tên file giữ theo định dạng cũ: nhóm đầu tiên từng là mưa |
| `calendar.parquet` | `flood_season` và các kênh giờ / ngày (không dùng) |
| `weather.parquet` | Rỗng (không dùng) |
| `samples.parquet` | Mỗi dòng một giờ phát hành: mực nước lúc t, nhãn +3 / +6 / +12 h, cột `split` (`train`, `validation`, `test_exploratory`, `purged`, `excluded_sensor_step`) |
| `normalizers.json` | Trung bình và độ lệch chuẩn tính trên train, giống `normalizers` trong bundle |
| `feature_schema.json`, `station_mapping.json`, `config.json`, `manifest.json` | Danh sách kênh theo nhóm, mã trạm, cách chia tập, cấu hình dựng (`manifest.json → config`) |
| `despiked_readings.csv` | Các số đo 10 phút đã bị lọc gai |

## 6. Cách chạy

Cần Python ≥ 3.10 với `numpy`, `pandas`, `torch` (CPU là đủ), `joblib`. Muốn đọc mực nước từ DB thì cần thêm `sqlalchemy` và driver PostgreSQL. Biến môi trường `DATABASE_URL` trỏ tới DB có bảng `hydrology.observations`.

```bash
cd nga_ba_sinh
python worker.py                                   # cập nhật dữ liệu Open-Meteo (nên chạy mỗi giờ)
python predict.py                                  # dự báo cho giờ tròn gần nhất (UTC)
python predict.py --time 2026-11-10T03:00Z         # một giờ phát hành
python predict.py --start 2026-09-01T00:00+07:00 --end 2026-10-31T23:00+07:00 --out forecasts.csv
python predict.py --time 2026-11-10T03:00Z --level-file readings.csv    # mực nước từ file số đo 10 phút
```

Gọi từ Python:

```python
import pandas as pd
from predict import Forecaster

f = Forecaster('model/nga_ba_sinh_combined_v1.joblib', data_dir='data/open_meteo')   # nạp một lần
table = f.forecast(pd.date_range('2026-11-10 00:00', '2026-11-10 06:00', freq='h', tz='UTC'))
```

- **Kết quả (một dòng cho mỗi giờ phát hành và mốc):** `issue_time_utc`, `horizon_h`, `valid_time_utc`, `forecast_mm`, `level_now_mm`, `member_short_3h_6h_mm`, `member_long_12h_mm`, `input_coverage`, `reliable`. `reliable` = False khi dưới 80% số giờ trong cửa sổ có số đo.
- **Thời gian chạy (CPU):** khoảng 3 giây cho một giờ phát hành (gồm cả nạp thư viện), khoảng 14 giây cho mọi giờ của 2 tháng. `predict.py` chỉ đọc những lượt chạy ECMWF cần dùng, và chỉ fit triều cho các giờ nằm trong cửa sổ đầu vào.
- **Lập lịch** (ví dụ crontab): chạy `worker.py` vào phút 15 mỗi giờ, rồi chạy `predict.py`.
  - Với một trạm mới, `worker.py` tự tải 40 ngày gần nhất.
  - Lượt chạy ECMWF chưa có trên API sẽ được thử lại ở lần chạy sau.
  - Gói miễn phí của Open-Meteo cho khoảng 10.000 lượt gọi mỗi ngày; mỗi điểm trong một request tính là một lượt.

## 7. Lưu ý
- **Phải giữ đúng quy tắc thời gian ở mục 4.3.** Dùng dự báo mới hơn (ví dụ gió `previous_day0`), hoặc lượt ECMWF chưa đủ 6 h, sẽ làm đầu vào khác với lúc train.
- Khi thiếu Open-Meteo, các kênh tương ứng nhận giá trị 0 và cờ thiếu bằng 1, model vẫn chạy. Chất lượng khi thiếu kéo dài chưa được đánh giá.
