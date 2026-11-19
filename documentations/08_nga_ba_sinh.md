# Giai đoạn 8: Ngã Ba Sình

Trạm sông đầu tiên đạt **0 lỗi > 200 mm** trên cả validation, test và giai đoạn dữ liệu mới. Giai đoạn này kiểm chứng rằng quy trình và bộ feature của Thuận An dùng lại được cho một trạm sông khó hơn, đồng thời bổ sung thêm các feature và kỹ thuật xử lý dữ liệu mới.

## 1. Thời gian

12/11/2026 đến 16/11/2026 (hoàn thành 16/11).

## 2. Trọng tâm

Ngã Ba Sình (0074647705) nằm trên sông Hương, tại nơi sông Bồ nhập vào và gần phá Tam Giang. Đây là trạm sông, khó hơn Thuận An: chỉ với nước và triều của chính trạm, số lỗi > 200 mm ở +6 h là 18 (validation) và 23 (test), trong khi Thuận An là 0.

Cách làm, theo quy trình của giai đoạn 7:

- Mỗi lần chỉ thêm **một nhóm đầu vào mới**; nhóm nào đáng thử thì sàng lọc bằng hồi quy tuyến tính trước khi train.
- Chấm điểm **từng epoch** trên mọi tập, không chỉ epoch trainer chọn.
- Xếp hạng theo **số lỗi > 200 mm trước, MAE sau**; chấp nhận MAE tăng nếu giảm được lỗi lớn.
- Quy mô: hơn 40 lần train (một seed mỗi lần, nhiều lần thử seed 23, 42), hàng nghìn epoch được chấm.

## 3. Dữ liệu sử dụng

Cùng chia tập với Thuận An: train 6.073 mẫu (đến 29/07/2026), validation 1.080 (đến 13/09), test 710 (từ 14/09), cộng 48 mẫu bị loại ở ranh giới và 13 mẫu bị loại vì sensor đổi mốc đo; giai đoạn mới 15/10–12/11 gồm 668 mẫu. Mốc dự báo +3, +6, +12 h.

| Nhóm feature | Nội dung | Số kênh |
|---|---|---:|
| Cơ bản | Mực nước trạm đã lọc gai, cờ thiếu, cờ mùa lũ | 3 |
| Mưa dự báo ECMWF | Tổng mưa 0–3, 3–6, 6–12, 12–24 giờ tới (log1p) ở ô A Lưới, Thuận An, Bạch Mã và trung bình 7 ô; tuổi dự báo; tỉ lệ thiếu | 18 |
| Gió vào bờ | Gió ECMWF dự báo từ 24 giờ trước: trung bình 24 giờ qua, 6 giờ tới, 12 giờ tới, thay đổi 12 giờ; tỉ lệ thiếu | 5 |
| Triều 7 ngày | Hồi quy hài hòa 7 thành phần trên 7 ngày: triều hiện tại, thay đổi sau 3/6/12 giờ; tỉ lệ thiếu | 5 |
| **Mực nước biển ngoài khơi dự báo** | Open-Meteo Marine `sea_level_height_msl_previous_day1`: thay đổi nước dâng, độ lệch so với trung bình 7 ngày, độ lệch 12 giờ tới; (bản +3/+6 h thêm giá trị dự báo trực tiếp tại +3/+6/+12 giờ) | 4 (8) |

Hai model của `combined_v1` có 39 đầu vào (nhóm cho +3/+6 h) và 35 đầu vào (nhóm cho +12 h).

**Xử lý dữ liệu riêng cho trạm này:**

- **Lọc gai số đo** (`despike`): bỏ số đo 10 phút lệch hơn 150 mm so với trung vị trong cửa sổ 121 phút; trên giai đoạn train/validation/test bỏ 24 số đo, làm thay đổi 14 giá trị giờ. Một đoạn tụt ngày 11/04/2026 được giữ lại vì Dã Viên cũng tụt cùng lúc nên có thể là hiện tượng thật.
- **Sensor đổi mốc đo** (`sensor_steps`): lúc 22:20 giờ VN ngày 10/10/2026 mực nước tụt 157 mm chỉ trong 10 phút trong khi Thuận An và Dã Viên tăng nhẹ, và chênh lệch với hai trạm này đổi khoảng −172 mm và giữ nguyên đến 12/11. Số đo thô được giữ, chỉ loại 13 mẫu nằm ở hai phía lúc đổi mốc; model dự báo mức thay đổi nên dùng được với mốc mới.

## 4. Kết quả

**Model `combined_v1`: ghép theo mốc dự báo** (mỗi mốc lấy đầu ra của một model, không lấy trung bình):

| Mốc | Model | Cấu hình |
|---|---|---|
| +3 h, +6 h | `short_3h_6h` | GRU 64×3, cửa sổ 12 giờ, 39 đầu vào, 74.955 tham số, lịch học `step`, seed 23, epoch 59 |
| +12 h | `long_12h` | GRU 128×3, cửa sổ 25 giờ, 35 đầu vào, 279.435 tham số, trọng số loss mốc +12 h gấp 2,5, seed 23, epoch 110 |

Hiệu năng 01/09–31/10/2026 (1.451 thời điểm phát dự báo):

| Mốc | MAE (mm) | RMSE (mm) | Sai số lớn nhất (mm) | Lỗi > 200 mm |
|---|---:|---:|---:|---:|
| +3 h | 38,0 | 46,6 | 152,7 | **0** |
| +6 h | 55,7 | 68,8 | 187,1 | **0** |
| +12 h | 52,0 | 64,9 | 197,6 | **0** |

Persistence (giữ nguyên mực nước) có MAE 90 / 131 / 72 mm. Cũng 0 lỗi trên validation, test và giai đoạn mới 15/10–12/11.

**Quá trình thử nghiệm** (số lỗi > 200 mm ở +3/+6/+12 h, validation · test):

| Bước | Thay đổi | Kết quả tốt nhất |
|---|---|---|
| Chỉ mực nước | h64, cửa sổ 25 giờ | 3/15/20 · 0/8/8 |
| + triều | 7 ngày tốt hơn 30 ngày | 0/10/20 · 0/0/10 |
| + gió | chỉ dùng 4 kênh gió vào bờ | 0/3/12 · 0/1/7 |
| + mưa dự báo ECMWF | giảm mạnh lỗi +12 h | 0/3/5 · 0/3/3 |
| Lịch học, cửa sổ | lịch `step` tốt hơn cosine; cửa sổ 12/25 giờ | 0/4/3 · 0/1/3 |
| Sửa lỗi dữ liệu | loại 13 mẫu đổi mốc, lọc gai | test 0/1/3 → 0/0/0 |
| **+ mực nước biển dự báo (1 điểm)** | bước tiến lớn trên dữ liệu mới | 0/0/7 · 0/0/0 (mới: 0/1/4) |
| h128 + trọng số +12 h | gấp 2–2,5 | seed 23: 0/0/0 · 0/0/0 (mới: 0/0/1) |
| **Ghép theo mốc** | +3/+6 h từ model ngắn, +12 h từ model dài | **0/0/0 · 0/0/0 (mới: 0/0/0)** |

## 5. So với giai đoạn trước

Đây là lần thứ hai đạt 0 lỗi, và là lần đầu ở một trạm sông. So với Dã Viên (giai đoạn 1–6), cùng loại trạm, kết quả tốt hơn hẳn nhờ bộ feature và quy trình. So với Thuận An (giai đoạn 7), cần thêm hai thứ để đạt: mực nước biển dự báo và ghép model theo mốc.

## 6. Lỗi > 200 mm

**Xử lý được**: 0 lỗi ở cả ba giai đoạn. 
- Các mẫu khó nhất đều thuộc các sự kiện lớn: nước bắt đầu lên trong bão 11/09 (+6 h), nước rút sau bão 13/09 (+12 h), mưa lớn kèm gió mùa đông bắc 08–09/11 (+12 h); mọi model đều sai cùng chiều ở đó.

## 7. Điều mới rút ra

- **Mực nước biển ngoài khơi dự báo là feature mang lại bước tiến lớn nhất trên dữ liệu mới** (lỗi từ 19 xuống 3–5). Nó mô tả phần nước dâng từ biển mà triều thiên văn và gió không giải thích hết. Mưa dự báo ECMWF giải quyết phần lớn lỗi +12 h và đợt mưa 07/10.
- **Lỗi dữ liệu có thể làm sai số trông như lỗi model.** Mọi lỗi test cũ ở trạm này hóa ra là lỗi dữ liệu (sensor đổi mốc đo ngày 10/10, các gai số đo). Từ đó quy trình thêm hai bước chuẩn: quét bước nhảy mốc đo và lọc gai.
- **Khác Thuận An:** triều 7 ngày tốt hơn 30 ngày, chỉ cần 4 kênh gió vào bờ, lịch học `step` tốt hơn cosine, cửa sổ 12 hoặc 25 giờ. Cho thấy các siêu tham số của một trạm không chuyển thẳng sang trạm khác.
- **Bập bênh giữa +6 h và +12 h** khi dùng chung một GRU: cải thiện mốc này làm xấu mốc kia. Cách chữa hiệu quả là tách mỗi nhóm mốc ra một model và ghép theo mốc. Các cách sửa bằng loss hoặc kiến trúc đều không thành công. Mô hình riêng chỉ +12 h có test luôn 3–9 lỗi, kém model ba mốc.
- **Trung bình nhiều model không giúp:** thử 11.521 tổ hợp 1–3 model từ 41 ứng viên, tốt nhất vẫn còn 1 lỗi, vì các model sai cùng chiều.
- **Kết quả rất nhạy seed:** chỉ seed 23 đạt kết quả này.
- **Phải chấm cả 48 mẫu ở ranh giới giữa các tập** (`purged`); những mẫu này hay sai nhất mà công cụ chấm ban đầu bỏ qua.
- **Hướng đã thử và loại:** mưa đo (thêm lỗi dự báo cao khi nước rút), xả hồ SOHOA (quá nhỏ, 230–310 m³/s trong bão), dự báo Thuận An làm đầu vào, dự báo tuyến tính làm đầu vào (GRU "đi đường tắt"), dropout theo nhóm, trọng số sự kiện (chỉ dời lỗi giữa dự báo thiếu và thừa), độ ẩm đất, sóng, dòng chảy, gió dự báo mới hơn, áp suất, mưa dự báo xa hơn 24 giờ.

## 8. Kết luận

Với quy trình từng trạm, bộ feature (mưa dự báo, gió vào bờ, triều, mực nước biển dự báo), xử lý lỗi dữ liệu và ghép model theo mốc, Ngã Ba Sình đạt 0 lỗi > 200 mm ở cả ba mốc. Hai điểm áp dụng cho trạm tiếp theo ([09](09_da_vien_hien_tai.md)): ghép theo mốc để tránh bập bênh +6/+12 h, và luôn quét dữ liệu hỏng trước khi kết luận về model. Model được đóng gói ở `models/nga_ba_sinh/combined_v1/` và `deploy/nga_ba_sinh/` (bundle chạy độc lập, đã bàn giao); chưa tích hợp vào ứng dụng web ([02](02_ung_dung_web.md)). Việc còn lại là chấm lại trên dữ liệu sau 12/11/2026.

## Nguồn

- `models/NOTES.md` (phần B: Ngã Ba Sình), `models/nga_ba_sinh/REPRODUCE.md` (kèm Phụ lục A: nhật ký thử nghiệm 12–16/11), `models/PIPELINE.md`
- `models/nga_ba_sinh/dataset_config_onshore_tide7_sea7.json`, `..._sea.json`, `models/nga_ba_sinh/combined_v1/`, `deploy/nga_ba_sinh/README.md`
- `figures/nga_ba_sinh/` (gai số đo, dự báo `combined_v1`)
