# Giai đoạn 9: Quay lại Dã Viên (tạm dừng 18/11)

Sau khi Thuận An và Ngã Ba Sình đạt 0 lỗi, nghiên cứu quay lại Dã Viên, trạm đã dừng ở hàng chục lỗi suốt các giai đoạn 1–6, với quy trình, feature và bài học của hai trạm kia. Mốc +3 h đã đạt 0 lỗi; mốc +6 h và +12 h chưa. Nghiên cứu được tạm dừng ngày 18/11/2026; giai đoạn này vì vậy chưa kết thúc theo nghĩa đạt mục tiêu.

## 1. Thời gian

17/11/2026 đến 18/11/2026 (tạm dừng). Nhật ký chi tiết (36 phần việc) được lưu ở `reports/da_vien_notes.md` (ngoài git).

## 2. Trọng tâm

Mục tiêu: **0 lỗi > 200 mm** trên mọi giờ ngoài tập train (validation 31/07–14/09, giờ ranh giới, test 15/09–14/10 và giai đoạn mới 15/10–12/11) ở +3, +6 và +12 h. Người hướng dẫn yêu cầu xử lý hết mọi lỗi này.

Cách làm, theo thứ tự:

1. **Làm sạch và phân tích dữ liệu Dã Viên**, kiểm chứng giả thuyết "Dã Viên khó hơn Ngã Ba Sình vì biến động lớn hơn".
2. **Quét thông tin đầu vào** cho từng mốc: dự báo hoặc quan trắc của Ngã Ba Sình, mực nước biển dự báo, cụm trạm mưa, ECMWF, lưu lượng sông.
3. **Quét cách huấn luyện:** nền persistence hay tuyến tính, độ dài cửa sổ (12, 25, 48 giờ), hệ số phạt thiếu (under-weight), trọng số mốc, kích thước mạng, finetune từ model Ngã Ba Sình, tách hoặc ghép mốc.
4. **Feature mới:** nhóm dòng chảy (flow), pha triều, mưa dạng API, lưu lượng GloFAS, triều 15 và 30 ngày, nhóm pha nước cường/nước kém (spring_neap), dự báo của chính model làm đầu vào (self_fc).
5. **Augmentation** (phóng to hoặc thu nhỏ phần nước dâng/lũ khi train).
6. **Dự báo cả quỹ đạo +1…+6 h**, ensemble chéo giữa các run, hậu xử lý.

## 3. Dữ liệu sử dụng

- **Dữ liệu được làm sạch trước tiên.** Hai đoạn sensor đứng (báo đúng 441,0 mm suốt 11,2 ngày từ 01/02 và 534,0 mm suốt 5,1 ngày từ 20/06) bị coi là thiếu dữ liệu; 5 gai số đo bị lọc (ngưỡng 85 mm), giữ lại một đoạn tụt ngày 11/04 vì Ngã Ba Sình cũng tụt cùng lúc; không có bước đổi mốc đo như ở Ngã Ba Sình. Sau khi làm sạch, mực nước giờ không còn gai.
- **Dataset `dv_h36_all`** (mốc +3 và +6 h, cửa sổ 12 giờ, tối đa 96 giờ, 58 đầu vào ở bản có thêm spring_neap), gồm các nhóm:
  - mực nước Dã Viên;
  - gió vào bờ (4 kênh, Previous Runs);
  - triều 7 ngày;
  - mưa dự báo ECMWF ở 7 ô;
  - mực nước biển dự báo (Open-Meteo Marine);
  - **mực nước quan trắc của Ngã Ba Sình** (đã lọc gai) trong những giờ gần nhất;
  - tùy chọn: cụm trạm mưa Tây/Đông/Đồng bằng, spring_neap.
- Nhãn: mực nước trung bình giờ tại t + h. Chia tập như các trạm trước.
- **Giới hạn dữ liệu:** quan trắc chỉ bắt đầu từ 14/11/2025 (cả cơ sở dữ liệu lẫn API); thử lấy dữ liệu mùa lũ cũ hơn (10–24/11/2024) từ API không trả về số đo nào.

## 4. Kết quả

Số lỗi > 200 mm ngoài tập train (validation + ranh giới + test + giai đoạn mới), 2.519 giờ:

| Mốc | Model tốt nhất | Lỗi > 200 mm | Ghi chú |
|---|---|---:|---|
| +1, +2 h | ứng viên `best_h1to6_sn_aug` | 0 | |
| **+3 h** | **`v1`** (ensemble 10 epoch cuối) | **0** | sai số lớn nhất 197 mm |
| **+6 h** | ứng viên `best_h1to6_sn_aug` (dự báo +1…+6 h, spring_neap, augmentation, seed 23, ensemble 20 checkpoint) | **8**, đều 200–217 mm | chỉ đạt với seed 23 |
| +6 h (đã đóng gói) | `v1` | 12 | lớn nhất 256 mm |
| +12 h | model chung +3/+6/+12 h | 37–72 | chưa có ứng viên dùng được |

**`v1`** (từng được đóng gói; hiện chỉ còn ghi lại dataset và lệnh train ở `models/da_vien/README.md`): mỗi mốc lấy trung bình dự báo của 10 epoch cuối (91–100) của một lần train; quy tắc đặt trước, không chọn epoch theo test.

- **+3 h:** GRU residual 64×3 trên nền persistence, cửa sổ 25 giờ, đầu ra chung (không station embedding), hệ số phạt thiếu bằng 0, seed 23, **có augmentation** nước dâng/lũ.
- **+6 h:** cùng cấu hình, không augmentation.

Chi tiết theo từng tập (số lỗi; MAE mm; sai số lớn nhất mm):

| Mốc | Train | Validation | Ranh giới | Test | Giai đoạn mới |
|---|---|---|---|---|---|
| +3 h | 0; 22,1; 193 | 0; 29,6; 166 | 0; 47,8; 116 | 0; 39,3; 170 | 0; 33,5; 197 |
| +6 h | 0; 29,3; 192 | 3; 48,2; 256 | 0; 69,9; 186 | 5; 56,7; 229 | 4; 46,2; 215 |

**Các hướng và kết quả (số lỗi +6 h ngoài train; baseline 10 tốt nhất / 12 trung bình):**

| Hướng | Kết quả |
|---|---|
| Dự báo Ngã Ba Sình làm đầu vào | Làm kém đi (MAE +9…+10 mm): hồi quy học trên train, nơi dự báo Ngã Ba Sình tốt gần gấp đôi thực tế |
| **Mực nước quan trắc Ngã Ba Sình làm đầu vào** | Giải thích 88–96% biến động; là đầu vào chính của `v1` |
| Finetune từ model Ngã Ba Sình | Không giúp |
| Cửa sổ 12 / 25 / 48 giờ | 25 giờ nhỉnh hơn nhẹ (ensemble 10 lỗi so với 12) |
| Nền persistence và nền tuyến tính | Persistence tốt hơn |
| Nhóm dòng chảy (flow) và mưa dạng API, thử trên GRU (từng nhóm và cả ba, kể cả pha triều) | Kém hơn rõ: +6 h trung vị 24–27 lỗi so với 15; GRU phản ứng mạnh hơn nhưng sai chiều trong bão |
| Pha triều riêng; lưu lượng GloFAS | **Chỉ mới sàng lọc bằng hồi quy tuyến tính**, chưa có lần train GRU riêng. GloFAS không giúp trên hồi quy tuyến tính, kể cả khi biết trước tương lai (dữ liệu theo ngày, trễ khoảng 1 ngày, không có hồ chứa) và chưa được thử trên GRU |
| Triều 15, 30 ngày | 15 ngày: +3 h ổn định hơn, +6 h kém nhẹ; 30 ngày: kém rõ. Giữ triều 7 ngày |
| spring_neap (pha nước cường/nước kém) | Hồi quy tuyến tính: không đổi số lỗi. **Trên GRU: có tác dụng**: validation +6 h từ 3 xuống 1 lỗi (có run 10 epoch đạt 0), nhưng dời một phần lỗi sang test và giai đoạn mới khi dùng riêng; kết hợp với dự báo quỹ đạo +1…+6 h thì tạo ra model +6 h ít lỗi nhất hiện có (8–10 lỗi) |
| Dự báo của chính model làm đầu vào (self_fc) | Không giúp +6 h |
| Loss chỉ phạt phần đuôi, hậu xử lý tuyến tính, hiệu chỉnh trực tuyến, time-lagged ensemble | Không giúp |
| **Augmentation** nước dâng/lũ | **+3 h: số epoch 0 lỗi tăng từ 15 lên 40–43**; +6 h, +12 h không cải thiện |
| **Dự báo cả quỹ đạo +1…+6 h kèm spring_neap** | **+6 h còn 8–10 lỗi**, nhưng chỉ với seed 23 (seed 42 cho 18, seed 17 cho 24) |
| Ghép chéo giữa các run (3.682 tổ hợp 1–3 run) | Tốt nhất 8 lỗi, không có tổ hợp nào về 0 |

## 5. So với giai đoạn trước

Tiến bộ rõ so với các giai đoạn trước ở Dã Viên: +3 h đạt 0 lỗi lần đầu, +6 h xuống 8–12 lỗi (các giai đoạn 1–6 dừng ở 20–60 lỗi). So với Ngã Ba Sình và Thuận An (0 lỗi ở cả ba mốc) thì chưa bằng: +6 h còn 8–12 lỗi và +12 h còn 37–72 lỗi.

## 6. Lỗi > 200 mm

**Xử lý được một phần.**

- **+3 h: 0 lỗi.** Nếu nhãn là số đo tức thời lúc hh:00 thay vì trung bình giờ thì `v1` có 4 lỗi ở +3 h và 19 lỗi ở +6 h (câu hỏi còn mở về định nghĩa nhãn).
- **+6 h: 8–12 lỗi**, đều rất gần ngưỡng (200–217 mm với ứng viên tốt nhất). **Ngoài giờ bão/lũ (79% thời gian) đã 0 lỗi** ở cả +3 và +6 h; mọi lỗi còn lại nằm trong sáu đợt: bão 11–13/09, 21/09, 13/10, 28–29/10 và lũ 09–11/11. Kiểu lỗi: nhận ra muộn đợt nước lên, hoặc đánh giá thấp tốc độ rút sau đỉnh.
- **+12 h: 37–72 lỗi**, chưa có ứng viên dùng được.

## 7. Điều mới rút ra

- **Dã Viên khó hơn Ngã Ba Sình không phải vì biến động chung lớn hơn.** Ở +6 h hai trạm biến động như nhau (tỉ lệ 1,03); ở +12 h Dã Viên biến động hơn 14%. Phần khó là **thành phần riêng của đoạn sông Ngã Ba Sình → Dã Viên** (độ lệch chuẩn khoảng 55 mm, lên 150–190 mm ở các giờ lỗi), cộng dồn lên sai số chung với Ngã Ba Sình. Ngã Ba Sình chỉ còn biên 12–15 mm ở +6 h dưới ngưỡng 200, nên thêm một thành phần nhỏ là đủ vượt.
- **Cơ chế "bập bênh" giữa các mốc.** Ở các giờ khó, lỗi +6 h là dự báo thiếu khi nước lên, còn lỗi +12 h là dự báo thiếu mức rút khi nước rút, đều do model nén biên độ (độ dốc dự báo/thật chỉ 0,80 và 0,63). Tổng độ lớn hai lỗi khoảng 390–410 mm, nên không có mức lệch chung nào giữ được cả hai dưới 200 mm. Ghép mốc +3/+6 h từ epoch sớm với +12 h từ epoch muộn của cùng một lần train cho 0 lỗi ở mọi giờ, và đây là lý do model Ngã Ba Sình ghép theo mốc.
- **Biết khi nào sai nhưng không biết sai về phía nào.** Mưa đo 24 giờ vượt phân vị 90 bắt được 9/10 giờ lỗi (AUC 0,89), nhưng trong các giờ đó sai số trung bình gần 0 và số giờ sai dương, âm xấp xỉ nhau (33/39). Mọi cách hiệu chỉnh theo chiều dự báo thay đổi đều không chữa được.
- **Nén biên độ không do thiếu ví dụ lớn.** Phóng to các đợt lũ trong train không đổi độ dốc dự báo/thật (0,76 so với 0,77); đó là phản ứng tối ưu của model khi một phần biến động không dự báo được. Augmentation chỉ giúp +3 h, nơi thông tin đủ.
- **Loss không sửa được lỗi chưa từng có trong train.** Train chỉ có khoảng 19 giờ sai > 150 mm ở +6 h (vì dự báo trên train là trong mẫu), quá ít để học phần đuôi.
- **Dự báo của chính model hay trạm khác làm đầu vào bị thổi phồng trên train.** Dự báo Ngã Ba Sình làm đầu vào hại vì trên train nó "quá tốt"; cách đúng cần dự báo out-of-fold (huấn luyện chéo theo khối thời gian). Nhưng mực nước **thật** của Ngã Ba Sình có giá trị rất lớn: giới hạn trên khi biết trước (oracle) là 27 → 3 lỗi.
- **Kết quả tốt là của riêng một seed (23).** Ghép seed không giúp (13 lỗi khi ghép seed 23 và 42; 12 khi ghép cả ba).
- **Dự báo thưa có thể "đạt 0" do chọn nhịp phát hành**: +6 h mỗi 6 giờ ở các giờ 03, 09, 15, 21 cho 0 lỗi (sai số lớn nhất 199 mm), nhưng chỉ vì các giờ lỗi tình cờ rơi ngoài nhịp đó; không phải cải thiện thật.
- **Hiệu quả của một feature trên hồi quy tuyến tính không dự đoán được hiệu quả trên GRU.** spring_neap không đổi số lỗi trên hồi quy tuyến tính, nhưng là thành phần của model GRU +6 h tốt nhất; ngược lại các feature giúp được hồi quy tuyến tính (nhóm flow có thêm thông tin về phần riêng của Dã Viên: R² từ 21% lên 40% ở validation) lại làm GRU kém đi. Vì vậy sàng lọc tuyến tính chỉ nên dùng để loại thô; những feature chỉ mới qua sàng lọc tuyến tính (GloFAS, pha triều riêng) chưa thể coi là đã bị loại cho GRU. Hiện chưa tách được riêng đóng góp của spring_neap và của dự báo cả quỹ đạo trong model +6 h tốt nhất.
- **Triều Dã Viên là hỗn hợp:** hai đỉnh mỗi ngày với độ cao khác nhau; biên độ đổi theo mùa (M2 từ 7 mm tháng 7 đến 104 mm tháng 10–11); pha nước cường/nước kém chỉ giải thích 1,7% phương sai biên độ triều ngày.
- **Giới hạn gốc:** chỉ một năm dữ liệu nên quá ít đợt bão lũ để học, và không có số đo theo giờ ở thượng lưu sông Hương.

## 8. Kết luận

Với quy trình từng trạm và bộ feature mới, Dã Viên đạt 0 lỗi ở +1, +2 và +3 h, và đưa +6 h từ hàng chục xuống 8–12 lỗi, nhưng chưa đạt mục tiêu ở +6 h và +12 h. Phần lỗi còn lại tập trung trong sáu đợt bão lũ và có nguyên nhân chủ yếu là thiếu thông tin (thượng nguồn, lịch sử mùa lũ) chứ không phải kiến trúc hay cách huấn luyện. Nghiên cứu tạm dừng ngày 18/11/2026.

**Hướng tiếp theo nếu nghiên cứu lại:**

1. Thử các feature mới chỉ qua sàng lọc tuyến tính trên GRU (lưu lượng GloFAS, pha triều riêng); tách đóng góp riêng của spring_neap và của dự báo quỹ đạo.
2. Tìm dữ liệu mực nước hoặc lưu lượng theo giờ ở thượng lưu (Kim Long, Thượng Nhật, hạ lưu Bình Điền và Tả Trạch) và các mùa lũ trước 11/2025.
3. Thêm thông tin 10 phút gần nhất (số đo mới nhất, độ dốc 30–60 phút) làm đầu vào.
4. Thêm thành phần loss phạt báo thừa khi nước rút nhanh.
5. Kiểm tra mọi model trên dữ liệu sau 12/11/2026.
6. Xác nhận định nghĩa nhãn (trung bình giờ hay tức thời).

## Nguồn

- `reports/da_vien_notes.md` (phần tạm dừng ở đầu file và Task 1–36), `models/da_vien/README.md` (kết quả tốt nhất, dataset, lệnh train) và `models/da_vien/dataset_config_h36_all.json`. Các checkpoint đã đóng gói (`v1`, `best_h1to6_sn_aug`, `h3_v1`) và các cấu hình dataset thử nghiệm khác không còn trong `models/`; xem lịch sử git (commit `783f894`)
- `research/scripts/score_runs_all_splits.py` (chấm mọi epoch trên mọi tập), `research/scripts/member_ensemble.py`
- `src/ai/augmentation.py`, `src/ai/station_dataset/features.py` (spring_neap, self_fc, flow)
- `figures/da_vien/`
