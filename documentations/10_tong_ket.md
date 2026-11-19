# 10. Tổng kết

## 10.1 Tóm tắt cả quá trình

Từ 30/09 đến 18/11/2026, nghiên cứu đi qua chín giai đoạn để dự báo mực nước ở Huế sao cho không còn lỗi tuyệt đối > 200 mm ở ba mốc +3, +6 và +12 giờ. Kết quả cuối cùng:

- **Thuận An:** 0 lỗi ở cả ba mốc, trên validation, test và giai đoạn dữ liệu mới.
- **Ngã Ba Sình:** 0 lỗi ở cả ba mốc, trên validation, test và giai đoạn dữ liệu mới.
- **Dã Viên:** +1, +2 và +3 h đạt 0 lỗi; +6 h còn 8–12 lỗi; +12 h chưa có model dùng được. Nghiên cứu tạm dừng.

Hai trạm thành công và mốc +3 h của trạm thứ ba đều đến từ một cách làm chung: **model nhỏ riêng cho từng trạm, đầu vào được thiết kế có chủ đích, và quy trình chọn model chặt**. Các hướng "tổng quát hơn" (model chung nhiều trạm, đổi kiến trúc, sửa trọng số, mô hình toán, kết hợp toán và mạng nơ-ron) không đạt mục tiêu.

## 10.2 Bảng so sánh các giai đoạn

| # | Giai đoạn | Thời gian | Trạm | Hướng | Kết quả lỗi > 200 mm | Hơn giai đoạn trước? |
|---|---|---|---|---|---|---|
| 1 | Model local Dã Viên | 30/09–~23/10 | Dã Viên | LSTM, ResidualLSTM, tree; mưa và dự báo mưa | model B: 48 lỗi trên test (legacy 97) | — |
| 2 | Ứng dụng web | 12/10–~23/10 | Dã Viên | tích hợp model vào hệ thống | không áp dụng | — |
| 3 | Regional | ~23/10–4/11 | 11 trạm | nhiều kiến trúc, loss, số trạm | MAE tốt hơn, lỗi lớn không giảm (23 so với 21 ở Dã Viên) | không về lỗi lớn; chọn ra GRU |
| 4 | Weight-space programming | 4–6/11 | Dã Viên, regional | sửa trực tiếp trọng số | giảm 3–12% trên validation, không khái quát | không; nhưng tìm ra chu kỳ triều 12,42 h |
| 5 | Mô hình toán học | ~3–10/11 | Dã Viên | M0–M7, hiệu chỉnh pha | tốt nhất 31 trên test (GRU: 26) | không; có neo quan trắc và feature triều |
| 6 | Hybrid | 10–11/11 | Dã Viên | H0–H8, CFW-TD, AMLMN, ERA, thiết kế lại feature | không vượt nền toán đóng băng (33–35 validation, 31–34 test) | không về model; có bộ feature đầu vào mới |
| 7 | Thuận An | ~9–12/11 | Thuận An | quy trình từng trạm | **0** | có: lần đầu đạt 0 |
| 8 | Ngã Ba Sình | 12–16/11 | Ngã Ba Sình | quy trình từng trạm, mực nước biển, ghép theo mốc | **0** | có |
| 9 | Dã Viên (quay lại) | 17–18/11 | Dã Viên | quy trình từng trạm | +3 h: 0; +6 h: 8–12; +12 h: 37–72 | có (so với giai đoạn 1–6) |

## 10.3 Vì sao cách làm của giai đoạn 7–9 thành công còn các hướng trước thì không

1. **Thông tin đầu vào đúng quan trọng hơn kiến trúc.** Từ giai đoạn 3 đến 6, nhiều kiến trúc và mô hình đều cho kết quả gần nhau; bước nhảy chất lượng đến khi thêm đúng loại thông tin: mưa dự báo nhiều ô ECMWF, gió vào bờ dự báo, mực nước biển dự báo, triều ước lượng trên 7–30 ngày.
2. **Model nhỏ, riêng từng trạm, cửa sổ ngắn.** Các GRU chỉ vài chục nghìn tham số với cửa sổ 12–25 giờ tổng quát hóa tốt hơn các model chung 11 trạm hoặc cửa sổ 96 giờ, vốn học thuộc ít đợt lũ của train.
3. **Dữ liệu sạch được kiểm tra có hệ thống.** Sensor đổi mốc đo, sensor đứng, gai số đo từng làm sai số trông như lỗi model; mọi lỗi test cũ ở Ngã Ba Sình hóa ra là lỗi dữ liệu.
4. **Chọn model chặt hơn.** Chấm từng epoch trên mọi tập (kể cả giờ ranh giới và giai đoạn mới), chọn epoch đạt 0 lỗi, ghép model (trung bình hoặc theo mốc) thay vì tin một epoch theo validation. Chọn trạm có thông tin đầy đủ (gần biển) cũng giúp: Thuận An vốn dễ hơn các trạm sông.
5. **Ghép theo mốc** giải quyết sự bập bênh giữa +6 h và +12 h khi dùng chung một GRU.

## 10.4 Những phát hiện đáng giữ

| Phát hiện | Giai đoạn |
|---|---|
| Lỗi lớn đến theo đợt (bão, lũ), không rải đều; 94% lỗi của model B nằm trong chuỗi từ 2 giờ liên tiếp | 1 |
| GRU là kiến trúc ổn định nhất về lỗi lớn | 3 |
| Cải thiện lỗi lớn trên validation sinh lỗi mới trên test, và ngược lại (lỗi bị dời đi, không mất đi) | 4, 5, 6, 9 |
| Chu kỳ triều 12,42 h là nền của feature triều; biên độ triều đổi theo mùa | 4, 5, 7 |
| Neo vào quan trắc hiện tại giải thích phần lớn gain của các mô hình toán | 5 |
| Phần lớn lỗi còn lại ở Dã Viên do thiếu thông tin thượng nguồn và lịch sử lũ, không do kiến trúc | 6, 9 |
| Đợt nước rút ở các trạm gần biển đi theo mực nước biển và gió, không theo lũ sông | 6 |
| Bộ feature: cụm trạm mưa, ô ECMWF chọn lọc, cửa sổ ngắn, gió Previous Runs, log1p, triều, mực nước biển dự báo | 6, 7, 8 |
| Bập bênh +6 h / +12 h trong một model; ghép theo mốc | 8, 9 |
| Biết khi nào model sai (mưa lớn) nhưng không biết sai về phía nào | 9 |
| Hiệu quả trên hồi quy tuyến tính không dự đoán được hiệu quả trên GRU (spring_neap: không giúp tuyến tính, nhưng nằm trong model +6 h tốt nhất của GRU) | 9 |

## 10.5 Các hướng đã thử và không thành công

| Hướng | Giai đoạn | Kết quả |
|---|---|---|
| XGBoost, LightGBM | 1, 3 | kém model sâu |
| Thêm hoặc đổi trạm mưa, dự báo 24 mốc | 1, 3 | không giảm lỗi lớn |
| Model chung nhiều trạm; Transformer, TCN, CNN-GRU, MLP | 3 | không giảm lỗi lớn ở Dã Viên |
| Loss lấy số lỗi > 200 mm làm mục tiêu | 3 | kém hơn |
| Sửa trọng số (7 nhánh), fine-tune từ model đã sửa | 4 | không khái quát |
| Mô hình toán (M0–M7) | 5 | không thắng GRU trên test |
| Kết hợp toán với GRU, định tuyến mưa thượng nguồn, nâng ma trận | 6 | không vượt nền toán đóng băng |
| GRU chỉ dùng mực nước, cân bằng lại dữ liệu train | 7 | trong mức nhiễu |
| Mưa quan trắc, xả hồ, độ ẩm đất, sóng, áp suất, dropout nhóm, trọng số sự kiện | 7, 8 | không giúp, hoặc chỉ dời lỗi |
| Dự báo trạm khác hoặc của chính model làm đầu vào, finetune giữa trạm | 8, 9 | không giúp |
| Nhóm flow và mưa dạng API (thử trên GRU), triều 15/30 ngày, dự báo của chính model, hậu xử lý | 9 | không giúp, hoặc dời lỗi |
| GloFAS, pha triều riêng | 9 | chỉ thử bằng hồi quy tuyến tính, chưa thử trên GRU; chưa kết luận |

## 10.6 Giới hạn và rủi ro

- **Chỉ có khoảng một năm dữ liệu** (từ 11/2025); quá ít đợt bão lũ cho model học, và các đợt này lại chồng lấn với validation và test.
- **Đánh giá lạc quan.** Các model cuối cùng được chọn sau khi đã xem validation, test và giai đoạn mới; không còn tập chưa dùng. Chỉ dùng một seed ở hầu hết các trạm, và kết quả rất nhạy seed (Ngã Ba Sình và Dã Viên chỉ đạt với seed 23). "0 lỗi" đúng trên các giai đoạn đã chấm, nhưng cần xác nhận trên dữ liệu sau 12/11/2026.
- **Biên an toàn mỏng:** sai số lớn nhất của các model 0 lỗi nằm ở 190–198 mm.
- **Dự báo thời tiết là hồi cứu.** Thời điểm dự báo ECMWF thực sự có sẵn trong quá khứ chưa kiểm chứng được (giả định trễ 6 giờ); dự báo ECMWF thường thấp hơn mưa thực trong đợt lớn.
- **Định nghĩa nhãn:** mọi kết quả chấm theo mực nước trung bình giờ. Nếu nhãn là số đo tức thời lúc hh:00 thì kết quả Dã Viên đổi (+3 h: 0 → 4 lỗi; +6 h: 12 → 19 lỗi).
- **Chưa tích hợp vào ứng dụng:** model Thuận An và Ngã Ba Sình được đóng gói độc lập (bundle `deploy/`, đã bàn giao, không nằm trong git) nhưng chưa deploy trong ứng dụng web. Cần thêm thu thập gió ngoài khơi, mực nước biển dự báo, dự báo mưa nhiều ô. Có lỗi đã biết: `predict.py` dừng khi không có lượt chạy ECMWF nào.
- **Lưu trữ:** các báo cáo gốc (`reports/`), `data/` và `scripts/local/` nằm ngoài git; cần sao lưu riêng.

## 10.7 Việc còn lại và hướng tiếp theo

1. Chấm lại các model đã chọn trên dữ liệu sau 12/11/2026 (nhất là Ngã Ba Sình và Dã Viên, biên mỏng).
2. Dã Viên +6 h và +12 h: thử trên GRU các feature mới chỉ qua sàng lọc tuyến tính (GloFAS, pha triều riêng); tách đóng góp riêng của spring_neap và của dự báo quỹ đạo; tìm dữ liệu mực nước hoặc lưu lượng theo giờ ở thượng lưu và các mùa lũ trước 11/2025; thêm thông tin 10 phút gần nhất; thành phần loss phạt báo thừa khi nước rút.
3. Chạy thêm seed và đánh giá ổn định cho các model đã chọn.
4. Làm rõ định nghĩa nhãn với người hướng dẫn.
5. Cập nhật ứng dụng web với các model và nguồn dữ liệu mới.
6. Mở rộng quy trình từng trạm sang các trạm còn lại.

## 10.8 Bài học phương pháp

- Mỗi lần chỉ thêm một nhóm đầu vào mới; sàng lọc bằng hồi quy tuyến tính trước khi train, nhưng chỉ dùng kết quả đó để loại thô: một feature không giúp hồi quy tuyến tính vẫn có thể giúp GRU (và ngược lại), nên feature quan trọng nên được thử trực tiếp trên GRU.
- Báo cáo kết quả trên mọi tập cùng lúc; chọn theo validation nhưng luôn đối chiếu với test và giai đoạn mới. Một kết quả tốt trên một tập thường chỉ là lỗi bị dời sang tập khác.
- Kiểm tra dữ liệu và nhãn trước khi kết luận về model.
- Phân biệt "tốt hơn trên tập đã xem" với "tốt hơn trên dữ liệu mới"; ghi rõ khi một con số là hồi cứu hoặc lạc quan.
- Kiểm chứng tái tạo được (dữ liệu, dataset, huấn luyện tất định) trước khi bàn giao.
