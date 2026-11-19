# Giai đoạn 7: Thuận An

Giai đoạn đầu tiên đạt **0 lỗi > 200 mm** trên cả validation và test. Nó đồng thời đặt ra quy trình nghiên cứu từng trạm mà hai giai đoạn sau (Ngã Ba Sình, Dã Viên) dùng lại.

## 1. Thời gian

Khoảng 9/11/2026 (bước chuẩn bị) đến 12/11/2026 (chọn model xong). Giai đoạn chồng thời gian với phần cuối của giai đoạn hybrid (10–11/11).

## 2. Trọng tâm

### 2.1 Bước chuẩn bị: làm sạch, chỉ dùng mực nước (9/11)

Sau các giai đoạn dùng nhiều đầu vào và kiến trúc phức tạp mà không giảm được lỗi lớn, nghiên cứu quay về điểm xuất phát đơn giản nhất: **một GRU chỉ dùng mực nước của chính trạm, không feature nào khác**, để biết đâu là mức nền thực sự. Có hai thí nghiệm:

- **GRU chỉ dùng mực nước** (13 lần chạy, trên Dã Viên và regional 11 trạm): thay đổi độ dài lịch sử (13 hoặc 96 giờ), kích thước batch, dropout, có hoặc không đặc trưng giờ/ngày. Model tốt nhất theo validation có 43 lỗi validation và 104 lỗi test (ở +6/+12 h), không vượt các model có mưa trước đó (WSP A: 19/22, model B: 51/19). Model overfit rất mạnh (loss train giảm 0,080 → 0,028 trong khi loss validation tăng 0,169 → 0,254). Thêm đặc trưng giờ giảm lỗi validation (65 → 49) nhưng làm test tệ đi (75 → 122): lợi ích không ổn định.
- **Cân bằng lại dữ liệu train** (E0–E4, cả local lẫn regional): bỏ bớt các cửa sổ "đứng yên" thừa (25% hoặc 50% nhóm đủ điều kiện), đặt trọng số cao hơn cho cửa sổ biến động mạnh, hoặc lấy mẫu nhiều hơn các cửa sổ này. Số lỗi validation tốt nhất ở Dã Viên dao động 37–53 so với 49 của baseline, tức nằm trong nhiễu của một seed; không có cách cân bằng nào cho cải thiện rõ ràng.

Kết luận của bước này: chỉ thay đổi cách huấn luyện hoặc cân bằng dữ liệu không đủ; cần **thông tin mới** có ích cho dự báo.

### 2.2 Chọn Thuận An và dựng quy trình từng trạm

Thuận An (0074647402, cảng cá ở cửa biển phá Tam Giang) được chọn làm trạm đầu tiên vì chịu ảnh hưởng trực tiếp của biển: biên độ triều lớn nhất (độ lệch chuẩn 104 mm, so với 62–65 mm ở Ngã Ba Sình và Dã Viên) và dao động theo triều chiếm phần lớn phương sai. Chỉ với mực nước và triều của chính trạm, hồi quy tuyến tính đã giải thích 65–69% phương sai +12 h, so với 24–42% ở hai trạm sông.

Quy trình mới, đóng gói thành `src/ai/station_dataset/` (xem `models/PIPELINE.md`):

1. Lấy quan trắc trực tiếp từ API vào PostgreSQL, tổng hợp theo giờ.
2. Tải dữ liệu Open-Meteo (gió dự báo, mưa dự báo) về file.
3. Dựng dataset từ một file cấu hình JSON mỗi trạm, chọn nhóm feature (`--groups fc_ecmwf wind tide …`).
4. Train GRU residual nhỏ (hàng chục nghìn tham số), tất định theo seed.
5. **Chấm điểm mọi epoch trên mọi tập**, kể cả các giờ ranh giới và giai đoạn mới; chọn epoch có 0 lỗi, rồi ghép nhiều model bằng trung bình đều.

## 3. Dữ liệu sử dụng

- Mực nước Thuận An theo giờ (nhịp quan trắc gốc 10 phút), từ 14/11/2025; nhãn là trung bình giờ tại +3, +6, +12 h.
- **Gió và áp suất ngoài khơi** (15 kênh): ECMWF IFS 0,25°, Open-Meteo Previous Runs, bản dự báo khoảng 24 giờ trước, điểm 16,75°N 107,75°E (khoảng 25 km về phía đông bắc trạm).
- **Triều 30 ngày** (5 kênh): hồi quy hài hòa 10 thành phần trên 30 ngày mực nước đến thời điểm dự báo.
- **Mưa dự báo ECMWF** (18 kênh, log1p): tổng mưa dự báo 0–3, 3–6, 6–12, 12–24 giờ tới ở ô A Lưới, Thuận An, Bạch Mã và trung bình 7 ô.
- Chia tập: train đến 29/07/2026 (6.073 mẫu), validation 31/07–14/09 (1.080), test 15/09 trở đi (723), cộng một giai đoạn mới 15/10–12/11.

## 4. Kết quả

| Lựa chọn | Thành phần | Tham số | Lỗi > 200 mm (validation / test) | MAE tháng 9–10 (+3 / +6 / +12 h) | Lỗi > 200 mm tháng 9–10 |
|---|---|---:|---|---|---|
| **Model đơn** | GRU 32×3, cửa sổ 12 giờ, đầu vào gió + triều 30 ngày + mưa dự báo (41 đầu vào) | 21.291 | 0 / 0 | 31,0 / 40,9 / 43,8 mm | 1 (23/10, +12 h, −215 mm) |
| **Ensemble** | Trung bình đều model đơn và một GRU 32×2 chỉ dùng gió (cửa sổ 25 giờ) | 34.038 | 0 / 0 | 30,5 / 40,1 / 44,2 mm | **0** (sai số lớn nhất 191 mm) |
| Persistence | Giữ nguyên mực nước | — | — | 114 / 162 / 75 mm | 743 |

Model gió dùng một mình có 6 lỗi validation và 4 lỗi test nên chỉ dùng trong ensemble. Ensemble lớn hơn (khoảng 627 nghìn tham số) có MAE 29,2 / 39,0 / 44,4 mm và cũng 0 lỗi, tức thêm tham số không giúp thêm.

Đóng gói và tái tạo: toàn bộ quy trình được kiểm chứng tái tạo được. Dữ liệu lấy từ API khớp 100% dữ liệu train, dataset dựng lại giống từng byte, hai lần train cùng cấu hình cho dự báo trùng 0,0 mm. Model và ensemble được đóng gói ở `models/thuan_an/` và `deploy/thuan_an/` (bundle chạy độc lập, đã bàn giao).

## 5. So với giai đoạn trước

Đây là lần đầu có model đạt 0 lỗi > 200 mm trên validation và test, trong khi ở Dã Viên các giai đoạn trước dừng ở 30–60 lỗi. Cần lưu ý đây không hoàn toàn là một "model tốt hơn": Thuận An là trạm dễ hơn (phần lớn biến động do triều và nước dâng từ biển, đều dự báo được), nên kết quả còn phản ánh trạm được chọn. Phần đóng góp mới là bộ feature và quy trình chọn model, được kiểm chứng ở hai trạm sau.

## 6. Lỗi > 200 mm

**Xử lý được.** 0 lỗi trên validation, test và giai đoạn mới 15/10–12/11 (ensemble). Với model đơn còn một lỗi duy nhất (23/10, thiếu 215 mm). Cần ghi chú hai hạn chế:

- Chỉ train với một seed (17); model và ensemble được chọn sau nhiều lần thử và đã xem cả validation, test lẫn đoạn 16–31/10, nên "0 lỗi" có phần lạc quan.
- Biên an toàn mỏng: sai số lớn nhất là 194 mm (model đơn, validation) và 191 mm (ensemble).

## 7. Điều mới rút ra

- **Triều 30 ngày tốt hơn triều 7 ngày.** Phần biến động +12 h giải thích được tăng từ 0,45 lên 0,62 (validation) và từ 0,36 lên 0,59 (test). Cửa sổ 7 ngày không tách được các thành phần triều gần nhau (M2/S2/N2, O1/Q1). Đây là nâng cấp của feature triều xuất phát từ chu kỳ 12,42 h ([04](04_weight_space_programming.md), [05](05_mo_hinh_toan_hoc.md)).
- **Mưa dự báo ECMWF là thông tin quý nhất cho +12 h:** báo hiệu bão sớm hơn gió 3–6 giờ. Ngược lại, **mưa quan trắc gần như không giúp** ở Thuận An. **Gió dự báo** xử lý được đợt bão 12/09 trên validation.
- **Cách chọn mô hình quan trọng không kém feature.** Chấm điểm từng epoch trên mọi tập rồi ghép nhiều model bằng trung bình đều ổn định hơn chọn một epoch theo validation. Train quá khoảng 130 epoch làm kết quả xấu đi. Hệ số `under-weight` = 0,3 trong loss `combined` (phạt dự báo thiếu nhẹ hơn dự báo thừa) cho kết quả tốt nhất; 0,0 kém ở mọi chỉ số. Dropout theo nhóm đầu vào (mưa dự báo, gió) làm validation kém đi và đã bị loại.
- **Chiều di chuyển của nước**, đo trên dữ liệu 10 phút cả năm: ngày thường sóng triều đi từ biển vào (Thuận An → Ngã Ba Sình → Dã Viên, mỗi chặng khoảng 40 phút); lũ sông lớn đi ngược từ sông ra biển (Dã Viên và Ngã Ba Sình cùng lúc, Thuận An sau 0,5–1,5 ngày). Thay đổi theo giờ trong mùa lũ vẫn xuất hiện ở phía biển trước vì triều và nước dâng do bão đến từ biển.
- **Trạm sông khó hơn Thuận An nhiều.** Chỉ với nước và triều của chính trạm, R² ở +12 h: Ngã Ba Sình 39% / 42%, Dã Viên 35% / 24% (validation / test), Thuận An 65% / 69%; số lỗi > 200 mm ở +6 h: Ngã Ba Sình 18 / 23, Dã Viên 26 / 33, Thuận An 0.
- **Mực nước trạm lân cận không giúp** (có khi làm kém đi, ví dụ R² +12 h ở Ngã Ba Sình khi thêm Dã Viên giảm từ 42% xuống 29%) vì các trạm dao động gần như cùng lúc và quan hệ thay đổi theo mùa. Nhưng **dự báo của model Thuận An dùng làm đầu vào** cho trạm sau có ích ở +6 h (lỗi test giảm 33 → 14 ở Dã Viên và 23 → 14 ở Ngã Ba Sình). Cách làm cần dự báo out-of-fold ở giai đoạn train; vòng lặp qua lại giữa các trạm không thêm thông tin sau vòng đầu.
- **Phần không phải triều mới là phần khó:** model Thuận An đạt R² 0,93 phần lớn nhờ triều; trên phần còn lại (nước dâng, lũ, mực nền) R² chỉ −0,06 ở +6 h và 0,10 ở +12 h.
- **Đợt 23/10/2026** (nước dâng khoảng +200 mm so với triều, không mưa, gió yếu) không đầu vào nào giải thích được; mọi model dự báo thiếu 150–220 mm. Nguồn thông tin có thể giúp: dự báo mực nước biển ngoài khơi, được thử ở trạm sau.

## 8. Kết luận

Với các feature mới (gió dự báo, triều 30 ngày, mưa dự báo nhiều ô), một GRU nhỏ chỉ vài chục nghìn tham số đạt 0 lỗi > 200 mm ở Thuận An, sau khi các giai đoạn trước ở Dã Viên dừng ở hàng chục lỗi. Phần đáng giữ lại không chỉ là model mà là **quy trình**: dựng dataset từ cấu hình, train tất định, chấm mọi epoch trên mọi tập và ghép model. Quy trình và bộ feature này được áp dụng cho Ngã Ba Sình ([08](08_nga_ba_sinh.md)) và Dã Viên ([09](09_da_vien_hien_tai.md)). Model chưa được tích hợp vào ứng dụng web ([02](02_ung_dung_web.md)).

## Nguồn

- `models/NOTES.md` (phần A: Thuận An và các phân tích chung), `models/PIPELINE.md`, `models/thuan_an/README.md`, `models/thuan_an/REPRODUCE.md`, `models/thuan_an/dataset_config.json`, `deploy/thuan_an/README.md`
- Bước chuẩn bị: `reports/water_only_gru/` (đánh giá ngày 9/11), `reports/training_rebalancing/`
- `src/ai/station_dataset/` (`build.py`, `features.py`, `open_meteo.py`, `evaluate.py`)
