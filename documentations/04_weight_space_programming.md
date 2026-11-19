# Giai đoạn 4: Weight-space programming (WSP)

## 1. Thời gian

4/11/2026 đến 6/11/2026. Việc đọc tài liệu nguồn bắt đầu ngày 4/11 và nhánh cuối cùng (GRU 96 hidden) hoàn thành 6/11.

## 2. Trọng tâm

Sau khi đổi kiến trúc không giảm được lỗi lớn (giai đoạn 3), thử một cách làm khác: **không train thêm, mà sửa trực tiếp trọng số của model đã train** (weight-space programming) để giảm lỗi > 200 mm. Cảm hứng từ bài "Sparks of Weight-Space Programming" (đăng 2/11/2026), nơi các tác giả lập trình thẳng trọng số của mạng để làm một nhiệm vụ.

Cách làm: mổ xẻ từng tensor và từng cổng GRU, tìm cơ chế gây lỗi bằng can thiệp có đối chứng (tắt, nhân đôi, thay thế phần mưa hoặc lịch sử), rồi sửa trọng số theo công thức toán học cố định, không dùng optimizer, không hồi quy trên sai số. Có ba loại việc:

1. **Sửa trọng số các model đã train:** GRU regional gốc, ba model riêng cho Dã Viên, model WSP A mở rộng ra toàn vùng, Transformer 64, GRU 96 hidden.
2. **Fine-tune từ model đã sửa (E0–E3)** để xem huấn luyện thêm có giữ được lợi ích không.
3. **Mạng lập trình hoàn toàn bằng trọng số giải tích** (48 chuyên gia hài hòa cố định, không train).

Nhánh phụ chuẩn bị: một ma trận GRU/Transformer rộng 64/96, có hoặc không station embedding sớm (`regional_capacity_experiments`), được train riêng để có nhiều checkpoint làm đối tượng sửa.

## 3. Dữ liệu sử dụng

- Dataset regional 11 trạm (giai đoạn 3), model gốc `regional_o24_gru_epoch005`: GRU 93.632 tham số, 96 giờ đầu vào, 24 đầu ra, 11 head riêng từng trạm.
- Chia dữ liệu: train đến 29/07, validation 31/07–13/09, test khám phá từ 15/09 (7.821 mẫu). Thêm tập **confirmation** có khoảng cách 120–168 giờ với validation để tránh chồng lấn lịch sử.

## 4. Kết quả

Số lỗi > 200 mm ở +6/+12 h cộng lại, dạng `trước → sau khi sửa`. Mọi sửa đổi được đóng băng trước khi chấm tập kiểm chứng.

| Nhánh | Thay đổi | Validation | Test / confirmation |
|---|---|---|---|
| GRU regional gốc | 256 tham số (0,27%) | 444 → 391 | 683 → 685 |
| Ba model Dã Viên A/B/C (382 khảo sát, 170 ứng viên) | thêm hoặc sửa 153–1.408 giá trị | A 40→19, B 51→32, C 48→33 | A 21→21, B 19→24, C 45→42 |
| WSP A mở rộng toàn vùng (32 ứng viên) | head từng trạm | 423 → 423 (bản chọn theo protocol) | 703 → 710 |
| Transformer 64 + embedding sớm | 3.904 giá trị | 445 → 404 | 702 → 669 |
| GRU 96 hidden + embedding sớm | 3.870 giá trị | 411 → 364 | 674 → 652 |
| Mạng lập trình hoàn toàn bằng trọng số giải tích | 135 cấu hình, 14.171 tham số | 406 (so với 411) | 764 (so với 674) |
| Fine-tune từ WSP A (E1–E3) | train thêm | không thắng WSP A | — |

Ghi chú:
- Ở nhánh toàn vùng từ WSP A, bản `history_protected_0.5` giảm test 703 → 682 nhưng tăng lỗi > 1.000 mm ở vài trạm nên bị loại theo ràng buộc đã đặt.
- Fine-tune: cả ba checkpoint E1–E3 tốt nhất đều ở epoch 0, tức giống hệt WSP A; E0 (train từ đầu) tốt hơn trên train nhưng tổng quát hóa kém hơn trên test.
- Hầu hết cải thiện ở validation đến từ **một đợt duy nhất (11–13/09)**; ở giai đoạn sau (từ 22/09) lợi ích chỉ còn 1,4%.

## 5. So với giai đoạn trước

Các model sau khi sửa giảm 3–12% lỗi lớn trên validation nhưng chỉ khoảng 1–3% (hoặc không giảm, hoặc tăng) trên dữ liệu chưa dùng để thiết kế sửa đổi. Không có model nào tốt hơn rõ ràng so với bản GRU regional gốc của giai đoạn 3.

## 6. Lỗi > 200 mm

Không xử lý được. Không nhánh nào về 0, và không có lợi ích nào khái quát được sang dữ liệu mới. Nhánh toàn vùng còn hàng trăm lỗi (vd. 652 trên test). Ở Dã Viên, các model A/B/C không giảm lỗi trên tập confirmation (A 21 → 21).

## 7. Điều mới rút ra

- **Cơ chế "triệt tiêu dao động":** trong model, đáp ứng của +6 h với thành phần dao động 12,42 h (triều) bị yếu hoặc sai pha. Có thể lập trình lại head để khôi phục đúng pha và độ khuếch đại ở một biên độ chuẩn, nhưng không chuyển được sang các biên độ khác, nên không phải lời giải đầy đủ cho lỗi lũ thật.
- **Nhánh mưa dự báo lấn át lịch sử:** nhánh thời tiết dùng ReLU không bị chặn, nên khi mưa dự báo lớn (kích hoạt tối đa 16,4 trên validation so với 5,6 trên train), nó lấn át nhánh lịch sử vốn bị chặn của GRU. Can thiệp nhân quả (thay đổi 0×, 1×, 2× lượng mưa dự báo) cho thấy dự báo của Huỳnh Liên đổi từ −211 đến +3.788 mm. Giảm một nửa nhánh này giảm lỗi nhưng không phải một sửa chữa phổ quát.
- **Rủi ro chọn lọc:** hơn 100 can thiệp trên dữ liệu validation đã xem nhiều lần làm cho kết quả tốt trên development dễ là ngẫu nhiên, ngay cả khi không dùng optimizer.
- **Lập trình trọng số không thay được thông tin còn thiếu:** thêm nhánh mới trong mạng cũng không tạo ra mưa thượng nguồn hay dữ liệu thủy văn chưa quan sát được.

- **Cải thiện lỗi lớn ở validation thì sinh lỗi mới ở test, và ngược lại.** Kết quả lặp lại ở nhiều nhánh: model B của Dã Viên giảm validation 51 → 32 nhưng confirmation tăng 19 → 24 (cứu 2 lỗi, tạo 7 lỗi mới); GRU regional gốc giảm validation 444 → 391 nhưng confirmation 683 → 685; nhánh toàn vùng từ WSP A không đổi ở validation (423) và tăng ở test (703 → 710); model C ở +6 h tăng 15 → 17 trong khi +12 h giảm 30 → 25. Mỗi sửa đổi chỉ dời lỗi từ chỗ này sang chỗ khác. Không có một báo cáo riêng nào viết thẳng kết luận này; bằng chứng nằm rải rác trong các bảng "cứu / tạo mới" (rescued / introduced) của các `FINAL_REPORT.md` ở trên, và lời giải thích cơ chế ở `reports/da_vien_notes.md` (Task 11: val là bão, dự báo thừa khi nước rút; test là triều mạnh, kéo theo hai hướng ngược nhau; Task 35: mức lệch chung của model đi xuống thì lỗi nước lên xấu đi còn lỗi nước rút tốt lên, và ngược lại, không có mức lệch nào giữ được cả hai dưới 200 mm). Hệ quả cho cách làm sau này: phải chấm mọi tập và mọi epoch cùng lúc, không chọn theo validation đơn lẻ.
- **Chu kỳ 12,42 h (thủy triều M2) là phát hiện hữu ích nhất của giai đoạn này.** Phân tích phổ của mực nước Dã Viên cho thấy đỉnh ở 12,42 h (kèm thành phần 24 h) và các model đều đáp ứng yếu hoặc sai pha với chu kỳ này ở +6 h. Từ đó rút ra rằng cần đưa **thành phần triều vào làm feature** thay vì để model tự học. Đây là cơ sở của feature triều (hồi quy điều hòa trên chính mực nước của trạm, cửa sổ 7 hoặc 30 ngày) dùng ở Thuận An, Ngã Ba Sình và Dã Viên (giai đoạn 7–9); các model đó đạt 0 lỗi > 200 mm ở Thuận An, Ngã Ba Sình và ở mốc +3 h của Dã Viên. Chu kỳ này cũng là nền cho các mô hình toán học ở giai đoạn 5. Chi tiết phân tích phổ ở `reports/da_vien_mathematical_model_research/TIDAL_HARMONIC_ANALYSIS.md`; hằng số `M2 = 12,4206 h` trong `src/ai/station_dataset/features.py`.

## 8. Kết luận

Sửa trực tiếp trọng số giúp hiểu rõ hơn cách model sai (triệt tiêu dao động, nhánh mưa lấn át), nhưng không đưa được số lỗi > 200 mm về 0 và lợi ích không khái quát. Giá trị lớn nhất của giai đoạn này là phát hiện chu kỳ triều 12,42 h, về sau trở thành feature triều giúp các trạm riêng đạt 0 lỗi. Model production không bị thay đổi; các checkpoint đã sửa được giữ làm tài liệu nghiên cứu, không chọn lại sau khi thấy kết quả. Từ đây hướng sang mô hình toán học tường minh ([05](05_mo_hinh_toan_hoc.md)) và về sau là quy trình từng trạm.

## Nguồn

- `reports/weight_space_programming/` (`FINAL_REPORT.md` ở gốc và ở `da_vien_multimodel/`, `regional_from_wsp_a/`, `t2_transformer_regional/`, `r3_gru_regional/`; tài liệu bài báo trong `sources/`)
- `reports/regional_capacity_experiments/` (ma trận GRU/Transformer, `evaluation/COMPARISON_REPORT.md`, `mechanistic_screening/`)
- `reports/local_finetuning/da_vien_wsp_a/evaluation_long/REPORT.md`
- `reports/programmed_regional_forecasting/FINAL_REPORT.md`
- `reports/model_comparisons/wsp_A_vs_original_B/REPORT.md`
