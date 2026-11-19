# Tài liệu nghiên cứu dự báo mực nước Huế

Tài liệu kể lại toàn bộ quá trình nghiên cứu từ 30/09/2026 đến 18/11/2026, theo thứ tự thời gian, mỗi giai đoạn một file. Người đọc mới nên đọc [10_tong_ket.md](10_tong_ket.md) trước để nắm toàn cảnh, rồi đọc từng giai đoạn khi cần chi tiết. Các tên như M3, H4, CFW-TD… là nhãn nội bộ đặt ra trong lúc nghiên cứu; ý nghĩa của chúng được giải thích ở các file tương ứng (xem phụ lục của [05](05_mo_hinh_toan_hoc.md)).

## Mục tiêu của cả quá trình

Dự báo mực nước tại các trạm ở Huế sao cho **không còn lỗi tuyệt đối > 200 mm** (chặt: `|dự báo − thực tế| > 200`, đúng 200 không tính). Mentor yêu cầu xử lý hết mọi lỗi này. MAE chỉ là tiêu chí phụ.

## Các giai đoạn

| # | File | Giai đoạn | Thời gian (dự kiến, cần đối chiếu) | Trạm | Kết quả với lỗi > 200 mm |
|---|---|---|---|---|---|
| 0 | [00_du_lieu_va_quy_uoc.md](00_du_lieu_va_quy_uoc.md) | Dữ liệu và quy ước chung | — | — | — |
| 1 | [01_da_vien_local.md](01_da_vien_local.md) | Model local ở Dã Viên (LSTM, tree, model B) | 30/09 – ~21/10 | Dã Viên | model B: 48 lỗi trên test (legacy 97); chưa hết |
| 2 | [02_ung_dung_web.md](02_ung_dung_web.md) | Ứng dụng web tích hợp model | 12/10 – ~23/10 | Dã Viên | không áp dụng |
| 3 | [03_regional.md](03_regional.md) | Model regional, nhiều kiến trúc | ~23/10 – ~4/11 | 11 trạm | MAE tốt hơn model B, nhưng không giảm lỗi lớn (23 so với 21 ở Dã Viên) |
| 4 | [04_weight_space_programming.md](04_weight_space_programming.md) | Weight-space programming | 4/11 – 6/11 | Dã Viên, regional | giảm 3–12% trên val nhưng không khái quát; không về 0; tìm ra chu kỳ triều 12,42 h |
| 5 | [05_mo_hinh_toan_hoc.md](05_mo_hinh_toan_hoc.md) | Mô hình toán học (M0–M7) | ~3/11 – 10/11 | Dã Viên | tốt nhất 31 lỗi trên test (GRU: 26); chưa thắng GRU; phát triển feature triều 12,42 h |
| 6 | [06_hybrid.md](06_hybrid.md) | Hybrid (H0–H8, CFW-TD, AMLMN, ERA) | ~10/11 – 11/11 | Dã Viên | không vượt nền toán đóng băng (33–35 val, 31–34 test); chưa về 0 |
| 7 | [07_thuan_an.md](07_thuan_an.md) | Thuận An | ~9/11 – 12/11 | Thuận An | 0 lỗi (val, test, giai đoạn mới) |
| 8 | [08_nga_ba_sinh.md](08_nga_ba_sinh.md) | Ngã Ba Sình | 12/11 – 16/11 | Ngã Ba Sình | 0 lỗi (val, test, giai đoạn mới) |
| 9 | [09_da_vien_hien_tai.md](09_da_vien_hien_tai.md) | Quay lại Dã Viên (tạm dừng 18/11) | 17/11 – 18/11 | Dã Viên | +3 h: 0; +6 h: 8–12; +12 h: 37–72 (chưa có ứng viên) |
| 10 | [10_tong_ket.md](10_tong_ket.md) | Tổng kết, bài học, việc còn lại | 30/09 – 18/11 | Tất cả | Thuận An, Ngã Ba Sình: 0; Dã Viên: +3 h 0 |

## Mỗi file giai đoạn có cùng cấu trúc

1. Thời gian (bắt đầu, kết thúc)
2. Trọng tâm: giai đoạn này làm gì
3. Dữ liệu sử dụng
4. Kết quả
5. So với giai đoạn trước: có tốt hơn không
6. Lỗi > 200 mm: có xử lý được không
7. Điều mới rút ra (chỉ ghi nếu có)
8. Kết luận
9. Nguồn (đường dẫn tới báo cáo gốc)

## Lưu ý lưu trữ

`reports/`, `data/` và `scripts/local/` nằm trong `.gitignore`. Báo cáo gốc mà tài liệu này trỏ tới chỉ có trên ổ đĩa, không có trong git. Nên sao lưu `reports/` riêng.
