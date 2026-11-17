# Cảng Cá Thuận An (0074647402): model đã chọn, dự báo +3/+6/+12 h

Hướng dẫn tái tạo từng bước: `REPRODUCE.md` (riêng trạm này) và `../PIPELINE.md` (phần dùng chung). Cấu hình dataset: `dataset_config.json`.

Chọn ngày 12/11/2026. Mỗi thư mục là bản sao nguyên vẹn của một epoch checkpoint (`model.pt`, `model.preprocessing.joblib` đã gồm bộ chuẩn hóa và danh sách cột đầu vào, cùng các file đánh giá), kèm `selection.json` ghi cấu hình, lệnh train, metric, nguồn gốc và sha256. Đã kiểm tra nạp lại từ thư mục này: dự báo validation khớp bản lưu (chênh ≤0,14 mm, do chạy CPU so với GPU).

## Hai lựa chọn

| Lựa chọn | Thành phần | Tham số | val / test (lỗi >200 mm) | MAE T9+T10/2026 (+3 / +6 / +12 h, chung) | Lỗi >200 mm T9+T10 |
|---|---|---|---|---|---|
| **Model đơn** | `wind_tide30_fc_gru32x3_seq12_cosine_ep77` | 21.291 | 0 / 0 | 31,0 / 40,9 / 43,8 (38,6) | 1 (23/10 03h UTC, +12 h, −215 mm) |
| **Không lỗi** (`ensemble_34k.json`) | trung bình đều của model đơn và `wind_gru32x2_seq25_ep33` | 34.038 | 0 / 0 | 30,5 / 40,1 / 44,2 (38,3) | 0 (sai số lớn nhất 191 mm) |

- MAE T9+T10 tính trên 1.464 mốc phát dự báo, từ 01/09 00:00 đến 31/10 23:00 giờ VN. Đầu vào sau 15/10 dựng lại từ PostgreSQL và từ dữ liệu Open-Meteo tải lại; trên đoạn trùng, kết quả khớp tuyệt đối với dataset gốc.
- `wind_gru32x2_seq25_ep33` **chỉ dùng trong ensemble**. Dùng một mình nó có val 6 / test 4 lỗi, MAE T9+T10 là 44,0.
- So sánh: ensemble lớn (wind h64x3 + wind+tide30 h128x3 + wind+tide30+fc h128x3, khoảng 627 nghìn tham số) có MAE 29,2 / 39,0 / 44,4 (37,5) và 0 lỗi. Persistence (giữ nguyên mực nước) có MAE 114 / 162 / 75.

## Đầu vào mỗi giờ (cửa sổ 12 h cho model đơn, 25 h cho model gió)

| Nhóm | Nguồn | Có trong |
|---|---|---|
| Mực nước, cờ thiếu, `flood_season` | Quan trắc trạm, trung bình mỗi giờ trên khoảng (t−1h, t] | cả hai |
| `wind` (15 kênh) | Gió và áp suất dự báo ECMWF IFS 0,25°, Open-Meteo Previous Runs, bản `_previous_day1` (dự báo từ 24 h trước), điểm 16,75N 107,75E. Đặc trưng tính bằng `src/ai/station_dataset/features.py:wind_features` | cả hai |
| `tide` (5 kênh) | Fit điều hòa 30 ngày trước t, 10 thành phần (`src/ai/station_dataset/features.py:tide_features`) | model đơn |
| `fc_ecmwf` (18 kênh) | Mưa dự báo ECMWF cho 0–3/3–6/6–12/12–24 h tới, dạng log1p, ở các ô Thuận An, A Lưới, Bạch Mã và trung bình 7 ô. Dùng lượt chạy mới nhất khởi tạo trước t−6 h, trong vòng 24 h (Open-Meteo Single Runs, `cell_selection=land`) | model đơn |

**Để chạy trong ứng dụng:** worker hiện chưa thu gió ngoài khơi và chỉ thu dự báo mưa cho ô A Lưới. Cần thêm phần thu gió (bản dự báo từ 24 h trước) và dự báo mưa cho 7 ô, theo đúng quy tắc chọn lượt chạy ở trên. Backend cũng cần hỗ trợ chạy 2 model rồi lấy trung bình mm theo từng mốc giờ.

## Lưu ý

- Mới chỉ train với một seed (17). Kế hoạch là chạy lại với seed 23, chọn epoch theo trainer và ghép bằng trung bình đều.
- Hai model được chọn sau rất nhiều run, đã chấm trên validation và test (15/9–15/10), nên test không còn độc lập. Ensemble được chọn trong 327 tổ hợp, sau khi đã xem cả 16–31/10, nên kết quả có phần lạc quan.
- Biên an toàn mỏng: sai số lớn nhất trên val là 194 mm (model đơn) và 191 mm (ensemble, T9+T10).
- Đợt 23/10 (nước dâng khoảng +200 mm, không có mưa, gió yếu) không có đầu vào nào giải thích được. Mọi model đều dự báo thiếu 150–220 mm.
