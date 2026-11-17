# Ghi chú chuyển session: Thuận An (xong) → Ngã Ba Sình (tiếp theo)

Ngày 12/11/2026. Tài liệu kỹ thuật chi tiết: `models/PIPELINE.md` (quy trình chung) và `models/thuan_an/REPRODUCE.md` (riêng Thuận An).

---

## 1. Trạm Thuận An (0074647402): ĐÃ XONG

### Model đã chọn (`models/thuan_an/`)

| Lựa chọn | Model | Tham số | val / test (lỗi >200 mm) | MAE tháng 9–10/2026 (+3 / +6 / +12 h) | Lỗi >200 mm tháng 9–10 |
|---|---|---|---|---|---|
| Model đơn | `wind_tide30_fc_gru32x3_seq12_cosine_ep77` | 21.291 | 0 / 0 | 31,0 / 40,9 / 43,8 (chung 38,6) | 1 (23/10) |
| Không lỗi | `ensemble_34k.json` = trung bình đều của model đơn và `wind_gru32x2_seq25_ep33` | 34.038 | 0 / 0 | 30,5 / 40,1 / 44,2 (chung 38,3) | 0 |

**Cấu hình model đơn:** GRU residual 32×3, cửa sổ 12 h, lr 0,0001 với warmup-cosine, 200 epoch, batch 64, loss `combined` với `under-weight 0.3`, seed 17. Đầu vào gồm:
- mực nước của trạm;
- triều 30 ngày, 10 thành phần;
- gió ECMWF (bản dự báo từ 24 h trước, Open-Meteo Previous Runs);
- mưa dự báo ECMWF ở 7 ô (Open-Meteo Single Runs).

### Bài học rút ra từ Thuận An
- **Triều 30 ngày** thay cho 7 ngày: tỉ lệ biến động +12 h giải thích được tăng từ 0,45 lên 0,62 (val), từ 0,36 lên 0,59 (test). Cửa sổ 7 ngày không tách được các thành phần triều M2/S2/N2 và O1/Q1.
- **Gió dự báo** xử lý được đợt bão 12/09 trên validation.
- **Mưa dự báo ECMWF** báo hiệu bão sớm hơn gió 3–6 h. Đây là thông tin quý nhất cho +12 h. Ngược lại, **mưa đo** gần như không giúp ở Thuận An.
- `under-weight`: 0,3 tốt nhất trên tháng 9–10. Mức 0,0 kém hơn. Mức 0,5 ổn định nhất trên val/test (94 epoch đạt 0/0) nhưng MAE tháng 9–10 cao hơn.
- Dropout nhóm đầu vào (mưa dự báo và gió) làm validation kém đi. Train quá khoảng 130 epoch cũng kém đi. Ghép model theo trung bình đều giúp ổn định hơn.
- **Đợt 23/10/2026** (nước dâng khoảng +200 mm so với triều, không mưa, gió yếu) không đầu vào nào giải thích được; mọi model đều dự báo thiếu 150–220 mm. Nguồn có thể giúp: dự báo mực nước biển ngoài khơi (Copernicus Marine).
- Mới train với một seed (17). Chạy thêm seed 23 vẫn là việc nên làm.

### Code và dữ liệu đã chuẩn bị (người dùng sẽ commit `src/` và `models/`)
- `src/backend/ingestion/api_backfill.py`: nạp lịch sử quan trắc từ API DCU (Postman HueNgapLut) vào PostgreSQL.
- `src/ai/station_dataset/`:
  - `open_meteo.py`: tải gió Previous Runs và mưa Single Runs;
  - `features.py`: tính đặc trưng gió, triều, mưa dự báo;
  - `build.py`: dựng dataset theo file config JSON; có `--grid-end-utc` và `--normalizers-from`;
  - `evaluate.py`: chấm từng epoch và ensemble.
- Đã kiểm chứng: dữ liệu lấy chỉ từ API trùng 100% với dữ liệu train; dataset dựng lại giống từng byte; train là tất định.
- Database đã đồng bộ đến 12/11/2026.
- Khi train: tối đa 2 run song song; **người dùng tự chạy train**, Claude chỉ đưa lệnh.

---

## 2. Chiều di chuyển của nước (đo bằng dữ liệu 10 phút, 11/2025–11/2026)

```
Ngày thường (sóng triều đi từ biển vào):
  Thuận An ──+40 phút──► Ngã Ba Sình ──+40 phút──► Dã Viên        (r = 0,50 ; 0,94)
Lũ sông lớn (đỉnh lũ đi từ sông ra biển):
  Dã Viên ≈ Ngã Ba Sình (gần như cùng lúc, < 1 h) ──0,5–1,5 ngày──► Thuận An   (ví dụ 17/11/2025: +33 h; 06/10/2026: +23 h)
Sông Bồ:    Tứ Phú → nhập sông Hương tại Ngã Ba Sình, liên hệ yếu (r = 0,25), đỉnh lũ đến không đều
Sông Ô Lâu: Huỳnh Liên → phá Tam Giang, hệ riêng (tương quan chỉ do cùng chịu một đợt mưa bão)
```
- Thay đổi theo giờ trong mùa lũ vẫn xuất hiện ở phía biển trước, vì triều và nước dâng do bão đến từ biển.
- Lần đầu tôi báo "Dã Viên đi trước Ngã Ba Sình 1 h". Đó là **đọc nhầm dấu**, đã đính chính.
- Biên độ triều (độ lệch chuẩn): Thuận An 104 mm, Ngã Ba Sình 62 mm, Dã Viên 65 mm.

## 3. Các phân tích cho Ngã Ba Sình và Dã Viên

Hồi quy tuyến tính, fit trên train. Kết quả ghi theo val / test.

- **Trạm sông khó hơn Thuận An nhiều.** Chỉ với lịch sử mực nước và triều 30 ngày của chính trạm, R² ở +12 h: Ngã Ba Sình 39% / 42%, Dã Viên 35% / 24%, so với Thuận An 65% / 69%. Số lỗi >200 mm ở +6 h: Ngã Ba Sình 18 / 23, Dã Viên 26 / 33, trong khi Thuận An là 0.
- **Mực nước quá khứ của trạm lân cận** (t, −3, −6, −12 h): **không giúp**. Ở +12 h còn kém đi, ví dụ Ngã Ba Sình khi thêm Dã Viên: R² test giảm từ 42% xuống 29%. Lý do: hai trạm dao động gần như cùng lúc, và quan hệ giữa chúng thay đổi theo mùa.
- **Dự báo của model Thuận An** (+3/+6/+12 h) làm đầu vào:
  - +6 h: số lỗi >200 mm trên test giảm từ 33 xuống 14 (Dã Viên) và từ 23 xuống 14 (Ngã Ba Sình).
  - +12 h: Dã Viên R² test tăng từ 24% lên 36%; Ngã Ba Sình gần như không đổi.
- **Vì sao dự báo kém xa "tương lai thật":** model Thuận An đạt R² 0,93 là nhờ phần triều. Trên **phần không phải triều** (nước dâng, lũ, mực nước nền), R² chỉ −0,06 ở +6 h và 0,10 ở +12 h. Mà đây lại chính là phần trạm sông cần: tương quan với phần thật là 0,45, với phần dự báo chỉ 0,17.

## 4. Các ý tưởng đã ghi nhận

| Ý tưởng | Kết luận |
|---|---|
| **Chuỗi model Thuận An → Ngã Ba Sình → Dã Viên** (dự báo của trạm trước làm đầu vào cho trạm sau) | Có ích ở +6 h. Chỉ làm **một bước**. Dự báo dùng cho giai đoạn train phải tạo bằng cách train chéo theo từng đoạn thời gian (out-of-fold), nếu không model hạ nguồn sẽ quá tin vào đầu vào này. |
| **Vòng lặp train lại qua lại** (Ngã Ba Sình ↔ Dã Viên, rồi Dã Viên → Thuận An…) đến khi cả hai tốt nhất | **Không khuyến nghị.** Mô phỏng: chỉ vòng 1 có thay đổi (Dã Viên +3 h R² val 54% → 61%). Từ vòng 2 kết quả đứng yên vì không có thông tin mới (một dự báo chỉ là kết quả tính từ đầu vào của model tạo ra nó). Ở +12 h còn kém đi, và kết quả hội tụ về một model gộp. Rủi ro: quá tin vào dự báo trên dữ liệu train, và chọn nhiều lần trên cùng tập validation. |
| **GAN** | Khác bản chất với vòng lặp trên: GAN có dữ liệu thật mới ở mỗi bước, và vòng lặp của nó tạo ra hàm loss chứ không tạo thông tin. Có thể dùng **loss đối kháng** để dự báo đỉnh lũ sắc nét hơn (giống DGMR của DeepMind). Nhưng nó không giải quyết được đợt kiểu 23/10, dễ báo động giả hơn và khó train. Để thử sau. |
| Dùng Dã Viên cho Thuận An +12 h khi có lũ sông lớn | Đáng thử: đỉnh lũ ở Dã Viên đến trước Thuận An 0,5–1,5 ngày. |
| Dữ liệu xả hồ SOHOA cho trạm sông | Đáng thử: dữ liệu đã có trong database (bảng facility_operations). |
| Dự báo mực nước biển ngoài khơi (Copernicus Marine) cho Thuận An | Đáng thử cho các đợt nước dâng kiểu 23/10. Cần kiểm tra bản lưu trữ có phải đúng bản phát hành tại thời điểm đó. |

## 5. Việc tiếp theo cho Ngã Ba Sình (0074647705, 16,528167N 107,574000E)

1. Tạo `models/nga_ba_sinh/dataset_config.json` theo mẫu của Thuận An: cùng lưới thời gian, cùng cách chia tập, cùng điểm gió và 7 ô dự báo mưa; chỉ đổi mã trạm và tọa độ.
2. Bổ sung vào `src/ai/station_dataset/build.py` các nhóm đầu vào còn thiếu:
   - `rain_obs`: mưa đo theo 3 cụm, lấy từ database. Quy tắc theo giờ: đủ 6 bản ghi thì cộng; 5 bản ghi thì cộng rồi nhân 1,2; ít hơn thì để trống. Các cụm: tây = 750007, 750062; đông = 750041, 750066; đồng bằng = 750033, 567011, 750024, 750046. Biến đổi log1p.
   - `ta_forecast`: dự báo của model Thuận An, giai đoạn train dùng dự báo train chéo.
   - Có thể thêm `reservoir`: dữ liệu xả hồ SOHOA.
3. Trước khi train, đo bằng hồi quy tuyến tính xem nhóm nào giải thích được nhiều nhất (giống cách đã làm cho Thuận An). Sau đó dựng các bản dataset để so sánh có và không có từng nhóm.
4. Train với công thức đã tốt ở Thuận An (GRU 32×3, lr 0,0001, warmup-cosine, under-weight 0,3), rồi chấm tròn tháng 9–10 và cả dữ liệu mới.

---

## 6. Tiến độ Ngã Ba Sình (cập nhật 12/11/2026)

Chi tiết và lệnh tái tạo nằm ở `models/nga_ba_sinh/REPRODUCE.md`.

- **Mục tiêu:** 0 lỗi > 200 mm trên cả val và test, rồi mới chuyển sang trạm khác.
- **Model tốt nhất hiện tại:** GRU 64×3, cửa sổ 12 h, lịch `step`, epoch 66, train trên `nbs_onshore_tide7_fc`. Đầu vào: mực nước, mưa dự báo ECMWF 17 kênh, 4 kênh gió vào bờ, triều 7 ngày. Lỗi > 200 mm (+3 / +6 / +12 h): **val 0/4/3, test 0/0/0**.
- **Lỗi sensor (đã xử lý):** lúc 2026-10-10 15:20 UTC (22:20 giờ Việt Nam), mực nước Ngã Ba Sình tụt 157 mm trong 10 phút, trong khi Thuận An và Dã Viên vẫn lên. Từ đó, chênh lệch với hai trạm này đổi khoảng −172 mm và giữ nguyên đến 12/11, tức sensor đã đổi mốc đo.
  - Cách xử lý: **giữ nguyên số đo thô và loại 13 mẫu test** có giờ nhãn vắt qua lúc đổi mốc (thời điểm phát dự báo 04:00–16:00 UTC ngày 10/10). Khai báo bằng khóa `sensor_steps` trong config; quy tắc chung ở `models/PIPELINE.md`, mục 3.1b.
  - Toàn bộ 4 lỗi test cũ (0/1/3) đều do lỗi này.
  - Nên báo đơn vị quản lý trạm để biết mốc nào đúng.
- **Bài học, khác với Thuận An:**
  - Triều 7 ngày tốt hơn triều 30 ngày.
  - Chỉ dùng 4 kênh gió vào bờ, không dùng đủ 15 kênh.
  - Lịch `step` và cửa sổ 12 h tốt hơn cosine và cửa sổ 25 h.
  - Mưa đo không giúp (đã train thử).
  - Xả hồ chưa thử, vì đã loại theo dữ liệu: trong đợt bão chỉ xả 230–310 m³/s, so với p99 lúc train là 3.872 m³/s.
  - Ensemble chỉ giảm lỗi val xuống 5.
- **Còn lại:** 7 lỗi val, tất cả trong đợt bão 12–13/09 (dự báo thiếu lúc nước lên, dự báo thừa lúc nước rút). Hướng tiếp theo: dùng dự báo của model Thuận An làm đầu vào. Trước tiên chẩn đoán bằng model Thuận An đã đóng gói, chưa cần train; nếu có tín hiệu thì mới train chéo.
