# Trạm Dã Viên (0074647403): ghi chú và ý tưởng

Bắt đầu ngày 17/11/2026. Quy trình chung theo `models/PIPELINE.md`. Bài học từ Ngã Ba Sình xem trong `models/NOTES.md` (phần B) và `models/nga_ba_sinh/REPRODUCE.md`.

## TẠM DỪNG NGHIÊN CỨU (18/11/2026): đọc phần này trước khi tiếp tục

**Mục tiêu:** 0 lỗi > 200 mm (dự báo trung bình giờ) trên mọi giờ ngoài train (val 31/07–14/09, ranh giới, test 15/09–14/10, giai đoạn mới 15/10–12/11/2026) ở +3 / +6 / +12 h. Người hướng dẫn yêu cầu xử lý hết mọi lỗi > 200 mm.

**Đạt tới đâu:**

| Mốc | Tốt nhất | Lỗi > 200 ngoài train | Ở đâu |
|---|---|---|---|
| +1, +2 h | ứng viên `best_h1to6_sn_aug` | 0 | `models/da_vien/best_h1to6_sn_aug/` |
| +3 h | `v1` (aug, ensemble 10 epoch cuối) | **0** (lớn nhất 197 mm) | `models/da_vien/v1/` (Task 25) |
| +6 h | ứng viên `best_h1to6_sn_aug` (+1…+6 h, spring_neap, aug, s23, ensemble 20 checkpoint) | **8**, đều 200–217 mm | Task 32, 36 |
| +6 h (đã đóng gói) | `v1` (gốc u0 s23, ensemble 10 epoch cuối) | 12 (lớn nhất 256) | `models/da_vien/v1/` |
| +12 h | model chung +3 / +6 / +12 h | 37–72 | chưa có ứng viên dùng được |

**Kết luận chính:**
- Ngoài giờ bão / lũ (79% thời gian), +3 h và +6 h đều 0 lỗi. Mọi lỗi +6 h còn lại nằm trong 6 đợt: bão 11–13/09, 21/09, 13/10, 28–29/10, lũ 09–11/11. Kiểu lỗi: nhận ra muộn đợt nước lên, hoặc đánh giá thấp tốc độ rút sau đỉnh (Task 30, biểu đồ quỹ đạo Task 32).
- DV và NBS biến động như nhau ở +6 h (tỷ lệ 1,03); DV khó hơn vì phần riêng của đoạn sông NBS → DV (SD khoảng 55 mm) (Task 18). Ở +12 h, DV biến động hơn NBS 14%.
- **Đã thử và không giải quyết được +6 h:**
  - feature mới: flow, tide_phase, rain_api (Task 19–20); triều 15/30 ngày (Task 28); hồ chứa; Thuận An; Tứ Phú; GloFAS (Task 21); self_fc (Task 34);
  - hậu xử lý: chồng tuyến tính, loss chỉ phạt phần đuôi, hiệu chỉnh sai số trực tuyến, time-lagged ensemble (Task 22, 23, 33, 35);
  - ghép chéo 3.682 tổ hợp run: tốt nhất 8 (Task 31).
- **Có giúp:**
  - augmentation `--augment-surge-scale` (Task 24): +3 h ổn định hơn rõ;
  - dự báo cả quỹ đạo +1…+6 h kèm spring_neap (Task 32): +6 h từ 12 xuống 8–10 lỗi, nhưng chỉ với seed 23 (s42 / s17 kém hơn, Task 36).
- **Giới hạn gốc:**
  - chỉ có 1 năm dữ liệu (DB và API DCU bắt đầu 14/11/2025, Task 30), nên quá ít đợt bão lũ để học;
  - thiếu số đo theo giờ ở thượng lưu sông Hương.
- **Định nghĩa nhãn:** chấm theo mực nước trung bình giờ. Nếu yêu cầu là số đo tức thời lúc hh:00 thì `v1` từ 0 / 12 thành 4 / 19 lỗi (+3 / +6 h) (Task 27). Cần xác nhận với người hướng dẫn.

**Hướng tiếp theo nếu nghiên cứu lại:**
1. Tìm dữ liệu mực nước hoặc lưu lượng theo giờ ở thượng lưu (Kim Long, Thượng Nhật, hạ lưu Bình Điền / Tả Trạch) và các mùa lũ trước 11/2025.
2. Thêm thông tin 10 phút gần nhất (số đo mới nhất, độ dốc 30–60 phút) làm đầu vào (Task 27; chưa sàng lọc).
3. Thành phần loss phạt báo thừa khi nước rút nhanh (Task 36, các lỗi còn lại của epoch tốt nhất).
4. Kiểm tra mọi model trên dữ liệu sau 12/11/2026.

**Dọn dẹp ngày 18/11/2026:**
- Đã xóa: mọi checkpoint từng epoch trong `model/experiments/` (kể cả các run Thuận An và Ngã Ba Sình), MLflow (`mlflow_local.db`, `mlartifacts/`), các dataset DV của thử nghiệm đã đóng, và các file thô lớn trong `reports/`. Các đường dẫn `model/experiments/...` trong notes này chỉ còn là ghi chép.
- Còn giữ:
  - dataset: `dv_water_only`, `dv_h36_all`, `dv_h36_sn`, `dv_tide7_rainobs_nbsobs_linfull_h3`, và mọi config `dataset_config_*.json` (dựng lại dataset bằng `build.py`);
  - model: `models/da_vien/v1`, `h3_v1`, `best_h1to6_sn_aug`;
  - lệnh train của mọi run nằm trong các Task tương ứng.
- Script chấm mọi epoch trên mọi tập: `research/scripts/score_runs_all_splits.py`. Các script ghi "scratchpad" trong notes không còn (logic chính đã có trong `research/scripts/`).

## Tóm tắt trạng thái (cuối ngày 17/11/2026)

**Dữ liệu Dã Viên (xong):**
- Xóa hai đoạn sensor đứng 01–12/02 và 20–25/06 (`invalid_ranges`).
- Lọc 5 gai, ngưỡng 85 mm, giữ sóng thật 11/04.
- Không có bước nhảy mốc đo.
- Mực nước giờ sau khi làm sạch không còn gai.
- Một tập mẫu gốc dùng cho cửa sổ 12/25/48/96 h (`sequence_length` 12, `max_sequence_length` 96).
- Dataset `train_data/da_vien/dv_water_only`; config `models/da_vien/dataset_config_base.json`.

**Phân tích ban đầu:**
- Persistence: test khó nhất ở +3/+6 h (triều tháng 9–10); val gần như chỉ có bão 10–13/09.
- Dự báo NBS giải thích được nhiều cho DV ở +3/+6 h. Mực nước NBS thật giải thích 88–96%.
- Dự báo NBS trên train là trong mẫu. Người dùng quyết định không train lại NBS kiểu out-of-fold; sẽ dùng group dropout.

**Model NBS mới (không station embedding, để finetune sang DV), khoảng 1.900 epoch của 16 run:**
- Không epoch nào đạt 0 lỗi trên mọi giờ (val, test, mới 15/10–12/11, giờ ranh giới).
- Tốt nhất có 1 lỗi sát ngưỡng (200,2–200,6 mm), ví dụ hw 1/1,5/2,5 s42 ep61.
- Ghép +3/+6 h từ ep37 và +12 h từ ep69 của cùng run đó: **0 lỗi trên mọi giờ**.
- **Bập bênh +6 h / +12 h:** các giờ khó nhất bị một mức lệch chung trong GRU kéo cùng nhau. Nguồn gốc là model nén biên độ trên dữ liệu chưa thấy (độ dốc 0,80 / 0,63, trong khi train khoảng 0,9).
- Đã thử và **không giúp**: tail penalty, event weight, batch 128, lr 3e-4, head riêng cho từng mốc, nhánh GRU riêng cho từng mốc. Nhánh GRU riêng cắt được bập bênh nhưng làm mọi mốc kém đi (trung vị 14 lỗi so với 5).
- Model riêng cho từng mốc: người dùng có thể train (lệnh ở cuối cuộc thảo luận ngày 17/11), chưa có kết quả.
- Code mới, mặc định đều tắt:
  - `--station-embedding-size 0`;
  - `--change-tail-weight` / `--change-tail-start-mm`;
  - `--horizon-heads separate`, `--horizon-head-size`, `--horizon-head-layers`, `--horizon-gru-layers`.

**Bước tiếp theo cho Dã Viên:** dựng dataset DV có nhóm đầu vào giống NBS (35 kênh, để finetune), thêm dự báo NBS làm đầu vào kèm group dropout, và thêm chức năng finetune (nạp trọng số NBS) vào `train.py`. Thứ tự do người dùng quyết định.

---

## Task 1. Kiểm tra chất lượng dữ liệu quan trắc (17/11/2026)

**Phạm vi:** toàn bộ số đo 10 phút của Dã Viên trong `hydrology.observations`, từ 14/11/2025 18:50 UTC đến 12/11/2026 03:20 UTC. Gồm 52.167 số đo, phủ train, val, test và giai đoạn mới. Dã Viên được so cùng lúc với Ngã Ba Sình (NBS) và Thuận An (TA).

**Biểu đồ:** `figures/da_vien/data_check.png`.

Mọi thời điểm dưới đây là **giờ Việt Nam**, trừ khi ghi rõ UTC.

### 1.1 Tóm tắt

| # | Hiện tượng | Thời gian | Giai đoạn | Mức độ | Đề xuất |
|---|---|---|---|---|---|
| 1 | **Sensor đứng**: báo đúng 441,0 mm suốt 1.618 số đo liên tiếp | 01/02 10:00 → 12/02 15:30 (11,2 ngày, 270 giờ) | train | **Nặng** | Coi là thiếu dữ liệu |
| 2 | **Sensor đứng**: báo đúng 534,0 mm suốt 732 số đo liên tiếp | 20/06 07:40 → 25/06 09:26 (5,1 ngày, 122 giờ) | train | **Nặng** | Coi là thiếu dữ liệu |
| 3 | Gai 1 số đo +290 mm | 29/03 03:10 | train | Vừa | `despike` 150 mm bắt được |
| 4 | Gai 1 số đo +181 mm | 04/09 10:20 | val | Vừa | `despike` 150 mm bắt được |
| 5 | Gai 1 số đo +388 mm | 26/09 09:50 | test | Vừa | `despike` 150 mm bắt được |
| 6 | Gai 1 số đo +119 mm và +98 mm | 17/08 04:10; 30/06 03:50 | val; train | Nhẹ (mực nước giờ lệch khoảng 20 mm) | Tùy chọn: ngưỡng 100 mm bắt được gai 17/08 |
| 7 | Sóng hạ ngắn lan từ NBS tới DV | 28/02 khoảng 15:00; 11/04 khoảng 08:00 | train | Có vẻ là hiện tượng thật | **Giữ nguyên** |
| 8 | Bước nhảy mốc đo (kiểu NBS 10/10) | không có | — | — | Không cần `sensor_steps` |

Ngoài các điểm trên, dữ liệu Dã Viên **sạch**:
- không có khoảng trống ≥ 1 giờ;
- không có thời điểm trùng lặp;
- không có giá trị âm hay bằng 0;
- độ nhiễu 10 phút thấp nhất trong ba trạm.

### 1.2 Chi tiết

**(1)–(2) Hai đoạn sensor đứng: phát hiện quan trọng nhất**
- Trong hai đoạn này, sensor vẫn gửi đều mỗi 10 phút nhưng giá trị không đổi (số nguyên 441,0 và 534,0). Trong cùng thời gian, NBS dao động theo triều:
  - đoạn tháng 2: 217–688 mm;
  - đoạn tháng 6: 383–572 mm.
  
  Vậy đây là **dữ liệu giả**: thiết bị treo hoặc hệ thống điền giá trị cuối cùng.
- Lúc kết thúc, cả hai đoạn đều có một số đo lệch giờ (12/02 14:52, 25/06 09:26 rồi 09:29, 09:51). Có vẻ thiết bị đã được khởi động lại.
- **Sau khi khởi động lại, mốc đo không đổi.** Chênh lệch DV − NBS (trung vị 3 ngày) trước và sau: 45 → 45 mm (tháng 2), 41 → 38 mm (tháng 6).
- Trạm khác không có đoạn đứng nào: TA và NBS không có chuỗi nào gồm ≥ 6 số đo giống nhau liên tiếp. Ở DV chỉ có thêm một chuỗi ngắn 02/09 23:40–00:30 (6 số đo 326,8 mm, 1 giờ), có thể chỉ là nước đứng yên.
- **Ảnh hưởng:**
  - khoảng 392 giờ mực nước "phẳng" nằm trong tập train, chiếm khoảng 6% số giờ train;
  - với các mẫu này, nhãn "mức thay đổi = 0" là sai, và triều 7 hay 30 ngày fit trên những đoạn đó cũng sai;
  - **các dataset Dã Viên cũ (`hue_regional_20261015_*`, ds1–ds5) đều chứa hai đoạn này như dữ liệu thật** (đã kiểm tra `water.parquet`). Đây có thể là một phần lý do các thử nghiệm trước không thành công.
- **Đề xuất:** thêm một khóa config mới, ví dụ `invalid_ranges: [{start_utc, end_utc, note}]`, để coi các số đo trong khoảng đó là thiếu. Khi đó:
  - mực nước giờ = NaN;
  - quy tắc ≥ 80% giờ có dữ liệu và "nhãn phải có giá trị" sẽ tự loại các mẫu bị ảnh hưởng;
  - database không bị sửa, giống cách làm của `despike`.
  
  Cần sửa `build.py`. **Chưa làm, chờ bạn quyết.** Cũng có thể thêm luật tự phát hiện (≥ 6 số đo liên tiếp giống hệt nhau), nhưng khai báo tường minh thì an toàn hơn.

**(3)–(6) Gai số đo**
- Chạy `build.despike_observations` (cửa sổ 121 phút, giống NBS) với ba ngưỡng:
  - ngưỡng 150 mm: 3 số đo (29/03, 04/09, 26/09);
  - ngưỡng 100 mm: thêm gai 17/08 04:10 (+118 mm);
  - ngưỡng 70 mm: thêm 30/06 03:50 (+91 mm), cùng các điểm 28/02 và 11/04 (là hiện tượng thật, xem (7)).
- Mọi gai đều là **1 số đo** nhảy lên rồi quay lại ngay ở lần đo kế tiếp, trong khi NBS gần như không đổi (độ lệch 0–5 mm).
- Dã Viên có ít gai hơn hẳn NBS: 13 sự kiện ở NBS, so với 3 ở DV theo ngưỡng 150 mm.
- **Đề xuất:** dùng `despike` với ngưỡng 150 mm giống NBS. Ngưỡng 100 mm cũng an toàn vì chỉ bắt thêm đúng 1 gai thật (17/08) và không bắt nhầm điểm nào.

**(7) Sóng hạ ngắn lan truyền**
- Ngày 28/02: NBS tụt khoảng 150 mm lúc 14:40–14:50 rồi hồi lại. DV tụt khoảng 80 mm lúc 15:00–15:10.
- Ngày 11/04: NBS tụt 07:30–07:40. DV tụt 07:50–08:10.
- DV luôn đi sau NBS khoảng 20–40 phút, khớp với độ trễ triều giữa hai trạm (30–40 phút, ổn định cả năm). Vậy đây là một sóng có thật truyền dọc sông, ví dụ do đóng mở công trình hoặc tàu thuyền, không phải lỗi sensor.
- Ở NBS, config đã giữ ca 11/04 (`despike.keep`). Ca 28/02 thì chưa từng được xem xét riêng.
- **Đề xuất:** giữ nguyên, không lọc. Với ngưỡng 150 mm, DV không có điểm nào ở hai ngày này bị bắt.

**(8) Không có bước nhảy mốc đo**
- Cách dò: so trung vị DV − NBS trong 1 h, 6 h và 24 h trước và sau mỗi thời điểm, rồi tìm bước nhảy đột ngột mà vẫn giữ nguyên về sau.
- Các vị trí bắt được:
  - lũ 11/2025, bão 13/09, các đợt mưa tháng 10–11: hiện tượng thủy lực thật, cả hai trạm cùng biến động;
  - **10/10 22:30: +174 mm**, chính là lần NBS đổi mốc đo. Điều này cho thấy cách dò có tác dụng.
- Không có bước nhảy nào do riêng DV.
- Độ trễ giữa DV và NBS (từ tương quan của mức thay đổi theo giờ) giữ ở 30–40 phút suốt các tháng. Không có dấu hiệu đồng hồ bị lệch.

### 1.3 Các đặc điểm thật của dữ liệu (không phải lỗi, nhưng cần biết)

- **Đầu chuỗi:** số đo bắt đầu từ 15/11/2025 01:50, lúc trận lũ lớn 15–18/11 đang lên.
  - Đỉnh cả năm: 3.011,8 mm lúc 17/11 14:10.
  - Cửa sổ lịch sử dài, ví dụ 96 h, sẽ làm mất các mẫu trong trận lũ này (đã thấy ở thử nghiệm 11/11).
  - Trận này chỉ có ở train và mạnh hơn hẳn mọi thứ ở val/test, nên có thể làm lệch bộ chuẩn hóa nhãn +12 h.
- **Chênh lệch DV − NBS theo tháng** (NBS đã bù lại 172 mm sau ngày 10/10 để so sánh):
  - mùa khô (tháng 1–8): 35–48 mm, giảm dần nhẹ khoảng 13 mm;
  - mùa lũ: 56–92 mm (tháng 9–11/2026), 225 mm (tháng 11/2025, lúc lũ).
  
  Đây là độ dốc mặt nước khi sông chảy mạnh, không phải lỗi. Mức nền mùa lũ cao hơn cho thấy dòng chảy sông có vai trò lớn ở DV hơn ở NBS.
- **Mực nước nền theo mùa:** trung vị tháng 8 là 360 mm, so với 506 mm tháng 7. NBS cũng hạ theo cùng cách (471 → 326). Đây là thay đổi chung của cả hệ, có thể do mực nước biển theo mùa. Đây cũng là chỗ dịch chuyển phân bố giữa train và val.
- **Mức thấp nhất năm:** 33 mm lúc 30/09 21:30. Đây là thật: lúc đó NBS xuống −46 mm và triều thấp.
- **Nhỏ, không ảnh hưởng:**
  - 7 số đo lệch khỏi lưới 10 phút (12/02 14:52, 15/04, 19/05, 3 lần ngày 25/06, 05/08);
  - 19 khoảng trống 20–30 phút.
  
  Cả hai đều vẫn nằm đúng giờ khi lấy trung bình theo giờ.

### 1.3b Kiểm tra lại trên mực nước TRUNG BÌNH GIỜ (dữ liệu model thật sự dùng)

Làm ngày 17/11/2026 theo yêu cầu của người dùng. Biểu đồ: `figures/da_vien/hourly_check.png`.
- Mực nước giờ được tính đúng như `build.py`: bỏ hai đoạn sensor đứng, lọc gai 85 mm, giữ ca 11/04, phủ 14/11/2025 → 12/11/2026.
- Ngã Ba Sình dùng config lọc gai của chính trạm đó; Thuận An để thô.

**Chỉ số dùng để dò:** độ "gồ" của mỗi giờ = giá trị giờ đó trừ trung bình của giờ trước và giờ sau. Một gai chỉ kéo dài 1 giờ sẽ hiện thành một vạch nhọn.

**Kết quả: sau khi làm sạch, mực nước giờ của DV không còn gai.**
- Độ gồ của DV rất nhỏ: trung vị 5 mm, p99 31 mm, p99,9 41 mm, nhỏ hơn NBS (p99,9 69 mm). Chỉ 36 trong 8.301 giờ (0,4%) vượt ±35 mm.
- Các giờ vượt ngưỡng nằm ở hai nhóm:
  - **Lũ và bão** (15–18/11/2025, 11–13/09, 09–10/11/2026) và các đợt triều lên mạnh buổi tối tháng 9–11. NBS cũng có cùng lúc, nên đó là thay đổi thật.
  - **Một vài giờ trong mùa khô** (01/01, 20/03, 01/06, 03/06, 05/06, 28/06, 15/07). Mọi ca đều có dao động tương tự ở NBS, sớm hơn khoảng 1 giờ. Ví dụ ca 28/06 06:00 (DV −57 mm/giờ): NBS dao động cùng kiểu lúc 05:00. Vậy đây là sóng hoặc dao động thật, không phải lỗi đo.
- **Đoạn răng cưa 17/08 13–17 h**: theo giờ thì mượt hoàn toàn. Nhiễu 10 phút đã triệt tiêu khi lấy trung bình 6 số đo.
- **Số đo trong mỗi giờ:**
  - 8.287 giờ đủ 6 số đo; 17 giờ có 5 số đo (do khoảng trống 20 phút);
  - 3 giờ có 2–3 số đo: giờ đầu tiên của chuỗi (15/11/2025 02:00) và hai giờ lúc khởi động lại sau sensor đứng (12/02 16:00, 20/06 08:00);
  - không ảnh hưởng đáng kể.

**Hiện tượng lạ đáng chú ý trên dữ liệu giờ (là thật, không phải lỗi):**
1. **Sụt mực nước đột ngột ngoài mùa lũ.** Ở 11/04 và 22/05:
   - NBS có 1 giờ tụt sâu (còn 352 và 312 mm) rồi hồi lại;
   - DV **tụt 60–80 mm trong 1–2 giờ rồi giữ mức thấp mới** (11/04: 541 → 461; 22/05: 490 → 407), không hồi lại.
   
   Đây có thể là tác động của vận hành công trình hoặc một biến cố thủy lực ở hạ lưu (chưa kiểm chứng nguyên nhân). Không đầu vào nào hiện có báo trước được loại sự kiện này. Nếu nó gây lỗi lớn cho model thì cần biết trước.
2. **Lũ lên ở DV nhanh và mạnh hơn NBS rất nhiều:**
   - 17/11/2025 00:00: DV +142 mm/giờ, NBS +5;
   - 09/11/2026 22:00: DV +168, NBS +109;
   - đỉnh 2.989 mm, so với NBS 1.703 mm.
   
   Lũ sông về DV trước và lớn hơn. Vì vậy câu "DV ≈ NBS khi lũ" trong `NOTES.md` A2 chỉ đúng về thời điểm, không đúng về biên độ.

### 1.4 Quyết định

**(a) Hai đoạn sensor đứng: XÓA HOÀN TOÀN khỏi train/val/test** (quyết định của người dùng, 17/11/2026). **Đã làm** ngày 17/11/2026:
- code khóa `invalid_ranges` trong `build.py`, mô tả ở `PIPELINE.md` mục 3.1d;
- khai báo trong `models/da_vien/dataset_config_base.json`.

**Kiểm tra bằng một dataset thử chỉ có mực nước** (trong scratchpad):
- 269 + 121 giờ thành NaN.
- Không còn mẫu nào có mực nước 441,0 hay 534,0.
- Số mẫu: train 5.465, val 1.080, test 723, purged 48, `excluded_invalid_range` 42. Phần lớn các mẫu khác chạm vào hai đoạn đã bị loại sẵn vì thiếu nhãn hoặc thiếu lịch sử.
- Dataset NBS dựng lại vẫn giống từng byte.

Cách làm:
- Thêm khóa config `invalid_ranges: [{start_utc, end_utc, note}]`.
- `build.py` bỏ mọi số đo trong các khoảng này, nên mực nước giờ ở đó là NaN. Database không bị sửa.
- Mọi mẫu có **giờ nhãn** hoặc **cửa sổ lịch sử** chạm vào khoảng này bị gán `split = excluded_invalid_range`, nên không thuộc train, val hay test. Cách này giống `sensor_steps`.
- Cả hai khoảng đều nằm trong train, nên val và test không bị mất mẫu.
- Fit triều vẫn chạy trên phần dữ liệu còn lại của cửa sổ (đã có quy tắc ≥ 70% giờ có dữ liệu, nếu không đủ thì dùng fit toàn cục).

**(b) Gai: người dùng xác nhận 17/08 04:10, 30/06 03:50 và 04/09 10:20 là gai** (17/11/2026). Hai gai lớn 29/03 và 26/09 (Task 1) cũng được lọc.

Config `despike`:
- ngưỡng **85 mm**, cửa sổ 121 phút, `keep` cho ca 11/04;
- bỏ **đúng 5 số đo**: 29/03, 30/06, 17/08, 04/09, 26/09;
- biên an toàn: gai nhỏ nhất bị bắt lệch +91 mm; điểm lớn nhất không bị bắt (và không thuộc `keep`) lệch −78 mm, là sóng thật 28/02.

**Giải thích panel 17/08 15:50** (người dùng hỏi):
- Mỗi chấm là **một số đo 10 phút thô**, không phải trung bình giờ. Đường chấm là trung vị 2 h mà bộ lọc gai dùng.
- Tiêu đề "DV đổi −67 mm trong 10 phút" là hiệu giữa số đo 15:50 và 15:40.
- Từ khoảng 13:00 đến 17:30, số đo DV nhảy lên xuống ±20–67 mm giữa hai lần đo liên tiếp, kiểu răng cưa. NBS trong cùng lúc chỉ đổi ±0–17 mm. Ghi trước đây "NBS cũng dao động" là **sai**: NBS êm hơn nhiều.
- Mực nước nền không đổi (khoảng 240–250 mm), nên đây là **nhiễu đo hoặc sóng mặt nước tại chỗ đặt sensor** (gió chiều, tàu thuyền…), không phải thay đổi mực nước thật.
- Mực nước giờ lấy trung bình 6 số đo nên nhiễu bị làm mượt: 13:00–17:00 là 253, 248, 247, 238, 245 mm, tức chỉ dao động khoảng ±10 mm.
- Không có số đo nào lệch quá 85 mm so với trung vị, nên không bị lọc. Không cần xử lý.

Lịch sử: ban đầu biểu đồ `figures/da_vien/small_spikes.png` (vẽ ngày 17/11/2026) được dùng để chọn ngưỡng.

Biểu đồ gồm 2 gai nhỏ, 1 gai đã bị bắt ở ngưỡng 150 mm (04/09) để so sánh, và 6 bước nhảy 60–70 mm trong 10 phút, mỗi panel ±3 h:
- **17/08 04:10 (+119 mm)** và **30/06 03:50 (+98 mm, lệch trung vị +91)**: đúng dạng gai như 04/09. Chỉ 1 số đo nhảy lên rồi quay lại ngay, các số đo xung quanh phẳng, NBS gần như không đổi (−2 và −4 mm). **Kết luận: là gai.**
- Các bước nhảy 60–70 mm **không phải gai**, vì không quay về mức cũ ngay hoặc NBS cũng biến động:
  - 30/06 18:10 và 10/07 14:00: nước lên thật, NBS cũng lên;
  - 17/08 15:50 và 27/09 16:20: dao động 10 phút ±30–60 mm lặp lại (sóng mặt nước), NBS cũng dao động;
  - **07/04 14:40 và 22/05 14:30**: lại là kiểu "NBS tụt trước, DV tụt sau khoảng 30 phút", giống 28/02 và 11/04 → sóng thật.
- **Nhận xét:** kiểu sóng hạ truyền NBS → DV xuất hiện ít nhất 4 lần: 28/02 14:50, 07/04 14:30, 11/04 07:30, 22/05 14:10 (giờ NBS tụt). Ba lần rơi vào khoảng 14–15 h chiều. Có thể do vận hành công trình ở hạ lưu, chưa rõ nguyên nhân.

Các lựa chọn ngưỡng (cửa sổ 121 phút):

| Ngưỡng | Số đo bị bỏ | Ghi chú |
|---|---|---|
| 150 mm (như NBS) | 3: 29/03, 04/09, 26/09 | sót 2 gai nhỏ |
| 100 mm | 4: thêm 17/08 | sót gai 30/06 (+91) |
| 90 mm + `keep` ca 11/04 | 5: thêm 30/06 | bắt đủ 5 gai; nếu không có `keep` thì bắt nhầm 1 số đo của sóng thật 11/04 (−98) |

---

## Việc còn lại cho phân tích và tiền xử lý (đề xuất ngày 17/11/2026, chưa làm)

1. **Chọn `sequence_length` trong config** (cần làm trước khi dựng dataset). Với 96 h, toàn bộ pha lên và đỉnh trận lũ 11/2025 bị mất. Cần quyết định cửa sổ ngắn hơn, và có giữ trận lũ này trong train không.
2. **Đo độ khó của từng tập bằng persistence**, trước khi train:
   - số lỗi > 200 mm và MAE ở +3/+6/+12 h cho train / val / test;
   - các lỗi đó rơi vào sự kiện nào (lũ, bão, triều, sụt đột ngột kiểu 11/04 và 22/05);
   - so phân bố mức thay đổi giữa các tập.
   
   Mục đích: biết lỗi sẽ đến từ đâu và tập nào khó. Riêng trận lũ 11/2025 có thể làm lệch bộ chuẩn hóa nhãn.
3. **Dành dữ liệu sau 12/11/2026 làm tập kiểm tra độc lập thật.** Không dùng để chọn model, chỉ chấm một lần ở cuối. Rút kinh nghiệm từ NBS, nơi không còn giai đoạn nào chưa dùng để chọn.
4. **Kiểm tra chất lượng từng nguồn đầu vào mới, ngay khi định dùng nó.** Đặc biệt là các nguồn về lượng nước từ thượng nguồn, vì lũ ở DV mạnh hơn NBS rất nhiều:
   - trạm mưa đo trên lưu vực;
   - dữ liệu xả hồ SOHOA (`facility_operations`).
   
   Gió, mưa ECMWF và mực nước biển dùng chung file với NBS, đã kiểm chứng.

## Nhiều cửa sổ lịch sử 12 / 25 / 48 / 96 h (người dùng muốn, 17/11/2026)

**Mục tiêu:** cửa sổ lịch sử không cố định ở một gốc 96 h. Run dùng cửa sổ ngắn sẽ có thêm mẫu trong trận lũ tháng 11/2025.

**Cách làm (không cần sửa code train):**
- Dựng **một** dataset với `sequence_length: 12` (cửa sổ nhỏ nhất) làm tập mẫu gốc.
- Khi train với `--lstm-seq-len L`, `training_selection.select_samples` tự lọc lại: chỉ giữ các mẫu có cửa sổ L giờ đủ ≥ 80% giờ có dữ liệu.

**Mô phỏng** (dataset thử trong scratchpad, `sequence_length` 12):

| `--lstm-seq-len` | train | val | test | mẫu đầu tiên (UTC) | mẫu train trong lũ 15–19/11/2025 |
|---|---|---|---|---|---|
| 12 | 5.717 | 1.080 | 723 | 15/11 04:00 | 116 |
| 25 | 5.690 | 1.080 | 723 | 15/11 17:00 | 103 |
| 48 | 5.629 | 1.080 | 723 | 16/11 16:00 | 80 |
| 96 | 5.505 | 1.080 | 723 | 18/11 16:00 | 32 |

- Val và test **giống hệt nhau** ở mọi cửa sổ (đã so từng thời điểm), nên so sánh giữa các cửa sổ là công bằng.
- Đỉnh lũ (17/11 07:10 UTC) chỉ có trong train khi cửa sổ ≤ 48 h.

**Vấn đề cần sửa trước khi làm:** quy tắc loại mẫu quanh hai đoạn sensor đứng (`invalid_ranges`) đang dùng `sequence_length` của config, tức 12 h. Khi train với cửa sổ dài hơn, vẫn còn mẫu có cửa sổ chạm vào giờ đã xóa:
- 25 h: 10 mẫu;
- 48 h: 18 mẫu;
- 96 h: 38 mẫu, mỗi cửa sổ có tới 19 giờ trống (được điền 0, kèm cờ thiếu dữ liệu).

Như vậy là trái với yêu cầu "xóa hoàn toàn".

**Đề xuất:** thêm khóa config `max_sequence_length` (96). Quy tắc loại mẫu quanh `invalid_ranges` sẽ dùng cửa sổ dài nhất này.

**ĐÃ LÀM** ngày 17/11/2026 (người dùng đồng ý):
- `build.py` có khóa `max_sequence_length`; `dataset_config_base.json` đặt `sequence_length: 12` và `max_sequence_length: 96`.
- Kiểm tra trên dataset dựng lại:

| Cửa sổ | train | val | test | lũ 15–19/11 | mẫu chạm giờ đã xóa |
|---|---|---|---|---|---|
| 12 h | 5.549 | 1.080 | 723 | 116 | 0 |
| 25 h | 5.536 | 1.080 | 723 | 103 | 0 |
| 48 h | 5.513 | 1.080 | 723 | 80 | 0 |
| 96 h | 5.465 | 1.080 | 723 | 32 | 0 |

- Tập mẫu gốc: train 5.549, val 1.080, test 723, purged 48, `excluded_invalid_range` 176.
- Dataset NBS dựng lại vẫn giống từng byte.
- Cái giá: run cửa sổ 12 h mất thêm khoảng 170 mẫu train ngay sau hai đoạn đã xóa, đều trong mùa khô.
- Lũ tháng 11 không bị ảnh hưởng.

**Lưu ý về chuẩn hóa:**
- Với tập mẫu gốc 12 h, trận lũ 11/2025 làm độ lệch chuẩn của nhãn tăng gấp đôi: 224 mm, so với khoảng 114 mm nếu bỏ trận lũ.
- Độ lệch chuẩn của mức thay đổi +12 h: 120 so với 90 mm.
- Mọi run dùng chung bộ chuẩn hóa này. Loss dùng mẫu số max(σ_h, 200), nên ảnh hưởng tới loss nhỏ (224 so với 200). Nhưng đầu vào mực nước bị nén khoảng 2 lần.
- Sẽ xem kỹ hơn ở bước đo độ khó bằng persistence.

## Task 2. Persistence và hồi quy tuyến tính với dự báo của Ngã Ba Sình (17/11/2026)

**Dữ liệu**
- Dataset `train_data/da_vien/dv_water_only`: chỉ có mực nước DV, +3/+6/+12 h, config `dataset_config_base.json` (`sequence_length` 12, `max_sequence_length` 96).
  - Số mẫu: train 5.549, val 1.080, test 723.
  - Dùng 5.536 / 1.080 / 723 mẫu. 13 mẫu đầu tiên bị bỏ vì model NBS +12 h cần cửa sổ 25 h.
- Dự báo NBS lấy từ `models/nga_ba_sinh/combined_v1`, chạy trên dataset NBS dựng lại với cửa sổ 12 h (scratchpad, bộ chuẩn hóa của dataset train NBS). Trên val, kết quả khớp bản gốc (sai khác < 0,0001 mm).
- Đầu vào NBS là **mức thay đổi dự báo** (dự báo − mực nước NBS hiện tại), nên không bị ảnh hưởng bởi lần NBS đổi mốc đo.
- Hồi quy tuyến tính (OLS) riêng cho từng mốc, nhãn là mức thay đổi của DV. Đầu vào "DV lịch sử 12h" gồm mực nước hiện tại và 12 hiệu DV(t) − DV(t−k).
- Script: scratchpad `dv/linear_nbs.py`, bị xóa khi sang session mới.

**Kết quả: fit trên train BỎ lũ 15–19/11/2025.** Fit có cả trận lũ thì kém hơn ở val +12 h (R² 2%), vì trận lũ chi phối OLS.

R² của mức thay đổi DV, tức phần trăm biến động được giải thích (val | test, +3 / +6 / +12 h):

| Đầu vào | val | test |
|---|---|---|
| persistence | 0 / 0 / 0 | 0 / 0 / 0 |
| DV lịch sử 12h | 47 / 27 / 14 | 81 / 73 / 19 |
| chỉ NBS dự báo (3 kênh) | 44 / 35 / 15 | 83 / 82 / 27 |
| DV lịch sử + NBS dự báo | 53 / 36 / 13 | 86 / 83 / 31 |
| DV lịch sử + NBS **thật** (giới hạn trên, không dùng được khi chạy thật) | 89 / 93 / 93 | 95 / 96 / 88 |

Số lỗi > 200 mm (+3 / +6 / +12 h):

| Đầu vào | train | val | test |
|---|---|---|---|
| persistence | 95 / 358 / 280 | 16 / 45 / 29 | 79 / 190 / 30 |
| DV lịch sử 12h | 25 / 131 / 127 | 4 / 22 / 35 | 0 / 27 / 21 |
| chỉ NBS dự báo | 17 / 38 / 62 | 0 / 12 / 17 | 0 / 8 / 16 |
| **DV lịch sử + NBS dự báo** | 17 / 35 / 58 | **0 / 9 / 22** | **0 / 3 / 13** |
| DV lịch sử + NBS thật | 11 / 19 / 43 | 0 / 0 / 3 | 0 / 0 / 0 |

MAE (mm), DV lịch sử + NBS dự báo: val 30 / 57 / 71, test 37 / 57 / 61. Persistence: val 37 / 61 / 72, test 102 / 147 / 75.

**Persistence: tập nào khó, lỗi đến từ đâu**
- **Test khó nhất ở +3/+6 h:** 79 / 190 lỗi.
  - Triều tháng 9–10 rất mạnh: |Δ6h| p95 là 352 mm ở test, so với 170 mm ở val. Lỗi trải đều trên nhiều ngày (15/09, 27–30/09, 09–12/10).
- **Val:** gần như toàn bộ lỗi nằm trong bão 10–13/09.
- **Train:** lỗi tập trung ở lũ 16–18/11/2025 và các đợt triều mạnh tháng 12–1.

**Kết luận**
1. **DV bám rất sát NBS.** Nếu biết trước mực nước thật của NBS thì giải thích được 88–96% biến động của DV và gần như hết lỗi lớn. Vậy giới hạn nằm ở **chất lượng dự báo NBS**, nhất là +12 h.
2. **Dự báo NBS giảm mạnh lỗi lớn ở +3/+6 h.** Lỗi +6 h còn 9 ở val (so với 22 khi chỉ dùng lịch sử DV) và 3 ở test (so với 27). Đây là bước tiến rõ nhất.
3. **Ở +12 h, dự báo NBS giúp ít.**
   - R² val gần như không đổi (14 → 13), test tăng 19 → 31.
   - Lỗi > 200 mm: val 35 → 22, test 21 → 13.
   - Lỗi còn lại ở val: bão 12–13/09 (dự báo thiếu khi nước lên, thừa khi nước rút). Ở test: 11–13/10.
   - Bản thân model NBS +12 h cũng chỉ đạt R² 18% (val) và 49% (test) trên chính NBS.
4. **Cảnh báo, kết quả đang lạc quan:**
   - Trên train, dự báo NBS là **trong mẫu**, vì model NBS được train trên chính giai đoạn này. R² của model NBS trên NBS: train 78 / 80 / 73, val chỉ 41 / 37 / 18. Hồi quy (hay GRU sau này) học trên train sẽ tin dự báo NBS hơn mức thật.
   - Model NBS lại được chọn sau khi xem cả val và test của NBS.
   - Muốn dùng làm đầu vào đúng cách thì phải tạo dự báo NBS **train chéo theo từng đoạn thời gian** (out-of-fold) cho tập train (xem `NOTES.md` A4). Việc này cần train lại model NBS nhiều lần.

## Task 3. Hồi quy tuyến tính riêng cho từng mốc, chỉ dùng dự báo NBS cùng mốc (17/11/2026)

**Cách làm:** giống Task 2 (OLS, nhãn là mức thay đổi DV, cùng 5.536 / 1.080 / 723 mẫu và thêm 48 giờ ranh giới). Khác ở chỗ model +h chỉ nhận dự báo NBS +h, không nhận hai mốc còn lại.
- A: DV lịch sử 12h (mực nước hiện tại và 12 hiệu DV(t) − DV(t−k)).
- B: A + mức thay đổi dự báo NBS (`combined_v1`) **cùng mốc**.
- Script: scratchpad `dv_linear_per_h.py`, bị xóa khi sang session mới. Dùng lại dự báo NBS của Task 2.

**Số lỗi > 200 mm, fit trên train bỏ lũ 11/2025 (+3 / +6 / +12 h):**

| Model | train | val | ranh giới (48) | test |
|---|---|---|---|---|
| persistence | 95 / 358 / 280 | 16 / 45 / 29 | 12 / 15 / 1 | 79 / 190 / 30 |
| A: DV lịch sử | 25 / 131 / 127 | 4 / 22 / 35 | 0 / 1 / 1 | 0 / 27 / 21 |
| **B: + NBS cùng mốc** | 16 / 37 / 59 | **0 / 14 / 21** | **0 / 0 / 0** | **0 / 11 / 12** |
| *Task 2: + NBS cả 3 mốc* | 17 / 35 / 58 | 0 / 9 / 22 | chưa chấm | 0 / 3 / 13 |

- MAE của B: val 30 / 57 / 69, test 38 / 63 / 62 mm. R² của B: val 52 / 37 / 16, test 85 / 80 / 29.
- Fit có cả lũ 11/2025: B được val 0 / 12 / 28, test 0 / 12 / 15, ranh giới 0 / 2 / 0. Tốt hơn ở val +6 h, kém hơn ở +12 h.
- Hệ số của kênh dự báo NBS (bỏ lũ): 0,83 / 0,97 / 1,05. DV đi theo NBS gần 1:1.

**Nhận xét**
1. **+3 h:** chỉ cần dự báo NBS +3 h là đủ, 0 lỗi ở val, test và ranh giới.
2. **+6 h:** chỉ dùng dự báo NBS cùng mốc thì **kém hơn** dùng cả 3 mốc: val 14 so với 9, test 11 so với 3. Dự báo NBS +3 h và +12 h có thêm thông tin cho +6 h, ví dụ về hình dạng sóng triều.
3. **+12 h:** gần như bằng khi dùng cả 3 mốc: val 21 so với 22, test 12 so với 13. Ở +12 h, các mốc ngắn không giúp gì thêm.
4. **Lỗi còn lại:** chủ yếu do **dự báo NBS sai**, không phải do quan hệ NBS→DV.
   - Ví dụ +12 h ngày 12/09 15–19 h: DV thật lên +317 đến +453 mm, nhưng NBS chỉ dự báo +21 đến +30 mm.
   - Ở +6 h, khi nước lên mạnh, model dự báo thiếu khoảng 200–240 mm: bão 11–12/09 ở val; 27–28/09, 09/10 ở test.
5. Dự báo NBS trên train vẫn là trong mẫu (xem cảnh báo ở Task 2), nên kết quả này còn lạc quan.

**Bổ sung: model +6 h với các tổ hợp mốc dự báo NBS** (fit bỏ lũ; số lỗi > 200 mm / MAE mm; val | ranh giới | test)

| Dự báo NBS | val | ranh giới | test |
|---|---|---|---|
| +6 | 14 / 57 | 0 / 64 | 11 / 63 |
| +3, +6 | 13 / 57 | 0 / 62 | 3 / 58 |
| +6, +12 | 14 / 57 | 0 / 64 | 11 / 63 |
| +3, +6, +12 | 9 / 57 | 0 / 61 | 3 / 57 |
| +3, +12 (không có +6) | 8 / 58 | 0 / 61 | 3 / 56 |

- Phần lớn mức cải thiện đến từ **dự báo NBS +3 h**: test giảm từ 11 xuống 3 lỗi. Riêng +12 h không giúp gì; chỉ khi đi cùng +3 h mới giảm thêm ở val (13 → 9).
- Bỏ hẳn dự báo +6 h không làm kết quả kém đi.
- Tổ hợp được chọn sau khi đã xem val và test, nên các khác biệt nhỏ (9 so với 8) không có ý nghĩa.

**Model +6 h nhận dự báo +3 h của chính DV (model B +3 h) thay cho dự báo NBS +3 h** (số lỗi > 200 mm / MAE mm; val | ranh giới | test)

| Đầu vào | val | ranh giới | test |
|---|---|---|---|
| DV lịch sử + NBS fc+6 | 14 / 56,7 | 0 / 64,0 | 11 / 63,4 |
| DV lịch sử + **DV dự báo +3** + NBS fc+6 | 13 / 57,0 | 0 / 61,9 | 3 / 57,8 |
| DV lịch sử + NBS fc+3 + NBS fc+6 | 13 / 57,0 | 0 / 61,9 | 3 / 57,8 |
| DV dự báo +3 + NBS fc+6 (bỏ DV lịch sử) | 13 / 56,1 | 0 / 63,4 | 7 / 59,8 |
| DV lịch sử + DV dự báo +3 + NBS fc+6, fc+12 | 9 / 57,5 | 0 / 61,4 | 3 / 57,3 |

- So với chỉ dùng NBS fc+6, thêm dự báo +3 h của DV có cải thiện: test 11 → 3, val 14 → 13.
- Nhưng kết quả **trùng hoàn toàn** với khi dùng dự báo NBS +3 h (sai khác 2·10⁻¹² mm). Lý do: model B +3 h là tổ hợp tuyến tính của DV lịch sử và NBS fc+3, nên khi đã có DV lịch sử, thêm đầu ra của nó cũng chỉ là thêm NBS fc+3. Với hồi quy tuyến tính, cách ghép này không mang thêm thông tin.
- Cách ghép này chỉ có thể khác đi nếu model +3 h là phi tuyến (GRU), hoặc nếu model +6 h không có DV lịch sử (dòng thứ tư: test 7, kém hơn).

**Model +3 h chỉ dùng DV lịch sử: có đạt 0 lỗi không?** (số lỗi > 200 mm; train | val | ranh giới | test)

| Cửa sổ | fit bỏ lũ | fit đủ |
|---|---|---|
| 12 h | 25 \| 4 \| 0 \| 0 | 17 \| 2 \| 0 \| 0 |
| 25 h | 36 \| 3 \| 0 \| 0 | 19 \| 2 \| 0 \| 0 |
| 48 h | 24 \| 4 \| 0 \| 0 | 10 \| 2 \| 0 \| 0 |
| 96 h | 3 \| 4 \| 0 \| 0 | 0 \| 4 \| 0 \| 0 |

- Test và ranh giới luôn 0 lỗi. Val không lần nào về 0, ít nhất còn 2 lỗi, đều trong bão 12–13/09.
- **Lỗi ở 12/09 19–20 h:** DV vẫn đang rút (−35, −11 mm/h, mực nước 693 → 682 mm), rồi lên +321 mm trong 3 h tiếp theo. Trong lịch sử chưa có dấu hiệu nào của đợt lên này.
- **Lỗi ở 13/09 00–01 h:** DV vừa lên +304 mm trong 3 h. Model dự báo nước sẽ chậm lại như một con triều bình thường, nhưng bão đẩy tiếp lên 1.047 → 1.232 mm. Model dự báo thiếu 250–284 mm.
- Thêm dự báo NBS +3 h (model B) thì val về 0. Thông tin cần cho các giờ này đến từ bên ngoài (đầu vào khí tượng và biển trong model NBS), không có trong mực nước DV.
- Chưa thử model phi tuyến chỉ dùng lịch sử. Có thể model học được kiểu "mực nước nền cao bất thường thì biên độ lên lớn hơn", nhưng train chỉ có một trận lũ (11/2025) nên khó kỳ vọng.

**Model +3 h: thêm từng nhóm đầu vào của NBS** (script scratchpad `dv_h3_groups.py`; giá trị tại thời điểm dự báo, không lấy chuỗi). Các mẫu thiếu dự báo mưa ECMWF bị bỏ, trong đó có trận lũ 11/2025, nên fit bỏ lũ và fit đủ cho cùng kết quả. Còn 5.362 mẫu train.

| Đầu vào | val n200 / MAE | test n200 / MAE | lỗi tại 12/09 19 h, 20 h, 13/09 00 h, 01 h |
|---|---|---|---|
| A: DV lịch sử | 4 / 32 | 0 / 43 | −217 −265 −303 −273 |
| A + NBS quan trắc 12 h | **2** / 28 | 0 / 33 | −218 −286 −197 −175 |
| A + triều | 5 / 26 | 0 / 36 | −204 −234 −249 −256 |
| A + biển | 3 / 31 | 0 / 40 | −197 −250 −297 −269 |
| A + gió | 4 / 32 | 0 / 43 | gần như không đổi |
| A + mưa ECMWF | 4 / 32 | 0 / 43 | gần như không đổi |
| A + NBS quan trắc + triều + biển | 4 / 24 | 0 / 28 | −187 −242 −192 −191 |
| A + cả 5 nhóm | 2 / 25 | 0 / 28 | −184 −237 −190 −190 |
| B: A + NBS fc+3 | **0** / 31 | 0 / 38 | −157 −197 −168 −191 |

- **Không có nhóm nào tự nó đưa về 0.** Với tuyến tính, gộp cả 5 nhóm vẫn còn 2 lỗi.
- **NBS fc+3 cũng chỉ vừa đủ:** lỗi lớn nhất ở val là 197 mm. Ở các giờ khó, nó chỉ giảm lỗi được 60–110 mm.
- **NBS quan trắc** giúp nhiều nhất trong 5 nhóm (MAE test 43 → 33) và sửa được 13/09 00–01 h, nhưng không sửa được 12/09 20 h. Nhóm này có sẵn khi chạy thật.
- **sea_anom** trong bão rất lớn (285–547 mm, trong khi p95 trên train là 294 mm). Đây là tín hiệu bão rõ, nhưng model tuyến tính không dùng được vì train gần như không có giá trị cỡ này.
- **Gió và mưa:** với tuyến tính, gần như không có tác dụng ở +3 h.
- **Kết luận:** NBS fc+3 tốt hơn là nhờ GRU kết hợp phi tuyến cả chuỗi 12 h của các nhóm, không phải nhờ một nhóm cụ thể. Muốn biết chính xác thì cần làm ablation trên chính GRU NBS (chưa làm).

**MAE +3 h: DV (model B) so với NBS (`combined_v1`), trên cùng các giờ** (bỏ 3 giờ có nhãn +3 h vắt qua bước nhảy mốc đo NBS 10/10)

| Tập | MAE DV | MAE NBS | persistence DV | persistence NBS |
|---|---|---|---|---|
| val | 30,3 | 32,7 | 37,3 | 34,3 |
| ranh giới | 43,3 | 44,9 | 96,9 | 92,3 |
| test | 37,8 | 37,6 | 101,6 | 102,7 |

Ở +3 h, MAE của hai trạm gần như bằng nhau; ở val, DV thấp hơn khoảng 2 mm. Mức dao động +3 h của hai trạm cũng tương đương (xem cột persistence). Cả hai đều 0 lỗi > 200 mm.

## Task 4. GRU chỉ dự báo +3 h: mực nước DV + triều (17/11/2026)

**Yêu cầu của người dùng:** model GRU chỉ có **một đầu ra +3 h**, không bị ảnh hưởng bởi các mốc khác, kể cả khi các mốc đó không được dùng.

**Cách đảm bảo**
- `--target-horizons 3`: `output_size` = 1, nên head chỉ có 1 đầu ra; loss và bộ chuẩn hóa nhãn chỉ tính trên +3 h.
- Dataset mới chỉ có `horizons_hours: [3]`. Mẫu chỉ cần nhãn +3 h có giá trị, không bị loại vì thiếu nhãn +6/+12 h. Quy tắc xóa mẫu quanh `invalid_ranges` cũng chỉ tính tới t + 3 h.

**Dataset `train_data/da_vien/dv_tide7_h3`** (config `models/da_vien/dataset_config_tide7_h3.json` = base + `tide.window_days` 7 + `horizons_hours` [3]; `--groups tide`):
- `water.parquet` giống từng byte với `dv_water_only`.
- Số mẫu: train 5.551 (thêm 2 mẫu, trước bị loại vì nhãn +12 h chạm đoạn sensor đứng), val 1.080, test 723, ranh giới 48.
- Triều fit trên chính mực nước DV đã làm sạch: `tide_now`, `tide_d3`, `tide_d6`, `tide_d12`, `tide_missing_frac`. Hai kênh `tide_d6`/`tide_d12` là thông tin triều thiên văn biết trước, chỉ là đầu vào, không phải đầu ra.
- Tương quan tide_d3 với Δ3 thật của DV: train 0,68, val 0,68, test 0,88. 3,6% giờ train dùng fit triều dự phòng (sau hai đoạn sensor đứng); val và test 0%.

**Lệnh train** (người dùng chạy; công thức giống model short của NBS, không station embedding):
```bash
.venv/bin/python -m src.ai.train --mlflow-ex-name da_vien --horizon 1h \
  --dataset-dir train_data/da_vien/dv_tide7_h3 --target-horizons 3 \
  --rain-stations all --use-weather-forecast false --feature-set water_level+rain+flood_season \
  --train-method gru --head-mode shared --station-embedding-size 0 --lstm-seq-len 12 --lstm-hidden-size 64 --lstm-num-layers 3 \
  --lstm-drop-out 0.2 --lstm-input-dropout 0.1 --lstm-head-dropout 0.1 --lstm-lr 0.0001 --lstm-weight-decay 0.001 \
  --lr-scheduler step --batch-size 64 --num-epochs 100 --early-stopping 100 \
  --change-loss combined --change-under-weight 0.3 --change-event-weight 0 --change-margin-mm 0 \
  --lstm-checkpoint-metric val_loss --skip-test --seed 42 --diagnose-change-dynamics \
  --checkpoint-dir model/experiments/da_vien/dv_tide7_h3/noemb_b64_lr0.0001_wd0.001_gru_h64x3_do0.2-0.1-0.1_seq12_e100_combinedu0.3_ev0_step_s42
```

**Mốc so sánh (tuyến tính +3 h, Task 3):** DV lịch sử: val 4 lỗi, test 0; + triều: val 5, test 0 (MAE val 26, test 36); + NBS fc+3: val 0.

### Kết quả (người dùng train; Claude chấm 100 epoch ngày 17/11/2026)

- **Chấm điểm:** script scratchpad `score_dv.py`. Có thêm giai đoạn mới 15/10–12/11 (668 giờ), dựng bằng `--grid-end-utc 2026-11-12T03:00Z --normalizers-from train_data/da_vien/dv_tide7_h3`. Phần trùng thời gian giống hệt dataset train.
- **Số lỗi > 200 mm, bỏ epoch 1:**
  - val 4–6, ranh giới 0, test 0–1, mới 6–9, train 15–21;
  - tổng ngoài train: nhỏ nhất 10 (ep6), trung vị 12;
  - MAE: val 27–28, test 36–37 (giống tuyến tính + triều), mới 37–39 mm.
- **Lỗi còn lại nằm ở hai trận và đều là dự báo thiếu khi nước bắt đầu lên mạnh:**
  - **Bão 11–13/09 (val):**
    - 12/09 08–09 h: Δ3 thật +190 đến +285 mm, dự báo khoảng 0;
    - 12/09 19–20 h: Δ3 thật +199 đến +321 mm, dự báo −56 đến +33 mm; sai ở 99/99 epoch;
    - 13/09 01–02 h: khoảng một nửa số epoch sai.
  - **Lũ 09–11/11 (giai đoạn mới):** DV lên từ 362 mm (09/11 08 h) đến 1.356 mm (11/11 04 h).
    - 09/11 08–10 h và 18–20 h: Δ3 thật +250 đến +397 mm, dự báo −72 đến +150 mm, lỗi tới −390 mm;
    - 10/11 11 h: sai ở 96 epoch;
    - 11/11 18 h: dự báo thừa khi nước rút, sai ở 35 epoch.
  - **Test 13/10 09 h:** sai ở 32 epoch, lỗi khoảng −198 mm, sát ngưỡng.
- Mọi lỗi khác chỉ xuất hiện ở epoch 1–2.
- **Các giờ lỗi thiếu thông tin gì** (script scratchpad `why_errors.py`; 15 giờ sai ở ≥ 30 epoch; mưa đo theo cụm đông Bạch Mã + Đỉnh Bạch Mã, tây Hồng Thái + Nhâm, đồng bằng Phú Bài + VP Huế + Bình Điền):
  - **Kiểu 1, 12/15 giờ: nước lên mạnh từ mức thấp hoặc đang rút** (12/09 08–09 h và 19–20 h, 13/10 09 h, 09/11 08–10 h và 18–20 h).
    - Cả hạ lưu lên cùng lúc: trong 3 h tới NBS lên +143…+332 mm, Thuận An +164…+270 mm, DV +152…+397 mm. Vậy nước lên từ phía biển (triều cường cộng nước dâng), không phải lũ từ thượng nguồn về.
    - **Triều fit trên DV báo sai:** tide_d3 chỉ −124…+167 mm, nhiều giờ còn báo xuống (12/09 19 h −96; 09/11 18 h −124). Trên train, ở các giờ DV lên > 200 mm, tide_d3 trung vị +178 mm, nghĩa là bình thường triều báo được.
    - Mực nước biển lệch rất cao: sea_anom 158–463 mm, trong khi p95 train là 261 mm. Có nước dâng do bão.
    - NBS quan trắc chỉ báo trước được ở vài giờ (09/11 09–10 h và 19 h: NBS đã lên +79…+121 mm/h, trên p95 train 44). Ở các giờ còn lại, NBS cũng chưa lên.
  - **Kiểu 2, 3/15 giờ: nước vẫn lên khi đã ở mức cao, trong khi phía biển đang xuống** (13/09 01–02 h, 10/11 11 h): Thuận An −143…−199 mm, mực biển dự báo −180…−440 mm.
    - Đây là **lũ từ thượng nguồn**: mưa đo 24 h ở cụm tây 392–490 mm và cụm đông 491–505 mm, **vượt mức lớn nhất trên train** (tây 567 mm nhưng chỉ một trận; đông 410 mm).
  - **11/11 18 h:** dự báo thừa khi nước rút.
  - **Lưu ý:**
    - Mực nước biển dự báo (sea_p3) dẫn trước DV khoảng 3 h: tương quan với Δ3 của DV là 0,54 khi lệch 3 h, gần 0 khi không lệch. Muốn dùng thì GRU phải tự học độ lệch này.
    - Mưa dự báo ECMWF chưa tải cho giai đoạn sau 15/10.
- **Triều giải thích được bao nhiêu** (triều fit trên DV; R² % của mức thay đổi DV; "hiệu chỉnh" = a·tide_d + b, fit trên train bỏ lũ 11/2025; train ở đây gồm cả lũ):

| Cửa sổ | Mốc | Triều dùng thẳng: train / val / test / mới | Triều hiệu chỉnh: train / val / test / mới |
|---|---|---|---|
| **7 ngày (đang dùng)** | +3 h | 37 / 29 / 74 / 61 | **46 / 45 / 76 / 66** |
| | +6 h | 42 / 49 / 83 / 75 | 47 / 55 / 82 / 76 |
| | +12 h | 1 / 33 / 3 / 35 | 12 / 31 / 22 / 34 |
| 15 ngày | +3 h | 37 / 40 / 79 / 68 | 43 / 38 / 78 / 70 |
| 30 ngày | +3 h | 31 / 36 / 69 / 82 | 38 / 29 / 59 / 75 |

  - Hệ số hiệu chỉnh khoảng 0,7: biên độ triều fit được lớn hơn biên độ thật của Δ3.
  - Giờ dùng fit dự phòng (train): 7 ngày 3,6%, 15 ngày 8,4%, 30 ngày 10,2%.
  - Không cửa sổ nào sửa được các giờ lỗi. Tại 8 giờ lỗi kiểu 1 (Δ3 thật +152…+397 mm), tide_d3 với 7 ngày là −124…+123, 15 ngày −60…+134, 30 ngày −33…+137 mm.
- **Mực nước biển giải thích thêm được bao nhiêu** (script scratchpad `sea_r2.py`; OLS fit trên train bỏ lũ; biển là nhóm `sea` của NBS (7 kênh tại t), thêm 6 kênh thay đổi 1–6 h trước của mực biển; mẫu train từ 18/11/2025 vì trước đó chưa có dữ liệu biển). R² % (val / test / mới) và số lỗi > 200 mm:

| Mốc | DV lịch sử + triều | + biển | Thêm được |
|---|---|---|---|
| +3 h | 59 / 87 / 79; lỗi 4 / 0 / 7 | 61 / 88 / 82; lỗi 4 / 0 / 5 | +2 / +1 / +3 điểm |
| +6 h | 47 / 84 / 72; lỗi 19 / 6 / 38 | 52 / 86 / 78; lỗi 13 / 3 / 30 | +5 / +2 / +6 |
| +12 h | 14 / 14 / 2; lỗi 26 / 19 / 65 | 26 / 34 / 25; lỗi 22 / 12 / 57 | +12 / +20 / +23 |

  - So với chỉ có triều, biển thêm 4–7 điểm ở +3 h (val 44 → 50, mới 67 → 74).
  - Biển một mình chỉ giải thích 16–35% ở +3 h.
  - Các kênh thay đổi 1–6 h trước gần như không thêm gì so với kênh tại t.
  - **Kết luận (tuyến tính):** ở +3 h, biển thêm rất ít và không giảm lỗi ở val; mực biển hữu ích hơn nhiều cho +12 h. GRU có thể dùng biển phi tuyến (chỉ khi có bão) tốt hơn, nhưng chưa thử.
- **Gió, mưa đo, mưa dự báo ECMWF giải thích thêm được bao nhiêu** (script scratchpad `groups_r2.py`). Cách làm:
  - OLS fit trên train bỏ lũ. Mọi model dùng cùng tập giờ: train 5.364, val 1.080, test 723, mới 668.
  - Gió: file `prev_offshore_to_20261116.json`.
  - Mưa đo: 3 cụm (đông Bạch Mã + Đỉnh Bạch Mã, tây Hồng Thái + Nhâm, đồng bằng Phú Bài + VP Huế + Bình Điền), tổng 1/3/6/12/24/48 h.
  - ECMWF: thư mục `ecmwf_single_runs_soil` (có tới 16/11). Đã kiểm tra: giống hệt `ecmwf_single_runs` trên toàn bộ giai đoạn dataset.

  **+3 h**, R² % (val / test / mới) và số lỗi > 200 mm (val / test / mới):

| Đầu vào | R² | Lỗi |
|---|---|---|
| DV lịch sử + triều | 59 / 87 / 79 | 4 / 0 / 7 |
| + biển | 61 / 88 / 82 | 4 / 0 / 5 |
| + gió (4 kênh vào bờ như NBS) | 61 / 87 / 79 | 4 / 0 / 7 |
| + gió (tất cả kênh) | **45** / 87 / 80 | 4 / 0 / 5 |
| + mưa đo 3 cụm (log1p) | 62 / 88 / 82 | 2 / 0 / 5 |
| + mưa đo 3 cụm (mm) | **65** / 88 / 82 | **0 / 0 / 3** |
| + ECMWF (log1p) | 59 / 87 / 79 | 4 / 0 / 7 |
| + ECMWF (mm) | 63 / 86 / 79 | 2 / 0 / 5 |
| + biển + gió 4 + mưa đo + ECMWF (log1p) | 58 / 88 / 83 | 2 / 0 / 4 |

  - Lỗi ở +6 h (val / test / mới): nền 19 / 6 / 38. Thêm mưa đo: 5–9 / 1–2 / 27–29. Thêm biển: 13 / 3 / 31. Thêm ECMWF (mm): 6 / 6 / 32. Thêm gió 4 kênh: 17 / 7 / 37.
  - Lỗi ở +12 h: mỗi nhóm thêm 6–23 điểm R², nhưng lỗi chỉ giảm ít. Gộp cả 4 nhóm: test 19 → 9, mới 65 → 49.
  - **Mưa đo + ECMWF cùng lúc**, R² val / test / mới (số lỗi):

| Mốc | Chỉ mưa đo (mm) | Chỉ mưa đo (log1p) | Mưa đo + ECMWF, tốt nhất trong 4 cách kết hợp mm/log1p |
|---|---|---|---|
| +3 h | 65 / 88 / 82 (0 / 0 / 3) | 62 / 88 / 82 (2 / 0 / 5) | mm + log1p: 63 / 88 / 82 (0 / 0 / 3); mm + mm: 65 / 87 / 80 (0 / 0 / 5) |
| +6 h | 56 / 87 / 78 (5 / 2 / 29) | 53 / 87 / 78 (9 / 1 / 27) | mm + log1p: 53 / 87 / 78 (6 / 0 / 31); log1p + mm: 55 / 86 / 77 (6 / 1 / 25) |
| +12 h | 31 / 24 / 6 (27 / 14 / 62) | 26 / 23 / 19 (26 / 18 / 61) | log1p + log1p: 28 / 28 / 25 (24 / 16 / 60) |

    → Khi đã có mưa đo, ECMWF gần như không thêm gì: ở +3 h tăng 0–1 điểm, có cách kết hợp còn kém hơn; ở +6 h test lỗi 2 → 0–1; ở +12 h tăng 2–6 điểm. Hai nhóm phần lớn trùng thông tin (với model tuyến tính).
  - **Mưa đo giúp nhiều nhất ở +3 h và +6 h.** Ở +3 h, bản mm đưa val về 0 lỗi, mới 7 → 3. Đây là thông tin sẵn có lúc dự báo, khớp với phát hiện cũ (11/11): các giờ nước lên ở val đã có ≥ 100 mm mưa đông đo được từ trước.
  - **ECMWF:** ở +3 h chỉ có tác dụng khi để đơn vị mm (val +4 điểm, lỗi 4 → 2).
  - **Gió:** gần như 0 ở +3 h. Dùng tất cả kênh gió làm val tụt mạnh (59 → 45 ở +3 h, −53 ở +12 h), tức là overfit.
  - **Lưu ý:**
    - Kết quả tuyến tính. Chọn mm hay log1p sau khi đã xem val và test, nên khác biệt nhỏ không đáng tin.
    - Train có rất ít trận mưa lớn, nên GRU có thể không học được giống vậy.
- **Kết luận:** chỉ với mực nước và triều thì không thấy trước được lúc nước bắt đầu lên mạnh do bão hoặc lũ. Đây đúng là các giờ mà ở Task 3 dự báo NBS +3 h giúp được. `combined_v1` của NBS có 0 lỗi +3 h ở giai đoạn mới.

## Task 5. GRU +3 h: thêm cụm mưa đo, và thêm cả mưa dự báo ECMWF (17/11/2026)

**Hai dataset**, đều chỉ có mốc +3 h, cùng mẫu với `dv_tide7_h3` (giống hệt), cùng mực nước và triều 7 ngày (giống hệt):

| Dataset | Config | Nhóm | Số đầu vào |
|---|---|---|---|
| `train_data/da_vien/dv_tide7_rainobs_h3` | `models/da_vien/dataset_config_tide7_rainobs_h3.json` | tide, rain_obs | 29 |
| `train_data/da_vien/dv_tide7_rainobs_fc_h3` | `models/da_vien/dataset_config_tide7_rainobs_fc_h3.json` | tide, rain_obs, fc_ecmwf | 47 |

- **rain_obs:** 3 cụm, mỗi cụm có tổng mưa 1/3/6/12/24/48 h (log1p) và tỉ lệ trạm thiếu.
  - Đông: Bạch Mã 750041, Đỉnh Bạch Mã 750066.
  - Tây: Hồng Thái 750062, Nhâm 750007.
  - Đồng bằng: Phú Bài 750014, VP Huế 750033, Bình Điền 750061.
  - Chọn log1p chứ không phải mm, vì GRU chuẩn hóa tuyến tính và mưa có đuôi dài (bài học ngày 11/11). Tuyến tính cho thấy bản mm tốt hơn một chút, nhưng chọn sau khi đã xem val và test.
- **fc_ecmwf:** giống hệt khối của NBS (18 kênh, log1p).
- **Kiểm tra:**
  - Không có giá trị thiếu ở train, val, test. Trạm thiếu: đông 0%, tây 0%, đồng bằng 0,5%.
  - |z| lớn nhất: train 11,2, val 9,7, test 9,4, mới 10,6 (đều là `sum_1h`). Không vượt xa train.
- **Dataset đánh giá tới 12/11** (scratchpad): `dv_tide7_rainobs_h3_ext`, `dv_tide7_rainobs_fc_h3_ext`. Dựng với `--normalizers-from`; bản có ECMWF lấy từ `ecmwf_single_runs_soil`. Phần trùng với dataset train giống hệt.

**Lệnh train:** công thức giống Task 4, chỉ đổi `--dataset-dir` và `--checkpoint-dir`.
```bash
.venv/bin/python -m src.ai.train --mlflow-ex-name da_vien --horizon 1h \
  --dataset-dir train_data/da_vien/dv_tide7_rainobs_h3 --target-horizons 3 \
  --rain-stations all --use-weather-forecast false --feature-set water_level+rain+flood_season \
  --train-method gru --head-mode shared --station-embedding-size 0 --lstm-seq-len 12 --lstm-hidden-size 64 --lstm-num-layers 3 \
  --lstm-drop-out 0.2 --lstm-input-dropout 0.1 --lstm-head-dropout 0.1 --lstm-lr 0.0001 --lstm-weight-decay 0.001 \
  --lr-scheduler step --batch-size 64 --num-epochs 100 --early-stopping 100 \
  --change-loss combined --change-under-weight 0.3 --change-event-weight 0 --change-margin-mm 0 \
  --lstm-checkpoint-metric val_loss --skip-test --seed 42 --diagnose-change-dynamics \
  --checkpoint-dir model/experiments/da_vien/dv_tide7_rainobs_h3/noemb_b64_lr0.0001_wd0.001_gru_h64x3_do0.2-0.1-0.1_seq12_e100_combinedu0.3_ev0_step_s42
```
Bản có ECMWF: thay `dv_tide7_rainobs_h3` bằng `dv_tide7_rainobs_fc_h3` ở cả hai chỗ.

### Kết quả Task 5 (người dùng train 7 run, 50 epoch; Claude chấm ngày 17/11/2026)

Số lỗi > 200 mm (trung vị qua các epoch ≥ 3; val / ranh giới / test / mới) và MAE trung vị:

| Run | Lỗi trung vị | Tổng tốt nhất (epoch) | Lỗi train | MAE val / test / mới |
|---|---|---|---|---|
| triều, h64 u0.3 (Task 4) | 5 / 0 / 0 / 7 | 10 (ep6) | 17 | 27,9 / 36,2 / 38,1 |
| mưa, h32 u0 | 4 / 0 / 0 / 6 | 8 | 13 | 29,2 / 38,4 / 39,8 |
| mưa, h32 u0.3 | 5 / 0 / 0 / 7 | 8 | 14 | 29,4 / 38,4 / 39,7 |
| mưa, h64 u0.3 | 3 / 0 / 0 / 6 | 4 (ep5) | 7 | 28,6 / 36,7 / 37,9 |
| mưa + ECMWF, h16 u0 | 1,5 / 0 / 0 / 3 | 2 (ep32) | 14 | 30,7 / 48,4 / 46,2 |
| mưa + ECMWF, h32 u0 | **0 / 0 / 0 / 3** | 2 (ep18) | 13 | 29,9 / 41,2 / 42,7 |
| mưa + ECMWF, h64 u0 | 2 / 0 / 0 / 5 | 4 | 5 | 30,0 / 40,3 / 40,0 |
| mưa + ECMWF, h64 u0.3 | 3 / 0 / 0 / 5 | 4 | 4 | 30,1 / 40,1 / 40,2 |

**Thêm run mưa + ECMWF, h16 u0, batch 32 (50 epoch):**
- Không epoch nào 0 lỗi ở giai đoạn mới. Ít nhất 2 lỗi (ep22, 23, 28). Tốt nhất toàn bộ: ep23 với 1 / 0 / 0 / 2, tổng 3.
- 09/11 09 h và 19 h sai ở cả 48/48 epoch từ ep3 (lỗi khoảng −220 và −240 mm). Đây là lúc lũ bắt đầu lên. Tuyến tính cũng sai 09/11 19 h.
- MAE test 45–46 mm, kém tuyến tính (36) và kém các GRU lớn hơn (40–41).

**Tuyến tính, cùng giờ và cùng đầu vào** (script scratchpad `linear_same_rows.py`; DV 12 h + các kênh tại t; fit trên toàn bộ train như GRU / bỏ lũ):

| Dataset | Lỗi val / ranh giới / test / mới | Lỗi train | MAE val / test / mới |
|---|---|---|---|
| triều | 4 / 0 / 0 / 5 (bỏ lũ: 4 / 0 / 0 / 7) | 13 | 24,3 / 36,1 / 39,0 |
| mưa | 0 / 0 / 0 / 4 (bỏ lũ: 2 / 0 / 0 / 4) | 11 | 25,3 / 35,6 / 36,9 |
| mưa + ECMWF | 0 / 0 / 0 / 3 (bỏ lũ: 2 / 0 / 0 / 4) | 11 | 26,8 / 36,2 / 37,7 |

**Vì sao tuyến tính bằng hoặc hơn GRU**
- Về số lỗi, GRU tốt nhất (mưa + ECMWF, h32 u0) chỉ **ngang** tuyến tính. Về MAE, mọi GRU đều kém hơn: val kém 3–5 mm, test và mới kém 1–12 mm.
- **Không phải do loss:** loss `combined` với event weight 0 là bình phương sai số chuẩn hóa, cùng mục tiêu với OLS.
- **Không phải do thiếu thông tin:** cả hai nhận cùng đầu vào.
- **Rất ít dữ liệu ở chỗ quan trọng:**
  - Train có 5.551 giờ, nhưng chỉ 95 giờ |Δ3| > 200 mm, thuộc 28 đợt, phần lớn là triều mùa đông.
  - Nước lên do mưa gần như chỉ có trận lũ 11/2025. Val có 6 đợt, tất cả trong bão 11–13/09.
- **GRU học thuộc các đợt lớn của train:** độ dốc (dự báo Δ / Δ thật, ở các giờ |Δ| > 150 mm).
  - GRU mưa h64: train tăng 0,65 → 0,81 theo epoch, nhưng val giữ ở 0,61–0,64, mới 0,69–0,76.
  - Tuyến tính: train 0,69, val 0,75, test 0,82, mới 0,80. Không có khoảng cách giữa train và dữ liệu chưa thấy.
  - Giống hiện tượng đã thấy ở NBS.
- **Bài toán gần như tuyến tính:** tide_d3 đã là mức thay đổi triều dự kiến, tổng mưa đã được tính sẵn. Phần phi tuyến còn lại quá nhỏ so với nhiễu khi học từ vài trận.

## Ý tưởng (người dùng, 17/11/2026): dùng model tuyến tính làm baseline trong GRU thay cho persistence

- **Hiện tại:** dự báo = persistence + phần hiệu chỉnh của GRU (head khởi tạo bằng 0; `residual_sequence.py` `_persistence`, `forward`).
- **Đề xuất:** dự báo = persistence + dự báo tuyến tính Δ3 + phần hiệu chỉnh của GRU. Ở epoch 0, model trùng đúng với model tuyến tính.
- **Kênh tuyến tính có sẵn:** nhóm `lin_fc` của builder (`features.linear_forecast_features`). Trên train là dự báo **out-of-fold** (5 đoạn liên tiếp, có khoảng đệm); val và test dùng fit trên toàn bộ train.
- **Đã kiểm tra lin_d3 như một model riêng** (số lỗi > 200 mm / MAE):

| Đầu vào | train (out-of-fold) | val | ranh giới | test | mới |
|---|---|---|---|---|---|
| triều + mưa đo | 29 / 33,8 | 0 / 25,4 | 0 / 41,1 | 0 / 35,9 | 4 / 36,7 |
| triều + mưa đo + ECMWF | 28 / 34,5 | 0 / 26,8 | 0 / 40,1 | 0 / 36,6 | 3 / 37,7 |

  Khớp với tuyến tính trong mẫu ở val, test và mới. Trên train, bản out-of-fold có 28–29 lỗi, so với 11 lỗi của bản trong mẫu. Đây là sai số thật, nên GRU sẽ học sửa đúng loại lỗi mà tuyến tính mắc khi gặp dữ liệu chưa thấy.

## Task 6. GRU với baseline tuyến tính (người dùng đồng ý; làm ngày 17/11/2026)

**Code (mặc định tắt):**
- `residual_sequence.py`: tham số `baseline_feature_indices` / `baseline_scale` / `baseline_offset`.
  - `forward` = persistence + (x_t[lin_d<h>]·scale + offset) + phần hiệu chỉnh.
  - `persistence()` giữ nguyên là persistence thuần, vì loss và các chẩn đoán dùng nó để tính mức thay đổi thật.
  - Hệ số lưu trong `architecture_config`, buffer không vào state_dict, nên checkpoint cũ không đổi.
  - Kênh baseline không được nằm trong group dropout hay history dropout, và luôn lấy trước mọi dropout.
- `regional_training_data.regional_linear_baseline`: scale = s_lin / s_target(h), offset = m_lin / s_target(h).
- `train.py`: cờ `--linear-baseline` (cần dataset có nhóm `lin_fc`); MLflow ghi `linear_baseline`.
- Test `src/ai/tests/test_linear_baseline.py` (6 test). Toàn bộ test: 4 lỗi, đều có từ trước (2 trong test_change_experiments, 1 test_event_aware_no_training, 1 test_residual_architectures; kiểm tra bằng `git stash`).

**Dataset** (các nhóm khác giống hệt bản không có lin; `lin_fc` gồm history 12 h, 5 fold):
- `train_data/da_vien/dv_tide7_rainobs_lin_h3`: config `dataset_config_tide7_rainobs_lin_h3.json`, nhóm tide rain_obs lin_fc.
- `train_data/da_vien/dv_tide7_rainobs_fc_lin_h3`: config `dataset_config_tide7_rainobs_fc_lin_h3.json`, nhóm tide rain_obs fc_ecmwf lin_fc.
- Dataset đánh giá tới 12/11 trong scratchpad (`*_ext`). Phần trùng khớp tới 1e-14.
- lin_d3 thiếu ở 2 mẫu train, các tập khác không thiếu.

**Kiểm tra end-to-end:** model chưa train, dựng giống `train.py`, cho ra đúng now + lin_d3 (sai khác ≤ 0,0001 mm).
- val: 0 lỗi, MAE 25,4 (bản mưa) / 26,8 (bản mưa + ECMWF);
- test: 0 lỗi, MAE 35,9 / 36,6.

Chấm điểm: `score_dv.py ... with0` để chấm cả epoch 0 (= model tuyến tính).

**Lệnh train** (h32×3, u0, 50 epoch, giống GRU tốt nhất ở Task 5):
```bash
.venv/bin/python -m src.ai.train --mlflow-ex-name da_vien --horizon 1h \
  --dataset-dir train_data/da_vien/dv_tide7_rainobs_lin_h3 --target-horizons 3 --linear-baseline \
  --rain-stations all --use-weather-forecast false --feature-set water_level+rain+flood_season \
  --train-method gru --head-mode shared --station-embedding-size 0 --lstm-seq-len 12 --lstm-hidden-size 32 --lstm-num-layers 3 \
  --lstm-drop-out 0.2 --lstm-input-dropout 0.1 --lstm-head-dropout 0.1 --lstm-lr 0.0001 --lstm-weight-decay 0.001 \
  --lr-scheduler step --batch-size 64 --num-epochs 50 --early-stopping 100 \
  --change-loss combined --change-under-weight 0 --change-event-weight 0 --change-margin-mm 0 \
  --lstm-checkpoint-metric val_loss --skip-test --seed 42 --diagnose-change-dynamics \
  --checkpoint-dir model/experiments/da_vien/dv_tide7_rainobs_lin_h3/linbase_noemb_b64_lr0.0001_wd0.001_gru_h32x3_do0.2-0.1-0.1_seq12_e50_combinedu0.0_ev0_step_s42
```
Bản có ECMWF: thay `dv_tide7_rainobs_lin_h3` bằng `dv_tide7_rainobs_fc_lin_h3` ở cả hai chỗ.

### Kết quả Task 6 (run mưa, h32 u0, 50 epoch; chấm cả epoch 0; script scratchpad `linbase_diag.py`)

| Epoch | Lỗi train / val / ranh giới / test / mới | MAE train / val / test / mới |
|---|---|---|
| 0 (= tuyến tính) | 29 / 0 / 0 / 0 / 4 | 33,8 / 25,4 / 35,9 / 36,7 |
| 2 | 25 / 0 / 0 / 0 / **1** | 33,7 / 26,1 / 38,4 / 37,3 |
| 10 | 14 / 9 / 0 / 0 / 4 | 31,5 / 27,7 / 37,8 / 38,5 |
| 50 | 7 / 11 / 0 / 0 / 6 | 29,1 / 29,3 / 34,7 / 36,9 |

**GRU đang làm gì:**
- **Sai số của baseline trên train gần như chỉ đến từ lũ 11/2025.**
  - Cả 29 lỗi train của tuyến tính out-of-fold nằm trong lũ. Lũ chiếm 2,1% số giờ nhưng 31% tổng bình phương sai số.
  - Ở các giờ lũ, tuyến tính out-of-fold dự báo thiếu trung bình +116 mm (RMS 186). Các giờ train khác: trung bình +1, RMS 41.
- **GRU học quy tắc "mưa lớn thì cộng thêm".** Nó dùng mưa đông 24 h để nhận ra trận lũ. Tương quan của phần hiệu chỉnh với mưa đông 24 h là 0,5–0,8 ở val và mới.
  - Trên train, quy tắc này sửa được lũ: lỗi 29 → 7, sửa được 38% bình phương sai số.
- **Ở val (bão 12–13/09, mưa đông 24 h 400–500 mm), quy tắc lại sai.**
  - Tuyến tính ở đó đã đúng hoặc còn dư (lỗi +2…+155). Tương quan giữa hiệu chỉnh và sai số thật ở val là −0,05.
  - GRU vẫn cộng +140…+225 mm, kể cả lúc nước đang rút (13/09 05–12 h, Δ thật −242…+43 mm). Kết quả: 11 lỗi, đều dự báo thừa.
- **Nguyên nhân gốc: train và val dùng hai "phiên bản" tuyến tính khác nhau.**
  - Trên train, fold chứa trận lũ được dự báo bằng tuyến tính chưa từng thấy lũ: thiếu +116 mm.
  - Ở val, test và mới, baseline là tuyến tính fit trên cả train, đã học lũ. Trên chính các giờ lũ, bản này chỉ thiếu +9 mm (RMS 109).
  - Vì vậy phần hiệu chỉnh GRU học được cộng **hai lần** thứ mà baseline lúc dùng đã có sẵn.
  - Thêm nữa, chỉ có một trận lũ, nên GRU không thể biết khi nào quy tắc "mưa lớn thì cộng" là đúng.
- Train có 132 giờ mưa đông 24 h > 200 mm (116 trong số đó là lũ), val 59, test 114.

## Task 7. Baseline tuyến tính fit trên toàn bộ train, dùng cho cả giờ train (người dùng chọn phương án 1; 17/11/2026)

**Code:** `features.linear_forecast_features(folds=0)`, tức config `lin_fc.folds: 0`.
- Không tính chéo nữa: mọi giờ, kể cả train, dùng một hồi quy fit trên toàn bộ train.
- `folds = 1` hoặc âm thì báo lỗi; mặc định 5 giữ nguyên.
- Test `src/ai/tests/test_lin_fc_folds.py`; ghi vào PIPELINE §3.

**Dataset `train_data/da_vien/dv_tide7_rainobs_linfull_h3`** (config `dataset_config_tide7_rainobs_linfull_h3.json`; bản đánh giá tới 12/11 trong scratchpad):
- Các kênh khác giống hệt `dv_tide7_rainobs_lin_h3`. lin_d3 ở val, ranh giới và test giống hệt (sai khác 0); chỉ giờ train thay đổi.
- Baseline trên train: 11 lỗi, MAE 30,3 (bản out-of-fold: 29 lỗi). Ở các giờ lũ 11/2025, sai số trung bình +15 mm (out-of-fold: +119).
- Model chưa train cho ra đúng now + lin_d3 (≤ 0,0005 mm).

**Lệnh train** (giống Task 6, chỉ đổi dataset):
```bash
.venv/bin/python -m src.ai.train --mlflow-ex-name da_vien --horizon 1h \
  --dataset-dir train_data/da_vien/dv_tide7_rainobs_linfull_h3 --target-horizons 3 --linear-baseline \
  --rain-stations all --use-weather-forecast false --feature-set water_level+rain+flood_season \
  --train-method gru --head-mode shared --station-embedding-size 0 --lstm-seq-len 12 --lstm-hidden-size 32 --lstm-num-layers 3 \
  --lstm-drop-out 0.2 --lstm-input-dropout 0.1 --lstm-head-dropout 0.1 --lstm-lr 0.0001 --lstm-weight-decay 0.001 \
  --lr-scheduler step --batch-size 64 --num-epochs 50 --early-stopping 100 \
  --change-loss combined --change-under-weight 0 --change-event-weight 0 --change-margin-mm 0 \
  --lstm-checkpoint-metric val_loss --skip-test --seed 42 --diagnose-change-dynamics \
  --checkpoint-dir model/experiments/da_vien/dv_tide7_rainobs_linfull_h3/linbase_noemb_b64_lr0.0001_wd0.001_gru_h32x3_do0.2-0.1-0.1_seq12_e50_combinedu0.0_ev0_step_s42
```

### Kết quả Task 7 (người dùng train; Claude chấm 51 epoch, kể cả epoch 0)

| Epoch | Lỗi train / val / ranh giới / test / mới | MAE train / val / test / mới | Lỗi lớn nhất ở mới |
|---|---|---|---|
| 0 (= tuyến tính) | 11 / 0 / 0 / 0 / 4 | 30,3 / 25,4 / 35,9 / 36,7 | 221 |
| 12–20 | 9–11 / 0 / 0 / 0 / **1** | 29,1–29,8 / 25,8–26,7 / 34,3–36,0 / 35,1–36,4 | 240–267 |
| 21–28 | 8–9 / 1 / 0 / 0 / 1 | khoảng 28,8 / 27 / 34,5 / 34,7 | 274–286 |
| 47–50 | 6 / 1 / 0 / 0 / 4 | 27,9 / 26,6–26,9 / 31,6–32,6 / 33,8–34,0 | 260–270 |

- **Tốt hơn hẳn bản out-of-fold.** Độ lệch chuẩn của phần hiệu chỉnh chỉ 3–24 mm (bản trước 17–45 mm). Không còn quy tắc "mưa lớn thì cộng 100–200 mm"; val chỉ 0–1 lỗi (bản trước 11).
- **Epoch 12–20: 1 lỗi ngoài train** (tuyến tính có 4).
  - Ba lỗi bị sửa đều là dự báo thiếu sát ngưỡng ở lũ 09–10/11 (−206…−213 → −169…−194).
  - Lỗi còn lại, 09/11 22 h, là lỗi dự báo thừa có sẵn của tuyến tính (+221). GRU còn làm nặng thêm (+252…+279).
- **Sau epoch 20, GRU lại bắt đầu cộng thêm trong bão và lũ:** val 12/09 22 h (hiệu chỉnh +70, lỗi +225), mới 10/11 07 h (+107, lỗi +220). Lỗi train tiếp tục giảm (11 → 6), nhưng lỗi ngoài train tăng lại (1 → 5). Đây là học thuộc, nhưng chậm hơn nhiều so với bản trước.
- MAE test và mới giảm đều (35,9 → 32,0; 36,7 → 33,8), MAE val tăng nhẹ (25,4 → 26,9). Phần GRU sửa được nằm ở giờ thường (triều), không phải ở các giờ bão.
- **Chọn epoch 12–20 là nhìn vào giai đoạn mới.** Val 0 lỗi suốt epoch 0–20, và val MAE tăng đều từ epoch 0, nên chọn chỉ theo val sẽ không ra được khoảng này. Mức cải thiện 4 → 1 lỗi là lạc quan.

**Kích thước hidden h16 và h8** (người dùng train 80 epoch; cùng dataset, cùng các tham số khác):

| Hidden | Số epoch có lỗi ngoài train ≤ 4 (bằng tuyến tính) / < 4 | Ít nhất | MAE val / test / mới ở epoch cuối | Độ lệch chuẩn hiệu chỉnh ở mới (ep 10 → cuối) |
|---|---|---|---|---|
| h32 (50 ep) | 47/50; 29 | 1 (ep 12–20) | 26,9 / 32,0 / 33,8 | 3,9 → 24,4 mm |
| h16 | 80/80; 47 | 3 (ep 21–67) | 26,9 / 35,6 / 36,9 | 1,9 → 14,1 mm |
| h8 | 80/80; 0 | 4 (mọi epoch) | 25,0 / 35,5 / 37,1 | 0,6 → 7,4 mm |

- **h8:** gần như chính là model tuyến tính. 4 lỗi ở mọi epoch, các giờ khó lệch dưới 5 mm. Val MAE tốt hơn 0,4 mm, mới kém hơn 0,4 mm.
- **h16:** không epoch nào kém hơn tuyến tính.
  - Từ epoch 21 đến 67, sửa được đúng 1 lỗi: 10/11 00 h (−213 → −122…−180).
  - Nhưng cùng lúc, nó đẩy hai giờ khác xa thêm:
    - 09/11 22 h (dự báo thừa có sẵn): +221 → +293;
    - val 12/09 22 h: +155 → +209, vượt ngưỡng từ khoảng epoch 70.
  - MAE test và mới gần như không đổi.
- **h64** (50 epoch; tên thư mục ghi `e80` nhưng MLflow ghi `num_epochs` 50):
  - Có 40/50 epoch với lỗi ngoài train ≤ 4, chỉ 5 epoch < 4. Ít nhất 3 lỗi (ep 8–12); không epoch nào 0.
  - Lỗi train giảm mạnh, 11 → 1 (ep 41–50): học thuộc gần hết sai số của trận lũ 11/2025.
  - Từ ep13 val có 1 lỗi: 12/09 22 h, hiệu chỉnh +75 mm, lỗi +203…+257.
  - MAE test và mới tốt nhất trong các kích thước (32,2 / 33,0 ở ep50; tuyến tính 35,9 / 36,7), nhưng MAE val kém nhất (28,0; tuyến tính 25,4).
  - Lỗi lớn nhất ở val tăng 186 → 205–257.
- **Kết luận:** mô hình càng nhỏ thì càng ít hại, nhưng cũng càng ít giúp. Phần GRU thêm được ở +3 h nhỏ (0–3 lỗi, khoảng 1 mm MAE) và luôn kèm việc đẩy giờ bão hoặc lũ về phía dự báo thừa.

### Task 8. Tuyến tính + mực nước NBS +3 h (17/11/2026; script scratchpad `lin_nbs3.py`)

- **Dự báo NBS +3 h:** `combined_v1/short_3h_6h` chạy trên dataset NBS mở rộng tới 12/11; ghép với file dự báo cũ.
  - Phần trùng sai khác ≤ 1,1 mm.
  - Giai đoạn mới: 0 lỗi, MAE 38,8 (khớp `combined.json`).
  - Bỏ các giờ mà khoảng (t−1 h, t+3 h] vắt qua lần đổi mốc đo NBS (4 giờ ở test).
- **Cách dựng:** giống baseline `lin_fc` (folds 0, fit trên toàn bộ train, DV 12 h + tide_now/tide_d3 + mưa đo), thêm một kênh NBS.

| Model | Lỗi train / val / ranh giới / test / mới | MAE train / val / test / mới | R² val / mới |
|---|---|---|---|
| A: baseline hiện tại | 11 / 0 / 0 / 0 / 4 | 30,3 / 25,4 / 35,8 / 36,7 | 62 / 84 |
| B: A + NBS dự báo +3 h | 12 / 0 / 0 / 0 / **1** | 26,1 / **29,7** / 37,0 / 38,4 | 53 / 84 |
| C: A + NBS quan trắc +3 h (giới hạn trên) | 5 / 0 / 0 / 0 / 0 | 16,5 / 13,3 / 22,9 / 21,2 | 89 / 94 |

- **B giảm lỗi ở giai đoạn mới (4 → 1), nhưng MAE kém hơn ở mọi tập ngoài train** (val +4,3, test +1,2, mới +1,7), còn train tốt hơn (30,3 → 26,1).
  - Nguyên nhân: dự báo NBS trên train là trong mẫu (R² NBS train 78%, val 41%, test 85%, mới 84%). Hồi quy tin dự báo NBS hơn mức đáng tin; tệ nhất ở val (bão).
  - Lỗi còn lại của B: 09/11 22 h, +272. NBS dự báo +159 mm, thực tế NBS +122, DV +88.
- **Thêm mực nước NBS quan trắc 12 h** (NBS(t) − DV(t) và NBS(t) − NBS(t−k), k = 1…12; mốc đo NBS sau 10/10 đã cộng +172 mm; thiếu 0,04%):

| Model | Lỗi train / val / ranh giới / test / mới | MAE train / val / ranh giới / test / mới | Lỗi lớn nhất val / test / mới |
|---|---|---|---|
| A: baseline hiện tại | 11 / 0 / 0 / 0 / 4 | 30,3 / 25,4 / 41,1 / 35,8 / 36,7 | 186 / 129 / 221 |
| **D: A + NBS quan trắc 12 h** | 14 / 0 / 0 / 0 / **0** | 27,6 / **23,6 / 37,1 / 29,5 / 30,7** | 195 / 109 / 179 |
| E: D + NBS dự báo +3 h | 13 / 0 / 0 / 0 / 1 | 24,9 / 27,2 / 40,1 / 32,7 / 33,7 | 192 / 134 / 243 |

  - **D tốt hơn A ở mọi tập ngoài train:** 0 lỗi ở cả 4 tập; MAE giảm 1,8 (val) đến 6,3 mm (test); R² mới 84 → 89. Chỉ train tăng nhẹ (11 → 14 lỗi).
  - 4 giờ lũ 09–10/11: −209 / −206 / +221 / −213 → −174 / −107 / +179 / −150.
  - Thêm dự báo NBS lên trên D (E) lại kém đi; cùng lý do dự báo NBS trên train là trong mẫu.
  - **Lưu ý:**
    - Việc dùng NBS quan trắc đã được gợi ý từ Task 3 (test 43 → 33), trước khi xem số liệu giai đoạn mới.
    - Lúc chạy thật cần xử lý các lần NBS đổi mốc đo, như phép cộng +172 mm ở đây. Giờ 16:00 UTC ngày 10/10 có lẫn một số đo trước bước nhảy.
- **C cho thấy giới hạn:** nếu biết đúng mực nước NBS 3 h tới thì MAE giảm khoảng một nửa và không còn lỗi nào ngoài train. Vậy chất lượng dự báo NBS là nút thắt.

**Rủi ro:** baseline trên train giờ là trong mẫu, nên phần dư trên train nhỏ hơn thực tế khoảng 41 → 38 mm RMS ở giờ thường. GRU có thể học quá ít, hoặc học thuộc phần dư còn lại của trận lũ.

## Task 9. GRU trên baseline D (tuyến tính + NBS quan trắc 12 h) (17/11/2026)

**Code mới (mặc định tắt):**
- `build.py`: nhóm `station_obs`; `hourly_level_from_db(offsets=...)` cộng một mức lệch cố định vào số đo 10 phút thô kể từ `from_utc`, trước despike và trước khi lấy trung bình giờ (`apply_offsets`).
- `features.station_obs_features`: `obs_<tên>_minus_self`, `obs_<tên>_d1..d12`, `obs_<tên>_missing_frac`.
- Test `src/ai/tests/test_station_obs.py`.

**Dataset `train_data/da_vien/dv_tide7_rainobs_nbsobs_linfull_h3`** (config `dataset_config_tide7_rainobs_nbsobs_linfull_h3.json`; nhóm tide rain_obs station_obs lin_fc; 45 đầu vào):
- NBS dùng despike giống config NBS, cộng +172 mm từ 10/10 15:20 UTC.
- Triều và mưa giống hệt bản trước.
- lin_d3 (folds 0) tái tạo đúng model D của Task 8: 14 / 0 / 0 / 0 / 0 lỗi; MAE 27,6 / 23,6 / 37,1 / 29,5 / 30,7.
- Model chưa train = baseline (≤ 0,0001 mm). Bản đánh giá tới 12/11 trong scratchpad.

**Lệnh train** (h32, 50 epoch, giống Task 7):
```bash
.venv/bin/python -m src.ai.train --mlflow-ex-name da_vien --horizon 1h \
  --dataset-dir train_data/da_vien/dv_tide7_rainobs_nbsobs_linfull_h3 --target-horizons 3 --linear-baseline \
  --rain-stations all --use-weather-forecast false --feature-set water_level+rain+flood_season \
  --train-method gru --head-mode shared --station-embedding-size 0 --lstm-seq-len 12 --lstm-hidden-size 32 --lstm-num-layers 3 \
  --lstm-drop-out 0.2 --lstm-input-dropout 0.1 --lstm-head-dropout 0.1 --lstm-lr 0.0001 --lstm-weight-decay 0.001 \
  --lr-scheduler step --batch-size 64 --num-epochs 50 --early-stopping 100 \
  --change-loss combined --change-under-weight 0 --change-event-weight 0 --change-margin-mm 0 \
  --lstm-checkpoint-metric val_loss --skip-test --seed 42 --diagnose-change-dynamics \
  --checkpoint-dir model/experiments/da_vien/dv_tide7_rainobs_nbsobs_linfull_h3/linbase_noemb_b64_lr0.0001_wd0.001_gru_h32x3_do0.2-0.1-0.1_seq12_e50_combinedu0.0_ev0_step_s42
```

### Kết quả Task 9 (người dùng train h32 thường và h32 + tail λ 3 từ 150 mm, 50 epoch; Claude chấm cả epoch 0)

Lỗi ngoài train = val + ranh giới + test + mới; baseline D có 0.

| Run | Epoch có 0 lỗi ngoài train | Lỗi train (ep 0 → 50) | MAE val / test / mới tốt nhất trong các epoch 0 lỗi |
|---|---|---|---|
| h32 thường | **0–31 (liên tục)**, sau đó 1–3 | 14 → 7 | ep30: 23,9 / 28,6 / 28,2 (D: 23,6 / 29,5 / 30,7) |
| h32 + tail | 0–11, 47, 50 | 14 → 4 | ep50: 23,7 / 27,5 / 28,0 |

- **Tail:** val có lỗi từ epoch 12 đến 49 (1–4 lỗi, ở 12/09 22 h và 13/09 06–08 h, đều dự báo thừa; hiệu chỉnh +34…+53 mm).
  - Ep47 và ep50 về 0, nhưng ep48–49 vẫn có 2 lỗi, và lỗi lớn nhất ở val ep50 là 193 mm (sát ngưỡng). Đây là dao động, chưa ổn định.
  - Tail đẩy lỗi train xuống nhanh hơn (4 so với 7), cùng cơ chế học thuộc trận lũ.
- **Thường:** 0 lỗi ngoài train suốt epoch 0–31, MAE test và mới giảm đều. Từ epoch 32, giờ 12/09 22 h vượt ngưỡng (hiệu chỉnh khoảng +43 mm).
- **Gợi ý:** h32 thường, khoảng ep 20–31. Nhưng khoảng này cũng được chọn sau khi xem mọi tập.

## Model +3 h tốt nhất: `models/da_vien/h3_v1` (người dùng chọn ngày 17/11/2026)

- Run h32 + tail λ 3 từ 150 mm, epoch 50, dataset `dv_tide7_rainobs_nbsobs_linfull_h3`. Thư mục đóng gói có `model.pt` và các file kèm theo, cùng `model_card.json` (sha256, đầu vào, lưu ý).
- Chấm lại trên bản đóng gói (lỗi > 200 mm / MAE / lỗi lớn nhất):

| Tập | Lỗi | MAE | Lỗi lớn nhất |
|---|---|---|---|
| train | 4 | 25,4 | 253 |
| val | 0 | 23,7 | 193 |
| ranh giới | 0 | 29,4 | 94 |
| test | 0 | 27,5 | 121 |
| mới | 0 | 28,0 | 195 |

- **Lưu ý:**
  - Epoch được chọn sau khi xem mọi tập.
  - Epoch 48–49 có 2 lỗi ở val; biên với ngưỡng mỏng (lỗi lớn nhất 193 và 195 mm).
  - Cần kiểm tra lại trên dữ liệu sau 12/11.

## Task 10. +6 h: hồi quy tuyến tính theo nhóm đầu vào (17/11/2026; script scratchpad `lin_h6.py`)

- **Cách làm:** như baseline +3 h (`lin_fc`, folds 0), nhãn là Δ6 tính từ mực nước giờ, cùng tập mẫu của dataset +3 h.
- **Đầu vào:**
  - Biển: nhóm `sea` của NBS.
  - ECMWF: 18 kênh, log1p.
  - NBS dự báo +6 h: `combined_v1` short, chạy tới 12/11.

Số lỗi > 200 mm (train / val / ranh giới / test / mới) và MAE val / test / mới:

| Đầu vào | Lỗi | Tổng ngoài train | MAE |
|---|---|---|---|
| persistence | 359 / 45 / 15 / 190 / 165 | 415 | 61,0 / 147,0 / 145,5 |
| DV lịch sử + triều | 60 / **12** / 0 / 5 / 20 | 37 | **41,0** / 55,6 / 61,5 |
| + mưa đo | 50 / 18 / 1 / 4 / 11 | 34 | 44,9 / 57,3 / 56,8 |
| + NBS quan trắc | 49 / 17 / 0 / 2 / 16 | 35 | 40,6 / 50,0 / 59,1 |
| D (triều + mưa + NBS quan trắc) | 36 / 16 / 1 / 3 / 13 | 33 | 43,3 / 51,7 / 54,1 |
| **D + biển** | 33 / 14 / 1 / 4 / **8** | **27** | 42,8 / 49,8 / **49,7** |
| D + ECMWF | 34 / 17 / 1 / **1** / 13 | 32 | 45,4 / 51,5 / 53,8 |
| D + biển + ECMWF | 32 / 17 / 1 / 2 / **7** | **27** | 45,4 / 50,3 / 49,9 |
| D + NBS dự báo +6 h | 26 / 16 / 1 / 9 / 10 | 36 | 53,0 / 60,3 / 57,6 |

- **+6 h khó hơn nhiều so với +3 h:** baseline tốt nhất vẫn còn 27 lỗi ngoài train (ở +3 h là 0).
- **Biển giúp rõ ở +6 h** (khác với +3 h): ở D, giai đoạn mới giảm 13 → 8, MAE mới 54 → 50.
- **Mưa đo làm val kém đi** (12 → 16–18 lỗi), dù test và mới tốt hơn.
- **Lỗi val tập trung ở 13/09 04–11 h** (dự báo thừa 211–428 mm, lúc nước rút sau đỉnh bão) và 11/09 10–14 h (thừa 229–302 mm). Mưa lớn khiến hồi quy chờ nước tiếp tục lên.
- **Lỗi ở mới** tập trung ở 09/11 07–08 h và 15–17 h (dự báo thiếu khi lũ lên) và 28/10 12–13 h.
- NBS dự báo +6 h vẫn có vấn đề trong mẫu: train tốt hơn hẳn (26 lỗi), ngoài train kém hơn (MAE val 53).
- **R² (%) của D và D + biển** (train / val / ranh giới / test / mới):

| Mốc | DV lịch sử + triều | D | D + biển |
|---|---|---|---|
| +3 h | 63 / 62 / 83 / 86 / 81 | 70 / 67 / 88 / 91 / 89 | 70 / 67 / 89 / 91 / 89 (thêm 0–1 điểm) |
| +6 h | 53 / 58 / 84 / 84 / 79 | 61 / 51 / 86 / 86 / 84 | 63 / 53 / 87 / 87 / 87 (thêm 1–2 điểm; mới +2,4) |

- **Gió** (`prev_offshore_to_20261116.json`; 4 kênh gió vào bờ như NBS, hoặc cả 14 kênh). R² val / mới, tổng lỗi ngoài train, MAE val / mới:

| Mốc | Model | R² val / mới | Tổng lỗi ngoài train | MAE val / mới |
|---|---|---|---|---|
| +3 h | D | 66,5 / 88,5 | 0 | 23,6 / 30,7 |
| | D + gió 4 kênh | 66,1 / 88,6 | 0 | 23,7 / 30,6 |
| | D + gió 14 kênh | 62,2 / 88,8 | 0 | 25,5 / 30,1 |
| +6 h | D + biển | 52,5 / 86,8 | 27 | 42,8 / 49,7 |
| | D + biển + gió 4 kênh | 52,1 / 86,9 | 26 | 42,4 / 49,2 |
| | D + biển + gió 14 kênh | **45,9** / 87,2 | 28 | **47,3** / 48,1 |

  - Gió 4 kênh: thêm 0–0,6 điểm R²; ở +6 h giảm 1 lỗi (ranh giới 1 → 0) và MAE 0,4–2 mm. Gió gần như trùng thông tin với biển (trên D, gió 4 kênh giảm 33 → 29 lỗi; trên D + biển chỉ 27 → 26).
  - Gió 14 kênh: tốt hơn ở test và mới, nhưng R² val giảm 4–7 điểm (overfit, giống ngày 11/11 và Task 4).

- **ECMWF trên nền D + biển** (thư mục `ecmwf_single_runs_soil`, không thiếu; 18 kênh, hoặc chỉ 4 kênh trung bình lưu vực `fc_basin_*`; log1p hoặc mm). R² val / mới, tổng lỗi ngoài train (val / ranh giới / test / mới), MAE val / mới:

| Mốc | Model | R² val / mới | Lỗi ngoài train | MAE val / mới |
|---|---|---|---|---|
| +3 h | D + biển | 66,9 / 89,3 | 0 | 23,6 / 29,8 |
| | + 18 kênh log / mm | 63,4–63,9 / 88,5–88,6 | 0 / 2 | 24,6–24,7 / 30,6–30,9 |
| | + lưu vực log / mm | 66,6 / 65,1, 89,3 | 0 / 2 | 23,6–24,1 / 29,5–29,8 |
| +6 h | D + biển | 52,5 / 86,8 | 27 (14 / 1 / 4 / 8) | 42,8 / 49,7 |
| | + 18 kênh log | 46,6 / 86,8 | 27 (17 / 1 / 2 / 7) | 45,4 / 49,9 |
| | + 18 kênh mm | 38,8 / 86,8 | 27 (21 / 0 / 0 / 6) | 48,2 / 49,0 |
| | + lưu vực log | 51,3 / 87,1 | 26 (14 / 1 / 4 / 7) | 43,4 / 49,0 |
| | + lưu vực mm | 43,3 / 87,2 | 23 (16 / 0 / 1 / 6) | 46,0 / 49,1 |

  - Khi đã có mưa đo và biển, ECMWF gần như không thêm gì ở cả +3 h và +6 h (R² test và mới thay đổi dưới 1 điểm).
  - **Luôn làm val kém đi** (−1…−14 điểm R²): mưa dự báo lớn trong bão 11–13/09 khiến hồi quy chờ nước lên tiếp, trong khi nước đang rút.
  - Bản mm lưu vực có tổng lỗi thấp nhất (23), nhưng là nhờ test và ranh giới, đổi lại val tệ hơn. Khác biệt nhỏ, lại chọn sau khi xem mọi tập.

- **Thêm dự báo +3 h của DV (`h3_v1`) vào +6 h** (nền D + biển; số lỗi train / val / ranh giới / test / mới, tổng ngoài train; MAE val / test / mới):

| Thêm | Lỗi | Tổng ngoài train | MAE |
|---|---|---|---|
| (không thêm) | 33 / 14 / 1 / 4 / 8 | 27 | 42,8 / 49,8 / 49,7 |
| tide_d3 | giống hệt | 27 | giống hệt |
| lin_d3 (baseline tuyến tính +3 h) | giống hệt | 27 | 42,9 / 49,8 / 49,7 |
| dự báo đầy đủ của h3_v1 | 17 / 16 / 0 / 2 / 10 | 28 | 45,9 / 56,3 / 53,0 |
| lin_d3 + phần hiệu chỉnh GRU (2 kênh) | 15 / 16 / 0 / 5 / 12 | 33 | 47,7 / 55,4 / 50,9 |

  - **Không giúp.**
    - lin_d3 là tổ hợp tuyến tính của các đầu vào đã có, nên không thêm thông tin (giống Task 3).
    - Phần hiệu chỉnh của GRU là trong mẫu trên train: lỗi train 33 → 15–17, nhưng ngoài train không giảm (27 → 28–33) và MAE kém đi 3–7 mm. Cùng cơ chế với NBS dự báo.

## Task 11. Dataset +6 h cho GRU: có và không có ECMWF; baseline persistence hoặc tuyến tính (17/11/2026)

**Dataset** (chỉ mốc +6 h; tập mẫu: train 5.551, val 1.080, test 723, ranh giới 48):

| Dataset | Config | Nhóm | Số đầu vào khi chọn `all` |
|---|---|---|---|
| `train_data/da_vien/dv_h6` | `dataset_config_h6.json` | tide, rain_obs, station_obs (NBS), sea, lin_fc | 53 |
| `train_data/da_vien/dv_h6_fc` | `dataset_config_h6_fc.json` | như trên + fc_ecmwf | 71 |

- **`lin_d6` (folds 0):** `lin_fc.inputs` = tide, rain_obs, station_obs, sea, tức model "D + biển" của Task 10. ECMWF **không** nằm trong baseline (nó làm R² val kém đi), chỉ là đầu vào của GRU. Vì vậy `lin_d6` giống hệt nhau ở cả hai dataset.
- **Baseline tái tạo đúng Task 10:** lỗi 33 / 14 / 1 / 4 / 8, MAE 46,6 / 42,8 / 54,5 / 49,8 / 49,7. Model chưa train cho ra đúng now + lin_d6 (≤ 0,0001 mm).
- **ECMWF** lấy từ `ecmwf_single_runs_soil` (giống bản gốc trên giai đoạn dataset, kéo dài tới 16/11).
- **Kiểm tra:**
  - |z| lớn nhất ngoài train ≤ 10,6 (train 13,0).
  - Biển thiếu 3,4% giờ train, do dữ liệu `previous_day1` chỉ có từ 18/11/2025, tức đầu trận lũ.
- Dataset đánh giá tới 12/11 trong scratchpad (`dv_h6_ext`, `dv_h6_fc_ext`); phần trùng khớp tới 1e-13.
- **Model baseline persistence:** dùng cùng dataset nhưng chọn nhóm bằng `--rain-stations tide+rain_obs+station_obs+sea[+fc_ecmwf]`, nên GRU không thấy kênh `lin_d6` (51 / 69 đầu vào).

**4 lệnh train** (h32×3, u0, 50 epoch, seed 42; giống công thức +3 h): xem câu trả lời ngày 17/11. Thư mục `model/experiments/da_vien/dv_h6[_fc]/{persbase,linbase}_…`.

### Kết quả Task 11 (người dùng train 11 run; Claude chấm mọi epoch)

Lỗi ngoài train = val + ranh giới + test + mới. Baseline tuyến tính (epoch 0) = 27 (14 / 1 / 4 / 8). Bảng xếp theo số lỗi ngoài train ít nhất.

| Run | Ít nhất (epoch; train / val / rg / test / mới) | Trung vị | Số epoch < 27 | Epoch cuối |
|---|---|---|---|---|
| không ECMWF, persistence, lr 3e-4, seq 12 | **18** (ep34; 10 / 8 / 0 / 6 / 4) | 29 | 22/50 | 5 / 14 / 0 / 5 / 5 |
| **không ECMWF, tuyến tính, lr 3e-4, seq 25** | 19 (ep45; 4 / 12 / 0 / 2 / 5) | **24,5** | **34/50** | 7 / 15 / 0 / 7 / 5 |
| ECMWF, persistence, lr 3e-4, seq 75 (75 ep) | 20 (ep43) | 24 | 48/75 | 3 / 14 / 0 / 1 / 9 |
| không ECMWF, tuyến tính, lr 1e-4, seq 12 | 21 (ep6) | 27 | 15/50 | 21 / 16 / 0 / 6 / 6 |
| ECMWF, persistence, h64, lr 3e-4, seq 25 (75 ep) | 22 (ep70; train 0) | 30 | 13/75 | 0 / 17 / 1 / 8 / 7 |
| không ECMWF, persistence, lr 3e-4, seq 25 | 23 | 28 | 11/50 | |
| ECMWF, tuyến tính, lr 1e-4, seq 12 | 23 | 31 | 15/50 | |
| không ECMWF, tuyến tính, lr 3e-4, seq 12 | 23 | 28 | 14/50 | |
| ECMWF, persistence, lr 3e-4, seq 12 | 29 | 42 | 0 | |
| không ECMWF, persistence, lr 1e-4, seq 12 | 33 | 40 | 0 | |
| ECMWF, persistence, lr 1e-4, seq 12 | 45 | 53 | 0 | |

- **GRU có giúp ở +6 h:** tốt nhất 18–20 lỗi so với 27 của tuyến tính. Ổn định nhất là bản tuyến tính, lr 3e-4, seq 25: 34/50 epoch dưới 27, trung vị 24,5.
- **Baseline tuyến tính giúp rõ** khi train chậm (lr 1e-4, seq 12): trung vị 27 so với 40 (persistence). Với lr 3e-4, hai loại baseline gần nhau.
- **ECMWF không giúp:** cùng công thức thì luôn kém hơn (persistence lr 1e-4: 45 so với 33; lr 3e-4: 29 so với 18; tuyến tính: 23 so với 21). Riêng bản seq 75 có ECMWF ổn định, nhưng khác cả độ dài cửa sổ.
- **Lỗi còn lại vẫn ở các giờ cũ** (bản tuyến tính, lr 3e-4, seq 25, trung bình mỗi epoch):
  - val 13/09 (6,3; chủ yếu dự báo thừa khi nước rút sau đỉnh bão) và 11/09 (4,5; thừa);
  - mới 09/11 (4,4; thiếu khi lũ lên);
  - test 21/09 (2,2; thừa).
  - Persistence lr 3e-4 thêm lỗi thiếu ở 12/09 và test 13/10.
- Train còn 0–10 lỗi: GRU học thuộc trận lũ 11/2025, nhất là h64 (train 0).
- Mọi kết quả chọn sau khi xem mọi tập; mỗi công thức mới chạy 1 seed.
- **Val thấp nhất là 6 lỗi:** không ECMWF, persistence, lr 3e-4, seq 25, epoch 14 (train / val / rg / test / mới = 17 / 6 / 0 / 11 / 7; tổng ngoài train 24).
  - Các epoch val ≤ 8 đều có test 6–11 lỗi; baseline tuyến tính có test 4.
  - Val (bão, dự báo thừa khi nước rút) và test (triều mạnh) kéo theo hai hướng ngược nhau.

## Task 12. +6 h thiếu thông tin gì? (người dùng: "phải thêm thông tin có ích"; 17/11/2026; scripts scratchpad `new_info_h6.py`)

**Thử thông tin "tương lai thật" (oracle, không dùng được khi chạy thật)** trên nền tuyến tính D + biển, +6 h. Lỗi > 200 mm và R² theo train / val / ranh giới / test / mới:

| Thêm | Lỗi | R² (%) |
|---|---|---|
| (nền) | 33 / 14 / 1 / 4 / 8 | 63 / 53 / 87 / 87 / 87 |
| Thuận An thay đổi 2/4/6 h tới | 33 / 14 / 1 / 4 / 8 | không đổi |
| Tứ Phú thay đổi 6 h tới | 31 / 13 / 0 / 0 / 14 | 69 / 61 / 90 / 89 / 83 |
| NBS − TA thay đổi tới (phần sông) | 20 / 7 / 0 / 0 / 1 | 77 / 63 / 90 / 89 / 90 |
| NBS thay đổi 2/4/6 h tới | **13 / 0 / 0 / 0 / 0** | 92 / 93 / 98 / 96 / 97 |

- Biết trước mực nước biển ở cửa (Thuận An) cũng không giúp; triều thiên văn ở TA đã giải thích 96% thay đổi của TA.
- Biết trước NBS thì hết lỗi. Phần sông (NBS − TA) gỡ được khoảng một nửa.
- Vậy thứ còn thiếu là **diễn biến dòng sông Hương** trong 6 h tới, không phải biển.

**Dữ liệu sẵn trong DB chưa dùng, sàng lọc bằng tuyến tính (tổng lỗi ngoài train; nền 27):**
- **Mực nước các trạm khác** (despike 150, chưa kiểm tra đổi mốc): Thuận An 33, Tứ Phú 30, Tư Hiền 30, Phú Xuân 26, Hương Sơ 24, An Cựu 27, Xóm Gióng 27, Ô Lâu 41. Thuận An và Tứ Phú làm test kém đi (4 → 6–9). Thặng dư nước dâng ở TA (quan trắc − triều TA): 28.
- **Mưa thêm ~50 trạm,** gom 4 cụm: thượng nguồn Hương 29, Tả Trạch 28, sông Bồ 26, thành phố 24. Không có cụm nào giúp rõ.
- Ở +3 h không nhóm nào thay đổi được (đã 0 lỗi).
- **Vận hành hồ đập** (`hydrology.facility_operations`, nguồn sohoa; chưa từng dùng làm đầu vào):
  - Có Bình Điền (Hữu Trạch, bước 3 h), Tả Trạch (12 h), Hương Điền (Bồ, 6 h), với `qdi` (lưu lượng xả) và `qden` (lưu lượng đến). Đủ cả năm, cập nhật gần thời gian thực.
  - Lũ 11/2025: xả rất lớn (Hương Điền tới 3.531 m³/s; Bình Điền tới 1.499).
  - Lũ 10/11/2026: Hương Điền tăng xả 176 → 1.561 m³/s từ 05 h, Tả Trạch 80 → 290, Bình Điền 62 → 205.
  - Bão 12–13/09: xả nhỏ và đều (Bình Điền khoảng 60, Tả Trạch 40–50, Hương Điền khoảng 190). Hồ **không** giải thích được lỗi val 13/09, cũng không giải thích lỗi 09/11 06–08 h và 15–18 h (trước khi tăng xả).

### Task 12b. Thử dữ liệu hồ đập (người dùng đồng ý; script scratchpad `reservoir_h6.py`)

- **Dữ liệu:** `hydrology.facility_operations` (sohoa) cho Bình Điền (sohoa:2), Tả Trạch (sohoa:3), Hương Điền (sohoa:1); cột `qdi` (xả) và `qden` (đến).
  - **Chỉ dùng số liệu đã công bố:** bản ghi dùng được tại t khi max(observed_at, source_updated_at) ≤ t. Bình Điền: 90% nhập trong 9,5 h, có bản muộn tới 236 h.
  - Bỏ 1 bản ghi sai: Bình Điền 1.379 m³/s lúc 15/11/2025 18:00 UTC, nằm giữa 215 và 212, nhập ngày 17/11.
  - Hương Điền mùa khô xả bật/tắt 0 ↔ 100–180 m³/s (chạy tua-bin), là dữ liệu thật.
  - Bình Điền trống từ 30/01 đến 19/05/2026 (train, mùa khô); có kênh tuổi số liệu.
- **Đặc trưng:** log1p(xả) gần nhất, thay đổi so với 6 h và 12 h trước, tuổi số liệu; tùy chọn thêm lưu lượng đến.
- **Kết quả tuyến tính trên nền D + biển** (lỗi train / val / ranh giới / test / mới; tổng ngoài train):

| Mốc | Model | Lỗi | Tổng |
|---|---|---|---|
| +6 h | nền | 33 / 14 / 1 / 4 / 8 | 27 |
| | + Tả Trạch | 34 / 12 / 1 / 4 / 8 | 25 |
| | + Bình Điền + Tả Trạch | 34 / 12 / 1 / 3 / 8 | 24 |
| | + 3 hồ (xả) | 33 / 13 / 1 / 3 / 7 | 24 |
| | + 3 hồ (xả + đến) | 34 / 16 / 1 / 7 / 6 | 30 |
| +3 h | mọi biến thể | train 13–14, ngoài train 0–1 | |

  R² và MAE thay đổi dưới 1 điểm.
- **Hồ gần như không giúp:** các giờ lỗi ngoài train (03/08, 11/09, 13/09, 21/09, 28/10, 09/11) đều rơi vào lúc hồ xả nhỏ và đều.
  - Tuyến tính cũng không sửa được lỗi lũ train 11/2025: lũ chỉ có một trận, quan hệ xả → mực nước có độ trễ và phi tuyến.
  - GRU có thể dùng tốt hơn trong các trận do xả hồ, nhưng val, test và mới gần như không có trận nào như vậy (chỉ 10/11).

### Task 12c. Dự báo +6 h của NBS (`combined_v1` short) làm đầu vào cho DV +6 h

- **Độ chính xác của chính dự báo NBS +6 h trên NBS** (R² / MAE / lỗi > 200 mm):

| Tập | R² | MAE | Lỗi |
|---|---|---|---|
| train | 80% | 31 | 25 |
| val | 37% | 53 | 0 |
| ranh giới | 82% | 62 | 0 |
| test | 85% | 54 | 0 |
| mới | 86% | 54 | 0 |

  Trên train là dự báo trong mẫu: MAE 31, so với 53–62 ở ngoài train.
- **Tuyến tính DV +6 h trên nền D + biển:**

| Model | Lỗi train / val / rg / test / mới | Tổng ngoài train | MAE val / test / mới |
|---|---|---|---|
| nền | 33 / 14 / 1 / 4 / 8 | 27 | 42,8 / 49,8 / 49,7 |
| + NBS dự báo +6 h | 27 / 15 / 1 / 8 / 9 | 33 | 52,4 / 59,6 / 58,4 |
| + NBS quan trắc tương lai (oracle) | 13 / 0 / 0 / 3 / 0 | 3 | 17,2 / 31,3 / 23,8 |

- **Kết luận:** dự báo NBS hiện tại làm DV kém đi (MAE +9…+10 mm), vì hồi quy học trên train nơi dự báo NBS tốt gần gấp đôi thực tế. Thông tin NBS rất có giá trị (oracle: 27 → 3 lỗi), nhưng muốn dùng đúng thì cần dự báo NBS out-of-fold trên train, tức train lại model NBS theo từng khối thời gian.

### Task 12d. Chẩn đoán: dự báo NBS +6 h ngoài mẫu, cân đúng trọng số, có giúp DV +6 h không? (người dùng cho phép fit trên val/test chỉ để chẩn đoán; script scratchpad `nbs_fc_diag.py`)

Ở val, test và mới, dự báo `combined_v1` là dự báo ngoài mẫu, nên fit hồi quy trên một tập và chấm trên tập khác. Kết quả không dùng để chọn model.

| Fit trên | Model | Lỗi val / test / mới (lỗi > 200 mm) | MAE val / test / mới |
|---|---|---|---|
| val | D + biển | (fit) / 13 / 26 | (fit) / 66,0 / 62,1 |
| | + NBS dự báo | (fit) / **48** / 25 | (fit) / **89,6** / 63,8 |
| test | D + biển | 34 / (fit) / 9 | 78,9 / (fit) / 46,9 |
| | + NBS dự báo | 32 / (fit) / 7 | 76,6 / (fit) / 47,1 |
| val + test | D + biển | (fit) / (fit) / 14 | (fit) / (fit) / 45,9 |
| | + NBS dự báo | (fit) / (fit) / 12 | (fit) / (fit) / 43,2 |
| (tham chiếu) train | D + biển / + NBS dự báo / + NBS thật | 14 / 4 / 8; 15 / 6 / 9; 0 / 0 / 0 | |

- **Khi được cân trọng số trên dữ liệu ngoài mẫu, dự báo NBS +6 h chỉ giúp rất ít và không nhất quán:**
  - fit trên test: val −2, mới −2 lỗi;
  - fit trên val + test: mới −2 lỗi, MAE −2,7 mm;
  - fit trên val (chỉ bão): test tệ đi nhiều (13 → 48).
  - So với oracle (NBS thật: 0 lỗi gần như mọi nơi), khoảng cách rất lớn. Nút thắt là **chất lượng dự báo NBS** (R² val 37%), không phải cách dùng.
- **Quyết định đề xuất:** không làm NBS out-of-fold cho DV. Muốn tiến xa hơn ở +6 h thì phải cải thiện chính dự báo NBS hoặc tìm nguồn dữ liệu mới.
- **Lưu ý:** mỗi tập chỉ khoảng 700–1.100 giờ, gần như 1–2 sự kiện, nên fit trên một tập riêng lẻ khái quát kém (D + biển fit trên val → test 13 lỗi, so với 4 khi fit trên train).
- Người dùng cho phép Claude tự chạy train trên GPU (17/11/2026).

### Task 12e. Ensemble các run +6 h đã có (không train thêm)

Trung bình dự báo của 10 epoch cuối mỗi run, rồi trung bình qua các run (lỗi train / val / ranh giới / test / mới; tổng ngoài train; MAE val / mới):

| Ensemble | Lỗi | Tổng | MAE |
|---|---|---|---|
| **6 run không ECMWF** (lr 3e-4 ×5 và tuyến tính lr 1e-4) | 8 / 10 / 0 / 2 / 3 | **15** | 44,8 / 39,3 |
| cả 11 run | 8 / 12 / 0 / 2 / 3 | 17 | 45,1 / 40,6 |
| run đơn tốt nhất (tuyến tính lr 3e-4 seq 25, 10 epoch cuối) | 4 / 12 / 0 / 4 / 5 | 21 | 47,7 / 42,8 |
| baseline tuyến tính | 33 / 14 / 1 / 4 / 8 | 27 | 42,8 / 49,7 |

Ensemble giảm lỗi mà không cần chọn epoch, nhưng val vẫn còn 10 lỗi (bão 11 và 13/09).

### Task 12f. Độ ẩm đất ECMWF (nhóm `soil`; script scratchpad `soil_h6.py`)

- **Dữ liệu:** `ecmwf_single_runs_soil`, 4 tầng (0–7, 7–28, 28–100, 100–255 cm), trung bình 7 ô; quy tắc chọn lượt chạy giống `fc_ecmwf`; không thiếu.
- **Ở cả ba sự kiện, đất đều gần bão hòa:** tầng 0–7 cm 0,41–0,50 m³/m³ trong lũ 11/2025, bão 09/2026 và lũ 11/2026 (mùa khô 05/2026: 0,27–0,43). Độ ẩm đất chủ yếu cho biết mùa, không phân biệt được các trận.
- **Tuyến tính trên D + biển** (lỗi train / val / ranh giới / test / mới; tổng ngoài train):

| Mốc | Model | Lỗi | Tổng |
|---|---|---|---|
| +6 h | nền | 33 / 14 / 1 / 4 / 8 | 27 |
| | + tầng 0–7 cm (và × mưa dự báo) | 33 / 14 / 1 / 4 / 8 | 27 |
| | + 4 tầng | 34 / 14 / 1 / 3 / 8 | 26 |
| | + độ ẩm × mưa đo 24 h | 24 / **36** / 4 / 3 / **28** | **71** |
| +3 h | + 4 tầng | 0 ngoài train | |
| | + độ ẩm × mưa đo | val 16, mới 9 | 25 |

- **Kết luận:**
  - Độ ẩm đất không giúp: R² và MAE thay đổi dưới 1 điểm.
  - Kênh tương tác độ ẩm × mưa đo (tính bằng mm) overfit mạnh vào trận lũ 11/2025: train tốt hơn, nhưng val và mới tệ hẳn.

## Task 13. Thêm thông tin cho +6 h (người dùng đồng ý; Claude tự chạy train từ 17/11/2026, mỗi bước chờ người dùng cho phép)

**Bước 1: dự báo Thuận An (`ensemble_34k`) làm đầu vào tuyến tính.**
- Dựng dataset TA tới 12/11 với cửa sổ mẫu 25 h (scratchpad `ta_*_ext`; phần trùng khớp, triều lệch ≤ 2,5 mm ở vài giờ dự phòng).
- TA +6 h trên chính TA: R² 93–94% ngoài train.
- Trên nền D + biển: +6 h vẫn 27 lỗi (thêm phần không phải triều cũng vậy), +3 h vẫn 0. **Không giúp**, vì phần TA dự báo được là triều, đã có trong dự báo biển.

**Bước 2: công thức model short của NBS cho DV +6 h.**
- Dataset `train_data/da_vien/dv_h6_all` (config `dataset_config_h6_all.json`): nhóm tide, rain_obs, station_obs, sea, wind 4 kênh, fc_ecmwf, lin_fc (baseline D + biển). Kênh chung giống hệt `dv_h6_fc`; gió giống hệt dataset NBS.
- Công thức: h64×3, seq 12, under-weight 0,3, lr 1e-4, 100 epoch, step.
- Script scratchpad `train_score.sh` (train rồi chấm), `queue_h6_nbsr.sh`, `summarize_h6.py`.

Lỗi ngoài train = val + ranh giới + test + mới; baseline tuyến tính = 27.

| Run (nhóm GRU đọc) | Ít nhất (epoch; train / val / rg / test / mới) | Trung vị | Epoch < 27 | TB 10 epoch cuối (lỗi; tổng) |
|---|---|---|---|---|
| persistence, s23 (gió + triều + ECMWF + biển + NBS quan trắc) | **16** (ep32; 15 / **3** / 1 / 5 / 7) | **19** | 85/100 | 2 / 6 / 0 / 5 / 7; 18 |
| tuyến tính, s42 (như trên + lin_fc) | 19 (ep45; 4 / 14 / 0 / 3 / 2) | 22 | 85/100 | 1 / 14 / 0 / 5 / 1; 20 |
| chỉ nhóm NBS, persistence, s23 (không NBS quan trắc) | 19 (ep97) | 25 | 61/100 | 7 / 3 / 3 / 9 / 8; 23 |
| tuyến tính, s23 | 20 (ep5) | 24 | 90/100 | 2 / 15 / 0 / 9 / 1; 25 |
| persistence, s42 | 21 (ep44) | 25 | 60/100 | 2 / 10 / 0 / 7 / 8; 25 |

- **Ensemble 5 run** (10 epoch cuối mỗi run): 0 / 8 / 0 / 2 / 2, tổng **12**; MAE val / mới 45,7 / 40,8.
- **Ensemble 5 run này + 2 run tốt nhất của Task 11:** 1 / 7 / 0 / 0 / 2, tổng **11**; MAE 44,7 / 39,5.
- **Lỗi còn lại của ensemble 5 run:**
  - val 03/08 17–18 h (+206, +233); 11/09 12–14 h (+226…+252); 13/09 05–07 h (+219…+248). Đều dự báo thừa.
  - test 21/09 01 h (−200) và 06/10 00 h (−201): sát ngưỡng.
  - mới 09/11 16 h (−218) và 11/11 18 h (+204).
- **Nhận xét:**
  - Công thức NBS tốt hơn công thức h32 (trung vị 19–25 so với 24,5–29).
  - Seed ảnh hưởng nhiều (s23 / s42: 19 / 25).
  - NBS quan trắc vẫn có ích trong GRU (bỏ đi: trung vị 25, test và mới kém hơn).
  - Chưa run nào đạt 0. Các lỗi còn lại phần lớn chỉ vượt ngưỡng 0–50 mm.

**Bước 3 (đã chuẩn bị, chưa chạy, chờ người dùng):**
- Code: nhóm `reservoir` (`features.reservoir_features`, `build.reservoir_ops_from_db`); nhóm `station_obs_extra`; tùy chọn `surge` trong station_obs (nước dâng = mực nước − triều 7 ngày của trạm đó).
- Test `test_reservoir_features.py` (2 test). Bản vector hóa khớp tuyệt đối với vòng lặp của Task 12b.
- Config `models/da_vien/dataset_config_h6_rt.json`: dv_h6_all + Thuận An (có nước dâng) + Tứ Phú + xả 3 hồ.

**Bước 3 (đã chạy theo yêu cầu người dùng): dataset `train_data/da_vien/dv_h6_rt`** (config `dataset_config_h6_rt.json`; 50 kênh thêm vào `dv_h6_all`, các kênh cũ giống hệt).
- **Lỗi dữ liệu mới phát hiện, đã xóa bằng `invalid_ranges`** (đều trong train):
  - Tứ Phú 21/04/2026 13:40–15:00 VN: số đo nhảy 740–1.898 mm trong 80 phút;
  - Thuận An 03/06/2026 12:10–14:50 VN: +250 mm giữ khoảng 3 h rồi tụt về.
- **Hiện tượng thật, giữ nguyên:**
  - Tứ Phú lên tới 3.600 mm ngày 10/11/2026 (lũ sông Bồ, khớp lúc Hương Điền tăng xả);
  - nước dâng ở Thuận An khoảng 1.000 mm ngày 19/11/2025.
- Công thức như bước 2, GRU đọc thêm station_obs_extra (Thuận An, có nước dâng; Tứ Phú) và reservoir:

| Run | Ít nhất | Trung vị | Epoch < 27 | TB 10 epoch cuối (train / val / rg / test / mới; tổng) | So với bước 2 (trung vị) |
|---|---|---|---|---|---|
| persistence, s23 | 31 | 35 | 0 | 1 / 6 / 0 / 4 / **23**; 33 | 19 |
| tuyến tính, s23 | 18 | 35 | 6 | 2 / 18 / 0 / 5 / 10; 33 | 24 |
| persistence, s42 | 26 | 37 | 1 | 0 / 10 / 0 / 6 / **21**; 37 | 25 |
| tuyến tính, s42 | 19 | 25 | 74 | 2 / 16 / 0 / 2 / 5; 23 | 22 |

- **Kết luận: bước 3 kém hơn bước 2.** Ensemble 4 run: 1 / 7 / 0 / 0 / 13, tổng 20 (bước 2: 12).
- **Nguyên nhân: lũ sông Bồ 10/11/2026.** Các run bước 3 dự báo thừa +367…+490 mm (trung vị) ở 10/11 trong 5–11 giờ, rồi thừa +212…+234 mm ngày 11/11.
  - Trên train chỉ có trận lũ 11/2025: Tứ Phú và xả hồ lên cao thì DV lên rất mạnh. GRU học quy tắc này từ đúng một trận.
  - Ngày 10/11/2026, sông Bồ lũ lớn (Tứ Phú 3.600 mm, Hương Điền xả 1.561 m³/s) nhưng DV trên sông Hương lên ít hơn nhiều.
  - Lại là học thuộc từ một sự kiện, cùng kiểu với Task 6 và Task 12f.
- Val không cải thiện (persistence s23: 6 lỗi, giống bước 2), tức Thuận An và nước dâng không sửa được bão 11 và 13/09.

## Task 14. Finetune model NBS short (`combined_v1/short_3h_6h`, ep59) sang DV +6 h (người dùng yêu cầu)

- **Code:** `src/ai/finetune.py` (`init_from_checkpoint`) và `train.py --init-checkpoint`.
  - Nạp mọi trọng số; head chỉ giữ hàng của các mốc đích (+3/+6/+12 → +6).
  - Giữ hệ số persistence và bộ chuẩn hóa đầu vào của dataset mới.
  - Báo lỗi nếu cột đầu vào hoặc kiến trúc khác nhau.
  - Test `src/ai/tests/test_finetune.py` (3 test).
- **Dataset `train_data/da_vien/dv_h6_nbsft`** (config `dataset_config_h6_nbsft.json`): đúng 39 cột của model NBS, cùng thứ tự (fc_ecmwf, wind 4, tide7 fit trên DV, sea); giá trị trùng `dv_h6_all`.
- **Kết quả** (GRU 64×3, station head, embedding 8, under-weight 0,3, seed 23; lỗi ngoài train = val + ranh giới + test + mới):

| Run | Ít nhất | Trung vị | TB 10 epoch cuối (train / val / rg / test / mới; tổng) | MAE val / mới |
|---|---|---|---|---|
| epoch 0 = model NBS dùng thẳng cho DV | 209 | | 290 / 38 / 0 / 72 / 99 | 85,6 / 117,9 |
| finetune lr 1e-4, 50 ep | 27 | 35 | 9 / 15 / 2 / 8 / 10; 35 | 53,2 / 51,9 |
| finetune lr 3e-5, 50 ep | 27 | 30 | 18 / 10 / 3 / 8 / 7; 28 | 53,6 / 51,6 |
| train từ đầu, cùng kiến trúc, 100 ep | 24 | 31 | 5 / 13 / 3 / 6 / 11; 33 | 50,1 / 51,1 |

- **Kết luận: finetune không giúp.**
  - Model NBS dùng thẳng cho DV rất kém (tệ hơn persistence ở test và mới). Trọng số NBS không áp được cho DV dù đầu vào được chuẩn hóa theo DV.
  - Sau finetune: ít nhất 27 lỗi, bằng mức baseline tuyến tính, và kém hơn bước 2 (trung vị 19–25, có NBS quan trắc).
  - Bộ 39 kênh này không có NBS quan trắc, nên kể cả train từ đầu cũng kém hơn (trung vị 31).

## Task 15. Một model dự báo song song +3 h và +6 h (người dùng yêu cầu)

- **Dataset `train_data/da_vien/dv_h36_all`** (config `dataset_config_h36_all.json`): giống hệt `dv_h6_all` (cùng mẫu, cùng kênh, `lin_d6` giống hệt), thêm nhãn +3 h và `lin_d3` (D + biển cho +3 h).
- **Công thức như bước 2 Task 13** (h64×3, seq 12, under-weight 0,3, 100 epoch, lr 1e-4), `--target-horizons 3 6`.
- **+6 h** (lỗi ngoài train = val + ranh giới + test + mới):

| Run | Ít nhất (epoch; train / val / rg / test / mới) | Trung vị | Epoch < 27 | TB 10 epoch cuối; tổng | Cùng run khi chỉ dự báo +6 h (ít nhất / trung vị) |
|---|---|---|---|---|---|
| persistence, s23 | **14** (ep98; 3 / 5 / 0 / 5 / 4) | **18** | 84 | 1 / 5 / 0 / 5 / 6; 16 | 16 / 19 |
| tuyến tính, s23 | 19 | 21 | 97 | 3 / 15 / 0 / 6 / 1; 22 | 20 / 24 |
| persistence, s42 | 19 | 25 | 70 | 1 / 8 / 0 / 7 / 8; 23 | 21 / 25 |
| tuyến tính, s42 | 21 | 27 | 44 | 3 / 16 / 0 / 8 / 5; 29 | 19 / 22 |

  Ensemble 4 run (10 epoch cuối): 3 / 10 / 0 / 1 / 1, tổng **12**; bằng ensemble 4 run tương ứng chỉ dự báo +6 h (1 / 10 / 0 / 0 / 1, tổng 11).
- **+3 h trong cùng model:**
  - 3 / 4 run có nhiều epoch đạt 0 lỗi ngoài train (79 và 70 epoch ở tuyến tính s23 và persistence s42).
  - Ensemble 4 run: train 1, ngoài train 0; MAE val / mới 25,3 / 27,3 (`h3_v1`: train 4, ngoài train 0; 23,7 / 28,0).
- **Kết luận:**
  - Học chung với +3 h không làm +6 h kém đi, và giúp nhẹ ở 2 / 4 run (seed 23). Khác biệt nằm trong mức dao động do seed.
  - +6 h vẫn còn khoảng 10 lỗi ở val (bão 11 và 13/09).
  - Phần +3 h của model chung đạt chuẩn (0 lỗi ngoài train) mà không cần chọn epoch, nếu lấy ensemble.

## Task 16. Cửa sổ lịch sử 25 h và 48 h (người dùng yêu cầu; cấu hình như Task 15)

- **Thời gian train:** đo trước khi chạy, mỗi epoch 1,76 s, trong đó:
  - nạp dữ liệu 0,19 s (~11%);
  - GPU 0,35 s (~20%);
  - còn lại khoảng 1,2 s là chấm val, `--diagnose-change-dynamics` (chấm train + val, ghi checkpoint mỗi epoch) và MLflow.
  - Nút thắt không phải ở vận chuyển dữ liệu. Người dùng quyết định không tối ưu.
- **+6 h** (lỗi ngoài train = val + ranh giới + test + mới):

| Cửa sổ | Run | Ít nhất | Trung vị | Epoch < 27 | TB 10 epoch cuối; tổng |
|---|---|---|---|---|---|
| 12 h | persistence s23 / s42 | 14 / 19 | 18 / 25 | 84 / 70 | 16 / 23 |
| | tuyến tính s23 / s42 | 19 / 21 | 21 / 27 | 97 / 44 | 22 / 29 |
| 25 h | persistence s23 / s42 | **12** / 15 | **17** / 24 | 87 / 78 | 17 / 28 |
| | tuyến tính s23 / s42 | 20 / 17 | 21 / 22 | 98 / 91 | 20 / 23 |
| 48 h | persistence s23 / s42 | 14 / 15 | 21 / 18 | 89 / 88 | 19 / 18 |
| | tuyến tính s23 / s42 | 18 / 17 | 21 / 21 | 98 / 91 | 22 / 21 |

  - Run đơn tốt nhất: cửa sổ 25 h, persistence, s23, epoch 32: train / val / rg / test / mới = 11 / 3 / 0 / 4 / 5 (tổng 12).
  - Ensemble 4 run theo cửa sổ (10 epoch cuối, val / rg / test / mới): 12 h 10 / 0 / 1 / 1 (12); 25 h 9 / 0 / 1 / 0 (**10**); 48 h 10 / 0 / 1 / 1 (12). Cả 12 run: 10 / 0 / 1 / 0 (11).
- **+3 h:** ensemble cả 12 run đạt **0 lỗi ở mọi tập, kể cả train**; MAE val / mới 25,4 / 27,2. 12 / 12 run có epoch đạt 0 lỗi ngoài train.
- **Kết luận:**
  - Cửa sổ dài hơn chỉ đổi rất ít; 25 h nhỉnh nhất, nhưng trong mức dao động do seed.
  - Val +6 h vẫn kẹt ở khoảng 9–10 lỗi (bão 11 và 13/09) với mọi cửa sổ, mọi seed, mọi baseline.

### So sánh model +6 h tốt nhất (`dv_h36_all/h36_pers_seq25_s23` ep32) với NBS +6 h (`combined_v1` short) tại cùng thời điểm

- **Hình:**
  - `figures/da_vien/h6_forecast_h36_pers_seq25_s23_ep32.png`;
  - `figures/nga_ba_sinh/h6_forecast_combined_v1_dv_error_windows.png` (cùng 6 đoạn);
  - script `research/scripts/plot_station_forecast.py` (thêm `--events`).
- **Tương quan sai số DV và sai số NBS trên mọi giờ ngoài train:** 0,74. Ở cả 12 giờ DV lỗi, NBS sai cùng chiều. MAE ngoài train: DV 48,0, NBS 53,8.
- **Hai loại lỗi:**

| Giờ (VN) | Δ6 thật DV / NBS | Lỗi DV / NBS | Loại |
|---|---|---|---|
| 11/09 12–14 h (val) | −178…−316 / −221…−267 | +238…+278 / +141…+185 | chung: cả hai báo thừa khi nước rút sâu trong bão |
| 21/09 00–01 h (test) | +174, +234 / +133, +172 | −203, −210 / −124, −112 | chung: cả hai lên chậm |
| 13/10 05–06 h (test) | −70, +42 / −126, +5 | −235, −241 / −31, −61 | riêng DV |
| 29/10 00–02 h (mới) | +141…−102 / −58…−258 | −228…−250 / −46…−69 | riêng DV: DV và NBS lệch pha nhau |
| 10/11 10–11 h (mới) | +197, +137 / +82, −19 | −217, −223 / −32, −54 | riêng DV (lũ sông Bồ) |

- **Cách đọc:**
  - Loại "chung": thông tin đầu vào không đủ để thấy trước, ở cả hai trạm. Lỗi DV lớn hơn vì biên độ ở DV lớn hơn.
  - Loại "riêng DV": NBS dự báo đúng, nhưng diễn biến ở DV khác NBS (lệch pha khoảng vài giờ).

## Task 17. Người dùng train 7 run (21:13–21:38): under-weight 0, trọng số mốc, h128, thêm +12 h

Cấu hình gốc: `dv_h36_all`, cửa sổ 25 h, persistence, lr 1e-4, 100 epoch.
- **Lưu ý về thư mục:**
  - Thư mục `..._u0_s19_horizonweight12.5` thực ra chạy **seed 17**.
  - `..._u0_s23_horizonweight12.5` có 3 run MLflow; checkpoint là của run 21:21 (seed 23, input dropout 0,1), hai lần sau lỗi ngay khi khởi động.
  - `_id01` = input dropout 0.

**+6 h** (lỗi ngoài train = val + ranh giới + test + mới):

| Run | Ít nhất (epoch; train / val / rg / test / mới) | Trung vị | TB 10 epoch cuối; tổng |
|---|---|---|---|
| gốc: under-weight 0,3, s23 (Task 16) | 12 (ep32; 11 / 3 / 0 / 4 / 5) | 17 | 1 / 6 / 0 / 6 / 5; 17 |
| **under-weight 0, s23** | **10** (ep61; 6 / 4 / 1 / 4 / 1) | **15** | 0 / 3 / 0 / 5 / 4; **12** |
| under-weight 0, trọng số +3/+6 = 1/2,5, s23 | 10 (ep86; 4 / 3 / 0 / 5 / 2) | 16 | 0 / 4 / 0 / 5 / 3; 12 |
| như trên, seed 17 | 22 | 28 | 0 / 14 / 0 / 5 / 10; 29 |
| như trên, input dropout 0 | 15 | 21 | 0 / 10 / 2 / 5 / 5; 22 |
| như trên, h128 | 13 | 22 | 0 / 12 / 0 / 4 / 5; 21 |
| +3/+6/+12, trọng số bằng nhau, u0 | 12 | 18 | 8 / 8 / 2 / 5 / 3; 18 |
| +3/+6/+12, trọng số 1/2,5/1, h128, u0 | **8** (ep35; 4 / 5 / 0 / 2 / 1) | 19 | 2 / 11 / 0 / 4 / 5; 20 |

- **Under-weight 0 tốt hơn 0,3** (cùng seed 23): ít nhất 10 so với 12, trung vị 15 so với 17, TB 10 epoch cuối 12 so với 17.
- Trọng số 2,5 cho +6 h không đổi nhiều (trung vị 16).
- Seed 17, input dropout 0 và h128 đều kém hơn. Seed vẫn là yếu tố lớn nhất.
- Thêm +12 h vào model: +6 h hơi kém hơn (trung vị 18–19); riêng h128 có một epoch lẻ đạt 8 (ep35), là epoch đơn tốt nhất đến nay, nhưng trung vị 19.
- **Ensemble:** 8 run trên 0 / 5 / 0 / 4 / 4 (tổng 13); 2 run u0 s23 tốt nhất 0 / 4 / 0 / 5 / 4 (tổng 13).
- **+3 h:** mọi run đều có epoch 0 lỗi ngoài train. Ensemble 8 run: 0 / 0 / 0 / 0 / 1.
- **+12 h trong model chung kém xa NBS long:** ít nhất 37–44, trung vị 52–59 lỗi ngoài train (NBS `combined_v1` +12 h: 0).

## Task 18. Kiểm chứng: DV khó hơn NBS vì biến động lớn hơn? (người dùng đặt giả thuyết; 18/11/2026)

**Giả thuyết của người dùng:** cùng thông tin mà NBS đạt 0 lỗi vì NBS biến động ít hơn; sai số NBS đã sát ±200 mm, nên DV chỉ cần biến động thêm khoảng 50 mm là vượt ngưỡng.

**Cách làm** (script scratchpad `levels.py`, `predict.py`): mực nước giờ từ DB tới 12/11 (despike, bỏ đoạn sensor đứng, bỏ cặp giờ vắt qua lần đổi mốc NBS 10/10). Dựng lại dataset đánh giá tới 12/11 (`--normalizers-from`). Dự báo mọi giờ của NBS `combined_v1` và DV `dv_h36_all/h36_pers_seq25_u0_s23` ep61; khớp đúng số liệu cũ (NBS 0 lỗi; DV 6 / 4 / 1 / 4 / 1).

**Độ biến động, ngoài train (val + ranh giới + test + mới), Δh = mực nước(t+h) − mực nước(t):**

| Mốc | SD Δ DV / NBS | Tỷ lệ | Lỗi persistence > 200 DV / NBS |
|---|---|---|---|
| +3 h | 101 / 98 | 1,03 | 178 / 159 |
| +6 h | 149 / 145 | 1,03 | 411 / 414 |
| +12 h | 106 / 93 | 1,14 | 150 / 76 |

Theo tập, tỷ lệ +6 h: val 1,08, test 1,00, mới 1,03. Ở +6 h DV hầu như không biến động hơn NBS; ở +12 h thì có (thêm 14%).

**Biên an toàn của NBS** (nhân sai số `combined_v1` với hệ số k, đếm lỗi > 200 ngoài train):

| k | 1,03 | 1,05 | 1,08 | 1,10 | 1,15 | 1,20 |
|---|---|---|---|---|---|---|
| +6 h | 0 | 0 | 4 | 8 | 21 | 30 |
| +12 h | 5 | 6 | 8 | 12 | 18 | 30 |

NBS +6 h có 62 giờ lỗi 150–188 mm; +12 h có 6 giờ trên 190 mm. Biên rất mỏng: đúng như người dùng nói.

**So sai số DV và NBS ở +6 h, cùng giờ, ngoài train (2.506 giờ):**
- DV không kém hơn về trung bình: MAE 49,6 so với 53,8; skill so với persistence 0,81 so với 0,79. Cả hai đều nén biên độ như nhau (độ dốc dự báo/thật 0,77 và 0,76).
- DV kém hơn ở đuôi: > 150 mm 102 giờ so với 62; > 200 mm 10 so với 0.
- Tương quan sai số 0,76; sai số DV ≈ 0,82 × sai số NBS + phần riêng DV (SD 41 mm). SD của (Δ6 DV − Δ6 NBS) là 55 mm.
- Ở 10 giờ DV lỗi (giờ phát hành), phần riêng DV = sai số DV − 0,82 × sai số NBS:
  - 14/09 15 h (ranh giới): +57; NBS đã sai 176. **Đúng kiểu "thêm khoảng 50 mm".**
  - 11/09 12–14 h, 21/09 00–01 h: ±97…145; NBS sai 112–185. Dù vậy Δ6 DV không lớn hơn NBS (11/09 12 h: −178 so với −221).
  - 13/09 05 h, 13/10 05–06 h, 29/10 01 h: ±156…194; NBS chỉ sai 30–63. Gần như toàn bộ là phần riêng DV (DV lệch pha so với NBS; 29/10: DV +14, NBS −177).
- Hiệu chỉnh bằng sai số đã biết lệch 24 h: không giúp (β fit trên val làm val/test/mới tệ hơn: 6 / 6 / 7 so với 4 / 4 / 1).

**Kết luận:**
- Người dùng đúng ở cơ chế: NBS chỉ còn biên 12–15 mm ở +6 h và 2–5 mm ở +12 h, nên chỉ cần thêm một thành phần nhỏ là vượt 200.
- Nhưng ở +6 h, thành phần thêm đó **không phải biên độ chung lớn hơn** (DV và NBS dao động như nhau). Nó là **phần riêng của đoạn sông NBS → DV** (SD khoảng 40–55 mm, lên tới 150–190 mm ở các giờ lỗi), cộng dồn lên phần sai số chung với NBS.
- Ở +12 h thì đúng là DV biến động hơn 14%, nên +12 h của DV sẽ khó hơn NBS ngay cả khi model tốt như nhau.

### Task 18b. Cộng/trừ ±41 mm vào các giờ DV hay sai lớn? (người dùng đề xuất; script scratchpad `corr41.py`)

Model gốc: DV u0 s23 ep61, +6 h; lỗi train / val / rg / test / mới = 6 / 4 / 1 / 4 / 1.
- **Giới hạn trên (biết trước giờ sai > 200 và dấu, trừ 41 mm):** 1 / 2 / 0 / 0 / 0, sửa được 8/10 lỗi ngoài train.
- **Biết được KHI NÀO thì có thể:** mưa đo 24 h (3 cụm) vượt phân vị 90 của train gắn cờ 478/2.506 giờ ngoài train và bắt được 9/10 giờ lỗi (AUC 0,89; mưa ECMWF 0–6 h AUC 0,85; |tide_d6| 0,74; |dự báo thay đổi| chỉ 0,57).
- **Biết được VỀ PHÍA NÀO thì không:** trong nhóm có cờ, sai số trung bình +3 mm; giờ sai > 150 mm có 33 giờ dương, 39 giờ âm.
- **Các cách sửa thử (không train), lỗi ngoài train (gốc 10):**
  - ±41 mm theo chiều dự báo thay đổi, chỉ trong giờ mưa: 30–33;
  - giãn biên độ (bù độ dốc 0,77), chọn trên val: 18;
  - đẩy 41 mm về phía dự báo NBS khi hai bên bất đồng > 100 mm: 10 (chỉ dời lỗi giữa các tập: val 3, test 2, mới 4);
  - trộn với NBS 25%/50%: 15/20;
  - hồi quy tuyến tính dự đoán sai số: fit train 10–14, fit val 22–31.
- **Kết luận:** thiếu thông tin về chiều sai, không phải về thời điểm. Lỗi tập trung trong giờ mưa lớn, ủng hộ hướng thêm feature về dòng chảy sông.

## Task 19. Sàng lọc tuyến tính 3 nhóm feature mới (người dùng yêu cầu; 18/11/2026; scripts scratchpad `prep_123.py`, `screen_123.py`)

**Nhóm feature** (đều biết tại t; NBS dùng mốc cũ, +172 mm sau 10/10, như `station_obs`):
1. **Dòng chảy / độ dốc mặt nước:**
   - 1a: `fl_dvres` = DV − triều DV; `fl_grad` = (DV − triều DV) − (NBS − triều NBS, fit 7 ngày cùng cách); thay đổi 1/3/6/12 h của cả hai.
   - 1b: mực nước DV trung bình 25 h (lọc triều) so với hiện tại và thay đổi 6/12/24 h; chênh lệch DV − NBS trung bình 25 h và thay đổi 6/12/24 h.
2. **Pha triều** (đường triều 7 ngày của DV cho t..t+14, tái tạo khớp `tide_d6` tới 1e-13): số giờ tới đỉnh/chân triều kế tiếp và độ cao so với hiện tại, biên độ 12 h tới, độ dốc tại t, t+6, t+12, độ cong tại t+6.
3. **Mưa suy giảm** (API, τ = 12/24/72 h): 3a trung bình lưu vực (có thêm bản cộng mưa ECMWF 0–6 h); 3b từng cụm (east/west/plain). log1p.

**Nền "D + biển"** (giống `lin_d6`, tái tạo khớp tuyệt đối). Lỗi > 200 mm train / val / rg / test / mới; tổng ngoài train:

| Thêm | +6 h | +12 h | +3 h |
|---|---|---|---|
| (nền) | 33 / 14 / 1 / 4 / 8; **27** | 118 / 35 / 3 / 42 / 34; **114** | 14 / 0…; 0 |
| 1a | 32 / 15 / 0 / 3 / 6; 24 | 111 | 0 |
| 1b | 28 | 113 | 0 |
| 1 (1a + 1b) | 28 | 107 | 0 |
| 2 | 28 | 111 | 0 (MAE mới 29,8 → 27,9) |
| 3a | 26 | 117 | 0 |
| 3b | 33 | 133 | 0 |
| 1 + 2 | 27 | 105 | 0 |
| 1 + 2 + 3a | 36 / 13 / 0 / 2 / 7; 22 | 101 | 0 |

- +6 h: thay đổi trong khoảng 22–28, R² val giảm nhẹ (52,5 → 51–52). Không có cải thiện rõ.
- +12 h: R² val tăng (0,8 → 18–21 với nhóm 1) nhưng R² mới giảm (54,5 → 47–50). Lẫn lộn.
- **Nhóm 1 có thông tin mới về phần riêng DV:** dự báo Δ6 DV − Δ6 NBS, R² val / test / mới từ 21 / 74 / 74 lên **40 / 80 / 81**. Nhóm 2 và 3 gần như không thêm (24 / 73 / 77 và 18 / 74 / 74). Phần này chỉ có SD 53 mm nên tổng số lỗi gần như không đổi.
- **Ở 10 giờ lỗi của GRU +6 h** (z-score ngoài train): `fl_grad_d12` rất lớn ở 13/09 05 h (+6,7, đúng chiều), vừa ở 14/09 (+1,3) và 11/09 (+0,4…+1,1); đúng chiều nhẹ ở 21/09 và 13/10 (−0,6…−1,1); **sai chiều ở 29/10** (+2,0 trong khi lỗi âm).
- `ph_curv6` (độ cong triều tại t+6) tương quan 0,30 với sai số GRU trên mọi giờ ngoài train (GRU làm tròn đỉnh/chân triều), nhưng không nổi bật ở giờ lỗi.
- Hiệu chỉnh hậu kỳ sai số GRU bằng các feature này (fit trên val): test / mới tệ đi (4 / 1 → 6–14 / 13–18).

## Task 20. GRU với 3 nhóm feature mới, từng nhóm riêng và cả ba (người dùng yêu cầu; 18/11/2026)

Người dùng: GRU có thể dùng thông tin khác với tuyến tính, nên thử cả 3 nhóm của Task 19 trên GRU dù sàng lọc tuyến tính không giúp rõ.

**Code** (`features.py`, `build.py`; test `src/ai/tests/test_flow_phase_api.py`, 4 test; khóa config ghi trong `PIPELINE.md` mục 3):
- nhóm `flow` (`flow_features`, 18 kênh + cờ);
- nhóm `tide_phase` (`tide_phase_features`, 9 kênh + cờ);
- nhóm `rain_api` (`rain_api_features`, 6 kênh + cờ: API lưu vực τ 12/24/72 h, và API + ECMWF 0–6 h).

**Dataset** (config `models/da_vien/dataset_config_h36_123.json` = `dataset_config_h36_all.json` + 3 khóa mới), trong `train_data/da_vien/`:

| Dataset | Nhóm thêm vào `dv_h36_all` | Đầu vào GRU (đọc wind+tide+fc_ecmwf+sea+station_obs + nhóm mới) |
|---|---|---|
| `dv_h36_flow` | flow | 72 |
| `dv_h36_tph` | tide_phase | 63 |
| `dv_h36_api` | rain_api | 60 |
| `dv_h36_123` | cả ba | 89 |

- Kênh cũ và tập mẫu giống hệt `dv_h36_all` (sai khác 0); `lin_d3/lin_d6` không đổi. Run gốc đọc 53 đầu vào.
- Kênh mới khớp tuyệt đối với bản sàng lọc Task 19.
- Bản đánh giá tới 12/11 trong scratchpad (`dv_h36_*_ext`, dựng với `--normalizers-from`); phần trùng khớp tới 1e-13.

**Lệnh train:** công thức của run gốc `dv_h36_all/h36_pers_seq25_u0_s23` (thông số lấy từ MLflow): h64×3, cửa sổ 25 h, persistence, under-weight 0, step, lr 1e-4, 100 epoch, seed 23. Chạy 2 run một lúc:
```bash
run() {  # $1 = dataset suffix, $2 = extra groups
.venv/bin/python -m src.ai.train --mlflow-ex-name da_vien --horizon 1h \
  --dataset-dir train_data/da_vien/dv_h36_$1 --target-horizons 3 6 \
  --rain-stations wind+tide+fc_ecmwf+sea+station_obs$2 --use-weather-forecast false --feature-set water_level+rain+flood_season \
  --train-method gru --head-mode shared --station-embedding-size 0 --lstm-seq-len 25 --lstm-hidden-size 64 --lstm-num-layers 3 \
  --lstm-drop-out 0.2 --lstm-input-dropout 0.1 --lstm-head-dropout 0.1 --lstm-lr 0.0001 --lstm-weight-decay 0.001 \
  --lr-scheduler step --batch-size 64 --num-epochs 100 --early-stopping 1000 \
  --change-loss combined --change-under-weight 0 --change-event-weight 0 --change-margin-mm 0 \
  --lstm-checkpoint-metric val_loss --skip-test --seed 23 --diagnose-change-dynamics \
  --checkpoint-dir model/experiments/da_vien/dv_h36_$1/h36_pers_seq25_u0_s23
}
run flow +flow & run tph +tide_phase & wait
run api +rain_api & run 123 +flow+tide_phase+rain_api & wait
```
So sánh với run gốc u0 s23: ít nhất 10 (ep61), trung vị 15, TB 10 epoch cuối 12 lỗi +6 h ngoài train.

### Kết quả Task 20 (người dùng train 11 run; Claude chấm mọi epoch trên mọi giờ; script scratchpad `score_runs.py`)

Không có run `tide_phase` riêng. Lỗi +6 h ngoài train = val + rg + test + mới; ensemble = trung bình 10 epoch cuối (train / val / rg / test / mới; tổng).

| Run | Ít nhất (epoch; train / val / rg / test / mới) | Trung vị | TB 10 epoch cuối | Ensemble 10 epoch cuối | +3 h: số epoch 0 lỗi |
|---|---|---|---|---|---|
| **gốc `dv_h36_all` u0 s23** | **10** (ep61; 6 / 4 / 1 / 4 / 1) | **15** | **12,9** | 0 / 3 / 0 / 5 / 4; **12** | 15 |
| flow s23 | 19 (ep22) | 27 | 27,3 | 0 / 14 / 0 / 8 / 4; 26 | 0 |
| flow s42 | 19 (ep21) | 29 | 31,8 | 0 / 11 / 0 / 10 / 9; 30 | 14 |
| flow s17 | 17 (ep52) | 24 | 25,6 | 0 / 13 / 0 / 6 / 6; 25 | 11 |
| rain_api s23 | 18 (ep44) | 25 | 29,9 | 1 / 16 / 0 / 8 / 7; 31 | 0 |
| rain_api s17 | 15 (ep24) | 24 | 27,5 | 4 / 11 / 0 / 10 / 6; 27 | 0 |
| rain_api s17 h128 (70 ep) | 18 | 30 | 35,3 | 34 | 3 |
| rain_api s17 h128 ev1 | 17 | 26 | 31,1 | 34 | 4 |
| rain_api s17 h128 ev2 | 15 | 23 | 26,1 | 25 | 6 |
| cả ba s23 | 16 (ep23) | 24 | 26,3 | 0 / 11 / 0 / 9 / 6; 26 | 7 |
| cả ba s23 ev1 | 17 (ep52) | 25 | 26,6 | 29 | 3 |
| cả ba s17 | 23 (ep74) | 28 | 27,1 | 27 | 0 |

- **Cùng seed 23, mọi nhóm đều kém hơn rõ:** trung vị 24–27 so với 15; ensemble 26–31 so với 12. Phần kém thêm chủ yếu ở val (11–16 lỗi so với 3).
- Lỗi mới nằm trong bão 11–13/09 (thêm 12/09 báo thiếu, 13/09 cả hai chiều) và 06/10. Feature mới làm GRU phản ứng mạnh hơn nhưng sai chiều trong bão, đúng như dấu hiệu sai chiều ở Task 19.
- Seed 17 và 42 không có run gốc cùng cấu hình (gần nhất: u0 s17 trọng số 1/2,5, Task 17: 22 / 28 / 29). Các run s17 có cỡ lỗi tương đương, không tốt hơn.
- event weight 1–2 với h128: không giúp rõ.
- +3 h cũng kém hơn: gốc có 15 epoch đạt 0 lỗi ngoài train; run mới 0–14.
- **Kết luận: cả ba nhóm (flow, rain_api, cả ba) đều không giúp GRU.** Giữ `dv_h36_all` làm nền.

## Task 21. Lưu lượng sông Hương và sông Bồ từ Open-Meteo Flood API (GloFAS v4) (người dùng yêu cầu; 18/11/2026; script scratchpad `screen_q.py`)

**Dữ liệu:**
- `data/external/open_meteo/glofas_v4_grid64_20251101_20261117.json`: lưới 8×8 điểm (16,25–16,60 N, 107,30–107,65 E, bước 0,05°), từ 01/11/2025 đến 17/11/2026, theo ngày (GMT).
- `data/external/open_meteo/glofas_v4_live_fetched_20261118.json`: dự báo lấy ngày 18/11 cho 3 ô. Lưu lại để sau này đối chiếu xem bản lưu giữ dự báo ở hạn nào.
- Chỉ có số liệu theo ngày. `consolidated_v4` trống cả năm; `seamless_v4` trùng hệt `forecast_v4`, tức bản lưu là dự báo GloFAS, không rõ hạn dự báo và giờ phát hành.
- Mạng sông GloFAS (0,05°):
  - nhánh Hương: 16,275/107,575 → 16,325/107,575 → 16,425/107,525 → **16,475/107,525**;
  - nhánh Bồ: 16,425/107,325 → … → **16,525/107,425**;
  - hai nhánh nhập ở 16,525/107,525; cửa ra 16,575/107,575 (trung bình 298 m³/s).

**GloFAS chậm khoảng 1 ngày so với thực tế:**
- Lũ 11/2025: DV đỉnh 17/11, GloFAS đỉnh 18/11.
- Bão 09/2026: DV đỉnh 12/09, GloFAS đỉnh 13–14/09.
- 29/10: DV đỉnh 28–29/10, GloFAS lên 29–30/10.
- Lũ 10/11: DV đỉnh 10/11, GloFAS đỉnh 11/11.

Tương quan log Q (Hương) với mực nước trung bình ngày của DV chỉ 0,38; với DV − NBS là 0,42. GloFAS không có hồ chứa, không có triều.

**Sàng lọc tuyến tính trên nền D + biển.** Lỗi > 200 mm ngoài train (+6 / +12 / +3 h; nền 27 / 114 / 0):

| Thêm | +6 h | +12 h | +3 h |
|---|---|---|---|
| ngày d−1 (an toàn), Hương + Bồ | 28 | 116 | 0 |
| ngày d | 28 | 118 | 0 |
| ngày d và d+1 | 31 | 126 | 0 |
| d…d+2 | 31 | 117 | 0 |
| chỉ Hương / chỉ Bồ / ô nhập lưu (d−1) | 28 / 27 / 28 | 114–115 | 0 |
| ORACLE: Q nội suy tại t+6/t+12 − t | 31 | 130 | 0 |

R² val luôn giảm (52,5 → 47–51 ở +6 h). **Không giúp, kể cả bản biết trước tương lai.** Lý do: thô theo thời gian (theo ngày), trễ khoảng 1 ngày, và không có hồ chứa. Mình không đề xuất đưa vào GRU.

## Task 22. Hồi quy tuyến tính chồng lên dự báo GRU (người dùng hỏi; 18/11/2026)

Model DV u0 s23 ep61. Đầu vào của hồi quy:
- dự báo thay đổi +3 và +6 h của GRU;
- thay đổi mực nước quan trắc 1/2/3/6 h tới t;
- sai số đã biết tại t: dự báo +3 h phát lúc t−3, +6 h phát lúc t−6, +3 h phát lúc t−24.

Không dùng được mực nước thật tại t+h. Lỗi > 200 train / val / rg / test / mới; MAE:

| Mốc | Cách | Lỗi | MAE test / mới |
|---|---|---|---|
| +3 h | GRU gốc | 0 / 0 / 0 / 0 / 0 | 38,5 / 33,5 |
| | fit train, đủ đầu vào | 0 / 0 / 0 / 0 / **1** | **34,4 / 29,5** |
| | fit val, đủ đầu vào | 2 / 0 / 0 / 0 / 1 | 30,9 / 27,9 (val đã dùng để fit) |
| +6 h | GRU gốc | 6 / 4 / 1 / 4 / 1 | 56,4 / 46,0 |
| | fit train, tốt nhất | 3 / 4 / 0 / 5 / 1 | 55,6 / 44,6 |
| | fit val | 4–9 / 5–7 / 0 / 12–17 / 5–10 | 53–57 / 44–46 |

- +3 h: MAE giảm 4 mm nhưng thêm 1 lỗi > 200 ở giai đoạn mới. +6 h: không giảm lỗi; fit trên val làm test/mới tệ hẳn.
- Lý do: dự báo GRU trên train là trong mẫu (tốt hơn thực tế) nên hồi quy fit trên train gần như không sửa gì; fit trên val thì học theo hai cơn bão của val. Sai số đã biết gần như không báo trước sai số sắp tới (tương quan −0,1…+0,1, Task 18).

## Task 23. Hiệu chỉnh chỉ nhắm vào lỗi lớn (người dùng hỏi; 18/11/2026; script scratchpad `tailfit.py`)

**Cách làm:** hiệu chỉnh tuyến tính cộng vào dự báo GRU (u0 s23 ep61), fit bằng loss chỉ phạt phần sai số vượt τ: Σ max(|e| − τ, 0)², τ = 100 hoặc 150 mm, ridge λ = 1e2 / 1e4. So với bình phương tối thiểu (MSE).
- Đầu vào: bộ chồng của Task 22; hoặc thêm feature Task 19 (dòng chảy, pha triều, mưa tích lũy) cùng mưa ECMWF, mưa đo 24 h, biển.
- Fit trên train, val, hoặc train + val.

**Kết quả** (lỗi > 200 ngoài train; GRU gốc: +3 h 0, lớn nhất 194; +6 h 10, có 102 giờ > 150):
- **+3 h:** không bản nào tốt hơn. Nhiều bản thêm 1–3 lỗi; bản giữ 0 lỗi thì giờ > 150 tăng hoặc giữ nguyên.
- **+6 h:**
  - fit train, hiệu chỉnh mạnh (λ 1e2): 16–19 lỗi (val và test tệ hơn);
  - fit train, hiệu chỉnh gần như bằng 0 (λ 1e4, SD 1–2 mm): 8–10 lỗi. Chỉ đẩy vài giờ sát ngưỡng xuống dưới 200, và là bản tốt nhất trong 32 biến thể, tức chọn sau khi xem;
  - fit val: test và mới 16–35 lỗi;
  - fit train + val: test và mới 8–21 lỗi (gốc 5).
- Số giờ > 150 giảm được (102 → 70–76) chỉ ở các bản fit có val, và đổi lại nhiều lỗi > 200 hơn ở test và mới.

**Kết luận:** đổi loss chỉ đổi mục tiêu tối ưu, không thêm thông tin. Ở giờ sai lớn, đầu vào không cho biết chiều sai (Task 18b), nên hiệu chỉnh hoặc gần bằng 0 hoặc học theo vài cơn bão. Trên train chỉ có 19 giờ (+6 h) sai > 150 mm, vì dự báo GRU trên train là trong mẫu: quá ít để học phần đuôi.

## Task 24. Data augmentation online: phóng to hoặc thu nhỏ phần nước dâng/lũ (người dùng đề xuất; 18/11/2026)

**Ý tưởng:** train chỉ có 1 trận lũ lớn và vài cơn bão, còn GRU nén biên độ trên dữ liệu chưa thấy (độ dốc 0,77). Tạo thêm các phiên bản to hơn hoặc nhỏ hơn của chính các đợt có trong train. Người dùng chọn cách online: mỗi lần nạp mẫu, ngẫu nhiên có hoặc không augment, với hệ số k mới mỗi epoch.

**Code:** `src/ai/augmentation.py` (`SurgeScaleAugmenter`); `train.py --augment-surge-scale KMIN:KMAX:PROB`; `regional_loaders(augment_surge_scale=...)`; test `src/ai/tests/test_augmentation.py` (6 test).
- Mực nước DV = triều (`tide_now`) + phần dư r. Với xác suất PROB, chọn k ~ U[KMIN, KMAX]; r' = r̄ + k(r − r̄), với r̄ là trung bình r trong cửa sổ lịch sử. Phần cộng thêm a = (k − 1)(r − r̄) được cộng vào mực nước lịch sử và vào nhãn +h, nên triều giữ nguyên và nhãn khớp đầu vào.
- NBS giả định nhận cùng phần cộng thêm: `obs_nbs_minus_self` giữ nguyên; `obs_nbs_d<k>` cộng a(τ) − a(τ−k); `flow_dvres*` và `flow_lp_dv*` đổi theo; `flow_grad*` giữ nguyên.
- Mưa, ECMWF, biển, gió, triều, pha triều giữ nguyên. Đây là điểm không nhất quán đã biết: cùng lượng mưa nhưng lũ to hơn hoặc nhỏ hơn.
- Từ chối các kênh `lin_*` (phải fit lại) và `obs_*_surge*`.
- Chỉ áp dụng trong collate của train loader. Val, test và phần chẩn đoán mỗi epoch (dùng loader riêng) vẫn thấy dữ liệu gốc.
- Đã kiểm tra trên `dv_h36_flow` (k = 1,3): phần dư của mực nước lịch sử và của nhãn đúng bằng r̄ + k(r − r̄) (sai khác ≤ 0,05 mm); `flow_lp_dv` khớp với bản tính lại từ mực nước đã augment (≤ 1e-4 mm); kênh ngoại sinh không đổi; k = 1 hoặc PROB = 0 cho dữ liệu y hệt.
- 4 test CLI cũ (`test_change_experiments` ×2, `test_event_aware_no_training`, `test_residual_architectures`) lỗi ở `validate_args` (đọc `args.dataset_dir` khi = None). Lỗi này không thuộc phần augmentation.

**Lệnh train** (nền `dv_h36_all`, công thức run gốc u0 s23 seq25; chỉ thêm `--augment-surge-scale`), mỗi run một lệnh riêng:

1. To và nhỏ, một nửa số mẫu:
```bash
.venv/bin/python -m src.ai.train --mlflow-ex-name da_vien --horizon 1h --dataset-dir train_data/da_vien/dv_h36_all --target-horizons 3 6 --rain-stations wind+tide+fc_ecmwf+sea+station_obs --use-weather-forecast false --feature-set water_level+rain+flood_season --train-method gru --head-mode shared --station-embedding-size 0 --lstm-seq-len 25 --lstm-hidden-size 64 --lstm-num-layers 3 --lstm-drop-out 0.2 --lstm-input-dropout 0.1 --lstm-head-dropout 0.1 --lstm-lr 0.0001 --lstm-weight-decay 0.001 --lr-scheduler step --batch-size 64 --num-epochs 100 --early-stopping 1000 --change-loss combined --change-under-weight 0 --change-event-weight 0 --change-margin-mm 0 --lstm-checkpoint-metric val_loss --skip-test --seed 23 --diagnose-change-dynamics --augment-surge-scale 0.8:1.3:0.5 --checkpoint-dir model/experiments/da_vien/dv_h36_all/h36_pers_seq25_u0_s23_aug0.8-1.3p0.5
```

2. Chỉ phóng to:
```bash
.venv/bin/python -m src.ai.train --mlflow-ex-name da_vien --horizon 1h --dataset-dir train_data/da_vien/dv_h36_all --target-horizons 3 6 --rain-stations wind+tide+fc_ecmwf+sea+station_obs --use-weather-forecast false --feature-set water_level+rain+flood_season --train-method gru --head-mode shared --station-embedding-size 0 --lstm-seq-len 25 --lstm-hidden-size 64 --lstm-num-layers 3 --lstm-drop-out 0.2 --lstm-input-dropout 0.1 --lstm-head-dropout 0.1 --lstm-lr 0.0001 --lstm-weight-decay 0.001 --lr-scheduler step --batch-size 64 --num-epochs 100 --early-stopping 1000 --change-loss combined --change-under-weight 0 --change-event-weight 0 --change-margin-mm 0 --lstm-checkpoint-metric val_loss --skip-test --seed 23 --diagnose-change-dynamics --augment-surge-scale 1.0:1.4:0.5 --checkpoint-dir model/experiments/da_vien/dv_h36_all/h36_pers_seq25_u0_s23_aug1.0-1.4p0.5
```

3. Mọi mẫu đều augment:
```bash
.venv/bin/python -m src.ai.train --mlflow-ex-name da_vien --horizon 1h --dataset-dir train_data/da_vien/dv_h36_all --target-horizons 3 6 --rain-stations wind+tide+fc_ecmwf+sea+station_obs --use-weather-forecast false --feature-set water_level+rain+flood_season --train-method gru --head-mode shared --station-embedding-size 0 --lstm-seq-len 25 --lstm-hidden-size 64 --lstm-num-layers 3 --lstm-drop-out 0.2 --lstm-input-dropout 0.1 --lstm-head-dropout 0.1 --lstm-lr 0.0001 --lstm-weight-decay 0.001 --lr-scheduler step --batch-size 64 --num-epochs 100 --early-stopping 1000 --change-loss combined --change-under-weight 0 --change-event-weight 0 --change-margin-mm 0 --lstm-checkpoint-metric val_loss --skip-test --seed 23 --diagnose-change-dynamics --augment-surge-scale 0.8:1.3:1.0 --checkpoint-dir model/experiments/da_vien/dv_h36_all/h36_pers_seq25_u0_s23_aug0.8-1.3p1.0
```

So sánh với run gốc u0 s23: ít nhất 10 (ep61), trung vị 15, TB 10 epoch cuối 12 lỗi +6 h ngoài train.

### Kết quả Task 24 (người dùng train 2/3 run; Claude chấm mọi epoch trên mọi giờ, 18/11/2026)

Lỗi ngoài train (val + rg + test + mới). Ensemble = trung bình 10 epoch cuối (train / val / rg / test / mới; tổng).

| Run (s23) | +6 h ít nhất | +6 h trung vị | +6 h TB 10 cuối | +6 h ensemble | +3 h: số epoch 0 lỗi | +3 h TB 10 cuối | +3 h ensemble (lớn nhất) |
|---|---|---|---|---|---|---|---|
| gốc | **10** (ep61) | **15** | **12,9** | 0 / 3 / 0 / 5 / 4; **12** | 15 | 1,1 | 1 (213 mm) |
| aug 0,8–1,3 p 0,5 | 11 (ep50) | 15 | 13,9 | 0 / 5 / 0 / 5 / 3; 13 | **43** | 0,4 | **0 (197 mm)** |
| aug 1,0–1,4 p 0,5 | 12 (ep96) | 18 | 15,5 | 3 / 5 / 0 / 5 / 3; 13 | **40** | **0,0** | **0 (188 mm)** |

- **+6 h: không cải thiện** (ngang hoặc kém nhẹ). Các giờ lỗi vẫn như cũ: 11/09, 21/09, 13/10, 28/10, 10/11.
- **Độ nén biên độ không đổi:** độ dốc dự báo/thật +6 h ngoài train 0,76 (gốc 0,77), cả khi chỉ phóng to (k 1,0–1,4). Nén biên độ không phải do thiếu ví dụ lớn. Đó là phản ứng tối ưu của model khi phần biến động không dự báo được: augmentation phóng to cả phần dự báo được lẫn phần không dự báo được theo cùng tỷ lệ, nên mức co về trung bình vẫn giữ nguyên.
- **+3 h ổn định hơn rõ:** số epoch đạt 0 lỗi ngoài train tăng từ 15 lên 40–43; ensemble 10 epoch cuối 0 lỗi, lỗi lớn nhất 188–197 mm (gốc 213). MAE gần như không đổi (33,8–34,3 so với 33,5).
- Mới có 1 seed. Run 3 (0,8–1,3, p 1,0) chưa train.

### Task 24b. Thêm 6 run augmentation (người dùng train; Claude chấm mọi epoch trên mọi giờ, 18/11/2026)

Lỗi ngoài train (val + rg + test + mới): ít nhất / trung vị / TB 10 epoch cuối / ensemble 10 epoch cuối.

| Run | +3 h | +6 h | +12 h |
|---|---|---|---|
| gốc s23 (+3/+6) | 0 / 1 / 1,1 / 1 | **10 / 15 / 12,9 / 12** | — |
| gốc +3/+6/+12 s23 (Task 17) | 0 / 1 / 1,1 / 1 | 12 / 18 / 17,2 / 18 | 37 / 52 / 53,4 / 54 |
| gốc +3/+6/+12 h128 trọng số 1/2,5/1 (Task 17) | 0 / 1 / 0,9 / 1 | 8 / 19 / 20,9 / 20 | 44 / 59 / 51,3 / 49 |
| aug 0,8–1,3 p 1,0 (+3/+6/+12) | 0 / 1 / 1,0 / 1 | 14 / 20 / 17,0 / 17 | 39 / 50 / 51,8 / 51 |
| aug p 1,0 ev1 (+3/+6/+12) | 0 / 1 / 0,2 / 0 | 17 / 28 / 18,8 / 19 | 42 / 61 / 55,1 / 53 |
| aug p 1,0 ev1 h128 | 0 / 1 / 0,8 / 1 | 12 / 17 / 14,9 / 14 | 51 / 72 / 68,8 / 72 |
| **aug p 1,0 ev1 trọng số 1/2,5/1 (mới nhất)** | 0 / **0** / 0,3 / **0** | 12 / 27 / 14,5 / 13 | 47 / 68 / 60,0 / 58 |
| aug p 0,5 s17 / s17 h128 (+3/+6) | 0 / 2 / 2,3 / 2 và 0 / 2 / 2,0 / 2 | 20 / 26 / 26,8 / 30 và 15 / 21 / 24,0 / 23 | — |

**Run mới nhất, xét riêng val** (người dùng thấy 3 lỗi val +6 h và 8 lỗi val +12 h):
- **+6 h:** val giữ ở 3 lỗi suốt epoch 44–75 (ổn định hơn gốc), nhưng cùng epoch đó ranh giới 2–3, test 4–9, mới 3–8. Tổng 12–15, bằng hoặc kém gốc (gốc ep61: val 4, tổng 10).
- **+12 h:** val thấp nhất 6 (ep51), nhưng cùng epoch test 10, mới 28. Tổng ngoài train 45–60 ở mọi epoch.
- Val (31/07–15/09) không đại diện cho +12 h: lỗi +12 h chủ yếu nằm ở test và giai đoạn mới (mùa lũ tháng 10–11).

**Kết luận:**
- Augmentation tiếp tục giúp +3 h (ensemble 0 lỗi, trung vị 0).
- Không giúp +6 h và +12 h. +12 h trong model chung vẫn 37–72 lỗi ngoài train (NBS long: 0; DV biến động hơn NBS 14% ở +12 h, Task 18).
- Seed 17 kém hơn s23, như mọi lần trước.

## Task 25. Đóng gói `models/da_vien/v1` (người dùng yêu cầu; 18/11/2026)

- **Quy tắc định trước:** mỗi mốc lấy trung bình dự báo (mm) của 10 epoch cuối (91–100) của một run; không chọn epoch theo test.
  - +3 h: `dv_h36_all/h36_pers_seq25_u0_s23_aug0.8-1.3p0.5` (augmentation).
  - +6 h: `dv_h36_all/h36_pers_seq25_u0_s23` (không augmentation).
- Thư mục `models/da_vien/v1/h3/ep91…ep100`, `h6/ep91…ep100` (6,5 MB). Thông tin, nguồn, chỉ số và lưu ý trong `package.json`.
- Script dự báo/chấm: `research/scripts/member_ensemble.py package.json <dataset>` (ensemble theo mốc, chạy trên GPU). Đã chạy lại trên dataset tới 12/11, khớp số liệu chấm trước đó:

| Mốc | train | val | ranh giới | test | mới |
|---|---|---|---|---|---|
| +3 h: lỗi; MAE; lớn nhất | 0; 22,1; 193 | 0; 29,6; 166 | 0; 47,8; 116 | 0; 39,3; 170 | 0; 33,5; 197 |
| +6 h | 0; 29,3; 192 | 3; 48,2; 256 | 0; 69,9; 186 | 5; 56,7; 229 | 4; 46,2; 215 |

## Task 26. Dự báo thưa: +3 h mỗi 3 giờ, +6 h mỗi 6 giờ (người dùng hỏi; 18/11/2026)

Dùng dự báo của `v1`, chỉ giữ các giờ phát hành theo nhịp (giờ VN). Lỗi > 200 ngoài train (val + rg + test + mới):

| Mốc | Giờ phát hành | Lỗi | Số lần dự báo | Lớn nhất (mm) |
|---|---|---|---|---|
| +3 h | mỗi giờ | 0 | 2.519 | 197 |
| | 3 cách chọn giờ | 0 / 0 / 0 | 840 mỗi cách | 170 / 197 / 179 |
| +6 h | mỗi giờ | 12 | 2.519 | 256 |
| | 00, 06, 12, 18 | 3 (11/09 12 h, 21/09 00 h, 13/10 06 h) | 420 | 223 |
| | 01, 07, 13, 19 | 2 | 420 | 256 |
| | 02, 08, 14, 20 | 3 | 420 | 241 |
| | **03, 09, 15, 21** | **0** | 420 | **199** |
| | 04, 10, 16, 22 | 1 | 420 | 209 |
| | 05, 11, 17, 23 | 3 | 419 | 216 |

- Model không đổi; tỷ lệ lỗi trên mỗi lần dự báo vẫn khoảng 0,5%. Ít lần dự báo hơn thì ít lỗi hơn (trung bình 12/6 = 2 mỗi cách chọn).
- Cách chọn 03/09/15/21 đạt 0 là nhờ các giờ lỗi tình cờ rơi ngoài nhịp đó, và lớn nhất 199 mm (sát ngưỡng). Chọn giờ sau khi đã xem kết quả; trận bão sau có thể rơi đúng nhịp.
- Dự báo thưa chỉ cho một giá trị mỗi 6 giờ, không thay được dự báo hằng giờ nếu cần theo dõi liên tục.

## Task 27. Định nghĩa mực nước: trung bình giờ hay số đo lúc hh:00? (người dùng hỏi; 18/11/2026)

- **Hiện tại** (mọi dataset DV và NBS, `build.hourly_level`): mực nước giờ t = trung bình các số đo 10 phút hợp lệ trong (t − 1 h, t]. Cả đầu vào lẫn nhãn +h đều theo định nghĩa này, nên nhất quán, nhưng chậm khoảng 25 phút so với số đo mới nhất.
- **Số đo DV** đều mỗi 10 phút tại phút :00, :10, …, :50 (8.302 số đo lúc hh:00).
- **Trung bình giờ − số đo lúc hh:00:** trung bình |sai| 10,7 mm, p95 30, p99 44, lớn nhất 106 mm. 34 giờ lệch > 50 mm, 2 giờ > 100 mm.
- **Chấm `v1` với nhãn là số đo lúc hh:00** thay vì trung bình giờ (ngoài train, 2.519 giờ):

| Mốc | Lỗi > 200 (TB giờ → số đo hh:00) | Lỗi > 150 | MAE |
|---|---|---|---|
| +3 h | 0 → **4** | 12 → 28 | 33,8 → 39,2 |
| +6 h | 12 → **19** | 103 → 105 | 50,5 → 52,0 |

- **Hệ quả:** nếu yêu cầu thật là mực nước tức thời lúc hh:00 thì mọi kết quả đến nay (cả NBS `combined_v1`) đều được chấm theo định nghĩa dễ hơn. Cần người dùng xác nhận định nghĩa trước khi đi tiếp.

## Task 28. Triều 15 / 30 ngày thay cho 7 ngày (người dùng yêu cầu sau khi trao đổi với người hướng dẫn về nhật triều / bán nhật triều; 18/11/2026)

**Bối cảnh:** người hướng dẫn chỉ ra DV có hai đỉnh mỗi ngày, cao thấp khác nhau (triều hỗn hợp). Kiểm tra trên dữ liệu (fit 7 thành phần theo tháng, đã bỏ trung bình 25 h):

| Tháng | M2 / S2 (mm) | K1 / O1 (mm) | F = (K1 + O1) / (M2 + S2) |
|---|---|---|---|
| 03/2026 | 23 / 19 | 28 / 8 | 0,88 (hỗn hợp, thiên bán nhật) |
| 07/2026 | 7 / 6 | 19 / 2 | 1,68 (hỗn hợp, thiên nhật triều) |
| 10–11/2026 | 104 / 30 | 43 / 44 | 0,65 (hỗn hợp, thiên bán nhật) |

NBS giống hệt. Triều 7 ngày hiện tại đã có cả M2, S2, N2, K1, O1, M4, MS4. Nhưng 7 ngày không đủ để tách M2 khỏi S2 (cần khoảng 14,8 ngày) và K1 khỏi O1 (khoảng 13,7 ngày); 15 ngày thì đủ.

**So sánh** (R² % của triều dùng thẳng cho mức thay đổi DV; hồi quy tuyến tính "D + biển" với triều tương ứng; lỗi > 200 ngoài train):

| Mốc | 7 ngày | 15 ngày | 30 ngày |
|---|---|---|---|
| +3 h: R² triều val / test / mới; lỗi tuyến tính | 29 / 74 / 61; 0 | 40 / 79 / 68; 0 | 36 / 69 / 82; 2 |
| +6 h | 49 / 83 / 75; **27** | 47 / 81 / 75; 36 | 40 / 68 / 83; 43 |
| +12 h | 33 / 3 / 35; **114** | 34 / 19 / 32; 117 | 35 / 13 / 43; 122 |

- Ở dạng triều riêng lẻ, 15 ngày tốt hơn ở +3 h và +12 h test. Trong hồi quy tuyến tính, 7 ngày tốt nhất ở +6 h (27 so với 36 / 43).
- **Lý do:** biên độ triều DV đổi theo mùa (M2 7 → 104 mm), nên cửa sổ dài bị trễ. Fit dự phòng ở train cũng nhiều hơn (3,6% / 8,4% / 10,2%).
- **Dataset** (người dùng muốn thử trên GRU): `train_data/da_vien/dv_h36_tide15`, `dv_h36_tide30` (config `dataset_config_h36_tide15.json`, `..._tide30.json`). Giống hệt `dv_h36_all` (mẫu và kênh không phải triều sai khác 0), chỉ khác nhóm triều. Bản tới 12/11 trong scratchpad.
- **Lệnh train** (công thức run gốc u0 s23, không augmentation, để so trực tiếp với 10 / 15 / 12,9):

Triều 15 ngày:
```bash
.venv/bin/python -m src.ai.train --mlflow-ex-name da_vien --horizon 1h --dataset-dir train_data/da_vien/dv_h36_tide15 --target-horizons 3 6 --rain-stations wind+tide+fc_ecmwf+sea+station_obs --use-weather-forecast false --feature-set water_level+rain+flood_season --train-method gru --head-mode shared --station-embedding-size 0 --lstm-seq-len 25 --lstm-hidden-size 64 --lstm-num-layers 3 --lstm-drop-out 0.2 --lstm-input-dropout 0.1 --lstm-head-dropout 0.1 --lstm-lr 0.0001 --lstm-weight-decay 0.001 --lr-scheduler step --batch-size 64 --num-epochs 100 --early-stopping 1000 --change-loss combined --change-under-weight 0 --change-event-weight 0 --change-margin-mm 0 --lstm-checkpoint-metric val_loss --skip-test --seed 23 --diagnose-change-dynamics --checkpoint-dir model/experiments/da_vien/dv_h36_tide15/h36_pers_seq25_u0_s23
```

Triều 30 ngày:
```bash
.venv/bin/python -m src.ai.train --mlflow-ex-name da_vien --horizon 1h --dataset-dir train_data/da_vien/dv_h36_tide30 --target-horizons 3 6 --rain-stations wind+tide+fc_ecmwf+sea+station_obs --use-weather-forecast false --feature-set water_level+rain+flood_season --train-method gru --head-mode shared --station-embedding-size 0 --lstm-seq-len 25 --lstm-hidden-size 64 --lstm-num-layers 3 --lstm-drop-out 0.2 --lstm-input-dropout 0.1 --lstm-head-dropout 0.1 --lstm-lr 0.0001 --lstm-weight-decay 0.001 --lr-scheduler step --batch-size 64 --num-epochs 100 --early-stopping 1000 --change-loss combined --change-under-weight 0 --change-event-weight 0 --change-margin-mm 0 --lstm-checkpoint-metric val_loss --skip-test --seed 23 --diagnose-change-dynamics --checkpoint-dir model/experiments/da_vien/dv_h36_tide30/h36_pers_seq25_u0_s23
```


### Kết quả Task 28 (người dùng train; Claude chấm mọi epoch trên mọi giờ)

| Run (u0 s23, không aug) | +6 h: ít nhất / trung vị / TB 10 cuối / ensemble | +3 h: số epoch 0 lỗi; ensemble |
|---|---|---|
| triều 7 ngày (gốc) | **10 / 15 / 12,9 / 12** | 15; 1 |
| triều 15 ngày | 12 / 17 / 17,1 / 16 (val 8 trong ensemble) | **25; 0** |
| triều 30 ngày | 21 / 26 / 24,1 / 25 | 3; 2 |

- 15 ngày: +6 h kém hơn nhẹ, +3 h ổn định hơn (giống tuyến tính: triều 15 ngày tốt hơn ở +3 h). 30 ngày kém rõ. Giữ triều 7 ngày cho +6 h.
- **Vị trí trong chu kỳ nước cường / nước kém (14,77 ngày):** chưa có kênh riêng. Triều 15/30 ngày chỉ chứa gián tiếp (đường triều dự báo có biên độ đổi theo M2 + S2); triều 7 ngày gần như không có.

## Task 29. Nhóm `spring_neap`: pha nước cường / nước kém (người dùng yêu cầu; 18/11/2026)

**Code:** `features.spring_neap_features`, nhóm `--groups spring_neap` (không cần config); test `src/ai/tests/test_spring_neap.py` (2 test: trăng mới và trăng tròn cho `sn_cos` > 0,9, trăng bán nguyệt < −0,9; chu kỳ 14,765 và 13,661 ngày).
- `sn_cos` / `sn_sin` = cos / sin 2D, với D là góc lệch trung bình Mặt Trăng − Mặt Trời (Meeus). Chu kỳ 14,77 ngày; +1 khi nước cường.
- `sn_decl_cos` / `sn_decl_sin` = cos / sin 2s, với s là kinh độ trung bình của Mặt Trăng. Chu kỳ 13,66 ngày (độ xích vĩ, điều khiển độ chênh giữa hai đỉnh trong ngày).
- Kênh thiên văn, biết trước cho mọi thời điểm. Độ trễ tại trạm (tuổi triều) là một phép dời pha, model học được qua tổ hợp sin và cos.

**Tín hiệu ở DV yếu:**
- Pha nước cường / nước kém chỉ giải thích 1,7% phương sai log biên độ triều ngày; độ xích vĩ 2,9%.
- Biên độ triều ngày (trung vị) chủ yếu đổi theo mùa: tháng 6–7 khoảng 111 mm, tháng 9–11 khoảng 229–252 mm, tháng 12 khoảng 298 mm.

**Tuyến tính "D + biển":**
- +3 h: 0 lỗi (không đổi), R² +0,1–0,4 điểm.
- +6 h: 27 → 27 lỗi, R² +0,2–0,5.
- +12 h: 114 → 113 lỗi, R² test −7,2 → −3,3.

Không thay đổi số lỗi.

**Dataset** `train_data/da_vien/dv_h36_sn` = `dv_h36_all` + spring_neap (mẫu và kênh cũ giống hệt; GRU 58 đầu vào). Bản tới 12/11 trong scratchpad.

**Lệnh train** (công thức gốc u0 s23, để so với 10 / 15 / 12,9):
```bash
.venv/bin/python -m src.ai.train --mlflow-ex-name da_vien --horizon 1h --dataset-dir train_data/da_vien/dv_h36_sn --target-horizons 3 6 --rain-stations wind+tide+fc_ecmwf+sea+station_obs+spring_neap --use-weather-forecast false --feature-set water_level+rain+flood_season --train-method gru --head-mode shared --station-embedding-size 0 --lstm-seq-len 25 --lstm-hidden-size 64 --lstm-num-layers 3 --lstm-drop-out 0.2 --lstm-input-dropout 0.1 --lstm-head-dropout 0.1 --lstm-lr 0.0001 --lstm-weight-decay 0.001 --lr-scheduler step --batch-size 64 --num-epochs 100 --early-stopping 1000 --change-loss combined --change-under-weight 0 --change-event-weight 0 --change-margin-mm 0 --lstm-checkpoint-metric val_loss --skip-test --seed 23 --diagnose-change-dynamics --checkpoint-dir model/experiments/da_vien/dv_h36_sn/h36_pers_seq25_u0_s23
```

## Task 30. Lỗi còn lại chỉ nằm trong bão/lũ; tìm dữ liệu cũ (18/11/2026)

- **Giờ bão/lũ** = mưa đo 24 h > 30 mm, hoặc mưa ECMWF 6 h tới > 10 mm, hoặc mực nước > 900 mm; chiếm 21% số giờ ngoài train.
  - `v1` +6 h: 12/12 lỗi > 200 và 70/103 giờ sai > 150 nằm trong các giờ này.
  - `v1` +3 h: 12/12 giờ sai > 150 nằm trong các giờ này.
  - **Ngoài giờ bão/lũ: 0 lỗi > 200 ở cả +3 h và +6 h.**
- **DB:** mực nước DV, NBS, Thuận An bắt đầu từ 14/11/2025. 51 trạm có dữ liệu sớm hơn đều là trạm mưa, bắt đầu từ 31/10/2025.
- **API nguồn (DCU):** người dùng chạy thử `api_backfill` cho DV từ 10/11 đến 24/11/2024: không trả về số đo nào. Nguồn không có lịch sử mùa lũ trước.
- **Hướng tiếp theo:** hỏi người hướng dẫn về số liệu giờ các mùa lũ cũ (DV, Kim Long, Phú Ốc); khi vận hành thì báo độ tin cậy thấp trong giờ bão/lũ.

### Kết quả Task 29 (người dùng train `dv_h36_sn` s23 và s23 ev1; Claude chấm mọi epoch trên mọi giờ, 18/11/2026)

| Run | +6 h: ít nhất (train / val / rg / test / mới) | Trung vị | TB 10 cuối | Ensemble | Val thấp nhất |
|---|---|---|---|---|---|
| gốc `dv_h36_all` s23 | **10** (6 / 4 / 1 / 4 / 1) | **15** | **12,9** | **12** | 3 |
| spring_neap s23 | 11 (5 / 1 / 0 / 3 / 7) | 16 | 14,6 | 15 | **1** |
| spring_neap s23 ev1 | 12 (4 / 0 / 0 / 6 / 6) | 19 | 16,9 | 16 | **0 (10 epoch)** |

- **Người dùng đúng ở val:** spring_neap đưa val +6 h từ 3 xuống 1; với ev1 có 10 epoch val = 0.
- Nhưng ở chính 10 epoch val = 0, ranh giới + test + mới có 12–18 lỗi. **Lỗi bị dời đi chứ không mất:** bão 11/09 được sửa, nhưng xuất hiện lỗi mới 07/10 (báo thừa 3 giờ) và 11/11 (báo thừa 4 giờ khi lũ rút).
- +3 h kém hơn nhẹ (số epoch 0 lỗi 39–45 so với 82 ở val; ensemble 2 lỗi so với 1).
- Chấm thêm: tide15 h32 (+6 h: ít nhất 22) và h3612 aug ev2 trọng số 1 / 2,5 / 1 (+6 h: 12 / 33 / 14,1; +12 h ít nhất 57). Không run nào tốt hơn.

- **Thêm 2 run spring_neap** (người dùng train; chấm mọi giờ):
  - `dv_h6_sn/h6_pers_seq25_u0_s23` (chỉ +6 h): ít nhất 14 (ep96: 2 / 2 / 0 / 7 / 5), trung vị 19, TB 10 cuối 17,8, ensemble 19. Lỗi mới 07/10 (+, 4 giờ) và 11/11 (+, 3 giờ), giống run ev1.
  - `dv_h3612_sn/h3612_pers_seq25_u0_s23_ev1`:
    - +3 h: trung vị 0, TB 10 cuối 0, ensemble 0 lỗi ngoài train (train 2);
    - +6 h: ít nhất 16, trung vị 22, val ít nhất 11 (13/09 báo thừa 7 giờ);
    - +12 h: ít nhất 49, trung vị 61.

### Task 31. Ensemble chéo giữa các run (+6 h; 18/11/2026)

Lấy trung bình 10 epoch cuối của từng run, rồi ghép 1–3 run trong 28 run có mốc +6 h (3.682 tổ hợp), chấm mọi giờ.
- **Tổ hợp tốt nhất: 8 lỗi ngoài train** (5 tổ hợp, ví dụ h3612 aug ev1 h128 + spring_neap s23 + tide30: 2 / 6 / 0 / 2 / 0; lỗi lớn nhất 237 mm).
- Phần lớn tổ hợp 3 run có 10–15 lỗi.
- Đây là kết quả chọn sau khi xem mọi tập (lạc quan nhất có thể), mà **vẫn không có tổ hợp nào đạt 0**. Các lỗi chung (11/09, 21/09, 13/10) bị mọi model sai cùng chiều.

### Task 32. Dự báo cả quỹ đạo +1…+6 h (người dùng train `dv_h123456_sn/h123546_pers_seq25_u0_s23` và `dv_h369_sn`; chấm mọi giờ, 18/11/2026)

| Run (spring_neap, s23) | Mốc | Ít nhất (train / val / rg / test / mới) | Trung vị | TB 10 cuối | Ensemble | Số epoch 0 lỗi |
|---|---|---|---|---|---|---|
| +1…+6 h | +1 | 0 | 0 | 0 | 0 | 100 |
| | +2 | 0 | 0 | 0 | 0 | 93 |
| | +3 | 0 | 1 | 1,0 | 1 | 35 |
| | +4 | 0 | 2 | 1,9 | 2 | 4 |
| | +5 | 2 | 7 | 6,0 | 6 | 0 |
| | **+6** | **8** (ep81: 4 / 1 / 0 / 4 / 3) | 17 | **11,0** | **10** (3 / 2 / 0 / 4 / 4) | 0 |
| +3 / +6 / +9 h | +6 | 11 (7 / 6 / 0 / 0 / 5) | 17 | 13,7 | 12 | 0 |
| | +9 | 28 | 36 | 37,5 | 38 | 0 |
| gốc `dv_h36_all` (+3 / +6) | +6 | 10 (6 / 4 / 1 / 4 / 1) | 15 | 12,9 | 12 | 0 |

- **+1…+6 h là run có +6 h tốt nhất đến nay theo ít nhất (8), TB 10 cuối (11,0) và ensemble (10).** Trung vị kém hơn (17 so với 15). Mức cải thiện nằm trong dao động do seed; chưa tách được phần do spring_neap và phần do dự báo cả quỹ đạo.
- Lỗi +6 h tại ep81: 11/09 (+), 21/09 (−, 2), 13/10 (−, 2), 10/11 (−), 11/11 (+, 2).
- Lợi ích phụ: có dự báo cho mọi giờ +1…+6, đúng nhu cầu một đường dự báo liên tục.

**Lệnh kiểm tra tiếp** (mỗi run một lệnh):

1. +1…+6 h không spring_neap (tách hiệu ứng spring_neap và hiệu ứng quỹ đạo):
```bash
.venv/bin/python -m src.ai.train --mlflow-ex-name da_vien --horizon 1h --dataset-dir train_data/da_vien/dv_h36_all --target-horizons 1 2 3 4 5 6 --rain-stations wind+tide+fc_ecmwf+sea+station_obs --use-weather-forecast false --feature-set water_level+rain+flood_season --train-method gru --head-mode shared --station-embedding-size 0 --lstm-seq-len 25 --lstm-hidden-size 64 --lstm-num-layers 3 --lstm-drop-out 0.2 --lstm-input-dropout 0.1 --lstm-head-dropout 0.1 --lstm-lr 0.0001 --lstm-weight-decay 0.001 --lr-scheduler step --batch-size 64 --num-epochs 100 --early-stopping 1000 --change-loss combined --change-under-weight 0 --change-event-weight 0 --change-margin-mm 0 --lstm-checkpoint-metric val_loss --skip-test --seed 23 --diagnose-change-dynamics --checkpoint-dir model/experiments/da_vien/dv_h123456_all/h123456_pers_seq25_u0_s23
```
2. +1…+6 h spring_neap, seed 42 (kiểm tra seed):
```bash
.venv/bin/python -m src.ai.train --mlflow-ex-name da_vien --horizon 1h --dataset-dir train_data/da_vien/dv_h36_sn --target-horizons 1 2 3 4 5 6 --rain-stations wind+tide+fc_ecmwf+sea+station_obs+spring_neap --use-weather-forecast false --feature-set water_level+rain+flood_season --train-method gru --head-mode shared --station-embedding-size 0 --lstm-seq-len 25 --lstm-hidden-size 64 --lstm-num-layers 3 --lstm-drop-out 0.2 --lstm-input-dropout 0.1 --lstm-head-dropout 0.1 --lstm-lr 0.0001 --lstm-weight-decay 0.001 --lr-scheduler step --batch-size 64 --num-epochs 100 --early-stopping 1000 --change-loss combined --change-under-weight 0 --change-event-weight 0 --change-margin-mm 0 --lstm-checkpoint-metric val_loss --skip-test --seed 42 --diagnose-change-dynamics --checkpoint-dir model/experiments/da_vien/dv_h123456_sn/h123456_pers_seq25_u0_s42
```
3. +1…+6 h spring_neap, seed 23, thêm augmentation:
```bash
.venv/bin/python -m src.ai.train --mlflow-ex-name da_vien --horizon 1h --dataset-dir train_data/da_vien/dv_h36_sn --target-horizons 1 2 3 4 5 6 --rain-stations wind+tide+fc_ecmwf+sea+station_obs+spring_neap --use-weather-forecast false --feature-set water_level+rain+flood_season --train-method gru --head-mode shared --station-embedding-size 0 --lstm-seq-len 25 --lstm-hidden-size 64 --lstm-num-layers 3 --lstm-drop-out 0.2 --lstm-input-dropout 0.1 --lstm-head-dropout 0.1 --lstm-lr 0.0001 --lstm-weight-decay 0.001 --lr-scheduler step --batch-size 64 --num-epochs 100 --early-stopping 1000 --change-loss combined --change-under-weight 0 --change-event-weight 0 --change-margin-mm 0 --lstm-checkpoint-metric val_loss --skip-test --seed 23 --diagnose-change-dynamics --augment-surge-scale 0.8:1.3:0.5 --checkpoint-dir model/experiments/da_vien/dv_h123456_sn/h123456_pers_seq25_u0_s23_aug0.8-1.3p0.5
```

### Task 33. Dùng lại dự báo của chính model làm đầu vào cho lần dự báo sau (người dùng hỏi; 18/11/2026)

**Cách thử** (run `dv_h123456_sn` s23, trung bình 10 epoch cuối; hiệu chỉnh tuyến tính cho +6 h):
- Đầu vào tại t: 6 dự báo phát hành lúc t; các dự báo phát hành lúc t−1, t−2, t−3 cho cùng giờ mục tiêu (dưới dạng mức điều chỉnh so với dự báo mới); sai số đã biết của dự báo +k h phát hành lúc t−k (hiệu lực đúng lúc t).

**Lỗi > 200 mm** (train / val / rg / test / mới; ngoài train):

| Cách | Lỗi | Ngoài train | MAE ngoài train |
|---|---|---|---|
| GRU +1…+6 | 3 / 2 / 0 / 4 / 4 | **10** | 47,6 |
| fit train, thêm dự báo t−1 | 2 / 4 / 0 / 6 / 4 | 14 | 49,3 |
| fit train, thêm dự báo t−1…t−3 và sai số đã biết | 0 / 2 / 0 / 4 / 5 | 11 | 48,7 |
| fit val (các biến thể) | — | 38–45 | 54,6–56,0 |

- Tương quan ngoài train giữa sai số +6 h sắp tới với sai số +1 h đã biết là −0,05; với sai số +3 h là 0,08; với mức điều chỉnh dự báo giữa hai lần phát hành là 0,01.
- **Không giúp.** Dự báo lúc t−1 là hàm của dữ liệu lúc t−1, mà dữ liệu đó GRU đã thấy trong cửa sổ 25 h lúc t. Thông tin mới duy nhất (số đo lúc t) cũng đã là đầu vào. Trạng thái ẩn của GRU vốn đã mang "niềm tin" từ các bước trước.

## Task 34. Dataset có dự báo của chính model làm đầu vào (người dùng yêu cầu dù Task 33 âm tính; 18/11/2026)

- **Dự báo nguồn:** `dv_h123456_sn/h123546_pers_seq25_u0_s23`, trung bình epoch 91–100, cho mọi giờ phát hành 15/11/2025–11/11/2026. Xuất bằng `research/scripts/export_self_forecasts.py` vào `data/derived/da_vien/self_fc_h123456_sn_s23_ep91-100.parquet`; khớp bản đã chấm (≤ 0,0004 mm). Đã thêm `research/__init__.py` để import được `research.scripts.member_ensemble`.
- **Nhóm `self_fc`** (`features.self_forecast_features`; khóa config `self_fc.forecasts_parquet`, `issue_lag_hours` = 1; test `src/ai/tests/test_self_forecast.py`):
  - `selffc_t<k>` (k = 0…5) tại giờ τ = dự báo phát hành lúc τ − 1 cho giờ τ + k, trừ mực nước đo lúc τ. `selffc_t0` là sai số đã biết của dự báo +1 h, đổi dấu (đã kiểm tra khớp ≤ 0,0004 mm).
  - Khoảng 10% giờ thiếu (đầu chuỗi, khoảng trống).
  - Augmentation từ chối nhóm này (`selffc_*` phụ thuộc mực nước).
- **Dataset** `train_data/da_vien/dv_h36_sn_selffc` (config `dataset_config_h36_sn_selffc.json`) = `dv_h36_sn` + `self_fc`. Mẫu và kênh cũ giống hệt; GRU đọc 65 đầu vào. Bản tới 12/11 trong scratchpad.
- **Rủi ro đã báo:** trên giờ train, dự báo nguồn là trong mẫu (tốt hơn thực tế), nên GRU dễ tin quá mức (giống Task 10, 12c, 22). Task 33: thông tin này gần như không tương quan với sai số +6 h sắp tới.
- **Lệnh train** (so với `dv_h123456_sn` s23: +6 h ít nhất 8, trung vị 17, TB 10 cuối 11,0, ensemble 10):

1. Không dropout nhóm:
```bash
.venv/bin/python -m src.ai.train --mlflow-ex-name da_vien --horizon 1h --dataset-dir train_data/da_vien/dv_h36_sn_selffc --target-horizons 1 2 3 4 5 6 --rain-stations wind+tide+fc_ecmwf+sea+station_obs+spring_neap+self_fc --use-weather-forecast false --feature-set water_level+rain+flood_season --train-method gru --head-mode shared --station-embedding-size 0 --lstm-seq-len 25 --lstm-hidden-size 64 --lstm-num-layers 3 --lstm-drop-out 0.2 --lstm-input-dropout 0.1 --lstm-head-dropout 0.1 --lstm-lr 0.0001 --lstm-weight-decay 0.001 --lr-scheduler step --batch-size 64 --num-epochs 100 --early-stopping 1000 --change-loss combined --change-under-weight 0 --change-event-weight 0 --change-margin-mm 0 --lstm-checkpoint-metric val_loss --skip-test --seed 23 --diagnose-change-dynamics --checkpoint-dir model/experiments/da_vien/dv_h123456_sn_selffc/h123456_pers_seq25_u0_s23
```
2. Dropout nhóm 0,3 (giảm phụ thuộc vào dự báo trong mẫu):
```bash
.venv/bin/python -m src.ai.train --mlflow-ex-name da_vien --horizon 1h --dataset-dir train_data/da_vien/dv_h36_sn_selffc --target-horizons 1 2 3 4 5 6 --rain-stations wind+tide+fc_ecmwf+sea+station_obs+spring_neap+self_fc --use-weather-forecast false --feature-set water_level+rain+flood_season --train-method gru --head-mode shared --station-embedding-size 0 --lstm-seq-len 25 --lstm-hidden-size 64 --lstm-num-layers 3 --lstm-drop-out 0.2 --lstm-input-dropout 0.1 --lstm-head-dropout 0.1 --lstm-lr 0.0001 --lstm-weight-decay 0.001 --lr-scheduler step --batch-size 64 --num-epochs 100 --early-stopping 1000 --change-loss combined --change-under-weight 0 --change-event-weight 0 --change-margin-mm 0 --lstm-checkpoint-metric val_loss --skip-test --seed 23 --diagnose-change-dynamics --group-dropout selffc_:0.3 --checkpoint-dir model/experiments/da_vien/dv_h123456_sn_selffc/h123456_pers_seq25_u0_s23_gd0.3
```

### Kết quả Task 32 và Task 34 (người dùng train 9 run; Claude chấm mọi epoch trên mọi giờ, 18/11/2026)

Mọi run dự báo +1…+6 h, cửa sổ 25 h, u0, step, 100 epoch. Lỗi +6 h ngoài train (val + rg + test + mới).

| Run | Ít nhất (epoch; train / val / rg / test / mới) | Trung vị | TB 10 cuối | Ensemble 10 cuối | Ensemble: lỗi lớn nhất (mm) | +3 h: số epoch 0 lỗi |
|---|---|---|---|---|---|---|
| gốc `dv_h36_all` +3 / +6 | 10 (ep61) | 15 | 12,9 | 12 | 256 | 15 |
| spring_neap s23 (Task 32) | 8 (ep81) | 17 | 11,0 | 10 | 217 | 35 |
| không spring_neap s23 | 14 | 18 | 17,5 | 16 | 250 | 25 |
| spring_neap s42 | 13 | 18 | 17,3 | 17 | 253 | 4 |
| **spring_neap s23 + aug 0,8–1,3 p 0,5** | **7** (ep78; 3 / 1 / 0 / 3 / 3) | 15,5 | **9,9** | **8** (2 / 2 / 0 / 3 / 3) | **216** | 47 |
| spring_neap s23 h128 | 14 | 33,5 | 36,4 | 36 | 289 | 11 |
| spring_neap s23 h32×3 / ×4 / ×6 | 22 / 25 / 19 | 25 / 33 / 32 | 23,5 / 35,3 / 34,4 | 23 / 37 / 34 | 267–353 | 0 |
| + self_fc s23 | 15 | 21 | 23,6 | 24 | 273 | **90** |
| + self_fc dropout nhóm 0,3 | 8 (ep39) | **13** | 15,9 | 15 | 246 | 67 |

- **Tốt nhất đến nay:** +1…+6 h, spring_neap, aug, s23. Ensemble 10 epoch cuối có 8 lỗi +6 h, **đều chỉ vượt ngưỡng 2–16 mm** (202–216 mm): 11/09 13–14 h (+), 21/09 00–01 h (−), 13/10 06 h (−), 10/11 01 h (−), 11/11 17–18 h (+). Với ngưỡng 220 mm thì 0 lỗi.
- Ensemble đó ở các mốc khác: +1 h 0 (lớn nhất 101), +2 h 0, +3 h 1 (209 mm, mới), +4 h 2, +5 h 5.
- spring_neap có vẻ giúp trong cấu hình +1…+6 h (s23: 8 / 17 / 11,0 so với 14 / 18 / 17,5 khi không có), nhưng s42 kém (13 / 18 / 17,3). Phụ thuộc seed vẫn lớn.
- Kích thước khác 64×3 đều kém hơn.
- self_fc: +3 h rất ổn định (90 epoch đạt 0 lỗi), +6 h không tốt hơn. Dropout nhóm giúp trung vị (13) nhưng ensemble 15.
- Ghép chéo các run +1…+6 h: không tổ hợp nào tốt hơn riêng run aug (8).
- Thư mục `h123546_pers_seq25_u0_s23_h32x3` có thêm một lần chạy FAILED ngay sau lần FINISHED, không ghi đè (đủ 100 epoch).

- **self_fc theo từng mốc** (TB 10 epoch cuối; MAE train / ngoài train; so với cùng cấu hình không self_fc):
  - +6 h: MAE 28 / 53 → 25 / 57 (gd 0,3: 27 / 56). Train tốt lên, ngoài train kém đi: dấu hiệu học theo dự báo trong mẫu, đúng rủi ro đã báo.
  - +3 h: MAE ngoài train 37 → 40, nhưng số epoch đạt 0 lỗi tăng từ 35 lên 90 (gd 0,3: 67).
  - **Kết luận:** self_fc không giúp +6 h; có thể giữ cho +3 h nếu cần ổn định khi chọn epoch.

**Lệnh tiếp theo** (cùng cấu hình tốt nhất, kiểm tra seed và mức augmentation; mỗi run một lệnh):

seed 42, aug 0,8–1,3 p 0,5:
```bash
.venv/bin/python -m src.ai.train --mlflow-ex-name da_vien --horizon 1h --dataset-dir train_data/da_vien/dv_h36_sn --target-horizons 1 2 3 4 5 6 --rain-stations wind+tide+fc_ecmwf+sea+station_obs+spring_neap --use-weather-forecast false --feature-set water_level+rain+flood_season --train-method gru --head-mode shared --station-embedding-size 0 --lstm-seq-len 25 --lstm-hidden-size 64 --lstm-num-layers 3 --lstm-drop-out 0.2 --lstm-input-dropout 0.1 --lstm-head-dropout 0.1 --lstm-lr 0.0001 --lstm-weight-decay 0.001 --lr-scheduler step --batch-size 64 --num-epochs 100 --early-stopping 1000 --change-loss combined --change-under-weight 0 --change-event-weight 0 --change-margin-mm 0 --lstm-checkpoint-metric val_loss --skip-test --seed 42 --diagnose-change-dynamics --augment-surge-scale 0.8:1.3:0.5 --checkpoint-dir model/experiments/da_vien/dv_h123456_sn/h123456_pers_seq25_u0_s42_aug0.8-1.3p0.5
```

seed 17, aug 0,8–1,3 p 0,5:
```bash
.venv/bin/python -m src.ai.train --mlflow-ex-name da_vien --horizon 1h --dataset-dir train_data/da_vien/dv_h36_sn --target-horizons 1 2 3 4 5 6 --rain-stations wind+tide+fc_ecmwf+sea+station_obs+spring_neap --use-weather-forecast false --feature-set water_level+rain+flood_season --train-method gru --head-mode shared --station-embedding-size 0 --lstm-seq-len 25 --lstm-hidden-size 64 --lstm-num-layers 3 --lstm-drop-out 0.2 --lstm-input-dropout 0.1 --lstm-head-dropout 0.1 --lstm-lr 0.0001 --lstm-weight-decay 0.001 --lr-scheduler step --batch-size 64 --num-epochs 100 --early-stopping 1000 --change-loss combined --change-under-weight 0 --change-event-weight 0 --change-margin-mm 0 --lstm-checkpoint-metric val_loss --skip-test --seed 17 --diagnose-change-dynamics --augment-surge-scale 0.8:1.3:0.5 --checkpoint-dir model/experiments/da_vien/dv_h123456_sn/h123456_pers_seq25_u0_s17_aug0.8-1.3p0.5
```

seed 23, aug 0,8–1,3 p 1,0:
```bash
.venv/bin/python -m src.ai.train --mlflow-ex-name da_vien --horizon 1h --dataset-dir train_data/da_vien/dv_h36_sn --target-horizons 1 2 3 4 5 6 --rain-stations wind+tide+fc_ecmwf+sea+station_obs+spring_neap --use-weather-forecast false --feature-set water_level+rain+flood_season --train-method gru --head-mode shared --station-embedding-size 0 --lstm-seq-len 25 --lstm-hidden-size 64 --lstm-num-layers 3 --lstm-drop-out 0.2 --lstm-input-dropout 0.1 --lstm-head-dropout 0.1 --lstm-lr 0.0001 --lstm-weight-decay 0.001 --lr-scheduler step --batch-size 64 --num-epochs 100 --early-stopping 1000 --change-loss combined --change-under-weight 0 --change-event-weight 0 --change-margin-mm 0 --lstm-checkpoint-metric val_loss --skip-test --seed 23 --diagnose-change-dynamics --augment-surge-scale 0.8:1.3:1.0 --checkpoint-dir model/experiments/da_vien/dv_h123456_sn/h123456_pers_seq25_u0_s23_aug0.8-1.3p1.0
```

## Task 35. Gợi ý của ChatGPT về dự báo cuốn chiếu (người dùng chuyển lại; 18/11/2026)

Kiểm tra trên run tốt nhất (+1…+6 h, spring_neap, aug, s23, trung bình 10 epoch cuối); lỗi > 200 ngoài train.

- **B. Time-lagged ensemble** (với mục tiêu t + h, kết hợp các dự báo phát hành lúc t, t − 1, …, t − (6 − h)):

| Mốc | Số dự báo chồng lấn | Mới nhất | Trung bình đều | Trọng số học trên train |
|---|---|---|---|---|
| +1 h | 6 | 0 (lớn nhất 101) | 0 (170) | 0 (84) |
| +2 h | 5 | 0 (154) | 0 (189) | 0 (145) |
| +3 h | 4 | 1 (209) | 1 (201) | **0 (190)** |
| +4 h | 3 | 2 (230) | 2 (208) | 1 (228) |
| +5 h | 2 | 5 (225) | 5 (213) | 5 (223) |
| +6 h | 1 | 8 (216) | — | 9 (hiệu chỉnh tuyến tính) |

  Giúp nhẹ ở +3/+4 h. Ở +6 h chỉ có đúng một dự báo (model dài nhất +6 h), nên không có gì để kết hợp.
- **A/C. Revision và độ bất đồng làm chỉ báo lỗi +6 h > 200** (AUC ngoài train): |revision| 0,52; độ bất đồng giữa các dự báo chồng lấn 0,63; |sai số đã biết +1 h| 0,54; |sai số +6 h đã biết| 0,53; |thay đổi dự báo| 0,50. So với mưa đo 24 h: **0,92**.
- **D. Online error correction:** tương quan sai số +6 h sắp tới với sai số +1 h vừa biết là −0,05 (giống Task 18, 22, 33). Không giúp.
- **"Nhận ra muộn" hay "thiếu biên độ"** (gợi ý mục 4 của ChatGPT): biểu đồ quỹ đạo (`figures/da_vien/h123456_sn_s23_trajectory_every6h_00.png`) cho thấy các lỗi 21/09, 13/10, 09–10/11 chủ yếu là nhận ra muộn. Lần phát hành trước đợt lên vẫn dự báo đi ngang hoặc xuống; lần sau thì bám kịp.

### Task 36. Kiểm tra seed cho cấu hình +1…+6 h + spring_neap + aug (người dùng train 3 run; chấm mọi giờ, 18/11/2026)

| Run | +6 h: ít nhất (train / val / rg / test / mới) | Trung vị | TB 10 cuối | Ensemble | Lớn nhất (ensemble) | +3 h: số epoch 0 lỗi |
|---|---|---|---|---|---|---|
| s23 aug p 0,5 (Task 32) | 7 (3 / 1 / 0 / 3 / 3) | 15,5 | 9,9 | **8** | 216 | 47 |
| **s23 aug p 1,0** | **6** (ep67; 6 / 2 / 0 / 1 / 3) | 15,5 | **9,3** | **8** | 217 | **66** |
| s42 aug p 0,5 | 16 | 20,5 | 19,6 | 18 | 256 | 8 |
| s17 aug p 0,5 | 14 | 25 | 24,5 | 24 | 318 | 0 |

- **Kết quả tốt là của riêng seed 23**, không giữ được ở seed 42 và 17 (giống mọi cấu hình trước đây).
- **Ghép seed không giúp** (+6 h, ensemble 10 epoch cuối mỗi run):
  - s23 + s42: 13;
  - s23 + s42 + s17: 12;
  - s23 p 0,5 + s23 p 1,0: 8 (lớn nhất 217).
- 8 lỗi của ghép hai run s23 vẫn ở các đợt cũ, đều 200–217 mm: 11/09 13 h (+210), 21/09 00–01 h (−210, −217), 13/10 06 h (−212), 28/10 20 h (−200), 10/11 01 h (−206), 11/11 17–18 h (+214, +210).

## Kế hoạch của người dùng về dự báo NBS (17/11/2026)

- **Không train lại model NBS** để tạo dự báo out-of-fold.
- Khi train DV: dùng dự báo NBS (`combined_v1`) làm đầu vào, kèm **dropout cho nhóm này** để model không quá phụ thuộc. Code đã có sẵn: `--group-dropout PREFIX:RATE`.
- Sau khi có model DV: dùng đầu ra DV để train lại một model NBS tốt hơn.

**Rủi ro đã nêu với người dùng:**
- Dropout làm giảm mức phụ thuộc, nhưng không sửa được việc dự báo NBS trên train là trong mẫu. Ở NBS, kênh `lin_fc` kèm group dropout 0,3 và 0,5 đều cho kết quả kém hơn.
- Vòng DV → NBS cũng gặp vấn đề tương tự (dự báo DV trên train là trong mẫu). Mô phỏng ở `NOTES.md` A4 cho thấy chỉ vòng đầu có thay đổi.

## Kiến trúc GRU hiện tại (đã kiểm chứng ngày 17/11/2026)

`src/ai/residual_sequence.py`, lớp `ResidualSequence(architecture='gru')`, dùng với `--train-method gru --head-mode station`. Đã nạp hai checkpoint của `combined_v1` và đếm tham số từng phần: khớp `PIPELINE.md` mục 4 và `REPRODUCE.md` (74.955 và 279.435).

Luồng xử lý (x: [B, L, F]):
1. (chỉ khi train, nếu bật) `--group-dropout` rồi `--history-dropout`;
2. `Dropout1d(input_dropout)`: mỗi mẫu bỏ nguyên một kênh trên cả cửa sổ, kênh còn lại nhân 1/(1−p). Có thể bỏ cả kênh mực nước, nhưng nhánh persistence không bị ảnh hưởng;
3. `GRU(F → H, N lớp, dropout giữa lớp)`, lấy trạng thái ẩn ở bước cuối;
4. ghép với `Embedding(1, 8)` của trạm, rồi `Linear(H + 8 → H)` + ReLU;
5. `Dropout(head_dropout)`;
6. `StationCorrectionHead`: W [1, 3, H] và b [1, 3], khởi tạo 0, ra Δ cho từng mốc;
7. ŷ_h = a_h · x[t, mực nước] + b_h + Δ_h, với a_h = σ_w/σ_h. Ví dụ NBS: σ_w 144,2, σ_h 107,8 / 106,5 / 104,2, nên a_h = 1,337 / 1,354 / 1,384, khớp checkpoint.

**Đã kiểm tra:**
- model mới khởi tạo cho đầu ra đúng bằng persistence;
- `Dropout1d` bỏ nguyên kênh;
- mọi nhóm đầu vào phụ (mưa ECMWF, gió, triều, mực nước biển) được đưa vào GRU ở **mọi bước** của cửa sổ (`weather_feature_indices` rỗng, không có late fusion).

**Station embedding trong model một trạm** (người dùng hỏi, 17/11/2026):
- Chỉ có 1 trạm, nên `Embedding(1, 8)` là **một vector hằng**, giống nhau cho mọi mẫu. Phần nó đóng góp vào `Linear(H + 8 → H)` (W_e·e) chỉ là một **bias cố định**.
- Đã kiểm chứng trên hai checkpoint `combined_v1`: gộp W_e·e vào bias của tầng `fusion` rồi đặt embedding = 0 thì đầu ra không đổi (sai khác 2·10⁻⁷).
- Phần dư thừa: 520 tham số (H = 64) và 1.032 tham số (H = 128).
- Cùng lý do, `StationCorrectionHead` với 1 trạm chính là một `Linear(H → 3)` khởi tạo 0.
- **Ảnh hưởng tới kết quả:** không đổi khả năng biểu diễn của model. Nó chỉ làm khác cách tối ưu một chút (bias được biểu diễn bằng nhiều tham số hơn) và làm khác dãy số ngẫu nhiên lúc khởi tạo.
- **Người dùng quyết định (17/11/2026): giữ nguyên ở các trạm đã train (Thuận An, NBS). Riêng Dã Viên thì BỎ.**
  - Đã thêm tùy chọn `--station-embedding-size 0`, bắt buộc đi cùng `--head-mode shared`. Khi đó model không có embedding, tầng ghép là `Linear(H → H)`, head là `Linear(H → 3)` khởi tạo 0. Mã trạm vẫn được dùng để lấy hệ số persistence.
  - Kiểm tra:
    - test mới `src/ai/tests/test_no_station_embedding.py` (4 test) đạt;
    - các test liên quan (kiến trúc, group/history dropout, transformer) vẫn chỉ có 2 lỗi cũ do `validate_args`, giống hệt trước khi sửa;
    - dự báo của `combined_v1` không đổi (sai khác 0,0 trên 24.011 dòng);
    - dựng thử model DV H64×3 trên `dv_water_only`: 67.523 tham số, a_h = 1,002 / 1,004 / 1,006.
  - Cờ khi train DV: `--head-mode shared --station-embedding-size 0`, thay cho `--head-mode station`.
- Nhận định ban đầu của Claude (giữ nguyên ở mọi trạm): bỏ đi không có lợi hiển nhiên. Ngược lại, nó làm thay đổi khởi tạo với cùng seed: kết quả NBS phụ thuộc rất nhiều vào seed và chỉ seed 23 đạt 0 lỗi, nên sẽ không còn tái tạo được. Ngoài ra còn phải sửa `train.py` và `SequenceRuntime`, vì `--head-mode station` yêu cầu có embedding.

## Model NBS không có station embedding, để finetune sang DV (17/11/2026)

- **Mục đích:** người dùng muốn finetune một model NBS trên dữ liệu DV. Vì model DV bỏ embedding, cần train lại một model NBS cùng kiến trúc (`--head-mode shared --station-embedding-size 0`).
- **Công thức:** model đơn tốt nhất của NBS là `sea3 h128x3 seq25 hw 1 1 2 ev0 step s23`, epoch 98. Trên NBS: val 0/0/0, test 0/0/0, giai đoạn mới 0/0/1. Thông số lấy từ MLflow `mlflow_local.db`: 100 epoch, early stopping 100, dataset `nbs_onshore_tide7_fc_sea_v3` (35 đầu vào).
- **Lưu ý:**
  - Bỏ embedding làm khởi tạo khác đi dù cùng seed 23, nên không chắc tái tạo được 0 lỗi (NBS rất nhạy với seed).
  - `model/experiments/` hiện **không còn** (hai run gốc của `combined_v1` đã mất). Chỉ còn bản đóng gói `models/nga_ba_sinh/combined_v1/`.
- **Để finetune được sau này:**
  - dataset DV phải có **đúng 35 kênh, cùng thứ tự** với dataset NBS (cùng nhóm `fc_ecmwf wind tide sea`, cùng config nhóm);
  - `train.py` hiện chưa có tùy chọn nạp trọng số cho GRU residual, nên phải thêm. Khi nạp cần thay hệ số persistence (a_h, b_h) và bộ chuẩn hóa bằng của DV.

### Kết quả hai run NBS không embedding (người dùng train, 120 epoch, seed 23 và 42; chấm ngày 17/11/2026)

**Cách chấm:** chấm mọi epoch trên GPU (`research/scripts/score_all_epochs.py`) trên val, test và giai đoạn mới 15/10–12/11. Giai đoạn mới dùng dataset mở rộng dựng từ `eval_config_ext_20261112_sea.json` với `--normalizers-from`; dataset này khớp tuyệt đối với dataset train trên đoạn chung. Số lỗi > 200 mm ghi theo +3/+6/+12 h.

| Run | Epoch tốt nhất | val | test | mới | Sai số lớn nhất val / test / mới (mm) | Trung vị epoch ≥ 30 (val · test · mới) | Epoch trainer chọn |
|---|---|---|---|---|---|---|---|
| s23 | 58 / 49 | 0/3/8 · 0/3/9 | 0/0/0 | 0/0/1 · 0/0/0 | 236–246 / 176–180 / 196–212 | 0/6/9 · 0/0/0 · 0/1/1 | ep11 (val 0/15/8): kém |
| **s42** | **63** | **0/0/0** | **0/0/0** | **0/1/0** | **199 / 177 / 206** | 0/0/1 · 0/0/0 · 0/1/2 | ep8 (val 0/11/3): kém |
| s42 | 69 | 0/0/0 | 0/0/0 | 0/1/0 | 198 / 175 / 214 | | |
| s42 | 90 | 0/0/0 | 0/0/0 | 0/1/1 | 195 / 166 / 206 | | |

- s42 có **10 epoch đạt 0 lỗi trên cả val và test**: 63, 65, 68, 69, 72, 73, 75, 78, 87, 90. Không epoch nào đạt 0 lỗi trên cả ba giai đoạn.
- Lỗi còn lại ở giai đoạn mới đều thuộc đợt 09/11: +6 h lúc 16:00 (−206 mm) và +12 h lúc 09:00 (khoảng −196…−206).
- **So với model cũ có embedding** (s23 ep98: val 0/0/0, test 0/0/0, mới 0/0/1): **chất lượng tương đương**. Seed tốt đổi từ 23 sang 42, đúng như dự đoán: bỏ embedding chỉ làm đổi "seed may mắn", không làm model tốt hay kém hơn.
- ~~Đề xuất làm model gốc để finetune sang DV: s42 epoch 63~~. Đã thay bằng kết quả bên dưới.

**Ba run thêm** (người dùng train, cùng công thức, chỉ đổi seed hoặc trọng số mốc; chấm ngày 17/11/2026):

| Run | Epoch tốt nhất | val | test | mới | Số epoch 0 lỗi val+test | Số epoch 0 lỗi cả 3 giai đoạn | TB tổng lỗi, epoch ≥ 30 |
|---|---|---|---|---|---|---|---|
| hw 1/1/2, s23 | 58 | 0/3/8 | 0/0/0 | 0/0/1 | 0 | 0 | 17,1 |
| hw 1/1/2, s42 | 63 / 69 | 0/0/0 | 0/0/0 | 0/1/0 | 10 | 0 | 5,2 |
| hw 1/1/2, s17 | 51 | 0/2/4 | 0/0/0 | 0/2/2 | 0 | 0 | 15,3 |
| hw 1/1/2,5, s42 | 63 | 0/1/0 | 0/0/0 | 0/0/0 | 0 | 0 | 4,4 |
| **hw 1/1,5/2,5, s42** | **116** | **0/0/0** | **0/0/0** | **0/0/0** | **17** | **6** | **4,4** |

- **Lần đầu tiên một model đơn đạt 0 lỗi trên cả val, test và giai đoạn mới.** Run hw 1/1,5/2,5 s42 có 6 epoch như vậy: 75, 90, 97, 114, 115, 116.
- Sai số lớn nhất trên cả ba giai đoạn: ep75 199,5; ep90 200,0; ep97 198,8; ep114 199,0; ep115 199,7; ep116 198,8 mm. **Biên độ rất mỏng.**
- **Đề xuất: epoch 116.** Đây là một trong 3 epoch liên tiếp (114–116) cùng đạt 0, tức vùng ổn định. Sai số lớn nhất 198,8 mm.
  - Sai số lớn nhất từng mốc (val / test / mới): +3 h 141 / 128 / 156; +6 h 191 / 170 / 193; +12 h 199 / 171 / 197.
  - MAE +3/+6/+12 h: val 30,2 / 50,3 / 65,3; test 35,1 / 49,6 / 50,6; mới 29,3 / 44,4 / 56,3.
- Seed 42 tốt ổn định với cả ba bộ trọng số. Seed 17 và 23 kém.
- **Lưu ý:**
  - Epoch được chọn sau khi xem cả ba giai đoạn, trên khoảng 600 epoch của 5 run, nên kết quả lạc quan.
  - Phép thử thật là dữ liệu sau 12/11/2026.
  - Thư mục run: `model/experiments/nga_ba_sinh/nbs_onshore_tide7_fc_sea_v3/noemb_hw11.52.5_…_s42`.

**So sánh ep116 với `combined_v1`** (17/11/2026; đã loại các mẫu sensor đổi mốc; +3 / +6 / +12 h):

| Giai đoạn | Model | Lỗi > 200 | MAE | Sai số lớn nhất | Lỗi > 150 | Độ lệch TB |
|---|---|---|---|---|---|---|
| 01/09–31/10 (1.451 giờ) | combined_v1 | 0/0/0 | 38,0 / 55,7 / 52,0 | 153 / 187 / 198 | 1/44/31 | +14 / +24 / +8 |
| | ep116 | **0/1/0** | **33,0 / 48,9** / 54,9 | 143 / **239** / 199 | 0/19/38 | +3 / +8 / +9 |
| val | combined_v1 | 0/0/0 | 32,7 / 53,0 / 60,8 | 145 / 185 / 198 | 0/26/36 | |
| | ep116 | 0/0/0 | 30,2 / 50,3 / 65,3 | 141 / 191 / 199 | 0/19/61 | |
| test | combined_v1 | 0/0/0 | 37,7 / 54,0 / 47,7 | 153 / 187 / 184 | 1/26/8 | |
| | ep116 | 0/0/0 | 35,1 / 49,6 / 50,6 | 128 / 170 / 171 | 0/6/13 | |
| mới 15/10–12/11 | combined_v1 | 0/0/0 | 38,8 / 54,3 / 53,8 | 168 / 188 / 196 | 2/5/10 | |
| | ep116 | 0/0/0 | 29,3 / 44,4 / 56,3 | 156 / 193 / 197 | 1/4/11 | |

- **Phát hiện quan trọng: `score_all_epochs.py` không chấm 48 mẫu `purged`** (24 h trước ranh giới train/val và val/test). Ở mẫu ranh giới **14/09 21:00 VN, +6 h**, cả 6 epoch "0 lỗi" đều dự báo thiếu: ep75 −209, ep90 −225, ep97 −234, ep114 −236, ep115 −233, ep116 −239 mm. `combined_v1` thì đạt 0 cả ở đây.
- **Kết luận:**
  - Theo tiêu chí lỗi lớn trên **mọi giờ**, `combined_v1` vẫn hơn (0 so với 1).
  - ep116 hơn rõ ở +3/+6 h: MAE thấp hơn 5–10 mm, số lỗi > 150 mm ở +6 h giảm một nửa, ít thiên cao hơn.
  - ep116 kém hơn một chút ở +12 h.
  - ep116 gọn hơn (1 model, 279 nghìn tham số, 1 bộ đầu vào) và là model duy nhất dùng được để finetune sang DV (không có embedding).
- **Chấm lại cả 600 epoch của 5 run, có tính 48 mẫu `purged`** (GPU, ngày 17/11/2026): **không có epoch nào đạt 0 lỗi trên mọi giờ**.
  - Tốt nhất có 1 lỗi, sát ngưỡng: hw 1/1/2,5 s42 ep63 (lỗi duy nhất val +6 h, 200,2 mm); hw 1/1,5/2,5 s42 ep61 (val +12 h, 200,3 mm); hw 1/1/2 s42 ep61 (200,6 mm).
  - Sáu epoch "0 lỗi" trước đó đều chỉ sai ở mẫu ranh giới 14/09 21:00 +6 h (209–239 mm).
- **Bài học:** từ nay chấm cả `purged`, hoặc chấm theo khoảng thời gian liên tục bằng `evaluate combined`, để tránh bỏ sót.

## Loss phạt lỗi lớn (tail penalty), thêm ngày 17/11/2026

- **Nhận xét của người dùng:** loss chỉ là MSE, không có tín hiệu trực tiếp cho thấy lỗi > 200 mm là rất tệ, nên sai số lớn nhất luôn sát 200.
- **Phân tích của Claude:** sát 200 chủ yếu là hiệu ứng chọn epoch. Sai số lớn nhất trên mọi giờ, các epoch ≥ 30 của run seed 42: thấp nhất 200, trung vị khoảng 222, cao nhất khoảng 238. Không epoch nào dưới 200.
- **Lựa chọn có sẵn đã loại:**
  - `large-error-count` (sigmoid quanh 200 ± 20 mm): gradient tắt khi sai số xa 200, không dùng chung được với under-weight; ở DV ngày 11/11 cho kết quả kém hơn.
  - `--lstm-tail-penalty`: không dùng được với `--change-loss`.
- **Người dùng đồng ý** thêm số hạng mới (luôn chấp nhận MAE tăng để đạt 0 lỗi):
  - `--change-tail-weight λ --change-tail-start-mm 150`;
  - khoản phạt λ·(max(|e| − 150, 0)/d)², tính trên sai số gốc, d = max(σ_h, 200);
  - code ở `change_objective.py` và `train.py`, test `src/ai/tests/test_change_tail.py` (5 test); các test loss cũ vẫn đạt;
  - MLflow giờ ghi thêm `change_margin_mm`, `change_tail_weight`, `change_tail_start_mm`.
- **Độ mạnh, so gradient phần tail với phần base tại e = 220 mm** (base có gradient ∝ e, tail ∝ λ·(e − 150)): λ = 3 tăng thêm khoảng 95%, λ = 10 tăng khoảng 3,2 lần. Với λ = 1 chỉ tăng khoảng 32%, quá yếu. **Đề xuất thử λ = 3 và λ = 10.**

### Kết quả tail penalty (người dùng train λ = 3 và 10 từ 150 mm; chấm ngày 17/11/2026, có cả train và giờ ranh giới)

- **Trên train, model đã gần như không còn sai số lớn.** Ở mọi epoch ≥ 30: 0 dự báo sai > 200 mm, trung vị chỉ **1 dự báo** (trên hơn 18 nghìn) sai > 150 mm, sai số lớn nhất khoảng 150–175 mm. Vì vậy phần tail (bắt đầu từ 150 mm) gần như không có gì để phạt, và λ = 10 hay λ = 3 khác nhau rất ít.
- **Lỗi còn lại đều trên dữ liệu model chưa thấy, và lặp lại ở cùng vài giờ trong gần như mọi epoch:**
  - val 13/09 07:00–08:00, +12 h, dự báo thừa 205–216 mm (100% epoch khi λ = 10);
  - mới 09/11 09:00–10:00, +12 h, dự báo thiếu 205–217 mm (43–92%);
  - val 31/08 05:00–01/09 01:00, +12 h, dự báo thừa (11–65%);
  - val 11/08 21:00, +12 h; ranh giới 14/09 21:00, +6 h.
- **So với run gốc đúng** (cùng công thức, under 0,3, không tail), tail **không giúp**. Bảng tính trên val + test + mới + ranh giới, epoch ≥ 30:

| Run | Tổng lỗi (trung vị) | Tổng lỗi (thấp nhất) | Sai số lớn nhất (trung vị) | Lỗi > 200 trên train |
|---|---|---|---|---|
| under 0 | 10 | 4 | 232 | 0 |
| under 0,2 | 6 | 1 | 222 | 0 |
| **under 0,3 (gốc)** | **5** | **1** | 223 | 0 |
| u0,3 + tail 3 | 7 | 3 | 225 | 0 |
| u0,3 + tail 10 | 6 | 3 | 223 | 0 |

  - Under-weight 0,3 vẫn là tốt nhất; bỏ under-weight (0) thì kém rõ.
  - Lúc đầu Claude so tail với nhầm run under 0 nên báo "tail có giúp ít"; đã sửa lại.
- **Kết luận:** loss chỉ "nhìn thấy" train. Không thể dùng loss để sửa lỗi trên dữ liệu chưa thấy khi lỗi đó không có trên train. Đây là giới hạn khái quát hóa và thông tin đầu vào, không phải do model "không chịu học".

### Thêm 5 run tail (người dùng train, chấm ngày 17/11/2026 trên mọi giờ, gồm cả ranh giới)

| Run (tất cả 1/1,5/2,5, seed 42, under 0,3) | Epoch tốt nhất | val | test | mới | ranh giới | Tổng | Sai số lớn nhất | Trung vị tổng, ep ≥ 30 |
|---|---|---|---|---|---|---|---|---|
| tail 10 từ 100, b64, lr 1e-4 | 72 | 0/0/3 | 0/0/0 | 0/0/1 | 0/0/0 | 4 | 217 | 6 |
| tail 10 từ 100 + event 1, b64, lr 1e-4 | 117 | 0/0/2 | 0/0/0 | 0/1/0 | 0/0/0 | 3 | 224 | 6 |
| tail 10 từ 150, b128, lr 1e-4 | 107 | 0/0/2 | 0/0/0 | 0/0/0 | 0/0/0 | 2 | 226 | 6 |
| tail 10 từ 150, b128, lr 3e-4, 120 ep | 40 | 0/0/2 | 0/0/0 | 0/0/0 | 0/0/0 | 2 | 208 | 18 |
| tail 10 từ 150, **b64** (thư mục ghi nhầm b128), lr 3e-4, 75 ep | 23 | 0/1/1 | 0/0/0 | 0/0/0 | 0/0/0 | 2 | 208 | 17,5 |

- **Không có epoch nào đạt 0 lỗi trên mọi giờ**, tính trên tất cả 14 run không embedding (khoảng 1.600 epoch).
- Tốt nhất vẫn là các epoch có 1 lỗi sát ngưỡng (200,2–200,6 mm) của các run không tail: hw 1/1/2,5 ep63, hw 1/1,5/2,5 ep61, hw 1/1/2 ep61.
- lr 3e-4 kém ổn định hơn: trung vị tổng 17–18 lỗi, so với 5–6 ở lr 1e-4.

### Bập bênh giữa +6 h và +12 h (người dùng phát hiện; Claude phân tích ngày 17/11/2026)

Lưu dự báo của mọi epoch của run gốc (1/1,5/2,5, seed 42, under 0,3) trên mọi giờ, rồi theo dõi các giờ khó:

| Giờ khó | Thay đổi thật | Sai số, ep 1–15 | ep 31–45 | ep 61–75 | ep 106–120 |
|---|---|---|---|---|---|
| 11/09 18:00 +6 h (nước lên) | +383 | −226 | −159 | −192 | −200 |
| 14/09 21:00 +6 h (nước lên) | +413 | −238 | −167 | −202 | −236 |
| 13/09 08:00 +12 h (nước rút) | −287 | +248 | +220 | +199 | +198 |
| 31/08 07:00 +12 h | −184 | +195 | +215 | +163 | +134 |
| 09/11 09:00 +12 h (nước lên) | +443 | −284 | −209 | −207 | −204 |

- **Các giờ khó di chuyển cùng nhau.** Tương quan qua các epoch (≥ 30) giữa sai số của chúng là 0,71–0,99, trộn lẫn cả +6 h và +12 h. Riêng 09/11 là yếu tố độc lập.
- **Cơ chế:**
  - Ở các giờ khó, lỗi +6 h là dự báo **thiếu khi nước lên** (sai số âm), còn lỗi +12 h là dự báo **thiếu mức rút khi nước rút** (sai số dương). Cả hai đều do model **nén biên độ**: trên tập đánh giá, độ dốc giữa thay đổi dự báo và thay đổi thật chỉ 0,80 (+6 h) và 0,63 (+12 h).
  - Khi train tiếp, độ dốc trên **train** tăng (0,60 → 0,91 ở +6 h; 0,47 → 0,86 ở +12 h), nhưng trên tập đánh giá **đứng yên** sau epoch khoảng 16. Model chỉ học thuộc train.
  - Thứ duy nhất thay đổi ở các giờ khó là **một mức lệch chung**. Độ lệch trung bình +6 h trên tập đánh giá giảm từ 28 xuống 17 mm. Mức lệch chung đi xuống thì lỗi nước lên (+6 h) xấu đi và lỗi nước rút (+12 h) tốt lên, và ngược lại.
  - Tổng độ lớn hai lỗi ≈ 390–410 mm, nên **không có mức lệch nào giữ được cả hai dưới 200 mm**. Vì vậy các epoch tốt nhất đều nằm ở 198–200 mm.
- **Kiểm chứng:** ghép theo mốc từ **cùng một run** (+3/+6 h từ epoch sớm, +12 h từ epoch muộn) cho **0 lỗi trên mọi giờ**: 120 cặp như vậy. Ví dụ +3/+6 h từ ep37 (sai số lớn nhất 187) và +12 h từ ep69 (198). Đây cũng là lý do `combined_v1` đạt 0.
- **Hệ quả:**
  - Một model dùng chung một thân GRU cho cả 3 mốc không đạt được 0 lỗi với thông tin hiện có, trừ khi sửa được việc nén biên độ.
  - Các hướng: (1) ghép theo mốc (từ hai epoch hoặc hai model); (2) tách phần dùng chung giữa các mốc; (3) thêm thông tin để model dự báo đúng biên độ ở bão.

### Hướng sửa (17/11/2026)

- **Hướng 2, đã code:** `--horizon-heads separate`. Mỗi mốc có tầng ghép `Linear(H → H)` + ReLU và head riêng; GRU vẫn dùng chung.
  - Test: `src/ai/tests/test_horizon_heads.py` (5 test, có test kiểm tra đổi tầng của mốc 0 không làm đổi đầu ra mốc khác).
  - Dự báo `combined_v1` không đổi.
  - Số tham số model H128: 278.403 → 311.427.
- **Hướng 3, chưa làm.** Chẩn đoán: nén biên độ chỉ xảy ra trên dữ liệu chưa thấy (độ dốc 0,80 / 0,63, trong khi train 0,9), tức là "co về trung bình" khi đầu vào không đủ để phân biệt. Ứng viên:
  - (a) **Thêm thông tin.** Phân tích ngày 13/11: dự báo của model Thuận An đã thấy trước 4 lần nước lên +6 h ở val (+264…+334 mm), trong khi model NBS chỉ dự báo +108…+228. Ứng viên tốt cho các lỗi +6 h. Các lỗi +12 h lúc nước rút thì model TA cũng sai.
  - (b) **Thêm ví dụ:** khi đã chốt công thức, train model cuối trên train + val + test (có bão 13/09). Chỉ kiểm chứng được trên dữ liệu sau 12/11.
  - (c) Kéo giãn biên độ sau khi dự báo, hoặc tăng regularization: không khuyến nghị (thêm báo động giả, hoặc không làm tăng độ dốc trên tập đánh giá).

### Kết quả head riêng cho từng mốc (người dùng train, chấm ngày 17/11/2026 trên mọi giờ)

| Run (1/1,5/2,5, seed 42, under 0,3) | Trung vị tổng lỗi, ep ≥ 30 | Thấp nhất | Tương quan lỗi giờ khó +6 h với +12 h |
|---|---|---|---|
| head chung (gốc) | 5 | 1 | 0,83 |
| head riêng 128 × 1 lớp | 7 | 4 | 0,76 |
| head riêng 32 × 3 lớp (100 epoch) | 13 | 9 | 0,67 |

- **Head riêng không giúp.** Lỗi nhiều hơn, và các giờ khó vẫn di chuyển cùng nhau. Vậy mức lệch chung nằm trong GRU, không nằm trong head.
- **Đã code hướng ở giữa:** `--horizon-gru-layers K`. K lớp GRU trên cùng thành nhánh riêng cho từng mốc; các lớp dưới (`--lstm-num-layers` − K) dùng chung; giữa phần chung và nhánh có dropout như GRU xếp chồng. Cần `--horizon-heads separate`.
  - Tham số, GRU 128×3 với 35 đầu vào: K = 1 cho 509.571; K = 2 cho 707.715.
  - Test trong `test_horizon_heads.py`: đổi nhánh của mốc +12 h không làm đổi +3/+6 h; đổi phần chung thì đổi tất cả.

- **Kết quả `--horizon-gru-layers 1`** (2 lớp GRU chung + 1 lớp riêng mỗi mốc, 120 epoch): mức gắn nhau giữa giờ khó +6 h và +12 h **giảm còn 0,32** (gốc 0,83), tức bập bênh đã bị cắt. Nhưng lỗi **tăng ở cả hai mốc**: trung vị tổng 14 (gốc 5), thấp nhất 7; trung vị +6 h là 7, +12 h là 8. Học chung giữa các mốc giúp model khái quát hóa. Tách ra thì hết bập bênh nhưng mỗi mốc đều kém đi, giống model chỉ +12 h trước đây (test 3–9 lỗi).
- Người dùng muốn thử phương án cuối: **mỗi mốc một model riêng**. Lưu ý: đã có sẵn cách "mỗi mốc một model" đạt 0 lỗi trên mọi giờ mà không cần train thêm, là lấy +3/+6 h và +12 h từ hai epoch của cùng run gốc (ví dụ ep37 và ep69).

## Ý tưởng (để thảo luận)

Phần này chưa chốt, sẽ cập nhật qua từng buổi thảo luận.

1. **Làm sạch dữ liệu trước:** loại hai đoạn sensor đứng và lọc gai (Task 1). Sau đó mới dựng lại một dataset cơ sở.
2. **Cửa sổ lịch sử ngắn (12–25 h)**, giống NBS, để giữ được trận lũ 11/2025 trong tập train.
   - Với `sequence_length` 96 trong config, mẫu đầu tiên là 18/11/2025 16:00 UTC. Toàn bộ pha lên và đỉnh lũ (đỉnh 17/11 07:10 UTC) bị mất.
   - Lý do: tập mẫu được lọc theo 80% của cửa sổ L trong config. Khi train với `--lstm-seq-len` ngắn hơn, chỉ có thể bớt mẫu, không thêm lại được.
   - Muốn giữ trận lũ thì phải đặt `sequence_length` nhỏ ngay trong config.
3. **Áp lại công thức của NBS**, mỗi lần thêm một nhóm và sàng lọc tuyến tính trước:
   - mưa dự báo ECMWF;
   - gió vào bờ;
   - triều 7 ngày (thử cả 30 ngày);
   - mực nước biển dự báo;
   - trọng số theo mốc;
   - ghép theo mốc.
4. **Dòng chảy sông có vai trò lớn hơn ở NBS** (chênh lệch DV − NBS tăng mạnh mùa lũ), nên có thể cần thêm thông tin về lượng nước từ thượng nguồn. Ví dụ: mưa dự báo ở các ô thượng nguồn (A Lưới, Bạch Mã) với cửa sổ cộng dồn dài hơn.
