# Trạm Dã Viên (0074647403): ghi chú và ý tưởng

Bắt đầu ngày 17/11/2026. Quy trình chung theo `models/PIPELINE.md`. Bài học từ Ngã Ba Sình xem trong `models/NOTES.md` (phần B) và `models/nga_ba_sinh/REPRODUCE.md`.

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
