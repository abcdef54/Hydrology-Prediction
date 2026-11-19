# Giai đoạn 6: Kết hợp mô hình toán học với mạng nơ-ron (hybrid)

Giai đoạn này nối tiếp trực tiếp giai đoạn mô hình toán học: giữ phần "triều tường minh có neo vào quan trắc" của giai đoạn 5, nhưng thêm các phần học bằng mạng nơ-ron hoặc thông tin ngoài. Gồm bốn đợt nghiên cứu liên tiếp trên cùng một dữ liệu và cùng quy tắc chấm (H0–H8, CFW-TD, AMLMN, ERA), rồi một đợt cuối làm lại dữ liệu đầu vào của Dã Viên. Các tên H0, H8, CFW-TD… cũng là nhãn nội bộ; phần dưới giải thích từng cái.

## 1. Thời gian

Khoảng 10/11/2026 đến 11/11/2026, ngay sau M7. Kế hoạch H0–H8 và các đợt sau đều được đóng băng (giao thức, mã nguồn, hash) **trước khi chạy**, và tập test chỉ chấm một lần sau khi đóng băng toàn bộ checkpoint.

## 2. Trọng tâm

Ý tưởng chung: giai đoạn 5 cho thấy mô hình toán (triều + neo) ổn định nhưng nghèo thông tin, còn GRU linh hoạt nhưng không tổng quát hóa. Vậy **kết hợp hai bên**: mô hình toán làm nền, phần học chỉ phải dự đoán phần còn lại.

**Mô hình nền dùng chung.** Là các bản đóng băng của giai đoạn 5, đều neo vào mực nước hiện tại H(t) (xem phụ lục của [05](05_mo_hinh_toan_hoc.md)):

| Nhãn | Công thức | Ý nghĩa |
|---|---|---|
| A1 | H(t) + ΔT(h) + trôi nền | triều từ bộ lọc Kalman, neo vào quan trắc, có trôi nền của M3 |
| A3 | H(t) + ΔT(h) | như A1 nhưng bỏ trôi nền |
| C1 | A1 với hệ số triều hiệu chỉnh lại (bán nhật 1,03; nhật triều 1,24) | |

### 2.1 Đợt 1: H0–H8, chín cách kết hợp

| Nhãn | Tên mô tả | Cách làm |
|---|---|---|
| H0 | GRU từ persistence | GRU đơn thuần học phần thay đổi so với mực hiện tại; một bản chỉ có mực nước (H0-raw), một bản thêm đặc trưng toán (H0-math) |
| H1 | Nền toán + GRU hiệu chỉnh | GRU học **phần dư** so với dự báo của A1, C1 hoặc A3 |
| H2 | Ensemble cố định | Trung bình cố định giữa dự báo toán và GRU (0,5 : 0,5, hoặc trọng số học trên train) |
| H3 | Ensemble thích nghi | Một cổng nhỏ chọn trọng số theo trạng thái (hằng số, theo mực nước, theo trạng thái toán) |
| H4 | Triều Kalman + GRU dự báo mức nền | Triều tường minh của A3; GRU chỉ dự báo mức nền đã trừ triều thay đổi bao nhiêu sau h giờ |
| H5 | Như H4, đầu ra cộng dồn | GRU dự báo các gia số từng giờ rồi cộng lại (có bản thêm mục tiêu trung gian) |
| H6 | Mạng "toán có cấu trúc" | Mạng MLP nhỏ (khoảng 3.000 tham số) điều khiển từng khối vật lý: quay về mức cũ sau lệch, thay đổi nền, tiếp tục theo độ dốc, hiệu chỉnh biên độ triều |
| H7 | Trạng thái khả vi | Bộ lọc Kalman của M3 cho phép học nhiễu Q, hằng số thời gian nền, hoặc một mạng nhỏ dự báo nền (5 đến 1.731 tham số) |
| H8 | Phần dư neo theo chế độ dòng chảy | Nền A1 cộng nhánh "quay về mức cũ" mạnh yếu theo trạng thái (γ_t, τ_t do mạng quyết định) và nhánh động học cộng dồn, kèm đầu phân loại (nước rút, thường, nước lên) và đầu phân vị |

Có thêm nhánh regional (6 trạm, nhúng định danh trạm, chỉ lấy dự báo của trạm đích khi suy luận) để kiểm tra chia sẻ giữa các trạm. Tổng cộng 33 cấu hình học × 3 seed, cộng 3 ensemble chưa học, 3 nền đóng băng và các model M tham chiếu; 99 checkpoint được băm trước khi chấm test.

### 2.2 Đợt 2: CFW-TD (sóng lũ nhân quả + tách triều)

Mô hình vật lý tường minh hơn nữa: nền A3 cộng các thành phần có ý nghĩa thủy văn: (1) quay về mức cũ của chuỗi đã trừ triều, (2) lan truyền sóng lũ từ lượng xả hồ chứa Bình Điền, Tả Trạch qua nhân trễ (3 giờ), (3) lan truyền mưa quan trắc trên phần lưu vực không bị điều tiết qua nhân Nash, (4) duy trì theo hàm mũ với hằng số riêng, (5) neo đúng tại h = 0. C0 đến C6 là các mức độ đủ dần các thành phần; C6 thêm mạng trạng thái 524 tham số điều chỉnh 6 tham số vật lý. Có 10 ablation bỏ từng thành phần.

### 2.3 Đợt 3: AMLMN (mạng toán học nâng ma trận từng nguyên tử)

Lấy hai công thức tốt nhất, M7-C1 và CFW-TD C6, và **nâng từng toán hạng** (hệ số, tần số, độ khuếch đại, hằng số thời gian) từ số vô hướng thành vector hoặc ma trận học được, giữ nguyên cây biểu thức và thứ tự phép tính. Mục đích: cho công thức nhiều bậc tự do hơn mà vẫn giữ cấu trúc vật lý. 13 biến thể × 3 seed × 2 họ công thức = 78 lần huấn luyện.

### 2.4 Đợt 4: ERA (định tuyến mưa thượng nguồn thích nghi theo sự kiện)

Mô hình nền A1 cộng một cách định tuyến lượng mưa quan trắc ở thượng nguồn đến Dã Viên, thích nghi theo từng sự kiện lũ, nhất quán với đường cong lưu lượng. **Không có tham số nào học bằng gradient**, nên không phụ thuộc việc train có chứa lũ lớn hay không. Chấm từng nhánh đúng một lần.

### 2.5 Đợt cuối: xây dựng lại bộ feature đầu vào cho Dã Viên và 25 lần train tự động (11/11)

Sau khi các hướng mô hình đều không thành công, nghiên cứu quay lại GRU thuần và **thiết kế lại dữ liệu đầu vào**. Đây là đợt có nhiều đóng góp lâu dài nhất của giai đoạn: phần lớn feature được tạo ra ở đây sau đó trở thành thành phần chuẩn của quy trình từng trạm (giai đoạn 7–9). Đợt này chạy bằng 25 lần train tự động (một seed, seed 17), với mục tiêu 0 lỗi > 200 mm ở +3/+6/+12 h; tiêu chí chọn là tổng lỗi validation cộng test, nên không còn tập độc lập và mọi con số đều lạc quan.

Các feature và kỹ thuật được xây dựng (chi tiết ở mục 7):

- **Cụm trạm mưa theo hướng:** Tây (A Lưới: Nhâm, Hồng Thái; là đầu vào gốc), Đông thượng nguồn (Bạch Mã, Đỉnh Bạch Mã) và Đồng bằng (Phú Bài, VP Huế, Bình Điền, các trạm còn lại).
- **Dự báo ECMWF theo ô lưới chọn lọc:** bảy ô lưới quanh lưu vực, chọn A Lưới, Thuận An và Bạch Mã thấp; thêm đặc trưng "dự báo sắp tới so với mưa vừa đo".
- **Cửa sổ lịch sử ngắn** (6, 12, 24, 25 giờ thay cho 96 giờ), lưới dữ liệu mở rộng lùi 120 giờ, mưa dựng lại từ số đo 10 phút gốc.
- **Gió và áp suất ngoài khơi** (Open-Meteo, Previous Runs), dùng bản dự báo khoảng 24 giờ trước nên mọi giờ đến t + 12 đều đã có sẵn tại thời điểm dự báo.
- **Triều thiên văn** (hồi quy hài hòa trên 7 ngày gần nhất) thêm vào làm nhóm feature riêng.
- **Biến đổi log1p** cho lượng mưa và lượng mưa dự báo, **dropout theo nhóm đầu vào** khi train, **trọng số cho đợt lũ**, loss đếm lỗi lớn, và mực nước các trạm khác (Ngã Ba Sình, Thuận An, Tứ Phú).

## 3. Dữ liệu sử dụng

- Mực nước Dã Viên (5.999 / 1.080 / 711 mốc, như giai đoạn 5) cho đợt 1–4; thêm cho riêng CFW-TD: lượng xả hồ chứa (độ sẵn có đã kiểm tra: múi giờ, thời điểm nhập, độ phủ), mưa quan trắc từng trạm, địa hình FABDEM để xác định lưu vực (Dã Viên 1.550 km²: khoảng 683 km² trên đập Tả Trạch, 517 km² trên Bình Điền, khoảng 350 km² không điều tiết).
- Đợt cuối dùng các dataset local một trạm (`train_data/da_vien_dataset_revision/`, mốc +3/+6/+12 h, chia cố định) gồm: cụm trạm mưa, ô ECMWF, gió và áp suất ngoài khơi (Open-Meteo Previous Runs, `ecmwf_ifs` 0,25°), triều 7 ngày, mực nước trạm khác.
- Ghi chú dữ liệu: **không có trạm mực nước nào ở thượng nguồn Dã Viên**; sáu trong tám trạm mưa A Lưới thoát nước sang Lào; không có phiên bản dự báo thời tiết lịch sử nào được xác minh độ sẵn có.

## 4. Kết quả

Số lỗi > 200 mm cộng ở +6/+12 h (chọn theo validation, test chấm một lần sau khi đóng băng):

| Model | Validation | Test khám phá |
|---|---:|---:|
| Nền toán đóng băng A1 | 33 | **34** |
| Nền toán đóng băng A3 | 35 | 32 |
| M6 (neo + trôi nền, giai đoạn 5) | 33 | 31 |
| H4 (triều Kalman + GRU mức nền), model thắng theo quy tắc chọn | **30** | 98 |
| H0-raw (GRU thuần) | 38 | — |
| H7-Q, H7-B, H7-J (trạng thái khả vi, mọi seed) | 34 | 32–34 |
| H8-E4 (phần dư neo theo chế độ) | 31 | — |
| CFW-TD, nhánh tốt nhất (C3/C4) | 33 | 32 |
| CFW-TD đầy đủ (C5) | 43 | 32 |
| AMLMN (tốt nhất mỗi họ) | 35 / 35 (các bản gốc: 33 / 34) | — |
| ERA | 32 | 39 |
| 25 lần train GRU ở đợt cuối, model đơn tốt nhất | 61 (val + test, 29 + 32) | |
| Ghép nhiều model (GRU + CFW-TD + H8-E4), chọn trên val + test | 37 (25 + 12) | |

Các điểm chính:

- **H4 thắng quy tắc chọn nhờ chọn seed:** các seed của H4 dao động 30–45 lỗi ở validation, và seed được chọn có 98 lỗi ở test, so với 34 của A1.
- **Mọi thành phần học khớp train gần như hoàn hảo** (H0: 0 lỗi trên train) nhưng không tổng quát hóa; kết quả validation và test thường **ngược nhau** giữa các seed và các ablation.
- **Mô hình bền nhất là các mô hình toán đóng băng** (A1, A3, M6), cùng các biến thể H7 chỉ học vài tham số.
- AMLMN: nâng ma trận **không cải thiện**; các biến thể giảm lỗi train gần một nửa (178 → 85 ở họ A) nhưng làm validation tệ hơn. Ở họ A đó là hệ quả toán học: nâng ma trận chỉ ở phần dự báo trùng đúng với nâng vector, tức một hàm tuyến tính của 11 trạng thái Kalman, tối ưu của lớp đó là 65 lỗi ở validation.
- Con số 37 lỗi ở đợt cuối chọn trên cả validation và test nên lạc quan; kiểm tra chéo cho ước lượng ngoài mẫu khoảng 45–55.

## 5. So với giai đoạn trước

Không tốt hơn. Các mô hình toán đóng băng của giai đoạn 5 (33–35 lỗi validation, 31–34 test) vẫn là mức bền nhất; các kết hợp với GRU hoặc thêm thông tin chỉ **đổi chỗ** lỗi giữa validation và test. So với GRU regional của giai đoạn 3 (validation 49, test 39), đợt cuối không có model đơn nào vượt.

## 6. Lỗi > 200 mm

**Không xử lý được.** Không model nào đạt 0 lỗi ở bất kỳ tập nào. Các lỗi còn lại tập trung ở +6/+12 h, trong ba đợt: 12/09 (mưa lớn rơi sau thời điểm phát hành), 13/09 (nước rút theo mực nước biển và gió), và 7–13/10 (ECMWF bỏ lỡ mưa).

## 7. Điều mới rút ra

### 7.1 Bộ feature đầu vào mới, được dùng lại ở các giai đoạn sau

Đợt cuối của giai đoạn tạo ra nhiều feature và kỹ thuật dựng dữ liệu mà các giai đoạn sau dùng lại. Lý do ra đời và nơi được dùng lại:

| Feature / kỹ thuật | Nội dung | Vì sao được tạo | Dùng lại ở |
|---|---|---|---|
| **Cụm trạm mưa theo hướng** | 3 cụm Tây, Đông, Đồng bằng; mỗi cụm có tổng mưa 1/3/6/12/24/48 giờ (log1p) và tỉ lệ trạm thiếu | Lỗi tháng 10 xảy ra vì các trạm phía Tây gần như không đo được mưa rơi ở phía Đông | Nhóm `rain_obs` của pipeline từng trạm; Dã Viên +3 h (giai đoạn 9) |
| **Ô lưới ECMWF chọn lọc** | 7 ô quanh lưu vực; ô A Lưới, Thuận An, Bạch Mã thấp; trung bình lưu vực; dự báo sắp tới so với mưa vừa đo | Trước đợt nước lên 12/09, dự báo 0–12 giờ ở A Lưới rất lớn (72–230 mm) trong khi mưa đo ở các trạm Tây chưa phản ánh | Nhóm `fc_ecmwf` (7 ô, `a_luoi`, `thuan_an`, `bach_ma`) ở Thuận An, Ngã Ba Sình, Dã Viên |
| **Cửa sổ lịch sử ngắn** | Cửa sổ 6–25 giờ, lưới mở rộng lùi 120 giờ, dựng lại mưa từ số đo 10 phút; bản `nonov` bỏ các mẫu đầu chuỗi | Cửa sổ 96 giờ làm mất trận lũ 15–18/11/2025 (đỉnh 2.989 mm); đưa lại thì model dự báo thừa sau đỉnh | Thuận An (cửa sổ 12 giờ, 25 giờ), Dã Viên (cửa sổ 12 giờ), cơ chế "một tập mẫu cho nhiều cửa sổ" của pipeline (`max_sequence_length`) |
| **Gió và áp suất ngoài khơi** | 15 cột: gió vào bờ, dọc bờ, tốc độ trung bình 24 giờ qua, 6 giờ tới, 12 giờ tới, thay đổi, áp suất và thay đổi áp suất; chỉ dùng dự báo `previous_day1` | Đợt nước rút 13/09 do gió vào bờ giảm (xem 7.2) | Nhóm `wind` ở Thuận An; xử lý được đợt bão 12/09 trên validation |
| **Triều 7 ngày** | Nhóm feature riêng: hồi quy hài hòa trên 7 ngày gần nhất, có dự phòng bằng hồi quy toàn cục trên train | Tách phần triều khỏi phần nước dâng và lũ | Nhóm `tide` (nâng lên 30 ngày ở Thuận An), liên hệ với chu kỳ 12,42 h ([04](04_weight_space_programming.md), [05](05_mo_hinh_toan_hoc.md)) |
| **Biến đổi log1p** | Lượng mưa và lượng mưa dự báo qua log(1 + x) thay cho mm | Mưa có đuôi rất dài; GRU chuẩn hóa tuyến tính nên ngoại suy sai khi mưa lớn | Toàn bộ nhóm mưa và dự báo mưa trong pipeline |
| **Dropout theo nhóm đầu vào** | Khi train, tắt ngẫu nhiên cả một nhóm feature (`--group-dropout`) | GRU dùng dự báo ECMWF như đường tắt, bỏ qua mưa đo (test 52 → 35 khi có dropout) | Nhóm dự báo của chính model; thử và loại ở Thuận An |
| **Loại đoạn dữ liệu hỏng** | Loại sensor bị treo, lọc nhiễu đột biến | Sensor Dã Viên báo giá trị không đổi nhiều ngày | Bước dựng dataset chung (`invalid_ranges`, `despike`) |

Các nhóm feature này nay là thành phần chuẩn của `src/ai/station_dataset` và có mặt trong các model đạt 0 lỗi > 200 mm ở Thuận An, Ngã Ba Sình và mốc +3 h của Dã Viên. Đây là đóng góp lâu dài nhất của đợt thử nghiệm này, dù bản thân các model Dã Viên của đợt này chưa đạt mục tiêu.

### 7.2 Phát hiện khác

- **Hai cơ chế lỗi có nguyên nhân xác định:**
  - *Ở validation* là một trận lũ do mưa: trạng thái mưa thượng nguồn gấp 2,5–9 lần giá trị lớn nhất của train, và các mẫu train gần nhất cách xa gấp ba lần so với mẫu thường.
  - *Ở test* là chuyển chế độ: biên độ dao động và khoảng biến thiên gấp khoảng hai lần train, và các mạng nơ-ron mang xu hướng quay về mức trung bình của train.
  - Không kiến trúc nào chữa được cả hai với dữ liệu này.
- **Thông tin còn thiếu, không phải kiến trúc, là giới hạn.** Dã Viên không có trạm mực nước thượng nguồn; hệ số truyền mưa học trên train nhỏ hơn 5–11 lần so với trận lũ tháng 9, kể cả khi có mưa tương lai hoàn hảo; ở đợt nước lên ngày 12/09 khoảng một nửa lượng mưa gây lũ rơi sau thời điểm phát hành. Dự báo ECMWF hồi cứu giảm sai số của đợt nước lên này khoảng 46 mm, nhưng độ sẵn có của dự báo đó chưa được xác minh.
- **Mưa quan trắc thượng nguồn có thật thông tin về pha rút sau đỉnh:** ERA loại bỏ 109–160 mm dự báo thừa +12 h ở mọi mốc từ 3h đến 8h ngày 13/09, không cần khớp tham số vào sự kiện đó. Nhưng nó hỏng ở test vì hệ số thích nghi bị nhiễu từ phần dư triều khi biên độ triều gấp đôi.
- **Đợt nước rút 13/09 do mực nước biển và gió, không phải lũ sông.** Dã Viên đạt đỉnh cùng Thuận An và Ngã Ba Sình rồi cùng rút, trong khi Tứ Phú và Ô Lâu còn lên. Gió vào bờ trung bình 24 giờ tương quan 0,69 với mực nước nền ở Thuận An; hồi quy gió cộng áp suất giải thích 51% phương sai. Khi gió vào bờ tụt từ 8,9 xuống 0,7 m/s, mực nước biển và Dã Viên rút theo. Đưa gió vào GRU của Dã Viên **không giúp** (test tệ hơn), nhưng đây là cơ chế quan trọng cho các trạm gần biển ở giai đoạn sau, nơi gió dự báo xử lý được đợt bão 12/09 (xem 7.1).
- **Tín hiệu báo trước từ trạm khác không nhất quán:** chênh lệch mực nước Dã Viên so với Thuận An tương quan −0,72 với thay đổi +12 h ở validation nhưng chỉ +0,11 ở train và +0,09 ở test. Chỉ có "độ vượt trên mức nền" của chính Dã Viên là ổn định, và GRU đã tự học được.
- **Mục tiêu 0 lỗi không khả thi với thông tin trong dữ liệu hiện có** (kết luận của ERA và của đợt cuối).

## 8. Kết luận

Kết hợp toán và mạng nơ-ron không thành công: không cách kết hợp nào, từ ensemble đến định tuyến vật lý, đến nâng ma trận, vượt được nền toán đóng băng một cách bền vững, và 25 lần train thêm với dữ liệu đầu vào được sửa cũng không đưa lỗi về 0. Khuyến nghị của báo cáo H0–H8: giữ nền triều neo đóng băng (A1 hoặc A3) làm tham chiếu, không triển khai model học nào của họ H. Ba kết quả đáng giữ: (1) bộ feature đầu vào mới (cụm trạm mưa, ô ECMWF chọn lọc, cửa sổ ngắn, gió, log1p…), nền tảng của các model thành công ở giai đoạn sau; (2) hiểu rõ giới hạn thông tin của Dã Viên (không có thượng nguồn, không có dự báo đã kiểm chứng); (3) cơ chế nước biển và gió cho đợt rút 13/09. Từ đây nghiên cứu tạm dừng ở Dã Viên và chuyển sang các trạm gần biển với quy trình mới, bắt đầu từ Thuận An ([07](07_thuan_an.md)).

## Nguồn

- `reports/hybird_experiments/plan.md`, `OPEN_ENDED_QUESTION.md` (kế hoạch và đề bài), `reports/da_vien_hybrid_h0_h8_experiments/` (`FINAL_REPORT.md`, `ARCHITECTURE_DEFINITIONS.md`, `SCIENTIFIC_SYNTHESIS.md`, `PREDICTABILITY_ANALYSIS.md`, `H8_NEW_METHOD.md`, `REGIONAL_EXPERIMENTS.md`)
- `reports/da_vien_cfwt_d_experiments/FINAL_REPORT.md`, `reports/da_vien_atomic_matrix_lifted_network/FINAL_REPORT.md`, `reports/da_vien_open_ended_experiments/FINAL_REPORT.md` (có `ERA2_PROPOSAL.md`, đề xuất chưa chạy)
- `reports/model_comparisons/da_vien_dataset_revision/` (`AUTONOMOUS_LOG_20261111.md`, `SESSION_SUMMARY_20261111.md`, `SELECTED_CHECKPOINTS.md`), `train_data/da_vien_dataset_revision/`
