# Giai đoạn 5: Mô hình toán học

Giai đoạn có nhiều model nhất: hơn 10 họ model (M0 đến M7, hiệu chỉnh pha...) với khoảng vài trăm lần hiệu chuẩn (fit), nên tài liệu chỉ tóm tắt từng họ và mối liên hệ giữa chúng.

## 1. Thời gian

Khoảng 3/11/2026 (hiệu chỉnh pha) đến 10/11/2026 (M7). Phần phân tích dữ liệu và M0–M2 làm ngày 9/11; M3 đến M7 làm trong 9–10/11. Chồng thời gian với giai đoạn WSP (4–6/11).

## 2. Trọng tâm

Chỉ làm **Dã Viên**. Sau khi đổi kiến trúc (giai đoạn 3) và sửa trọng số (giai đoạn 4) đều không hết lỗi, thử một hướng hoàn toàn khác: thay mạng nơ-ron bằng **mô hình có phương trình tường minh** về cách mực nước Dã Viên dao động theo triều và trôi theo nền, hy vọng nắm đúng cơ chế mà GRU nén biên độ. Mỗi model ít tham số (3 đến 26), có ý nghĩa vật lý, huấn luyện bằng gradient trên cả khối dữ liệu train.

Thứ tự các họ model (M0, M1… chỉ là nhãn nội bộ đặt ra trong lúc nghiên cứu; công thức và cấu trúc từng model ở [phụ lục bên dưới](#phụ-lục-đặc-tả-các-mô-hình-toán-học)):

| Nhãn | Tên mô tả | Ý tưởng |
|---|---|---|
| Hiệu chỉnh pha | Hiệu chỉnh pha cho GRU | Sửa dự báo của 3 model GRU hiện có bằng quỹ đạo dao động 12,42 h; có hồi quy ridge nhỏ |
| M0 | Persistence + triều | Mực nước hiện tại cộng phần thay đổi do triều (10 tham số) |
| M1 | + quán tính, mưa, trạm lân cận | Thêm quán tính, hai bể chứa mưa và độ dốc ở hai trạm lân cận (18 tham số) |
| M2 | Triều biên độ/pha thích nghi | Biên độ và pha triều thay đổi theo trạng thái dòng chảy (26 tham số) |
| M3 | Bộ dao động nhân quả (Kalman) | Bộ lọc Kalman theo dõi nền và 5 dao động triều trên 168 giờ lịch sử (7 tham số) |
| M3 không +1 h | | M3 bỏ mục tiêu +1 h khỏi hàm loss để kiểm tra có ảnh hưởng không |
| M4 | Khuếch đại triều theo từng mốc | Mỗi mốc +6 h, +12 h có bộ hệ số riêng; thử thêm các dạng trôi nền |
| M5 | Hai thang thời gian + neo + cổng | Thêm neo quan trắc, hai vận tốc nền nhanh/chậm, cổng pha trộn |
| Raw-unit | Tái hiệu chuẩn bằng MSE thô | Huấn luyện lại M0–M5 bằng mm hoặc cm thô thay cho chuẩn hóa |
| M6 | Neo + trôi nền bền vững | Neo vào quan trắc hiện tại, 2 tham số trôi nền mỗi trạm; thử chia sẻ giữa các trạm |
| M7 | Các biến thể kiểm tra M6 | Không phải model mới: tách tác dụng của neo, trôi nền, triều thích nghi |

Trước đó có một bước phân tích dữ liệu và cơ chế (không phải một model): phổ triều, quan hệ mưa, dữ liệu cửa đập, lỗi của GRU.

## 3. Dữ liệu sử dụng

- Chủ yếu **chỉ mực nước Dã Viên** (trung bình giờ). Không dùng mưa, trạm khác hoặc dự báo thời tiết trong các cấu hình chính; các đầu vào đó chỉ thử ở thí nghiệm phụ.
- Chia chuẩn: train 18/11/2025–29/07/2026 (5.999 mốc), validation 31/07–13/09 (1.080), test khám phá 15/09–14/10 (711). Mọi model, kể cả GRU, chấm trên đúng cùng mẫu.
- Mốc dự báo: M0–M2 dùng +1/+6/+12 h; M3, M4 dùng +6/+12 h; M5 trở đi thêm +3 h.
- Phân tích phụ dùng dữ liệu cửa đập Thảo Long (694 báo cáo, cách nhau khoảng 12 giờ, đơn vị chưa xác nhận).

## 4. Kết quả

Số lỗi > 200 mm cộng ở +6/+12 h, chọn model chỉ theo validation:

| Model | Validation | Test khám phá |
|---|---:|---:|
| GRU (đối chứng) | 50 | **26** |
| Persistence + triều / + quán tính, mưa / triều thích nghi (M0 / M1 / M2) | 52 / 65 / 44 | 149 / 178 / 83 |
| Bộ dao động nhân quả (M3, 7 tham số) | 34 | 49 |
| Khuếch đại triều theo mốc (M4, nhóm) | 37 | 41 |
| Hai thang thời gian + cổng (M5) | 34 | 48 |
| Hai thang thời gian, trộn cố định (M5-B) | 35 | 43 |
| Neo + trôi nền bền vững (M6) | 33 | 31 |
| Biến thể kiểm tra M6 (M7) | 33–36 | 34–39 |
| Hiệu chỉnh pha trên GRU A / B / C (validation → confirmation) | 40→33 / 51→47 / 48→45 | 21→22 / 19→19 / 45→38 |

Các nhận xét đi kèm:

- M0–M2: M2 thắng GRU ở validation (44 so với 50) nhưng thua nặng ở test (83 so với 26).
- M3: sửa được lỗi pha +6 h của M2, validation 34 (thắng GRU 50), nhưng test 49 (thua GRU 26) và tệ hơn ở +12 h.
- M4, M5, raw-unit: chỉ chênh vài lỗi, không có lợi ích khái quát ổn định. Đổi đơn vị mm/cm gần như không thay đổi gì.
- M6: test giảm 49 → 31, validation hầu như không đổi (34 → 33); MAE tệ hơn M3. Persistence và chia sẻ regional **không được chứng minh có lợi** (tham số π tự co về gần 0).
- M7: không có biến thể nào giảm được lỗi nặng so với M6.

## 5. So với giai đoạn trước

Các mô hình toán rất ít tham số mà vẫn gần bằng GRU ở validation (33–37 so với 50 lỗi), nhưng **trên test chưa model nào thắng GRU** (31–49 so với 26). Không tốt hơn giai đoạn 3 (GRU) về số lỗi lớn.

## 6. Lỗi > 200 mm

Không xử lý được. Tốt nhất là 31 lỗi (M6) trên test so với 26 của GRU; không model nào giảm được lỗi nặng (> 300 mm). Phân tích 53 cấu hình chồng lấn cho thấy 11 lỗi validation bị **mọi** model mắc (chủ yếu trong đợt 11–13/09, 27 giờ liên tiếp), tức là không phải lỗi của một kiến trúc cụ thể.

## 7. Điều mới rút ra

- **Triều ở Dã Viên không cố định:** phân tích phổ cho thấy biên độ M2 khoảng 95 mm vào tháng 12 nhưng chỉ khoảng 2 mm vào tháng 5; bộ hài hòa cố định chỉ giải thích khoảng 1% phương sai cả năm nhưng 52% riêng tháng 12. Không có một bộ hài hòa cố định nào đủ cho mọi giai đoạn.
- **Bộ dao động nhân quả (M3)** sửa được lỗi pha ở +6 h, nhưng không xử lý được thay đổi nền nhanh (đợt bão, lũ). Mô hình tường minh mô tả triều tốt, không mô tả được thủy văn.
- **Gain của M6 chủ yếu là "neo vào quan trắc hiện tại" (anchoring)** chứ không do persistence hay chia sẻ regional: M7 cho thấy riêng anchoring giảm test 49 → 34. Anchoring cũng giảm RMSE ở +3 h trên mọi tập.
- **Độ dốc nền mang dấu âm** (tương quan −0,12 đến −0,69 với thay đổi phi triều sau đó) là hiện tượng **quay về mức cũ sau các lệch ngắn hạn** của mực nước đã trừ triều, chứ không phải nhiễu hay pha triều.
- **Triều thích nghi theo biên độ không chuyển giao:** khớp tốt ở train, sụp ở validation (36 lỗi khi chọn, 49 khi hội tụ).
- **Chu kỳ triều 12,42 h được liên hệ và phát triển thành feature.** Từ phát hiện ở giai đoạn WSP ([04](04_weight_space_programming.md)), giai đoạn này đi từ phân tích đến công cụ: phân tích phổ (thô 10 phút và theo giờ, thành phần M2 và các thành phần lân cận, chu kỳ nước kém – nước cường spring–neap), rồi dựng thành các cách biểu diễn triều ngày càng tốt hơn: hồi quy hài hòa cố định (M0), hồi quy có biên độ/pha thích nghi (M2), bộ lọc Kalman theo dõi biên độ và pha theo thời gian thực (M3, M5), hệ số khuếch đại riêng cho từng mốc (M4), và chuỗi "mực nước đã trừ triều" để tính độ dốc nền (M5, M6, M7). Cùng bộ chu kỳ (M2 12,4206 h, S2 12 h, N2 12,6583 h, K1 23,9345 h, O1 25,8193 h) về sau được dùng nguyên văn trong feature triều của các trạm riêng (giai đoạn 7–9), nơi triều được ước lượng bằng hồi quy trên mực nước của chính trạm với cửa sổ 7 hoặc 30 ngày. Đây là đóng góp của giai đoạn này mang sang các giai đoạn thành công.
- **Tương quan mưa bị đánh giá quá cao:** hệ số tương quan mưa Nhâm với thay đổi +12 h là 0,41 trên toàn train nhưng chỉ 0,04 trên các mốc thật sự đủ lịch sử, vì do vài ngày đầu chuỗi chi phối.
- **Có dao động chung rất mạnh giữa Ngã Ba Sình và Dã Viên** (gợi ý dùng nước của trạm này làm đầu vào cho trạm kia); M6 thử chia sẻ nhưng không thấy lợi ích.

## 8. Kết luận

Hơn 10 họ mô hình toán cho thấy mô hình tường minh mô tả được thành phần triều nhưng không thay được GRU và không giảm được lỗi lớn; kết quả tốt trên validation không giữ trên test. Có ba thứ đáng giữ lại: (1) anchoring vào quan trắc hiện tại, (2) hiểu rõ cấu trúc triều/nền của Dã Viên và (3) feature triều 12,42 h, về sau trở thành đầu vào then chốt của các trạm riêng. Từ đây hướng sang **kết hợp** mô hình toán với GRU (giai đoạn hybrid, [06](06_hybrid.md)), trong đó triều tường minh làm nền và GRU học phần còn lại.

## Phụ lục: Đặc tả các mô hình toán học

Phần này giải thích **từng mô hình toán học đã thử ở giai đoạn này** (các phần trên): nó là gì, cấu trúc ra sao, công thức dự báo là gì. Các tên M0, M1, … M7 chỉ là **nhãn nội bộ** đặt ra trong lúc nghiên cứu theo thứ tự thử; mỗi mô hình dưới đây có thêm một tên mô tả để người đọc không cần nhớ nhãn.

Các mô hình này đều **chỉ dành cho trạm Dã Viên**, dự báo mực nước từng giờ ở một số mốc (+1, +3, +6, +12 h), và có rất ít tham số (3 đến 26). Chúng không phải mạng nơ-ron: mỗi tham số có một ý nghĩa vật lý hoặc thống kê rõ ràng, và được tối ưu bằng gradient.

### A.0 Ký hiệu và khối chung

**Ký hiệu**

| Ký hiệu | Nghĩa |
|---|---|
| t | Thời điểm phát hành dự báo (anchor) |
| h | Mốc dự báo, số giờ sau t (3, 6 hoặc 12) |
| H(u) | Mực nước trung bình giờ tại Dã Viên, trong khoảng (u − 1 h, u], đơn vị mm |
| Ĥ(t+h) | Dự báo của mô hình cho thời điểm t + h |
| e | Sai số = dự báo − thực tế |

**Mọi mô hình đều chỉ dùng lịch sử mực nước của chính Dã Viên** (96 hoặc 168 giờ gần nhất) và đồng hồ thiên văn cố định. Không dùng mưa quan trắc, dự báo thời tiết, hay trạm khác trong các cấu hình chính (trừ nhánh phụ ghi rõ bên dưới).

**Thành phần triều (dùng chung).** Mực nước được coi là tổng của một mực nền chậm và năm dao động tuần hoàn có chu kỳ cố định (các chu kỳ triều thiên văn):

| Thành phần | Chu kỳ (giờ) |
|---|---:|
| M2 | 12,4206 |
| S2 | 12,0000 |
| N2 | 12,6583 |
| K1 | 23,9345 |
| O1 | 25,8193 |

Mỗi thành phần k có tần số góc ω_k = 2π / chu kỳ_k và hai hệ số (a_k, b_k) hoặc hai trạng thái (x_k, y_k) cho biên độ và pha. Các chu kỳ **không được học**, chỉ biên độ/pha/độ khuếch đại được ước lượng. Đây là dao động hiệu dụng trong mực nước sông, không phải triều biển được đo độc lập.

**Hàm mất mát và cách huấn luyện (dùng chung mọi mô hình M).**

```
ℓ(e) = Huber(e/100) + 0,25 · softplus((|e| − 200)/25) + 0,02 · (e/200)²
```

Thành phần Huber giữ độ chính xác chung, thành phần softplus phạt các lỗi gần và vượt ngưỡng 200 mm, thành phần bình phương nhỏ giữ ổn định. Loss là trung bình ℓ(e) trên các mốc h có trọng số (thường +6 h và +12 h trọng số 1, +3 h trọng số 0,5), cộng phạt co tham số 10⁻⁴·mean(θ²).

Huấn luyện bằng AdamW toàn bộ batch (full-batch), learning rate 0,025, cắt gradient ở 10, tối đa 350 epoch (dừng sớm nếu 90 epoch không cải thiện), độ chính xác float64 trên CPU, 3 seed (17, 42, 77). **Chọn epoch chỉ theo validation:** ưu tiên số lỗi > 200 mm ở +6 và +12 h, sau đó RMSE, kèm các "rào chắn" (không được tệ hơn model chuẩn ở từng mốc, sai số lớn nhất ≤ 500–600 mm).

---

### A.1 Persistence cộng triều (M0)

**Là gì:** mô hình đơn giản nhất. Giả định mực nước tương lai bằng mực nước hiện tại cộng thêm phần thay đổi do triều.

**Công thức:**

```
Ĥ(t+h) = H(t) + T(t+h) − T(t)
T(s)   = Σ_k [ a_k · sin(ω_k s) + b_k · cos(ω_k s) ]     (k = M2, S2, N2, K1, O1)
```

**10 tham số** (a_k, b_k cho 5 thành phần), mỗi tham số bị chặn trong ±200 mm. Các hệ số này là biên độ hài hòa thực nghiệm của mực nước, không phải hằng số triều biển.

### A.2 Triều + quán tính + mưa + trạm lân cận (M1)

**Là gì:** M0 thêm ba thứ mà thủy văn gợi ý: (1) quán tính của mực nước (đang lên thì còn lên thêm một lúc), (2) phản ứng chậm với mưa, (3) ảnh hưởng của độ dốc mực nước ở hai trạm lân cận.

**Công thức:**

```
Ĥ(t+h) = H(t) + [T(t+h) − T(t)]
         + τ_v · v_t · (1 − e^(−h/τ_v))            ← quán tính
         + Σ_j β_j · S_j(t) · (e^(−h/τ_j) − 1)      ← bể chứa mưa (nhanh, chậm)
         + Σ_i γ_i · U_i(t, h)                      ← độ dốc ở Ngã Ba Sình, Thuận An
```

- v_t là vận tốc gần đây của mực nước, tính bằng trung bình có trọng số mũ của 23 hiệu số giờ liền kề trên mực nước đã trừ triều và mưa.
- S_j là hai "bể chứa mưa" (nhanh 1–8 giờ, chậm 1–24 giờ) được cập nhật bằng lượng mưa trung bình của Nhâm và Hồng Thái; mưa tương lai được giả định bằng 0.
- U_i là độ dốc mực nước 6 giờ gần nhất ở trạm lân cận, nhân với một hệ số suy giảm 12 giờ.

18 tham số khai báo. **Trong cấu hình chính, đường mưa và đường trạm lân cận bị tắt** (vì dữ liệu mưa quan trắc không có thời điểm nhận chính thức), nên chỉ còn 12 tham số có tác dụng: triều + quán tính. Phiên bản đầy đủ chỉ chạy như thí nghiệm phụ.

### A.3 Triều biên độ/pha thích nghi (M2)

**Là gì:** M1 nhưng biên độ và pha của triều **thay đổi theo trạng thái** của dòng chảy (khi nước lớn, triều bị nén hoặc lệch pha).

```
T_t(s) = g(z_t) · Σ_k [ a_k sin(ω_k s + ψ(z_t)) + b_k cos(ω_k s + ψ(z_t)) ]
g(z)   = 2 · sigmoid(c · z)        ∈ (0, 2)   — độ khuếch đại
ψ(z)   = (π/2) · tanh(d · z)       ∈ (−π/2, π/2) — độ lệch pha
```

z gồm bốn đại lượng đã chuẩn hóa: mực nước trung bình 48 giờ, biên độ dao động 25 điểm gần nhất, thay đổi 6 giờ qua và mưa 24 giờ. 26 tham số khai báo (18 có tác dụng khi tắt mưa).

---

### A.4 Bộ dao động nhân quả (M3)

**Là gì:** thay vì ước lượng hệ số triều bằng hồi quy, dùng **bộ lọc Kalman** để theo dõi liên tục mực nền chậm và biên độ/pha của năm dao động triều. Bộ lọc chạy trên 168 giờ lịch sử; đây là bước đột phá của họ model này, vì nó theo kịp biên độ triều thay đổi theo mùa (xem mục 7 ở trên).

**Trạng thái (12 thành phần):**

```
s = [ B, v, x_M2, y_M2, x_S2, y_S2, x_N2, y_N2, x_K1, y_K1, x_O1, y_O1 ]
```

B là mực nền (mm), v là vận tốc nền (mm/h), (x_k, y_k) là hai tọa độ vuông góc của dao động k (biên độ = √(x²+y²), pha = atan2(−y, x)).

**Quan sát:** H(u) = B + Σ_k x_k + nhiễu, tức mực nước đo được là nền cộng năm dao động.

**Chuyển trạng thái mỗi giờ:**
- Nền: B' = B + 24·(1 − e^(−1/24))·v, v' = e^(−1/24)·v (vận tốc nền suy giảm theo hằng số thời gian 24 giờ).
- Mỗi dao động k quay một góc ω_k: x' = x·cos ω_k + y·sin ω_k, y' = −x·sin ω_k + y·cos ω_k.

**Nhiễu cố định (không học):** phương sai nhiễu quan sát R = 9 mm², nhiễu quá trình Q = diag(1; 0,25; 0,01 × 10). Vì các ma trận hiệp phương sai không phụ thuộc dữ liệu, toàn bộ bộ lọc quy về một phép nhân ma trận cố định kích thước 12 × 168 áp lên 168 giờ lịch sử. Mỗi lần dự báo bắt đầu lại bộ lọc từ đầu cửa sổ của nó, không mang trạng thái từ lần trước.

**Công thức dự báo (bản B, bản chọn):**

```
Ĥ(t+h) = B_t + g_v · v_t · τ · (1 − e^(−h/τ))
         + Σ_k g_k · [ x_k · cos(ω_k h) + y_k · sin(ω_k h) ]
```

- τ = 1 + 71·sigmoid(θ₀) giờ, g_v = 2·sigmoid(θ₁) là hai tham số của phần trôi nền.
- g_k = 2·sigmoid(θ_k) ∈ (0, 2) là **độ khuếch đại học được cho từng dao động**; dự báo là các dao động của bộ lọc quay thêm h giờ và nhân với g_k.
- Tổng 7 tham số (bản B). **Mô hình được chọn:** cửa sổ 168 giờ, không dùng mục tiêu +1 h; τ = 3,11 giờ, g_v = 0,31, g = (0,71; 1,04; 0,99; 0,48; 0,31) cho (M2, S2, N2, K1, O1).

**Ba biến thể:** A (g_k = 1, chỉ 2 tham số), B (7 tham số như trên), C (g_k phụ thuộc vào trạng thái nền, vận tốc và biên độ gần đây; 22 tham số). Thử 18 cấu hình (độ dài cửa sổ × biến thể).

**Điểm đáng chú ý:** M3 **không neo vào quan trắc hiện tại**: tại h = 0, dự báo là B_t + Σ g_k x_k chứ không bằng H(t). Đây là điểm sẽ được cải thiện ở M6.

**Biến thể "không +1 h":** cùng mô hình M3, nhưng bỏ mục tiêu +1 h khỏi hàm loss (chỉ dùng +6 h và +12 h). Dùng để kiểm tra xem mốc +1 h có làm lệch việc học hay không; kết quả: số lỗi tổng bằng nhau, chỉ đổi chỗ 3 lỗi mỗi chiều.

### A.5 Hệ số khuếch đại theo từng mốc (M4)

**Là gì:** M3 dùng chung một bộ g_k cho mọi mốc h. M4 cho **mỗi mốc có bộ hệ số riêng**, vì dự báo +6 h và +12 h có thể cần nén biên độ khác nhau. Bộ lọc Kalman và các hằng số khác giữ nguyên.

| Biến thể | Hệ số khuếch đại | Số tham số |
|---|---|---:|
| M4-A | 5 hệ số chung cho mọi mốc (tiếp tục huấn luyện M3) | 7 |
| M4-B | 2 nhóm (bán nhật M2/S2/N2 và nhật triều K1/O1) × 2 mốc | 6 |
| M4-C | 5 hệ số × 2 mốc, độc lập | 12 |
| M4-D | như M4-C, thêm phạt λ·Σ(g_k(6) − g_k(12))² (λ = 0,001; 0,01; 0,1) | 12 |

Một chuỗi biến thể phụ thử **thay phần trôi nền** (cố định bộ hệ số triều):
- **BG-DV:** vận tốc nền suy giảm đơn giản (như M3).
- **BG-ACC:** thêm gia tốc: B' = v, v' = −v/τ_v + a, a' = −a/τ_a (nghiệm chính xác bằng hàm mũ ma trận).
- **BG-MS (đa thang thời gian):** hai vận tốc nền, nhanh và chậm, ước lượng bằng độ dốc bình phương tối thiểu của B trên 6 và 24 giờ gần nhất: drift = g·[α·s_f·τ_f·(1 − e^(−h/τ_f)) + (1 − α)·s_s·τ_s·(1 − e^(−h/τ_s))].

### A.6 Hai thang thời gian + neo + cổng thích nghi (M5)

**Là gì:** thêm ba ý tưởng vào M3: (1) **neo vào quan trắc hiện tại** (phần dư giữa mực đo và mô hình giảm dần theo thời gian), (2) **hai vận tốc nền** (nhanh, chậm) được ước lượng **vững chắc** (robust) trên chuỗi đã trừ triều, (3) một **cổng** quyết định pha trộn giữa hai thang theo mức "ngạc nhiên" của dữ liệu mới.

**Chuỗi đã trừ triều:** D_u = H_u − T(u − t | t), tức mực nước quá khứ trừ phần triều ước lượng cho 24 giờ gần nhất. Từ đó:

- Vận tốc nhanh v_f và chậm v_s = độ dốc của D trên 6 giờ và 24 giờ gần nhất, tính bằng hồi quy Huber lặp 6 vòng (ít nhạy với điểm ngoại lai).
- Phần dư neo: r = H(t) − B_t − Σ g_k x_k.
- Đổi mới trung bình (innovation) z̄: trung bình 3 giờ gần nhất của sai lệch giữa quan trắc và dự đoán một bước của bộ lọc, chuẩn hóa.

**Các biến thể:**

```
M5-A:  Ĥ = B + g_v v τ (1 − e^(−h/τ)) + T(h) + r · e^(−h/t_c)
M5-B:  Ĥ = B + T(h) + v_s·t_s·(1 − e^(−h/t_s)) + 0,5·(v_f − v_s)·t_f·(1 − e^(−h/t_f)) + r·e^(−h/t_c)
M5-C:  như M5-B nhưng thay 0,5 bằng cổng p ∈ (0,1) học được:
       p = sigmoid( b₀ + b₁·|v_f − v_s|/s_v + b₂·|z̄| + b₃·tanh((v_f − v_s)·z̄/s_v) )
M5-D:  như M5-C nhưng tăng nhiễu nền của bộ lọc theo p (một tham số κ); κ được chọn bằng 0
       nên M5-D trùng M5-C.
```

Ở đây T(h) = Σ g_k [x_k cos(ω_k h) + y_k sin(ω_k h)] và các hệ số g_k của triều được **đóng băng** từ M3. Chỉ thang thời gian (t_c, t_f, t_s) và hệ số cổng được học (3 đến 7 tham số). Dự báo trong M5 gồm +3, +6 và +12 h; mốc +3 h là yêu cầu mới ở giai đoạn này. Hai biến thể kiểm soát: M5-0R (M3 huấn luyện lại với +3/+6/+12) và M5-0R-no3 (không có +3 h) để tách việc thêm mốc +3 h ra khỏi việc chỉ huấn luyện thêm.

### A.7 Tái hiệu chuẩn bằng đơn vị thô (raw-unit)

Không phải mô hình mới. Lấy các phương trình M0–M5 và **huấn luyện lại bằng sai số bình phương trung bình (MSE) thuần trên mực nước vật lý** (mm hoặc cm) thay cho chuẩn hóa và hàm loss ℓ ở trên, bắt đầu từ các checkpoint cũ. Dùng để kiểm tra việc đổi đơn vị hoặc đổi hàm loss có thay đổi kết quả không (kết quả: gần như không).

---

### A.8 Neo vào quan trắc + trôi nền bền vững (M6)

**Là gì:** từ kết quả M3 và M5, M6 xây lại một mô hình gọn hơn: bộ lọc triều M3 đóng băng, **neo thẳng vào H(t)**, và chỉ học **hai tham số** cho phần trôi nền, gồm một "độ bền" π (phần vận tốc chậm tiếp tục tồn tại về lâu dài) và một tốc độ suy giảm ρ.

**Công thức:**

```
Ĥ(t+h) = H(t) + ΔT(h) + ΔB(h)
ΔT(h)  = Σ_k g_k · [ x_k cos(ω_k h) + y_k sin(ω_k h) ] − Σ_k g_k · x_k     ← phần thay đổi của triều
ΔB(h)  = h · π · v_s + (v_f − π · v_s) · S(h),     S(h) = Σ_{j=1..h} ρ^j    ← trôi nền
π = sigmoid(a) ∈ [0,1],   ρ = sigmoid(b) ∈ (0,1)
```

Tại h = 0 cả hai phần đều bằng 0, nên dự báo trùng đúng H(t). Vận tốc v_f, v_s là độ dốc nhanh và chậm của chuỗi đã trừ triều, tính bằng chính ước lượng robust của M5. Hệ số khuếch đại g_k đóng băng từ M3.

**Các nhánh thí nghiệm:**

| Nhánh | Khác biệt |
|---|---|
| L0 | π cố định bằng 0 (chỉ còn một tham số ρ): không có tính bền |
| L1 | π, ρ cùng học: mô hình đầy đủ |
| L2 | như L1, nhưng chuỗi đã trừ triều lấy bằng "quay ngược" từ trạng thái ở thời điểm dự báo (cách của M5) |
| R0 | thí nghiệm chia sẻ giữa các trạm: mỗi trạm học (π, ρ) riêng |
| R1 | các trạm cùng nhóm bị ràng buộc co về chung nhau (hệ số phạt λ = 0,01); các biến thể bỏ Tứ Phú, bỏ Ngã Ba Sình, hoặc gộp 5 trạm |

Khi dự báo cho một trạm, M6 chỉ cần hai tham số của trạm, bộ lọc của trạm và 168 giờ nước của chính nó; không cần dữ liệu trạm khác.

### A.9 Các biến thể kiểm tra M6 (M7)

M7 **không phải một mô hình mới**, mà là một chuỗi thí nghiệm có đối chứng để hiểu vì sao M6 tốt hơn M3 trên test. Tất cả dùng bộ lọc và hệ số triều đóng băng của M3.

| Giai đoạn | Biến thể | Nội dung |
|---|---|---|
| A (không tham số) | A0 | Ĥ = B + T(h) + drift của M3 (chính là M3) |
| | A1 | A0 cộng phần dư neo: Ĥ = H(t) + T(h) − T(0) + drift |
| | A2 | B + T(h), không drift |
| | A3 | H(t) + T(h) − T(0), neo, không drift |
| B | chẩn đoán | Không phải dự báo: so sánh độ dốc đã trừ triều với thay đổi phi triều thực tế |
| C (hệ số triều) | C0 | khuếch đại 1 (= A1) |
| | C1 | hai hệ số hằng a_S (bán nhật), a_D (nhật triều), mỗi hệ số trong (0, 3) |
| | C2 | hai hệ số phụ thuộc vào khoảng biến thiên 24 giờ gần nhất của mực nước |
| D (trôi nền trên triều C1) | D0 | không trôi nền |
| | Ddrift | drift của M3 |
| | D1 | v_f · S(h) |
| | D2 | h·π·v_s + (v_f − π·v_s)·S(h) |
| | D3 | tổ hợp tuyến tính tự do của v_s và v_f theo từng mốc |
| | D4 | như D2 nhưng cho phép π âm (−1, 1) |

---

### A.10 Hiệu chỉnh pha cho các model GRU (phase-aware)

**Là gì:** không dựng model mới, mà **sửa dự báo của ba model GRU đã có** bằng các quy tắc toán học lên quỹ đạo dự báo của chúng. Ba GRU gốc: A (có dự báo mưa ECMWF, 93.632 tham số), B (không có nhánh dự báo, 90.080 tham số), C (thêm 4 đặc trưng thời gian tuần hoàn, 90.848 tham số). Lý do: phân tích cho thấy lỗi của GRU liên quan đến chu kỳ 12,42 h bị nén hoặc lệch pha.

Có 22 phương pháp khai báo trước; các nhóm chính:

- **Quy tắc quỹ đạo:** nếu biên độ dao động của quỹ đạo dự báo 7 điểm của GRU ≥ 150 mm, cộng một hiệu chỉnh c₆ = clip(P₆ − mean(H₀, P₁…P₆), ±200 mm) vào dự báo +6 h (P là quỹ đạo dự báo của GRU); bản "một nửa cường độ" nhân hiệu chỉnh với 0,5. Đây là phương pháp được chọn cho A và B.
- **Quy tắc loại bỏ theo điểm ngoặt:** chỉ bật hiệu chỉnh khi dấu của nó khớp với dấu mà đồng hồ 12,42 h dự đoán.
- **Mô hình hài hòa thích nghi:** hồi quy bình phương tối thiểu có trọng số mũ trên 96 giờ quá khứ, hàm có nền tuyến tính cộng cặp sin/cos 12,42 h và 24 h, tính lại mỗi giờ.
- **Mô hình Kalman trạng thái:** [nền, vận tốc, hai dao động 12,42 h và 24 h], nhiễu nhỏ (q = 2 hoặc 8 mm).
- **Hiệu chỉnh lai:** chiếu quỹ đạo GRU lên cùng cơ sở hài hòa, thay phần *gia số dao động* từ 0 đến h bằng ước lượng từ lịch sử, giữ phần nền của GRU.
- **Dự báo phần dư nhân quả:** học phần sai của chính các dự báo đã trưởng thành (đã có kết quả thật) và cộng vào dự báo hiện tại (trung bình mũ, hài hòa, hoặc Kalman).
- **Hồi quy ridge nhỏ (cho GRU C):** 10 đặc trưng, mỗi mốc +6 h và +12 h một hồi quy, tổng cộng 22 hệ số, hiệu chỉnh phần dư của GRU. Đây là phương pháp được chọn cho C.

Tất cả các phương pháp được đóng băng (tập cấu hình, hash mã nguồn) **trước khi chấm trên validation mới**, chọn cho từng GRU theo kết quả trên 5 khối thời gian của tập train (có khoảng cách 288 giờ giữa các khối).

---

### Tóm tắt mối quan hệ giữa các mô hình

```
M0  persistence + triều (10)
 └─ M1 + quán tính + mưa + trạm lân cận (18)
     └─ M2 triều biên độ/pha thích nghi (26)
M3  bộ lọc Kalman dao động (7): theo dõi biên độ/pha triều → thay hồi quy
 ├─ M4 hệ số khuếch đại theo từng mốc; thay phần trôi nền
 ├─ M5 + neo + hai thang thời gian + cổng
 │    └─ raw-unit: huấn luyện lại bằng MSE thuần
 └─ M6 neo vào H(t) + trôi nền bền vững (2 tham số / trạm)
      └─ M7 các biến thể kiểm tra: neo × trôi nền, hệ số triều thích nghi, trôi nền có dấu
Hiệu chỉnh pha: sửa trực tiếp dự báo của các GRU có sẵn
```

## Nguồn


- `reports/da_vien_mathematical_model_research/` (phân tích dữ liệu, phổ triều, cơ chế, lỗi của GRU), `da_vien_mathematical_model_experiments/` (M0–M2)
- `reports/da_vien_m3_experiments/`, `da_vien_m3_ablation_no_h1/`, `da_vien_m4_experiments/`, `da_vien_m5_experiments/`
- `reports/da_vien_raw_unit_experiments/`, `da_vien_m6_preflight/`, `da_vien_m6_experiments/`, `da_vien_m7_experiments/`
- `reports/phase_aware_water_level_forecasting/` (hiệu chỉnh pha, `FINAL_REPORT.md`, `MATHEMATICAL_MODELS.md`)
