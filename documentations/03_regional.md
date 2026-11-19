# Giai đoạn 3: Model regional, nhiều kiến trúc

Đây là giai đoạn có nhiều thí nghiệm nhất (`reports/model_training/` có 48 thư mục, MLflow ghi hơn 90 run regional). Tài liệu chỉ tóm tắt theo các báo cáo tổng hợp.

## 1. Thời gian

Khoảng 23/10/2026 (commit ResidualLSTM regional đầu tiên) đến 4/11/2026. Phân tích 19 trạm làm trước, đến 26/10; dataset regional dựng ngày 27/10; phần lớn thí nghiệm ở 27/10–3/11. Từ khoảng 2/11, train trên máy GPU khác. Một số thí nghiệm hiệu chỉnh pha ở Dã Viên (3–4/11) nằm cuối giai đoạn.

## 2. Trọng tâm

Model B ở Dã Viên không hết lỗi > 200 mm (giai đoạn 1), nên thử một hướng khác: **train một model chung cho nhiều trạm**, hy vọng học được từ nhiều chuỗi và nhiều đợt lũ hơn. Gồm các bước:

1. **Chọn trạm.** Phân tích dữ liệu 19 trạm (796.846 bản ghi): chất lượng, độ đầy đủ, mức tương đồng. Kết luận: có thể thử Regional Model nhưng không nên gộp mọi trạm chỉ vì cùng ở Huế; các nhóm trạm không tách biệt rõ. Từ đó dựng dataset 11 trạm.
2. **Quét kiến trúc.** LSTM, GRU, CNN-GRU, TCN, Transformer; ghép dự báo mưa sớm (early) hoặc muộn (late fusion); đầu ra chung hay riêng từng trạm (linear, MLP); MLP; XGBoost.
3. **Quét loss và đầu vào.** Cân bằng theo trạm, trọng số sự kiện, phạt bất đối xứng, DILATE, **dùng thẳng số lỗi > 200 mm làm mục tiêu**; có hoặc không đặc trưng lịch (giờ, ngày); có hoặc không dự báo thời tiết; đầu ra +6/+12 h hoặc 24 mốc (O24).
4. **Thử số trạm** (5, 6, 11 trạm) và so với model local chỉ Huỳnh Liên.
5. **Chẩn đoán lỗi:** theo sự kiện lũ, báo động giả, riêng trạm Huỳnh Liên và Phú Xuân, và hiệu chỉnh pha ở Dã Viên (E1–E4).

## 3. Dữ liệu sử dụng

- Dataset `hue_regional_20261015_h6_h12_fixed_split_v1`: 11 trạm nước (Thuận An, Dã Viên, Tứ Phú, Ngã Ba Sình, Tư Hiền, Huỳnh Liên, Hòa Xuân, Phú Xuân, An Cựu, Thôn Trung Làng, Thôn Phò Nam A), chuỗi 96 giờ, +6/+12 h (hoặc 24 mốc), nhúng định danh trạm (8 chiều).
- Mưa quan trắc ở 2–3 trạm (Nhâm, Hồng Thái ở bản cũ; Hồng Thái, Bạch Mã, Phú Bài ở bản mới), dự báo mưa ECMWF, cờ mùa lũ, tùy chọn đặc trưng lịch.
- Chia train / val / test: 62.500 / 11.880 / 7.953 mẫu (xem [00](00_du_lieu_va_quy_uoc.md)). Mỗi trạm chung có 1.080 mốc validation và 723 mốc test.

## 4. Kết quả

**Các model regional chọn trên validation (11 trạm, +6/+12 h):**

| Model | MAE trung bình (mm) | Lỗi > 200 mm (val) |
|---|---:|---:|
| CNN-GRU early, shared | 35,4 (tốt nhất về MAE) | 469 |
| GRU late fusion, đầu ra riêng từng trạm | 36,5 | **380** |
| Transformer late, MLP từng trạm (32/32) | 37,0 (RMSE tốt nhất) | 400 |

Không có model thắng tuyệt đối: mỗi bên thắng ở một chỉ số, khác nhau về seed, đầu vào và ngân sách train nên không tách được tác dụng riêng của kiến trúc.

**So sánh ở Dã Viên với model B** (592 mốc test chung, +6/+12 h):

| Model | MAE (mm) | Lỗi > 200 mm |
|---|---:|---:|
| Model B (local) | 69,1 | 21 |
| Regional GRU late fusion (đầu ra riêng từng trạm) | 60,4 | 23 |
| Regional Transformer (MLP từng trạm 32/32) | — | 52 |

**Các hướng loại bỏ:**
- XGBoost regional: validation 743 lỗi so với 400 của Transformer, test 1.213 so với 741.
- Số lỗi > 200 mm làm mục tiêu duy nhất: validation 528 lỗi (so với 400), test 956 (so với 741); số đợt nước lên phát hiện được tụt từ 319 xuống 172.
- Hiệu chỉnh pha (E1–E4, 3–4/11): giảm ròng chỉ 1–2 lỗi (25 → 23), MAE và RMSE tăng.

### Dự báo 24 mốc (O24)

Ngoài +6/+12 h, giai đoạn này còn thử cho model dự báo trực tiếp **cả 24 mốc +1…+24 h trong một lần** (cùng dataset, cùng công thức train, chỉ đổi số đầu ra từ 2 lên 24; thử cả ở Dã Viên local và regional).

| Cấu hình | Lỗi > 200 mm ở +6/+12 h (validation, 11 trạm) |
|---|---:|
| GRU late fusion, 2 đầu ra (+6/+12) | 380 |
| GRU late fusion, 24 đầu ra (epoch 5) | 444 (196 + 248) |

- Dự báo 24 mốc **không giảm được lỗi lớn**: ở +6/+12 h còn hơi tệ hơn (444 so với 380). Ở Dã Viên local (giai đoạn 1) kết quả tương tự: 236 lỗi so với 209 của model B.
- Cái được là **có cả quỹ đạo dự báo**: so sánh các lần dự báo liên tiếp của cùng một thời điểm đích (revision). Revision dương có tương quan với mực nước thực sự dâng giữa hai lần phát, nhưng độ chính xác phát hiện nước dâng > 200 mm thấp và recall rất thấp, nên không dùng làm cảnh báo sớm.
- Dự báo cả quỹ đạo về sau được dùng lại ở Dã Viên (+1…+6 h, giai đoạn 9) và là model nền của WSP.
- Có chuẩn bị loss DILATE (căn quỹ đạo 24 mốc theo hình dạng và thời gian) nhưng không thấy báo cáo kết quả.

## 5. So với giai đoạn trước

Regional **chính xác hơn model B ở độ lớn trung bình** tại Dã Viên (MAE 60 so với 69 mm) nhưng **không giảm được lỗi lớn** (23 so với 21). Về số lỗi > 200 mm, kết quả không tốt hơn giai đoạn 1.

## 6. Lỗi > 200 mm

Không xử lý được. Trên toàn vùng vẫn còn 380–470 lỗi trong 23.760 dự báo validation. Ở +12 h, cả model B lẫn các model regional đều không phát hiện được đợt nước lên > 200 mm nào ở Dã Viên (0/11).

## 7. Điều mới rút ra

- **GRU là kiến trúc ổn định nhất về lỗi > 200 mm.** GRU late fusion có ít lỗi lớn nhất trên validation (380, so với 400 của Transformer và 469 của CNN-GRU) và ở Dã Viên (23 lỗi, so với 52 của Transformer). Các kiến trúc khác (Transformer, CNN-GRU, TCN) có thể thắng ở MAE hoặc RMSE nhưng không giảm được lỗi lớn. Model lớn hơn hoặc phức tạp hơn không thắng. GRU được chọn và dùng cho mọi giai đoạn sau (WSP, Thuận An, Ngã Ba Sình, Dã Viên).
- **Cái giúp được (nhỏ):** đầu ra riêng từng trạm (RMSE +12 h từ 111 xuống 103 mm, báo động giả giảm). **Cái làm hại:** đặc trưng lịch giờ/ngày (MAE validation 37,0 → 39,1 mm).
- **Trạm Huỳnh Liên chi phối bài toán.** Ở +12 h, số lần mực nước thay đổi > 1.000 mm: 55 trên 6.073 mẫu train nhưng 73 trên 723 mẫu test. Model nhìn thấy quá ít đợt lớn lúc train; mọi model đều dự báo thiếu biên độ.
- **Hàm loss bất cân xứng giữa các trạm:** sai số cùng 1 mm ở Dã Viên đóng góp vào loss khoảng 6,3 lần so với Huỳnh Liên (do mẫu số `max(scale, 200 mm)`). Đã đo được xung đột gradient giữa trạm trong encoder dùng chung (cosine âm), nhưng chưa chứng minh được là nguyên nhân.
- **Giới hạn về thông tin:** nút thắt có thể là đợt lũ hiếm và thiếu dữ liệu thượng nguồn, hơn là kiến trúc hay loss. Đây là giả thuyết, chưa được chứng minh.
- **Giới hạn đánh giá:** tập test đã được xem quá nhiều lần nên không còn là tập độc lập; báo cáo tổng hợp ngày 31/10 nhấn mạnh cần giai đoạn dữ liệu mới.
- Cập nhật dự báo theo giờ cho MAE 10,5 mm so với 46,1 mm của dự báo mỗi ngày một lần, nhưng chỉ vì mọi dự báo cập nhật đều có lead +1 h; đây không phải cải thiện dự báo xa.

## 8. Kết luận

Train chung nhiều trạm giúp MAE nhưng không giúp lỗi > 200 mm ở Dã Viên; đổi kiến trúc không giải quyết được vấn đề; điểm nghẽn là các đợt lũ lớn hiếm và thông tin đầu vào. Tuy vậy giai đoạn này chọn ra được **GRU** là kiến trúc ổn định nhất về lỗi lớn và dùng nó cho đến nay. Các model regional GRU, Transformer và event-aware được giữ làm nền cho giai đoạn tiếp theo; model GRU regional 24 mốc (`regional_o24_gru_epoch005`, 93.632 tham số) là đối tượng của giai đoạn weight-space programming ([04](04_weight_space_programming.md)). Hướng "đổi kiến trúc" dừng ở đây.

## Nguồn

- `reports/model_training/regional_experiment_comprehensive_review_20261031.md` (tổng hợp, có bảng run ID), `large_error_primary_experiment_20261031.md`, `regional_xgboost_20261102/`, `regional_mlp_experiments_20261031/`, `station_subset_*`, `huynh_lien_error_investigation_20261030/`, `hydrology_event_forensics_20261102/`, `event_checkpoint_analysis_20261102/`, `o24_daily_rolling_revision_20261103/`, `da_vien_e1_phase_correction_20261104/`, `da_vien_e3_*`, `da_vien_e4_*`
- `reports/model_comparisons/regional_shortlist_vs_local_da_vien_20261030/`, `regional_vs_local_da_vien_20261030/`, `reports/model_training/da_vien_model_b_vs_regional_final/`
- `reports/station_analysis/report.md`, `reports/model_selection/EXPERIMENT_HISTORY.md`
- `research/REGIONAL_O24.md`, `research/CHANGE_DYNAMICS_EXPERIMENTS.md`, `TRAINING_REGIONAL.md`
