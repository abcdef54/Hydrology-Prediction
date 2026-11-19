# Giai đoạn 1: Model local ở Dã Viên

## 1. Thời gian

30/09/2026 (commit đầu tiên) đến khoảng 23/10/2026, khi bắt đầu làm model regional. Model B hoàn thiện khoảng 18–19/10; các thí nghiệm mở rộng (mưa, mốc đầu ra) kéo dài đến 23/10.

## 2. Trọng tâm

Đây là giai đoạn đầu tiên của cả quá trình nghiên cứu: dự báo mực nước **riêng trạm Dã Viên** ở năm mốc +1, +3, +6, +12, +24 h, và tích hợp model vào ứng dụng web (xem [02](02_ung_dung_web.md)).

1. **Thử model tree trước (đầu 10/2026).** XGBoost và LightGBM được thử đầu tiên, với vài bộ feature (chỉ mực nước, thêm mưa, thêm chỉ báo mùa lũ). Không tốt bằng deep learning, nên từ đó chuyển sang deep learning. Các run này đã bị xóa khi dọn dẹp nên không còn số liệu cụ thể.
2. **Deep learning.** LSTM, ResidualLSTM (học phần hiệu chỉnh so với persistence, tức giữ nguyên mực nước hiện tại), có thử backbone pretrained và attention LSTM; hai cái sau bị bỏ. ResidualLSTM 3 lớp, hidden 32, chuỗi 96 giờ trở thành kiến trúc chính.
3. **Cải tiến dần để giảm lỗi lớn:**
   - Tăng số trạm mưa đầu vào.
   - Thêm dự báo mưa ECMWF làm feature (**model B**).
   - Phạt phần lỗi quanh ngưỡng 200 mm trong loss (tail loss), chọn checkpoint theo số lỗi > 200 mm.
   - Thử 5 hoặc 24 mốc đầu ra (O5, O24).
   - Thử huấn luyện thêm dữ liệu và lấy mẫu theo sự kiện.

## 3. Dữ liệu sử dụng

- Mực nước Dã Viên theo giờ, từ 15/11/2025.
- Mưa giờ ở các trạm lân cận (Hồng Thái, Nhâm, A Lưới, Hồ A Lá, rồi Bạch Mã, Phú Bài, Bình Điền, VP QLKTCTTL), tổng mưa 3/6/12/24 giờ, chỉ báo mùa lũ, cờ thiếu dữ liệu.
- Với model B và các bản sau: dự báo mưa ECMWF tổng ở 0–3, 3–6, 6–12, 12–24 giờ tới (cả 4 trạm mưa rơi vào cùng một ô lưới, xem [00](00_du_lieu_va_quy_uoc.md)), kèm tuổi của lần chạy dự báo.
- Dataset `da_vien_1h_2026-10-15`: 6.095 train / 985 validation / 592 test sequence. Test là các mốc từ 19/09 23:00 đến 14/10 14:00. Test đã được xem nhiều lần nên là kết quả khám phá.

## 4. Kết quả

Cùng 592 mốc test × 5 horizon = 2.960 dự báo, các model Dã Viên-only:

| Model | Trạm mưa | Dự báo ECMWF | Lỗi > 200 mm | Tỷ lệ | MAE (mm) |
|---|---|---|---:|---:|---:|
| Legacy (production đầu tiên) | Hồng Thái, Nhâm, A Lưới, A Lá | không | 97 | 3,28% | 64,5 |
| **Model B** (seed 42) | Hồng Thái, Nhâm | có | **48** | 1,62% | 57,0 |
| Three-station | Hồng Thái, Bạch Mã, Phú Bài | có | 50 | 1,69% | 58,1 |
| Expanded-train, hidden 16 | như three-station | có | 32 | 1,08% | 54,0 |

Model expanded-train **không được triển khai**: nó huấn luyện thêm cả validation cũ và chọn checkpoint bằng test, nên không có tập độc lập, con số 32 lạc quan.

Trên toàn bộ dataset (7.910 mốc × 5 horizon): legacy 260 lỗi, model B 209 lỗi, ResidualLSTM 24 đầu ra 236 lỗi. Theo mốc:

| Mốc | Legacy | Model B |
|---|---:|---:|
| +1 h | 0 | 0 |
| +3 h | 5 | 1 |
| +6 h | 77 | 46 |
| +12 h | 61 | 52 |
| +24 h | 117 | 110 |

## 5. So với giai đoạn trước

Đây là giai đoạn đầu, nên không có giai đoạn trước. Mốc so sánh là model tree (kém hơn deep learning) và production legacy.

## 6. Lỗi > 200 mm

**Không xử lý hết.** Model B giảm khoảng một nửa so với legacy (97 → 48 trên test; 260 → 209 trên toàn dataset), nhưng lỗi còn lại tập trung ở +6/+12/+24 h. +1 h và +3 h gần như không còn lỗi. Hơn nữa B sửa 116 lỗi của legacy nhưng tạo thêm 65 lỗi mới.

## 7. Điều mới rút ra

- **Lỗi lớn đến theo đợt, không rải đều:** 94% lỗi của B nằm trong các chuỗi từ 2 giờ liên tiếp; ba chuỗi dài nhất chiếm 28%. Hai trường hợp điển hình là đợt mưa 12–13/09 (nước lên +371 mm trong 24 h, model dự báo chỉ +25 mm) và đợt nước rút 13–14/09.
- **Model phản ứng quá yếu hơn là sai hướng:** 70% lỗi đúng hướng nhưng biên độ thay đổi quá nhỏ, 30% sai hướng. Tăng mọi hiệu chỉnh theo một chiều sẽ không giải quyết nhóm nước rút.
- **Thêm mưa không sửa được các lỗi này:** tương quan giữa mưa vùng và thay đổi mực nước ở +12/+24 h có thể đạt khoảng 0,3 trên train nhưng gần 0 trên validation. Thay hoặc thêm trạm mưa (Bạch Mã, Phú Bài, Bình Điền…) cũng không ổn định tốt hơn model B: trong thí nghiệm 36 lượt train ở +6/+12 h (B có 21 lỗi), chỉ 1 trong 18 cấu hình có một seed thắng B. Bỏ dự báo ECMWF 12–24 h còn giảm lỗi ở 12/18 cặp so sánh.
- **Nguyên nhân gốc chưa nằm ở mưa của Dã Viên.** Điều này dẫn đến hướng tiếp theo: đưa dữ liệu nhiều trạm vào chung một model.

## 8. Kết luận

Model deep learning (ResidualLSTM) tốt hơn model tree. Thêm trạm mưa, dự báo mưa ECMWF và loss tập trung vào lỗi lớn giúp giảm khoảng một nửa số lỗi > 200 mm ở Dã Viên nhưng không loại bỏ được: model B (48 lỗi trên test) vẫn sai ở các đợt nước lên và rút nhanh ở +6/+12/+24 h. Vì chỉ dùng thông tin của riêng Dã Viên không đủ, giai đoạn tiếp theo thử train chung nhiều trạm ([03](03_regional.md)). Model B được giữ làm baseline cho các giai đoạn sau.

## Nguồn

- `README.md` (bảng production model), `model/README.md`, `src/ai/RESIDUAL_ARCHITECTURES.md`
- `reports/model_b_full_dataset_20261019_complete/`, `reports/model_b_sequence_diagnosis_20261019/`, `reports/four_models_full_dataset_20261019/`, `reports/rain_regions_h12_h24_20261019/`
- `reports/model_comparisons/` (`hong_thai_*`, `expanded_train/`, `20261023_weather_outputs_matrix/`)
- `reports/mds/` (`TAIL_LOSS.md`, `RAIN_STATION_ABLATION.md`, `OUTPUT_HORIZON_ABLATION.md`, `new_rain_station_expieriment.md`)
- Log git 30/09 – 23/10; `reports/model_selection/MLFLOW_RUN_INVENTORY.csv` (tên các run tree, không có metric)
