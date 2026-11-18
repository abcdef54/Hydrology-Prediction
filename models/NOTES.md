# Ghi chú các trạm (Thuận An, Ngã Ba Sình)

Cập nhật 16/11/2026. Quy trình chung: `models/PIPELINE.md`. Chi tiết tái tạo: `models/thuan_an/REPRODUCE.md` và `models/nga_ba_sinh/REPRODUCE.md`.

Phần A: Thuận An và các phân tích chung (ghi 12/11/2026). Phần B: Ngã Ba Sình (xong 16/11/2026, kèm các việc tiếp theo).

---

# Phần A. Thuận An và phân tích chung

## A1. Trạm Thuận An (0074647402): ĐÃ XONG

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

### Code và dữ liệu đã chuẩn bị
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

## A2. Chiều di chuyển của nước (đo bằng dữ liệu 10 phút, 11/2025–11/2026)

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

## A3. Các phân tích cho Ngã Ba Sình và Dã Viên

Hồi quy tuyến tính, fit trên train. Kết quả ghi theo val / test.

- **Trạm sông khó hơn Thuận An nhiều.** Chỉ với lịch sử mực nước và triều 30 ngày của chính trạm, R² ở +12 h: Ngã Ba Sình 39% / 42%, Dã Viên 35% / 24%, so với Thuận An 65% / 69%. Số lỗi >200 mm ở +6 h: Ngã Ba Sình 18 / 23, Dã Viên 26 / 33, trong khi Thuận An là 0.
- **Mực nước quá khứ của trạm lân cận** (t, −3, −6, −12 h): **không giúp**. Ở +12 h còn kém đi, ví dụ Ngã Ba Sình khi thêm Dã Viên: R² test giảm từ 42% xuống 29%. Lý do: hai trạm dao động gần như cùng lúc, và quan hệ giữa chúng thay đổi theo mùa.
- **Dự báo của model Thuận An** (+3/+6/+12 h) làm đầu vào:
  - +6 h: số lỗi >200 mm trên test giảm từ 33 xuống 14 (Dã Viên) và từ 23 xuống 14 (Ngã Ba Sình).
  - +12 h: Dã Viên R² test tăng từ 24% lên 36%; Ngã Ba Sình gần như không đổi.
- **Vì sao dự báo kém xa "tương lai thật":** model Thuận An đạt R² 0,93 là nhờ phần triều. Trên **phần không phải triều** (nước dâng, lũ, mực nước nền), R² chỉ −0,06 ở +6 h và 0,10 ở +12 h. Mà đây lại chính là phần trạm sông cần: tương quan với phần thật là 0,45, với phần dự báo chỉ 0,17.

## A4. Các ý tưởng đã ghi nhận

| Ý tưởng | Kết luận |
|---|---|
| **Chuỗi model Thuận An → Ngã Ba Sình → Dã Viên** (dự báo của trạm trước làm đầu vào cho trạm sau) | Có ích ở +6 h. Chỉ làm **một bước**. Dự báo dùng cho giai đoạn train phải tạo bằng cách train chéo theo từng đoạn thời gian (out-of-fold), nếu không model hạ nguồn sẽ quá tin vào đầu vào này. |
| **Vòng lặp train lại qua lại** (Ngã Ba Sình ↔ Dã Viên, rồi Dã Viên → Thuận An…) đến khi cả hai tốt nhất | **Không khuyến nghị.** Mô phỏng: chỉ vòng 1 có thay đổi (Dã Viên +3 h R² val 54% → 61%). Từ vòng 2 kết quả đứng yên vì không có thông tin mới (một dự báo chỉ là kết quả tính từ đầu vào của model tạo ra nó). Ở +12 h còn kém đi, và kết quả hội tụ về một model gộp. Rủi ro: quá tin vào dự báo trên dữ liệu train, và chọn nhiều lần trên cùng tập validation. |
| **GAN** | Khác bản chất với vòng lặp trên: GAN có dữ liệu thật mới ở mỗi bước, và vòng lặp của nó tạo ra hàm loss chứ không tạo thông tin. Có thể dùng **loss đối kháng** để dự báo đỉnh lũ sắc nét hơn (giống DGMR của DeepMind). Nhưng nó không giải quyết được đợt kiểu 23/10, dễ báo động giả hơn và khó train. Để thử sau. |
| Dùng Dã Viên cho Thuận An +12 h khi có lũ sông lớn | Đáng thử: đỉnh lũ ở Dã Viên đến trước Thuận An 0,5–1,5 ngày. |
| Dữ liệu xả hồ SOHOA cho trạm sông | Đáng thử: dữ liệu đã có trong database (bảng facility_operations). |
| Dự báo mực nước biển ngoài khơi (Copernicus Marine) cho Thuận An | Đáng thử cho các đợt nước dâng kiểu 23/10. Cần kiểm tra bản lưu trữ có phải đúng bản phát hành tại thời điểm đó. |

---

# Phần B. Ngã Ba Sình

## B1. Kết quả

**Model `models/nga_ba_sinh/combined_v1/`**: ghép theo mốc dự báo, không lấy trung bình.
- **+3 h và +6 h** từ `short_3h_6h`: dataset `nbs_onshore_tide7_fc_sea7_v3` (39 đầu vào), GRU 64×3, cửa sổ 12 h, lịch `step`, seed 23, **epoch 59**, 74.955 tham số.
- **+12 h** từ `long_12h`: dataset `nbs_onshore_tide7_fc_sea_v3` (35 đầu vào), GRU 128×3, cửa sổ 25 h, `--lstm-horizon-weights 1 1 2.5`, 130 epoch, seed 23, **epoch 110**, 279.435 tham số.
- **0 lỗi > 200 mm** trên val, test và giai đoạn mới 15/10–12/11.
- **01/09–31/10/2026 (1.451 giờ):** MAE **38,0 / 55,7 / 52,0 mm** (+3 / +6 / +12 h); persistence 90 / 131 / 72.
- **Lưu ý:**
  - Model được chọn sau khi xem cả ba giai đoạn, nên cần kiểm tra lại trên dữ liệu sau 12/11 bằng `evaluate combined` (REPRODUCE mục 4.6).
  - Biên độ ở +12 h mỏng (lỗi lớn nhất 197,6 mm).
  - Chỉ seed 23 đạt được kết quả này; các seed khác có 3–7 lỗi.

## B2. Những gì quyết định kết quả (theo thứ tự)
1. **Mưa dự báo ECMWF** (17 kênh): giải quyết phần lớn lỗi +12 h và lỗi đợt mưa 07/10.
2. **Lỗi dữ liệu:**
   - sensor đổi mốc đo lúc 2026-10-10 15:20 UTC: loại 13 mẫu bằng `sensor_steps`;
   - lọc gai số đo bằng `despike` (trung vị trượt 121 phút, ngưỡng 150 mm; giữ ca 11/04).
   
   Mọi lỗi test cũ đều là lỗi dữ liệu.
3. **Mực nước biển ngoài khơi dự báo** (Open-Meteo Marine `sea_level_height_msl_previous_day1`): bước tiến lớn nhất trên dữ liệu mới (19 → 3–5 lỗi).
4. **Model h128 với trọng số loss +12 h gấp 2–2,5:** đưa +12 h về sát 0.
5. **Ghép theo mốc:** +3/+6 h và +12 h đánh đổi nhau khi dùng chung một GRU, nên tách mỗi mốc ra một model.

**Đã thử và loại:** mưa đo, xả hồ SOHOA, dự báo Thuận An làm đầu vào, dự báo tuyến tính làm đầu vào, dropout nhóm, event-weight (chỉ dời lỗi giữa dự báo thiếu và thừa), độ ẩm đất, sóng, dòng chảy, gió dự báo mới hơn, áp suất, mưa dự báo xa hơn 24 h, model chỉ +12 h (test luôn 3–9 lỗi), ensemble trung bình (các model sai cùng chiều).

**Khác Thuận An:**
- triều 7 ngày tốt hơn 30 ngày;
- chỉ dùng 4 kênh gió vào bờ;
- lịch `step` tốt hơn cosine;
- seq12 hoặc seq25;
- under-weight vẫn là 0,3.

## B3. Code mới (đã commit ngày 17/11/2026)
- `build.py`:
  - nhóm mới `rain_obs`, `lin_fc`, `soil`, `sea`, `sea_multi`, `wave`;
  - khóa config `wind.channels`, `wind.onshore_pos`, `fc_ecmwf.channels`, `sea.direct`, `sensor_steps`, `despike`;
  - xuất `despiked_readings.csv`.
- `features.py`: hàm cho từng nhóm trên (`sea_level_features`, `sea_multi_features`, `wave_features`, `soil_moisture_features`, `linear_forecast_features`, `rain_cluster_features`…).
- `open_meteo.py`:
  - lệnh `marine`;
  - `single-runs --hourly/--pause`;
  - tự chờ và thử lại khi gặp HTTP 429 (gói miễn phí tính mỗi điểm trong request là một lượt: khoảng 5.000 lượt/giờ, 10.000 lượt/ngày);
  - `load_single_runs_vars`.
- `evaluate.py`: lệnh `combined` (ghép theo mốc, chấm trên một khoảng thời gian).
- `research/scripts/score_all_epochs.py`: chấm mọi epoch của một run trên val, test và giai đoạn mới; chạy trên GPU.
- Tài liệu: `models/PIPELINE.md` (mục 3.1b đổi mốc đo, 3.1c gai, danh sách khóa config, `evaluate combined`), `models/nga_ba_sinh/REPRODUCE.md`.

## B4. Dữ liệu
- **Open-Meteo** (`data/external/open_meteo/`):
  - `prev_offshore.json` và `prev_offshore_to_20261112.json` (gió);
  - `ecmwf_single_runs/` (dùng cho train) và `ecmwf_single_runs_soil/` (đến 12/11, có thêm độ ẩm đất);
  - `marine_offshore_ta.json` (1 điểm) và `marine_multi_points.json` (8 điểm, có sóng và dòng chảy).
- **Dataset:** `train_data/nga_ba_sinh/` chỉ còn `nbs_onshore_tide7_fc_sea7_v3` và `nbs_onshore_tide7_fc_sea_v3`. Config: `models/nga_ba_sinh/dataset_config_onshore_tide7_{sea7,sea}.json`; config đánh giá mở rộng đến 12/11: `models/nga_ba_sinh/eval_config_ext_20261112_{sea7,sea}.json`.
- **Đã dọn ngày 16/11/2026:** xóa mọi config, dataset, run thử nghiệm và model đóng gói không thuộc `combined_v1` (khoảng 4,5 GB). `model/experiments/nga_ba_sinh/` chỉ còn hai run gốc của `combined_v1`.
- **Dataset đánh giá mở rộng** nằm trong scratchpad, **bị xóa khi sang session mới**. Dựng lại theo REPRODUCE mục 4.6, nhớ dùng `--normalizers-from`.
- **Biểu đồ gai số đo:** `figures/nga_ba_sinh/sensor_spikes*.png`.

## B5. Quy tắc làm việc (đã lưu trong bộ nhớ Claude)
- **Người dùng tự chạy train**; Claude chỉ đưa lệnh. Tối đa 2 run cùng lúc, chạy với `nice`.
- Xếp hạng model theo **số lỗi > 200 mm trước, MAE sau**. Chấp nhận MAE tăng để giảm lỗi lớn.
- Mỗi lần chỉ thêm **một nhóm đầu vào mới**. Sàng lọc bằng hồi quy tuyến tính trước khi train.
- Luôn chấm **từng epoch**, không chỉ epoch mà trainer chọn.
- Mục tiêu của người dùng là **0 lỗi trên toàn bộ thời gian**: val, test và dữ liệu mới.
- Chấm điểm nên chạy trên GPU (`SequenceRuntime(..., device='cuda')`).

## B5b. Phát hiện thêm ngày 17/11/2026 (chi tiết trong `models/da_vien/notes.md`)
- **Model NBS không có station embedding** (`--head-mode shared --station-embedding-size 0`), train để finetune sang Dã Viên.
  - Công thức tốt nhất: `nbs_onshore_tide7_fc_sea_v3`, GRU 128×3, cửa sổ 25 h, trọng số mốc 1/1,5/2,5, under 0,3, step, seed 42, 120 epoch.
  - Tốt nhất có 1 lỗi sát ngưỡng trên mọi giờ.
  - Ghép theo mốc từ hai epoch của cùng run (+3/+6 h từ ep37, +12 h từ ep69) cho 0 lỗi.
- **Phải chấm cả 48 mẫu `purged`** (giờ ranh giới giữa các tập). `research/scripts/score_all_epochs.py` bỏ qua chúng; mẫu 14/09 21:00 +6 h là chỗ hay sai.
- **Bập bênh +6 h / +12 h** trong một model là lý do `combined_v1` (ghép theo mốc) đạt 0 lỗi mà model đơn thì không. Các cách sửa bằng loss hay kiến trúc đều không thành công.
- Phải train qua MLflow server (experiment `nga_ba_sinh` có nơi lưu artifact là `mlflow-artifacts:/7`).

## B5c. Đóng gói triển khai (18/11/2026)
- Folder bàn giao cho hệ thống chung: `deploy/thuan_an/`, `deploy/nga_ba_sinh/`. Mỗi folder gồm bundle joblib (dict và mảng numpy, không phụ thuộc repo), dataset train, dữ liệu Open-Meteo hiện có, `predict.py` và `worker.py` tự chứa, cùng `README.md`.
- Dựng lại: `PYTHONPATH=. .venv/bin/python research/scripts/make_deploy_folders.py` (bản gốc của script nằm ở `src/ai/station_dataset/deploy_templates/`). `src/ai/station_dataset/deploy.py` là bản dùng trong repo.
- Đã kiểm tra: `predict.py` chạy độc lập cho MAE T9–T10 đúng như đã báo cáo (Ngã Ba Sình 38,0 / 55,7 / 52,0; Thuận An 30,5 / 40,1 / 44,2), chênh với dự báo đã chấm ≤ 0,08 mm.

## B5d. Dọn dẹp và tạm dừng (18/11/2026)
- Đã xóa mọi checkpoint từng epoch (`model/experiments/`, kể cả hai run gốc của `combined_v1`), MLflow (`mlflow_local.db`, `mlartifacts/`) và các file thô lớn trong `reports/`.
- Epoch đã chọn của mọi model vẫn còn trong `models/thuan_an/`, `models/nga_ba_sinh/combined_v1/`, `models/da_vien/`. Các đường dẫn `run_dir` trong `combined.json`, `REPRODUCE.md`, `selection.json` chỉ còn là ghi chép nguồn gốc. Muốn tái tạo thì train lại theo lệnh trong `REPRODUCE.md`; MLflow sẽ tạo lại database mới.
- Nghiên cứu Dã Viên tạm dừng: tóm tắt ở đầu `models/da_vien/notes.md`.

## B6. Việc tiếp theo
1. **Kiểm tra `combined_v1` trên dữ liệu sau 12/11:**
   - đồng bộ database (`python -m src.backend.ingestion --once`);
   - tải thêm Open-Meteo (gió, ECMWF, Marine);
   - dựng dataset đánh giá với `grid_end_utc` mới và `--normalizers-from`;
   - chạy `evaluate combined`.
2. **Báo đơn vị quản lý trạm** về lần đổi mốc đo ngày 10/10/2026.
3. **Trạm tiếp theo theo kế hoạch: Dã Viên.** Áp dụng các bài học từ Ngã Ba Sình:
   - lọc gai; quét bước nhảy mốc đo;
   - mực nước biển dự báo;
   - mưa dự báo ECMWF;
   - trọng số theo mốc;
   - ghép theo mốc.
   
   Thuận An có thể cần lọc gai: có một gai 2 số đo ±351 mm ngày 18/08, đúng trong giai đoạn model Thuận An từng sai ở +6 h trên val.
5. Repo đã được dọn từ 35 GB xuống 15 GB. Muốn ổ D: (file `D:\WSL\Ubuntu\ext4.vhdx`) thực sự nhẹ đi thì cần co file đó lại: `wsl --shutdown`, rồi `diskpart` với `select vdisk file=...`, `attach vdisk readonly`, `compact vdisk`.
