# Ghi chú chuyển session: Ngã Ba Sình (xong, 16/11/2026)

Đọc kèm: `models/nga_ba_sinh/REPRODUCE.md` (chi tiết đầy đủ), `models/PIPELINE.md` (quy trình chung), `models/NOTES_thuan_an_to_nga_ba_sinh.md` (session trước).

## 1. Kết quả

**Model `models/nga_ba_sinh/combined_v1/`**: ghép theo mốc dự báo, không lấy trung bình.
- **+3 h và +6 h** từ `short_3h_6h`: dataset `nbs_onshore_tide7_fc_sea7_v3` (39 đầu vào), GRU 64×3, cửa sổ 12 h, lịch `step`, seed 23, **epoch 59**, 74.955 tham số.
- **+12 h** từ `long_12h`: dataset `nbs_onshore_tide7_fc_sea_v3` (35 đầu vào), GRU 128×3, cửa sổ 25 h, `--lstm-horizon-weights 1 1 2.5`, 130 epoch, seed 23, **epoch 110**, 279.435 tham số.
- **0 lỗi > 200 mm** trên val, test và giai đoạn mới 15/10–12/11.
- **01/09–31/10/2026 (1.451 giờ):** MAE **38,0 / 55,7 / 52,0 mm** (+3 / +6 / +12 h); persistence 90 / 131 / 72.
- **Lưu ý:**
  - Model được chọn sau khi xem cả ba giai đoạn, nên cần kiểm tra lại trên dữ liệu sau 12/11 bằng `evaluate combined` (REPRODUCE mục 4.6).
  - Biên độ ở +12 h mỏng (lỗi lớn nhất 197,6 mm).
  - Chỉ seed 23 đạt được kết quả này; các seed khác có 3–7 lỗi.

## 2. Những gì quyết định kết quả (theo thứ tự)
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

## 3. Code mới (chưa commit; `src/ai/station_dataset/` và `api_backfill.py` đều chưa được git theo dõi)
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

## 4. Dữ liệu
- **Open-Meteo** (`data/external/open_meteo/`):
  - `prev_offshore.json` và `prev_offshore_to_20261112.json` (gió);
  - `ecmwf_single_runs/` (dùng cho train) và `ecmwf_single_runs_soil/` (đến 12/11, có thêm độ ẩm đất);
  - `marine_offshore_ta.json` (1 điểm) và `marine_multi_points.json` (8 điểm, có sóng và dòng chảy).
- **Dataset:** `train_data/nga_ba_sinh/` chỉ còn `nbs_onshore_tide7_fc_sea7_v3` và `nbs_onshore_tide7_fc_sea_v3`. Config: `models/nga_ba_sinh/dataset_config_onshore_tide7_{sea7,sea}.json`; config đánh giá mở rộng đến 12/11: `models/nga_ba_sinh/eval_config_ext_20261112_{sea7,sea}.json`.
- **Đã dọn ngày 16/11/2026:** xóa mọi config, dataset, run thử nghiệm và model đóng gói không thuộc `combined_v1` (khoảng 4,5 GB). `model/experiments/nga_ba_sinh/` chỉ còn hai run gốc của `combined_v1`.
- **Dataset đánh giá mở rộng** nằm trong scratchpad, **bị xóa khi sang session mới**. Dựng lại theo REPRODUCE mục 4.6, nhớ dùng `--normalizers-from`.
- **Biểu đồ gai số đo:** `figures/nga_ba_sinh/sensor_spikes*.png`.

## 5. Quy tắc làm việc (đã lưu trong bộ nhớ Claude)
- **Người dùng tự chạy train**; Claude chỉ đưa lệnh. Tối đa 2 run cùng lúc, chạy với `nice`.
- Xếp hạng model theo **số lỗi > 200 mm trước, MAE sau**. Chấp nhận MAE tăng để giảm lỗi lớn.
- Mỗi lần chỉ thêm **một nhóm đầu vào mới**. Sàng lọc bằng hồi quy tuyến tính trước khi train.
- Luôn chấm **từng epoch**, không chỉ epoch mà trainer chọn.
- Mục tiêu của người dùng là **0 lỗi trên toàn bộ thời gian**: val, test và dữ liệu mới.
- Chấm điểm nên chạy trên GPU (`SequenceRuntime(..., device='cuda')`).

## 6. Việc tiếp theo
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
4. **Commit** `src/ai/station_dataset/`, `src/backend/ingestion/api_backfill.py`, `models/`, `research/scripts/score_all_epochs.py`.
5. Repo đã được dọn từ 35 GB xuống 15 GB. Muốn ổ D: (file `D:\WSL\Ubuntu\ext4.vhdx`) thực sự nhẹ đi thì cần co file đó lại: `wsl --shutdown`, rồi `diskpart` với `select vdisk file=...`, `attach vdisk readonly`, `compact vdisk`.
