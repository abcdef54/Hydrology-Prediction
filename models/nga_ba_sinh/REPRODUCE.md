# Trạm Ngã Ba Sình (0074647705): model dự báo mực nước `combined_v1`

Cập nhật ngày 16/11/2026. Tài liệu này mô tả model đã chọn cho trạm Ngã Ba Sình: nó gồm những gì, hiệu năng ra sao, và cách tái tạo từ đầu (dữ liệu, dataset, train, chọn epoch, chấm điểm). Các phần dùng chung cho mọi trạm (API, Open-Meteo, quy tắc dựng dataset, kiến trúc, loss, vòng lặp train) được giải thích chi tiết trong `models/PIPELINE.md`; tài liệu này chỉ nhắc lại những gì riêng của Ngã Ba Sình.

---

## 0. Tóm tắt

- **Model:** `models/nga_ba_sinh/combined_v1/`, **ghép theo mốc dự báo** từ 2 model GRU, không lấy trung bình:
  - **+3 h và +6 h** lấy từ `short_3h_6h`: GRU 64×3, cửa sổ 12 h, 39 đầu vào, 74.955 tham số.
  - **+12 h** lấy từ `long_12h`: GRU 128×3, cửa sổ 25 h, 35 đầu vào, 279.435 tham số, train với trọng số loss của mốc +12 h gấp 2,5.
  - Tổng cộng **354.390 tham số**.
- **Hiệu năng từ 01/09 đến 31/10/2026**, mọi giờ (1.451 thời điểm phát dự báo):

| Mốc | **MAE (mm)** | RMSE (mm) | Sai số lớn nhất (mm) | Số lỗi > 200 mm |
|---|---|---|---|---|
| +3 h | **38,0** | 46,6 | 152,7 | **0** |
| +6 h | **55,7** | 68,8 | 187,1 | **0** |
| +12 h | **52,0** | 64,9 | 197,6 | **0** |

- **Lưu ý quan trọng:** hai thành phần được chọn sau khi xem cả val, test và giai đoạn 15/10–12/11. Phải kiểm tra lại trên dữ liệu sau 12/11/2026 (mục 8).

---

## 1. Thông tin trạm và dữ liệu

| | |
|---|---|
| Mã trạm | `0074647705`, Trạm Ngã Ba Sình (sông Hương), collection `water_level_depth`, đơn vị mm |
| Tọa độ | 16,528167N 107,574000E, nơi sông Hương và sông Bồ hợp lưu, gần phá Tam Giang |
| Dữ liệu quan trắc | Từ 14/11/2025 18:50 UTC, nhịp 10 phút |
| Lưới giờ của dataset train | 2025-11-14T17:00Z → 2026-10-15T07:00Z (8.031 giờ) |
| Mốc dự báo | +3, +6, +12 h, tính từ thời điểm phát dự báo t |
| Chia tập (giống Thuận An) | train: t < 2026-07-29T17:00Z (6.073 mẫu); val: 2026-07-30T17:00Z → 2026-09-13T16:00Z (1.080 mẫu); test: t ≥ 2026-09-14T17:00Z (710 mẫu); 48 mẫu bị loại ở ranh giới (24 h trước mỗi ranh giới); 13 mẫu bị loại do sensor đổi mốc (mục 3.3) |
| Giai đoạn đánh giá thêm | 15/10–12/11/2026 (668 mẫu), dựng bằng lưới mở rộng (mục 4.6) |

**Đặc điểm thủy văn:**
- Triều lan từ biển vào: Thuận An → (khoảng 40 phút) → Ngã Ba Sình → (khoảng 40 phút) → Dã Viên.
- Trong lũ lớn, Ngã Ba Sình và Dã Viên lên gần như cùng lúc.
- Biên độ triều (độ lệch chuẩn) khoảng 62 mm. Dao động 12 h chủ yếu do triều nhật, nước dâng và lũ.

---

## 2. Model `combined_v1`

### 2.1 Ghép theo mốc

```
                       ┌─ short_3h_6h (GRU 64×3, cửa sổ 12 h, 39 đầu vào) ─► dự báo +3 h, +6 h  ─┐
đầu vào tại thời điểm t ┤                                                                        ├─► combined_v1
                       └─ long_12h   (GRU 128×3, cửa sổ 25 h, 35 đầu vào) ─► dự báo +12 h        ─┘
```

- Cả hai model đều được train để dự báo đủ 3 mốc (+3, +6, +12 h), nhưng mỗi mốc chỉ **dùng đầu ra của một model**. Đầu ra +12 h của `short_3h_6h`, và +3/+6 h của `long_12h`, bị bỏ.
- Lý do ghép như vậy: trong mọi thử nghiệm, mỗi khi dồn sức cho +12 h thì +3/+6 h kém đi, và ngược lại, vì ba mốc dùng chung một GRU (Phụ lục A). Hai model này mỗi cái đạt 0 lỗi ở mốc của mình trên cả ba giai đoạn, nên ghép theo mốc giữ được cả hai.
- Quy tắc ghép, đường dẫn và mọi tham số train nằm trong `models/nga_ba_sinh/combined_v1/combined.json`.

### 2.2 Kiến trúc chung: `ResidualGRU` (`src/ai/residual_sequence.py`)

Đầu vào x có kích thước [1, L, F] (L giờ gần nhất, F kênh đã chuẩn hóa):
1. `Dropout1d(0,1)` trên đầu vào: lúc train, mỗi mẫu bỏ ngẫu nhiên nguyên một kênh.
2. `GRU(F → H, 3 lớp, dropout 0,2 giữa các lớp)`. Lấy trạng thái ẩn ở bước cuối (t).
3. Ghép với embedding của trạm (8 chiều), đi qua `Linear(H + 8 → H)` + ReLU, rồi `Dropout(0,1)`.
4. Head riêng cho trạm, khởi tạo bằng 0: Δ_h = W·z + b, mỗi mốc h một đầu ra.
5. **Dự báo dạng residual:** ŷ_h = persistence + Δ_h, trong không gian nhãn đã chuẩn hóa. Persistence là "giữ nguyên mực nước hiện tại". Khi bắt đầu train, model đúng bằng persistence, rồi học dần phần thay đổi.
6. Đổi về mm bằng bộ chuẩn hóa của train.

### 2.3 Thông số từng thành phần

| | `short_3h_6h` | `long_12h` |
|---|---|---|
| Dùng cho mốc | +3 h, +6 h | +12 h |
| Dataset | `nbs_onshore_tide7_fc_sea7_v3` | `nbs_onshore_tide7_fc_sea_v3` |
| Số kênh đầu vào F | 39 | 35 |
| Cửa sổ lịch sử L | 12 h | 25 h |
| Kích thước ẩn H × số lớp GRU | 64 × 3 | 128 × 3 |
| Dropout (GRU / đầu vào / head) | 0,2 / 0,1 / 0,1 | 0,2 / 0,1 / 0,1 |
| **Số tham số** | **74.955** | **279.435** |
| Trọng số loss theo mốc (+3 / +6 / +12 h) | 1 / 1 / 1 | **1 / 1 / 2,5** |
| Loss | `combined`, under-weight 0,3, event-weight 0 | `combined`, under-weight 0,3, event-weight 0 |
| Optimizer | AdamW, lr 0,0001, weight decay 0,001, cắt gradient L2 = 1 | như bên trái |
| Lịch learning rate | `step` (nhân 0,9 sau mỗi 10 epoch) | `step` |
| Batch, số epoch, early stopping | 64, 100, 100 | 64, 130, 200 |
| Seed | 23 | 23 |
| **Epoch được chọn** | **59** | **110** |
| sha256 `model.pt` | `2bff6ff3c964c32d58b844fd98df7ce7089e70a93fa48f316e2d0fcbbeea9d29` | `9ded77daea1a3d2d496faf95bb3acb8ae00142ea4409c17d922956a3653f48eb` |

**Chi tiết số tham số.** Công thức: 3·(F·H + H·H + 2H) cho lớp GRU đầu, cộng 3·(2H² + 2H) cho mỗi lớp GRU sau, cộng 8 (embedding), (H + 8)·H + H (tầng tổng hợp) và 3H + 3 (head).

| Phần | `short_3h_6h` (F = 39, H = 64) | `long_12h` (F = 35, H = 128) |
|---|---|---|
| GRU lớp 1 | 20.160 | 63.360 |
| GRU lớp 2 + 3 | 49.920 | 198.144 |
| Embedding trạm | 8 | 8 |
| Tầng tổng hợp `Linear(H + 8 → H)` | 4.672 | 17.536 |
| Head (3 mốc) | 195 | 387 |
| **Tổng** | **74.955** | **279.435** |

**Loss `combined`** (`src/ai/change_objective.py`, xem `PIPELINE.md` mục 5). Gọi e là sai số (mm), c là mức thay đổi thật so với persistence (mm), d = max(σ_h, 200):
- base = (e/d)², cộng thêm 0,3 · (max(−e, 0)/d)² khi c > 200 mm, tức phạt thêm khi dự báo thiếu trong lúc nước lên mạnh.
- Loss là tổng có trọng số theo mốc. `long_12h` dùng trọng số 1 / 1 / 2,5 để dồn sức cho +12 h mà vẫn học cùng +3/+6 h.
- Đã thử model chỉ dự báo +12 h: đạt 0 lỗi trên val nhưng test luôn có 3–9 lỗi. Có vẻ việc học cùng các mốc ngắn giúp model ổn định hơn.

### 2.4 Các kênh đầu vào

Thứ tự kênh: 3 kênh cơ bản trước, rồi các nhóm theo thứ tự `--groups fc_ecmwf wind tide sea`. Mọi kênh dự báo chỉ dùng thông tin có sẵn tại t.

| Nhóm | Kênh | `short_3h_6h` | `long_12h` |
|---|---|---|---|
| Cơ bản | `water_level_depth` (mực nước đã lọc gai), `water_level_missing`, `flood_season` (1 vào tháng 9–12) | 3 | 3 |
| `fc_ecmwf`: mưa dự báo ECMWF | lượng mưa 0–3, 3–6, 6–12, 12–24 h tới cho 3 ô có tên (A Lưới, Thuận An, Bạch Mã) và trung bình 7 ô lưu vực, log1p(mm); `fc_age_hours`; `fc_missing_frac`. Lấy từ lượt chạy mới nhất khởi tạo trong khoảng 6–24 h trước t | 18 | 18 |
| `wind`: gió vào bờ ECMWF, dự báo từ 24 h trước | `wind_onshore_past24`, `_next6`, `_next12`, `_chg12`, `wind_missing_frac`. Điểm 16,85N 107,85E (ô 16,75/107,75) | 5 | 5 |
| `tide`: triều thiên văn | fit điều hòa 7 thành phần trên 7 ngày gần nhất: `tide_now`, `tide_d3`, `tide_d6`, `tide_d12`, `tide_missing_frac` | 5 | 5 |
| `sea`: mực nước biển ngoài khơi dự báo (Open-Meteo Marine, `previous_day1`) | `sea_setup_chg` (TB t+1…t+12 trừ TB t−23…t), `sea_anom` (giá trị tại t trừ TB 7 ngày), `sea_next12_anom` (TB t+1…t+12 trừ TB 7 ngày), `sea_missing_frac`. Điểm 16,70N 107,75E (ô 16,708/107,708) | 4 | 4 |
| `sea`, phần bổ sung (`direct: true`) | `sea_now`, `sea_p3`, `sea_p6`, `sea_p12` (giá trị dự báo tại t+h trừ giá trị tại t) | 4 | – |
| **Tổng** | | **39** | **35** |

Tất cả kênh dự báo chỉ nhìn tới t+12 h. Giá trị `previous_day1` của giờ H là dự báo phát khoảng 24 h trước H, nên mọi giá trị dùng đều đã có sẵn ít nhất 12 h trước thời điểm dự báo.

---

## 3. Dữ liệu và tiền xử lý

### 3.1 Nguồn dữ liệu

| Dữ liệu | Nguồn | File / bảng | sha256 |
|---|---|---|---|
| Mực nước 10 phút | API DCU → PostgreSQL (`src/backend/ingestion/api_backfill.py`, worker) | `hydrology.observations`, trạm `0074647705` | đồng bộ đến 12/11/2026 |
| Gió và áp suất, train | Open-Meteo Previous Runs, `ecmwf_ifs025` | `data/external/open_meteo/prev_offshore.json` (01/11/2025–21/10/2026) | `846a6db4270d8117b1d14593281db725bb18987e53c8b1b8837f758af17ae224` |
| Gió, đánh giá đến 12/11 | như trên | `data/external/open_meteo/prev_offshore_to_20261112.json` (đến 12/11/2026; trùng tuyệt đối với file trên ở đoạn chung) | `ba6edbe46b2cc4d23f6f96a65b57d7106eb2525b0a64c5bb29208f4b2a12dbde` |
| Mưa dự báo ECMWF, train | Open-Meteo Single Runs, `ecmwf_ifs`, 7 điểm | `data/external/open_meteo/ecmwf_single_runs/` (1.407 lượt chạy, 14/11/2025–01/11/2026) | |
| Mưa dự báo ECMWF, đánh giá | như trên, tải lại kèm độ ẩm đất | `data/external/open_meteo/ecmwf_single_runs_soil/` (1.458 lượt chạy, 13/11/2025–12/11/2026; lượng mưa trùng tuyệt đối với bản train) | |
| Mực nước biển dự báo | Open-Meteo Marine, `sea_level_height_msl_previous_day1` | `data/external/open_meteo/marine_offshore_ta.json` (01/11/2025–12/11/2026; có `previous_day1` từ 18/11/2025) | `8f241394f12b1a0604b4ee2aaf1c9070982d455f2ddf83343590f7fcc169c550` |

### 3.2 Mực nước theo giờ và lọc gai số đo (`despike`)

- **Mực nước theo giờ** tại t = trung bình các số đo hợp lệ trong khoảng (t − 1 h, t] (`build.hourly_level`).
- **Lọc gai** (`build.despike_observations`, khóa config `despike`): bỏ một số đo 10 phút nếu nó lệch hơn **150 mm** so với **trung vị các số đo trong cửa sổ 121 phút đặt giữa**. Mực nước giờ được tính từ các số đo còn lại. Không dự đoán hay nội suy giá trị thay thế, vì mực nước giờ cũng là nhãn để train và chấm điểm. Database không bị sửa.
- **Giai đoạn train/val/test: bỏ 24 số đo**, làm thay đổi 14 giá trị giờ. Danh sách nằm trong `despiked_readings.csv` của mỗi dataset. Theo ngày (giờ UTC):

| Ngày | Số đo bị bỏ | Độ lệch so với trung vị (mm) |
|---|---|---|
| 17/11/2025 | 1 | +397 |
| 03/04/2026 | 4 | +178…+189 |
| 07/04/2026 (sáng 08/04 giờ VN) | 5 | +183…+296 |
| 15/04 | 1 | −157 |
| 28/04 | 5 | +188…+325 |
| 29/08 | 4 | +242…+310 |
| 03/09 (sáng 04/09 giờ VN) | 1 | +342 |
| 28/09 | 1 | +230 |

- Giai đoạn đánh giá thêm có 2 gai nữa: 16/10 04:20 (1 số đo) và 17/10 17:30–17:40 giờ VN (2 số đo 1.773 và 1.417 mm, trong khi xung quanh khoảng 250 mm).
- **Ngoại lệ đã duyệt:** đoạn tụt −175/−284 mm lúc 07:30–07:40 ngày 11/04/2026 (giờ VN) được **giữ nguyên**, vì Dã Viên cũng tụt gần cùng lúc nên có thể là hiện tượng thật (`despike.keep`).
- Biểu đồ: `figures/nga_ba_sinh/sensor_spikes.png` và `figures/nga_ba_sinh/sensor_spikes_new_candidates.png`.

### 3.3 Sensor đổi mốc đo ngày 10/10/2026 (`sensor_steps`)

- Lúc **2026-10-10 15:20 UTC (22:20 giờ VN)**, mực nước tụt 157 mm chỉ trong 10 phút, trong khi Thuận An lên 30 mm và Dã Viên lên 17 mm.
- Từ đó, chênh lệch với hai trạm này đổi khoảng −172 mm và giữ nguyên đến 12/11/2026, tức sensor đã đổi mốc đo.
- **Xử lý:** giữ nguyên số đo thô, chỉ **loại 13 mẫu** có giờ hiện tại hoặc giờ nhãn nằm ở phía bên kia lúc đổi mốc (thời điểm phát dự báo 04:00–16:00 UTC ngày 10/10). Các mẫu này có `split = excluded_sensor_step`.
- Không hiệu chỉnh số đo, vì không biết mốc nào đúng. Model dự báo mức thay đổi, nên dùng được với số đo thô theo mốc mới.
- **Nên báo đơn vị quản lý trạm**, vì mức báo động phụ thuộc vào mực nước tuyệt đối.

---

## 4. Tái tạo từng bước

Mọi lệnh chạy từ thư mục gốc repo, có file `.env` chứa `DATABASE_URL` (xem `PIPELINE.md` mục 0–1).

### 4.1 Quan trắc vào PostgreSQL

```bash
.venv/bin/python -m src.backend.ingestion.api_backfill --station 0074647705 --start 2025-11-01T00:00:00+00:00 --end 2026-11-12T23:59:59+00:00 --apply
```

### 4.2 Tải dữ liệu Open-Meteo

Gói miễn phí tính **mỗi điểm trong một request là một lượt**, giới hạn khoảng 5.000 lượt/giờ và 10.000 lượt/ngày. Với 7 điểm, mỗi lượt chạy ECMWF tính là 7 lượt, nên phải đặt `--pause 5`. Code tự chờ rồi thử lại khi gặp lỗi HTTP 429.

```bash
# gió ECMWF dự báo từ 24 h trước (train và đánh giá)
.venv/bin/python -m src.ai.station_dataset.open_meteo previous-runs --lat 16.85 --lon 107.85 --start 2025-11-01 --end 2026-10-21 \
    --out data/external/open_meteo/prev_offshore.json
.venv/bin/python -m src.ai.station_dataset.open_meteo previous-runs --lat 16.85 --lon 107.85 --start 2025-11-01 --end 2026-11-12 \
    --out data/external/open_meteo/prev_offshore_to_20261112.json

# mưa dự báo ECMWF theo từng lượt chạy, 7 điểm (cùng 7 điểm với Thuận An)
.venv/bin/python -m src.ai.station_dataset.open_meteo single-runs --start 2025-11-13T00:00 --end 2026-11-12T06:00 --pause 5 \
    --hourly precipitation,soil_moisture_0_to_7cm,soil_moisture_7_to_28cm,soil_moisture_28_to_100cm,soil_moisture_100_to_255cm \
    --out-dir data/external/open_meteo/ecmwf_single_runs_soil --points \
    16.458944,107.571361 16.554968,107.63527 16.251017,107.873289 16.199675,107.857476 \
    16.122778,107.694444 16.576556,107.347889 16.221111,107.179444
# dataset train đọc từ data/external/open_meteo/ecmwf_single_runs (chỉ lượng mưa). Lượng mưa của hai thư mục trùng tuyệt đối,
# nên có thể chép hoặc tạo liên kết ecmwf_single_runs -> ecmwf_single_runs_soil.

# mực nước biển ngoài khơi dự báo (1 điểm, 1 request cho cả năm)
.venv/bin/python -m src.ai.station_dataset.open_meteo marine --lat 16.70 --lon 107.75 --start 2025-11-01 --end 2026-11-12 \
    --out data/external/open_meteo/marine_offshore_ta.json
```

Lượt chạy ECMWF 2026-06-10T12:00 không có trên API. Hai lượt chạy 18:00 ngày 10/06 và 06:00 ngày 11/06 không có độ ẩm đất, nhưng vẫn có lượng mưa. File tải lại có thể khác ở các trường metadata; các biến được dùng không đổi.

### 4.3 Dựng hai dataset train

```bash
set -a && . ./.env && set +a
.venv/bin/python -m src.ai.station_dataset.build --config models/nga_ba_sinh/dataset_config_onshore_tide7_sea7.json \
    --name nbs_onshore_tide7_fc_sea7_v3 --groups fc_ecmwf wind tide sea --output-dir train_data/nga_ba_sinh
.venv/bin/python -m src.ai.station_dataset.build --config models/nga_ba_sinh/dataset_config_onshore_tide7_sea.json \
    --name nbs_onshore_tide7_fc_sea_v3 --groups fc_ecmwf wind tide sea --output-dir train_data/nga_ba_sinh
```

- Hai config chỉ khác nhau ở khóa `sea.direct`: `true` cho bản 7 kênh, không có cho bản 3 kênh. Cả hai đều có `wind.channels` (4 kênh vào bờ), `tide.window_days: 7`, `despike` và `sensor_steps`.
- **Số mẫu mong đợi:** `{'train': 6073, 'validation': 1080, 'test_exploratory': 710, 'purged': 48, 'excluded_sensor_step': 13}`.
- **sha256** (đã kiểm chứng ngày 16/11/2026: dựng lại cho kết quả giống từng byte, kể cả `normalizers.json`):

| File | `nbs_onshore_tide7_fc_sea7_v3` | `nbs_onshore_tide7_fc_sea_v3` |
|---|---|---|
| `water.parquet` | `b9a4a308ac239bd3bdf27c344615b3f6855495c9a6b1307efbe37877c1370bdf` | giống bên trái |
| `samples.parquet` | `ca018361d7f8c918d9a45b6d711be75b74aee9940ec14e3009c349d31f28b682` | giống bên trái |
| `rainfall.parquet` | `9d8b1905a05c4be7f25f2f6be175f051cefbe8fc172fc43ae025092319c17839` | `e20404578dc24ac6a9bb8bb74d780acc26b7f76fc54c95f32970edce02cb5049` |
| `calendar.parquet` | `31ddea952f210a56f64971f1f702f32a68bec467e4dd69daa4485b78a49e1d0f` | giống bên trái |

### 4.4 Train hai model

```bash
# short_3h_6h
nice -n 19 .venv/bin/python -m src.ai.train --mlflow-ex-name nga_ba_sinh --horizon 1h \
  --dataset-dir train_data/nga_ba_sinh/nbs_onshore_tide7_fc_sea7_v3 --target-horizons 3 6 12 \
  --rain-stations all --use-weather-forecast false --feature-set water_level+rain+flood_season \
  --train-method gru --head-mode station --lstm-seq-len 12 --lstm-hidden-size 64 --lstm-num-layers 3 \
  --lstm-drop-out 0.2 --lstm-input-dropout 0.1 --lstm-head-dropout 0.1 --lstm-lr 0.0001 --lstm-weight-decay 0.001 \
  --lr-scheduler step --batch-size 64 --num-epochs 100 --early-stopping 100 \
  --change-loss combined --change-under-weight 0.3 --change-event-weight 0 --change-margin-mm 0 \
  --lstm-checkpoint-metric val_loss --skip-test --seed 23 --diagnose-change-dynamics \
  --checkpoint-dir model/experiments/nga_ba_sinh/nbs_onshore_tide7_fc_sea7_v3/b64_lr0.0001_wd0.001_gru_h64x3_do0.2-0.1-0.1_seq12_e100_combinedu0.3_step_s23

# long_12h
nice -n 19 .venv/bin/python -m src.ai.train --mlflow-ex-name nga_ba_sinh --horizon 1h \
  --dataset-dir train_data/nga_ba_sinh/nbs_onshore_tide7_fc_sea_v3 --target-horizons 3 6 12 --lstm-horizon-weights 1 1 2.5 \
  --rain-stations all --use-weather-forecast false --feature-set water_level+rain+flood_season \
  --train-method gru --head-mode station --lstm-seq-len 25 --lstm-hidden-size 128 --lstm-num-layers 3 \
  --lstm-drop-out 0.2 --lstm-input-dropout 0.1 --lstm-head-dropout 0.1 --lstm-lr 0.0001 --lstm-weight-decay 0.001 \
  --lr-scheduler step --batch-size 64 --num-epochs 130 --early-stopping 200 \
  --change-loss combined --change-under-weight 0.3 --change-event-weight 0 --change-margin-mm 0 \
  --lstm-checkpoint-metric val_loss --skip-test --seed 23 --diagnose-change-dynamics \
  --checkpoint-dir model/experiments/nga_ba_sinh/nbs_onshore_tide7_fc_sea_v3/hw112.5_b64_lr0.0001_wd0.001_gru_h128x3_do0.2-0.1-0.1_seq25_e100_combinedu0.3_ev0_step_s23_v2
```

- Các tham số trên khớp với bản ghi MLflow của hai run gốc (`mlflow_local.db`, experiment `nga_ba_sinh`), trừ `--change-margin-mm 0`, vốn không được ghi trong MLflow.
- Thư mục của `long_12h` có đuôi `e100_..._v2` nhưng run thực tế chạy **130 epoch** với early stopping 200.
- `--diagnose-change-dynamics` lưu mọi epoch vào `…_best.dynamics/epoch_XXX/`. Bắt buộc phải có, vì epoch được chọn **không phải** epoch mà trainer chọn theo val loss.
- Train là tất định trên cùng GPU và cùng phiên bản thư viện (`PIPELINE.md` mục 0). Chưa chạy lại cả hai lần train để kiểm chứng việc này riêng cho Ngã Ba Sình.

### 4.5 Chọn epoch và đóng gói

1. Chấm mọi epoch (từ epoch 10 trở đi) trên val, test và giai đoạn 15/10–12/11 (mục 4.6). Đếm số lỗi > 200 mm riêng cho từng mốc.
2. **`short_3h_6h`:** chọn epoch có 0 lỗi ở +3 h và +6 h trên cả ba giai đoạn, và có sai số lớn nhất nhỏ nhất. Đó là **epoch 59**, sai số lớn nhất 187,6 mm. Các epoch 52–62 của run này cũng đều 0 lỗi, nên đây là một vùng ổn định.
3. **`long_12h`:** chọn epoch có 0 lỗi ở +12 h trên cả ba giai đoạn. Trong toàn bộ thử nghiệm chỉ có 3 epoch như vậy (102, 110, 121, cùng run này). Chọn **epoch 110** vì có biên độ tốt nhất (197,6 / 184,2 / 195,6 mm trên val / test / mới).
4. Chép `model.pt`, `model.preprocessing.joblib`, `model.architecture.json`, `model.loss.json`, `summary.json` của hai epoch vào `models/nga_ba_sinh/combined_v1/short_3h_6h/` và `.../long_12h/`. Mỗi thư mục tự dùng được, không cần dataset: bộ chuẩn hóa, danh sách kênh, độ dài cửa sổ và kiến trúc đều nằm trong các file này.

### 4.6 Chấm điểm 01/09–31/10/2026

Giai đoạn này vượt quá lưới train (kết thúc 15/10), nên phải dựng dataset trên lưới mở rộng đến 12/11 và **dùng lại bộ chuẩn hóa của dataset train**:

```bash
set -a && . ./.env && set +a
.venv/bin/python -m src.ai.station_dataset.build --config models/nga_ba_sinh/eval_config_ext_20261112_sea7.json \
    --name nbs_ext_sea7_v3 --groups fc_ecmwf wind tide sea --output-dir <thư mục đánh giá> \
    --normalizers-from train_data/nga_ba_sinh/nbs_onshore_tide7_fc_sea7_v3
.venv/bin/python -m src.ai.station_dataset.build --config models/nga_ba_sinh/eval_config_ext_20261112_sea.json \
    --name nbs_ext_sea_v3 --groups fc_ecmwf wind tide sea --output-dir <thư mục đánh giá> \
    --normalizers-from train_data/nga_ba_sinh/nbs_onshore_tide7_fc_sea_v3

.venv/bin/python -m src.ai.station_dataset.evaluate combined \
    --member <thư mục đánh giá>/nbs_ext_sea7_v3 models/nga_ba_sinh/combined_v1/short_3h_6h 3,6 \
    --member <thư mục đánh giá>/nbs_ext_sea_v3  models/nga_ba_sinh/combined_v1/long_12h   12 \
    --start 2026-09-01T00:00:00+07:00 --end 2026-10-31T23:00:00+07:00
```

- Lệnh `evaluate combined` lấy **mọi** thời điểm phát dự báo trong khoảng thời gian đã cho, kể cả các mẫu nằm ở ranh giới giữa các tập. Mặc định nó loại các mẫu `excluded_sensor_step`; thêm `--keep-sensor-step` để giữ lại.
- Đã kiểm chứng ngày 16/11/2026: lệnh này cho đúng bảng ở mục 5.
- Hai config đánh giá chỉ khác config train ở `grid_end_utc` (2026-11-12T03:00Z) và đường dẫn tới các file Open-Meteo dài hơn.

---

## 5. Hiệu năng từ 01/09 đến 31/10/2026

Dự báo mọi giờ từ **00:00 ngày 01/09 đến 23:00 ngày 31/10/2026 (giờ Việt Nam)**: 1.464 thời điểm phát dự báo, trừ 13 mẫu ở lúc sensor đổi mốc, còn **1.451**. Mực nước đã lọc gai.

| Mốc | **MAE (mm)** | RMSE (mm) | Độ lệch trung bình (mm) | Sai số lớn nhất (mm) | Số lỗi > 200 mm | MAE persistence (mm) | Mức giảm MAE so với persistence |
|---|---|---|---|---|---|---|---|
| **+3 h** | **38,0** | 46,6 | +13,7 | 152,7 | **0** | 90,2 | 58 % |
| **+6 h** | **55,7** | 68,8 | +23,6 | 187,1 | **0** | 130,6 | 57 % |
| **+12 h** | **52,0** | 64,9 | +8,2 | 197,6 | **0** | 71,5 | 27 % |
| Cả ba mốc | 48,6 | | | | **0** | | |

Persistence là "giữ nguyên mực nước hiện tại". Độ lệch trung bình dương nghĩa là model dự báo hơi cao.

**MAE theo tháng (mm):**

| Tháng | +3 h | +6 h | +12 h |
|---|---|---|---|
| 9/2026 | 37,2 | 55,7 | 52,3 |
| 10/2026 | 38,7 | 55,8 | 51,6 |

**Ghi chú:**
- Nếu giữ cả 13 mẫu ở lúc sensor đổi mốc, MAE +12 h là 52,6 mm. Các mốc khác không đổi, và vẫn không có lỗi nào > 200 mm.
- Giai đoạn này gồm:
  - một phần val (01/09–13/09, 312 mẫu), trong đó có đợt bão 11–13/09;
  - 24 mẫu ở ranh giới giữa val và test;
  - test (14/09–14/10);
  - giai đoạn đánh giá thêm (15/10–31/10).
- Sai số lớn nhất ở +12 h (197,6 mm) rơi vào đợt bão 13/09 lúc nước rút, sát ngưỡng 200 mm.
- So sánh: model Thuận An trên cùng hai tháng đạt 31,0 / 40,9 / 43,8 mm. Ngã Ba Sình khó dự báo hơn vì chịu cả triều, lũ sông và nước dâng.

**Theo từng giai đoạn** (số lỗi > 200 mm bằng 0 ở mọi mốc):

| | Val (1.080 mẫu) | Test (710 mẫu) | Mới 15/10–12/11 (668 mẫu) |
|---|---|---|---|
| MAE +3 / +6 / +12 h (mm) | 32,7 / 53,0 / 60,8 | 37,7 / 54,0 / 47,7 | 38,8 / 54,3 / 53,8 |
| Sai số lớn nhất (mm) | 145 / 185 / 197,6 | 153 / 187 / 184 | 168 / 187,6 / 195,6 |

---

## 6. Dùng model

**Trên một dataset có sẵn:**
```python
from src.ai.station_dataset.evaluate import predict
s = predict('models/nga_ba_sinh/combined_v1/short_3h_6h/model.pt', '<dataset sea7>', 'all')   # dùng +3, +6 h
l = predict('models/nga_ba_sinh/combined_v1/long_12h/model.pt',    '<dataset sea>',  'all')   # dùng +12 h
forecast = pd.concat([s[s.horizon_hours.isin([3, 6])], l[l.horizon_hours == 12]])
```

**Chạy thật** (`src/ai/sequence_checkpoint.SequenceRuntime`):
```python
from src.ai.sequence_checkpoint import SequenceRuntime
short = SequenceRuntime('models/nga_ba_sinh/combined_v1/short_3h_6h/model.pt')
long_ = SequenceRuntime('models/nga_ba_sinh/combined_v1/long_12h/model.pt')
p_short = short.predict_mm(x_short, station_ids)   # x_short: [1, 12, 39], đã chuẩn hóa theo bộ chuẩn hóa trong preprocessing.joblib
p_long  = long_.predict_mm(x_long,  station_ids)   # x_long:  [1, 25, 35]
forecast = {3: p_short[0, 0], 6: p_short[0, 1], 12: p_long[0, 2]}   # mm
```

Đầu vào lúc chạy thật cần có:
- mực nước 10 phút đã lọc gai, chỉ so với các số đo trước đó, vì chưa có số đo 1 giờ sau;
- lượt chạy ECMWF mới nhất khởi tạo trong khoảng 6–24 h trước, cho đủ 7 ô;
- gió và mực nước biển `previous_day1` cho 12 h tới;
- fit triều 7 ngày.

Thứ tự kênh đúng như `feature_schema.json` của dataset tương ứng.

---

## 7. Đã kiểm chứng

| Kiểm tra | Kết quả |
|---|---|
| Dựng lại hai dataset train từ PostgreSQL và các file Open-Meteo | Giống từng byte (`water`, `samples`, `rainfall`, `calendar`, `normalizers.json`) |
| Lượng mưa ECMWF tải lại (thư mục `_soil`) so với bản train | Trùng tuyệt đối trên 1.104 lượt chạy đã so sánh lúc tải (so theo mã ô và thời điểm) |
| Gió `previous_day1` tải lại đến 12/11 so với bản train | Trùng tuyệt đối trên 8.520 giờ chung |
| Dataset đánh giá mở rộng so với dataset train trên đoạn chung | Trùng tuyệt đối (mực nước, mưa, gió, triều) |
| `evaluate combined` với hai model đã đóng gói | Cho đúng bảng ở mục 5 |
| Lọc gai không ảnh hưởng trạm khác | Dataset Thuận An dựng lại vẫn giống từng byte |
| Train lại hai model | **Chưa chạy lại** để kiểm chứng |

---

## 8. Hạn chế và việc cần làm

1. **Không còn giai đoạn nào chưa dùng để chọn model.** Hai thành phần và epoch được chọn sau khi đã xem val, test và giai đoạn 15/10–12/11, trong hàng nghìn epoch (hơn 40 run). Kết quả 0 lỗi là đúng trên các giai đoạn đó, nhưng **phép thử thật là dữ liệu sau 12/11/2026**. Chấm lại bằng `evaluate combined` (mục 4.6) khi có dữ liệu mới: dựng lại dataset đánh giá với `grid_end_utc` mới, và tải thêm Open-Meteo.
2. **Biên độ an toàn ở +12 h mỏng.** Sai số lớn nhất là 197,6 mm (val) và 195,6 mm (giai đoạn mới). Chỉ 3 epoch trong toàn bộ thử nghiệm đạt 0 lỗi +12 h trên cả ba giai đoạn, và đều thuộc cùng một run với seed 23. Các seed khác của cùng công thức có 3–7 lỗi. Phần +3/+6 h ổn định hơn nhiều.
3. **Các mẫu khó nhất** đều trong sự kiện lớn:
   - lúc nước bắt đầu lên trong bão 11/09 18:00 (+6 h);
   - lúc nước rút sau bão 13/09 07:00–09:00 (+12 h);
   - đợt mưa lớn kèm gió mùa đông bắc 08–09/11 (+12 h).
   
   Mọi model đều sai cùng chiều ở những mẫu này. Đó là giới hạn thông tin của các đầu vào hiện có.
4. **Sensor đổi mốc đo ngày 10/10/2026** (mục 3.3): cần đơn vị quản lý trạm xác nhận mốc nào đúng. Lúc chạy thật dùng số đo thô theo mốc mới.
5. **Lúc chạy thật** cần duy trì 4 nguồn dữ liệu: quan trắc DCU, ECMWF Single Runs cho 7 ô, Previous Runs cho gió, Marine `previous_day1`. Hai thành phần cần hai bộ đầu vào khác nhau (39 và 35 kênh).

---

## Phụ lục A. Nhật ký thử nghiệm (12–16/11/2026)

Tất cả đều là GRU, lr 0,0001, loss `combined` với under-weight 0,3. Các con số là lỗi > 200 mm (+3 / +6 / +12 h) trên val · test; "mới" là giai đoạn 15/10–12/11. Ngày 16/11/2026 đã dọn ổ cứng: xóa mọi checkpoint, dataset và config của các thử nghiệm dưới đây, chỉ giữ hai run gốc của `combined_v1`, hai dataset train `nbs_onshore_tide7_fc_{sea7,sea}_v3` và config của chúng. Kết quả vẫn nằm trong MLflow (`mlflow_local.db`, experiment `nga_ba_sinh`).

| Dataset | Thay đổi | Kết quả tốt nhất | Kết luận |
|---|---|---|---|
| `nbs_water_only` | chỉ mực nước | h64 seq25: 3/15/20 · 0/8/8 | h32 không học kịp |
| `nbs_tide7` / `nbs_tide30` | triều | triều 7 ngày: 0/10/20 · 0/0/10 | 7 ngày tốt hơn 30 ngày (khác Thuận An) |
| `nbs_wind_tide7` / `nbs_onshore_tide7` | gió 15 kênh / 4 kênh vào bờ | 4 kênh: 0/3/12 · 0/1/7 | dùng 4 kênh; cắt gió tại 0 kém hơn |
| `nbs_onshore_tide7_fc` | mưa dự báo ECMWF | 0/3/5 · 0/3/3 | giảm mạnh lỗi +12 h; chỉ dùng ô Thuận An thì kém hơn |
| `..._rainobs` | mưa đo 3 cụm | không tốt hơn | thêm lỗi dự báo cao lúc nước rút |
| `nbs_onshore_tide7_fc` | lịch `step`, cửa sổ 12/25/48 h, h128, under-weight 0,5 | step seq12 ep66: 0/4/3 · 0/1/3 | `step` tốt hơn cosine |
| `..._lin` | dự báo tuyến tính làm đầu vào (kể cả dropout nhóm) | val +12 h ≥ 12 | GRU "đi đường tắt" theo dự báo tuyến tính; đóng hướng này |
| `..._v2` | loại 13 mẫu quanh lúc sensor đổi mốc | test 0/1/3 → 0/0/0 | lỗi test cũ là lỗi dữ liệu |
| `..._v3` | lọc gai số đo | trung vị val 0/4/7, test 0/0/0; mới 1/5/13 | mốc mới |
| `..._soil_v3` | độ ẩm đất ECMWF | trung vị val 0/8/22 | kém hơn (độ ẩm đất phản ánh mùa nhiều hơn bão) |
| `..._sea_v3` | **mực nước biển dự báo, 1 điểm** | seq12: 0/0/7 · 0/0/0; mới 0/1/4 | **bước tiến lớn trên dữ liệu mới** |
| `..._sea7_v3` | + giá trị trực tiếp +3/+6/+12 h | seed 23: 0/0/6 · 0/0/0; mới 0/0/2 | tương đương; là gốc của `short_3h_6h` |
| `..._seamulti_v3` | mực nước biển ở 8 điểm (trung bình, độ dốc bắc–nam, xa bờ–gần bờ) | seq25 seed 23 ep83: 0/1/2 · 0/0/0; mới 0/0/0 | tốt trên dữ liệu mới |
| `..._seamulti_wave_v3` | + sóng | 0/4/8 · 0/0/0 | kém hơn; đóng hướng này |
| nhiều bản | event-weight 1 / 3 | không cải thiện | chỉ dời lỗi từ thiếu sang thừa |
| nhiều bản | model chỉ +12 h | val +12 h = 0 ở nhiều epoch, nhưng test luôn 3–9 lỗi | kém hơn model 3 mốc |
| `..._sea_v3` | **h128, trọng số mốc 1 1 2 / 1 1 2,5** | seed 23 ep98: 0/0/0 · 0/0/0; mới 0/0/1 | gốc của `long_12h` (1 1 2,5, ep110) |
| tổ hợp | trung bình 1–3 model (11.521 tổ hợp từ 41 ứng viên) | tốt nhất còn 1 lỗi | các model sai cùng chiều nên trung bình không giúp |
| **ghép theo mốc** | **+3/+6 h từ sea7 ep59, +12 h từ h128 ep110** | **0/0/0 · 0/0/0; mới 0/0/0** | **`combined_v1`** |

**Các hướng đã loại bằng phân tích** (không train): xả hồ SOHOA (trong bão chỉ 230–310 m³/s, quá nhỏ), gió dự báo mới hơn (dùng gió thực tế cũng không giúp), áp suất (chỉ phản ánh mùa), mưa dự báo xa hơn 24 h và mưa ECMWF đã rơi (không thêm thông tin), dự báo của model Thuận An (không học được cách dùng từ dữ liệu train).

## Phụ lục B. Các model đóng gói trước đó

Hai thư mục dưới đây đã bị xóa ngày 16/11/2026 khi dọn ổ cứng; chỉ còn lại kết quả.

| Thư mục | Mô tả | Val · Test · Mới |
|---|---|---|
| `models/nga_ba_sinh/seq12_step_gru64x3_ep66/` | mưa dự báo + gió + triều, chưa lọc gai (15/11) | 0/4/3 · 0/0/0 (đã loại 13 mẫu) · 1/5/10 |
| `models/nga_ba_sinh/sea3_h128x3_seq25_hw112_s23_ep98/` | mực nước biển, h128, trọng số mốc 1 1 2, epoch 98 (16/11) | 0/0/0 · 0/0/0 · 0/0/1 |
