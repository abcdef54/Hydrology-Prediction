# Nghiên cứu và phân tích mô hình

Thư mục này lưu công cụ tạo dữ liệu và các notebook dùng trong quá trình phát
triển mô hình. Backend của ứng dụng nằm trong `src/backend/`; mã train và đánh
giá mô hình nằm trong `src/ai/`.

| File | Mục đích |
|---|---|
| [TAIL_LOSS.md](TAIL_LOSS.md) | Loss phạt quanh ngưỡng 200 mm, chọn checkpoint theo lỗi lớn và lệnh train |
| [OUTPUT_HORIZON_ABLATION.md](OUTPUT_HORIZON_ABLATION.md) | Dataset 24 target giờ, loss chính/phụ, command tự train O5/O24 và so B/production |
| [TREE_HORIZON_EXPERIMENT.md](TREE_HORIZON_EXPERIMENT.md) | XGBoost/LightGBM trên dataset mới, lịch sử96h, native checkpoint và so B/production theo lỗi >200mm |
| [weather_forecast/](weather_forecast/) | Phân tích dự báo mưa và tạo dataset; collector nằm ở `src/backend/ingestion/weather_forecast.py` |
| `build_training_dataset_1h_top4.py` | Tạo dataset theo giờ với bốn trạm mưa đã chọn, chia Train/Val/Test và kiểm tra dữ liệu |
| `seasonal_analysis.ipynb` | Phân tích mùa mưa, tương quan mưa–mực nước và các đợt mưa |
| `mlflow_metrics.ipynb` | Tổng hợp, đối chiếu metric từ MLflow và đánh giá theo mùa |
| `refresh_da_vien_dataset.py` | Xuất snapshot PostgreSQL chỉ đọc, tạo dataset có phiên bản và audit |
| `resplit_hourly_dataset.py` | Chia lại snapshot theo mốc thời gian, purge horizon và giữ dataset nguồn |
| `rank_rain_stations.py` | So correlation trên train và toàn snapshot, đề xuất top 1/top 2 từ train |
| [RAIN_STATION_ABLATION.md](RAIN_STATION_ABLATION.md) | Lệnh train chọn 1/2/4 trạm mưa và cách tách checkpoint |
| [MODEL_COMPARISON.md](MODEL_COMPARISON.md) | Pipeline tự so model mới với model app sau mỗi lần train |
| `plot_forecast_error_histograms.py`, `summarize_error_histograms.py` | Histogram theo horizon và báo cáo thống kê theo khoảng ngày |

## Sử dụng

Chạy lệnh từ thư mục gốc dự án:

```bash
.venv/bin/python research/build_training_dataset_1h_top4.py --help
```

Mở các notebook tại đây bằng Jupyter hoặc VS Code. Các đường dẫn dữ liệu và
artifacts vẫn tính từ thư mục gốc dự án. Cần có dữ liệu/model/MLflow tương ứng
trên máy để chạy; việc chuyển thư mục không tạo lại dataset hoặc chạy model.
Kết quả xuất vào `reports/`, được bỏ qua trong Git.

## Giới hạn của các công cụ phân tích hiện tại

Dataset hiện dùng: `train_data/da_vien_1h_2026-10-15` (đổi tên từ
`1h_top4_pg_2026-10-15_val_aug01_sep15`; tên cũ là symlink tương thích local).
CSV/split/manifest gốc giữ nguyên, lịch sử đổi tên nằm trong `dataset_identity.json`.

```bash
.venv/bin/python research/rank_rain_stations.py \
  --dataset-dir train_data/da_vien_1h_2026-10-15 \
  --output-dir reports/rain_station_correlation/<run_id>
```

Bảng chính dùng Pearson giữa tổng mưa 6 giờ và thay đổi nước 12 giờ tiếp theo,
cùng các timestamp hợp lệ ở cả bốn trạm. Có thêm bảng đủ 25 cặp cửa sổ/horizon
và ma trận tương quan giữa trạm. Chỉ dùng ranking train đề xuất nguồn cho
ablation; toàn bộ thời gian dùng để đối chiếu mô tả, không dùng chọn trạm.

Chạy từ thư mục gốc repository. Công cụ đọc PostgreSQL dùng `DATABASE_URL` trong
môi trường hoặc `.env`; không đưa thông tin đăng nhập vào mã nguồn. Việc export
không train model hoặc sửa quan trắc.

Các script dùng một lần để viết báo cáo, vẽ các đợt cụ thể và khảo sát DB đã
được dọn sau khi sử dụng. Kết quả phân tích vẫn ở `reports/`. Các hàm tính metric
còn dùng trong pipeline được giữ tại `src/ai/metrics.py`.

Dataset PostgreSQL mới và checkpoint thử nghiệm giữ local; model/scaler đang
phục vụ và các dataset đã được track trước đó không đổi. Dữ liệu mưa tổng hợp
giữ phép cộng số đo nguồn của pipeline; khoảng tích lũy của từng bản ghi chưa
được xác minh. Các tập test đã phân tích nhiều lần là kết quả khám phá.

Kế hoạch tiếp theo nằm tại
[FORECAST_RAIN_ARCHITECTURE.md](../src/FORECAST_RAIN_ARCHITECTURE.md): thử số trạm
mưa, thêm dự báo mưa mới nhất đã có, cuối cùng so sánh 5 với 24 output.

## Thử tổ hợp trạm mưa bằng tên

Dataset chung cho các thử nghiệm hiện tại:
`train_data/da_vien_1h_2026-10-15_rain_pool_o24`.
Dataset có 6 trạm, 50 feature trong pool và 24 nhãn giờ; khi chọn 4 trạm thì
còn 38 inputs, chọn `hong_thai+nham` thì còn đúng 26 inputs của B, cùng thứ tự
và giá trị. Các feature ngoài mưa (kể cả dự báo mưa), nhãn, split và purge giữ
nguyên từ dataset B O24. Model B và registry production không đổi.

| Tên nhập | Mã | Trạm |
|---|---|---|
| `hong_thai` | `750062` | Hồng Thái |
| `nham` | `750007` | Nhâm |
| `bach_ma` | `750041` | Bạch Mã, khác Đỉnh Bạch Mã `750066` |
| `phu_bai` | `750014` | Hồ chứa nước Phú Bài |
| `vp` | `750033` | VP QLKTCTTL TP Huế |
| `binh_dien` | `750061` | Trạm mưa tại Thủy điện Bình Điền |

Tạo lại dataset vào một thư mục mới (builder từ chối ghi đè):

```bash
.venv/bin/python -m research.build_rain_station_dataset
```

Builder đọc parent `da_vien_1h_2026-10-15_forecast_mask_o24` và snapshot local
`reports/rain_regions_h12_h24_20261019/{hourly_rain.csv.gz,station_catalog.csv}`.
Đổi nguồn/thư mục đích bằng `--source-dir`, `--snapshot-dir`, `--output-dir`.
Không truy cập DB hoặc mạng. Dataset mới đang được ignore theo quy tắc
`train_data/da_vien_1h_*`; cần copy dataset riêng nếu chạy trên máy khác.

Chạy một tổ hợp với defaults ResidualLSTM/loss/seed theo cấu hình B, 5 outputs:

```bash
.venv/bin/python -m research.output_horizon_experiment \
  --dataset-dir train_data/da_vien_1h_2026-10-15_rain_pool_o24 \
  --rain-stations hong_thai+nham+bach_ma+phu_bai \
  --outputs 5 --device cuda \
  --mlflow-ex-name da_vien_rain_combinations
```

Ba tổ hợp cần thử lần lượt là:

- `hong_thai+nham+bach_ma+phu_bai`
- `hong_thai+bach_ma+phu_bai+vp`
- `hong_thai+bach_ma+binh_dien+vp`

Đối chứng train lại: `hong_thai+nham`. Thứ tự tên nhập không đổi thứ tự feature
trong dataset. Thêm `--check-only --no-mlflow --device cpu` để chỉ kiểm tra
data/model/loss/backward, không train hoặc tạo artifact. Khi train, bỏ các cờ
kiểm tra này; mặc định dùng GPU. Có thể dùng `--no-mlflow` để chạy offline.
MLflow chỉ nhận settings/metrics, không upload model. Checkpoint và preprocessing
được lưu local, mặc định dưới `model/experiments/rain_<combination>/` với thư mục
run duy nhất; `--output-dir` chỉ định một thư mục mới cụ thể.

Checkpoint vẫn chọn trên validation theo **cả 5 horizon** như quy tắc B;
không thay loss hoặc checkpoint rule cùng lúc với thay trạm. Báo cáo tự động
so với B/production trên cùng anchor sau training. Test đã được dùng để khám phá,
chưa phải xác nhận độc lập. Không gọi lại builder hay train chỉ để đổi thứ tự tên.

Entry point `src.ai.train` cũng nhận `--rain-stations ... --outputs 5`, dùng
`--feature-set water_level+rain+flood_season+weather_forecast`; các tham số model/loss
cần nhập như lệnh train B cũ. Không dùng cùng `--num-rain-station` (top-N tự xếp
hạng trên train). Nếu không có `--outputs`, entry point này giữ toàn bộ nhãn của
dataset như trước. Các checkpoint explicit selection được tách theo ID và O5/O24.

Không giới hạn 4 trạm: có thể chọn cả 6 trạm hiện có. Để bổ sung nguồn khác,
build một pool mới bằng `--rain-stations all --output-dir <new-directory>` hoặc
chỉ định các alias/ID cần thiết. `all` ở builder là toàn bộ catalog snapshot,
còn `all` ở train là toàn bộ trạm đã có trong dataset. Manifest lưu cả tên đầy
đủ đã chuẩn hóa không dấu thành `snake_case`, có thể dùng những tên đó hoặc ID;
các alias ngắn ở trên nằm trong `src/ai/rain_stations.py`.

Tổng mưa mới theo cùng quy ước B: cộng số đo giờ `(t-1h,t]`, rolling 3/6/12/24h
đòi đủ số giờ; NaN không được điền 0 trước scaler. Giữ giờ có một phần số đo
hợp lệ như pipeline B, loại số đo invalid/âm; không áp dụng quy tắc 6/6 số đo
nghiêm ngặt của notebook hồi cứu vào training. Phú Bài thiếu khoảng 11,5% giờ
validation, đã lưu độ phủ trong manifest. Không lọc mẫu theo độ phủ mưa;
scaler fit train, dùng chung cho validation/test; NaN sau chuẩn hóa thành 0 như B.

## Thử bước dữ liệu 12h/24h và model chỉ một horizon

Giữ hai nguồn B (`hong_thai+nham`) và các feature theo đơn vị giờ vật lý.
Lấy bản ghi cách nhau 12h/24h, không lấy trung bình thành một giá trị ngày:
12h giữ mốc 00:00/12:00 Việt Nam; 24h giữ mốc 00:00 Việt Nam. Mỗi dòng giữ nguyên
feature tại mốc đó, kể cả water lag 1h/3h và tổng mưa 3h/6h/12h/24h. Forecast
cũng là bản đã có tại chính mốc input. Target là mực nước giờ tại t+12h/t+24h,
không phải trung bình của 12h/24h tương lai. Split và purge giữ từ parent.

Tạo dataset (đã tạo local; chỉ chạy lại vào thư mục mới):

```bash
.venv/bin/python -m research.build_temporal_experiment_dataset \
  --step-hours 12 --target-hours 12 24 \
  --output-dir train_data/da_vien_12h_2026-10-15_rain_pool_h12_h24

.venv/bin/python -m research.build_temporal_experiment_dataset \
  --step-hours 24 --target-hours 24 \
  --output-dir train_data/da_vien_24h_2026-10-15_rain_pool_h24
```

| Trường hợp | Bước dữ liệu | Sequence | Outputs |
|---|---|---:|---|
| Dữ liệu 12h | 12h | 9 (t−96h đến t) | +12h, +24h |
| Dữ liệu 24h | 24h | 5 (t−96h đến t) | +24h |
| Chuyên +12h | 1h | 96 (t−95h đến t) | +12h |
| Chuyên +24h | 1h | 96 (t−95h đến t) | +24h |

Chạy từ root repo, định nghĩa hàm dùng chung để giữ các tham số B:

```bash
train_time_case() {
  .venv/bin/python -m src.ai.train \
    --dataset-dir "$1" \
    --horizon "$2" \
    --lstm-seq-len "$3" \
    --target-horizons "${@:4}" \
    --train-method residual_lstm \
    --feature-set water_level+rain+flood_season+weather_forecast \
    --rain-stations hong_thai+nham \
    --seed 42 --batch-size 128 --num-worker 0 \
    --num-epochs 50 --early-stopping 25 --gradient-accumulation 1 \
    --lstm-hidden-size 32 --lstm-num-layers 3 \
    --lstm-drop-out 0.7 --lstm-input-dropout 0.2 --lstm-head-dropout 0.2 \
    --lstm-lr 0.0009 --lstm-weight-decay 0.001 \
    --lstm-correction-penalty 0.1 \
    --lstm-tail-penalty 0.5 --lstm-tail-threshold-mm 200 \
    --lstm-tail-transition-mm 20 --lstm-checkpoint-metric large_error_rate \
    --device cuda \
    --checkpoint-dir model/experiments/temporal_horizons \
    --mlflow-ex-name da_vien_temporal_horizons
}

train_time_case train_data/da_vien_12h_2026-10-15_rain_pool_h12_h24 12h 9 12 24
train_time_case train_data/da_vien_24h_2026-10-15_rain_pool_h24 24h 5 24
train_time_case train_data/da_vien_1h_2026-10-15_rain_pool_o24 1h 96 12
train_time_case train_data/da_vien_1h_2026-10-15_rain_pool_o24 1h 96 24
```

`--target-horizons` nhận giờ vật lý, tự xác định output size; không dùng cùng
`--outputs`. Không truyền weights thì tail loss dùng trọng số 1 cho từng output.
Checkpoint tách theo danh sách target và cadence; metadata lưu resolution.
MLflow chỉ lưu settings/metrics và các metadata đang có, không upload model.

So sánh tự động chạy production trên dữ liệu hourly gốc và candidate trên
cadence riêng; ghép bằng timestamp, kiểm tra nhãn khớp và chấm horizon chung.
Model thưa có ít mẫu hơn nên không so tổng số lỗi của nó với 2.960 dự báo B cũ;
cần chấm B trên chính các mốc/horizon tương ứng. Đánh giá riêng model +12h cũng
không trộn tổng lỗi với model có 5 outputs.

## Tìm hyperparameter: Grid Search / Optuna TPE

Entry point nằm trong `src/ai/hyperparam_finding.py`. Hỗ trợ XGBoost,
LightGBM, LSTM và ResidualLSTM. Mỗi trial train lại từ đầu,
giữ split cố định và có thể tìm cả sequence length. Không đọc test trong lúc tìm tham số.
Mặc định ưu tiên giảm số dự báo lỗi >200 mm trên **+12/+24h** của validation;
khi bằng nhau mới chọn RMSE nhỏ hơn. Dùng `--focus-horizons 1 3 6 12 24`
nếu muốn chọn theo cả năm mốc, hoặc `--objective rmse` nếu ưu tiên RMSE.

Phạm vi TPE mặc định đã được mở rộng:

| Tham số tree | Khoảng |
|---|---|
| `max_depth` | 2–16 |
| `learning_rate` | 0.001–0.3, log |
| `reg_lambda` | 0.0001–1000, log |
| `reg_alpha` | 1e-8–100, log |
| `subsample`, `colsample_bytree` | 0.3–1.0 |
| `min_child_weight` — XGBoost | 0.1–100, log |
| `num_leaves` — LightGBM | 8–256, log |
| `min_data_in_leaf` — LightGBM | 3–256, log |

TPE neural: hidden size 8–256 (bước 8), 1–5 layer, learning rate 1e-5–0.01 (log),
LSTM dropout 0–0.85, input/head dropout 0–0.6, weight decay 1e-8–0.1 (log),
batch size 32/64/128/256. Sequence length mặc định cũng là tham số tìm kiếm:
TPE chọn số giờ nguyên bất kỳ từ **12 đến 168**; Grid chọn 12/24/48/72/96/120/144/168.
Quy tắc loss giữ nguyên trừ các tham số được khai báo trong custom space. Khoảng rộng hơn không bảo đảm tối ưu toàn cục;
có thể cần tăng trial/epoch/vòng boosting theo kết quả validation.

Ví dụ TPE cho ResidualLSTM (chạy tại root repo):

```bash
.venv/bin/python -m src.ai.hyperparam_finding \
  --search-method tpe --train-method residual_lstm \
  --dataset-dir train_data/da_vien_1h_2026-10-15_forecast_mask_o24 \
  --outputs 5 --device cuda \
  --focus-horizons 12 24 --n-trials 100 \
  --num-epochs 50 --early-stopping 25 \
  --output-dir model/experiments/tpe_residual_lstm
```

Tree: đổi `--train-method xgb --device cuda` hoặc `--train-method lgbm --device cpu`.
`--n-estimators 5000` là số vòng boosting tối đa mặc định cho mỗi horizon;
`--num-epochs` chỉ dùng cho neural. Tree mặc định học residual với squared error,
không dùng tail/correction loss của LSTM. `--target-mode direct` đổi sang học
mực nước tuyệt đối. Các trial chạy tuần tự; `--n-jobs` là số thread của tree.

- `--search-method grid`: thử đủ lưới mặc định (tree 2.016 tổ hợp; neural 1.440 tổ hợp, đã gồm 8 sequence length),
  hoặc giới hạn bằng `--n-trials`. Grid không pruning để so các tổ hợp đầy đủ.
- `--search-method tpe`: Optuna đề xuất tham số dựa trên kết quả trial trước;
  `MedianPruner` so tiến độ validation và dừng sớm trial kém. Mặc định đợi
  5 trial đầu; warmup 15 epoch với neural, hoặc 100 bước tiến độ của head cuối
  trong nhóm focus với tree. Tiến độ tree = vòng thực tế × min(1, learning_rate/0.05),
  để trial learning rate nhỏ không bị so quá sớm với trial học nhanh. Ví dụ lr=0.001
  thì 100 bước tương đương 5.000 vòng thực tế. Đây là quy đổi xấp xỉ, không bảo đảm
  tốc độ hội tụ ngang nhau. Early stopping vẫn hoạt động riêng: patience tree tăng
  theo cùng tỷ lệ, tối đa bằng giới hạn `--n-estimators`. Patience=50 ở lr=0.05
  sẽ thành 2.500 vòng ở lr=0.001. Có thể đổi bằng
  `--pruning-startup-trials`, `--pruning-warmup`, `--pruning-interval`, hoặc tắt
  bằng `--no-pruning`.
- TPE mặc định **100 trial**, trong đó 20 trial đầu lấy mẫu ngẫu nhiên để khảo sát;
  đổi số trial đầu bằng `--tpe-startup-trials`. Số này khác với số trial chờ của pruner.
- Sau search, console thông báo nếu trial tốt nhất hoặc phần lớn top 5 trial ở sát
  biên tìm kiếm. Với khoảng log, khoảng cách tới biên cũng tính theo log. Đây là dấu
  hiệu cần khảo sát thêm, không chứng minh tối ưu nằm ngoài biên. Mở rộng bằng study
  mới; không âm thầm thay space trong study đang chạy hoặc khi resume.
- `--search-space path.json`: tự định nghĩa tham số. Grid dùng danh sách, ví dụ
  `{"max_depth":[3,4],"learning_rate":[0.03,0.05]}`. TPE cũng nhận khoảng như
  `{"learning_rate":{"type":"float","low":0.01,"high":0.15,"log":true}}`.
  Các tham số không nằm trong space giữ mặc định. Tên neural dùng underscore
  như `lstm_hidden_size`, `lstm_lr`, `lstm_tail_penalty`; xem `default_space()`
  trong entry point để biết space mặc định.
- Bỏ `--sequence-length` để TPE tự tìm trong 12–168 giờ. Ghi `--sequence-length 96`
  sẽ cố định 96 giờ và bỏ sequence khỏi space mặc định. `--sequence-lengths 12 24 48 96 168`
  chỉ thử các giá trị được liệt kê; hai cờ CLI này không dùng cùng nhau.
- Với `--search-space`, có thể khai báo `"sequence_length":{"type":"int","low":12,"high":168}`
  cho TPE hoặc danh sách cho Grid. Custom space không chứa sequence và không có cờ CLI
  sẽ dùng cố định 96 giờ. Không khai báo sequence ở cả CLI và custom space.
- Mọi trial dùng cùng anchor/nhãn train và validation sau warmup của sequence lớn nhất
  trong space. Sequence ngắn dùng cửa sổ input ngắn hơn nhưng không được thêm mẫu.
  Scaler chỉ fit một lần trên train và dùng chung. Tree cắt đúng prefix lag mới nhất;
  neural dùng đúng cửa sổ tương ứng. Checkpoint/preprocessing lưu sequence thực của trial.
  Khi chấm test tùy chọn, vẫn dùng warmup chung đó. Vì vậy số mẫu có thể khác các báo cáo
  cũ dùng sequence 96; cần so model trên giao các anchor, không đối chiếu trực tiếp tổng lỗi.
- `--check-only`: chỉ kiểm tra space/train/val, không train hoặc tạo thư mục.
- `--resume`: tiếp tục đúng thư mục study. `--n-trials` là tổng ngân sách kể cả
  trial đã chạy, không phải số trial thêm. Search space mặc định mới cần thư mục
  study mới nếu trước đó đã chạy bằng space cũ. TPE khi resume không bảo đảm cùng
  chuỗi đề xuất như chạy liên tục. `--timeout` giới hạn thời gian gọi search,
  không ngắt cứng trial đang train.
- Kết quả trong đúng `--output-dir`: `trials.csv`, `best_trial.json`, cấu hình,
  SQLite Optuna local và checkpoint riêng từng trial. Không cần PostgreSQL,
  không upload model MLflow, không ghi đè B/production.
- Chỉ thêm `--evaluate-best` khi muốn chấm trial đã chọn trên test và so với
  B/production sau khi search kết thúc. Test không tham gia chọn tham số;
  test hiện đã xem nhiều lần vẫn chỉ là đánh giá khám phá.

## Phân bố thay đổi mực nước dự báo — 21/10/2026

Chạy lại ba artifact phục vụ trong `src/backend/prediction/model_configs.json`: production cũ,
Model B seed 42 và Hồng Thái + Bạch Mã + Phú Bài seed 42. Không train, không đổi checkpoint,
không ghi forecast vào DB. ΔH = dự báo H(t+h) − quan trắc H(t), không phải sai số dự báo.
Dấu dương/âm là tăng/giảm; TB |ΔH| đo độ lớn để tránh hai chiều triệt tiêu.

7.672 anchor chung: train 6.095 (19/11/2025 01:00–30/07/2026 23:00), validation 985
(04/08/2026 23:00–14/09/2026 23:00), test 592 (19/09/2026 23:00–14/10/2026 14:00), giờ VN.
Mỗi model chạy đủ +1/+3/+6/+12/+24h, tổng 115.080 dự báo. Giữ warmup 96 giờ riêng từng split,
lọc cùng anchor đủ năm nhãn và dùng scaler riêng, không fit lại. Dataset gốc production:
`train_data/1h_top4_pg_2026-10-15_val_aug01_sep15`; hai model mới dùng
`train_data/da_vien_1h_2026-10-15_rain_pool_o24`. Batch inference kiểm tra khớp single forward
(sai khác ≤0,002 mm). Test tái hiện RMSE/N200: cũ 87,03/97; B 75,78/48; ba trạm 77,04/50.

### Bảng chính: 1.577 anchor validation + test, đơn vị mm

| Horizon | Model | TB ΔH | Median | Min | Max | TB \|ΔH\| | % dương | Khoảng phổ biến (rộng 20 mm) | % trong khoảng |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| +1h | Production cũ | 0.84 | 0.82 | -53.52 | 61.91 | 11.42 | 52.57 | [0, 20) | 43.75 |
| +1h | Model B | 4.00 | 2.71 | -54.68 | 67.89 | 16.37 | 59.48 | [0, 20) | 40.84 |
| +1h | Ba trạm | 4.11 | 3.98 | -56.81 | 72.03 | 15.31 | 65.25 | [0, 20) | 49.90 |
| +3h | Production cũ | 5.07 | 6.08 | -167.81 | 185.22 | 36.30 | 55.10 | [-20, 0) | 20.99 |
| +3h | Model B | 10.27 | 8.07 | -164.42 | 193.08 | 45.91 | 59.29 | [0, 20) | 19.72 |
| +3h | Ba trạm | 14.36 | 12.84 | -162.16 | 207.90 | 45.55 | 64.17 | [0, 20) | 20.29 |
| +6h | Production cũ | 10.92 | 12.29 | -292.45 | 277.23 | 58.06 | 57.77 | [20, 40) | 12.05 |
| +6h | Model B | 13.86 | 13.81 | -280.61 | 283.71 | 72.25 | 58.40 | [-20, 0) | 11.10 |
| +6h | Ba trạm | 20.29 | 19.25 | -285.84 | 314.59 | 71.74 | 60.43 | [-20, 0) | 11.60 |
| +12h | Production cũ | 13.01 | 14.72 | -204.91 | 209.18 | 49.25 | 59.92 | [40, 60) | 13.63 |
| +12h | Model B | 14.96 | 23.09 | -162.00 | 158.28 | 51.40 | 63.22 | [60, 80) | 14.39 |
| +12h | Ba trạm | 23.88 | 25.19 | -181.23 | 189.29 | 55.94 | 65.31 | [0, 20) | 11.86 |
| +24h | Production cũ | 9.42 | 14.98 | -154.71 | 132.06 | 38.42 | 63.92 | [20, 40) | 20.86 |
| +24h | Model B | 20.47 | 27.52 | -146.25 | 158.18 | 41.29 | 77.49 | [20, 40) | 30.88 |
| +24h | Ba trạm | 31.79 | 36.38 | -133.21 | 182.46 | 47.30 | 77.49 | [20, 40) | 21.81 |

Các ΔH dự báo không có giá trị đúng 0 trong mẫu này: % âm = 100 − % dương.
Khoảng histogram là [trái, phải); mode phụ thuộc độ rộng/vị trí bin, không phải giá trị chính xác
xuất hiện nhiều nhất. Train là mô tả in-sample; không trộn với validation/test để đánh giá tổng quát.

### Đối chiếu độ lớn thực tế và nhóm biến động mạnh

Chọn theo quan trắc |H(t+h) − H(t)| >200 mm, riêng từng horizon, không chọn theo lỗi model.
Các model dùng cùng mốc. Các giờ/horizon chồng lấn không phải nhiều đợt độc lập.

| Horizon | N mạnh | TB \|ΔH thực tế\| | Cũ | B | Ba trạm |
|---|---:|---:|---:|---:|---:|
| +1h | 0 | — | — | — | — |
| +3h | 94 | 243.54 | 110.07 | 125.65 | 127.14 |
| +6h | 213 | 297.08 | 133.23 | 166.54 | 167.79 |
| +12h | 60 | 267.27 | 76.86 | 67.95 | 73.70 |
| +24h | 95 | 301.73 | 51.51 | 48.11 | 42.20 |

Những khoảng anchor biến động mạnh tập trung: 10–14/09, 25/09–02/10, 05–07/10
và 08–14/10/2026; đây là mốc đầu vào, actual nằm ở t+h. Không dùng nhãn tương lai này để chọn model online.

| Trường hợp | N | TB ΔH thực tế | Cũ | B | Ba trạm |
|---|---:|---:|---:|---:|---:|
| +3h, Tăng >200 mm | 51 | +258.33 | +110.75 | +132.23 | +133.94 |
| +3h, Giảm >200 mm | 43 | -225.99 | -109.25 | -117.85 | -119.06 |
| +6h, Tăng >200 mm | 107 | +294.56 | +129.27 | +173.39 | +172.88 |
| +6h, Giảm >200 mm | 106 | -299.62 | -134.95 | -158.93 | -162.24 |
| +12h, Tăng >200 mm | 29 | +259.64 | +32.22 | +44.10 | +51.70 |
| +12h, Giảm >200 mm | 31 | -274.41 | -94.69 | -77.12 | -88.71 |
| +24h, Tăng >200 mm | 63 | +293.07 | -5.63 | +21.76 | +21.40 |
| +24h, Giảm >200 mm | 32 | -318.79 | -75.95 | -52.43 | -44.11 |

### Diễn giải và giới hạn

- +3/+6h thường nhận đúng chiều khi nước đổi mạnh (B: 100%/98,59%), nhưng biên độ vẫn nhỏ.
- +12h mạnh: tỷ lệ đúng chiều cũ/B/ba trạm = 81,67%/88,33%/95,00%; +24h = 55,79%/74,74%/71,58%.
- +24h tăng mạnh: B dự báo trung bình +21,76 mm trong khi thực tế +293,07 mm; khi giảm mạnh:
  B −52,43 mm, thực tế −318,79 mm. Ba trạm không khắc phục được co biên độ +24h.
- Trên toàn validation/test, B và ba trạm dự báo tăng ở 77,49% mốc +24h, thực tế 50,54%.
  Riêng test, tỷ lệ tăng B/ba trạm = 54,05%/58,45%; không coi 77,49% là thiên lệch cố định mọi thời kỳ.
- TB |ΔH| thực tế +6h =97,51 mm, +24h =71,43 mm: không mặc định horizon xa phải đổi nhiều hơn.
- Những mẫu có ít nhất một model lỗi >200 mm, dùng cùng tập union: +12h có 53 mốc,
  TB |ΔH thực tế| 245,59 mm, dự báo cũ/B/ba trạm 59,04/38,19/45,35 mm; +24h có 81 mốc,
  thực tế 310,05 mm, dự báo 46,20/40,68/36,54 mm. Đây là chọn mẫu theo kết quả, chỉ mô tả hậu nghiệm.
- Không suy ra nguyên nhân do thiếu dữ liệu/loss/kiến trúc chỉ từ thống kê này; test đã xem nhiều lần.

### Thống kê toàn bộ 7.672 anchor (bao gồm train)

| Horizon | Model | TB ΔH | Median | Min | Max | TB \|ΔH\| | % dương | % \|ΔH\|≤20 mm |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| +1h | Production cũ | -0.88 | -1.16 | -64.03 | 63.18 | 8.88 | 43.27 | 87.28 |
| +1h | Model B | 1.05 | -0.16 | -55.20 | 67.89 | 10.48 | 49.09 | 82.55 |
| +1h | Ba trạm | 1.11 | 0.46 | -57.28 | 72.03 | 10.86 | 52.39 | 83.47 |
| +3h | Production cũ | -2.47 | -3.78 | -191.32 | 205.45 | 28.64 | 44.86 | 55.14 |
| +3h | Model B | 2.51 | -0.18 | -167.41 | 193.08 | 29.91 | 49.57 | 60.43 |
| +3h | Ba trạm | 4.88 | 2.29 | -162.16 | 208.64 | 32.55 | 53.81 | 51.99 |
| +6h | Production cũ | -1.90 | -3.20 | -313.80 | 360.99 | 46.45 | 47.28 | 35.35 |
| +6h | Model B | 2.75 | -0.26 | -280.61 | 319.04 | 48.72 | 49.73 | 39.17 |
| +6h | Ba trạm | 6.39 | 3.05 | -290.60 | 326.88 | 52.73 | 52.82 | 34.29 |
| +12h | Production cũ | -1.97 | -1.73 | -249.89 | 288.68 | 42.48 | 48.66 | 33.73 |
| +12h | Model B | 2.86 | 5.10 | -208.18 | 281.15 | 39.69 | 54.90 | 35.58 |
| +12h | Ba trạm | 5.90 | 7.52 | -262.99 | 347.69 | 45.10 | 56.56 | 32.06 |
| +24h | Production cũ | -3.34 | 0.39 | -277.83 | 285.03 | 31.45 | 50.52 | 46.30 |
| +24h | Model B | 2.16 | 2.22 | -268.83 | 170.70 | 25.71 | 53.38 | 57.26 |
| +24h | Ba trạm | 5.08 | 5.26 | -293.30 | 266.25 | 30.18 | 57.96 | 53.79 |

## Hai thí nghiệm train mở rộng — chuẩn bị ngày 21/10/2026

Mục tiêu: bổ sung các đợt biến động thật từ validation cũ vào train, rồi tách riêng
tác dụng của lấy mẫu ưu tiên. Giữ B seed 42 và production cũ làm tham chiếu đóng băng;
không ghi đè artifact hoặc đổi registry production.

| Trường hợp | Dataset | Cách lấy mẫu train |
|---|---|---|
| 1 — `uniform` | Train mở rộng | Shuffle thông thường, không lặp mẫu trong epoch |
| 2 — `event-balanced` | Giống trường hợp 1 | Khoảng 80% tự nhiên +10% ưu tiên tăng mạnh +10% ưu tiên giảm mạnh |

Hai trường hợp train **từ đầu**, cùng seed, cùng scaler fit lại trên train mở rộng,
cùng số lượt lấy mẫu và số batch mỗi epoch. Giữ nguyên cấu hình B: ResidualLSTM,
Hồng Thái + Nhâm, 26 inputs, chuỗi 96 giờ, outputs +1/+3/+6/+12/+24h; hidden 32,
3 layers; dropout/input/head 0,7/0,2/0,2; AdamW lr 0,0009, weight decay 0,001;
correction penalty 0,1; tail penalty 0,5, threshold 200 mm, transition 20 mm;
horizon weights đều 1, batch 128, clip gradient 1, StepLR(10;0,9), tối đa 50 epochs,
patience 25; checkpoint chọn theo (N200 validation, RMSE), có persistence epoch 0.

### Dataset và ranh giới

`research.build_expanded_train_dataset` tạo dataset offline từ hourly pool, không
nối thẳng CSV. Khôi phục 24 giờ purge nội bộ giữa train/val cũ vì nay cùng thuộc
train; giữ nguyên 24 giờ purge trước validation mới. Kiểm tra nhãn bằng lookup
quan trắc đúng t+h, không nội suy, không đổi predictor hoặc forecast vintage.

Dataset: `train_data/da_vien_1h_2026-10-15_expanded_train_o24`.

| Tập | Khoảng bản ghi đầu vào (giờ Việt Nam) | Rows | Chuỗi hợp lệ 96 giờ |
|---|---|---:|---:|
| Train mới | 15/11/2025 02:00–14/09/2026 23:00 | 7.294 | 7.199 |
| Validation mới = test cũ | 16/09/2026 00:00–15/10/2026 14:00 | 711 | 592 |

Không tạo `test.csv`: **không còn test độc lập**. Validation mới đã xem nhiều lần
và được dùng early stopping/chọn checkpoint; mọi so sánh đều là khám phá. Giữ
cùng 592 anchor trước đây, từ 19/09 23:00 đến 14/10 14:00, để so các model.
`reference_validation.csv` giữ nguồn quan trắc nguyên bản của production cũ,
tránh thiếu các feature trạm 750003/750011 khi so với rain pool.

### Quy tắc lấy mẫu ưu tiên

- Chỉ tính nhãn nhóm từ các **anchor train hợp lệ**. Chọn ΔH có độ lớn lớn hơn
  giữa +12/+24h; nếu bằng nhau dùng +12h. ΔH >200 mm là tăng mạnh, <−200 mm là
  giảm mạnh. Nhãn nhóm không trở thành feature và không cần khi inference.
- Gộp anchor mạnh thành nhóm thời gian khi khoảng cách ≤12 giờ; đây là quy ước
  nhóm mẫu, không khẳng định các nhóm là những trận mưa/lũ độc lập.
- Trong mỗi nhóm tăng/giảm: chọn nhóm thời gian đều nhau rồi chọn anchor đều nhau.
  Thực thi bằng trọng số nghịch số anchor của nhóm; không để nhóm dài chiếm hết.
- Mỗi epoch vẫn 7.199 lượt: 5.761 tự nhiên không hoàn lại, 719 tăng và 719 giảm
  có hoàn lại. Shuffle toàn bộ rồi chia batch, nên tỷ lệ từng batch chỉ xấp xỉ.
  Phần tự nhiên vẫn có thể chứa mẫu mạnh. Train mới có 409 anchor mạnh
  (179 tăng, 230 giảm), nên tỷ lệ mạnh kỳ vọng khoảng 24,5%, không chỉ 20%.
- Không thay chuỗi 96 giờ, không ghép/sinh dữ liệu và không oversample validation.
  Thiếu nhóm được yêu cầu thì báo lỗi, không âm thầm đổi tỷ lệ thí nghiệm.

### Lệnh tái lập

Tạo dataset một lần; output đã tồn tại sẽ bị từ chối để tránh ghi đè:

```bash
.venv/bin/python -m research.build_expanded_train_dataset

.venv/bin/python -m research.expanded_train_experiment \
  --variant uniform --check-only
.venv/bin/python -m research.expanded_train_experiment \
  --variant event-balanced --check-only
```

Chạy trường hợp 1 trước:

```bash
.venv/bin/python -m research.expanded_train_experiment \
  --variant uniform \
  --device cuda --seed 42 --num-epochs 50 --early-stopping 25 \
  --output-dir model/experiments/expanded_train_20261021 \
  --mlflow-ex-name da_vien_expanded_train
```

Trường hợp 2, giữ nguyên các tham số còn lại:

```bash
.venv/bin/python -m research.expanded_train_experiment \
  --variant event-balanced \
  --device cuda --seed 42 --num-epochs 50 --early-stopping 25 \
  --output-dir model/experiments/expanded_train_20261021 \
  --mlflow-ex-name da_vien_expanded_train
```

Các giá trị mặc định theo B và có args để thay đổi. Có thể chọn tổ hợp bằng
`--rain-stations`, nhưng hai trường hợp đầu giữ mặc định `hong_thai+nham` để cô lập
tác dụng. Mỗi lượt chạy tự có run name riêng, không cần đổi tên thủ công.
Output tách `<combination>/<variant>/seed_<seed>/<run_name>/`, không resume/ghi đè.
MLflow mặc định localhost:5000, có `--mlflow-tracking-uri`
hoặc `--no-mlflow`; chỉ log settings/metrics, không upload model hay bảng artifact.
Cuối run reload checkpoint và so với B/cũ trên cùng anchor, đủ năm horizon;
log N200, RMSE, độ lớn ΔH và độ lớn ΔH ở nhóm biến động mạnh dưới nhãn validation.

`--checkpoint-metric` chọn cả checkpoint và tiêu chí early stopping:
`large_error_rate` (mặc định, ưu tiên N200 rồi RMSE), `rmse` (RMSE gộp cả năm
horizon bằng mm, không ưu tiên N200), hoặc `val_loss` (HorizonWeightedTailLoss).
Các metric còn lại vẫn được log đầy đủ. Run name chứa sequence length, hidden size,
tiêu chí checkpoint, timestamp UTC và mã ngẫu nhiên ngắn. Muốn so sampler thì giữ
cùng tiêu chí hai nhánh.

Các hyperparameter đều được truyền từ CLI vào dataset/model/optimizer/loss thực tế,
không chỉ ghi vào log. Các cờ `--lstm-*` tương ứng của `src.ai.train` cũng được nhận
làm alias. Danh sách cờ chính và mặc định:

| Args | Mặc định |
|---|---|
| `--sequence-length` | 96 |
| `--hidden-size`, `--num-layers` | 32; 3 |
| `--dropout`, `--input-dropout`, `--head-dropout` | 0.7; 0.2; 0.2 |
| `--learning-rate`, `--weight-decay` | 0.0009; 0.001 |
| `--correction-penalty` | 0.1 |
| `--horizon-weights` | 1 1 1 1 1, theo +1/+3/+6/+12/+24h |
| `--tail-penalty`, `--tail-threshold-mm`, `--tail-transition-mm` | 0.5; 200; 20 |
| `--batch-size`, `--num-worker`, `--gradient-accumulation`, `--gradient-clip` | 128; 0; 1; 1.0 |
| `--scheduler`, `--scheduler-step-size`, `--scheduler-gamma` | step; 10; 0.9 |

`--scheduler none` tắt lịch LR; `--tail-penalty 0` dùng weighted MSE không có tail.
Thay ngưỡng tail không thay định nghĩa N200 trong checkpoint/report (vẫn >200 mm).
Input size tự tính từ feature của tổ hợp trạm; outputs vẫn là cả năm horizon.
Đổi sequence length sẽ đổi số anchor hợp lệ; báo cáo dùng giao các anchor với
hai model tham chiếu. Có thể chạy lại cùng command; mỗi lần tạo thư mục mới tự động.

`--output-dir` và `--report-dir` đều tùy chọn, nay là **thư mục gốc** có thể tái sử dụng.
Mặc định lần lượt là `model/experiments/expanded_train` và
`reports/model_comparisons/expanded_train`. Trong cả hai, các thư mục con chứa cùng
`<combination>/<variant>/seed_<seed>/<run_name>` để đối chiếu dễ dàng. Các thư mục
và checkpoint cũ giữ nguyên. MLflow cũng dùng tên riêng mỗi run; terminal in đường
dẫn trước khi train. `--check-only` chỉ in dự kiến, không tạo thư mục.

Đã check-only cả hai nhánh, chạy smoke 1 epoch CPU mỗi nhánh với checkpoint ở
`/tmp`, kiểm tra reload và so sánh tham chiếu end-to-end. Smoke không đánh giá
hiệu quả thí nghiệm; chưa chạy các run đầy đủ hoặc thay model production.

### Báo cáo tự động sau mỗi lượt train

`expanded_train_experiment` tự reload checkpoint tốt nhất và tạo báo cáo ba model:
production cũ, Model B và candidate. Lệnh train bên trên không cần thêm cờ.
Output mặc định:
`reports/model_comparisons/expanded_train/<combination>/<variant>/seed_<seed>/<run_name>/`.
Có thể chọn thư mục gốc bằng `--report-dir`; mỗi lượt luôn tạo thư mục run mới.

- `comparison.md`: tổng hợp RMSE/MAE/N200, đủ năm horizon, chênh lệch với từng model
  tham chiếu, biên độ ΔH và nhóm biến động mạnh. Ghi rõ validation mới không phải test độc lập.
- `metrics.csv`: metric theo model/horizon và thống kê phân bố ΔH.
- `predictions.csv.gz`: cả ba model trên cùng anchor, actual, current, target time.
- `manifest.json`: seed/variant, epoch được chọn, dataset, model/version/scaler paths,
  feature order, thiết bị và cấu hình train nếu báo cáo được tạo trong run mới.

Các file này chỉ lưu local; không upload model/báo cáo lên MLflow. Model tham chiếu
không bị sửa, không dùng chung scaler với candidate. CPU/GPU có thể lệch số lẻ nhỏ.

Lượt uniform seed 42 đã chạy xong trước khi thêm tính năng này: epoch 43, RMSE
76,87 mm, N200=51; B 75,78/48; production cũ 87,03/97. Báo cáo đã tạo tại
[uniform seed 42](../reports/model_comparisons/expanded_train/nham+hong_thai/uniform/seed_42/044689eeb379414a82a15f73eabb68c8/comparison.md).
Nó gần B nhưng chưa tốt hơn theo hai chỉ số tổng hợp; cần xem nhánh lấy mẫu ưu tiên.

Tạo báo cáo từ checkpoint có sẵn, **không train lại**:

```bash
.venv/bin/python -m research.expanded_train_experiment \
  --variant uniform --seed 42 --device cpu \
  --report-only --best-epoch 43 \
  --checkpoint 'model/experiments/expanded_train_20261021/nham+hong_thai/uniform/seed_42/1h/ResidualLSTM_water_level+rain+flood_season+weather_forecast_best.pt'
```

`--best-epoch` chỉ ghi metadata đã biết, không chọn checkpoint; nếu không có thì
bỏ cờ, báo cáo ghi chưa có metadata. Nhánh report-only dùng scaler cạnh checkpoint,
không tạo checkpoint mới, không ghi MLflow và không yêu cầu output train mới.
Process train đã khởi động trước thay đổi code không tự nhận tính năng mới;
nếu đang chạy nhánh thứ hai thì để nó chạy xong rồi dùng report-only.
