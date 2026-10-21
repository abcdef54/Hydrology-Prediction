# Nghiên cứu và phân tích mô hình

Thư mục này lưu công cụ tạo dữ liệu và các notebook dùng trong quá trình phát
triển mô hình. Backend của ứng dụng nằm trong `src/backend/`; mã train và đánh
giá mô hình nằm trong `src/ai/`.

| File | Mục đích |
|---|---|
| [TAIL_LOSS.md](TAIL_LOSS.md) | Loss phạt quanh ngưỡng 200 mm, chọn checkpoint theo lỗi lớn và lệnh train |
| [OUTPUT_HORIZON_ABLATION.md](OUTPUT_HORIZON_ABLATION.md) | Dataset 24 target giờ, loss chính/phụ, command tự train O5/O24 và so B/production |
| [TREE_HORIZON_EXPERIMENT.md](TREE_HORIZON_EXPERIMENT.md) | XGBoost/LightGBM trên dataset mới, lịch sử96h, native checkpoint và so B/production theo lỗi >200mm |
| [weather_forecast/](weather_forecast/README.md) | Thu thập ECMWF Single Runs, lưu PostgreSQL và kiểm tra độ phủ; chưa tạo dataset |
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
