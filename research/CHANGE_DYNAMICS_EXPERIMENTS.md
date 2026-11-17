# Pipeline thử nghiệm phần 4: dự báo lớn đúng thời điểm

**Trạng thái: đã chuẩn bị code và lệnh, chưa chạy training.** Agent chỉ chạy unit
tests trên dữ liệu giả, không cập nhật weights qua optimizer. Người dùng thực hiện
toàn bộ training và đánh giá trên dữ liệu thực sau khi review.

Mục tiêu là phân biệt: chưa học được extreme ngay trên train, tổng quát hóa kém,
và đánh đổi khi nhiều trạm dùng chung mô hình. Chưa có kết luận rằng thay head,
time features hay event weight sẽ cải thiện kết quả thực tế.

## 1. Bắt đầu bằng một lần chẩn đoán D0

Bộ lệnh đã chuẩn bị ở
[prepared_experiments_20261030](../reports/model_training/checkpoint_dynamics_20261029/prepared_experiments_20261030/RUNBOOK.md).
Chỉ chạy D0 sau khi review; không chạy cả ma trận ngay:

```bash
bash reports/model_training/checkpoint_dynamics_20261029/prepared_experiments_20261030/commands/D0_s42.sh
```

D0 giữ cấu hình A: Transformer late fusion, history 96, +6/+12h, Nhâm/Hồng Thái,
không thêm hour/day features, seed 42, batch 256, hidden 64, 3 lớp, B1 F=200,
event/under weights 0, tail/correction penalty 0, 20 epoch và checkpoint val_loss.
Toàn bộ lệnh đều có `--skip-test`: không chạy test hay so sánh tự động sau training.
Phú Xuân và dữ liệu gốc được giữ nguyên.

Cờ mới dùng được trên regional station-conditioned models:

| CLI | Mặc định | Chức năng |
|---|---|---|
| `--diagnose-change-dynamics` | Tắt | Đánh giá train/validation ở eval mode, lưu snapshots và dự báo validation |
| `--dynamics-every N` | 1 | Chu kỳ chẩn đoán; luôn thêm epoch 0 và epoch cuối hoặc lúc early stop |
| `--diagnose-station-gradients` | Tắt | Đo gradient encoder trên các anchors train cố định; yêu cầu dynamics |
| `--gradient-probe-samples N` | 32 | Tối đa anchors mỗi trạm cho gradient probe, N ≥2 |
| `--head-mode shared\|station` | shared | Giữ head chung hoặc dùng projection correction riêng theo trạm |
| `--horizon-decoder none\|transformer` | none | Decoder attention dùng query riêng cho từng horizon; yêu cầu late fusion và station heads |
| `--skip-test` | Tắt | Bỏ qua test và post-training comparison; áp dụng sequence models |

Với `--horizon-decoder transformer`, toàn bộ chuỗi đầu ra của encoder là memory của
Transformer decoder. Query được học riêng theo thứ tự horizon và kết hợp với fusion
đã chứa forecast cùng station embedding; từng query tạo correction tương ứng qua
weights riêng của trạm. `--transformer-heads` và `--transformer-ff-size` điều khiển
decoder một lớp. Đầu ra cuối vẫn khởi tạo bằng 0 và cộng vào persistence. Cấu hình
này là thử nghiệm opt-in, cần so sánh với cùng kiến trúc không decoder và không chứng
minh cải thiện cho đến khi chạy đánh giá dữ liệu thực.

Các cờ dynamics hiện yêu cầu tail/correction penalty bằng 0. Khi tắt dynamics,
không thêm lượt đánh giá hay snapshot. Default head và cách chọn best checkpoint
giữ nguyên; checkpoint cũ vẫn đọc được. `station-balanced` nay cảnh báo rõ nếu
event/under weight khác 0 vì hai cờ này không tác động trong mode đó. MLflow ghi
riêng giá trị được yêu cầu và giá trị weight có hiệu lực.

## 2. Đầu ra để kiểm tra

Bên cạnh checkpoint chính có thư mục `<checkpoint_stem>.dynamics/`. Mỗi epoch
được chẩn đoán có một thư mục riêng `epoch_000/`, `epoch_001/`, ... gồm:

- `model.pt`, `.architecture.json`, `.preprocessing.joblib` và `.loss.json`
  nếu dùng ChangeRiskLoss: weights, kiến trúc và scaler/mapping của chính epoch đó.
  Có thể đọc bằng `SequenceRuntime` trên tensor đã được chuẩn hóa đúng.
- `train_eval_metrics.csv`, `val_eval_metrics.csv`: model ở `eval()` mode,
  tắt dropout, weights cố định. Có station/horizon, pooled theo horizon và pooled
  toàn bộ; nhóm `all`, `extreme`, `rise`, `fall`, `non_extreme`.
- `validation_predictions.parquet`: mỗi hàng là một sample/horizon, gồm sample ID,
  station ID, anchor/target UTC, mực nước hiện tại, nhãn, dự báo, thay đổi thực/dự báo,
  sai số có dấu và đóng góp objective. Khóa đối chiếu là `sample_id + horizon_hours`.
- `summary.json`: epoch, checkpoint tốt nhất đang được chọn, trạng thái hoàn tất,
  target horizons, head, time features, loss và fingerprint anchors/nhãn.
- Nếu bật gradient: `gradient_norms.csv`, `gradient_cosines.csv`; danh sách anchors
  cố định nằm trong `gradient_samples.csv` ở thư mục dynamics.

Định nghĩa nhóm dựa trên **thay đổi thực tế**: extreme khi |ΔH| >T, rise khi ΔH >T,
fall khi ΔH <−T, non_extreme gồm cả đúng ±T. T lấy từ change threshold hoặc ngưỡng
validation khi dùng baseline. Lỗi lớn dùng ngưỡng lỗi validation, tách biệt T.
Các nhãn này chỉ dùng để đánh giá/loss, không bổ sung làm input.

Metric gồm RMSE, MAE, bias có dấu, biên độ thực/dự báo, skill với persistence,
shortfall khi nước dâng, recall và false alarms, sai dấu, lỗi lớn và objective.
`wrong_sign_rate` chỉ đếm ΔH thực × ΔH dự báo <0; correction 0 không được gọi là sai dấu.
`rise_false_alarm_ratio` là false alarms / số dự báo dâng, còn false-positive rate
là false alarms / số trường hợp thực tế không dâng >T. Nhóm không có mẫu hoặc
mẫu số bằng 0 được ghi NaN, không biến thành 0. Các chỉ số recall/FAR chính cần đọc
trên nhóm `all`; phân nhóm theo nhãn có thể khiến một số ratio trở thành hiển nhiên.

`objective_contribution` đã gồm horizon weights và chia theo số anchors của split;
cộng station/horizon trong nhóm `all` khôi phục loss. Pooled RMSE tính từ tất cả
sai số, không phải macro mean. Loss train_eval cân theo samples; train_loss cũ
vẫn là số đo online có dropout, trung bình các batch, nên không đọc như cùng phép đo.
MLflow lưu metric mới dưới `dynamics/...`, có step bằng epoch.

## 3. Gradient và head riêng theo trạm

Gradient probe chỉ lấy **train**, không đọc sampler của lượt học. Mỗi trạm có tối
đa N/2 anchors extreme và phần còn lại non_extreme, chọn đều theo thứ tự thời gian.
Extreme anchor nghĩa là ít nhất một horizon có |ΔH| >T. Trọng số population/selected
khôi phục tỷ lệ nhóm gốc; gradient norm phản ánh ước lượng đóng góp vào loss hiện
tại, không phải một objective cố ý cân bằng số mẫu extreme.

Cosine chỉ đo trên tham số **encoder dùng chung**, khi tắt dropout. Probe không gọi
optimizer hoặc ghi vào `.grad`; RNG Python/NumPy/CPU/CUDA và chế độ từng module
được khôi phục. Cosine âm là tín hiệu cần điều tra, chưa chứng minh negative transfer.
Với GRU/LSTM, probe dùng đường native thay cho cuDNN để có backward khi model ở
eval mode; cờ cuDNN được khôi phục sau probe. Các unit tests trong lượt chuẩn bị
này chạy trên CPU; chưa kiểm chứng hiệu năng hoặc đường CUDA bằng dữ liệu thực.
Epoch 0 có thể có gradient encoder bằng 0 vì correction head khởi tạo bằng 0;
khi đó ghi `zero_gradient`, cosine NaN, không kết luận “không xung đột”.

`--head-mode station` chỉ thay projection correction cuối bằng một projection riêng
cho từng station. Encoder, forecast MLP, fusion và station embedding vẫn chia sẻ
như trước. Các projection khởi tạo bằng 0; shared components và RNG lúc khởi tạo
giống control. Số tham số tăng, nên đây không phải đối chứng ngang số tham số.
Metadata lưu head mode; đường dẫn thêm `head_station` để tránh lẫn với shared head.
Không tự đăng ký hay triển khai model lên app.

## 4. Các đối chứng sau D0

| Nhóm | Lệnh đã chuẩn bị | Chỉ thay đổi |
|---|---:|---|
| Diagnostic D0 | 1 | Thêm chẩn đoán, giữ cấu hình A |
| Calendar | 12 | none/hour/day_of_year/all × seeds 17/29/42 |
| Head | 6 | shared/station × seeds 17/29/42; bật gradient probe |
| Loss | 9 | B1 hoặc combined α=.25/.5, β=0 × seeds 17/29/42 |

Mỗi lệnh riêng, không có tự chuyển giai đoạn hay tự gọi cả ma trận. Đọc D0 trước
khi chọn nhóm phù hợp; các controls lặp lại là có chủ ý để so sánh trong cùng pipeline.
Không xếp hạng các loss khác nhau bằng giá trị val_loss trực tiếp.

Để tạo bộ lệnh vào thư mục **mới**, không chạy training:

```bash
.venv/bin/python -m research.scripts.prepare_change_experiments \
  --output-dir /tmp/change_experiments_review
```

Script từ chối ghi đè thư mục đã có. Từng command cũng từ chối dùng lại thư mục
results đã tồn tại. Khi cần chạy lại, tạo một bộ lệnh với output root mới.

Sau khi người dùng có kết quả:

```bash
.venv/bin/python -m research.scripts.summarize_change_experiments \
  --manifest reports/model_training/checkpoint_dynamics_20261029/prepared_experiments_20261030/manifest.json \
  --output reports/model_training/checkpoint_dynamics_20261029/prepared_experiments_20261030/summary.csv
```

Tổng hợp chỉ đọc metric đã lưu, không load weights/inference. `summary.csv` lấy
checkpoint được chọn theo objective của từng run; `.history.csv` giữ toàn bộ
epoch/trạm/nhóm để nhìn phần cải thiện và suy giảm. `.gradient_norms.csv` và
`.gradient_cosines.csv` gom các probe qua từng epoch, giữ nguyên station IDs.
Fingerprint cho biết anchor/nhãn
có khớp không. Nếu đổi `dynamics-every` khiến best epoch không có snapshot, summary
ghi `selected_epoch_not_saved`, không tự thay bằng epoch gần nhất. Trạng thái
`not_run`/`partial` giúp phân biệt chưa chạy và chưa hoàn tất.

Tiêu chí review là giảm lỗi extreme/shortfall và kiểm soát MAE non_extreme cùng
false alarms tại cả +6/+12, theo từng trạm. Ngưỡng đánh đổi vận hành phải được xác
định trước; pipeline không tự chọn ngưỡng hay dùng test để chọn cấu hình.

## 5. Chi phí và kiểm thử

Chẩn đoán đầy đủ thêm **một lượt train_eval và một lượt val_eval** ngoài lượt
validation chọn checkpoint, kèm I/O snapshots/Parquet và các gradient probe nhỏ.
Vì vậy thời gian/dung lượng tăng khi bật cờ; không có benchmark tốc độ trên dữ liệu
thực trong thay đổi này. Snapshots chứa weights và preprocessing, không chứa
optimizer/scheduler state để resume training.

Unit tests kiểm tra threshold strict, empty groups, inverse transform, weighting
gradient, snapshots/runtime, station heads, RNG/modes/.grad, train-sampler isolation,
CLI và ma trận lệnh. Không có lượt fit hay optimizer step trong các test mới:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MLFLOW_DISABLE_AGENT_HINT=1 \
.venv/bin/python -m unittest src.ai.tests.test_change_experiments -v
```
