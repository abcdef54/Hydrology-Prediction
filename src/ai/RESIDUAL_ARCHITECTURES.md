# Residual sequence architectures

All models predict corrections to persistence in normalized target units. They
default to two outputs (+6h/+12h), support configurable regional target horizons,
and share the same losses, chronological validation,
checkpoint metrics, feature selections, and train-fitted preprocessing.

| `--train-method` | History encoder | Forecast fusion |
|---|---|---|
| `lstm` | Shared residual wrapper with an LSTM encoder | Early |
| `residual_lstm` | Original local ResidualLSTM path | Original policy |
| `gru` | Stacked unidirectional GRU | Early |
| `tcn` | Causal dilated residual convolution blocks | Early |
| `cnn-gru` | Causal temporal CNN followed by a unidirectional GRU | Early |
| `gru-late-fusion` | Stacked GRU | Anchor forecast MLP |
| `tcn-late-fusion` | Causal TCN | Anchor forecast MLP |
| `cnn-gru-late-fusion` | Causal CNN + GRU | Anchor forecast MLP |
| `transformer` | Causal Transformer encoder with sinusoidal positions | Early |
| `transformer-late-fusion` | Causal Transformer encoder with sinusoidal positions | Anchor forecast MLP |

`lstm` now names the requested residual baseline in the common architecture
wrapper, including the same nonlinear fusion head and anchor-only policy as
the new models. The previous non-residual CLI
option remains available as `vanilla_lstm`. The existing `residual_lstm` local
path, `LSTM`/`ResidualLSTM` classes and their state-dict layouts are unchanged;
old checkpoints still load without new metadata. Regional `lstm` additionally
uses station conditioning; it is not a legacy local
checkpoint with extra weights silently attached.

## Fusion and causality

Early Fusion passes the selected feature sequence to the history encoder.
For new models, ECMWF is available only at the last timestep; earlier forecast
values/provenance are neutralized and their explicit missing flags are set.

Late Fusion removes **all** forecast columns from the history encoder,
including rainfall totals, masks, age and provenance. Only the forecast vector
at the prediction anchor enters a separate two-layer MLP. The history encoding,
forecast encoding and optional station embedding are concatenated before a
shared nonlinear fusion projection and final correction head. Nonlinear fusion
allows station identity/forecast to modulate the history response rather than
contributing only an additive station bias.

When forecast is disabled, the forecast encoder is not instantiated and the
fusion input width shrinks. It is not fed a vector of pretend zero rainfall.
Without forecast, Early/Late variants reduce to the same functional design;
these are controls rather than independent sources of information.

TCN blocks contain two left-padded convolutions, activations/dropout, and an
identity or 1×1 residual projection. Dilations are `1,2,4,...`. There is no
normalization pooling across future timesteps, bidirectional recurrence,
centered padding, or look-ahead pooling. The default five blocks with kernel 3
have a receptive field of 125 timesteps, covering the 96-hour input. CNN-GRU
uses causal convolutions with dilations `1,2,...`, then a GRU over all timesteps.

The final correction layer starts at zero, so a fresh model equals persistence
exactly. Station-specific coefficients are indexed by integer station IDs:

```text
prediction = normalized_current_water * persistence_scale[station]
             + persistence_offset[station] + learned_correction
```

Station embeddings enter the fusion head in both new Early/Late variants.
There are no station embeddings in a single-station model. For regional tail
loss and validation metrics, physical errors use each sample's station/horizon
target scales, rather than a pooled scaler. Test output includes per-station
metrics and pooled RMSE/MAE/large errors; pooled peaks/NSE are not calculated
across unrelated stations.

## Data selection and CLI

The existing CSV datasets remain supported. A dataset containing
`samples.parquet` is recognized as a prepared regional pool. The adapter does
not rebuild it or refit its normalizers. All stations share the selected rain
and weather channels. Examples below are usage references, not completed full
training experiments.

```bash
.venv/bin/python -m src.ai.train \
  --dataset-dir train_data/hue_regional_20261015_h6_h12_fixed_split_v1 \
  --train-method gru-late-fusion \
  --horizon 1h \
  --target-horizons 6 12 \
  --feature-set water_level+rain+flood_season+weather_forecast \
  --rain-station nham+bach_ma+ho_hoa_my+hong_thai \
  --use-weather-forecast true \
  --time-features all \
  --seed 42 \
  --batch-size 128 \
  --num-worker 0 \
  --num-epochs 50 \
  --early-stopping 25 \
  --gradient-accumulation 1 \
  --lstm-seq-len 96 \
  --hidden-size 32 \
  --num-layers 3 \
  --dropout 0.7 \
  --lstm-input-dropout 0.2 \
  --lstm-head-dropout 0.2 \
  --lstm-lr 0.0009 \
  --lstm-weight-decay 0.001 \
  --lstm-correction-penalty 0.1 \
  --lstm-horizon-weights 1 1 \
  --lstm-tail-penalty 0.5 \
  --lstm-tail-threshold-mm 200 \
  --lstm-tail-transition-mm 20 \
  --lstm-checkpoint-metric large_error_rate \
  --station-embedding-size 8 \
  --forecast-hidden-size 32 \
  --tcn-channels 32 32 32 32 32 \
  --tcn-kernel-size 3 \
  --cnn-channels 32 32 \
  --cnn-kernel-size 3 \
  --device cuda \
  --checkpoint-dir model/experiments/regional_architectures/run_001 \
  --mlflow-ex-name regional_architectures
```

- Change only `--train-method` for an architecture comparison. Preserve data
  selection, fold, seed, losses and validation criterion. Model parameter counts
  differ; equal hyperparameters do not imply equal capacity or an optimal setup.
- Add `--water-station 0074647404` to train only Tứ Phú from the same regional
  pool, with that station's saved normalization and no station embedding.
- `--use-weather-forecast false` removes the weather group. `--rain-station none`
  selects no observed rain. `--rain-stations` is an alias of `--rain-station`.
- Regional feature sets retain `water_level+flood_season`; rain and weather
  groups may be omitted. An explicit weather boolean overrides the weather
  group. Calendar selection uses `--time-features none/hour/day_of_year/all`.
- The corrected dataset saves one fixed train/validation/test split, with
  validation July 31–September 14 and test September 15–October 15. Omit
  `--regional-fold`; the loader uses the dataset's saved split and train-only
  normalizers. That flag remains only for older research datasets.
- `--hidden-size`, `--num-layers`, `--dropout` are aliases of the existing LSTM
  options. TCN block widths/count and CNN widths/count are controlled separately
  by their channel lists; TCN does not interpret recurrent layer count as blocks.
- New-model checkpoint directories include the method name, preventing Early
  and Late variants from overwriting one another within a run root.
- `--mlflow-tracking-uri` can override the existing default tracking server.

The post-cutoff regional split stays labelled exploratory. Historical forecast
availability remains an EDA2 research assumption; the architecture does not
convert hindcasts into verified operational forecasts. Model B and production
registrations are not changed.

## Training throughput

The regional adapter caches compact sample metadata and standardized targets,
then constructs sequence windows with batched array indexing. It does not
materialize the entire overlapping `[samples,96,features]` tensor. Rain/weather
masks, normalization, station mappings and the sequence order are unchanged.

Start with `--num-worker 0` on the current WSL laptop. The optimized path no
longer needs workers to hide pandas row construction. If using workers, they
are persistent across epochs, with CUDA pinned memory and nonblocking transfers.
The first epoch can still be expensive under Python/WSL multiprocessing; compare
both startup and steady epoch time rather than increasing workers blindly.

Static station IDs are validated once on CPU before the trainer's trusted fast
path; public model/inference calls retain range validation. Zero correction
penalty skips its unused computation, and metric reductions stay on device
until epoch end. No AMP, precision, model, loss, dataset or checkpoint-format
change is required.

## Checkpoints and inference

Each checkpoint retains raw tensor weights in `.pt` and gains:

- `.architecture.json`: family, fusion, sizes, convolution parameters, forecast
  indices/mask values, embedding dimensions and persistence tables.
- `.preprocessing.joblib`: existing scalers/feature order plus `model_spec`.
  Regional artifacts include the station mapping, fold normalizers, feature
  selections and source dataset-manifest hash.

`src.ai.comparison.load_candidate` restores new single-station checkpoints from
this metadata; its legacy loading path is retained. The existing backend
`ResidualLSTMRuntime` also accepts the additional residual model-type identifiers for
single-station artifacts that satisfy its feature/preprocessing contract.
No new model has been registered or deployed.

For prepared single-station or regional tensors:

```python
from src.ai.sequence_checkpoint import SequenceRuntime

runtime = SequenceRuntime("path/to/checkpoint_best.pt", device="cpu")
prediction_mm = runtime.predict_mm(x)                   # single station
prediction_mm = runtime.predict_mm(x, station_ids=ids) # regional, IDs shaped [B]
```

Inputs must use saved preprocessing and feature order. Regional runtime uses
the saved station-specific inverse transform. Live regional API input assembly
and registration are separate from prepared-tensor inference; the current live
app preprocessor is still a single-station contract.

## Verification without full training

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  -m unittest src.ai.tests.test_residual_architectures -v

OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  -m research.scripts.smoke_residual_architectures \
  --output-dir model/experiments/new_architecture_smoke \
  --include-station-specific
```

The smoke matrix runs one epoch on at most 22 samples per split for each case:
eight models × forecast on/off × local CSV/regional pool/single station from the
regional pool = 48 cases. It uses real data, real gradients, validation,
checkpoint selection, save/load and physical-unit inference checks. MLflow log
calls are mocked only in this smoke utility to avoid needing an external server.
These sampled scores must not be used to rank forecasting performance.
This optional real-data utility was not run for the Transformer extension.
The Transformer integration smoke tests instead generate all observations in a
temporary directory and run with `python -m unittest src.ai.tests.test_transformer`.

## Opt-in extreme-change objectives

Regional station-conditioned training accepts `--change-loss baseline|station-balanced|event-weighted|asymmetric|combined`.
Baseline remains the default. New modes require zero tail/correction penalties; they preserve model/scaler/checkpoint formats.
See [research, equations, metrics and full commands](../../reports/model_training/extreme_change_audit_20261028/README.md).
No real-data effectiveness claim is made; only synthetic smoke/regression tests were run for this change.

Validation prints pooled MAE alongside RMSE, in physical mm. MLflow records
`val/mae_mm` and `val/<target>/mae_mm`; station/horizon MAE remains available
in the change diagnostics tables. The `Validation >200 mm` count uses each
individual absolute prediction error, `abs(predicted_mm-observed_mm)>200`,
not an RMSE or MAE threshold. Exactly 200 mm is not counted. These extra metrics
do not change the training loss or checkpoint selection.

## Configurable regional history and targets

`--lstm-seq-len` selects a positive history length, including 48, 96 or longer
windows when sufficient history exists. `--target-horizons` selects distinct
positive integer hours, for example `3 6 12 18`. Horizons are ordered ascending;
provide horizon weights in that order. With four horizons, use
`--lstm-horizon-weights 1 1 1 1`, or omit weights for equal weighting.

These settings work with all residual encoders, Early/Late Fusion, and the
station-aware losses. Rain stations remain selectable independently, including
`--rain-station nham+hong_thai+bach_ma`. Native 96-hour/+6h/+12h behavior and
existing checkpoints remain compatible.

The regional training adapter uses the immutable pool's saved prediction anchors
and split membership. It changes each requested history window to end exactly at
the anchor, enforces the saved minimum water-history coverage, and drops samples
whose exact target timestamps are missing. It does not add new anchors just
because a shorter window or fewer targets would make them eligible. New labels
come from exact timestamps in the saved hourly `water.parquet`, without
interpolation. Existing input/target scalers are retained for saved channels;
additional horizon scalers are fitted only on eligible training samples and
shared with validation/test. A longer horizon extends the effective purge when
necessary. No dataset artifact is overwritten.

The actual history length, target columns, selected target scalers, persistence
coefficients and sample filtering policy are saved in checkpoint preprocessing.
Checkpoint paths include the selected sequence length. Sample counts can change
when history coverage, future labels or the required purge make samples invalid;
inspect the printed split counts before comparing experiments.

Examples of argument changes (keep the remaining training options as desired):

```bash
--lstm-seq-len 48 --target-horizons 6 12 --lstm-horizon-weights 1 1
```

```bash
--lstm-seq-len 48 --target-horizons 3 6 12 18 --lstm-horizon-weights 1 1 1 1
```

Only synthetic tests were run for this extension; no real-data training or
full-dataset inference was performed.

## Station-specific MLP experiment

For regional station-conditioned sequence models, `--head-mode station-mlp`
replaces the per-station linear correction with separate station MLPs. Each is
`shared feature width -> 32 -> 32 -> 32 -> output horizons`, with ReLU after
each hidden layer. Use `--station-mlp-hidden-size 64` to change the width of all
three hidden layers to 64; the default is 32. Its output layer is initialized to zero, preserving the
persistence prediction at initialization. The shared encoder, forecast fusion,
station embedding, and existing head dropout remain in place. This is opt-in;
`shared` and `station` retain their previous layouts and checkpoint metadata.

For an 11-station, 64-wide GRU with two outputs, these heads contain 46,838
parameters, replacing 1,430 in the linear station heads. Architecture metadata
records the mode and width, including when the default is used. Earlier station
MLP metadata without a width loads as 32. Nondefault widths use separate checkpoint
directories such as `head_station-mlp_width64`. This head currently requires
`--horizon-decoder none`. Compare it
with the existing linear station head using the same data, seed and settings;
it has not been established to improve real-data performance.

## Transformer experiment

`transformer-late-fusion` replaces the recurrent history encoder with a linear
input projection, sinusoidal positional encoding, and a stack of pre-norm
self-attention/feed-forward layers (GELU), followed by a final LayerNorm.
Each layer has independently initialized matrix weights. A causal mask prevents
position t from using later positions; the final history token can attend to the
whole observed window. Sequence length is dynamic, with no learned position-table
limit. The existing forecast MLP, station embedding, fusion head, persistence
scaling and zero-initialized correction head are retained.

The shared `--lstm-hidden-size`, `--lstm-num-layers`, `--lstm-drop-out`,
`--lstm-input-dropout` and `--lstm-head-dropout` flags also control this model.
`--transformer-heads` defaults to 4 and must divide hidden size;
`--transformer-ff-size` defaults to 128. Both are stored in architecture metadata
and logged with the training settings. Old checkpoints and default methods keep
their original architecture metadata and parameter layouts.

This is a small encoder experiment, not an implementation of a specialized
hydrological Transformer. The building blocks follow
[Attention Is All You Need](https://arxiv.org/abs/1706.03762) and
[PyTorch TransformerEncoderLayer](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html).
These sources establish the mechanism, not improved performance on this dataset.

Start with the B1 objective, two targets and the same settings as the existing
GRU late-fusion control. This command uses batch 256 and history 96; compare with
a GRU control using those same settings. Do not mix this comparison with auxiliary
3/18-hour targets, a different rain set, sequence length or loss.

```bash
.venv/bin/python -m src.ai.train \
  --dataset-dir train_data/hue_regional_20261015_h6_h12_fixed_split_v1 \
  --train-method transformer-late-fusion \
  --horizon 1h \
  --target-horizons 6 12 \
  --feature-set water_level+rain+flood_season+weather_forecast \
  --rain-station nham+hong_thai \
  --use-weather-forecast true \
  --time-features none \
  --regional-fold fixed \
  --seed 42 \
  --batch-size 256 \
  --num-worker 0 \
  --num-epochs 20 \
  --early-stopping 30 \
  --gradient-accumulation 1 \
  --lstm-seq-len 96 \
  --lstm-hidden-size 64 \
  --lstm-num-layers 3 \
  --transformer-heads 4 \
  --transformer-ff-size 128 \
  --lstm-drop-out 0.3 \
  --lstm-input-dropout 0.2 \
  --lstm-head-dropout 0.2 \
  --station-embedding-size 8 \
  --forecast-hidden-size 32 \
  --lstm-lr 0.0009 \
  --lstm-weight-decay 0.001 \
  --lstm-correction-penalty 0 \
  --lstm-horizon-weights 1 1 \
  --lstm-tail-penalty 0 \
  --lstm-tail-threshold-mm 200 \
  --lstm-tail-transition-mm 20 \
  --lstm-checkpoint-metric val_loss \
  --change-loss station-balanced \
  --change-scale-floor-mm 200 \
  --change-event-weight 0 \
  --change-under-weight 0 \
  --change-threshold-mm 200 \
  --device cuda \
  --checkpoint-dir "model/experiments/regional_b1_transformer/$(date -u +%Y%m%dT%H%M%S%NZ)" \
  --mlflow-tracking-uri http://localhost:5000 \
  --mlflow-ex-name regional_b1_architectures
```

Judge validation MAE/RMSE per station/horizon, persistence skill, errors >200 mm,
extreme-change amplitude and error, rise recall, false-alarm ratio and time per
epoch together. Attention does not guarantee that rare rises become predictable.
The repeatedly inspected test remains exploratory; final confirmation needs an
untouched holdout. Synthetic tests establish gradient flow, temporal masking,
forecast isolation, dynamic shapes and checkpoint round trips only. They do not
rank this model against GRU or Model B. No real-data training or inference was
run to introduce this architecture.
