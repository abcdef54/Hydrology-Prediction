# Regional direct +1h…+24h output

This uses the unchanged 11-station pool `train_data/hue_regional_20261015_h6_h12_fixed_split_v1`. No original artifact is rewritten. The existing regional dataset selection constructs a **view** on demand: `--outputs 24` is an alias for `--target-horizons 1 2 ... 24`. Every requested label is an exact lookup at issue time + horizon in `water.parquet`, never interpolation or recursive predictions. The 24 outputs are a single forward pass. Missing targets exclude the whole anchor from that O24 view; saved input grid, sequence length, features, split membership, 24h purge and weather availability policy do not change.

The original train-fitted feature scalers and the original +6h/+12h target scalers are retained. New per-station/per-horizon target means and scales are fitted only on selected training anchors; station-specific persistence affine coefficients are reconstructed for all 24 outputs. Validation and test reuse those training statistics. The original h6/h12 scalers were fitted on the original train pool: retain them deliberately for preprocessing parity, even if missing 24-hour targets cause the O24 train pool to be smaller on a different dataset. The station-balanced loss uses current train station counts and 24 equal horizon weights. Predictions use the existing residual direct multi-output heads; two-output weights cannot be loaded as 24-output checkpoint weights. This prepares training code and configurations; it does not assert an O24 checkpoint exists.

Run the no-training real-data audit from repository root:

```bash
.venv/bin/python -m research.scripts.check_regional_o24
.venv/bin/python -m unittest src.ai.tests.test_regional_training_selection src.ai.tests.test_evaluation_summary -v
```

The audit checks train-only target moments, original feature schema and feature normalizers, saved memberships, chronology and 24h purge, every exact target, shared train/validation/test normalizers, and finite 24-output forward/loss for both *new, untrained* model instances. It does not run an optimizer or write a checkpoint. On the checked dataset the O24 view retains 62,500 training anchors, 11,880 validation anchors and 7,821 test anchors; 132 of 7,953 original test anchors lack the complete target range. These are station-anchor counts. For fair O2 vs O24 metrics, intersect samples and compare identical horizons; the larger O2 test pool is not identical. Saved station test CSVs include MAE/RMSE, strict error counts >200 mm, and persistence MAE/RMSE for each horizon and all horizons.

## Optional training commands — not executed by the audit

Commands reproduce the recorded model configurations apart from 24 rather than 2 output horizons. They start training and require a reachable MLflow tracking server; run sequentially on a shared GPU. Use new checkpoint directories. Original R2 was run with test skipped; R1 was not. Test is historically inspected, not an untouched holdout. Historical forecast publication times and observations' issue-time availability are not independently established.

GRU R2 (seed 17):

```bash
.venv/bin/python -m src.ai.train \
  --mlflow-ex-name regional_o24_gru --dataset-dir train_data/hue_regional_20261015_h6_h12_fixed_split_v1 \
  --train-method gru-late-fusion --horizon 1h --outputs 24 --regional-fold fixed \
  --feature-set water_level+rain+flood_season+weather_forecast --rain-stations nham+hong_thai \
  --time-features none --lstm-seq-len 96 --head-mode station --station-embedding-size 8 --forecast-hidden-size 32 \
  --lstm-hidden-size 64 --lstm-num-layers 3 --lstm-drop-out 0.3 --lstm-input-dropout 0.2 --lstm-head-dropout 0.2 \
  --seed 17 --batch-size 256 --num-epochs 20 --early-stopping 30 --gradient-accumulation 1 \
  --lstm-lr 0.0009 --lstm-weight-decay 0.001 --lr-scheduler step --lstm-checkpoint-metric val_loss \
  --change-loss station-balanced --change-scale-floor-mm 200 --change-event-weight 0 --change-under-weight 0 \
  --diagnose-change-dynamics --diagnose-station-gradients --gradient-probe-samples 32 --skip-test \
  --checkpoint-dir model/experiments/regional_o24_gru_r2
```

Transformer R1 (seed 42):

```bash
.venv/bin/python -m src.ai.train \
  --mlflow-ex-name regional_o24_transformer --dataset-dir train_data/hue_regional_20261015_h6_h12_fixed_split_v1 \
  --train-method transformer-late-fusion --horizon 1h --outputs 24 --regional-fold fixed \
  --feature-set water_level+rain+flood_season+weather_forecast --rain-stations hong_thai+bach_ma+phu_bai \
  --time-features all --lstm-seq-len 96 --head-mode station-mlp --station-mlp-hidden-size 32 \
  --station-embedding-size 8 --forecast-hidden-size 32 --transformer-heads 4 --transformer-ff-size 128 \
  --lstm-hidden-size 32 --lstm-num-layers 3 --lstm-drop-out 0.3 --lstm-input-dropout 0.2 --lstm-head-dropout 0.2 \
  --seed 42 --batch-size 512 --num-epochs 60 --early-stopping 30 --gradient-accumulation 1 \
  --lstm-lr 0.0009 --lstm-weight-decay 0.001 --lr-scheduler step --lstm-checkpoint-metric val_loss \
  --change-loss station-balanced --change-scale-floor-mm 200 --change-event-weight 0 --change-under-weight 0 \
  --diagnose-change-dynamics --diagnose-station-gradients --gradient-probe-samples 32 \
  --checkpoint-dir model/experiments/regional_o24_transformer_r1
```

## DILATE objective (opt-in)

Select `--change-loss dilate --dilate-alpha 0.5 --dilate-gamma 0.01`. The sequence shape component is Soft-DTW; temporal distortion is the expected soft alignment path weighted by `(i-j)^2/24^2`. The loss is `alpha*shape + (1-alpha)*temporal`; shape is raw Soft-DTW and may be negative. Following the existing station-balanced objective, each physical level is represented relative to current water level and normalized by the corresponding station/horizon `max(target_scale, 200 mm)`; station sample weights are computed from training counts. Gamma is in squared dimensionless normalized units. It is not a 200 mm error-count loss.

This research implementation uses batched PyTorch dynamic programming with quadratic sequence-length work and a differentiable temporal path. Benchmark memory and runtime before any long training. `--diagnose-change-dynamics`, event-aware heads and custom horizon weights are rejected because DILATE's alignment couples all horizons. Existing per-horizon change diagnostics describe errors, not additive DILATE contributions. The existing training process logs the parameters and stores the loss configuration alongside the checkpoint. GPU testing requires an available CUDA device.

Example (starts training only if executed): take the GRU command above, remove `--diagnose-change-dynamics --diagnose-station-gradients --gradient-probe-samples 32`, replace `--change-loss station-balanced` with `--change-loss dilate --dilate-alpha 0.5 --dilate-gamma 0.01`, and choose a new `--checkpoint-dir`. Add `--mlflow-tracking-uri "sqlite:///$PWD/mlflow_local.db"` if no tracking server is running.
