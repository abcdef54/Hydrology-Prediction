# Train the regional model on another machine

This guide is for the 11-station +6h/+12h experiments. The dataset is included in `train_data/hue_regional_20261015_h6_h12_fixed_split_v1/`; training reads its local Parquet files and does not require PostgreSQL, Docker, application configuration, pretrained weights or an existing MLflow server.

## Install

Use Python 3.10 or newer, Git, and a working NVIDIA driver. On Linux or WSL, from the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

If CUDA is unavailable, install a GPU-enabled PyTorch build appropriate for the machine and driver using the instructions at https://pytorch.org/get-started/locally/, then run the check again. `--device cuda` requires a working GPU-enabled PyTorch installation. Requirements include a pinned Git commit for the Google hydrology library, so Git and network access are needed for installation. Hardware, OS and package-version differences can affect reproducibility. The dataset manifest contains original source paths as provenance; the prebuilt dataset reader uses local files inside the dataset directory, not those original paths.

## Train the event-aware model

The following command trains experiment A, the event-aware residual Transformer with supervised rise classification. It writes a new timestamped checkpoint and logs locally to `mlflow.db` and MLflow's local artifact directory; no MLflow server needs to be running. It skips the repeatedly inspected test period and saves validation diagnostics. Do not run it if another job is already using the GPU.

```bash
.venv/bin/python -m src.ai.train \
  --dataset-dir train_data/hue_regional_20261015_h6_h12_fixed_split_v1 \
  --train-method transformer-late-fusion --horizon 1h --target-horizons 6 12 \
  --feature-set water_level+rain+flood_season+weather_forecast \
  --rain-station hong_thai+bach_ma+phu_bai --use-weather-forecast true \
  --time-features all --regional-fold fixed --seed 42 \
  --batch-size 512 --num-worker 0 --num-epochs 60 --early-stopping 30 \
  --gradient-accumulation 1 --lstm-seq-len 96 \
  --lstm-hidden-size 32 --lstm-num-layers 3 --transformer-heads 4 \
  --transformer-ff-size 128 --lstm-drop-out 0.3 \
  --lstm-input-dropout 0.2 --lstm-head-dropout 0.2 \
  --station-embedding-size 8 --forecast-hidden-size 32 \
  --head-mode station-mlp --station-mlp-hidden-size 32 \
  --prediction-mode residual --horizon-decoder none \
  --event-aware --rise-classification-weight 0.1 --rise-positive-weight 1 \
  --lstm-lr 0.0009 --lstm-weight-decay 0.001 \
  --lstm-correction-penalty 0 --lstm-horizon-weights 1 1 \
  --lstm-tail-penalty 0 --lstm-tail-threshold-mm 200 --lstm-tail-transition-mm 20 \
  --lstm-checkpoint-metric val_loss --change-loss station-balanced \
  --change-scale-floor-mm 200 --change-event-weight 0 --change-under-weight 0 \
  --change-threshold-mm 200 \
  --diagnose-change-dynamics --dynamics-every 1 \
  --diagnose-station-gradients --gradient-probe-samples 32 \
  --skip-test --device cuda \
  --checkpoint-dir "model/experiments/event_aware/A/$(date -u +%Y%m%dT%H%M%S%NZ)" \
  --mlflow-tracking-uri sqlite:///mlflow.db \
  --mlflow-ex-name regional_event_aware
```

The model uses the existing strict definition of a large rise: future level minus current level **>200 mm**. `--head-mode station-mlp` keeps the station-specific MLP base heads. The extra rise heads are shared and station-conditioned. Classification weight `0.1` is a starting point, not a validated optimum.

For other controlled variants, see `research/configs/event_aware_regional_variants.json`. Starting from the command above:

- **Baseline:** remove `--event-aware --rise-classification-weight 0.1 --rise-positive-weight 1`.
- **B:** use the baseline and add `--rise-event-weight 0.5` (or `1`, or `2`).
- **C:** retain the event-aware flags and add `--rise-event-weight 0.5` (or `1`, or `2`).

Use a separate `--checkpoint-dir` for each run. Checkpoint selection uses unweighted station-balanced regression validation loss for every variant. The original, repeatedly inspected test period is not an independent holdout. Classifier thresholds should be selected on validation only, then frozen before a genuinely new holdout.

To inspect local MLflow results on that machine:

```bash
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db
```

To evaluate a saved validation snapshot after training, run:

```bash
.venv/bin/python -m src.ai.event_evaluation \
  --predictions /path/to/validation_predictions.parquet \
  --output-dir /path/to/new/evaluation-directory
```

To run only the synthetic, non-training event-aware checks:

```bash
MPLCONFIGDIR=/tmp/hydrology-mpl .venv/bin/python -m unittest src.ai.tests.test_event_aware_no_training -v
```

These tests use synthetic tensors and do not run an optimizer step or training epoch. Dataset forecast availability is retrospective and assumes a release lag; the snapshot does not prove real-time availability of every historical forecast or observation.
