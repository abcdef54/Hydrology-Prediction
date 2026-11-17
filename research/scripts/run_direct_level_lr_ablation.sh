#!/usr/bin/env bash
# The three requested learning-rate experiments, run sequentially.
set -euo pipefail
cd "$(dirname "$0")/../.."

run_tag="$(date -u +%Y%m%dT%H%M%S%NZ)"
log_root="reports/model_training/direct_level_lr_20261031/${run_tag}"
mkdir -p "$log_root"
printf 'Run directory: %s\n' "$log_root"

run_one() {
  local label="$1" rate="$2" scheduler="$3"
  if pgrep -f '[p]ython.*-m src.ai.train' >/dev/null; then
    printf '%s\n' 'Another training process is running; stopping before the next run.' >&2
    exit 2
  fi
  printf 'Starting experiment %s: lr=%s scheduler=%s\n' "$label" "$rate" "$scheduler"
  .venv/bin/python -m src.ai.train \
    --dataset-dir train_data/hue_regional_20261015_h6_h12_fixed_split_v1 \
    --train-method transformer-late-fusion \
    --horizon 1h --target-horizons 6 12 \
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
    --horizon-decoder none --prediction-mode direct \
    --lstm-lr "$rate" --lstm-weight-decay 0.001 \
    --lr-scheduler "$scheduler" --lr-warmup-epochs 3 --lr-min 0.000001 \
    --lstm-correction-penalty 0 --lstm-horizon-weights 1 1 \
    --lstm-tail-penalty 0 --lstm-tail-threshold-mm 200 \
    --lstm-tail-transition-mm 20 --lstm-checkpoint-metric val_loss \
    --change-loss station-balanced --change-scale-floor-mm 200 \
    --change-event-weight 0 --change-under-weight 0 --change-threshold-mm 200 \
    --diagnose-change-dynamics --dynamics-every 1 \
    --diagnose-station-gradients --gradient-probe-samples 32 \
    --device cuda \
    --checkpoint-dir "model/experiments/regional_direct_lr/${run_tag}/${label}" \
    --mlflow-tracking-uri http://localhost:5000 \
    --mlflow-ex-name regional_direct_lr \
    > "${log_root}/${label}.log" 2>&1
  printf 'Experiment %s finished.\n' "$label"
}

run_one A 0.0003 step
run_one B 0.0001 step
run_one C 0.0002 warmup-cosine
