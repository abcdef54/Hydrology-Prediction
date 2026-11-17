#!/usr/bin/env bash
# Controlled comparison with the 11-station two-output Transformer baseline.
set -euo pipefail
cd "$(dirname "$0")/../.."

if pgrep -f '[p]ython.*-m src.ai.train' >/dev/null; then
  printf '%s\n' 'Another training process is running; leaving the GPU alone.' >&2
  exit 2
fi

run_id="$(date -u +%Y%m%dT%H%M%S%NZ)"
exec .venv/bin/python -m src.ai.train \
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
  --head-mode station-mlp --station-mlp-hidden-size 32 --horizon-decoder none \
  --lstm-lr 0.0009 --lstm-weight-decay 0.001 \
  --lstm-correction-penalty 0 --lstm-horizon-weights 1 1 \
  --lstm-tail-penalty 0 --lstm-tail-threshold-mm 200 \
  --lstm-tail-transition-mm 20 --lstm-checkpoint-metric val_loss \
  --change-loss large-error-count --change-scale-floor-mm 200 \
  --change-event-weight 0 --change-under-weight 0 --change-threshold-mm 200 \
  --large-error-penalty 0.2 --large-error-threshold-mm 200 \
  --large-error-transition-mm 20 \
  --diagnose-change-dynamics --dynamics-every 1 \
  --diagnose-station-gradients --gradient-probe-samples 32 \
  --device cuda \
  --checkpoint-dir "model/experiments/regional_large_error_count/${run_id}" \
  --mlflow-tracking-uri http://localhost:5000 \
  --mlflow-ex-name regional_large_error_count
