#!/usr/bin/env bash
# Start the complete stack, choosing CUDA only when it works inside Docker.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# This entry point always chooses automatically, including with an older .env
# that still sets PREDICTION_DEVICE=cuda.
export PREDICTION_DEVICE=auto

COMPOSE=(docker compose -f docker-compose.yaml --profile ingestion)
GPU_COMPOSE=(docker compose -f docker-compose.yaml -f docker/compose.gpu.yaml --profile ingestion)

# Keep the CPU image available for ingestion and for GPU fallback.
"${COMPOSE[@]}" build

PROBE_LOG=$(mktemp)
trap 'rm -f "$PROBE_LOG"' EXIT

if docker run --rm --gpus 'device=0' --entrypoint nvidia-smi \
    hydrology-backend:local -L >"$PROBE_LOG" 2>&1; then
    "${GPU_COMPOSE[@]}" build backend
    if docker run --rm --gpus 'device=0' --entrypoint python \
        hydrology-backend:cuda -c \
        'from src.backend.prediction.device import resolve_prediction_device; raise SystemExit(0 if resolve_prediction_device() == "cuda" else 1)' \
        >"$PROBE_LOG" 2>&1; then
        echo "Docker CUDA probe passed; starting backend with GPU access."
        exec "${GPU_COMPOSE[@]}" up -d "$@"
    fi
fi

echo "Docker GPU unavailable; starting with CPU inference. Probe details:"
cat "$PROBE_LOG"
exec "${COMPOSE[@]}" up -d "$@"
