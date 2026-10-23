# Dã Viên Water-Level Forecasting

Water-level forecasting and hydrology monitoring for Huế, Vietnam. The application
predicts **hourly mean water levels** at **Dã Viên (`0074647403`)** at **+1, +3,
+6, +12 and +24 hours**, in **millimetres**.

The research objective is to reduce forecasts with **absolute error >200 mm**,
especially at +12/+24 hours during rapid rises and falls.

## Application

- Interactive station map, latest observations, hourly history and reservoir-operation data.
- Selectable production models, single-anchor forecasts and forecasts over a date range.
- Charts comparing forecasts with observations; missing values remain visible as gaps.
- Saved forecasts identified by model/pipeline version; edited inputs produce unsaved scenario forecasts.
- Scheduled observation and ECMWF rainfall-forecast ingestion, PostgreSQL storage and Redis caching.

**Stack:** PyTorch, FastAPI, SQLAlchemy/Alembic, PostgreSQL, Redis, React/Vite,
MapLibre and Nginx. Forecasting is currently enabled for Dã Viên; other stations
supply observations or appear in the monitoring interface.

## Production models

The [model registry](src/backend/prediction/model_configs.json) defines three
independent, selectable **ResidualLSTM** models. They are not an ensemble.
Each has its own checkpoint, fitted preprocessing and feature order in [model/](model/README.md).

| Model ID | Rain gauges | Inputs | ECMWF forecast | Selection |
|---|---|---:|---|---|
| `legacy` | Hồng Thái, Nhâm, Thị trấn A Lưới, Hồ A Lá | 29 | No | Default |
| `model_b` | Hồng Thái, Nhâm | 26 | Yes | Selectable |
| `three_station` | Hồng Thái, Bạch Mã, Hồ chứa nước Phú Bài | 32 | Yes | Selectable |

All three use **3 recurrent layers, hidden size 32 and 96 input timesteps**.
Legacy uses valid observed steps and permits time gaps; B and the three-station
model use a consecutive hourly grid with missing-value handling.

Features include current/lagged water levels, rainfall and rolling 3/6/12/24-hour
totals, missingness flags and flood-season indicators. Weather-enabled models
share an ECMWF grid associated with Nhâm, using rainfall over 0–3, 3–6, 6–12 and
12–24 future hours, missingness flags and forecast age.

## Recorded results

Comparison on **592 common anchors**, from **19 September 2026, 23:00** through
**14 October 2026, 14:00 (UTC+7)**: 5 horizons × 592 anchors = **2,960 forecasts**.
RMSE below is **pooled across all predictions**, not the mean of horizon RMSEs.

| Model | Deployment | RMSE (mm) | MAE (mm) | Errors >200 mm | Rate |
|---|---|---:|---:|---:|---:|
| Legacy | Production, default | 87.03 | 64.47 | 97 | 3.28% |
| Model B, seed 42 | Production | 75.78 | 57.00 | 48 | 1.62% |
| Three-station, seed 42 | Production | 77.04 | 58.11 | 50 | 1.69% |
| Expanded-train three-station, hidden size 16 | Experimental | **72.30** | **54.03** | **32** | **1.08%** |

Large errors by horizon:

| Horizon | Legacy | Model B | Production three-station | Expanded candidate |
|---|---:|---:|---:|---:|
| +1h | 0 | 0 | 0 | 0 |
| +3h | 0 | 0 | 0 | 0 |
| +6h | 41 | 5 | 10 | 10 |
| +12h | 25 | 16 | 11 | 7 |
| +24h | 31 | 27 | 29 | 15 |

The best expanded candidate uses uniform sampling, seed 42, correction penalty -0.2
and checkpoint epoch 86 (run `seq96_h16_large_error_rate_20261022T061755947891Z_b872679c`).
It is **not deployed**. Results are exploratory: this period has been examined
repeatedly and, for expanded training, is validation used for checkpoint selection.
There is **no independent test set for the expanded experiment**.

## Repository layout

```text
.
├── src/
│   ├── ai/
│   │   ├── lstm.py                      # LSTM and ResidualLSTM
│   │   ├── train.py                     # General training entry point
│   │   ├── expanded_train_experiment.py # Expanded training and reference inference
│   │   ├── build_expanded_train_dataset.py
│   │   ├── expanded_train_report.py     # Paired comparison reports
│   │   └── tests/
│   ├── backend/
│   │   ├── main.py                     # FastAPI entry point
│   │   ├── api/                        # Station, observation and forecast routes
│   │   ├── prediction/                 # Registry, preprocessing and inference
│   │   ├── ingestion/                  # Observation, facility and weather collectors
│   │   ├── db/                         # SQLAlchemy models and Alembic migrations
│   │   ├── cache/                      # Redis and input/forecast caching
│   │   └── security/                   # CORS and request/inference limits
│   └── frontend/                       # React/Vite interface and frontend tests
├── research/
│   ├── scripts/                        # Offline experiments and dataset utilities
│   ├── notebooks/                      # Correlation, seasonal and model analyses
│   └── README.md
├── train_data/
│   ├── da_vien_1h_2026-10-15_forecast_mask_o24/
│   └── da_vien_1h_2026-10-15_expanded_train_o24/
├── model/                              # Serving checkpoints and preprocessing
├── docker/                             # Nginx, automatic startup and GPU override
├── Dockerfile                          # Backend/frontend targets; Python installs use uv
├── docker-compose.yaml
├── requirements.txt                    # Research/training dependencies
└── .env.example
```

Serving artifacts and the two datasets above are versioned. Raw data, database
contents, experimental checkpoints, MLflow artifacts, `reports/` and archives
are local/ignored. **Cloning does not restore the operational database.**

## Run with Docker Compose

Requires Docker Compose and an existing PostgreSQL database with the application
schema and observation data. Copy [.env.example](.env.example) to `.env` and set
`DATABASE_URL`, hydrology-source credentials and, where applicable, the SOHOA token.

`DOCKER_DATABASE_URL` overrides `DATABASE_URL` inside containers. Use an address
reachable from Docker: `127.0.0.1` points to the container itself. For PostgreSQL
in WSL, use its reachable IP and allow the Docker network in PostgreSQL;
`host.docker.internal` may point to Windows. WSL IP changes require updating
`.env` and recreating the affected containers.

Start the application **and ingestion worker**, automatically using NVIDIA CUDA
when Docker and PyTorch can execute on the GPU, otherwise CPU:

```bash
bash docker/up.sh
```

Open **http://localhost:5173**. Inspect the worker or stop the stack:

```bash
docker compose logs -f --tail 50 ingestion
docker compose --profile ingestion down
```

Compose provides Nginx/frontend, backend and Redis; **PostgreSQL is external**.
The launcher builds the CPU image, checks NVIDIA access inside Docker, then builds
and checks a CUDA 12.8 backend image when GPU access is available. GPU use requires
a compatible NVIDIA driver and Docker GPU support (NVIDIA Container Toolkit on
Linux, or GPU-enabled Docker Desktop with WSL2). It sets `PREDICTION_DEVICE=auto`,
including when an older `.env` explicitly requests CUDA. The selected device is
logged at backend startup. CUDA images take more disk space than CPU images.

Plain `docker compose --profile ingestion up --build -d` remains available and
uses the CPU image without GPU access; set `PREDICTION_DEVICE=auto` or `cpu` in
`.env` for that command. Re-run `bash docker/up.sh` to detect GPU support again.

The backend uses one process, with serving artifacts mounted
read-only. Without the `ingestion` profile, automatic collection is off. Its worker
collects observations, facilities and recent ECMWF runs; weather status appears
under `weather`. Set `WEATHER_SYNC_ENABLED=false` to disable weather collection.

Change `HTTP_PORT`/`HTTP_BIND_ADDRESS` and matching `CORS_ORIGINS` when changing
the frontend address. Migration, when required, is explicit:
`docker compose --profile tools run --rm migrate`; it does not restore historical
data. Container health checks API liveness, not database readiness.

## Local development

Python 3.14 is used by Docker. Frontend development requires Node.js 22.12+
(Docker uses Node 24). PostgreSQL and Redis must be available separately.
From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r src/backend/requirements.txt
HYDROLOGY_SYNC_AUTOSTART=false .venv/bin/python -m uvicorn src.backend.main:app \
  --env-file .env --host 127.0.0.1 --port 8000 --workers 1 --no-proxy-headers
```

In separate terminals, run `npm ci && npm run dev` from `src/frontend/`, and
`.venv/bin/python -m src.backend.ingestion` from the repo root. Avoid an additional
local worker while Compose ingestion is active. Frontend: **http://localhost:5173**;
local API/Swagger: **http://127.0.0.1:8000/docs**.

## Datasets and training

Both datasets are fixed snapshots through **15 October 2026**. Runtime ingestion
does not update their CSV files.

| Dataset suffix | Contents | Split |
|---|---|---|
| `forecast_mask_o24` | 26 B features; 24 hourly targets (+1…+24h) | Original train/validation/test |
| `expanded_train_o24` | Pool of 6 rain gauges, 50 features; 24 targets | Former train+validation → train; former test → validation |

Selecting 2/3/4 rain gauges gives 26/32/38 inputs with water, season and weather
features. The expanded runner predicts the five standard horizons and uses
**7,199 training sequences / 592 validation sequences** at sequence length 96.
Uniform and event-balanced sampling use the same dataset. Historical builders
may require source snapshots that are no longer retained here.

In the Python environment created above, add the training dependencies and
GoogleHydrology adapter dependencies imported by the model module:

```bash
.venv/bin/python -m pip install mlflow optuna tqdm matplotlib xgboost lightgbm \
  ruamel.yaml more_itertools xarray dask \
  "git+https://github.com/google-research/flood-forecasting.git"
```

Use a CUDA-compatible PyTorch installation for GPU training; the Docker backend
image is for inference. Start MLflow in a separate terminal:

```bash
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Configuration of the expanded three-station run reported above:

```bash
.venv/bin/python -m src.ai.expanded_train_experiment \
  --dataset-dir train_data/da_vien_1h_2026-10-15_expanded_train_o24 \
  --variant uniform --rain-stations hong_thai+bach_ma+phu_bai \
  --device cuda --seed 42 --num-epochs 100 --early-stopping 25 \
  --sequence-length 96 --hidden-size 16 --num-layers 3 \
  --dropout 0.5 --input-dropout 0.2 --head-dropout 0.2 \
  --learning-rate 0.001 --weight-decay 0.001 --correction-penalty -0.2 \
  --horizon-weights 1 1 1 1 1 \
  --tail-penalty 0.5 --tail-threshold-mm 200 --tail-transition-mm 20 \
  --checkpoint-metric large_error_rate
```

Run folders are named automatically. Each completed run reloads its selected
checkpoint and compares it with legacy/B on common validation anchors. Reports
are written under `reports/model_comparisons/expanded_train/`; MLflow stores
settings/metrics without model uploads. Training does not promote a model to
production. Use `--help` for all options, `--check-only` for preparation checks,
or `--no-mlflow` to train without the tracking server.

## Further documentation

[Research](research/README.md) · [Backend/API](src/backend/README.md) ·
[Prediction contracts](src/backend/prediction/README.md) ·
[Ingestion](src/backend/ingestion/README.md) · [Database](src/backend/db/README.md) ·
[Frontend](src/frontend/README.md) · [API limits](src/backend/security/README.md)
