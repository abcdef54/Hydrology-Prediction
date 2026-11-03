# Dã Viên Water-Level Forecasting

Water-level forecasting and hydrology monitoring for Huế, Vietnam. The application
predicts **hourly mean water levels** at **Dã Viên (`0074647403`)** at **+1, +3,
+6, +12, and +24 hours**, in **millimetres**. A regional model (11 stations,
+6/+12 h) is the current best research model; the serving backend still runs the
three registered Dã Viên ResidualLSTM models.

The research objective is to reduce forecasts with **absolute error >200 mm**,
especially at +12/+24 hours during rapid rises and falls.

## Current best model — regional ResidualGRULateFusion

The best model so far is a **ResidualGRULateFusion** (`gru-late-fusion`)
regional checkpoint in [model/selected/gru_high_precision/](model/selected/gru_high_precision/).

- **11 stations**, including Dã Viên, Thuận An, Tư Hiền, Tứ Phú, Ngã Ba Sình,
  Sông Ô Lâu, Phú Xuân and An Cựu. Trained on the fixed regional pool
  `train_data/hue_regional_20261015_h6_h12_fixed_split_v1` (see
  [TRAINING_REGIONAL.md](TRAINING_REGIONAL.md)).
- **Two outputs: +6 h and +12 h** (not the full five-horizon set).
- Stacked GRU history encoder (**hidden size 64, 3 layers, 96-hour input**),
  station embeddings (size 8), per-station persistence coefficients and a
  zero-initialized correction head.
- **Late fusion of the ECMWF forecast**: all forecast columns are removed from
  the history encoder and only the forecast vector at the prediction anchor
  enters a separate MLP. Inputs are the `water_level+rain+flood_season+weather_forecast`
  feature set (42 inputs).
- Station-balanced loss with a 200 mm target-scale floor; checkpoint selected on
  validation `val_loss` at **epoch 7**.

Validation is the fixed regional split **July 31 – September 14, 2026**
(1,080 anchors per station per horizon; 11 × 2 × 1,080 = **23,760 predictions**):

| Scope | RMSE (mm) | MAE (mm) | Errors >200 mm | Rate |
|---|---:|---:|---:|---:|
| Pooled, all stations and horizons | 93.08 | 36.47 | 380 | 1.60% |
| All stations, +6 h | 79.56 | 32.58 | 170 | 1.43% |
| All stations, +12 h | 104.88 | 40.36 | 210 | 1.77% |
| Dã Viên, +6 h | 63.64 | 45.34 | 18 | 1.67% |
| Dã Viên, +12 h | 73.68 | 54.04 | 19 | 1.76% |

This selection is high-precision on rise alarms: pooled +6 h rise recall
**0.62** with false-alarm ratio **0.102**. Dã Viên rise precision at +6 h is
higher (12 hits, 1 false alarm) at the cost of lower recall (0.48).

Caveats, inherited from the research process and kept deliberately:

- This validation period has been inspected repeatedly during the architecture
  study; the September 15 – October 15 test period remains unused. Results are
  therefore **exploratory until an untouched holdout confirms them**.
- The checkpoint is **not registered** in
  [src/backend/prediction/model_configs.json](src/backend/prediction/model_configs.json);
  no model has been promoted or deployed. The live app is still a
  single-station contract, and live regional API input assembly is future work.

## Production (serving) models

The [model registry](src/backend/prediction/model_configs.json) defines three
independent, selectable **ResidualLSTM** models for Dã Viên. They are not an
ensemble. Each has its own checkpoint, fitted preprocessing and feature order in
[model/](model/README.md).

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

### Recorded comparison (Dã Viên, five horizons)

On **592 common anchors**, from **19 September 2026, 23:00** through
**14 October 2026, 14:00 (UTC+7)**: 5 horizons × 592 anchors = **2,960
forecasts**. RMSE below is **pooled across all predictions**, not the mean of
horizon RMSEs. Note this comparison covers a different period, targets and
anchor set from the regional GRU results above; the two tables are not directly
comparable.

| Model | Deployment | RMSE (mm) | MAE (mm) | Errors >200 mm | Rate |
|---|---|---:|---:|---:|---:|
| Legacy | Production, default | 87.03 | 64.47 | 97 | 3.28% |
| Model B, seed 42 | Production | 75.78 | 57.00 | 48 | 1.62% |
| Three-station, seed 42 | Production | 77.04 | 58.11 | 50 | 1.69% |
| Expanded-train three-station, hidden size 16 | Experimental | **72.30** | **54.03** | **32** | **1.08%** |

The best expanded candidate (uniform sampling, seed 42, correction penalty −0.2,
checkpoint epoch 86, run `seq96_h16_large_error_rate_20261022T061755947891Z_b872679c`)
is **not deployed**; it has no independent test set. All models in this table
are Dã Viên-only; none uses the 11-station regional training of the GRU.

## Application

- Interactive station map, latest observations, hourly history and reservoir-operation data.
- Selectable production models, single-anchor forecasts and forecasts over a date range.
- Charts comparing forecasts with observations; missing values remain visible as gaps.
- Saved forecasts identified by model/pipeline version; edited inputs produce unsaved scenario forecasts.
- Scheduled observation and ECMWF rainfall-forecast ingestion, PostgreSQL storage and Redis caching.

**Stack:** PyTorch, FastAPI, SQLAlchemy/Alembic, PostgreSQL, Redis, React/Vite,
MapLibre and Nginx. Forecasting is currently enabled for Dã Viên; other stations
supply observations or appear in the monitoring interface.

## Repository layout

```text
.
├── src/
│   ├── ai/
│   │   ├── lstm.py                       # Original ResidualLSTM
│   │   ├── residual_sequence.py          # GRU/TCN/CNN-GRU/Transformer residual wrappers
│   │   ├── train.py                      # Training entry point (local and regional pools)
│   │   ├── regional_training_data.py     # Regional pool dataset adapter
│   │   ├── sequence_checkpoint.py        # Inference runtime for regional checkpoints
│   │   ├── RESIDUAL_ARCHITECTURES.md     # Architecture reference
│   │   ├── expanded_train_experiment.py  # Expanded Dã Viên training and reference inference
│   │   ├── event_evaluation.py           # Rise-alarm evaluation for validation snapshots
│   │   └── tests/
│   ├── backend/
│   │   ├── main.py                       # FastAPI entry point
│   │   ├── api/                          # Station, observation and forecast routes
│   │   ├── prediction/                   # Registry, preprocessing and inference
│   │   ├── ingestion/                    # Observation, facility and weather collectors
│   │   ├── db/                           # SQLAlchemy models and Alembic migrations
│   │   ├── cache/                        # Redis and input/forecast caching
│   │   └── security/                     # CORS and request/inference limits
│   └── frontend/                         # React/Vite interface and frontend tests
├── research/
│   ├── station_analysis/                 # DEM, read-only station EDA and exploratory grouping
│   ├── scripts/                          # Offline experiments and dataset utilities
│   ├── notebooks/                        # Correlation, seasonal and model analyses
│   └── README.md
├── train_data/
│   ├── da_vien_1h_2026-10-15_*           # Fixed Dã Viên snapshots (see below)
│   └── hue_regional_*                    # Fixed regional pools (11-station experiments)
├── model/
│   ├── *.pt + *_preprocessing.joblib     # Registered serving checkpoints (Dã Viên)
│   ├── selected/gru_high_precision/      # Current best: regional GRU late-fusion candidate
│   ├── selected/                         # Other selected regional candidates
│   └── experiments/                      # Local experiment checkpoints (not served)
├── docker/                               # Nginx, automatic startup and GPU override
├── Dockerfile                            # Backend/frontend targets; Python installs use uv
├── docker-compose.yaml
├── requirements.txt                      # Research/training dependencies
├── TRAINING_REGIONAL.md                  # How to train the 11-station regional models
└── .env.example
```

Serving artifacts and the Dã Viên snapshot datasets above are versioned. Raw
data, database contents, experimental checkpoints, MLflow artifacts, `reports/`
and archives are local/ignored. **Cloning does not restore the operational
database.**

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
The launcher builds the CPU image, checks NVIDIA access inside Docker, then
builds and checks a CUDA 12.8 backend image when GPU access is available. GPU use
requires a compatible NVIDIA driver and Docker GPU support (NVIDIA Container
Toolkit on Linux, or GPU-enabled Docker Desktop with WSL2). It sets
`PREDICTION_DEVICE=auto`, including when an older `.env` explicitly requests
CUDA. The selected device is logged at backend startup. CUDA images take more
disk space than CPU images.

Plain `docker compose --profile ingestion up --build -d` remains available and
uses the CPU image without GPU access; set `PREDICTION_DEVICE=auto` or `cpu` in
`.env` for that command. Re-run `bash docker/up.sh` to detect GPU support again.

The backend uses one process, with serving artifacts mounted read-only. Without
the `ingestion` profile, automatic collection is off. Its worker collects
observations, facilities and recent ECMWF runs; weather status appears under
`weather`. Set `WEATHER_SYNC_ENABLED=false` to disable weather collection.

Change `HTTP_PORT`/`HTTP_BIND_ADDRESS` and matching `CORS_ORIGINS` when changing
the frontend address. Migration, when required, is explicit:
`docker compose --profile tools run --rm migrate`; it does not restore
historical data. Container health checks API liveness, not database readiness.

## Local development

Python 3.14 is used by Docker. Frontend development requires Node.js 22.12+
(Docker uses Node 24). PostgreSQL and Redis must be available separately. From
the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r src/backend/requirements.txt
HYDROLOGY_SYNC_AUTOSTART=false .venv/bin/python -m uvicorn src.backend.main:app \
  --env-file .env --host 127.0.0.1 --port 8000 --workers 1 --no-proxy-headers
```

In separate terminals, run `npm ci && npm run dev` from `src/frontend/`, and
`.venv/bin/python -m src.backend.ingestion` from the repo root. Avoid an
additional local worker while Compose ingestion is active. Frontend:
**http://localhost:5173**; local API/Swagger: **http://127.0.0.1:8000/docs**.

## Datasets and training

The Dã Viên CSV datasets are fixed snapshots through **15 October 2026**.
Runtime ingestion does not update their CSV files.

| Dataset suffix | Contents | Split |
|---|---|---|
| `forecast_mask_o24` | 26 B features; 24 hourly targets (+1…+24h) | Original train/validation/test |
| `forecast_mask_o2_h6_h12` | Same 26 inputs; only +6/+12h targets | Exact O24 splits and eligible anchors |
| `forecast_mask_o2_h6_h12_weather12` | 24 inputs: remove forecast rain 12–24h and its mask; same +6/+12h targets | Exact two-output splits and anchors |
| `o2_h6_h12_weather12_rain_pool` | Pool of 6 rain gauges, 48 inputs; +6/+12h targets and forecast windows through +12h | Original two-output splits and anchors; not expanded train |
| `o2_h6_h12_weather24_rain_pool` | Matched control: same rain pool and targets, retaining forecast rain 12–24h and its mask (50 inputs) | Same original splits and anchors as weather12 pool |
| `expanded_train_o24` | Pool of 6 rain gauges, 50 features; 24 targets | Former train+validation → train; former test → validation |

The regional experiments additionally use prebuilt Parquet pools such as
`train_data/hue_regional_20261015_h6_h12_fixed_split_v1` (11 stations,
+6/+12 h, fixed split: validation July 31 – September 14, test September 15 –
October 15). These datasets are recognized as prepared regional pools; the
loader does not rebuild them or refit their normalizers. See
[TRAINING_REGIONAL.md](TRAINING_REGIONAL.md) for a portable training guide.

Single-station and regional residual architectures (GRU, TCN, CNN-GRU,
Transformer, and their late-fusion variants) are documented in
[src/ai/RESIDUAL_ARCHITECTURES.md](src/ai/RESIDUAL_ARCHITECTURES.md). History
length (`--lstm-seq-len`), target horizons (`--target-horizons`), station
conditioning and the change-objective losses are shared across all of them.

### Rebuilding the paired Dã Viên datasets

Rebuild the paired two-output dataset with
`python -m src.ai.build_horizon_subset_dataset`. It retains sequence length 96,
the 24-hour purge and all input rows, including forecast rain through +24h.
Both selected targets are masked to NaN on rows missing any of the original 24
labels, so reducing outputs never adds eligible anchors: **6,095 / 985 / 592**
train/validation/test sequences. Retained labels use exact future timestamps.
`preprocessing.joblib` contains train-only fitted input/target scalers and the
two persistence coefficients per type. The existing `src.ai.train` loader
accepts this dataset with `--horizon 1h --lstm-seq-len 96 --target-horizons 6 12`;
`--outputs` remains restricted to 5 or 24. No model is trained by the builder.

Build the 24-input feature ablation with
`python -m src.ai.build_weather_horizon_ablation`. It projects the parent input
scaler to the retained columns and reuses the target scaler without fitting.
Historical 24-hour rain sums, forecast age, all remaining masks, target masks,
weather-vintage selection and forecast-history policy stay unchanged. Metadata
and `preprocessing.joblib` contain updated feature and weather-mask indices.
The same training CLI accepts this dataset without architecture or loop changes.

The fixed 24-input dataset contains only Hồng Thái and Nhâm. For other
combinations, build the pool with
`python -m src.ai.build_weather_horizon_ablation --rain-stations all
--output-dir train_data/da_vien_1h_2026-10-15_o2_h6_h12_weather12_rain_pool`.
It copies matching rain predictors from the retained expanded dataset's hourly
snapshot, while keeping the original two-output splits and masked labels. Added
input scales are fitted on the original train split only. Select sources with
`--rain-stations hong_thai+bach_ma+phu_bai` when training: 2/3/4/6 gauges give
24/30/36/48 inputs. The original two-station ablation stays unchanged. Use
`--forecast-hours 24` with a separate output directory to build the matched
control containing forecast rain through +24h (26/32/38/50 selected inputs).

### Comparison and promotion policy

After `src.ai.train` finishes on a Dã Viên dataset, comparison defaults to
`--comparison-model-id model_b`. It evaluates only shared horizons on identical
eligible timestamps, with each model's saved scalers. For the two-output
experiment, Model B's five-target reference table is resolved from the O24
parent without restoring masked samples. The console and
`reports/model_comparisons/<run_id>_<suffix>/comparison.md` show MAE/RMSE and
errors >200 mm. The verdict prioritizes fewer large errors, then lower pooled
RMSE; it never promotes a model automatically. Comparison failures are reported
separately and preserve the completed training checkpoint.

Expanded-train runs likewise compare their selected checkpoint with legacy/B on
common validation anchors and write reports under
`reports/model_comparisons/expanded_train/`; MLflow stores settings/metrics
without model uploads. Training does not promote a model to production. Use
`--help` for all options, `--check-only` for preparation checks, or `--no-mlflow`
to train without the tracking server.

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

The 2026-10-23 two-output study completed 36 cases (three configurations, three
rain combinations, two seeds, two forecast-input windows). On the same 592
anchors and only +6h/+12h, Model B has 21 errors >200 mm (RMSE 85.69 mm). The
best new case has 18 (87.49 mm), but its other seed has 38; the existing
expanded model has 17 (81.10 mm), using a larger training split. Removing the
12–24h forecast and its mask improves 12 of 18 matched pairs, with six worse.
These exploratory results do not establish a replacement for production. The
local detailed report and all case metrics are under
`reports/model_comparisons/20261023_weather_outputs_matrix/`.

## Station and terrain analysis

The reusable [station analysis pipeline](research/station_analysis/README.md)
clips the supplied FABDEM/Copernicus rasters to a documented Huế ROI, samples
100/200/300 m neighborhoods and analyzes exact +6h/+12h water changes. It reads
PostgreSQL in a read-only transaction and stores derived results in GeoTIFF,
Parquet and a separate local SQLite file; application tables are unchanged.

The initial run covers 19 catalog stations and 796,846 readings through
2026-10-21. It flags sensor transitions for review and finds weak exploratory
clustering (silhouette 0.226). Station datums/physical target definitions remain
unverified; DEM elevation must not be subtracted from station readings or used
alone to define regional model groups. No regional model has been trained by
this pipeline. Local results: `reports/station_analysis/report.md`.

## Further documentation

The [Phase 2 hydrological EDA extension](research/station_analysis/phase2/README.md)
follows `EDA2.md`. Its Stage 1 inventories 19 water stations, 55 rain gauges and
forecast vintages without training or changing source data. Rain accumulation
semantics and historical forecast availability remain review gates. The local
report is `reports/station_analysis/phase2/stage1_inventory/report.md`; later
stages require review before implementation.

[Research](research/README.md) · [Backend/API](src/backend/README.md) ·
[Prediction contracts](src/backend/prediction/README.md) ·
[Ingestion](src/backend/ingestion/README.md) · [Database](src/backend/db/README.md) ·
[Frontend](src/frontend/README.md) · [API limits](src/backend/security/README.md) ·
[Residual architectures](src/ai/RESIDUAL_ARCHITECTURES.md) ·
[Regional training](TRAINING_REGIONAL.md)
