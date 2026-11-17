# Stage 4: local XGBoost feature ablation

This is a research-only extension of the existing Phase 2 pipeline. It reads the
frozen Stage 1/2 observations and Stage 3B spatial forecast artifacts. It never
writes to PostgreSQL, original DEM, `train_data/`, production models or `src/ai/`.

The configuration is `config.yaml`. The reviewed design comes from
`stage4_spatial_ablation_spec.json` through the Stage 3B latest pointer. Inputs,
execution code and configuration are hashed; use a new `run_name` if they change.
The master plan currently lives at `reports/EDA2.md`.

Run the full study (Dã Viên first, then all eligible stations):

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/hydrology-stage4-mpl \
  .venv/bin/python -m research.station_analysis.phase2.feature_ablation.pipeline
```

The same command resumes completed fits. The current run is `comprehensive_v2`: it adds the December 15–31 validation fold. Compatible winter/spring/summer fold models from `comprehensive_v1` are reused only after frozen-input, archived-code, schema, parameter, seed, target and timestamp checks; final refits are recalculated from the expanded fold set. A model fit uses one station, feature
schema and seed, with separate +6h/+12h regressors and chronological folds. After
validation, it refits before the cutoff using the median selected number of
trees. Semantic aliases (for example E3/W0 = E1) reuse the identical fit but keep
separate ledger labels. Completion markers are atomic; incomplete fits are
rerun. GPU temperature is sampled every two seconds and training pauses at
65°C until it falls to 59°C. No models are selected using post-cutoff metrics.

Run through the common entrypoint with explicit opt-in configuration:

```bash
.venv/bin/python -m research.station_analysis.phase2.pipeline --stage features \
  --feature-config research/station_analysis/phase2/feature_ablation/config.yaml
```

For an isolated station, use `--stations 0074647403` on the Stage 4 entrypoint.
A station-subset run is **partial**, never a completed comprehensive study. Resume
the default all-station command to restore the full eligibility audit.

Regenerate analysis/report from completed models without fitting:

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/hydrology-stage4-mpl \
  .venv/bin/python -m research.station_analysis.phase2.feature_ablation.pipeline --report-only
```

Configuration controls time folds, minimum population, 96-hour context, station
history coverage, horizons, five paired seeds, XGBoost parameters, forecast release
assumptions and thermal policy. This study uses `anchor_only` forecast inputs,
not an unannounced change to the production sequence contract. XGBoost handles
native NaNs; rain missingness stays explicit. No scaler or imputer is fit on
validation/evaluation.

Modules:

- `data.py`: frozen sources, exact-hour features and common populations.
- `planning.py`: E0–E5/W0–W7, train-only discovery, finalists and conditional D1–D9.
- `training.py`: residual XGBoost, checkpoint selection, thermal guard and resume.
- `evaluation.py`, `uncertainty.py`: paired effects, strata, events, peaks and bootstrap.
- `explanations.py`: grouped gain, TreeSHAP and block-shift sensitivity.
- `recommendations.py`, `reporting.py`: regional schema proposals, evidence tables/plots.

Output paths:

- `data/processed/station_analysis/phase2/feature_ablation/<run_name>/`: execution
  manifest, common weather caches, per-station populations/discovery/selection,
  models, predictions, metrics, explanation caches and thermal log.
- `reports/station_analysis/phase2/stage4_feature_ablation/<run_name>/`: Vietnamese
  report, complete CSV/Parquet tables, figures, regional recommendation matrix,
  candidate-schema JSON and result manifest.

D8 within-version retraining requires enough chronological train/validation
samples and at least two algorithmic rainfall episodes in train. D2 requires a
multi-grid finalist. D9 requires the predeclared earliest-train residual signal.
Non-triggered and inestimable branches remain visible in the coverage table.

Historical availability is hypothetical, old forecasts include hindcasts, and
previous EDA inspected some periods now used as inner validation. The post-cutoff
period is exploratory. Local-tree improvements do not prove a Regional LSTM will
improve; static station features cannot be assessed by a separate local model.

Tests:

```bash
MPLCONFIGDIR=/tmp/hydrology-stage4-mpl \
  .venv/bin/python -m unittest discover -s research/station_analysis/tests -v
```

## Coordinated two-worker execution

After measuring spare GPU capacity, the current study also supports two disjoint
station workers. They use the exact same scientific executor, numerical config,
seeds, folds and checkpoints; only output namespaces and cache-library location
differ. A coordinator publishes station completion only after all referenced fits
are copied and verified, and generates one final report. Child reporting is
deferred with `HYDROLOGY_STAGE4_COORDINATED_WORKER=1`.

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/hydrology-stage4-mpl \
  .venv/bin/python -m research.station_analysis.phase2.feature_ablation.parallel --workers 2
```

`parallel_manifest.json`, per-worker YAML/logs and child execution manifests retain
provenance. The resume library references immutable completed models from the
serial runs; old datasets and production checkpoints are never modified. Each
worker monitors the same GPU temperature and pauses at the shared threshold. Two
workers were selected because WSL RAM is limited; do not launch additional serial
or parallel runners on the same station namespaces concurrently.
