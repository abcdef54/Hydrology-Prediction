# Phase 2 Hydrological EDA

This package implements the master plan in [EDA2.md](../../../EDA2.md) as an
extension of the existing [station analysis pipeline](../README.md). It remains
research code. The application, training pipeline, original datasets, source
observations and Phase 1 outputs are not modified.

## Architecture and review boundaries

| Stage | Responsibility | Prerequisites | Outputs and review gate |
|---|---|---|---|
| 1. Data Inventory & Integrity | Inventory water/rain metadata and observations, forecast vintages and provenance, existing datasets and terrain coverage; check units, timestamps, missingness and availability. | Read-only DB snapshot and retained artifacts. | Availability matrices, sourced semantic contracts, integrity/leakage report and frozen evidence. Implemented in this slice. |
| 2. Rainfall EDA & Water-Level Relationships | Analyze rain amounts/events, past-only accumulation and 0–72h lag relationships; assess stability by season and event. | Stage 1 evidence, confirmed rain measurement semantics, explicit end-of-hour convention and train-only selection periods. | Rainfall EDA, relationship matrices and station recommendations. Implemented and finalized; read the Stage 2 limitations before Stage 3. |
| 3. Weather Forecast Analysis | Evaluate forecast quality and conditional association with water changes using matched valid intervals and spatial support. | Stage 1 vintage inventory, known availability policy, compatible rain ground truth. | Separate operational-evidenced and assumption-dependent forecast assessments. Implemented; review required before Stage 4. |
| 4. Feature Ablation & Regional Recommendations | Compare E0–E5 lightweight baselines on fixed common anchors, then recommend minimal/rain/weather-aware schemas. | Approved Stage 2/3 conclusions and frozen temporal/event evaluation protocol. | Incremental predictive value, stability and feature recommendations. No regional training in this phase. Not implemented; review required. |

The implementation separates source extraction (`snapshot.py`), observation
integrity and semantic aggregation (`observations.py`), forecast availability
(`forecasts.py`), existing forecast provenance (`provenance.py`), artifact/dataset
contracts (`artifacts.py`), reporting and orchestration. These are reusable Python
functions over DataFrames, not command-specific scripts. Existing common JSON,
hash and SQLite writers and missing-run logic are reused from Phase 1. The prior
DEM rasters, metadata and water-level EDA are inspected rather than regenerated.

The data flow for this stage is:

```text
PostgreSQL, REPEATABLE READ / READ ONLY
  -> immutable versioned Parquet snapshot + manifest
  -> observation inventory and canonical (t-1h,t] coverage
  -> forecast run integrity + per-anchor availability by policy
  -> retained dataset/provenance/target checks + Phase1 artifact inventory
  -> reusable Parquet / local SQLite / CSV matrices / figures / readable report
```

Later stages must consume the frozen snapshot and these tables, not silently
refetch data or select a different vintage. Each result carries station/location
identity and absolute UTC timestamps. A forecast sample always uses one vintage
for all four accumulation windows. Rainfall amounts and observation presence are
separate products: presence is not a total, and a total is not proof of historical
availability. Later stage-specific modules will be added only after review;
there are no placeholder analysis results or automatic stage transitions.

## Running Stage 1

Use the existing project environment plus `../requirements.txt` dependencies.
Run from the repository root:

```bash
.venv/bin/python -m research.station_analysis.phase2.pipeline --stage inventory
```

The first execution reads `DATABASE_URL` from environment/root `.env`; source
access uses the existing backend engine with a read-only snapshot transaction.
Only metadata, raw observations and forecast records are read. API credentials,
connection URLs and raw provider request objects are not exported.

Subsequent runs validate hashes and reuse the same frozen snapshot and completed
outputs. `--force` reruns inventory calculations against that snapshot;
`--refresh-snapshot` explicitly takes a new versioned source snapshot. A failed
extraction cannot replace the previous complete snapshot. Changing selected IDs
requires another work directory or an explicit refresh. Analysis dates may change
without re-querying the all-dates snapshot. Without `DATABASE_URL`, a valid saved
snapshot can still be inspected offline.

```bash
.venv/bin/python -m research.station_analysis.phase2.pipeline \
  --stage inventory --force

.venv/bin/python -m research.station_analysis.phase2.pipeline \
  --stage inventory --refresh-snapshot
```

Configuration is [config.yaml](config.yaml). It supports selected water/rain
stations, analysis dates, source dataset directories, forecast assumptions,
rain semantic contracts and work/report locations. Dates require explicit
timezones; `end` is exclusive. With no end, the last whole-hour water measurement
anchor is included. Raw source inventory retains all dates, including rain that
precedes the water series and forecasts issued/retrieved after the analysis
period. These counts are clearly separate from analysis-period coverage.

```bash
.venv/bin/python -m research.station_analysis.phase2.pipeline \
  --water-station-ids 0074647403 0074647404 \
  --rain-station-ids 750062 750007 \
  --start 2025-11-15T00:00:00+07:00 \
  --end 2026-10-22T00:00:00+07:00 \
  --work-dir data/processed/phase2_subset \
  --output-dir reports/phase2_subset
```

The `future_analysis` configuration records design defaults for later work; it
does not cause Stage 1 to compute correlations, event selection or ablation.
Other `--stage` values fail explicitly until implemented and reviewed.

## Time, rainfall and provenance contracts

- Canonical observation anchor `t` contains `(t-1h,t]`, matching the existing
  training/serving builder. Phase 1's `[t,t+1h)` labels remain descriptive and
  are explicitly audited; their water/cluster statistics must be recalculated
  before operational feature selection under the new convention.
- Each hour reports record presence, finite/valid readings, and completeness on
  the expected 10-minute grid. One reading is not treated as six. Missing rain
  is not zero rain. Unknown units remain unverified.
- `received_at` is first local DB insertion, and `updated_at` is the latest local
  revision. Historical backfills do not reconstruct the upstream feed as it was
  at prediction time. Receipt/update-after-anchor counts are evidence limits,
  not automatic assertions that all historical observations leaked.
- Default rain semantics are **unknown**. The old raw-value hourly sum is saved
  only as `legacy_rain_sum_mm_diagnostic`. No physical rain amount, intensity,
  event, correlation or feature ranking is approved from that diagnostic.
- Pure aggregation helpers support confirmed interval amounts, rates and
  cumulative counters. Rates require explicit mm/h integration; counter resets
  and missing intervals remain missing. Unsupported intervals crossing hourly
  boundaries are rejected rather than allocated by inventing a within-interval
  rain pattern. Rolling sums require every past hour.

A future confirmed contract must include all of the following; this example is
illustrative and is **not a factual claim about any current station**:

```yaml
rainfall_semantics:
  "EXAMPLE_ONLY":
    confirmed: true
    kind: interval_amount  # alternatively rate or cumulative_counter
    interval_minutes: 10
    unit: mm              # rate requires mm/h
    source: "Specific verified device/provider documentation and date"
```

Strict forecast coverage accepts only operational runs with documented local
retrieval evidence, valid chronology and `retrieved_at <= available_at <= anchor`.
Hindcasts and unknown archive availability cannot become operationally verified
just because `init + 6h <= anchor`. A second, explicitly diagnostic policy
reproduces that assumed-release backtest selection. Vintage availability is
filtered before taking the latest revision; future-retrieved revisions cannot
replace older evidenced versions in the strict branch.

The provider describes precipitation as the preceding-hour amount. The windows
therefore correspond to `(a,a+3h]`, `(a+3h,a+6h]`, `(a+6h,a+12h]` and
`(a+12h,a+24h]`, using consecutive hourly point timestamps from one run.
Forecast skill and any feature benefit remain Stage 3/4 questions.

## Outputs and reuse

```text
data/processed/station_analysis/phase2/inventory/
  current_snapshot.json
  snapshots/<snapshot_id>/       immutable source Parquet and manifest
  tables/                       reusable inventory evidence, no Stage2+ analysis
  inventory.sqlite              dedicated derived database
  run_manifest.json             config, code/plan/snapshot hashes, stage status
  cache.json
reports/station_analysis/phase2/stage1_inventory/
  report.md                     readable findings and implications
  tables/                       CSV and compressed long availability tables
  figures/                      coverage/availability figures
  configuration.json
  pipeline.log
```

`observation_availability` is the data-availability matrix by station and hour;
`forecast_anchor_availability` separates each location/model/policy at the same
water anchors. Their long formats can be joined without inferring physical
rainfall totals or mapping a grid value to multiple independent gauges.
`dataset_forecast_rows` preserves per-row checks for the previously retained
training datasets. The manifests capture code, data, master plan and temporal
boundaries; original training/test files are never rewritten.

The first report will distinguish confirmed findings, unresolved semantics,
blocking research assumptions and recommended changes to the master plan. A
completed inventory is not permission to run Stage 2; it is the review input.

## Tests

```bash
.venv/bin/python -m unittest discover -s research/station_analysis/tests -v
```

Coverage includes timezones, exact-hour boundaries, incomplete hours, rainfall
amount/rate/counter semantics, counter resets, missing versus zero values,
forecast windows, vintage/retrieval ordering, hindcast exclusion, retained-target
alignment, immutable snapshot/cache behavior and rejection of later stages.

## Running Stage 2 and reproducing the finalized reports

```bash
.venv/bin/python -m research.station_analysis.phase2.pipeline --stage rainfall
.venv/bin/python -m unittest discover -s research/station_analysis/tests -v
```

Stage 2 reuses the frozen inventory snapshot and writes both `report.md` and
`stage2_clarification_report.md` under `reports/station_analysis/phase2/stage2_rainfall/`.
Both reports use the same computed DataFrames. The run manifest records config,
source snapshot hash, code hashes, output hashes and the temporal boundary.
The rainfall stage currently recomputes its analysis on each run; it does not
query source observations again when the frozen snapshot is reused.

The selection cutoff is `future_analysis.grouping_end` (explicit UTC+7).
Train anchors must satisfy `anchor + 12h <= cutoff`, while exploratory evaluation
anchors satisfy `anchor > cutoff`. All Ridge combinations share the same valid
anchors; station/lag selection and collinearity are train-only. The search grid
uses rainfall windows 1/3/6/12/24/48h and lags 0..72h in 2h steps. The
full-period figures, event sensitivity, clusters and conditional correlations
are descriptive and must not be treated as train-only feature selection.

Stage 2 uses the project's assumed 10-minute interval rainfall contract,
separately from the independently verified per-provider contracts required by
Stage 1. No new provider documentation was established during finalization.
Retain this assumption when interpreting Stage 3 forecast-vs-rain skill.

Tables 12–15 audit rainfall totals by split, optimal-lag counts, timestamped water
episodes and purged split sizes. Table 09 separates policy sensitivity on
each policy's own anchors from the difference on identical common anchors.
Stage 3 and Stage 4 remain unimplemented and do not run automatically.

## Focused Stage 2 clarification audit

```bash
.venv/bin/python -m research.station_analysis.phase2.stage2_clarification
```

This reads the saved rainfall Parquet, canonical inventory availability, station
catalog and Stage 2 CSVs. It regenerates both reports and targeted diagnostics
under `tables/clarification/` and one influential-episode figure, without rerunning
the full EDA or training forecasting models. The old cutoff/window grid is
recomputed only for Dã Viên to reconcile historical report/CSV values. The
existing Ridge objective is replayed analytically to verify saved scores and
inspect conditional correlations; no feature/model selection uses evaluation
results. `clarification_manifest.json` records input, output and code hashes.
Earlier report/table versions are retained under `audit_history/`.

## Stage 3: one shared city weather source

```bash
.venv/bin/python -m research.station_analysis.phase2.pipeline --stage weather
# Recompute only Stage 3 if explicitly needed:
.venv/bin/python -m research.station_analysis.phase2.pipeline --stage weather --force
```

`weather.py` validates the frozen snapshot, reuses the reviewed rain tables,
and reuses the Stage 1 forecast selection/integrity functions.
`weather_quality.py` matches exact future intervals to explicit gauge comparators;
`weather_metrics.py` contains missing-safe rain metrics and train-only nuisance
projections. `weather_relationships.py` assesses water associations, and
`weather_reporting.py` renders the findings. All code stays in research.

Forecast is ONE city-wide input source shared by all water stations. HT/Nham
request records are lineage views, not independent forecasts. Selection retains
one run for all four windows at an anchor, after source-specific availability
filtering, preferring complete 24h coverage and then the newest initialized/retrieved
run. Sources must share the same returned grid before city deduplication.
Raw request tables are kept only for provenance and duplicate-value checks.

`weather_analysis` config controls reference rain gauge IDs, number of nearest
grid comparators, wet-event threshold and heavy-rain mean-rate threshold.
Comparators are selected by declared station ID or geography, not evaluation
skill. No city rainfall ground-truth average is invented. Quality uses a 24h
target purge; water associations use the same 12h purge as Stage 2.
Pearson/Spearman, partial correlations and conditional checks are diagnostic;
E0–E5 predictive ablation remains Stage 4 and is not executed.

Outputs: `reports/station_analysis/phase2/stage3_weather/report.md`, `tables/`,
`figures/`, `run_manifest.json`; reusable Parquet and matched forecast/gauge
intervals live in `data/processed/station_analysis/phase2/weather/`. A fingerprint
of config, snapshot, upstream artifacts and code enables safe cache reuse;
changing an output invalidates that cache. Source DB and Stage 1/2 artifacts
are not modified. The default strict branch has no available historical anchors;
metrics from hindcast/unverified archives remain explicitly assumption-dependent.

## Stage 3B: spatial weather forecasts (no model training)

The modules under `spatial_weather/` extend the existing forecast audit, rain
metrics, water targets and train-only nuisance-projection helpers. Production
models, SQL observations, DEM sources and training datasets are never mutated.

Run the small validation gate first:

```bash
.venv/bin/python -m research.station_analysis.phase2.spatial_weather.validation
```

This checks six initialization times spanning hindcast/archived versions and
requires exact reproduction of three retained A Lưới runs. It records requested
coordinates separately from returned grid coordinates and reuses prior nearest
selection evidence only for a small mapping sensitivity check. A failed gate
blocks bulk collection. The model stays `ecmwf_ifs` and baseline selection `land`.

Resume historical collection (explicit paths preserve the frozen source):

```bash
.venv/bin/python -m research.station_analysis.phase2.spatial_weather.collect_archive \
  --folder data/processed/station_analysis/phase2/spatial_weather/collection_v1 \
  --snapshot data/processed/station_analysis/phase2/inventory/snapshots/20261026T035126Z_7f98ea8a \
  --start 2025-11-15T00:00:00+07:00 \
  --end 2026-10-21T19:00:00+07:00
```

The collector batches six new regions, reuses retained A Lưới cycles, fills
missing reference cycles, checkpoints every cycle, and retains immutable raw
responses. Its conservative ledger counts each coordinate separately, with
headroom under the provider's minute/hour/day limits. Budget pauses are normal;
progress and exact failures remain in `collection_progress.json`. Reruns skip
completed requests. A missing archive is reported rather than substituted with
observations or reanalysis. All UTC cycles are built after timezone conversion.

Once collection has finished, execute only Stage 3B:

```bash
.venv/bin/python -m research.station_analysis.phase2.pipeline --stage spatial-weather
.venv/bin/python -m unittest discover -s research/station_analysis/tests -v
```

Configuration uses `spatial_weather` in `config.yaml`, plus the existing start,
end, temporal cutoff, forecast age and rain-verification thresholds. Changing
validated locations requires a separate collection version and another gate.
Stage 3B writes versioned tables, raw/processed run identities and synchronized
features under `data/processed/station_analysis/phase2/spatial_weather/analyses/`.
Reports live under `reports/station_analysis/phase2/stage3b_spatial_weather/<version>/`;
`latest.json` points to the latest report. Identical inputs/config/code reuse the
completed analysis; changed inputs create another version instead of overwriting
old results. Stage 4 is a JSON specification only and is not executed.

Individual-cell verification may select that cell's own latest eligible run.
Cross-grid comparisons and every proposed feature vector instead use a common
initialization across all selected cells, with the same six-hour hypothetical
release and maximum-age policy. The evidence-only branch remains separate.
Gauge comparisons stay separate and terrain is descriptive; neither station
means nor grid centers are treated as a measured catchment-wide rainfall truth.
W2 means nearest *collected* cell and explicitly reports that spatial proxy's
distance; it does not falsely claim an unqueried exact-target API mapping.

## Stage 4: reviewed comprehensive feature ablation

The implementation and full execution configuration are documented in
[`feature_ablation/README.md`](feature_ablation/README.md). This stage uses local
XGBoost models and the existing frozen observations and spatial forecast archive.
It runs Dã Viên first, then all eligible stations, with five paired seeds and
chronological validation. It does not prepare a regional training dataset.

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/hydrology-stage4-mpl \
  .venv/bin/python -m research.station_analysis.phase2.feature_ablation.pipeline
```

The shared entrypoint requires explicit `--feature-config` for `--stage features`;
this prevents an old generic inventory command from launching training by mistake.
