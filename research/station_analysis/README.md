# Huế station and terrain analysis

Reusable research code, outside application `src/`. It prepares evidence for a
future Regional ResidualLSTM; it does not train models, modify serving artifacts,
or write to the source PostgreSQL database.

## Install and run

From the repository root, with the project's Python environment:

```bash
.venv/bin/python -m pip install -r research/station_analysis/requirements.txt
.venv/bin/python -m research.station_analysis.pipeline
```

`DATABASE_URL` is read from the environment or root `.env`, never printed or saved.
The first extraction opens a repeatable-read, read-only transaction. All catalog
water stations are retained, including stations without readings. Raw units come
from the database contract; unknown units are not silently treated as millimetres.

Defaults are in [default.yaml](default.yaml). Paths are relative to the repo root.
Change dates, selected IDs, DEM paths, terrain radii and output directories via
YAML or CLI:

```bash
.venv/bin/python -m research.station_analysis.pipeline \
  --config research/station_analysis/default.yaml \
  --station-ids 0074647403 0074647404 \
  --radii-m 100 200 300 \
  --start 2025-11-15T00:00:00+07:00 \
  --end 2026-10-22T00:00:00+07:00 \
  --work-dir data/processed/station_analysis_subset \
  --output-dir reports/station_analysis_subset
```

Dates require timezones; `end` is exclusive. By default the last partial hour is
excluded, and the data snapshot is frozen until explicitly refreshed. Use a new
work/output pair when keeping multiple analysis configurations side by side.

## Individual stages and reuse

```bash
# First read of the source DB, or explicit refresh on newly ingested data.
.venv/bin/python -m research.station_analysis.pipeline --stages extract --refresh-snapshot
# Rebuild all derived stages from that local snapshot, without DB access.
.venv/bin/python -m research.station_analysis.pipeline --stages dem statistics grouping plots report
# Regenerate just figures/report after the prerequisite stages exist.
.venv/bin/python -m research.station_analysis.pipeline --stages plots report
# Recalculate statistics and all its downstream consumers with changed settings.
.venv/bin/python -m research.station_analysis.pipeline --stages statistics grouping plots report --force
```

Stages run in the requested order. Missing prerequisites fail explicitly. A
cache records configuration/code/input identities and output hashes. Repeated
runs reuse matching results; `--force` recomputes selected stages. A full run is
recommended after changing the snapshot, dates, station selection, radii or
clustering cutoff, to keep derived consumers aligned. Reporting and plotting
never implicitly refresh the DB. `--help` lists all CLI options.

The boundary is downloaded only if `boundary_file` is absent. Offline users can
supply a WGS84 GeoJSON FeatureCollection plus an optional adjacent
`.source.json`. The first supplied geoBoundaries boundary describes **2008**,
not today's administrative border. The ROI is that polygon buffered by 1 km,
unioned with station buffers; it is not a watershed delineation. The metadata
file retains the pinned download URL, represented year, source and license.
The provided FABDEM covers this ROI; Copernicus N16/E107 covers only part of it.
Missing raster coverage is reported, never fabricated. GeoTIFF sources are
already extracted; no archive unpacking is necessary for the supplied files.

## Organization and outputs

Responsibilities are separated into configuration, database access, DEM,
metadata, statistics, quality screens, grouping, visualization and reporting.
The entrypoint is `pipeline.py`; each stage can also be imported as Python.

```text
research/station_analysis/       reusable modules, config, sources, tests
 data/DEM/data/                 supplied originals (never changed)
 data/processed/station_analysis/
   inputs/                     administrative boundary and provenance
   snapshot/                   frozen raw/station/hourly Parquet + manifest
   dem/                        clipped FABDEM/Copernicus + analysis region
   tables/                     reusable intermediate Parquet tables
   cache/                      stage fingerprints and output checks
   analysis.sqlite             derived metadata, terrain, hourly series, EDA
 reports/station_analysis/
   report.md                   findings, limitations and training proposal
   stations/<station_id>.md    detailed per-station pages
   tables/                     CSV tables, hourly/change CSV.gz
   figures/                    eight PNG figures
   configuration.json          resolved analysis settings, no DB credentials
   metadata_sources.json       curated references and verification status
   snapshot_manifest.json      source transaction, dates and aggregation
   pipeline.log                stage execution and errors
```

Original DEMs and large derived files stay local under the existing ignored
`data/` and `reports/` directories. Code, configuration and curated references
are versionable. SQLite provides a reusable analysis store without PostgreSQL
raster extensions or an application migration. Later application consumers can
import selected verified fields into the main schema through a separate change.

## Interpretation and safeguards

- Hourly water is the mean of finite, source-status-observed readings inside
  `[hour,hour+1h)`, matching the current project's label convention. A future
  online dataset must define issuance after the observation hour is complete.
- Missing hours remain missing. +6h/+12h joins use exact timestamps, never the
  next available row. Zero and negative values are preserved. Raw cadence and
  hourly missingness are reported separately.
- Strict 96-hour windows require complete water history and finite targets;
  80%-history counts are a separate feasibility measure, not a training policy.
  Neither count yet verifies rain/forecast availability or purged split borders.
- Quality flags are screening hypotheses, not automatic rejection. Defaults:
  1,000 mm hourly jump/raw within-hour range, 100 mm spike-return tolerance,
  24-hour exact plateau, 72-hour rolling range <=5 mm. See `quality.py` and YAML
  keys for overrides. No flagged values are dropped or clipped.
- DEM elevations are metres EGM2008. They are not station zero, sensor height,
  river bathymetry or a shared observational datum. Metric neighborhood disks
  use UTM48N; statistics use valid pixel centers at 100/200/300 m.
- Flood months default to September–December in Vietnam local time; this is a
  calendar label, not an event catalog. Event clusters are not independent floods.
- Descriptive full-period tables are distinct from pre-`grouping_end` features.
  Default grouping ends before 2026-08-01 UTC+7, the existing validation boundary.
  Future-horizon changes cannot cross this cutoff. Clustering uses dynamic,
  offset/scale-resistant descriptors, not absolute levels or DEM elevation.
- Spectral 12/24-hour fractions and correlations are proxies, not proof of tide,
  hydraulic connectivity or causal flow direction. Silhouette and feature-removal
  sensitivity are reported. Terrain associations are exploratory and uncorrected
  for multiple comparisons; no claim of model improvement is made.
- Curated metadata preserves verified facts, database fields and geographic
  hypotheses separately. Station quantity/datum remain unknown until confirmed;
  no Kim Long/Phú Ốc alert thresholds are copied to Dã Viên/Tứ Phú.

FABDEM Vietnam V1-0's local license is **CC BY 4.0**. Its README, attribution and
license remain beside the original raster; the generated report carries the
required Copernicus attribution. This is specific to the supplied Vietnam subset.

## Tests

```bash
.venv/bin/python -m unittest discover -s research/station_analysis/tests -v
```

Tests cover coordinate ordering/projection, metric buffers, nodata, source
preservation, timestamp holes, signed changes, valid windows, constant/sparse
series, grouping cutoff, metadata/alerts, quality screens and cache/storage.

## Phase 2: rainfall and forecast inventory

The four-stage extension is documented in [phase2/README.md](phase2/README.md)
and follows the repository's `EDA2.md` master plan. Only its first stage is
implemented so far:

```bash
.venv/bin/python -m research.station_analysis.phase2.pipeline --stage inventory
```

It reuses existing DEM/metadata outputs and takes one frozen read-only source
snapshot for water, rain and forecasts. Its end-of-hour `(t-1h,t]` inventory
matches the training/serving convention. The original Phase 1 `[t,t+1h)` results
remain descriptive and must be recalculated before reuse as operational feature
selection evidence. Rain accumulation semantics and forecast historical
availability are explicit review gates; no Stage 2/3/4 analyses run implicitly.
