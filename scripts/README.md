# Hue hydrology data audit

Run the complete analysis from the project root:

```bash
.venv/bin/pip install -r requirements.txt
MPLCONFIGDIR=/tmp/hydrology-matplotlib .venv/bin/python scripts/run_all.py
```

The pipeline reads from `data/` without modifying raw files. It writes tabular
and JSON evidence to `reports/`, plots to `figures/`, and the synthesis to
`reports/data_analysis_report.md`.

Each stage can also run independently:

```bash
.venv/bin/python scripts/inspect_datasets.py
.venv/bin/python scripts/plot_observation_counts.py
.venv/bin/python scripts/analyze_temporal_quality.py
.venv/bin/python scripts/analyze_spatial_relationships.py
.venv/bin/python scripts/analyze_hydrography_topology.py
.venv/bin/python scripts/analyze_hydrologic_signal.py
.venv/bin/python scripts/run_baselines.py
.venv/bin/python scripts/evaluate_training_feasibility.py
```

When no training table exists, `run_all.py` builds the derived table at
`reports/hydrology_training_10min.csv`. To rebuild a separate copy without
touching raw data:

```bash
.venv/bin/python scripts/build_training_dataset.py \
  --output reports/hydrology_training_10min_rebuilt.csv
```

Use `--force` only to replace that explicitly selected derived output. The
builder refuses to write into the raw `data/` directory.

Build the topology-filtered Dã Viên v2 dataset and its audit report:

```bash
.venv/bin/python scripts/build_training_dataset_v2.py
```

This builder reads `reports/da_vien_candidate_input_stations.csv`, cross-checks
it against `reports/station_hydrologic_relations.csv`, requires exactly 14
topology-approved rain stations and four upstream reservoirs, and writes
`reports/hydrology_training_10min_v2.csv` plus missingness and leakage-audit
companion files. Reservoir values are forward-filled from the past for at most
60 minutes by default, with separate source-missing and forward-filled flags;
other sources are not filled. The wind inputs are the explicitly requested
Cảng Thuận An and Cảng Tư Hiền stations, with a network-missing flag that is
set only when both stations are unavailable. Use `--force` to rebuild
existing derived outputs.

Build the comparable 30-minute and 1-hour datasets without changing the
existing 10-minute dataset:

```bash
.venv/bin/python scripts/build_multiresolution_training_datasets.py
```

The 30-minute dataset uses Dã Viên targets at +1h, +3h, +6h, and +12h.
The 1-hour dataset additionally includes +24h. Rain rolling windows, water
level and reservoir lags, and the 60-minute reservoir fill cap are converted
from physical time to the correct number of bins for each resolution.
