# Regional sequence dataset

This package prepares and reads data only. It does not train a Regional LSTM or
change the existing local-model, Model B, or production pipelines.

The research source adapter reuses frozen EDA1 station groups, EDA2 hourly
observations and rainfall preprocessing, and the Stage 3B forecast archive.
It never queries or writes PostgreSQL. Keep those source artifacts to rebuild.

## Build and validate

Run from the repository root with the project's `.venv`. Dependencies already
used by the EDA pipelines are listed in
`research/station_analysis/requirements.txt`; core reading requires NumPy,
pandas and PyArrow. There is no GPU requirement.

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m research.scripts.build_regional_dataset \
  --execution-manifest data/processed/station_analysis/phase2/feature_ablation/comprehensive_v2/execution_manifest.json \
  --groups reports/station_analysis/tables/cluster_assignments.csv \
  --output-dir train_data/hue_regional_h6_h12_new_run

OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest src.ai.tests.test_regional_data -v
```

Omit `--output-dir` to generate a timestamped directory. Existing directories
are rejected, so an interrupted build cannot overwrite an earlier dataset.
Optional `--start` and `--end` accept inclusive timestamps with explicit
timezones. They restrict the source range, including history; a range too short
to populate the chronological training folds is rejected.

To audit an existing build without rebuilding or training:

```python
from src.ai.regional_data.validation import validate_dataset
validate_dataset("train_data/hue_regional_20261015_h6_h12_fixed_split_v1")
```

The validator checks every sample's exact labels and history positions, every
rain window, every fold's normalizers, and all forecast window sums against the
archived hourly points. It also checks source hashes, same-anchor shared inputs,
feature selection, local-calendar flood season, purge and persistence transforms.
These are retrospective data checks, not proof of historical real-time receipt.

## Choose features without rebuilding

```python
from src.ai.regional_data import RegionalDataset

data = RegionalDataset(
    "train_data/hue_regional_20261015_h6_h12_fixed_split_v1",
    rain_stations="nham+bach_ma+ho_hoa_my+hong_thai",
    use_weather_forecast=True,
    time_features="none",
    split="train",
)
sample = data[0]
print(sample["x"].shape)       # (96, 68): default four gauges and forecast
print(sample["y"].shape)       # (2,): standardized absolute levels at +6h/+12h
print(sample["station_index"]) # separate integer ID, suitable for later conditioning
print(data.feature_columns)   # exact selected channel order

without_weather = RegionalDataset(
    data.path,
    rain_stations="nham+bach_ma",
    use_weather_forecast=False,
    split="train",
)
assert len(without_weather) == len(data)
```

The examples correspond to the future training controls `--rain-station` and
`--use-weather-forecast`; those flags are **not** added to the existing trainer
in this task. The reader accepts actual Python booleans, not string `"false"`.
Use numeric gauge IDs or catalog aliases; `none` selects no rain gauges and
`all` selects the entire stored pool. Unknown/duplicate IDs are rejected.
The whole pool has 55 gauges, including Nhâm (750007), Bạch Mã (750041),
Hồ Hòa Mỹ (682816) and Hồng Thái (750062). Bạch Mã is distinct from the
Vườn Quốc gia and Đỉnh Bạch Mã gauges.

All four cyclic time features are stored, but selection is deferred to reading:

| `time_features` | Calendar channels | Default input width with four gauges and weather |
|---|---|---:|
| `none` | `flood_season` | 68 |
| `hour` | `flood_season`, `hour_sin`, `hour_cos` | 70 |
| `day_of_year` | `flood_season`, `day_sin`, `day_cos` | 70 |
| `all` | All five | 72 |

Use `RegionalDataset(..., time_features="all")` to enable both groups. This does
not change samples, targets, splits, rainfall/weather selection, or normalizers.
These are reader options, not new CLI flags in the existing training pipeline.
For local time in Vietnam, the formulas are:

```text
hour_angle = 2*pi*local_hour/24
day_angle = 2*pi*(local_day_of_year-1)/(366 if local_year_is_leap else 365)
hour_sin, hour_cos = sin(hour_angle), cos(hour_angle)
day_sin, day_cos = sin(day_angle), cos(day_angle)
```

Jan 1 starts at day angle zero. The annual encoding advances by one daily step
on Feb 29 in leap years. Day-of-year features are constant within a local day;
hour-of-day features provide the daily cycle. All cyclic channels retain their
original range `[-1, 1]`, without fitting an extra scaler. They are shared at
each timestamp across all water stations and contain no observation-derived
future information.

Each timestep has:

- Two water channels: the target station's standardized level and missing flag.
- `flood_season`: 1 during September–December, 0 otherwise, using
  `Asia/Ho_Chi_Minh`, as in the existing datasets. This is a calendar indicator,
  not a confirmed flood-event label. It remains binary.
- Thirteen channels per selected rain gauge: past totals over 1/3/6/12/24/48
  hours, six matching missing flags, and a flag for an estimated hourly total.
- When enabled, thirteen weather channels: four A Lưới forecast totals,
  four missing flags, forecast age and its missing flag, and the three
  observed-availability/assumed-availability/hindcast flags.

Input order is deterministic: water, calendar, caller-selected gauge blocks,
then weather. Explicitly save `feature_columns` with any future checkpoint.
Batching will produce `[B, 96, F]`, IDs `[B]` and targets `[B, 2]`.

## Artifact contract

```text
dataset/
  manifest.json                    source/code hashes, policies and counts
  config.json                      exact folds and build parameters
  station_mapping.json             stable IDs, EDA1 groups, coordinates and datum caveats
  feature_schema.json              gauge blocks, aliases, masks, units and channel order
  normalizers.json                 separate train-only statistics for each fold/station
  samples.parquet                  station, anchor/history positions, exact targets, memberships
  water.parquet                    complete hourly UTC grid, one column per water station
  rainfall.parquet                 separate engineered blocks for all gauges
  rainfall_hourly_standard.parquet canonical hourly observations, raw NaN retained
  rainfall_hourly_strict.parquet   strict 6/6 reference
  weather.parquet                  only A Lưới features, run identity and provenance
  calendar.parquet                 flood season plus four optional cyclic time features
  rain_station_catalog.parquet    gauge IDs, names and coordinates
  eda1_station_groups.csv
  station_summary.csv
  rain_summary.csv
  validation.json
  REPORT.md                        self-contained Vietnamese build report
```

The build first creates a `BUILDING.json` marker; successful validation removes
it. Treat a build with that marker, no `validation.json`, or a failed validation
as incomplete. Do not use it for training. The manifest records SHA-256 hashes
of all generated artifacts except itself.

Water history, rain and calendar cover a complete hourly grid, so sequence
construction cannot jump over missing hours or cross station boundaries.
Shared values are stored once per timestamp, not duplicated for each station.
Changing features does not drop samples, including those with missing rainfall
or forecast. The source mean-level convention is `(t−1h, t]`; targets are those
hourly observed means at exactly `t+6h` and `t+12h`, without interpolation.

## Splits and normalization

All stations in the corrected dataset use one fixed train/validation/test split:

- Nominal train period: before **2026-07-31 00:00 UTC+7**.
- Nominal validation period: **2026-07-31 through 2026-09-14**.
- Nominal test period: **2026-09-15 through 2026-10-15**. The retained local
  observation snapshot ends at **2026-10-15 14:00 UTC+7**.
- Purge is applied to the last 24 anchor hours of train and validation. Thus
  actual train anchors end July 29 at 23:00, validation anchors start July 31
  at 00:00 and end September 13 at 23:00, and test anchors start September 15
  at 00:00 and end October 15 at 02:00 (the +12h label needs the final observation).
- Gap anchors stay labelled `purged`. Test is still labelled `test_exploratory`
  because this period has been examined in earlier research.

Use `train_data/hue_regional_20261015_h6_h12_fixed_split_v1`; omit `fold` in the
reader and `--regional-fold` in training. The saved normalizer key `fixed` is an
implementation detail, not another temporal cross-validation scheme. Earlier
v1/v2/v3 regional datasets retain their historical EDA2 folds for traceability;
they have not been overwritten and are not the corrected split.

Builder defaults now use these dates, configurable with `--validation-start`,
`--test-start`, and `--end`. The `refit_train` column is metadata only and is not
used by the standard fixed-split training loader.

Water normalization is per station; absolute target normalization is per
station and horizon. Shared rain scalers use the union of timestamps present
in training input windows, each timestamp once. Forecast scalers use training
anchors only. No validation/test values are used to fit either.

For each sample the reader returns `persistence_scale` and
`persistence_offset`, satisfying:

```text
normalized_persistence = normalized_current_water * persistence_scale + persistence_offset
```

It also returns raw `target_mm`, `delta_mm`, `water_now_mm` and the anchor.
Later training must route these station-specific coefficients correctly;
this reader does not change the existing model's persistence buffers.
Raw missing values remain NaN in storage; standardized missing channels become
0 while binary masks remain explicit. An all-missing train channel gets mean=0,
scale=1 and an `all_missing` marker; its presence is not evidence of usefulness.

Forecast history is **anchor_only**. The reader withholds weather from the
first 95 timesteps by setting values/provenance to 0 and missing flags to 1;
the last timestep retains the forecast selected for that anchor. Metadata
strings, future targets and future observations are never dynamic input columns.

## Research limitations

Forecast uses only the existing ECMWF IFS A Lưới grid, `cell_selection=land`,
with the same retrospective +6h assumed release and 24h maximum age policy as
EDA2. All four windows come from one run; historical point values are not
substituted with observations. Hindcasts and unverified archives stay labelled.
Retrieval time after an anchor is not treated as observed operational availability.
Single-cell selection is rebuilt from Stage 3B's raw run/point artifacts, so
unavailability of other grid cells cannot remove an A Lưới sample.

The post-August period has already been examined and is not an independent
final test. EDA1 group/eligibility selection has also seen pre-cutoff data beyond
some early inner folds. Station-level units follow the project's mm convention;
physical quantities and vertical datums are not independently established for
all sensors. DEM elevation is not converted into a station zero.
