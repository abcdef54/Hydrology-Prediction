# Offline evaluation of regional water-level predictions

This evaluator reads **existing prediction and observation artifacts**. It does not train, fit a threshold, invoke inference, or alter model/data artifacts. It provides regression, large-rise detection and observation-defined event analysis. The rise event is not a flood warning and no official water-level warning threshold is inferred.

## Usage

```bash
.venv/bin/python -m src.ai.hydrology_evaluation \
  --predictions path/to/validation_predictions.parquet \
  --observations train_data/hue_regional_20261015_h6_h12_fixed_split_v1/water.parquet \
  --output-dir reports/model_training/my_evaluation \
  --event-window-hours 12 \
  --rise-threshold-mm 200 \
  --event-merge-gap-hours 6 \
  --warning-delta-mm 200 \
  --warning-merge-gap-hours 1 \
  --tolerance-mm 20
```

Use the same predeclared arguments and observation coverage when comparing models. Prediction parquet must contain `station_id`, `anchor_utc`, `target_utc`, `horizon_hours`, `current_mm`, `actual_mm`, `predicted_mm`, one unique row per station/anchor/horizon. Horizons must have contiguous hourly target coverage. `target_utc` must equal issue time plus horizon. Observations are hourly wide parquet (`timestamp_utc` and one column per station) or long parquet (`timestamp_utc`, `station_id`, `water_level_mm`). UTC timestamps are normalized by pandas. Predictions, labels, current levels and observations must match within 0.01 mm, including every anchor/target. Duplicate keys, missing observations at anchors/targets, nonfinite predictions, inconsistent timestamps and gaps in forecast coverage raise errors. `water.parquet` contains dataset hourly values, not verified raw sensor records.

## Window-level evaluation

Regression: pooled, per horizon, per station/horizon MAE, RMSE, bias, absolute error p90 and count >200 mm; regime tables contain observed/predicted changes and ratio for observed rises >200. Labels use **actual future change >200 mm**, not absolute change. Detection is from predicted delta, with strict precision, recall, F1 and sklearn average precision (AP). `precision_recall.csv` gives station/horizon PR points; AP is undefined when there are no positives. FP categories are based on observed delta relative to the configured threshold: within 20 mm below the threshold; 20–100 mm below; more than 100 mm below. Conditional mean/median forecast error are also exported.

Gray-zone sensitivity ignores true changes in the closed interval [threshold−tolerance, threshold+tolerance], then computes precision and recall on the remaining samples, with predicted-delta warning threshold unchanged. This changes the evaluated population and is not a calibrated operating threshold. Use an independent calibration period and held-out evaluation; repeated validation inspection is exploratory.

## Observation-defined events

The user-selected research rule flags observation time `t` when `H(t)−H(t−12h)>200 mm`, with all 13 hourly observation values finite. The window and threshold are configurable. Flagged times for a station are merged when there are no more than six consecutive unflagged hours between them (configurable). An unknown/missing observation or invalid rolling window always breaks a cluster. Each station receives one shared event list; there is no duplicate event merely because a forecast has two horizons. Events are not cross-station merged. Results include flagged start/end, maximum observed level and time in the event, and largest rolling rise.

Events are constructed only within each station's **common target-time range across horizons**, with preceding observations used to compute the first rolling value. A flag close to the coverage edge or to missing observations is marked censored. The rule is conservative and may not identify all uncertain boundaries. A 12h rolling rise >200 mm is a research event, not automatically a flood, and predicted +6h delta >200 mm and a 12h observed event are different quantities.

## Warnings and matching

For each station, forecast rows with predicted delta above `--warning-delta-mm` form warning episodes; issue times up to one hour apart are grouped by default, across horizons in `combined` mode. A warning episode is eligible for an event only when at least one of its forecast **target times** lies within the observed event interval. Eligible episode/event pairs are greedily matched one-to-one by earliest eligible issue time, then event start. This is deterministic but not necessarily maximum-cardinality and may differ from another operational matching policy. One warning episode cannot claim multiple events; one event cannot claim multiple episodes.

Warning statuses: `matched`, `unmatched_competing` (eligible for an event but not selected due to one-to-one matching), and `unmatched_no_overlap`. The last category is **not automatically a false alarm**: an event can be outside the observed/evaluated period, hidden by missing observations, not captured by the 12-hour rule, or a warning may have a real effect outside the permitted interval. Inspect censoring and coverage before using it operationally. Matching is recalculated separately for +6h, +12h and their combined warning stream; all modes use the *same observed event list and IDs*.

For a match, `lead_to_event_start_hours = event_start − issue_time`; negative values are late. `lead_to_peak_hours` uses the observed maximum level within the event. The selected issue is the earliest eligible forecast in the matched warning episode, which can differ from the episode's first issue. For peak error, use a forecast with target exactly equal to observed event peak time, choosing the latest available issue within the evaluated horizon(s); absence is reported missing. This is distinct from the peak of a predicted hydrograph or the peak change between anchor and target.

## Outputs and limitations

- `regression_detection.csv`, `regimes.csv`, `precision_recall.csv`.
- `events_{combined,h6,h12}.csv`, `warnings_*.csv`, `matches_*.csv`, `event_summary_*.csv`. Station IDs with leading zeros should be loaded as strings (`dtype={'station_id':str}`).
- `config.json` records policy, coverage, artifact paths and SHA-256 hashes. Store it with the outputs.

Censored events are marked, with both all-event and uncensored counts. Missing observations prevent rolling-event construction; no reliable denominator exists for intervals without data. The evaluator does not supply flood-level thresholds, independent hydrological flood delineation, spatial event consolidation, calibrated thresholds or prospective operational performance. Validate policy and sensitivity, then freeze it before comparing on untouched future events. An observation series with retrospectively reconstructed or questionable readings needs independent quality review.
