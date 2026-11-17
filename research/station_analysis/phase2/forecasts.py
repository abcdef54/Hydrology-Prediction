"""Inventory forecast vintages without claiming historical operational availability.

This module audits provenance and timestamp integrity only. It does not evaluate
forecast skill, correlations, or train a model. The legacy diagnostic selection
is deliberately separated from conservative, evidenced operational selection.
"""
from __future__ import annotations

import math
import json

import numpy as np
import pandas as pd

HOUR_NS = pd.Timedelta(hours=1).value
DEFAULT_WINDOWS = ((0, 3), (3, 6), (6, 12), (12, 24))


def _utc(values, *, allow_null=False):
    """Require absolute instants instead of silently interpreting naive local time."""
    for value in values:
        if pd.isna(value):
            if not allow_null:
                raise ValueError('Forecast timestamps must not be missing')
        elif pd.Timestamp(value).tzinfo is None:
            raise ValueError('Forecast timestamps require explicit timezones')
    return pd.to_datetime(values, utc=True)


def _windows(config):
    windows = config.get('forecast_windows', config.get('windows', DEFAULT_WINDOWS))
    result = []
    last = 0
    for a, b in windows:
        if isinstance(a, bool) or isinstance(b, bool) or int(a) != a or int(b) != b:
            raise ValueError('Forecast interval bounds must be integer hours')
        a, b = int(a), int(b)
        if a < 0 or b <= a or a < last:
            raise ValueError('Forecast accumulation intervals must be ordered and non-overlapping')
        result.append((a, b))
        last = b
    if not result:
        raise ValueError('At least one forecast accumulation interval is required')
    return result


def _run_integrity(runs, points, locations, config):
    known_locations = set(locations.location_id.astype(str))
    duplicates = set(runs.loc[runs.run_id.duplicated(keep=False), 'run_id'])
    all_points = {str(key): value for key, value in points.groupby('run_id', sort=False)}
    strict_kinds = set(config.get('strict_forecast_archive_kinds', ['operational_single_run']))
    rows, valid_values = [], {}
    # A repeated metadata identity cannot safely be resolved by inventing order.
    metadata = runs.drop_duplicates('run_id', keep='first').copy()
    for run in metadata.itertuples(index=False):
        part = all_points.get(run.run_id, points.iloc[:0])
        # pandas3 may retain microsecond resolution; lookup keys explicitly use
        # nanoseconds like Timestamp.value, independent of source precision.
        stamps = pd.DatetimeIndex(part.valid_at).as_unit('ns')
        values = pd.to_numeric(part.precipitation_mm, errors='coerce')
        finite = np.isfinite(values)
        negative = values.lt(0)
        duplicate_points = int(part.duplicated(['run_id', 'valid_at'], keep=False).sum())
        quality = part.quality_status.astype(str)
        quality_mismatch = ((quality.eq('forecast') & (~finite | negative))
                            | (quality.eq('missing') & values.notna())
                            | ~quality.isin(['forecast', 'missing']))
        whole_hour = stamps == stamps.floor('h')
        after_init = stamps > run.initialized_at
        unique_stamps = stamps.unique().sort_values()
        hourly_missing = 0
        if len(unique_stamps):
            expected = int((unique_stamps.max() - unique_stamps.min()) / pd.Timedelta(hours=1)) + 1
            hourly_missing = max(0, expected - len(unique_stamps))
        usable = finite & ~negative & quality.eq('forecast') & whole_hour & after_init
        metadata_ambiguous = run.run_id in duplicates or run.location_id not in known_locations
        if not duplicate_points and not metadata_ambiguous:
            valid_values[run.run_id] = dict(zip(stamps[usable].asi8, values[usable].astype(float)))
        else:
            valid_values[run.run_id] = {}
        kind = str(run.archive_kind)
        basis = str(run.availability_basis)
        hindcast = 'hindcast' in kind.lower() or 'reanalysis' in kind.lower()
        reasons = []
        payload_units = getattr(run, 'hourly_units', None)
        if isinstance(payload_units, str):
            try:
                payload_units = json.loads(payload_units)
            except (ValueError, TypeError):
                payload_units = None
        units_supplied = 'hourly_units' in metadata
        unit_verified = isinstance(payload_units, dict) and payload_units.get('precipitation') == 'mm'
        if units_supplied and not unit_verified:
            reasons.append('payload_precipitation_unit_not_verified_mm')
        utc_offset = getattr(run, 'utc_offset_seconds', 0)
        try:
            utc_offset = float(utc_offset)
        except (TypeError, ValueError):
            utc_offset = np.nan
        if pd.isna(utc_offset) or utc_offset != 0:
            reasons.append('payload_time_offset_not_verified_utc')
        if (units_supplied and not unit_verified) or not np.isfinite(utc_offset) or utc_offset != 0:
            valid_values[run.run_id] = {}
        if kind not in strict_kinds or hindcast:
            reasons.append('hindcast_or_unverified_archive_kind')
        if basis != 'observed_retrieval':
            reasons.append('availability_not_evidenced_by_retrieval')
        if pd.isna(run.available_at):
            reasons.append('available_at_missing')
        elif run.available_at < run.retrieved_at or run.available_at < run.initialized_at:
            reasons.append('availability_precedes_retrieval_or_initialization')
        if run.retrieved_at < run.initialized_at:
            reasons.append('retrieval_precedes_initialization')
        if metadata_ambiguous or duplicate_points:
            reasons.append('ambiguous_run_identity_or_duplicate_hours')
        if pd.isna(run.available_at):
            provenance = 'hindcast_assumption_only' if hindcast else 'archive_availability_unknown'
        elif not reasons:
            provenance = 'operational_retrieval_evidenced'
        else:
            provenance = 'timestamp_present_but_unverified'
        rows.append({
            'run_id': run.run_id, 'location_id': run.location_id, 'model': run.model,
            'initialized_at': run.initialized_at, 'retrieved_at': run.retrieved_at,
            'available_at': run.available_at, 'archive_kind': kind, 'availability_basis': basis,
            'grid_latitude': run.grid_latitude, 'grid_longitude': run.grid_longitude,
            'point_rows': len(part), 'valid_forecast_hours': int(usable.sum()),
            'missing_value_rows': int(values.isna().sum()), 'negative_value_rows': int(negative.sum()),
            'nonfinite_value_rows': int((values.notna() & ~finite).sum()),
            'quality_value_mismatch_rows': int(quality_mismatch.sum()),
            'duplicated_hour_rows': duplicate_points,
            'off_hour_valid_times': int((~whole_hour).sum()),
            'valid_times_not_after_initialization': int((~after_init).sum()),
            'missing_hours_between_first_last': hourly_missing,
            'first_valid_at': stamps.min() if len(stamps) else pd.NaT,
            'last_valid_at': stamps.max() if len(stamps) else pd.NaT,
            'max_lead_hours': float((stamps.max() - run.initialized_at) / pd.Timedelta(hours=1)) if len(stamps) else np.nan,
            'initialization_on_6h_cycle': bool(run.initialized_at == run.initialized_at.floor('6h')),
            'duplicate_metadata_id': run.run_id in duplicates,
            'unknown_location_id': run.location_id not in known_locations,
            'hindcast_or_reanalysis_label': hindcast,
            'strict_eligible_run': not reasons,
            'strict_exclusion_reason': ';'.join(reasons),
            'provenance_class': provenance,
            'hours_retrieval_after_init': float((run.retrieved_at - run.initialized_at) / pd.Timedelta(hours=1)),
            'hours_available_after_init': float((run.available_at - run.initialized_at) / pd.Timedelta(hours=1)) if pd.notna(run.available_at) else np.nan,
            'payload_precipitation_unit_verified_mm': unit_verified,
            'precipitation_unit_evidence': ('payload_hourly_units' if units_supplied else 'collector_parser_contract_only'),
            'payload_utc_offset_seconds': utc_offset,
        })
    columns = [
        'run_id', 'location_id', 'model', 'initialized_at', 'retrieved_at', 'available_at',
        'archive_kind', 'availability_basis', 'grid_latitude', 'grid_longitude', 'point_rows',
        'valid_forecast_hours', 'missing_value_rows', 'negative_value_rows', 'nonfinite_value_rows',
        'quality_value_mismatch_rows', 'duplicated_hour_rows', 'off_hour_valid_times',
        'valid_times_not_after_initialization', 'missing_hours_between_first_last',
        'first_valid_at', 'last_valid_at', 'max_lead_hours', 'initialization_on_6h_cycle',
        'duplicate_metadata_id', 'unknown_location_id', 'hindcast_or_reanalysis_label',
        'strict_eligible_run', 'strict_exclusion_reason', 'provenance_class',
        'hours_retrieval_after_init', 'hours_available_after_init',
        'payload_precipitation_unit_verified_mm', 'precipitation_unit_evidence', 'payload_utc_offset_seconds',
    ]
    return pd.DataFrame(rows, columns=columns), valid_values


def _anchor_inventory(run_audit, valid_values, locations, anchors, config, windows):
    lag = float(config.get('assumed_release_lag_hours', 6))
    max_age = float(config.get('max_forecast_age_hours', 24))
    primary_age = float(config.get('partial_forecast_max_age_hours', 12))
    if not all(math.isfinite(v) and v >= 0 for v in [lag, max_age, primary_age]) or max_age < lag:
        raise ValueError('Invalid forecast age or assumed release lag')
    rows = []
    horizons = sorted(set([6, 12, 24] + [b for _, b in windows]))
    lookahead = max(horizons)
    for location in locations.itertuples(index=False):
        location_runs = run_audit.loc[run_audit.location_id.eq(location.location_id)]
        models = list(location_runs.model.unique()) or ['no_runs']
        for model in models:
            frame = location_runs.loc[location_runs.model.eq(model)].sort_values(
                ['initialized_at', 'retrieved_at', 'run_id'], ascending=[True, False, True])
            records = list(frame.itertuples(index=False))
            initialized = np.array([run.initialized_at.value for run in records], dtype=np.int64)
            for anchor in anchors:
                anchor_ns = anchor.value
                lower = np.searchsorted(initialized, anchor_ns - int(max_age * HOUR_NS), side='left')
                for policy in ['legacy_assumed_release', 'strict_operational']:
                    policy_lag = lag if policy == 'legacy_assumed_release' else 0
                    upper = np.searchsorted(initialized, anchor_ns - int(policy_lag * HOUR_NS), side='right')
                    recent = records[lower:upper]
                    eligible = []
                    for run in recent:
                        if run.duplicate_metadata_id or run.duplicated_hour_rows or run.unknown_location_id:
                            continue
                        if policy == 'strict_operational':
                            if not run.strict_eligible_run or run.available_at > anchor:
                                continue
                        elif pd.notna(run.available_at) and run.available_at > anchor:
                            continue
                        eligible.append(run)
                    # Availability filtering precedes selecting the latest vintage.
                    eligible.sort(key=lambda run: (run.initialized_at.value, run.retrieved_at.value, run.run_id), reverse=True)
                    seen_cycles = set()
                    cycle_runs = []
                    for run in eligible:
                        if run.initialized_at not in seen_cycles:
                            cycle_runs.append(run)
                            seen_cycles.add(run.initialized_at)
                    selected, partial = None, None
                    for run in cycle_runs:
                        age = (anchor - run.initialized_at) / pd.Timedelta(hours=1)
                        point_map = valid_values[run.run_id]
                        if partial is None and age <= primary_age:
                            partial = run
                        if all(anchor_ns + h * HOUR_NS in point_map for h in range(1, lookahead + 1)):
                            selected = run
                            break
                    selected = selected or partial
                    row = {
                        'location_id': location.location_id, 'source_station_id': location.station_id,
                        'model': model, 'anchor_utc': anchor, 'policy': policy,
                        'eligible_cycles': len(cycle_runs), 'selected_run_id': None,
                        'selected_initialized_at': pd.NaT, 'selected_retrieved_at': pd.NaT,
                        'selected_available_at': pd.NaT, 'selected_archive_kind': None,
                        'selected_availability_basis': None, 'age_hours': np.nan,
                        'retrieved_after_anchor': False, 'hindcast_selected': False,
                        'availability_status': 'no_eligible_run',
                    }
                    point_map = {}
                    if selected is not None:
                        point_map = valid_values[selected.run_id]
                        row.update(
                            selected_run_id=selected.run_id,
                            selected_initialized_at=selected.initialized_at,
                            selected_retrieved_at=selected.retrieved_at,
                            selected_available_at=selected.available_at,
                            selected_archive_kind=selected.archive_kind,
                            selected_availability_basis=selected.availability_basis,
                            age_hours=float((anchor - selected.initialized_at) / pd.Timedelta(hours=1)),
                            retrieved_after_anchor=bool(selected.retrieved_at > anchor),
                            hindcast_selected=bool(selected.hindcast_or_reanalysis_label),
                            availability_status=('operational_retrieval_evidenced' if policy == 'strict_operational'
                                                 else selected.provenance_class),
                        )
                    for horizon in horizons:
                        row[f'complete_{horizon}h'] = bool(point_map) and all(
                            anchor_ns + h * HOUR_NS in point_map for h in range(1, horizon + 1))
                    for a, b in windows:
                        values = [point_map.get(anchor_ns + h * HOUR_NS, np.nan) for h in range(a + 1, b + 1)]
                        row[f'rain_{a}_{b}h_mm'] = float(np.sum(values)) if np.isfinite(values).all() else np.nan
                    rows.append(row)
    columns = [
        'location_id', 'source_station_id', 'model', 'anchor_utc', 'policy', 'eligible_cycles',
        'selected_run_id', 'selected_initialized_at', 'selected_retrieved_at', 'selected_available_at',
        'selected_archive_kind', 'selected_availability_basis', 'age_hours', 'retrieved_after_anchor',
        'hindcast_selected', 'availability_status',
        *[f'complete_{h}h' for h in horizons], *[f'rain_{a}_{b}h_mm' for a, b in windows],
    ]
    return pd.DataFrame(rows, columns=columns)


def audit_forecasts(runs, points, locations, anchors: pd.DatetimeIndex, config: dict):
    """Audit one frozen source snapshot under two explicitly different policies.

    Strict eligibility requires observed successful retrieval, an operational run
    label and ordered timestamps init <= retrieval <= availability <= anchor.
    Historical release assumptions and hindcasts can only enter the separate
    legacy diagnostic branch. Neither branch assesses forecast accuracy.

    Selection uses one vintage for every accumulation window: latest eligible
    complete 24h (or longer configured interval), with the production-style
    <=12h-old partial-run fallback. Hourly rainfall at valid_at t represents the
    preceding hourly interval, so (a,b] contributes exactly b-a timestamps.
    """
    if not isinstance(anchors, pd.DatetimeIndex) or anchors.tz is None:
        raise ValueError('Forecast anchors must be a timezone-aware DatetimeIndex')
    anchors = anchors.tz_convert('UTC')
    if anchors.has_duplicates or not anchors.is_monotonic_increasing or not anchors.equals(anchors.floor('h')):
        raise ValueError('Forecast anchors must be unique, sorted and on whole hours')
    windows = _windows(config)
    runs, points, locations = runs.copy(), points.copy(), locations.copy()
    if 'run_id' not in runs and 'id' in runs:
        runs = runs.rename(columns={'id': 'run_id'})
    if 'location_id' not in locations and 'id' in locations:
        locations = locations.rename(columns={'id': 'location_id'})
    required_runs = {'run_id', 'location_id', 'model', 'initialized_at', 'retrieved_at', 'available_at',
                     'availability_basis', 'archive_kind', 'grid_latitude', 'grid_longitude', 'checksum'}
    if not required_runs <= set(runs) or not {'run_id', 'valid_at', 'precipitation_mm', 'quality_status'} <= set(points):
        raise ValueError('Forecast snapshot is missing required columns')
    if not {'location_id', 'station_id', 'latitude', 'longitude'} <= set(locations):
        raise ValueError('Forecast locations are missing required columns')
    if locations.location_id.duplicated().any():
        raise ValueError('Forecast locations require unique IDs')
    for frame, columns in [(runs, ['run_id', 'location_id']), (points, ['run_id']),
                           (locations, ['location_id', 'station_id'])]:
        for column in columns:
            if frame[column].isna().any():
                raise ValueError(f'Forecast identity {column} must not be missing')
            frame[column] = frame[column].astype(str)
    for column in ['initialized_at', 'retrieved_at', 'available_at']:
        runs[column] = _utc(runs[column], allow_null=column == 'available_at')
    points['valid_at'] = _utc(points.valid_at)
    audit, values = _run_integrity(runs, points, locations, config)
    anchors_table = _anchor_inventory(audit, values, locations, anchors, config, windows)
    rows = []
    for keys, part in audit.groupby(['location_id', 'model', 'archive_kind', 'availability_basis'], dropna=False):
        location_id, model, kind, basis = keys
        rows.append({
            'location_id': location_id, 'model': model, 'archive_kind': kind, 'availability_basis': basis,
            'run_count': len(part), 'point_rows': int(part.point_rows.sum()),
            'valid_forecast_hours': int(part.valid_forecast_hours.sum()),
            'first_initialized_at': part.initialized_at.min(), 'last_initialized_at': part.initialized_at.max(),
            'strict_eligible_runs': int(part.strict_eligible_run.sum()),
            'missing_available_at_runs': int(part.available_at.isna().sum()),
            'hindcast_runs': int(part.hindcast_or_reanalysis_label.sum()),
            'grid_coordinate_pairs': len(part[['grid_latitude', 'grid_longitude']].drop_duplicates()),
        })
    inventory = pd.DataFrame(rows, columns=[
        'location_id', 'model', 'archive_kind', 'availability_basis', 'run_count', 'point_rows',
        'valid_forecast_hours', 'first_initialized_at', 'last_initialized_at', 'strict_eligible_runs',
        'missing_available_at_runs', 'hindcast_runs', 'grid_coordinate_pairs',
    ])
    coverage_rows = []
    for keys, part in anchors_table.groupby(['location_id', 'model', 'policy'], dropna=False):
        location_id, model, policy = keys
        row = {'location_id': location_id, 'model': model, 'policy': policy,
               'anchor_count': len(part), 'selected_anchors': int(part.selected_run_id.notna().sum()),
               'retrieved_after_anchor_count': int(part.retrieved_after_anchor.sum()),
               'hindcast_selected_count': int(part.hindcast_selected.sum())}
        for horizon in [6, 12, 24]:
            count = int(part[f'complete_{horizon}h'].sum())
            row[f'complete_{horizon}h_anchors'] = count
            row[f'complete_{horizon}h_fraction'] = count / len(part) if len(part) else np.nan
        row['age_hours_median'] = part.age_hours.median()
        row['age_hours_p95'] = part.age_hours.quantile(.95)
        coverage_rows.append(row)
    coverage = pd.DataFrame(coverage_rows, columns=[
        'location_id', 'model', 'policy', 'anchor_count', 'selected_anchors',
        'retrieved_after_anchor_count', 'hindcast_selected_count',
        *[column for h in [6, 12, 24] for column in [f'complete_{h}h_anchors', f'complete_{h}h_fraction']],
        'age_hours_median', 'age_hours_p95',
    ])
    interval_table = pd.DataFrame([
        {'start_offset_hours': a, 'end_offset_hours': b, 'hour_count': b-a,
         'interval_convention': '(anchor+a, anchor+b]',
         'valid_time_semantics': 'precipitation accumulated over preceding hour; provider documentation',
         'source_url': 'https://open-meteo.com/en/docs'} for a, b in windows])
    orphan_points = points.loc[~points.run_id.isin(set(runs.run_id))].copy()
    cycles, cycle_gaps = _cycle_inventory(audit)
    return {'forecast_run_integrity': audit, 'forecast_inventory': inventory,
            'forecast_anchor_availability': anchors_table, 'forecast_coverage': coverage,
            'forecast_intervals': interval_table, 'forecast_orphan_points': orphan_points,
            'forecast_cycle_inventory': cycles, 'forecast_cycle_gaps': cycle_gaps,
            'forecast_location_grids': _spatial_inventory(audit, locations)}


def _cycle_inventory(audit):
    summary, gaps = [], []
    for keys, part in audit.groupby(['location_id', 'model'], dropna=False):
        location_id, model = keys
        cycles = pd.DatetimeIndex(part.initialized_at.unique()).sort_values()
        deltas = pd.Series(cycles).diff().dt.total_seconds().dropna() / 3600
        expected = int((cycles[-1]-cycles[0]) / pd.Timedelta(hours=6))+1 if len(cycles) else 0
        summary.append({
            'location_id': location_id, 'model': model, 'run_vintages': len(part),
            'distinct_initializations': len(cycles), 'expected_6h_cycles_within_span': expected,
            'missing_6h_cycles_within_span': max(0, expected-len(cycles)),
            'initializations_with_multiple_vintages': int(part.groupby('initialized_at').size().gt(1).sum()),
            'median_initialization_interval_hours': deltas.median(),
            'max_initialization_interval_hours': deltas.max(),
        })
        for left, right in zip(cycles[:-1], cycles[1:]):
            hours = float((right-left) / pd.Timedelta(hours=1))
            if hours != 6:
                gaps.append({'location_id': location_id, 'model': model,
                             'previous_initialized_at': left, 'next_initialized_at': right,
                             'elapsed_hours': hours,
                             'missing_expected_6h_cycles': max(0, int(hours/6)-1)})
    return pd.DataFrame(summary, columns=[
        'location_id', 'model', 'run_vintages', 'distinct_initializations',
        'expected_6h_cycles_within_span', 'missing_6h_cycles_within_span',
        'initializations_with_multiple_vintages', 'median_initialization_interval_hours',
        'max_initialization_interval_hours',
    ]), pd.DataFrame(gaps, columns=[
        'location_id', 'model', 'previous_initialized_at', 'next_initialized_at',
        'elapsed_hours', 'missing_expected_6h_cycles',
    ])


def _spatial_inventory(audit, locations):
    rows = []
    all_grids = audit[['location_id', 'grid_latitude', 'grid_longitude']].drop_duplicates()
    for grid in all_grids.itertuples(index=False):
        location = locations.loc[locations.location_id.eq(grid.location_id)]
        requested_lat, requested_lon = (location.iloc[0].latitude, location.iloc[0].longitude) if len(location) else (np.nan, np.nan)
        valid_coordinates = all(pd.notna(v) and np.isfinite(v) for v in [
            requested_lat, requested_lon, grid.grid_latitude, grid.grid_longitude])
        distance = np.nan
        if valid_coordinates:
            lat_a, lat_b = math.radians(requested_lat), math.radians(grid.grid_latitude)
            delta_lat, delta_lon = lat_b-lat_a, math.radians(grid.grid_longitude-requested_lon)
            haversine = math.sin(delta_lat/2)**2 + math.cos(lat_a)*math.cos(lat_b)*math.sin(delta_lon/2)**2
            distance = 6371 * 2 * math.asin(math.sqrt(min(1, max(0, haversine))))
        peers = all_grids.loc[all_grids.grid_latitude.eq(grid.grid_latitude)
                             & all_grids.grid_longitude.eq(grid.grid_longitude), 'location_id'].unique()
        rows.append({
            'location_id': grid.location_id, 'requested_latitude': requested_lat,
            'requested_longitude': requested_lon, 'grid_latitude': grid.grid_latitude,
            'grid_longitude': grid.grid_longitude, 'request_to_grid_distance_km': distance,
            'locations_sharing_returned_grid': len(peers),
            'shared_location_ids': ';'.join(sorted(peers)),
            'interpretation': 'grid-model precipitation; not a gauge measurement or citywide spatial mean',
        })
    return pd.DataFrame(rows, columns=[
        'location_id', 'requested_latitude', 'requested_longitude', 'grid_latitude',
        'grid_longitude', 'request_to_grid_distance_km', 'locations_sharing_returned_grid',
        'shared_location_ids', 'interpretation',
    ])
