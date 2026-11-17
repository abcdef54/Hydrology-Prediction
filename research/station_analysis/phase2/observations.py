"""Non-destructive inventory of observation timing, coverage and rainfall meaning.

Canonical inventory hour t contains (t-1h,t], so it never includes an observation
whose measurement timestamp is later than t. Coverage is distinct from physical
rainfall totals and from proof that a reading was available operationally at t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.station_analysis.statistics import missing_runs


UNIT_TO_MM = {'mm': 1.0, 'cm': 10.0, 'm': 1000.0}


def _utc(values, name):
    series = pd.Series(values)
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        return pd.to_datetime(series, utc=True)
    for value in series.dropna():
        try:
            stamp = pd.Timestamp(value)
        except (ValueError, TypeError):
            continue
        if pd.isna(stamp):
            continue
        if stamp.tzinfo is None:
            raise ValueError(f'{name} must contain timezone-aware timestamps')
    return pd.to_datetime(series, utc=True, errors='coerce', format='mixed')


def _anchors(values):
    index = pd.DatetimeIndex(values)
    if index.tz is None or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError('Anchors must be ordered unique timezone-aware timestamps')
    index = index.tz_convert('UTC')
    if not index.equals(index.floor('h')):
        raise ValueError('Anchors must fall on exact hour boundaries')
    return index


def _contract(entry):
    if not isinstance(entry, dict):
        return False
    return (entry.get('confirmed') is True
            and bool(str(entry.get('source', '')).strip())
            and entry.get('kind') in {'interval_amount', 'rate', 'cumulative_counter'})


def verified_rain_hourly(series, anchors, contract):
    """Aggregate only a sourced, explicitly confirmed rainfall measurement contract.

    Input timestamps mark interval ends. Rate units must be mm/h; amount/counter
    units must be mm, cm or m. Non-hour-divisor intervals and intervals crossing
    an hourly boundary are rejected, rather than split by invented rainfall rates.
    Counter reset intervals remain missing: the rain before a reset is unknown.
    An hour is returned only when every expected interval has a valid reading.
    """
    index = _anchors(anchors)
    if not _contract(contract):
        raise ValueError('Rainfall accumulation semantics require confirmed kind and source')
    minutes = float(contract.get('interval_minutes', 0))
    if not np.isfinite(minutes) or minutes <= 0 or not (60 / minutes).is_integer():
        raise ValueError('A known interval dividing an hour exactly is required')
    values = series.copy()
    if not isinstance(values.index, pd.DatetimeIndex) or values.index.tz is None:
        raise ValueError('Rainfall timestamps must be timezone aware')
    values.index = values.index.tz_convert('UTC')
    if values.index.has_duplicates:
        raise ValueError('Duplicate rainfall timestamps must be resolved before accumulation')
    values = pd.to_numeric(values.sort_index(), errors='coerce').astype(float)
    values = values.where(np.isfinite(values) & values.ge(0))
    interval = pd.Timedelta(minutes=minutes)
    if len(values) and (values.index.as_unit('ns').asi8 % interval.value != 0).any():
        raise ValueError('Rainfall intervals may cross hour boundaries; allocation is unverified')
    kind, unit = contract['kind'], contract.get('unit')
    if kind == 'rate':
        if unit != 'mm/h':
            raise ValueError('Confirmed rain rate requires explicit mm/h units')
        amounts = values * minutes / 60
    else:
        if unit not in UNIT_TO_MM:
            raise ValueError('Confirmed rainfall amount/counter requires explicit length units')
        amounts = values * UNIT_TO_MM[unit]
        if kind == 'cumulative_counter':
            contiguous = values.index.to_series().diff().eq(interval)
            amounts = amounts.diff().where(contiguous)
            amounts = amounts.where(amounts.ge(0))
    grouped = amounts.resample('h', closed='right', label='right')
    count = grouped.count()
    total = grouped.sum(min_count=int(60 / minutes))
    total = total.where(count.eq(int(60 / minutes))).reindex(index)
    total.name = 'rain_hourly_verified_mm'
    return total


def strict_accumulation(hourly, windows=(1, 3, 6, 12, 24, 48)):
    """Past-only totals on a full hourly grid, retaining holes as missing values."""
    index = _anchors(hourly.index)
    result_index = (pd.date_range(index.min(), index.max(), freq='h') if len(index) else index)
    values = pd.to_numeric(hourly, errors='coerce').reindex(result_index)
    values = values.where(np.isfinite(values) & values.ge(0))
    result = pd.DataFrame(index=result_index)
    for hours in windows:
        if not isinstance(hours, int) or hours <= 0:
            raise ValueError('Accumulation windows must be positive whole hours')
        result[f'rain_sum_{hours}h'] = values.rolling(hours, min_periods=hours).sum()
    return result


def _gap_position(row, first, last):
    if pd.isna(first):
        return 'no_observations_in_requested_period'
    if row.end_utc < first:
        return 'leading'
    if row.start_utc > last:
        return 'trailing'
    return 'internal'


def audit_observations(raw, stations, anchors, config):
    """Return Stage-1 inventory tables without choosing rain/water relationships.

    All metadata stations remain visible. Raw sums are explicitly diagnostics
    when rainfall semantics have not been confirmed. Receipt/update timestamps
    describe this database's history, not an independently archived upstream feed.
    """
    index = _anchors(anchors)
    metadata = stations.copy()
    if 'station_id' not in metadata and 'id' in metadata:
        metadata = metadata.rename(columns={'id': 'station_id'})
    metadata['station_id'] = metadata.station_id.astype(str)
    if metadata.station_id.duplicated().any():
        raise ValueError('Station metadata IDs must be unique')
    frame = raw.copy()
    required = {'station_id', 'observed_at', 'depth', 'quality_status'}
    if not required <= set(frame):
        raise ValueError(f'Missing raw observation columns: {sorted(required-set(frame))}')
    frame['station_id'] = frame.station_id.astype(str)
    frame['observed_at'] = _utc(frame.observed_at, 'observed_at').to_numpy()
    # to_numpy preserves aware Timestamp objects, then normalize to one UTC dtype.
    frame['observed_at'] = pd.to_datetime(frame.observed_at, utc=True)
    frame['value'] = pd.to_numeric(frame.depth, errors='coerce').astype(float)
    frame['finite'] = np.isfinite(frame.value)
    for column in ['received_at', 'updated_at']:
        frame[column] = (_utc(frame[column], column).to_numpy() if column in frame
                         else pd.NaT)
        frame[column] = pd.to_datetime(frame[column], utc=True)
    known_ids = set(metadata.station_id)
    orphans = frame.loc[~frame.station_id.isin(known_ids)].groupby('station_id').size()
    orphan_table = orphans.rename('raw_records').reset_index()
    cadence_minutes = float(config.get('observation_expected_interval_minutes', 10))
    if cadence_minutes <= 0 or not (60/cadence_minutes).is_integer():
        raise ValueError('Expected observation interval must divide 60 minutes')
    expected_records = int(60/cadence_minutes)
    interval_ns = pd.Timedelta(minutes=cadence_minutes).value
    rain_limit = float(config.get('rainfall_record_screen_mm', 100))
    rain_contracts = config.get('rainfall_semantics', {})
    inventory, availability, gaps, monthly, semantics, alignment = [], [], [], [], [], []
    groups = {sid: part for sid, part in frame.groupby('station_id', sort=False)}
    for station in metadata.itertuples(index=False):
        sid = station.station_id
        kind = station.station_type
        part = groups.get(sid, frame.iloc[:0]).sort_values('observed_at')
        good_time = part.observed_at.notna()
        timed = part.loc[good_time].copy()
        timed['hour'] = timed.observed_at.dt.ceil('h')
        timed['off_grid'] = timed.observed_at.dt.as_unit('ns').astype('int64') % interval_ns != 0
        timed['duplicate'] = timed.duplicated('observed_at', keep=False)
        valid = timed.finite & timed.quality_status.eq('observed')
        if kind == 'rainfall':
            valid &= timed.value.ge(0)
        timed['valid'] = valid
        timed['valid_value'] = timed.value.where(valid)
        unit = getattr(station, 'measurement_unit', None)
        factor = UNIT_TO_MM.get(unit, np.nan)
        timed['valid_mm'] = timed.valid_value * factor
        timed['finite_raw_mm'] = timed.value.where(timed.finite) * factor
        timed['receipt_by_hour'] = timed.valid & timed.received_at.le(timed.hour)
        timed['receipt_after_hour'] = timed.valid & timed.received_at.gt(timed.hour)
        timed['receipt_unknown'] = timed.valid & timed.received_at.isna()
        timed['update_after_hour'] = timed.valid & timed.updated_at.gt(timed.hour)
        timed['finite_value'] = timed.finite.astype(int)
        aggregated = timed.groupby('hour').agg(
            raw_record_count=('value', 'size'), finite_value_count=('finite_value', 'sum'),
            valid_value_count=('valid', 'sum'), off_grid_count=('off_grid', 'sum'),
            duplicate_rows=('duplicate', 'sum'), receipt_by_anchor_count=('receipt_by_hour', 'sum'),
            receipt_after_anchor_count=('receipt_after_hour', 'sum'),
            receipt_unknown_count=('receipt_unknown', 'sum'), update_after_anchor_count=('update_after_hour', 'sum'),
        ).reindex(index, fill_value=0)
        aggregated['has_records'] = aggregated.raw_record_count.gt(0)
        aggregated['has_valid_value'] = aggregated.valid_value_count.gt(0)
        aggregated['expected_records'] = expected_records
        aggregated['complete_expected_grid'] = (
            aggregated.valid_value_count.eq(expected_records)
            & aggregated.raw_record_count.eq(expected_records)
            & aggregated.off_grid_count.eq(0) & aggregated.duplicate_rows.eq(0)
        )
        aggregated['water_mean_mm'] = np.nan
        aggregated['legacy_rain_sum_mm_diagnostic'] = np.nan
        aggregated['rain_hourly_verified_mm'] = np.nan
        verified = False
        if kind == 'water_level':
            means = timed.groupby('hour').valid_mm.mean().reindex(index)
            aggregated['water_mean_mm'] = means.where(aggregated.duplicate_rows.eq(0))
        elif kind == 'rainfall':
            # Reproduce the legacy sum of finite source values for diagnosis,
            # including negative values if present; it is never an approved total.
            totals = timed.groupby('hour').finite_raw_mm.sum(min_count=1).reindex(index)
            aggregated['legacy_rain_sum_mm_diagnostic'] = totals
            contract = rain_contracts.get(sid, {})
            verified = _contract(contract)
            if verified:
                series = pd.Series(timed.valid_value.to_numpy(), index=pd.DatetimeIndex(timed.observed_at))
                aggregated['rain_hourly_verified_mm'] = verified_rain_hourly(series, index, contract)
            previous = timed.valid_mm.shift()
            same_grid = timed.observed_at.diff().eq(pd.Timedelta(minutes=cadence_minutes))
            pairs = same_grid & timed.valid_mm.notna() & previous.notna()
            delta = timed.valid_mm - previous
            positive_pairs = pairs & timed.valid_mm.gt(0) & previous.gt(0)
            semantics.append({
                'station_id': sid, 'name': station.name, 'measurement_unit': unit,
                'semantics_status': 'confirmed_by_supplied_contract' if verified else 'unknown_do_not_build_rainfall_features',
                'contract_kind': contract.get('kind', 'unknown'), 'contract_source': contract.get('source', ''),
                'nondecreasing_pair_fraction': float(delta[pairs].ge(0).mean()) if pairs.any() else np.nan,
                'positive_unchanged_pair_fraction': float(delta[positive_pairs].eq(0).mean()) if positive_pairs.any() else np.nan,
                'negative_raw_values': int((part.finite & part.value.lt(0)).sum()),
                'record_values_above_screen': int(timed.valid_mm.gt(rain_limit).sum()),
                'screen_threshold_mm_per_record': rain_limit,
                'interpretation': 'empirical diagnostics cannot establish interval, rate, cumulative counter or reset semantics',
            })
        aggregated.index.name = 'timestamp_utc'
        aggregated.insert(0, 'station_type', kind)
        aggregated.insert(0, 'station_id', sid)
        aggregated['rainfall_semantics_confirmed'] = verified if kind == 'rainfall' else False
        availability.append(aggregated.reset_index())
        presence = pd.Series(np.where(aggregated.has_valid_value, 1.0, np.nan), index=index)
        missing = missing_runs(presence)
        first, last = presence.first_valid_index(), presence.last_valid_index()
        if first is None:
            first, last = pd.NaT, pd.NaT
        missing.insert(0, 'station_id', sid)
        missing.insert(1, 'station_type', kind)
        missing['position'] = [_gap_position(row, first, last) for row in missing.itertuples()]
        gaps.append(missing)
        active = presence.loc[first:last] if pd.notna(first) else presence.iloc[:0]
        delta_minutes = timed.observed_at.diff().dt.total_seconds().div(60).dropna()
        inventory.append({
            'station_id': sid, 'name': station.name, 'station_type': kind, 'measurement_unit': unit,
            'raw_records': len(part), 'finite_raw_records': int(part.finite.sum()),
            'invalid_or_missing_values': int((~part.finite).sum()),
            'invalid_timestamps': int((~good_time).sum()),
            'negative_values': int((part.finite & part.value.lt(0)).sum()),
            'source_missing_status': int(part.quality_status.eq('missing').sum()),
            'duplicate_timestamp_extra_rows': int(timed.duplicated('observed_at').sum()),
            'off_expected_grid_records': int(timed.off_grid.sum()),
            'first_observed_utc': timed.observed_at.min(), 'last_observed_utc': timed.observed_at.max(),
            'median_interval_minutes': float(delta_minutes.median()) if len(delta_minutes) else np.nan,
            'p95_interval_minutes': float(delta_minutes.quantile(.95)) if len(delta_minutes) else np.nan,
            'max_interval_minutes': float(delta_minutes.max()) if len(delta_minutes) else np.nan,
            'requested_hours': len(index), 'hours_with_records': int(aggregated.has_records.sum()),
            'hours_with_valid_values': int(aggregated.has_valid_value.sum()),
            'missing_rate': float(presence.isna().mean()) if len(presence) else np.nan,
            'active_span_hours': len(active),
            'active_span_missing_rate': float(active.isna().mean()) if len(active) else np.nan,
            'complete_grid_hours': int(aggregated.complete_expected_grid.sum()),
            'partial_or_invalid_grid_hours': int((aggregated.has_records & ~aggregated.complete_expected_grid).sum()),
            'longest_missing_hours': int(missing.missing_hours.max()) if len(missing) else 0,
            'receipt_after_anchor_records': int(aggregated.receipt_after_anchor_count.sum()),
            'updates_after_anchor_records': int(aggregated.update_after_anchor_count.sum()),
            'unit_convertible_to_mm': unit in UNIT_TO_MM,
            'rainfall_semantics_confirmed': verified,
        })
        month_labels = index.tz_convert(config.get('timezone', 'Asia/Ho_Chi_Minh')).strftime('%Y-%m')
        for month in pd.unique(month_labels):
            selected = aggregated.loc[month_labels == month]
            monthly.append({'station_id': sid, 'station_type': kind, 'month': month,
                            'expected_hours': len(selected), 'hours_with_valid_values': int(selected.has_valid_value.sum()),
                            'complete_grid_hours': int(selected.complete_expected_grid.sum()),
                            'missing_rate': float((~selected.has_valid_value).mean())})
        # Quantify the difference from Phase-1's forward-looking label convention.
        left_hour = timed.observed_at.dt.floor('h')
        left_counts = timed.loc[timed.valid].groupby(left_hour).size().reindex(index, fill_value=0)
        alignment.append({
            'station_id': sid, 'station_type': kind,
            'right_closed_hours_with_values': int(aggregated.has_valid_value.sum()),
            'left_closed_hours_with_values': int(left_counts.gt(0).sum()),
            'hours_with_different_record_counts': int(left_counts.ne(aggregated.valid_value_count).sum()),
            'phase1_left_label_readings_after_label': int((timed.valid
                & timed.observed_at.gt(left_hour) & left_hour.isin(index)).sum()),
            'canonical_window': '(anchor-1h,anchor]',
            'phase1_window': '[label,label+1h)',
            'availability_caveat': 'measurement-time alignment only; receipt and revision history do not reconstruct the original upstream as-of feed',
        })
    return {
        'observation_inventory': pd.DataFrame(inventory),
        'observation_availability': pd.concat(availability, ignore_index=True) if availability else pd.DataFrame(),
        'observation_missing_gaps': pd.concat(gaps, ignore_index=True) if gaps else pd.DataFrame(),
        'observation_monthly_coverage': pd.DataFrame(monthly),
        'rainfall_semantics_audit': pd.DataFrame(semantics),
        'hourly_alignment_audit': pd.DataFrame(alignment),
        'observation_orphans': orphan_table,
    }
