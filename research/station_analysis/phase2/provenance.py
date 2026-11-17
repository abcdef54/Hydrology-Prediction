"""Trace retained dataset weather columns back to the recorded forecast vintage.

A matching archived value proves reconstruction consistency only. It never proves
that a hindcast or retrospectively retrieved file existed at its historic anchor.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .forecasts import HOUR_NS, _run_integrity, _utc, _windows

ROW_COLUMNS = [
    'dataset', 'dataset_path', 'anchor_utc', 'split_memberships', 'provenance_present',
    'source_run_id', 'source_run_found', 'duplicate_provenance_anchor',
    'source_initialized_at', 'source_retrieved_at', 'archive_kind', 'availability_basis',
    'db_archive_kind', 'db_availability_basis', 'initialized_at_agrees_with_db',
    'retrieved_at_agrees_with_db', 'archive_kind_agrees_with_db',
    'source_retrieved_after_anchor', 'hindcast_label', 'assumed_availability',
    'historical_availability_verified', 'release_lag_satisfied', 'max_age_satisfied',
    'calculated_age_hours', 'provenance_age_consistent', 'dataset_age_consistent',
    'fallback_flag_valid', 'was_filled', 'compared_forecast_values',
    'mismatched_forecast_values', 'feature_values_match_db_vintage',
    'feature_masks_match_values', 'compared_feature_columns', 'reconstruction_interpretation',
]


def _timestamp_column(frame):
    for name in ['timestamp_utc', 'anchor_utc']:
        if name in frame:
            return name
    raise ValueError('Dataset/provenance requires timestamp_utc or anchor_utc')


def _same_number(left, right):
    if pd.isna(left) and pd.isna(right):
        return True
    try:
        return bool(np.isfinite(left) and np.isfinite(right) and np.isclose(left, right, atol=1e-6, rtol=1e-7))
    except TypeError:
        return False


def _summary(dataset, dataset_path, split, frame, status='audited'):
    columns = [
        'provenance_present', 'source_run_found', 'duplicate_provenance_anchor',
        'initialized_at_agrees_with_db', 'retrieved_at_agrees_with_db', 'archive_kind_agrees_with_db',
        'source_retrieved_after_anchor', 'hindcast_label', 'assumed_availability',
        'historical_availability_verified', 'release_lag_satisfied', 'max_age_satisfied',
        'provenance_age_consistent', 'dataset_age_consistent', 'fallback_flag_valid', 'was_filled',
        'feature_values_match_db_vintage', 'feature_masks_match_values',
    ]
    row = {'dataset': dataset, 'dataset_path': dataset_path, 'split': split,
           'anchor_count': len(frame), 'status': status}
    for column in columns:
        row[f'{column}_rows'] = int(frame[column].fillna(False).sum()) if column in frame else 0
    row['compared_forecast_value_count'] = int(frame.compared_forecast_values.sum()) if 'compared_forecast_values' in frame else 0
    row['mismatched_forecast_value_count'] = int(frame.mismatched_forecast_values.sum()) if 'mismatched_forecast_values' in frame else 0
    return row


def audit_dataset_provenance(dataset_dirs, runs, config, points=None):
    """Audit saved per-hour provenance and optional exact feature reconstruction.

    Returns a row per dataset/hour plus summaries for the full hourly table and
    its named split memberships. Purged hourly rows are retained as unassigned.
    No model or dataset file is changed. Missing source evidence fails closed.
    """
    runs = runs.copy()
    if 'run_id' not in runs and 'id' in runs:
        runs = runs.rename(columns={'id': 'run_id'})
    runs['run_id'] = runs.run_id.astype(str)
    for column in ['initialized_at', 'retrieved_at', 'available_at']:
        runs[column] = _utc(runs[column], allow_null=column == 'available_at')
    ambiguous_runs = set(runs.loc[runs.run_id.duplicated(keep=False), 'run_id'])
    lookup = runs.drop_duplicates('run_id', keep='first').set_index('run_id')
    # Reuse the strict provenance rule, including payload unit checks, without
    # introducing a second definition of historically evidenced availability.
    locations = pd.DataFrame({'location_id': runs.location_id.drop_duplicates()})
    if points is None:
        audit_points = pd.DataFrame(columns=['run_id', 'valid_at', 'precipitation_mm', 'quality_status'])
    else:
        audit_points = points.copy()
        audit_points['run_id'] = audit_points.run_id.astype(str)
    audit_points['valid_at'] = _utc(audit_points.valid_at)
    run_audit, point_maps = _run_integrity(runs, audit_points, locations, config)
    strict_run_ids = set(run_audit.loc[run_audit.strict_eligible_run, 'run_id'])
    windows = _windows(config)
    lag = float(config.get('assumed_release_lag_hours', 6))
    max_age = float(config.get('max_forecast_age_hours', 24))
    all_rows, summaries = [], []
    reconstruction_cache = {}
    for directory in dataset_dirs:
        directory = Path(directory)
        dataset_path, dataset = str(directory.resolve()), directory.name
        if not (directory / 'hourly.csv').is_file():
            summaries.append(_summary(dataset, dataset_path, 'all_hourly', pd.DataFrame(), 'hourly_file_missing'))
            continue
        header = pd.read_csv(directory/'hourly.csv', nrows=0).columns
        time_column = 'timestamp_utc' if 'timestamp_utc' in header else header[0]
        feature_columns = [c for c in header if c.startswith('weather_forecast_')]
        hourly = pd.read_csv(directory/'hourly.csv', usecols=[time_column, *feature_columns])
        hourly['anchor_utc'] = _utc(hourly[time_column])
        if hourly.anchor_utc.duplicated().any():
            summaries.append(_summary(dataset, dataset_path, 'all_hourly', pd.DataFrame(), 'duplicate_hourly_anchor'))
            continue
        hourly = hourly.set_index('anchor_utc')
        provenance_path = directory/'weather_provenance.csv.gz'
        if provenance_path.is_file():
            provenance = pd.read_csv(provenance_path, dtype={'run_id': str})
            provenance['anchor_utc'] = _utc(provenance[_timestamp_column(provenance)])
        else:
            provenance = pd.DataFrame(columns=['anchor_utc'])
        duplicate_anchors = set(provenance.loc[provenance.anchor_utc.duplicated(keep=False), 'anchor_utc'])
        provenance = provenance.drop_duplicates('anchor_utc', keep='first').set_index('anchor_utc')
        for column in ['initialized_at', 'source_retrieved_at']:
            if column in provenance:
                provenance[column] = _utc(provenance[column], allow_null=True)
        memberships = {}
        for split in ['train', 'val', 'validation', 'test', 'reference_validation']:
            split_path = directory/f'{split}.csv'
            if split_path.is_file():
                split_header = pd.read_csv(split_path, nrows=0).columns
                split_time = 'timestamp_utc' if 'timestamp_utc' in split_header else split_header[0]
                split_frame = pd.read_csv(split_path, usecols=[split_time])
                memberships[split] = set(_utc(split_frame[split_time]))
        rows = []
        for anchor, values in hourly.iterrows():
            present = anchor in provenance.index
            source = provenance.loc[anchor] if present else pd.Series(dtype=object)
            run_id = source.get('run_id')
            found = pd.notna(run_id) and str(run_id) in lookup.index and str(run_id) not in ambiguous_runs
            db_run = lookup.loc[str(run_id)] if found else pd.Series(dtype=object)
            initialized = source.get('initialized_at', pd.NaT)
            retrieved = source.get('source_retrieved_at', pd.NaT)
            age = float((anchor-initialized) / pd.Timedelta(hours=1)) if pd.notna(initialized) else np.nan
            basis = str(source.get('availability_basis', 'unknown'))
            kind = str(source.get('archive_kind', 'unknown'))
            filled = source.get('was_filled', None)
            valid_filled = isinstance(filled, (bool, np.bool_)) or str(filled).lower() in {'true', 'false'}
            was_filled = bool(filled) if isinstance(filled, (bool, np.bool_)) else str(filled).lower() == 'true'
            declared_init_ok = bool(found and pd.notna(initialized) and initialized == db_run.initialized_at)
            declared_retrieval_ok = bool(found and pd.notna(retrieved) and retrieved == db_run.retrieved_at)
            declared_kind_ok = bool(found and kind == str(db_run.archive_kind))
            strict = bool(found and str(run_id) in strict_run_ids and db_run.available_at <= anchor
                          and anchor not in duplicate_anchors and declared_init_ok and declared_retrieval_ok
                          and declared_kind_ok and basis == 'observed_retrieval')
            point_map = point_maps.get(str(run_id), {})
            compared, mismatches, masks_ok = 0, 0, True
            compared_columns = []
            if points is not None and found and anchor not in duplicate_anchors:
                for a, b in windows:
                    name = f'weather_forecast_rain_{a}_{b}h'
                    if name not in values:
                        continue
                    key = (str(run_id), anchor.value, a, b)
                    if key not in reconstruction_cache:
                        hours = [point_map.get(anchor.value+h*HOUR_NS, np.nan) for h in range(a+1, b+1)]
                        reconstruction_cache[key] = float(np.sum(hours)) if np.isfinite(hours).all() else np.nan
                    expected = reconstruction_cache[key]
                    compared += 1
                    compared_columns.append(name)
                    mismatches += int(not _same_number(values[name], expected))
                    mask_column = name + '_missing'
                    if mask_column in values:
                        masks_ok &= _same_number(values[mask_column], float(pd.isna(values[name])))
            rows.append({
                'dataset': dataset, 'dataset_path': dataset_path, 'anchor_utc': anchor,
                'split_memberships': ';'.join(name for name, timestamps in memberships.items() if anchor in timestamps) or 'unassigned_hourly',
                'provenance_present': present, 'source_run_id': str(run_id) if pd.notna(run_id) else None,
                'source_run_found': bool(found), 'duplicate_provenance_anchor': anchor in duplicate_anchors,
                'source_initialized_at': initialized, 'source_retrieved_at': retrieved,
                'archive_kind': kind, 'availability_basis': basis,
                'db_archive_kind': db_run.get('archive_kind'), 'db_availability_basis': db_run.get('availability_basis'),
                'initialized_at_agrees_with_db': declared_init_ok,
                'retrieved_at_agrees_with_db': declared_retrieval_ok,
                'archive_kind_agrees_with_db': declared_kind_ok,
                'source_retrieved_after_anchor': bool(pd.notna(retrieved) and retrieved > anchor),
                'hindcast_label': 'hindcast' in kind.lower() or 'reanalysis' in kind.lower(),
                'assumed_availability': 'assum' in basis.lower(),
                'historical_availability_verified': strict,
                'release_lag_satisfied': bool(np.isfinite(age) and age >= lag),
                'max_age_satisfied': bool(np.isfinite(age) and age <= max_age),
                'calculated_age_hours': age,
                'provenance_age_consistent': bool(np.isfinite(age)) and _same_number(source.get('age_hours', np.nan), age) if present else False,
                'dataset_age_consistent': bool(np.isfinite(age)) and _same_number(values.get('weather_forecast_age_hours', np.nan), age) if present else False,
                'fallback_flag_valid': valid_filled if present else False, 'was_filled': was_filled,
                'compared_forecast_values': compared, 'mismatched_forecast_values': mismatches,
                'feature_values_match_db_vintage': compared > 0 and mismatches == 0,
                'feature_masks_match_values': bool(compared > 0 and masks_ok),
                'compared_feature_columns': ';'.join(compared_columns),
                'reconstruction_interpretation': 'value match proves archive consistency, not historical availability',
            })
        frame = pd.DataFrame(rows, columns=ROW_COLUMNS)
        all_rows.append(frame)
        summaries.append(_summary(dataset, dataset_path, 'all_hourly', frame))
        for split, timestamps in memberships.items():
            summaries.append(_summary(dataset, dataset_path, split, frame.loc[frame.anchor_utc.isin(timestamps)]))
        unassigned = frame.loc[frame.split_memberships.eq('unassigned_hourly')]
        if len(unassigned):
            summaries.append(_summary(dataset, dataset_path, 'unassigned_hourly', unassigned))
    summary_columns = list(_summary('', '', '', pd.DataFrame()))
    return {'dataset_forecast_rows': pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(columns=ROW_COLUMNS),
            'dataset_forecast_integrity': pd.DataFrame(summaries, columns=summary_columns)}
