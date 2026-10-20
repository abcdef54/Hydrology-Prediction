"""Resolve missing weather anchors from an older eligible archived run.

Writes a curated weather-only product, never overwrites raw forecasts in DB and
never joins observations/targets into a training dataset.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.weather_forecast.analyze_dataset_readiness import WINDOWS, read_times, sequence_audit


def fill_from_older(primary, fallback):
    """Replace an incomplete anchor with one complete older run; no run mixing."""
    if not primary.index.equals(fallback.index):
        raise ValueError('Anchor indexes differ')
    columns=[f'forecast_{a}_{b}h' for a,b in WINDOWS]
    needed=primary[columns].isna().any(axis=1)
    available=fallback[columns].notna().all(axis=1)
    replace=needed & available
    selected=primary.copy()
    selected.loc[replace,primary.columns]=fallback.loc[replace,primary.columns]
    cycle=pd.to_datetime(selected.initialized_at,utc=True)
    age=pd.Series((selected.index-cycle).dt.total_seconds().to_numpy()/3600,index=selected.index)
    nonnull=cycle.notna()
    if ((age[nonnull]<6)|(age[nonnull]>24)).any():
        raise ValueError('Selected run violates assumed availability or max age')
    primary_cycle=pd.to_datetime(primary.initialized_at,utc=True)
    known=replace & primary_cycle.notna()
    if (cycle[known]>primary_cycle[known]).any():
        raise ValueError('Replacement must not use a newer run')
    selected['age_hours']=age
    selected['was_filled']=replace
    selected['original_initialized_at']=primary.initialized_at
    selected['assumed_available_at']=cycle+pd.Timedelta(hours=6)
    selected['availability_basis']='assumed_6h_lag'
    for col in columns:
        selected[col+'_missing']=selected[col].isna()
    # Existing complete rows retain exactly their original values.
    pd.testing.assert_frame_equal(selected.loc[~needed,columns],primary.loc[~needed,columns])
    return selected


def main():
    root=Path(__file__).resolve().parents[2]
    source=root/'reports/weather_collection/dataset_readiness'
    output=root/'reports/weather_collection/resolved_forecasts'
    output.mkdir(parents=True,exist_ok=True)
    meta=json.loads((source/'manifest.json').read_text())
    cube_path=source/'forecast_runs_hourly.csv.gz'
    checksum=hashlib.sha256(cube_path.read_bytes()).hexdigest()
    if checksum!=meta['full_cube_sha256'] or not meta['part_2_complete']:
        raise ValueError('Full-run source differs from completed audit')
    dataset=root/'train_data/da_vien_1h_2026-10-15'
    splits={key:read_times(dataset/f'{key}.csv') for key in ['train','val','test']}
    for key in splits:
        if hashlib.sha256((dataset/f'{key}.csv').read_bytes()).hexdigest()!=meta['split_sha256'][f'{key}.csv']:
            raise ValueError('Dataset split changed since audit')
    diagnostic_path=source/'diagnostic_forecast_windows.csv.gz'
    diagnostics=pd.read_csv(diagnostic_path,index_col=0)
    diagnostics.index=pd.to_datetime(diagnostics.index,utc=True)
    primary=diagnostics.filter(like='primary_').rename(columns=lambda c:c.removeprefix('primary_'))
    fallback=diagnostics.filter(like='fallback_').rename(columns=lambda c:c.removeprefix('fallback_'))
    for frame in [primary,fallback]:
        frame['initialized_at']=pd.to_datetime(frame.initialized_at,utc=True)
    selected=fill_from_older(primary,fallback)
    cube=pd.read_csv(cube_path)
    for key in ['initialized_at','valid_at','retrieved_at']:
        cube[key]=pd.to_datetime(cube[key],utc=True)
    groups={stamp:group.set_index('valid_at') for stamp,group in cube.groupby('initialized_at')}
    provenance=[]
    filled_hours=[]
    # Check every selected window against the actual values of its ONE source run.
    for anchor,row in selected.iterrows():
        if pd.isna(row.initialized_at):continue
        group=groups[row.initialized_at]
        stamps=pd.date_range(anchor+pd.Timedelta(hours=1),periods=24,freq='h')
        future=group.reindex(stamps)
        values=future.precipitation_mm
        for a,b in WINDOWS:
            expected=values.iloc[a:b].sum(min_count=b-a)
            actual=row[f'forecast_{a}_{b}h']
            if not (pd.isna(expected) and pd.isna(actual)) and not np.isclose(expected,actual,rtol=1e-10,atol=1e-9):
                raise ValueError('Window does not match its source run')
        info=group.iloc[0]
        provenance.append(dict(anchor=anchor,run_id=info.run_id,source_retrieved_at=info.retrieved_at,
                               archive_kind=info.archive_kind))
        if row.was_filled:
            if not values.notna().all():raise ValueError('Filled run has missing future hours')
            for stamp,value in values.items():
                filled_hours.append(dict(anchor_utc=anchor,source_initialized_at=row.initialized_at,
                    valid_at=stamp,precipitation_mm=value,source_run_id=info.run_id))
    selected=selected.join(pd.DataFrame(provenance).set_index('anchor'))
    selected.to_csv(output/'weather_forecast_resolved.csv.gz',index_label='anchor_utc',compression='gzip')
    filled=selected[selected.was_filled].copy()
    filled['anchor_local']=filled.index.tz_convert('Asia/Ho_Chi_Minh')
    filled.to_csv(output/'filled_anchors.csv',index_label='anchor_utc')
    pd.DataFrame(filled_hours).to_csv(output/'filled_hourly_values.csv',index=False)
    columns=[f'forecast_{a}_{b}h' for a,b in WINDOWS]
    complete=selected[columns].notna().all(axis=1)
    coverage,_=sequence_audit(splits,complete,None,output)
    if not (coverage.baseline_sequences==coverage.complete_weather_all_96).all():
        raise ValueError('Some baseline sequences still contain missing forecast')
    manifest=dict(source_cube_sha256=checksum,
        source_diagnostics_sha256=hashlib.sha256(diagnostic_path.read_bytes()).hexdigest(),
        split_sha256=meta['split_sha256'],filled_anchors=int(selected.was_filled.sum()),
        remaining_incomplete_anchors=int((~complete).sum()),
        policy='Retain complete primary; replace incomplete anchor with one older run, age 6–24h; all next24h finite.',
        availability_basis='assumed_6h_lag; historical publication not independently verified',
        raw_forecasts_preserved=True,training_dataset_created=False)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    lines=['# Forecast đã xử lý các mốc thiếu','',
        f'Đã điền {manifest["filled_anchors"]} anchor; còn {manifest["remaining_incomplete_anchors"]} anchor thiếu.',
        'Tất cả cửa sổ của một anchor lấy từ cùng một bản cũ, có đủ 24 giờ tương lai. '
        'Không dùng mưa quan trắc để điền. Giữ giờ khởi tạo, độ cũ, run_id, archive_kind và cờ was_filled.', '',
        'Đây là bảng thời tiết đã xử lý, **chưa phải dataset training**. '
        'Response/bản dự báo gốc trong PostgreSQL vẫn giữ nguyên để không gán giá trị của bản cũ sang bản khác.', '',
        '| Split | Sequence ban đầu | Đủ forecast cả 96 giờ sau điền |','|---|---:|---:|']
    for r in coverage.itertuples():lines.append(f'| {r.split} | {r.baseline_sequences} | {r.complete_weather_all_96} |')
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    print('Filled anchors:',len(filled));print(coverage.to_string(index=False));print('Output:',output)


if __name__=='__main__':main()
