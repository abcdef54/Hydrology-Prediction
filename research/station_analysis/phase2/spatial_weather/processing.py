"""Build deduplicated, provenance-preserving research tables and common-run features."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from .collection import flatten_records,grid_identity,classify


def prepare_archive(folder,snapshot):
    folder=Path(folder);snapshot=Path(snapshot)
    records=[]
    for path in sorted((folder/'raw').glob('*_land_*.json')):
        record=json.loads(path.read_text())
        record['_source_path']=str(path)
        records.append(record)
    runs,points,mapping=flatten_records(records)
    old=pd.read_parquet(snapshot/'forecast_runs.parquet')
    old_points=pd.read_parquet(snapshot/'forecast_points.parquet')
    source=old[old.location_id.str.contains('750007')].sort_values('retrieved_at').drop_duplicates('initialized_at',keep='last')
    existing={(r.location_id,r.initialized_at):r.run_id for r in runs.itertuples()}
    rows=[];p_rows=[];m_rows=[];checks=[]
    for r in source.itertuples():
        gid=grid_identity(r.model,r.grid_latitude,r.grid_longitude)
        part=old_points[old_points.run_id.eq(r.run_id)].copy()
        if (gid,r.initialized_at) in existing:
            new_id=existing[(gid,r.initialized_at)]
            left=points[points.run_id.eq(new_id)].set_index('valid_at').precipitation_mm
            right=part.set_index('valid_at').precipitation_mm
            idx=left.index.intersection(right.index)
            mismatch=int(((left.loc[idx]-right.loc[idx]).abs()>1e-8).sum())+int((left.loc[idx].isna()!=right.loc[idx].isna()).sum())
            checks.append({'grid_id':gid,'initialized_at':r.initialized_at,'common_n':len(idx),'mismatch_n':mismatch})
            if mismatch:raise ValueError('Fresh API response disagrees with frozen Stage 3 reference')
            continue
        item=r._asdict();item['location_id']=gid
        item['source_artifact']=str(snapshot/'forecast_runs.parquet')
        item['request_parameters_json']=None
        item['source_snapshot_run_id']=r.run_id
        item['provenance_class']=classify(r.initialized_at)[1]
        item['utc_offset_seconds']=float(item['utc_offset_seconds'])
        rows.append(item);p_rows.append(part)
        m_rows.append({'request_id':'750007','request_station_id':'750007','request_latitude':16.221111,
                       'request_longitude':107.179444,'initialized_at':r.initialized_at,'grid_id':gid,
                       'grid_latitude':r.grid_latitude,'grid_longitude':r.grid_longitude,
                       'cell_selection':'land','run_id':r.run_id,'checksum':r.checksum,'model':r.model,
                       'model_version':classify(r.initialized_at)[2],
                       'version_basis':'official transition; original Stage 1 snapshot',
                       'source_snapshot_run_id':r.run_id,
                       'source_artifact':str(snapshot/'forecast_runs.parquet'),
                       'request_parameters_json':None})
    if rows:runs=pd.concat([runs,pd.DataFrame(rows)],ignore_index=True)
    if p_rows:points=pd.concat([points,*p_rows],ignore_index=True)
    if m_rows:mapping=pd.concat([mapping,pd.DataFrame(m_rows)],ignore_index=True)
    for column in ['initialized_at','retrieved_at','available_at']:runs[column]=pd.to_datetime(runs[column],utc=True)
    points['valid_at']=pd.to_datetime(points.valid_at,utc=True)
    registry=mapping.groupby('grid_id',as_index=False).agg(grid_latitude=('grid_latitude','first'),
                grid_longitude=('grid_longitude','first'),request_latitude=('request_latitude','first'),
                request_longitude=('request_longitude','first'),request_ids=('request_id',lambda s:';'.join(sorted(set(s)))),
                first_run=('initialized_at','min'),last_run=('initialized_at','max'),run_count=('initialized_at','nunique'))
    registry['location_id']=registry.grid_id
    registry['station_id']=registry.request_ids.str.split(';').str[0]
    registry['latitude']=registry.request_latitude;registry['longitude']=registry.request_longitude
    labels={'750007':'A Lưới','750062':'A Lưới','0074647403':'Dã Viên','0074647402':'Thuận An',
            '750041':'Bạch Mã thấp','750066':'Đỉnh Bạch Mã','750012':'Thượng Nhật','0074647606':'Sông Ô Lâu'}
    registry['grid_name']=registry.request_ids.map(lambda ids:labels.get(ids.split(';')[0],ids))
    return runs,points,mapping,registry,pd.DataFrame(checks)


def synchronized_features(selected,grid_ids,anchors,lag_hours=6,max_age_hours=24,run_audit=None,points=None):
    """Select one initialization available in ALL chosen cells; never mix cycles."""
    if run_audit is None or points is None:raise ValueError('Audited run metadata and hourly values are required')
    columns=['rain_0_3h_mm','rain_3_6h_mm','rain_6_12h_mm','rain_12_24h_mm']
    # Individual-grid selection may use different cycles; synchronized vectors are built separately.
    valid=points[points.quality_status.eq('forecast') & np.isfinite(points.precipitation_mm) & points.precipitation_mm.ge(0)]
    point_maps={rid:part.set_index('valid_at').precipitation_mm.to_dict() for rid,part in valid.groupby('run_id',sort=False)}
    metadata=run_audit.sort_values(['initialized_at','retrieved_at']).drop_duplicates(['location_id','initialized_at'],keep='last')
    by_cycle={cycle:part.set_index('location_id') for cycle,part in metadata.groupby('initialized_at')
              if set(grid_ids)<=set(part.location_id)}
    cycles=pd.DatetimeIndex(sorted(by_cycle))
    records=[]
    for anchor in anchors:
        eligible=cycles[(cycles<=anchor-pd.Timedelta(hours=lag_hours))&(cycles>=anchor-pd.Timedelta(hours=max_age_hours))]
        for cycle in reversed(eligible):
            frame=by_cycle[cycle];vector={};ok=True
            for gid in grid_ids:
                row=frame.loc[gid]
                if pd.notna(row.available_at) and row.available_at>anchor:ok=False;break
                values=point_maps.get(row.run_id,{})
                hourly=[values.get(anchor+pd.Timedelta(hours=h),np.nan) for h in range(1,25)]
                if not np.isfinite(hourly).all():ok=False;break
                for (a,b),col in zip([(0,3),(3,6),(6,12),(12,24)],columns):vector[f'{gid}|{col}']=sum(hourly[a:b])
                vector[f'{gid}|run_id']=row.run_id
                vector[f'{gid}|archive_kind']=row.archive_kind
            if ok:
                records.append({'anchor_utc':anchor,'initialized_at':cycle,'age_hours':(anchor-cycle)/pd.Timedelta(hours=1),**vector})
                break
    return pd.DataFrame(records)
