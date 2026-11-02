"""Rebuild the 11-station fixed-split dataset from frozen observations at 30 min.

Weather at :30 reuses the previous hourly anchor's forecast vector; windows
therefore start 30 minutes before the sample anchor, not at the anchor.
No training or model inference is performed.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.ai.regional_data.builder import build_dataset, write_json

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'train_data/hue_regional_20261015_h6_h12_fixed_split_v1'
SNAPSHOT=ROOT/'data/processed/station_analysis/phase2/inventory/snapshots/20261026T035126Z_7f98ea8a'


def aggregate_water(raw, station_ids, index):
    obs=raw[raw.station_id.isin(station_ids)].copy()
    obs['observed_at']=pd.to_datetime(obs.observed_at,utc=True)
    obs['value']=pd.to_numeric(obs.depth,errors='coerce')
    obs['bucket']=obs.observed_at.dt.ceil('30min')
    obs['duplicate']=obs.duplicated(['station_id','observed_at'],keep=False)
    obs['valid']=np.isfinite(obs.value)&obs.quality_status.eq('observed')
    obs['value']=obs.value.where(obs.valid)
    grouped=obs.groupby(['bucket','station_id'],sort=False).agg(value=('value','mean'),duplicates=('duplicate','sum'))
    grouped.loc[grouped.duplicates>0,'value']=np.nan
    return grouped.value.unstack('station_id').reindex(index=index,columns=station_ids)


def aggregate_rain(raw,station_ids,index):
    obs=raw[raw.station_id.isin(station_ids)].copy()
    obs['observed_at']=pd.to_datetime(obs.observed_at,utc=True)
    val=pd.to_numeric(obs.depth,errors='coerce')
    # Exact 10-minute slots only; a duplicate makes its whole 1-hour window unavailable.
    obs['duplicate']=obs.duplicated(['station_id','observed_at'],keep=False)
    obs['bucket']=obs.observed_at.dt.ceil('30min')
    obs['valid']=np.isfinite(val)&val.ge(0)&obs.quality_status.eq('observed')&obs.observed_at.eq(obs.observed_at.dt.floor('10min'))
    obs['val']=val.where(obs.valid)
    g=obs.groupby(['bucket','station_id'],sort=False).agg(total=('val','sum'),count=('valid','sum'),duplicates=('duplicate','sum'))
    count=g['count'].unstack('station_id').reindex(index=index,columns=station_ids).fillna(0)
    total=g['total'].unstack('station_id').reindex(index=index,columns=station_ids).fillna(0)
    duplicate=g['duplicates'].unstack('station_id').reindex(index=index,columns=station_ids).fillna(0)
    n=count+count.shift(1);s=total+total.shift(1);dup=duplicate+duplicate.shift(1)
    strict=s.where(n.eq(6)&dup.eq(0));standard=s.where(n.eq(6)&dup.eq(0)).combine_first((s*1.2).where(n.eq(5)&dup.eq(0)))
    return standard,strict


def prior_hour_forecast(old_weather,index):
    source=old_weather.set_index('timestamp_utc').sort_index()
    # Select the historical row at floor(anchor); no future forecasts or interpolation.
    source_time=index.floor('h');w=source.reindex(source_time).copy();w.index=index
    w['forecast_source_anchor_utc']=source_time
    w['weather_forecast_age_hours']=(w.index.to_series()-pd.to_datetime(w.initialized_at,utc=True))/pd.Timedelta(hours=1)
    invalid=w.run_id.isna()|(w.weather_forecast_age_hours>24)
    columns=['weather_forecast_rain_0_3h','weather_forecast_rain_3_6h','weather_forecast_rain_6_12h','weather_forecast_rain_12_24h','weather_forecast_age_hours']
    for col in columns:
        w.loc[invalid,col]=np.nan;w[col+'_missing']=w[col].isna().astype('uint8')
    for col in ['weather_forecast_observed_availability','weather_forecast_assumed_availability','weather_forecast_hindcast']:
        w.loc[invalid,col]=0;w[col]=w[col].fillna(0).astype('uint8')
    for col in ['run_id','archive_kind','initialized_at','retrieved_at','available_at','assumed_available_at','availability_basis','provenance_class','model','grid_latitude','grid_longitude','grid_id','cell_selection','model_version']:
        w.loc[invalid,col]=None
    w.loc[invalid,'strict_eligible_run']=False
    return w



def write_30min_report(output):
    output=Path(output);samples=pd.read_parquet(output/'samples.parquet');manifest=json.loads((output/'manifest.json').read_text())
    counts=samples.split.value_counts().to_dict();parts=[
      '# Bộ dữ liệu khu vực 11 trạm — lưới 30 phút', '',
      'Dataset mới, tách biệt dataset 1h cũ. Không huấn luyện mô hình trong bước build.', '',
      '## Cấu hình', '',
      '- 11 trạm nước và cùng danh mục trạm mưa, cùng mốc phân chia fixed và khoảng purge 24h với dataset 1h.',
      '- Lưới anchor cách nhau 30 phút; anchor thiếu hiện tại, target hoặc không đủ lịch sử được loại, nên có thể có khoảng trống.',
      '- 96 timestep từ t−47,5h đến t. Mỗi giá trị nước là trung bình quan trắc hợp lệ trong (t−30 phút, t], không phải trung bình 1 giờ như dataset cũ.',
      '- Hai nhãn là mực nước tại đúng t+6h và t+12h (12 và 24 bước 30 phút); mỗi nhãn cũng là trung bình 30 phút.',
      '- Các scaler được fit lại chỉ trên train của split fixed; không tái sử dụng scaler dataset 1h.', '',
      '| Split | Số sample |', '|---|---:|']
    for split in ['train','validation','test_exploratory','purged']:parts.append(f'| {split} | {counts.get(split,0):,} |')
    parts += ['', '## Quan trắc mưa và forecast', '',
      'Mưa được tính từ các quan trắc 10 phút. Tại từng anchor 30 phút, tổng 1h sử dụng 6/6 mẫu hoặc ước tính 5/6 ×6/5; ít hơn 5 mẫu là thiếu. Quan trắc trùng timestamp làm cửa sổ thiếu. Các tổng 3/6/12/24/48h cộng các khoảng 1h không chồng lấn, không cộng hai lần.', '',
      'Nguồn forecast có độ phân giải 1 giờ. Tại anchor :30, dùng nguyên vector forecast đã chọn tại giờ tròn trước đó, không dùng tương lai hay nội suy; tuổi forecast tăng thêm 0,5h. Do vậy các cửa sổ forecast 0–3/3–6/6–12/12–24h tại :30 bắt đầu sớm hơn anchor hiện tại 30 phút. Đây là khác biệt quan trọng so với forecast đồng bộ chính xác từng anchor. Các giả định availability hồi cứu từ dataset cũ vẫn áp dụng; chưa chứng minh được dữ liệu thật sự khả dụng theo thời gian thực.', '',
      'Nước thiếu được giữ NaN trong parquet và có mask; bộ đọc chuẩn hóa bằng scaler train rồi điền 0 trong không gian chuẩn hóa. Hai mẫu liên tiếp có lịch sử chồng lấn nên không thể xem số mẫu tăng gần gấp đôi là số sự kiện độc lập tăng gấp đôi.', '',
      '## Sử dụng', '',
      'Các lệnh training cần `--dataset-dir` trỏ tới thư mục này, `--horizon 30min`, `--lstm-seq-len 96` và `--target-horizons 6 12`. Không dùng checkpoint/scaler 1h như thể chúng là mô hình 30 phút.', '',
      'Xác minh: `PYTHONPATH=. .venv/bin/python research/scripts/validate_regional_30min_dataset.py '+str(output)+'`. Xem validation.json để biết những gì đã kiểm tra. Nguồn và hash được lưu trong manifest.json.']
    (output/'REPORT.md').write_text('\n'.join(parts)+'\n',encoding='utf-8')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path,default=ROOT/'train_data/hue_regional_20261015_h6_h12_fixed_split_30min_v1')
    a=ap.parse_args()
    if a.output_dir.exists():raise FileExistsError(f'Refuse to overwrite {a.output_dir}')
    mapping=json.loads((OLD/'station_mapping.json').read_text());config=json.loads((OLD/'config.json').read_text())
    config['resolution_minutes']=30;config['sequence_length']=96
    water_old=pd.read_parquet(OLD/'water.parquet');hour_index=pd.DatetimeIndex(water_old.timestamp_utc)
    index=pd.date_range(hour_index.min(),hour_index.max(),freq='30min')
    catalog=pd.read_parquet(OLD/'rain_station_catalog.parquet');catalog['station_id']=catalog.station_id.astype(str)
    rain_ids=catalog.station_id.tolist();station_ids=list(mapping)
    print('Reading frozen 10-minute observations',flush=True)
    raw=pd.read_parquet(SNAPSHOT/'observations.parquet',filters=[('station_id','in',station_ids+rain_ids)])
    raw.station_id=raw.station_id.astype(str)
    water=aggregate_water(raw,station_ids,index);standard,strict=aggregate_rain(raw,rain_ids,index)
    weather=prior_hour_forecast(pd.read_parquet(OLD/'weather.parquet'),index)
    clusters=pd.read_csv(OLD/'eda1_station_groups.csv',dtype={'station_id':str})
    sources=[SNAPSHOT/'observations.parquet',OLD/'manifest.json',OLD/'station_mapping.json',OLD/'weather.parquet',OLD/'rain_station_catalog.parquet',OLD/'eda1_station_groups.csv',OLD/'config.json']
    print('Building 30-minute dataset; no training',flush=True)
    build_dataset(a.output_dir,level=water,rain_standard=standard,rain_strict=strict,weather=weather,
                  stations=mapping,clusters=clusters,config=config,source_paths=sources,rain_catalog=catalog)
    schema=json.loads((a.output_dir/'feature_schema.json').read_text());schema.update(
        sequence_order='oldest to newest: t-47.5h through t, 96 samples at 30 minute cadence',
        time_feature_formulas={**schema['time_feature_formulas'],'hour':'sin/cos(2*pi*(local_hour+local_minute/60)/24)'},
        forecast_half_hour_policy='At :30 reuse previously selected hourly anchor forecast; forecast intervals start 30 min before current anchor. No interpolation.',
        rain_policy='At each 30min anchor, 1h accumulation from 6 ten-minute observations; 5/6 estimated *6/5, fewer missing. Multi-hour windows use non-overlapping 1h accumulations spaced 1h.')
    write_json(a.output_dir/'feature_schema.json',schema)
    manifest=json.loads((a.output_dir/'manifest.json').read_text());manifest.update(
        observation_interval='30min',water_aggregation='mean observed finite 10-minute records in (t-30min,t]; duplicate timestamps invalidate bucket',
        weather_half_hour_policy='previous hourly anchor vector; forecast intervals offset 30 min at :30; no interpolation; forecast ages updated',
        rainfall_policy='1-hour totals at each 30-minute anchor, from 6/6 or 5/6 10-minute values; multihour uses 1-hour spaced windows',
        source_dataset=str(OLD),history_span_hours=47.5)
    write_json(a.output_dir/'manifest.json',manifest)
    write_30min_report(a.output_dir)
    print(f'Built {a.output_dir}; run the separate validator before use',flush=True)

if __name__=='__main__':main()
