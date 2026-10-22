"""Build paired mask/-1 datasets from the fixed step-1 snapshot and resolved forecasts."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from research.scripts.weather_dataset_readiness import WINDOWS, read_times


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def encode_pair(frame, feature_columns):
    """Preserve labels; remove input flags only in the sentinel variant."""
    flags=[c for c in feature_columns if c.endswith('_missing')]
    values=[c for c in feature_columns if c not in flags]
    if (frame[values]==-1).any().any():
        raise ValueError('Valid feature value -1 collides with sentinel; cannot encode this snapshot safely')
    mask=frame.copy()
    sentinel=frame.drop(columns=flags).copy()
    sentinel[values]=sentinel[values].fillna(-1.)
    return mask,sentinel,flags


def build(root):
    parent=root/'train_data/da_vien_1h_2026-10-15'
    weather_dir=root/'reports/weather_collection/resolved_forecasts'
    weather_path=weather_dir/'weather_forecast_resolved.csv.gz'
    provenance=json.loads((weather_dir/'manifest.json').read_text())
    weather=pd.read_csv(weather_path,index_col=0)
    weather.index=pd.to_datetime(weather.index,utc=True)
    if weather.index.has_duplicates:raise ValueError('Duplicate weather anchors')
    weather['initialized_at']=pd.to_datetime(weather.initialized_at,utc=True)
    if ((weather.index-weather.initialized_at).dt.total_seconds()<6*3600).any():
        raise ValueError('Weather run violates assumed release lag')
    parent_manifest=json.loads((parent/'manifest.json').read_text())
    base=read_times(parent/'hourly.csv')
    base_columns=[c for c in base if c.startswith('water_level') or
        c.startswith(('rain_750007_','rain_750062_')) or c=='flood_season']
    assert len(base_columns)==17
    targets=[c for c in base if c.startswith('target_')]
    if len(targets)!=5:raise ValueError('Expected the five step-1 targets')
    combined=base[base_columns+targets].copy()
    forecast_columns=[]
    for a,b in WINDOWS:
        column=f'weather_forecast_rain_{a}_{b}h'
        combined[column]=weather[f'forecast_{a}_{b}h'].reindex(combined.index)
        forecast_columns.append(column)
    for column in forecast_columns:
        combined[column+'_missing']=combined[column].isna().astype(int)
    combined['weather_forecast_age_hours']=weather.age_hours.reindex(combined.index)
    features=base_columns+forecast_columns+[c+'_missing' for c in forecast_columns]+['weather_forecast_age_hours']
    if combined[forecast_columns+['weather_forecast_age_hours']].isna().any().any():
        raise ValueError('Resolved weather does not cover every dataset hour')
    combined=combined[features+targets]
    mask,sentinel,flags=encode_pair(combined,features)
    outputs=[]
    for mode,frame in [('mask',mask),('neg1',sentinel)]:
        out=root/f'train_data/da_vien_1h_2026-10-15_forecast_{mode}'
        if out.exists():raise FileExistsError(f'Refuse to overwrite dataset: {out}')
        outputs.append((mode,frame,out))
    checks=[]
    split_frames={}
    for key in ['train','val','test']:
        path=parent/f'{key}.csv'
        if sha(path)!=provenance['split_sha256'][f'{key}.csv']:
            raise ValueError('Source split changed since forecast resolution')
        original=read_times(path)
        for mode,frame,out in outputs:
            split_combined=combined.reindex(original.index).copy()
            split_combined[base_columns+targets]=original[base_columns+targets]
            split_mask,split_sentinel,_=encode_pair(split_combined,features)
            subset=split_mask if mode=='mask' else split_sentinel
            split_frames[(mode,key)]=subset
            pd.testing.assert_frame_equal(subset[targets],original[targets],check_exact=True)
            valid=original[targets].notna().all(axis=1)&original.water_level_depth.notna()
            count=int((valid & (np.arange(len(original))>=95)).sum())
            checks.append(dict(variant=mode,split=key,rows=len(subset),sequences=count))
    for mode,frame,out in outputs:
        out.mkdir(parents=True)
        input_features=[c for c in frame if c not in targets]
        frame.to_csv(out/'hourly.csv',index_label='timestamp_utc')
        for key in ['train','val','test']:
            original=read_times(parent/f'{key}.csv')
            split_frames[(mode,key)].to_csv(out/f'{key}.csv',index_label='timestamp_utc')
        # Provenance is not a model input, and is intentionally outside split CSVs.
        weather[['initialized_at','age_hours','was_filled','run_id','source_retrieved_at',
                 'archive_kind','availability_basis']].to_csv(out/'weather_provenance.csv.gz',
                                                           index_label='timestamp_utc',compression='gzip')
        manifest=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),resolution_minutes=60,
            variant=mode,parent_dataset=str(parent),parent_manifest_sha256=sha(parent/'manifest.json'),
            comparison_observations_dir='../da_vien_1h_2026-10-15',
            source_split_sha256=provenance['split_sha256'],source_weather_sha256=sha(weather_path),
            rain_station_ids=['750062','750007'],sequence_length=96,feature_columns=input_features,
            target_columns=targets,input_size=len(input_features),output_size=len(targets),
            missing_encoding='nan_with_flags' if mode=='mask' else 'sentinel_minus_one',
            sentinel_columns=[] if mode=='mask' else input_features,
            removed_missing_flags=[] if mode=='mask' else flags,
            raw_missing_value=None if mode=='mask' else -1.,
            normalized_missing_value=0. if mode=='mask' else -1.,
            feature_set='water_level+rain+flood_season+weather_forecast',
            availability_policy='run init + assumed 6h <= anchor; use older complete run max age24h',
            forecast_windows_hours=WINDOWS,forecast_grid=[16.274164,107.26496],
            observation_cutoff_utc=parent_manifest['cutoff_utc'],
            splits=[r for r in checks if r['variant']==mode],
            caveats=['Historical forecast publication time is assumed, not independently verified.',
                     'Archive includes provider-labelled hindcasts and a model-cycle transition.',
                     'Missing targets remain NaN; never encode labels as -1.',
                     'Test has already been used for exploratory model comparisons.',
                     'Minus one in scaled space can overlap legitimate standardized values; experimental encoding.'],
            outputs_sha256={p.name:sha(p) for p in out.glob('*.csv')})
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
        lines=[f'# Dã Viên + forecast — {mode}','',
            f'Input: {len(input_features)} features × 96 giờ; output: 5 mốc nước. Hai trạm: Nhâm/Hồng Thái.',
            'Forecast: tổng 0–3/3–6/6–12/12–24h và độ cũ. Không có cờ fallback trong input.', '',
            'Giữ nguyên timestamp, split/purge và nhãn của dataset bước 1. Không thêm quan trắc sau cutoff 15/10/2026.',
            'Variant mask giữ 3 cờ cũ và thêm 4 cờ forecast. Variant neg1 bỏ cả 7 cờ; '
            'mọi feature thực sự thiếu thành -1, target thiếu vẫn NaN.', '',
            'Loader đọc manifest: scaler chỉ fit trên giá trị hợp lệ của train; '
            'neg1 được giải mã thành NaN nội bộ trước fit, rồi gán -1 sau chuẩn hóa. '
            'Mã này không đảm bảo -1 nằm ngoài mọi giá trị hợp lệ đã StandardScaler. '
            'Các giá trị hợp lệ, kể cả 0 mm mưa, không bị thay đổi.', '',
            '`weather_provenance.csv.gz` chỉ dùng truy vết, không đưa vào model.', '',
            'Feature set khi train: `water_level+rain+flood_season+weather_forecast`.',
            'Chưa train model hoặc thay cấu hình serving.']
        (out/'README.md').write_text('\n'.join(lines)+'\n')
        print(f'{mode}: {out}; {len(input_features)} inputs')
    return checks


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    args=parser.parse_args()
    print(json.dumps(build(args.root),indent=2))
