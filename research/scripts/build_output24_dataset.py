"""Build direct hourly water targets 1..24 from the fixed B snapshot, offline.

No interpolation, database access or changes to original files. The current
ablation requires identical complete-label populations for O5/O24; if a new
snapshot violates that assumption, fail before writing instead of comparing
different samples or manufacturing missing labels in O5.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd

from src.ai.hourly_dataset import COMMON, TARGETS, read_frame, add_targets, assert_matched_population

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_SOURCE=ROOT/'train_data/da_vien_1h_2026-10-15_forecast_mask'
DEFAULT_OUTPUT=ROOT/'train_data/da_vien_1h_2026-10-15_forecast_mask_o24'


def build(source_dir,output_dir):
    source_dir,output_dir=Path(source_dir).resolve(),Path(output_dir).resolve()
    if output_dir.exists():raise FileExistsError(f'Refuse to overwrite: {output_dir}')
    spec=json.loads((source_dir/'manifest.json').read_text())
    if spec.get('variant')!='mask':raise ValueError('This paired experiment requires the B mask dataset')
    source=read_frame(source_dir/'hourly.csv')
    features=spec['feature_columns']
    if len(features)!=26:raise ValueError('Expected the 26 B input features')
    labels=add_targets(source)
    assert_matched_population(labels,source.water_level_depth)
    combined=pd.concat([source[features],labels],axis=1)
    splits={};coverage={}
    for split in ['train','val','test']:
        original=read_frame(source_dir/f'{split}.csv')
        if not original.index.isin(combined.index).all():raise ValueError('Split contains times outside snapshot')
        frame=combined.reindex(original.index).copy()
        # Preserve exact split predictor values, including original missingness.
        frame[features]=original[features]
        for c in COMMON:
            if not np.allclose(frame[c],original[c],rtol=0,atol=1e-9,equal_nan=True):
                raise ValueError(f'{split} has changed common targets')
        count=assert_matched_population(frame[TARGETS],frame.water_level_depth)
        splits[split]=frame
        coverage[split]=dict(rows=len(frame),complete_24h=int(frame[TARGETS].notna().all(axis=1).sum()),
            sequences_96=count,start_utc=str(frame.index.min()),end_utc=str(frame.index.max()))
    for left,right in [('train','val'),('val','test')]:
        if splits[left].index.max()+pd.Timedelta(hours=24)>=splits[right].index.min():
            raise ValueError('24h target purge is not preserved')
    output_dir.mkdir(parents=True)
    combined.to_csv(output_dir/'hourly.csv',index_label='timestamp_utc')
    for name,frame in splits.items():frame.to_csv(output_dir/f'{name}.csv',index_label='timestamp_utc')
    if (source_dir/'weather_provenance.csv.gz').exists():
        shutil.copyfile(source_dir/'weather_provenance.csv.gz',output_dir/'weather_provenance.csv.gz')
    manifest=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),experiment='O5_O24',
        parent_dataset=str(source_dir),resolution_minutes=60,variant='mask',sequence_length=96,
        feature_columns=features,target_columns=TARGETS,common_target_columns=COMMON,input_size=26,output_size=24,
        target_definition='Observed hourly mean H(t+h), hours 1..24, exact timestamp lookup; no interpolation',
        matching_policy='O5 and O24 require identical complete-label rows; fail if extra target holes change eligibility',
        splits=coverage,purge_hours=24,feature_set=spec['feature_set'],
        availability_policy=spec['availability_policy'],rain_station_ids=spec['rain_station_ids'],
        observation_cutoff_utc=spec['observation_cutoff_utc'],
        caveats=spec.get('caveats',[])+['Test reused for exploratory comparisons; 24 labels do not imply 24x independent observations.'])
    (output_dir/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (output_dir/'README.md').write_text('# Dã Viên O5/O24\n\n26 B features; 24 direct observed hourly targets. '
        'Use research.scripts.output_horizon_experiment with --outputs 5 or 24 on the same directory. '
        'No interpolation, no source modification. Manifest lists identical eligible populations.\n')
    print(json.dumps(coverage,ensure_ascii=False,indent=2))
    print(output_dir)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',type=Path,default=DEFAULT_SOURCE)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args();build(args.source_dir,args.output_dir)
