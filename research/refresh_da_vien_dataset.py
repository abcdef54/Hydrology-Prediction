"""Build a versioned Da Vien hourly dataset from a read-only PostgreSQL snapshot.

Preserves the existing top-four-rain aggregation and schema. Never replaces raw
project CSVs, existing datasets or serving models. Splits may be specified as
explicit aware timestamps, otherwise the legacy 70/15/15 rule is used.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.backend.db.session import build_engine
from research.build_training_dataset_1h_top4 import build_table, split_table, audit, RAIN_STATIONS, WIND_STATIONS


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--snapshot-dir',type=Path,required=True)
    parser.add_argument('--validation-start',help='Timezone-aware timestamp')
    parser.add_argument('--test-start',help='Timezone-aware timestamp')
    args=parser.parse_args()
    if bool(args.validation_start)!=bool(args.test_start):parser.error('Specify both split timestamps')
    if args.output_dir.exists() or args.snapshot_dir.exists():raise FileExistsError('Use new output and snapshot directories')
    args.output_dir.mkdir(parents=True);args.snapshot_dir.mkdir(parents=True)
    load_dotenv(ROOT/'.env',override=False)
    counts={}
    with build_engine().connect().execution_options(isolation_level='REPEATABLE READ') as c:
        c.execute(text('SET TRANSACTION READ ONLY'))
        c.execute(text("SET LOCAL statement_timeout='90s'"))
        cutoff=c.execute(text("SELECT date_trunc('hour',now())")).scalar_one()
        for name,ids in [('water_level_observations',['0074647403']),('rain_observations',list(RAIN_STATIONS))]:
            df=pd.read_sql(text('SELECT station_id,observed_at AS time_point,depth FROM hydrology.observations WHERE station_id=ANY(:ids) AND observed_at<=:cutoff ORDER BY station_id,observed_at'),c,params={'ids':ids,'cutoff':cutoff})
            counts[name]=len(df);df.to_csv(args.snapshot_dir/f'{name}.csv.gz',index=False)
        stations=pd.read_sql(text("SELECT id AS database_id,source_station_id AS id,name FROM hydrology.stations WHERE station_type='wind' AND name=ANY(:names) ORDER BY name"),c,params={'names':list(WIND_STATIONS)})
        if len(stations)!=2 or set(stations.name)!=set(WIND_STATIONS):raise ValueError('Wind station names must resolve exactly once')
        wind=pd.read_sql(text('SELECT s.source_station_id AS sid,o.observed_at AS time_point,o.ws,o.wsg,o.wd,o.wdg FROM hydrology.wind_observations o JOIN hydrology.stations s ON s.id=o.station_id WHERE o.station_id=ANY(:ids) AND o.observed_at<=:cutoff ORDER BY o.station_id,o.observed_at'),c,params={'ids':stations.database_id.tolist(),'cutoff':cutoff})
        counts['wind_observations']=len(wind)
        wind.to_csv(args.snapshot_dir/'wind_observations.csv.gz',index=False)
        stations[['id','name']].to_csv(args.snapshot_dir/'wind_stations.csv.gz',index=False)
    table=build_table(args.snapshot_dir)
    table=table.loc[table.index<=cutoff]
    if args.validation_start:
        val_start=pd.Timestamp(args.validation_start);test_start=pd.Timestamp(args.test_start)
        if val_start.tzinfo is None or test_start.tzinfo is None:raise ValueError('Split times must include timezone')
        if not table.index.min()<val_start<test_start<=table.index.max():raise ValueError('Split boundaries outside dataset')
        splits={'train':table.loc[table.index+pd.Timedelta(hours=24)<val_start],
                'val':table.loc[(table.index>=val_start)&(table.index+pd.Timedelta(hours=24)<test_start)],
                'test':table.loc[table.index>=test_start]}
        split_rule=f'train before {val_start}; validation before {test_start}; purge 24h before each boundary'
    else:
        splits=split_table(table);split_rule='Legacy chronological 70/15/15, purged 24h'
    checks=audit(table,splits)
    table.to_csv(args.output_dir/'hourly.csv',index=True)
    targets=[col for col in table if col.startswith('target_')];features=[col for col in table if col not in targets]
    for name,df in splits.items():df.to_csv(args.output_dir/f'{name}.csv',index=True)
    summary={name:dict(rows=len(df),complete_targets=int(df[targets].notna().all(axis=1).sum()),start=str(df.index.min()),end=str(df.index.max())) for name,df in splits.items()}
    old=pd.read_csv(ROOT/'train_data/1h_top4/train.csv',nrows=0)
    if list(old.columns)!=['timestamp_utc',*table.columns]:raise AssertionError('Feature/target schema changed')
    manifest=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),source='PostgreSQL repeatable-read read-only snapshot',cutoff_utc=str(cutoff),snapshot_dir=str(args.snapshot_dir.resolve()),source_counts=counts,rows=len(table),first_hour=str(table.index.min()),last_hour=str(table.index.max()),feature_columns=features,target_columns=targets,splits=summary,split_rule=split_rule,audit_checks=checks,source_sha256={p.name:sha(p) for p in args.snapshot_dir.glob('*.gz')},outputs_sha256={p.name:sha(p) for p in args.output_dir.glob('*.csv')},caveats=['Original four rain sources retained; selection was exploratory over historical available data','October has been inspected repeatedly and is not a pristine test set','No model trained or deployed by this builder'])
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    table.isna().mean().rename('missing_rate').to_csv(args.output_dir/'missingness.csv')
    print(json.dumps({k:manifest[k] for k in ['rows','first_hour','last_hour','source_counts','splits','split_rule']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
