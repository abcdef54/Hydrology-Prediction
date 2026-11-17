"""Small historical-run validation gate, explicitly separate from bulk downloads."""
import argparse
import json
from pathlib import Path
import pandas as pd
from .collection import Collector,flatten_records
from research.station_analysis.config import ROOT
from ..config import load_config


DATES=['2025-11-14T12:00Z','2025-11-20T00:00Z','2026-05-12T00:00Z',
       '2026-05-12T06:00Z','2026-10-15T00:00Z','2026-10-21T06:00Z']


def validate(config):
    settings=config['spatial_weather']
    inventory=Path(config['work_dir'])/'inventory'
    snapshot=Path(json.loads((inventory/'current_snapshot.json').read_text())['directory'])
    stations=pd.read_parquet(snapshot/'stations.parquet').set_index('station_id')
    locations=[{'id':sid,'station_id':sid,'name':stations.loc[sid,'name'],
                'latitude':float(stations.loc[sid,'latitude']),'longitude':float(stations.loc[sid,'longitude'])}
               for sid in settings['requested_station_ids']]
    folder=Path(config['work_dir'])/'spatial_weather'/settings['collection_version']
    collector=Collector(folder/'raw',{'retries':3,'request_interval_seconds':1.1})
    records=[collector.fetch(locations,date) for date in DATES]
    runs,points,mapping=flatten_records(records)
    for name,frame in [('validation_runs',runs),('validation_points',points),('validation_mapping',mapping)]:
        frame.to_parquet(folder/f'{name}.parquet',index=False)
    oldruns=pd.read_parquet(snapshot/'forecast_runs.parquet');oldpoints=pd.read_parquet(snapshot/'forecast_points.parquet')
    checks=[]
    for date in [DATES[0],DATES[3],DATES[4]]:
        fresh=runs[runs.initialized_at.eq(pd.Timestamp(date))&runs.grid_latitude.eq(16.274164)&runs.grid_longitude.eq(107.26496)].iloc[0]
        old=oldruns[oldruns.initialized_at.eq(pd.Timestamp(date))&oldruns.location_id.str.contains('750007')].sort_values('retrieved_at').iloc[-1]
        left=points[points.run_id.eq(fresh.run_id)].set_index('valid_at').precipitation_mm
        right=oldpoints[oldpoints.run_id.eq(old.run_id)].set_index('valid_at').precipitation_mm
        common=left.index.intersection(right.index)
        delta=(left.loc[common]-right.loc[common]).abs()
        checks.append({'run':date,'common_hours':len(common),'max_abs_difference_mm':float(delta.max()),
                       'missing_mismatch':int((left.loc[common].isna()!=right.loc[common].isna()).sum())})
    if not all(c['common_hours']>=36 and c['max_abs_difference_mm']==0 and c['missing_mismatch']==0 for c in checks):
        raise ValueError('Preflight differs from frozen Stage 3 data; stop before collection')
    (folder/'validation_comparison.json').write_text(json.dumps(checks,indent=2)+'\n')
    (folder/'locations.json').write_text(json.dumps(locations,ensure_ascii=False,indent=2)+'\n')
    # Reuse the already captured nearest probe; never multiply full historical downloads for it.
    prior=Path(config['output_dir'])/'stage3_weather/grid_verification'
    if (prior/'same_run_nearest.json').exists():
        nearest=json.loads((prior/'same_run_nearest.json').read_text())
        land=json.loads((prior/'same_run_land.json').read_text())
        rows=[]
        order=json.loads((prior/'manifest.json').read_text())['station_order']
        for source,a,b in zip(order,land,nearest):
            rows.append({'request_id':source['id'],'initialized_at':'2026-10-15T00:00:00Z',
                         'land_latitude':a['latitude'],'land_longitude':a['longitude'],
                         'nearest_latitude':b['latitude'],'nearest_longitude':b['longitude'],
                         'grid_changes':a['latitude']!=b['latitude'] or a['longitude']!=b['longitude'],
                         'evidence_path':str(prior/'same_run_nearest.json')})
        pd.DataFrame(rows).to_csv(folder/'nearest_sensitivity.csv',index=False)
    return checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).parents[1]/'config.yaml')
    args=parser.parse_args();print(json.dumps(validate(load_config(args.config)),indent=2))


if __name__=='__main__':main()
