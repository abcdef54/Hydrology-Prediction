"""Resume full Stage 3B collection after the preflight gate has passed."""
import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from .collection import Collector

LOG=logging.getLogger(__name__)


def collect(folder,snapshot,start,end):
    folder=Path(folder)
    validation=json.loads((folder/'validation_comparison.json').read_text())
    if not validation or any(r['max_abs_difference_mm']!=0 or r['missing_mismatch'] for r in validation):
        raise ValueError('Preflight equality checks must pass before bulk collection')
    locations=json.loads((folder/'locations.json').read_text())
    others=locations[2:]
    reference=locations[:1]
    old=pd.read_parquet(Path(snapshot)/'forecast_runs.parquet')
    retained=set(old[old.location_id.str.contains('750007')].initialized_at)
    cycles=pd.date_range((pd.Timestamp(start).tz_convert('UTC')-pd.Timedelta(hours=6)).floor('6h'),
                         (pd.Timestamp(end).tz_convert('UTC')-pd.Timedelta(hours=6)).floor('6h'),freq='6h')
    client=Collector(folder/'raw',{'retries':4,'request_interval_seconds':1.1})
    failures=[];completed=0
    # Interleave months so an interrupted collection still spans the full research period.
    ordered=sorted(cycles,key=lambda t:(t.day,t.hour,t.month,t.year))
    for cycle in ordered:
        covered=[]
        for path in (folder/'raw').glob(f'{cycle.strftime("%Y%m%dT%H%M")}_land_*.json'):
            record=json.loads(path.read_text());covered.extend(r['id'] for r in record['locations'])
        wanted=[p for p in others if p['id'] not in covered]
        if cycle not in retained:
            wanted.extend(p for p in reference if p['id'] not in covered)
        if wanted:
            try:client.fetch(wanted,cycle)
            except RuntimeError as error:
                failures.append({'initialized_at':str(cycle),'request_ids':[p['id'] for p in wanted],'error':str(error)})
        completed+=1
        progress={'planned_cycles':len(cycles),'processed_cycles':completed,'failed_batches':failures,
                  'last_cycle':str(cycle),'complete':completed==len(cycles)}
        temporary=folder/'collection_progress.tmp';temporary.write_text(json.dumps(progress,indent=2));temporary.replace(folder/'collection_progress.json')
        if completed%20==0:LOG.info('Archive progress %s/%s cycles; failed batches=%s',completed,len(cycles),len(failures))
    return progress


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder',type=Path,required=True);p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--start',default='2025-11-15T00:00:00+07:00')
    p.add_argument('--end',default='2026-10-21T19:00:00+07:00')
    args=p.parse_args();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    collect(args.folder,args.snapshot,args.start,args.end)


if __name__=='__main__':main()
