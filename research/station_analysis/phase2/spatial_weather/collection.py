"""Cached individual-run collection with bounded requests and resumable checkpoints."""
from datetime import datetime,timezone
import hashlib
import json
import logging
from pathlib import Path
import time

import pandas as pd
import requests

from src.backend.ingestion.weather_forecast import request_params,parse_response

LOG=logging.getLogger(__name__)
TRANSITION=pd.Timestamp('2026-05-12T06:00:00Z')
ENDPOINT='https://single-runs-api.open-meteo.com/v1/forecast'


def grid_identity(model,latitude,longitude):
    """Coordinates identify the returned cell; station/request names do not."""
    return f'{model}:{float(latitude):.6f}:{float(longitude):.6f}'


def classify(cycle):
    return ('hindcast_49r1','hindcast/reconstructed','49r1') if pd.Timestamp(cycle)<TRANSITION else (
        'archive_50r1_unverified','assumed_availability','50r1')


class Collector:
    def __init__(self,folder,settings):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.settings=settings;self.client=requests.Session()
        self.ledger=self.folder/'request_ledger.jsonl'

    def throttle(self,cost):
        """Conservative accounting treats each coordinate as one charged call."""
        while True:
            now=time.time()
            records=[json.loads(line) for line in self.ledger.read_text().splitlines()] if self.ledger.exists() else []
            wait=0.
            for seconds,limit in [(60,540),(3600,4800),(86400,9500)]:
                recent=[r for r in records if now-r['time']<seconds]
                if sum(r['cost'] for r in recent)+cost>limit:
                    wait=max(wait,seconds-(now-min(r['time'] for r in recent))+1)
            if wait<=0:return
            LOG.info('API budget pause %.0fs; checkpoint safe',wait)
            time.sleep(min(30,wait))

    def fetch(self,locations,cycle,selection='land'):
        cycle=pd.Timestamp(cycle)
        if cycle.tzinfo is None:raise ValueError('Cycle requires an explicit timezone')
        cycle=cycle.tz_convert('UTC')
        if cycle.hour%6 or cycle.minute or cycle.second:raise ValueError('Expected a six-hour UTC initialization cycle')
        params=request_params(locations[0],cycle.to_pydatetime())
        params['latitude']=','.join(str(p['latitude']) for p in locations)
        params['longitude']=','.join(str(p['longitude']) for p in locations)
        params['cell_selection']=selection
        key=hashlib.sha256(json.dumps(params,sort_keys=True).encode()).hexdigest()[:20]
        path=self.folder/f'{cycle.strftime("%Y%m%dT%H%M")}_{selection}_{key}.json'
        if path.exists():return json.loads(path.read_text())
        last=None
        for attempt in range(self.settings.get('retries',4)):
            self.throttle(len(locations))
            started=time.time()
            with self.ledger.open('a') as stream:stream.write(json.dumps({'time':started,'cost':len(locations),'key':key})+'\n')
            try:
                response=self.client.get(ENDPOINT,params=params,timeout=(15,90))
                if response.status_code==429:
                    delay=max(10,float(response.headers.get('Retry-After','60')))
                    for _ in range(int(delay//30)):time.sleep(30)
                    time.sleep(delay%30)
                    raise requests.HTTPError('Rate limited')
                response.raise_for_status();payload=response.json()
                entries=payload if isinstance(payload,list) else [payload]
                if len(entries)!=len(locations):raise ValueError('Response location count differs from request')
                for item in entries:parse_response(item,cycle.to_pydatetime())
                record={'endpoint':ENDPOINT,'parameters':params,'initialized_at':cycle.isoformat(),
                        'retrieved_at':datetime.now(timezone.utc).isoformat(),'locations':locations,'payload':entries}
                temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(record,ensure_ascii=False));temporary.replace(path)
                time.sleep(self.settings.get('request_interval_seconds',1.1))
                return record
            except (requests.RequestException,ValueError) as error:
                last=f'{type(error).__name__}: {str(error)[:180]}'
                LOG.warning('Run %s attempt %s failed: %s',cycle,attempt+1,last)
                time.sleep(min(30,2**attempt))
        raise RuntimeError(last)


def flatten_records(records):
    """Deduplicate equal cell/run payloads, preserving request mappings and conflicts."""
    runs=[];points=[];mapping=[];seen={};conflicts=[]
    for record in records:
        cycle=pd.Timestamp(record['initialized_at'])
        kind,provenance,version=classify(cycle)
        for position,(location,item) in enumerate(zip(record['locations'],record['payload'])):
            if item.get('location_id',position)!=position:
                raise ValueError('Provider response location order differs from the requested coordinate order')
            gid=grid_identity('ecmwf_ifs',item['latitude'],item['longitude'])
            parsed=parse_response(item,cycle.to_pydatetime())
            canonical=[(str(p['valid_at']),p['precipitation_mm']) for p in parsed]
            checksum=hashlib.sha256(json.dumps(canonical).encode()).hexdigest()
            run_id=hashlib.sha256(f'{gid}|{cycle.isoformat()}|{checksum}'.encode()).hexdigest()
            mapping.append({'request_id':location['id'],'request_station_id':location['station_id'],
                            'request_latitude':location['latitude'],'request_longitude':location['longitude'],
                            'initialized_at':cycle,'grid_id':gid,'grid_latitude':item['latitude'],
                            'grid_longitude':item['longitude'],'cell_selection':record['parameters']['cell_selection'],
                            'run_id':run_id,'checksum':checksum,'model':'ecmwf_ifs','model_version':version,
                            'version_basis':'official provider transition; not returned payload metadata',
                            'source_artifact':record.get('_source_path'),
                            'response_position':position,
                            'request_parameters_json':json.dumps(record['parameters'],sort_keys=True)})
            identity=(gid,cycle)
            if identity in seen:
                if seen[identity]!=checksum:conflicts.append({'grid_id':gid,'initialized_at':cycle,'conflicting_checksum':checksum})
                continue
            seen[identity]=checksum
            runs.append({'run_id':run_id,'location_id':gid,'model':'ecmwf_ifs','initialized_at':cycle,
                         'retrieved_at':pd.Timestamp(record['retrieved_at']),'available_at':pd.NaT,
                         'availability_basis':'unknown_archive','archive_kind':kind,
                         'grid_latitude':item['latitude'],'grid_longitude':item['longitude'],
                         'checksum':checksum,'hourly_units':json.dumps(item['hourly_units']),
                         'utc_offset_seconds':item['utc_offset_seconds'],'provenance_class':provenance,
                         'source_artifact':record.get('_source_path'),
                         'request_parameters_json':json.dumps(record['parameters'],sort_keys=True)})
            points.extend({'run_id':run_id,**p} for p in parsed)
    if conflicts:raise ValueError(f'{len(conflicts)} conflicting cell/run payloads; investigate before analysis')
    return pd.DataFrame(runs),pd.DataFrame(points),pd.DataFrame(mapping)
