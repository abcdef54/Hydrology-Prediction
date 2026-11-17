"""Run disjoint station workers without changing the reviewed scientific executor."""
import argparse,json,logging,os,shutil,subprocess,sys,time
from pathlib import Path
import pandas as pd
import yaml
from research.station_analysis.config import ROOT
from .data import Source,load_config,sha
from .training import reuse_key,write_json
LOG=logging.getLogger(__name__)


def build_library(work,c):
    library=work/'parallel_resume_library';(library/'fits').mkdir(parents=True,exist_ok=True)
    manifest=json.loads((work/'execution_manifest.json').read_text())
    write_json(library/'execution_manifest.json',manifest)
    archive=library/'execution_source';archive.mkdir(exist_ok=True)
    for source,expected in manifest['code'].items():
        if sha(source)!=expected:raise ValueError('Scientific execution code changed before parallel dispatch')
        shutil.copy2(source,archive/Path(source).name)
    candidates={}
    for origin in [ROOT/c['reuse_previous_run'],work]:
        old=json.loads((origin/'execution_manifest.json').read_text())
        if old['inputs']!=manifest['inputs']:raise ValueError('Resume library contains different source data')
        for done in sorted((origin/'fits').glob('*/done.json')):
            if not (done.parent/'identity.json').exists():continue
            identity=json.loads((done.parent/'identity.json').read_text());key=reuse_key(identity)
            count=json.loads(done.read_text())['trained_models']
            if key not in candidates or count>=candidates[key][0]:candidates[key]=(count,done.parent)
    for _,folder in candidates.values():
        destination=library/'fits'/folder.name
        if not destination.exists():destination.symlink_to(folder.resolve(),target_is_directory=True)
    write_json(library/'origins.json',{'canonical':str(work),'previous':c['reuse_previous_run'],
               'selected_cases':len(candidates),'rule':'identical numerical config/schema/seed/source; prefer more completed folds',
               'execution_source_sha256':{p.name:sha(p) for p in archive.iterdir()}})
    return library


def merge_station(source,work):
    ledger=json.loads((source/'ledger.json').read_text());child_work=source.parents[1]
    for key in sorted({r['fit_key'] for r in ledger if r.get('fit_key')}):
        origin=child_work/'fits'/key;dest=work/'fits'/key
        if not dest.exists():shutil.copytree(origin,dest)
        elif sha(dest/'identity.json')!=sha(origin/'identity.json'):raise ValueError('Fit-key collision across workers')
    target=work/'stations'/source.name
    if (target/'complete.json').exists():return
    target.mkdir(parents=True,exist_ok=True)
    # Publish the completion marker last so report/audit workers never see a partial station.
    for path in source.iterdir():
        if path.is_file() and path.name!='complete.json':shutil.copy2(path,target/path.name)
    shutil.copy2(source/'complete.json',target/'complete.json')


def execute(config_path,workers=2):
    if workers!=2:raise ValueError('Use the measured two-worker schedule; more workers were not resource-validated')
    c=load_config(config_path);work=Path(c['work_root'])/c['run_name'];out=Path(c['output_root'])/c['run_name']
    source=Source(c,work/'shared');eligibility=[]
    for row in source.water.itertuples():
        x,_,_=source.base(row.station_id);folds=source.folds(x.index)
        eligibility.append(dict(station_id=row.station_id,station_name=row.name,eligible=bool(folds),anchors=len(x),
            folds=';'.join(f['id'] for f in folds),reason='eligible' if folds else 'insufficient chronological training/validation before cutoff'))
    write_json(work/'eligibility.json',eligibility)
    complete={p.parent.name for p in (work/'stations').glob('*/complete.json')}
    pending=[r['station_id'] for r in eligibility if r['eligible'] and r['station_id'] not in complete]
    if not pending:
        from .reporting import finalize
        return finalize(work,out,c)
    library=build_library(work,c);raw=yaml.safe_load(Path(config_path).read_text());children=[];processes=[];handles=[]
    try:
        for i in range(workers):
            station_ids=pending[i::workers]
            if not station_ids:continue
            child_name=c['run_name']+f'_parallel_w{i+1}'
            child={**raw,'run_name':child_name,'reuse_previous_run':str(library.relative_to(ROOT))}
            child_config=work/f'parallel_w{i+1}.yaml';child_config.write_text(yaml.safe_dump(child,allow_unicode=True,sort_keys=False))
            child_work=Path(c['work_root'])/child_name;(child_work/'shared').mkdir(parents=True,exist_ok=True)
            for cache in (work/'shared').glob('weather_lag*.parquet'):shutil.copy2(cache,child_work/'shared'/cache.name)
            logfile=work/f'parallel_w{i+1}.log';handle=logfile.open('a');handles.append(handle)
            process=subprocess.Popen([sys.executable,'-m','research.station_analysis.phase2.feature_ablation.pipeline','--config',str(child_config),'--stations',*station_ids],stdout=handle,stderr=subprocess.STDOUT,env={**os.environ,'HYDROLOGY_STAGE4_COORDINATED_WORKER':'1'})
            processes.append(process);children.append(dict(worker=i+1,stations=station_ids,work=str(child_work),config=str(child_config),log=str(logfile),pid=process.pid))
            LOG.info('START worker=%s pid=%s stations=%s',i+1,process.pid,station_ids)
        write_json(work/'parallel_manifest.json',{'status':'running','workers':children,'canonical_signature':json.loads((work/'execution_manifest.json').read_text())['signature'],
                   'scientific_parameters_unchanged':True,'thermal_policy':'each worker pauses at 65C until <=59C; same physical GPU'})
        while True:
            for child in children:
                child_work=Path(child['work']);manifest=child_work/'execution_manifest.json'
                if manifest.exists():
                    parent=json.loads((work/'execution_manifest.json').read_text());record=json.loads(manifest.read_text())
                    ignore=['run_name','reuse_previous_run']
                    if {k:v for k,v in parent['config'].items() if k not in ignore}!={k:v for k,v in record['config'].items() if k not in ignore} or parent['inputs']!=record['inputs'] or parent['code']!=record['code']:
                        raise ValueError('Parallel worker changed scientific configuration, sources, or execution code')
                for done in (child_work/'stations').glob('*/complete.json'):
                    if done.parent.name not in complete:
                        merge_station(done.parent,work);complete.add(done.parent.name)
                        LOG.info('MERGED %s; completed=%s/%s',done.parent.name,len(complete),sum(r['eligible'] for r in eligibility))
            for process in processes:
                if process.poll() not in [None,0]:raise RuntimeError(f'Worker {process.pid} failed with exit {process.returncode}; inspect worker log and resume, do not report completion')
            if all(p.poll() is not None for p in processes):break
            time.sleep(2)
        # Preserve real telemetry across the coordinated run rather than reporting only the old serial temperatures.
        parts=[]
        for p in [work/'thermal.jsonl',*[Path(x['work'])/'thermal.jsonl' for x in children]]:
            if p.exists():parts.append(pd.read_json(p,lines=True))
        if parts:
            thermal=pd.concat(parts,ignore_index=True).drop_duplicates(['utc','c']).sort_values('utc')
            thermal.to_json(work/'thermal.jsonl',orient='records',lines=True,date_format='iso')
        write_json(work/'parallel_manifest.json',{'status':'complete','workers':children,'completed_stations':sorted(complete),
                   'scientific_parameters_unchanged':True,'child_manifests':{x['work']:sha(Path(x['work'])/'execution_manifest.json') for x in children}})
    finally:
        for handle in handles:handle.close()
        # An exception keeps child fits on disk for a later resume; stop active children to avoid orphan fitting.
        for process in processes:
            if process.poll() is None:process.terminate()
    from .reporting import finalize
    return finalize(work,out,c)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=Path(__file__).with_name('config.yaml'));p.add_argument('--workers',type=int,default=2)
    a=p.parse_args();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s');print(execute(a.config,a.workers),flush=True)
if __name__=='__main__':main()
