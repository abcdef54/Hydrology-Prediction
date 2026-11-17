"""Execute the complete reviewed local XGBoost study; safe to resume per fitted case."""
import argparse,json,logging,time
from pathlib import Path
import numpy as np
import pandas as pd
from .data import Source,load_config,sha,BASE
from .planning import principal,diagnostics,discover,choose_finalists,arms_for
from .training import Thermal,run_case,write_json,digest
LOG=logging.getLogger(__name__)


def normalized(task,discovery):
    e=task['experiment'];ids=sorted(task['grids']);mode=task.get('mode','normal');lag=task.get('lag',6)
    rain=sorted(discovery['selected_rain'] if e=='E5' else discovery['fixed_rain'])
    if e=='E5':e='E3'
    if not ids and e in ['E3','E4']:e='E1'
    if not ids and e=='E2':e='E0'
    if e in ['E0','E2']:rain=[]
    if e in ['E0','E1']:ids=[];mode='strict' if mode=='strict' and rain else 'normal';lag=6
    if mode=='duplicate':ids=[BASE]*len(ids)
    return dict(experiment=e,grids=ids,rain_ids=rain,mode=mode,lag=lag,version=task.get('version'))


def execute(config_path,station_ids=None,report_only=False):
    c=load_config(config_path);work=Path(c['work_root'])/c['run_name'];out=Path(c['output_root'])/c['run_name']
    work.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    if report_only:
        from .reporting import finalize
        return finalize(work,out,c)
    source=Source(c,work/'shared')
    execution_files=[Path(__file__),Path(__file__).with_name('data.py'),Path(__file__).with_name('training.py'),Path(__file__).with_name('planning.py')]
    provenance={'config':c,'inputs':{str(p):sha(p) for p in source.input_paths},'code':{str(p):sha(p) for p in execution_files},
                'stage3b_report':str(source.report),'specification':str(source.report/'stage4_spatial_ablation_spec.json')}
    signature=digest(provenance);manifest=work/'execution_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['signature']!=signature:
        raise ValueError('Execution inputs/code changed; use a new run_name, never mix incompatible cached fits')
    write_json(manifest,{**provenance,'signature':signature,'created_at':pd.Timestamp.now(tz='UTC').isoformat(),
                        'not_executed':['regional_training','production_changes','regional_dataset_preparation']})
    stations=['0074647403']+[s for s in source.water.station_id if s!='0074647403']
    if station_ids:stations=[s for s in stations if s in station_ids]
    thermal=Thermal(work,c);eligibility=[]
    try:
        for sid in stations:
            station=source.water.set_index('station_id').loc[sid]
            x,y,h=source.base(sid);folds=source.folds(x.index)
            eligible=bool(folds)
            eligibility.append(dict(station_id=sid,station_name=station['name'],eligible=eligible,
                anchors=len(x),folds=';'.join(f['id'] for f in folds),
                reason='eligible' if eligible else 'insufficient chronological training/validation before cutoff'))
            write_json(work/'eligibility.json',eligibility)
            if not eligible:LOG.info('INELIGIBLE %s %s',sid,station['name']);continue
            sd=work/'stations'/sid;sd.mkdir(parents=True,exist_ok=True)
            discovery_path=sd/'discovery.json'
            if discovery_path.exists():discovery=json.loads(discovery_path.read_text())
            else:discovery=discover(source,sid,x,y,folds);write_json(discovery_path,discovery)
            # Freeze sample provenance, fold membership, and physically verified targets before fitting.
            population=pd.DataFrame({'anchor_utc':x.index,'water_now_mm':h.to_numpy(),
                                     'dh6_mm':y.h6.to_numpy(),'dh12_mm':y.h12.to_numpy()})
            for f in folds:
                population[f['id']+'_train']=f['train'];population[f['id']+'_validation']=f['val']
            population.to_parquet(sd/'population.parquet',index=False)
            ledger=[];metrics=[];start=time.monotonic()
            def task_run(task):
                norm=normalized(task,discovery);design=source.features(x,norm['grids'],norm['rain_ids'],norm['experiment'],norm['mode'],norm['lag'])
                local_y=y;local_h=h;local_folds=folds
                if norm['version']:
                    kind=source.weather[6][f'{BASE}|archive_kind'].reindex(x.index)
                    mask=kind.str.contains(norm['version'],regex=False).fillna(False)
                    design=design.loc[mask];local_y=y.loc[mask];local_h=h.loc[mask];local_folds=source.folds(design.index)
                    # Within-version retraining needs multiple training rainfall episodes; never fake event replication.
                    allowed=[]
                    for f in local_folds:
                        ti=design.index[f['train']]
                        count=sum(((ti>=pd.Timestamp(ep.start_utc))&(ti<=pd.Timestamp(ep.end_utc))).sum()>=12 for ep in source.rain_events.itertuples())
                        if count>=2:allowed.append(f)
                    local_folds=allowed
                    if not local_folds:
                        ledger.append({**task,'station_id':sid,'status':'not_estimable','reason':'D8: no within-version fold has required train/validation and >=2 rainfall episodes'})
                        return
                alias={**task,'station_id':sid,'station_name':station['name'],'mode':task.get('mode','normal'),'lag':task.get('lag',6)}
                for seed in c['seeds']:
                    meta={'station_id':sid,**norm,'execution_signature':signature}
                    key=run_case(work,c,thermal,design,local_y,local_h,local_folds,meta,seed)
                    ledger.append({**alias,'seed':seed,'fit_key':key,'status':'completed','feature_count':len(design.columns)})
                    fit=pd.read_csv(work/'fits'/key/'metrics.csv')
                    for record in fit.to_dict('records'):metrics.append({**alias,'seed':seed,'fit_key':key,**record})
                pd.DataFrame(ledger).to_json(sd/'ledger.json',orient='records',indent=2,force_ascii=False)
                pd.DataFrame(metrics).to_csv(sd/'metrics.csv',index=False)
                LOG.info('TASK %s %s/%s mode=%s lag=%s %s aliases=%s elapsed=%.1fm',sid,task['experiment'],task['arm'],norm['mode'],norm['lag'],task.get('diagnostic','principal'),len(ledger),(time.monotonic()-start)/60)
            LOG.info('STATION START %s %s folds=%s anchors=%s',sid,station['name'],[f['id'] for f in folds],len(x))
            for task in principal(source,sid):task_run(task)
            finalists,scores=choose_finalists(metrics);scores.to_csv(sd/'validation_selection.csv',index=False)
            write_json(sd/'finalists.json',{'arms':finalists,'selection_scope':'pre-cutoff chronological validation only','seed_policy':'all five paired seeds averaged'})
            tasks=diagnostics(source,sid,finalists,discovery)
            if not discovery['challenger']:ledger.append(dict(station_id=sid,diagnostic='D9',status='not_triggered',reason='no excluded grid reaches prespecified earliest-train residual-association threshold 0.05'))
            for task in tasks:task_run(task)
            # Save even when last diagnostic is inestimable.
            pd.DataFrame(ledger).to_json(sd/'ledger.json',orient='records',indent=2,force_ascii=False)
            pd.DataFrame(metrics).to_csv(sd/'metrics.csv',index=False)
            write_json(sd/'complete.json',{'station_id':sid,'completed_at':pd.Timestamp.now(tz='UTC').isoformat(),
                'requested_principal':len(principal(source,sid)),'requested_diagnostic':len(tasks),'seed_count':len(c['seeds']),
                'ledger_rows':len(ledger),'unique_fits':len({r.get('fit_key') for r in ledger if r.get('fit_key')})})
            LOG.info('STATION COMPLETE %s aliases=%s unique_fits=%s elapsed=%.1fm',sid,len(ledger),len({r.get('fit_key') for r in ledger if r.get('fit_key')}),(time.monotonic()-start)/60)
    finally:thermal.close()
    from .reporting import finalize
    return finalize(work,out,c)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path(__file__).with_name('config.yaml'))
    p.add_argument('--stations',nargs='+');p.add_argument('--report-only',action='store_true')
    args=p.parse_args();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    print(execute(args.config,args.stations,args.report_only),flush=True)
if __name__=='__main__':main()
