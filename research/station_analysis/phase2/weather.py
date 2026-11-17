"""Stage 3 orchestration over frozen inputs; no database writes or model training."""
from datetime import datetime,timezone
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from research.station_analysis.config import sha256,digest,write_json
from .forecasts import audit_forecasts
from .rainfall_preprocessing import preprocess_canonical_water_levels
from .weather_quality import compare_forecasts
from .weather_relationships import assess_relationships
from .weather_reporting import write_weather_report

LOG=logging.getLogger(__name__)



def canonical_city_forecast(selected):
    """One city series, one vintage for all windows; keep raw request lineage."""
    if selected.model.nunique()!=1:
        raise ValueError('Configure one city forecast model before combining requests')
    ranked=selected.copy()
    ranked['has_run']=ranked.selected_run_id.notna()
    ranked=ranked.sort_values(['has_run','complete_24h','selected_initialized_at',
                               'selected_retrieved_at','location_id'],na_position='first')
    city=ranked.drop_duplicates(['policy','anchor_utc'],keep='last').copy()
    city['request_location_id']=city.location_id
    city['location_id']='hue_city_shared'
    return city.drop(columns='has_run').sort_values(['policy','anchor_utc']).reset_index(drop=True)


def run_weather(config, *, force=False, refresh_snapshot=False):
    if refresh_snapshot:
        raise ValueError('Stage 3 uses the reviewed frozen snapshot; refresh inventory separately first')
    work=Path(config['work_dir'])/'weather'
    output=Path(config['output_dir'])/'stage3_weather'
    inventory=Path(config['work_dir'])/'inventory'
    rainwork=Path(config['work_dir'])/'rainfall/tables'
    stage2=Path(config['output_dir'])/'stage2_rainfall/tables'
    pointer=json.loads((inventory/'current_snapshot.json').read_text())
    snapshot=Path(pointer['directory'])
    snapshot_manifest=json.loads((snapshot/'manifest.json').read_text())
    if sha256(snapshot/'manifest.json')!=pointer['manifest_sha256']:raise ValueError('Snapshot manifest changed')
    for name,expected in snapshot_manifest['sha256'].items():
        if sha256(snapshot/name)!=expected:raise ValueError(f'Snapshot changed: {name}')
    sources=[snapshot/'manifest.json',inventory/'tables/observation_availability.parquet',
             rainwork/'hourly_rainfall_standard.parquet',rainwork/'hourly_rainfall_strict.parquet',
             stage2/'06_single_station_candidates.csv']
    code={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}
    inputs={str(p):sha256(p) for p in sources}
    fingerprint=digest({'configuration':config,'inputs':inputs,'code':code})
    work.mkdir(parents=True,exist_ok=True);output.mkdir(parents=True,exist_ok=True)
    cache=work/'cache.json'
    if cache.exists() and not force:
        record=json.loads(cache.read_text())
        if record['fingerprint']==fingerprint and all(Path(p).exists() and sha256(p)==v for p,v in record['outputs'].items()):
            LOG.info('Reuse Stage 3 cache; no analyses or source queries repeated')
            return output/'report.md'
    LOG.info('Load frozen forecast catalog and point data; reuse Stage 2 rainfall')
    frames={name:pd.read_parquet(snapshot/f'{name}.parquet') for name in ['stations','forecast_runs','forecast_points','weather_locations']}
    rain=pd.read_parquet(rainwork/'hourly_rainfall_standard.parquet')
    strict=pd.read_parquet(rainwork/'hourly_rainfall_strict.parquet')
    anchors=rain.index
    start=pd.Timestamp(config['start']);end=pd.Timestamp(config['end']) if config.get('end') else anchors.max()+pd.Timedelta(hours=1)
    anchors=anchors[(anchors>=start)&(anchors<end)]
    if len(anchors)==0:raise ValueError('No anchors in requested interval')
    stations=frames['stations']
    water=stations[stations.station_type.eq('water_level')]
    if config.get('water_station_ids'):water=water[water.station_id.isin(config['water_station_ids'])]
    if config.get('rain_station_ids'):
        ids=[sid for sid in config['rain_station_ids'] if sid in rain.columns]
        if not ids:raise ValueError('No selected rain stations available')
        rain=rain[ids];strict=strict[ids]
    # Retain full past context for rolling inputs, then restrict analysis anchors explicitly.
    _,targets=preprocess_canonical_water_levels(pd.read_parquet(inventory/'tables/observation_availability.parquet'),water,rain.index)
    candidates=pd.read_csv(stage2/'06_single_station_candidates.csv',dtype={'water_station_id':str,'rain_station_id':str})
    LOG.info('Revalidate run integrity and reproduce strict vs assumed-release selection')
    tables=audit_forecasts(frames['forecast_runs'],frames['forecast_points'],frames['weather_locations'],anchors,config)
    request_selected=tables['forecast_anchor_availability']
    selected=canonical_city_forecast(request_selected)
    tables['weather_city_selection']=selected
    grids=tables['forecast_location_grids']
    unique_grids=grids[['grid_latitude','grid_longitude']].drop_duplicates()
    if len(unique_grids)!=1:
        raise ValueError('City forecast requests returned different grids; define a canonical spatial source before analysis')
    city_grid=grids.iloc[[0]].copy()
    city_grid['location_id']='hue_city_shared'
    coverage=[]
    for policy,part in selected.groupby('policy'):
        coverage.append({'policy':policy,'anchor_count':len(part),'selected_n':int(part.selected_run_id.notna().sum()),
                         'complete24_n':int(part.complete_24h.sum()),
                         'hindcast_selected_n':int(part.hindcast_selected.sum()),
                         'retrieved_after_anchor_n':int(part.retrieved_after_anchor.sum()),
                         'age_median_h':part.age_hours.median(),'age_p95_h':part.age_hours.quantile(.95)})
    tables['weather_city_coverage']=pd.DataFrame(coverage)
    LOG.info('Verify exact rainfall intervals against gauges, split by provenance/season/age')
    comparators,pairs,skill=compare_forecasts(selected,rain,strict,stations,city_grid,config)
    tables.update(weather_comparators=comparators,weather_skill=skill)
    LOG.info('Assess all water stations and train-fitted partial associations')
    # All target grids share the original context; masks exclude anchors outside requested analysis.
    trimmed_targets={sid:frame.loc[anchors] for sid,frame in targets.items()}
    tables.update(assess_relationships(selected,trimmed_targets,rain,candidates,stations,
                                      config['future_analysis']['grouping_end']))
    # Compare duplicate-grid source views on common anchors; do not double sample size.
    views=request_selected[request_selected.policy.eq('legacy_assumed_release')]
    locations=sorted(views.location_id.unique());duplicates=[]
    for i,left in enumerate(locations):
        for right in locations[i+1:]:
            a=views[views.location_id.eq(left)].set_index('anchor_utc')
            b=views[views.location_id.eq(right)].set_index('anchor_utc')
            for col in ['rain_0_3h_mm','rain_3_6h_mm','rain_6_12h_mm','rain_12_24h_mm']:
                aligned=pd.concat([a[col],b[col]],axis=1).dropna()
                difference=(aligned.iloc[:,0]-aligned.iloc[:,1]).abs()
                duplicates.append({'location_a':left,'location_b':right,'feature':col,'common_n':len(aligned),
                                   'equal_within_1e_9':int(difference.le(1e-9).sum()),'max_abs_difference_mm':difference.max()})
    tables['weather_duplicate_grid_comparison']=pd.DataFrame(duplicates)
    for name,frame in tables.items():
        (work/'tables').mkdir(exist_ok=True)
        frame.to_parquet(work/'tables'/f'{name}.parquet',index=False)
    pairs.to_parquet(work/'matched_rainfall.parquet',index=False)
    LOG.info('Write readable Stage 3 report, figures, tables and manifest')
    report=write_weather_report(tables,pairs,frames,config,output)
    write_json(output/'run_manifest.json',{'stage':'weather','completed_at_utc':datetime.now(timezone.utc).isoformat(),
               'snapshot_directory':str(snapshot),'snapshot_manifest_sha256':sha256(snapshot/'manifest.json'),
               'configuration':config,'code_sha256':code,'input_sha256':inputs,
               'fingerprint':fingerprint,'analysis_start_utc':str(anchors.min()),'analysis_end_utc':str(anchors.max()),
               'not_executed':['stage4','regional_training','training_dataset_changes'],
               'rows_by_table':{name:len(frame) for name,frame in tables.items()}})
    paths=[p for p in work.rglob('*.parquet')]+[p for p in output.rglob('*') if p.is_file() and p.name!='pipeline.log']
    write_json(cache,{'fingerprint':fingerprint,'outputs':{str(p):sha256(p) for p in paths}})
    LOG.info('Stage 3 complete: %s',report)
    return report
