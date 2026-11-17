"""Execute the reviewed Stage 3B over a resumable, versioned research archive."""
from datetime import datetime,timezone
import json
import logging
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from research.station_analysis.config import digest,sha256,write_json
from ..forecasts import audit_forecasts
from ..rainfall_preprocessing import preprocess_canonical_water_levels,extract_rain_station_dem
from ..weather_quality import compare_forecasts,haversine_km
from .processing import prepare_archive,synchronized_features
from .analysis import intergrid_statistics,spatial_water_relationships,event_rain_skill,recommendations,same_gauge_comparisons
from .reporting import report_spatial_weather

LOG=logging.getLogger(__name__)


def run_spatial_weather(config,*,force=False,refresh_snapshot=False):
    if refresh_snapshot:raise ValueError('Stage 3B must use the reviewed frozen snapshot')
    settings=config['spatial_weather']
    if settings['model']!='ecmwf_ifs' or settings['cell_selection']!='land':
        raise ValueError('Stage 3B baseline requires ecmwf_ifs and land selection')
    root=Path(config['work_dir'])/'spatial_weather'
    collection=root/config['spatial_weather']['collection_version']
    output_root=Path(config['output_dir'])/'stage3b_spatial_weather'
    pointer=json.loads((Path(config['work_dir'])/'inventory/current_snapshot.json').read_text())
    snapshot=Path(pointer['directory'])
    if sha256(snapshot/'manifest.json')!=pointer['manifest_sha256']:raise ValueError('Source snapshot manifest changed')
    manifest=json.loads((snapshot/'manifest.json').read_text())
    for name,expected in manifest['sha256'].items():
        if sha256(snapshot/name)!=expected:raise ValueError(f'Frozen snapshot file changed: {name}')
    progress=json.loads((collection/'collection_progress.json').read_text())
    locations=json.loads((collection/'locations.json').read_text())
    if set(p['id'] for p in locations)!=set(settings['requested_station_ids']):
        raise ValueError('Configured requests differ from the validated collection; create a new collection version')
    if not progress.get('complete'):raise ValueError('Collection still running; do not report partial progress as full archive')
    raw=list((collection/'raw').glob('*_land_*.json'))
    paths=raw+[snapshot/'manifest.json',collection/'validation_comparison.json',
               collection/'collection_progress.json',collection/'locations.json']
    paths += [collection/name for name in ['validation_runs.parquet','validation_points.parquet',
              'validation_mapping.parquet','nearest_sensitivity.csv'] if (collection/name).exists()]
    source_tables=Path(config['work_dir'])/'inventory/tables'
    rain_tables=Path(config['work_dir'])/'rainfall/tables'
    stage2=Path(config['output_dir'])/'stage2_rainfall/tables'
    paths += [source_tables/'observation_availability.parquet',rain_tables/'hourly_rainfall_standard.parquet',
              rain_tables/'hourly_rainfall_strict.parquet',stage2/'06_single_station_candidates.csv',
              stage2/'05_independent_rainfall_events.csv']
    code={str(p.relative_to(Path(__file__).parents[1])):sha256(p) for p in Path(__file__).parents[1].rglob('*.py')}
    inputs={str(p):sha256(p) for p in paths}
    fingerprint=digest({'config':config,'inputs':inputs,'code':code})
    latest=root/'latest_analysis.json'
    if latest.exists() and not force:
        previous=json.loads(latest.read_text())
        if previous['fingerprint']==fingerprint and all(Path(p).exists() and sha256(p)==h for p,h in previous['outputs'].items()):
            LOG.info('Reuse completed Stage 3B analysis %s',previous['report']);return Path(previous['report'])
    version=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:8]
    work=root/'analyses'/version;output=output_root/version
    work.mkdir(parents=True);output.mkdir(parents=True)
    LOG.info('Build immutable research archive version %s',version)
    runs,points,mapping,grids,checks=prepare_archive(collection,snapshot)
    stations=pd.read_parquet(snapshot/'stations.parquet')
    rain=pd.read_parquet(rain_tables/'hourly_rainfall_standard.parquet')
    rain_strict=pd.read_parquet(rain_tables/'hourly_rainfall_strict.parquet')
    anchors=rain.index
    if config.get('end'):anchors=anchors[anchors<pd.Timestamp(config['end'])]
    anchors=anchors[anchors>=pd.Timestamp(config['start'])]
    earliest=(anchors.min()-pd.Timedelta(hours=6)).floor('6h');last=(anchors.max()-pd.Timedelta(hours=6)).floor('6h')
    planned=pd.date_range(earliest,last,freq='6h')
    runs=runs[runs.initialized_at.between(earliest,last)]
    points=points[points.run_id.isin(runs.run_id)]
    mapping=mapping[mapping.run_id.isin(runs.run_id)].copy()
    grids=grids[grids.grid_id.isin(runs.location_id)].copy()
    spans=runs.groupby('location_id').initialized_at.agg(['min','max','nunique'])
    grids['first_run']=grids.grid_id.map(spans['min'])
    grids['last_run']=grids.grid_id.map(spans['max'])
    grids['run_count']=grids.grid_id.map(spans['nunique'])
    for name,frame in [('runs',runs),('points',points),('request_mapping',mapping),('grid_registry',grids)]:frame.to_parquet(work/f'{name}.parquet',index=False)
    LOG.info('Audit spatial run availability and forecast integrity')
    tables=audit_forecasts(runs,points,grids,anchors,config)
    coverage=[];missing=[]
    for grid in grids.itertuples():
        observed=runs.loc[runs.location_id.eq(grid.grid_id),'initialized_at'].drop_duplicates()
        absent=planned.difference(observed)
        coverage.append({'grid_id':grid.grid_id,'expected_cycles':len(planned),'actual_cycles':len(observed),
                         'missing_cycles':len(absent),'coverage_pct':100*len(observed)/len(planned)})
        missing.extend({'grid_id':grid.grid_id,'initialized_at':t} for t in absent)
    tables.update(grid_registry=grids,grid_run_coverage=pd.DataFrame(coverage),
                  missing_grid_runs=pd.DataFrame(missing,columns=['grid_id','initialized_at']),
                  request_mapping=mapping,reference_replay=checks)
    grid_dem_input=grids[['grid_id','grid_name','grid_latitude','grid_longitude']].rename(columns={
        'grid_id':'station_id','grid_name':'name','grid_latitude':'latitude','grid_longitude':'longitude'})
    grid_dem=extract_rain_station_dem(grid_dem_input).rename(columns={'station_id':'grid_id','elevation_m':'center_dem_elevation_m'})
    tables['grid_terrain']=grid_dem
    tables['preflight_checks']=pd.DataFrame(json.loads((collection/'validation_comparison.json').read_text()))
    if (collection/'nearest_sensitivity.csv').exists():
        tables['nearest_sensitivity']=pd.read_csv(collection/'nearest_sensitivity.csv',dtype={'request_id':str})
    integrity=tables['forecast_run_integrity'].copy()
    integrity['empty_run']=integrity.valid_forecast_hours.eq(0)
    tables['grid_integrity_summary']=integrity.groupby('location_id',as_index=False).agg(
        run_count=('run_id','size'),valid_hours=('valid_forecast_hours','sum'),
        missing_value_rows=('missing_value_rows','sum'),empty_runs=('empty_run','sum'),
        duplicate_hour_rows=('duplicated_hour_rows','sum'),negative_rows=('negative_value_rows','sum'))
    selected=tables['forecast_anchor_availability']
    LOG.info('Build common-initialization vectors for %s grids',len(grids))
    synced=synchronized_features(selected,grids.grid_id.tolist(),anchors,
                                  config['assumed_release_lag_hours'],config['max_forecast_age_hours'],
                                  tables['forecast_run_integrity'],points)
    synced.to_parquet(work/'synchronized_features.parquet',index=False)
    LOG.info('Verify rainfall skill grid by grid, retaining gauge comparisons separately')
    quality=[];comparators=[];event_skill=[]
    terrain=extract_rain_station_dem(stations[stations.station_type.eq('rainfall')])
    episodes=pd.read_csv(stage2/'05_independent_rainfall_events.csv').rename(columns={'event_id':'episode'})
    reference_by_grid={gid:[] for gid in grids.grid_id}
    quality_config={**config,'weather_analysis':{**config['weather_analysis'],
                    'reference_gauges_by_location':reference_by_grid,
                    'nearby_gauges':settings['nearby_gauges']}}
    for gid in grids.grid_id:
        part=selected[selected.location_id.eq(gid)&selected.policy.eq('legacy_assumed_release')]
        comp,pairs,skill=compare_forecasts(part,rain,rain_strict,stations,grids,quality_config)
        comp=comp.merge(terrain[['station_id','elevation_m']],left_on='gauge_id',right_on='station_id',how='left')
        comp=comp.merge(grid_dem[['grid_id','center_dem_elevation_m']],left_on='location_id',right_on='grid_id',how='left')
        comp['gauge_minus_gridcenter_elevation_m']=comp.elevation_m-comp.center_dem_elevation_m
        comparators.append(comp);quality.append(skill)
        event_skill.append(event_rain_skill(pairs,episodes))
        pairs.to_parquet(work/f'matched_rain_{gid.replace(":","_")}.parquet',index=False)
        LOG.info('Rainfall verification complete for %s',gid)
    tables.update(spatial_rain_skill=pd.concat(quality,ignore_index=True),
                  gauge_comparators=pd.concat(comparators,ignore_index=True),
                  rainfall_event_skill=pd.concat(event_skill,ignore_index=True),rain_gauge_terrain=terrain)
    LOG.info('Compare spatial redundancy and synchronized forecast signals')
    intergrid,spread=intergrid_statistics(synced,grids,config['future_analysis']['grouping_end'])
    tables.update(intergrid_correlations=intergrid,spatial_heterogeneity=spread)
    tables['same_gauge_grid_comparisons']=same_gauge_comparisons(synced,tables['gauge_comparators'],rain,config['future_analysis']['grouping_end'])
    water=stations[stations.station_type.eq('water_level')]
    _,targets=preprocess_canonical_water_levels(pd.read_parquet(source_tables/'observation_availability.parquet'),water,rain.index)
    candidates=pd.read_csv(stage2/'06_single_station_candidates.csv',dtype={'water_station_id':str,'rain_station_id':str})
    baseline=grids.loc[(grids.grid_latitude-16.274164).abs().lt(1e-5)&(grids.grid_longitude-107.264960).abs().lt(1e-5),'grid_id'].iloc[0]
    LOG.info('Assess water relationships and train-only additional associations')
    relationships,partial,controls=spatial_water_relationships(synced,grids,targets,rain,candidates,stations,
                                         config['future_analysis']['grouping_end'],baseline)
    tables.update(spatial_water_relationships=relationships,spatial_partial_associations=partial,water_controls=controls)
    arms=recommendations(grids,stations,candidates,partial,tables['grid_run_coverage'],config['future_analysis']['grouping_end'])
    coverage_map=tables['grid_run_coverage'].set_index('grid_id').coverage_pct.to_dict()
    coordinates=grids.set_index('grid_id')[['grid_latitude','grid_longitude']].to_dict('index')
    pair_lookup=intergrid[intergrid.scope.eq('train')&intergrid.season.eq('all')].groupby(['grid_a','grid_b']).pearson_r.mean().to_dict()
    def arm_details(row):
        ids=row.grid_ids.split(';') if row.grid_ids else []
        pairs=[pair_lookup.get((a,b),pair_lookup.get((b,a),np.nan)) for i,a in enumerate(ids) for b in ids[i+1:]]
        return pd.Series({'grid_coordinates_json':json.dumps({g:coordinates[g] for g in ids}),
                          'minimum_run_coverage_pct':min([coverage_map[g] for g in ids],default=np.nan),
                          'max_mean_pairwise_train_r':max(pairs,default=np.nan),
                          'planned_grid_cycles':len(planned)*len(ids),
                          'common_anchor_table':'synchronized_features.parquet',
                          'skill_table':'spatial_rain_skill.csv',
                          'availability_status':'hypothetical 6h release; no independent operational evidence',
                          'provenance_warning':'49r1 hindcasts / 50r1 archives; initialization is not publication'})
    arms=pd.concat([arms,arms.apply(arm_details,axis=1)],axis=1)
    tables['candidate_arms']=arms
    table_dir=work/'tables';table_dir.mkdir()
    for name,frame in tables.items():frame.to_parquet(table_dir/f'{name}.parquet',index=False)
    LOG.info('Write Vietnamese report, maps, comparison figures and Stage 4 specification')
    report=report_spatial_weather(tables,synced,stations,config,output,work,progress)
    write_json(output/'manifest.json',{'stage':'spatial_weather','version':version,'completed_at_utc':datetime.now(timezone.utc).isoformat(),
               'configuration':config,'snapshot_directory':str(snapshot),'input_sha256':inputs,'code_sha256':code,
               'collection_version':str(collection),'fingerprint':fingerprint,
               'synchronized_anchors':len(synced),'analysis_start_utc':str(anchors.min()),'analysis_end_utc':str(anchors.max()),
               'rows_by_table':{name:len(frame) for name,frame in tables.items()},
               'not_executed':['Stage4','XGBoost_training','RegionalLSTM_training','production_changes']})
    outputs={str(p):sha256(p) for base in [work,output] for p in base.rglob('*') if p.is_file()}
    write_json(latest,{'fingerprint':fingerprint,'report':str(report),'outputs':outputs})
    output_root.mkdir(exist_ok=True)
    write_json(output_root/'latest.json',{'version':version,'report':str(report)})
    LOG.info('Stage 3B complete: %s',report)
    return report
