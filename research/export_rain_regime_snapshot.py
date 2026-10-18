"""Export a read-only consistent snapshot for rain regime diagnostics to /tmp.
Run from repository root with DATABASE_URL in the environment or local .env.
"""
import sys,json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from sqlalchemy import text
from dotenv import load_dotenv
from src.backend.db.session import build_engine
load_dotenv(ROOT / '.env', override=False)
out=Path('/tmp/hydrology-rain-regimes');out.mkdir(exist_ok=True)
model=json.loads(Path('src/backend/prediction/model_configs.json').read_text())[0]
with build_engine().connect().execution_options(isolation_level='REPEATABLE READ') as c:
 c.execute(text('SET TRANSACTION READ ONLY'))
 c.execute(text("SET LOCAL statement_timeout='90s'"))
 forecasts=pd.read_sql(text('''SELECT DISTINCT ON(f.anchor_observed_at,p.horizon_minutes) f.id AS forecast_id,f.anchor_observed_at,f.generated_at,f.model_version,f.pipeline_version,p.horizon_minutes,p.forecast_at,p.water_level_mm FROM hydrology.forecasts f JOIN hydrology.forecast_points p ON p.forecast_id=f.id WHERE f.station_id=:sid AND f.model_version=:model ORDER BY f.anchor_observed_at,p.horizon_minutes,f.generated_at DESC,f.id'''),c,params={'sid':model['station_id'],'model':model['model_version']})
 if forecasts.empty: raise RuntimeError('No saved forecasts')
 forecasts.to_csv(out/'forecasts.csv',index=False)
 start=pd.to_datetime(forecasts.anchor_observed_at,utc=True).min()-pd.Timedelta(hours=24)
 end=min(pd.to_datetime(forecasts.forecast_at,utc=True).max(),pd.Timestamp.now(tz='UTC').floor('h'))
 ids=model['rain_station_ids']+['750010','228005','750066']
 obs=pd.read_sql(text('''SELECT station_id,observed_at,depth FROM hydrology.observations WHERE station_id=ANY(:ids) AND observed_at>:start AND observed_at<=:end ORDER BY station_id,observed_at'''),c,params={'ids':ids+[model['station_id']],'start':start.to_pydatetime(),'end':end.to_pydatetime()})
 obs.to_csv(out/'observations.csv',index=False)
 stations=pd.read_sql(text('SELECT id,name,latitude,longitude FROM hydrology.stations WHERE id=ANY(:ids)'),c,params={'ids':ids})
 stations.to_csv(out/'stations.csv',index=False)
 print(json.dumps({'saved_forecast_points':len(forecasts),'anchors':forecasts.anchor_observed_at.nunique(),'first_anchor':str(start+pd.Timedelta(hours=24)),'last_anchor':str(forecasts.anchor_observed_at.max()),'observation_rows':len(obs),'end':str(end),'model':model['model_version'],'output':str(out)},ensure_ascii=False))
