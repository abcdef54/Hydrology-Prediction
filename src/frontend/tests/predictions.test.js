import assert from 'node:assert/strict';
import test from 'node:test';
import { createPrediction, fetchForecastDetail } from '../src/api/predictions.js';
import { buildComparisonRows, splitActualSegments, getWaterLevelDomain } from '../src/predictions.js';

function prediction() {
  return {
    station_id: '0074647403', unit: 'mm', forecast_id: '12345678-1234-1234-1234-123456789abc',
    model_version: 'm1', feature_schema_version: 'f1', pipeline_version: 'p1',
    anchor_observed_at: '2026-10-13T00:00:00Z', generated_at: '2026-10-14T01:00:00Z',
    is_stale: false, overrides_applied: {},
    predictions: [60, 180].map((horizon_minutes, index) => ({ horizon_minutes,
      forecast_at: new Date(Date.parse('2026-10-13T00:00:00Z') + horizon_minutes * 60000).toISOString(),
      water_level_mm: index ? -20 : 0 })),
  };
}
function detail() {
  return { ...prediction(), window_end: '2026-10-13T03:00:00Z',
    actuals: [0, null, -30].map((value, index) => ({ timestamp: `2026-10-13T0${index+1}:00:00Z`,
      actual_water_level_mm: value, actual_sample_count: value === null ? 0 : 6 })) };
}

test('POST giữ trạm/mốc, không gửi overrides; GET đọc đúng ID, giữ 0/âm/null', async context => {
  const calls = [];
  context.mock.method(globalThis, 'fetch', async (url, options) => {
    calls.push({url, options});return Response.json(options.method === 'POST' ? prediction() : detail());
  });
  const result = await createPrediction('0074647403', '2026-10-13T07:00:00+07:00');
  assert.deepEqual(JSON.parse(calls[0].options.body), {station_id:'0074647403',reference_time:'2026-10-13T07:00:00+07:00'});
  assert.equal(calls[0].options.method,'POST');
  const comparison = await fetchForecastDetail(result);
  assert.equal(calls[1].url,`/api/v1/forecasts/${result.forecast_id}`);
  assert.deepEqual(comparison.actuals.map(point=>point.actual_water_level_mm),[0,null,-30]);
});

test('từ chối sai đơn vị/mốc/model/ID, actual thiếu hàng và giá trị sai', async context => {
  let response;
  context.mock.method(globalThis, 'fetch', async () => Response.json(response));
  for(const change of [{unit:'m'},{forecast_id:null},{predictions:[{...prediction().predictions[0],forecast_at:'2026-10-13T05:00:00Z'}]}]) {
    response={...prediction(),...change};await assert.rejects(createPrediction('0074647403','2026-10-13T00:00:00Z'));
  }
  for(const change of [{station_id:'other'},{pipeline_version:'other'},{actuals:[]},
    {actuals:detail().actuals.map(point=>({...point,actual_sample_count:0}))}]) {
    response={...detail(),...change};await assert.rejects(fetchForecastDetail(prediction()));
  }
});

test('so sánh đúng timestamp, sai lệch có dấu; không ghép giờ lân cận hoặc nối qua null', () => {
  const rows=buildComparisonRows(prediction().predictions,detail().actuals);
  assert.deepEqual(rows.map(row=>[row.actual,row.difference,row.absoluteError]),[[0,0,0],[-30,10,10]]);
  assert.equal(buildComparisonRows(prediction().predictions,[])[0].difference,null);
  assert.equal(splitActualSegments(detail().actuals).length,2);
  assert.deepEqual(splitActualSegments(detail().actuals).map(segment=>segment.length),[1,1]);
  assert.deepEqual(splitActualSegments(detail().actuals.map(point=>({...point,actual_water_level_mm:null}))),[]);
  const [low,high]=getWaterLevelDomain([-30,0]);assert.ok(low < -30 && high>0);
  assert.deepEqual(getWaterLevelDomain([0,0]),[-1,1]);
});

test('lỗi thiếu dữ liệu và quota an toàn, không tự retry POST', async context => {
  let calls=0;
  context.mock.method(globalThis,'fetch',async()=>{calls++;return Response.json({error:{code:'INSUFFICIENT_HISTORY',message:'private'}},{status:400});});
  await assert.rejects(createPrediction('0074647403','2026-10-13T00:00:00Z'),/Chưa đủ lịch sử/);
  assert.equal(calls,1);
});

test('dự báo có overrides gửi đúng giá trị, nhận metadata null và không chấp nhận bản lưu', async context => {
  const overrides={water_level_depth:-25,rain_750062_depth:null};
  let response={...prediction(),forecast_id:null,generated_at:null,pipeline_version:null,overrides_applied:overrides};
  let sent;
  context.mock.method(globalThis,'fetch',async(url,options)=>{sent=JSON.parse(options.body);return Response.json(response);});
  const result=await createPrediction('0074647403','2026-10-13T00:00:00Z',undefined,overrides);
  assert.deepEqual(sent.overrides,overrides);assert.equal(result.forecast_id,null);
  response={...response,forecast_id:prediction().forecast_id};
  await assert.rejects(createPrediction('0074647403','2026-10-13T00:00:00Z',undefined,overrides));
  response={...prediction(),forecast_id:null,generated_at:null,pipeline_version:null,overrides_applied:{water_level_depth:0}};
  await assert.rejects(createPrediction('0074647403','2026-10-13T00:00:00Z',undefined,overrides));
});

test('thực tế cho kịch bản lấy đúng anchor/horizon POST, không gửi overrides hoặc ID lưu', async context => {
  const { fetchScenarioActuals } = await import('../src/api/predictions.js');
  const scenario={...prediction(),forecast_id:null,pipeline_version:null,generated_at:null,overrides_applied:{water_level_depth:900}};
  let body={station_id:scenario.station_id,unit:'mm',anchor_observed_at:scenario.anchor_observed_at,
    window_end:detail().window_end,actuals:detail().actuals};
  let requested;
  context.mock.method(globalThis,'fetch',async(url,options)=>{requested={url:new URL(url,'http://local'),options};return Response.json(body);});
  const result=await fetchScenarioActuals(scenario);
  assert.equal(requested.options.method,'GET');
  assert.equal(requested.url.pathname,'/api/v1/stations/0074647403/forecast-actuals');
  assert.equal(requested.url.searchParams.get('anchor_observed_at'),scenario.anchor_observed_at);
  assert.equal(requested.url.searchParams.get('horizon_minutes'),'180');
  assert.equal(requested.url.searchParams.has('overrides'),false);
  assert.equal(requested.options.body,undefined);
  assert.deepEqual(result.actuals.map(point=>point.actual_water_level_mm),[0,null,-30]);
  for(const change of [{station_id:'other'},{unit:'m'},{anchor_observed_at:'2026-10-13T01:00:00Z'},{window_end:'2026-10-13T04:00:00Z'},{actuals:[]}]) {
    const original=body;body={...body,...change};await assert.rejects(fetchScenarioActuals(scenario));body=original;
  }
});
