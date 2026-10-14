import assert from 'node:assert/strict';
import test from 'node:test';
import { parseVietnamHour,getPredictionRangeError } from '../src/predictionRange.js';
import { fetchRangeChunk,fetchPredictionOptions } from '../src/api/predictionRange.js';

test('khoảng theo giờ Việt Nam, ngày thật và giới hạn 31 ngày',()=>{
  assert.equal(parseVietnamHour('2026-10-13T10:00'),Date.parse('2026-10-13T03:00:00Z'));
  for(const value of ['2026-02-30T00:00','2026-10-13T10:30',''])assert.ok(Number.isNaN(parseVietnamHour(value)));
  assert.equal(getPredictionRangeError('2026-10-01T00:00','2026-11-01T00:00'),'');
  assert.ok(getPredictionRangeError('2026-10-01T00:00','2026-11-02T00:00'));
  assert.ok(getPredictionRangeError('2026-10-02T00:00','2026-10-01T00:00'));
});

test('nhóm range giữ identity và xác thực 0/âm/thiếu thay vì suy diễn điểm',async context=>{
  const request={station_id:'0074647403',start:'2026-10-13T01:00:00Z',end:'2026-10-13T03:00:00Z',horizon_minutes:60,model_version:'m',pipeline_version:'p'};
  let data={...request,unit:'mm',created_count:0,reused_count:1,skipped_count:1,points:[
    {forecast_at:request.start,anchor_observed_at:'2026-10-13T00:00:00Z',status:'ready',forecast_id:'12345678-1234-1234-1234-123456789abc',predicted_water_level_mm:0,actual_water_level_mm:-20,actual_sample_count:6,reused:true},
    {forecast_at:'2026-10-13T02:00:00Z',status:'anchor_unavailable',forecast_id:null,anchor_observed_at:null,predicted_water_level_mm:null,actual_water_level_mm:null,actual_sample_count:0,reused:null},
  ]};
  let sent;
  context.mock.method(globalThis,'fetch',async(url,options)=>{sent=JSON.parse(options.body);return Response.json(data);});
  assert.equal((await fetchRangeChunk(request)).points[0].predicted_water_level_mm,0);
  assert.deepEqual(sent,request);assert.equal(sent.overrides,undefined);
  for(const change of [{pipeline_version:'other'},{points:[]},{reused_count:2},{unit:'m'}]){
    const previous=data;data={...data,...change};await assert.rejects(fetchRangeChunk(request));data=previous;
  }
});

test('horizon lấy từ API metadata, không hardcode và không chuẩn bị feature',async context=>{
  let path;
  context.mock.method(globalThis,'fetch',async url=>{path=url;return Response.json({station_id:'0074647403',model_version:'m',forecast_horizons_minutes:[90,120]});});
  assert.deepEqual((await fetchPredictionOptions('0074647403')).forecast_horizons_minutes,[90,120]);
  assert.ok(path.endsWith('/prediction-options'));
});
