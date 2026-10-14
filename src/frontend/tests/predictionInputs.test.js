import assert from 'node:assert/strict';
import test from 'node:test';
import { fetchPredictionInputs } from '../src/api/predictionInputs.js';

function inputResponse() {
  return {
    station_id: '0074647403', model_version: 'test', feature_schema_version: 'v1',
    requested_reference_time: null, hourly_reference_time: '2026-10-14T03:00:00Z',
    anchor_observed_at: '2026-10-14T03:00:00Z', sequence_length: 96, feature_count: 3,
    forecast_horizons_minutes: [60, 180], editable_features: ['water', 'rain', 'lag'],
    missing_features: ['lag'], input_ready: true, is_stale: false,
    features: [
      { name: 'water', value: -20, unit: 'mm', source_station_id: '0074647403', editable: true },
      { name: 'rain', value: 0, unit: 'mm', source_station_id: '750003', editable: true },
      { name: 'lag', value: null, unit: null, source_station_id: null, editable: false },
    ],
  };
}

test('auto-fill giữ mã, timestamp timezone, 0/âm/null và metadata thay vì hardcode số feature', async (context) => {
  let url;
  const payload = inputResponse();
  context.mock.method(globalThis, 'fetch', async (path) => { url = new URL(path, 'http://localhost'); return Response.json(payload); });
  const reference = '2026-10-14T10:00:00+07:00';
  await fetchPredictionInputs('0074647403', reference);
  assert.equal(url.pathname, '/api/v1/stations/0074647403/prediction-inputs');
  assert.equal(url.searchParams.get('reference_time'), reference);
  const inputs = await fetchPredictionInputs('0074647403', null);
  assert.equal(url.search, '');
  assert.deepEqual(inputs.features.map((feature) => feature.value), [-20, 0, null]);
  assert.equal(inputs.input_ready, true);
});

test('từ chối response sai trạm, feature không hợp lệ và metadata thiếu', async (context) => {
  let payload;
  context.mock.method(globalThis, 'fetch', async () => Response.json(payload));
  for (const change of [
    { station_id: 'other' }, { feature_count: 29 }, { anchor_observed_at: 'bad' },
    { features: [null, null, null] }, { features: inputResponse().features.map((feature) => ({ ...feature, value: '0' })) },
    { input_ready: undefined }, { forecast_horizons_minutes: [] },
  ]) {
    payload = { ...inputResponse(), ...change };
    await assert.rejects(fetchPredictionInputs('0074647403', null), /không hợp lệ/);
  }
});

test('thiếu lịch sử báo rõ; lỗi backend không lộ nội dung riêng tư', async (context) => {
  let code = 'INSUFFICIENT_HISTORY';
  context.mock.method(globalThis, 'fetch', async () => Response.json({ error: { code, message: 'private detail' } }, { status: 400 }));
  await assert.rejects(fetchPredictionInputs('0074647403', null), /Chưa đủ lịch sử/);
  code = 'UNKNOWN';
  await assert.rejects(fetchPredictionInputs('0074647403', null), (error) => !error.message.includes('private detail'));
});

test('nhãn dễ hiểu dùng tên trạm, thời gian và chữ thay cờ số', async () => {
  const { describePredictionFeature } = await import('../src/predictionInputs.js');
  const stations = [{ station_id: '750062', name: 'Hồng Thái' }];
  const describe = (name, value = 0) => describePredictionFeature({ name, value, unit: 'mm', source_station_id: '750062' }, stations, 0);
  assert.equal(describe('water_level_lag_60m').label, 'Mực nước 1 giờ trước');
  assert.equal(describe('water_level_lag_180m').label, 'Mực nước 3 giờ trước');
  assert.equal(describe('rain_750062_depth').label, 'Mưa đầu vào');
  assert.equal(describe('rain_750062_depth').sourceName, 'Hồng Thái');
  assert.equal(describe('rain_750062_sum_3h').label, 'Tổng số đo mưa 3 giờ');
  assert.equal(describe('rain_750062_missing', 1).value, 'Thiếu dữ liệu');
  assert.equal(describe('water_level_missing', 0).value, 'Có dữ liệu');
  assert.equal(describe('flood_season', 1).value, 'Có');
  assert.equal(describe('water_level_depth', -25).value, -25);
  assert.equal(describe('water_level_depth', null).value, '');
  assert.equal(describe('unknown_technical_name').label, 'Thông tin bổ sung 1');
});
