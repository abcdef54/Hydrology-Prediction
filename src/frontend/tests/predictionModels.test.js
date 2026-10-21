import assert from 'node:assert/strict';
import test from 'node:test';
import { fetchPredictionModels } from '../src/api/predictionModels.js';

function validModelsResponse() {
  return {
    station_id: '0074647403',
    default_model_id: 'legacy',
    models: [
      {
        model_id: 'legacy',
        name: 'Production cũ',
        model_version: 'm_legacy_v1',
        rain_station_ids: ['750062', '750007', '750003', '750011'],
        forecast_horizons_minutes: [60, 180, 360, 720, 1440],
        feature_count: 29,
        sequence_length: 96,
        is_default: true,
      },
      {
        model_id: 'model_b',
        name: 'Model B — Hồng Thái + Nhâm',
        model_version: 'm_b_v1',
        rain_station_ids: ['750062', '750007'],
        forecast_horizons_minutes: [60, 180, 360, 720, 1440],
        feature_count: 26,
        sequence_length: 96,
        is_default: false,
      },
      {
        model_id: 'three_station',
        name: '3 trạm — Hồng Thái + Bạch Mã + Phú Bài',
        model_version: 'm_3st_v1',
        rain_station_ids: ['750062', '750041', '750014'],
        forecast_horizons_minutes: [60, 180, 360, 720, 1440],
        feature_count: 32,
        sequence_length: 96,
        is_default: false,
      },
    ],
  };
}

test('đọc danh mục mô hình hợp lệ, giữ đúng default_model_id và danh sách mô hình', async (context) => {
  let requestedUrl;
  const payload = validModelsResponse();
  context.mock.method(globalThis, 'fetch', async (path) => {
    requestedUrl = path;
    return Response.json(payload);
  });

  const result = await fetchPredictionModels('0074647403');
  assert.equal(requestedUrl, '/api/v1/stations/0074647403/prediction-models');
  assert.equal(result.station_id, '0074647403');
  assert.equal(result.default_model_id, 'legacy');
  assert.equal(result.models.length, 3);
  assert.deepEqual(
    result.models.map((m) => m.model_id),
    ['legacy', 'model_b', 'three_station']
  );
  assert.equal(result.models[0].name, 'Production cũ');
  assert.equal(result.models[1].name, 'Model B — Hồng Thái + Nhâm');
  assert.equal(result.models[2].name, '3 trạm — Hồng Thái + Bạch Mã + Phú Bài');
});

test('từ chối danh mục mô hình sai trạm, thiếu default_model_id hoặc model_id trùng lặp', async (context) => {
  let payload;
  context.mock.method(globalThis, 'fetch', async () => Response.json(payload));

  for (const invalid of [
    { station_id: 'other_station' },
    { default_model_id: 'unknown_model' },
    { models: [] },
    { models: null },
    {
      models: [
        validModelsResponse().models[0],
        { ...validModelsResponse().models[1], model_id: 'legacy' },
      ],
    },
    {
      models: [
        { ...validModelsResponse().models[0], feature_count: 0 },
      ],
    },
    {
      models: [
        { ...validModelsResponse().models[0], sequence_length: -1 },
      ],
    },
    {
      models: [
        { ...validModelsResponse().models[0], forecast_horizons_minutes: [] },
      ],
    },
    {
      models: [
        { ...validModelsResponse().models[0], is_default: 'yes' },
      ],
    },
  ]) {
    payload = { ...validModelsResponse(), ...invalid };
    await assert.rejects(fetchPredictionModels('0074647403'), /không hợp lệ|trùng lặp/);
  }
});

test('báo lỗi an toàn khi trạm chưa có mô hình hoặc chưa bật dự báo', async (context) => {
  context.mock.method(globalThis, 'fetch', async () =>
    Response.json({ error: { code: 'MODEL_UNAVAILABLE', message: 'Internal detail' } }, { status: 404 })
  );
  await assert.rejects(
    fetchPredictionModels('0074647403'),
    /Mô hình của trạm chưa sẵn sàng hoặc chức năng dự báo chưa được bật/
  );
});
