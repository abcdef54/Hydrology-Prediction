import assert from 'node:assert/strict';
import { setTimeout } from 'node:timers/promises';
import test from 'node:test';
import { fetchLatestObservation, fetchStationReadings } from '../src/api/stations.js';
import { formatObservationTime, getLatestDataTime } from '../src/stations.js';

test('header ưu tiên nước và chỉ fallback mưa khi nước chưa có quan trắc', () => {
  const stations = [
    { station_id: 'water-a', station_type: 'water_level' },
    { station_id: 'water-b', station_type: 'water_level' },
    { station_id: 'rain', station_type: 'rainfall' },
  ];
  const readings = {
    'water-a': { observation: { observed_at: '2026-10-13T13:20:00Z' } },
    'water-b': { observation: { observed_at: '2026-10-13T14:20:00Z' } },
    rain: { observation: { observed_at: '2026-10-13T14:30:00Z' } },
  };
  assert.equal(getLatestDataTime(stations, readings), '2026-10-13T14:20:00Z');
  delete readings['water-a'];
  delete readings['water-b'];
  assert.equal(getLatestDataTime(stations, readings), '2026-10-13T14:30:00Z');
  assert.equal(getLatestDataTime(stations, {}), null);
  assert.match(formatObservationTime('2026-10-13T14:20:00Z'), /21:20/);
});

test('latest giữ 0/âm/đơn vị chưa biết, cô lập lỗi và giới hạn đồng thời', async (context) => {
  let active = 0;
  let maximumActive = 0;
  context.mock.method(globalThis, 'fetch', async (url) => {
    active++;
    maximumActive = Math.max(maximumActive, active);
    await setTimeout(2);
    active--;
    const stationId = url.split('/').at(-2);
    if (stationId === 'missing') {
      return Response.json({ error: { code: 'OBSERVATION_NOT_FOUND' } }, { status: 404 });
    }
    if (stationId === 'failed') return Response.json({}, { status: 503 });
    return Response.json({
      station_id: stationId, observed_at: '2026-10-13T14:20:00Z',
      value: stationId === 'water' ? -25.5 : 0,
      unit: stationId === 'future' ? null : 'mm',
    });
  });
  const stations = ['water', 'rain', 'future', 'missing', 'failed', 'another']
    .map((station_id) => ({ station_id }));
  const readings = await fetchStationReadings(stations, new AbortController().signal);
  assert.equal(readings.water.observation.value, -25.5);
  assert.equal(readings.rain.observation.value, 0);
  assert.equal(readings.rain.observation.unit, 'mm');
  assert.equal(readings.future.observation.unit, null);
  assert.equal(readings.missing.status, 'missing');
  assert.equal(readings.failed.status, 'error');
  assert.equal(readings.another.status, 'ready');
  assert.equal(maximumActive, 4);
});

test('hủy đợt tải không tiếp tục gọi API cho trạm trong hàng đợi', async (context) => {
  const controller = new AbortController();
  const fetchMock = context.mock.method(globalThis, 'fetch', async (_url, options) => {
    await setTimeout(100, undefined, { signal: options.signal });
    return Response.json({});
  });
  const stations = Array.from({ length: 10 }, (_, index) => ({ station_id: String(index) }));
  const loading = fetchStationReadings(stations, controller.signal);
  controller.abort();
  await assert.rejects(loading, { name: 'AbortError' });
  assert.equal(fetchMock.mock.callCount(), 4);
});

test('gió giữ số raw, 0 và ws thiếu khi các trường khác vẫn có dữ liệu', async (context) => {
  const measurements = {
    ws: { value: null, unit: null, unit_confirmed: false },
    wd: { value: 961, unit: null, unit_confirmed: false },
    wsg: { value: 0, unit: null, unit_confirmed: false },
    wdg: { value: 1292, unit: null, unit_confirmed: false },
  };
  const payload = {
    station_id: 'wind:228005', station_type: 'wind', value: null, unit: null,
    observed_at: '2026-10-14T12:40:00Z', measurements,
  };
  context.mock.method(globalThis, 'fetch', async () => Response.json(payload));
  const reading = await fetchLatestObservation('wind:228005');
  assert.equal(reading.value, null);
  assert.equal(reading.measurements.wd.value, 961);
  assert.equal(reading.measurements.wsg.value, 0);
  measurements.wd.value = '961';
  await assert.rejects(fetchLatestObservation('wind:228005'), /không hợp lệ/);
  measurements.wd.value = 961;
  delete measurements.wdg;
  await assert.rejects(fetchLatestObservation('wind:228005'), /không hợp lệ/);
});
