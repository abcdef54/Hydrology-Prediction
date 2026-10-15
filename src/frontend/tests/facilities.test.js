import assert from 'node:assert/strict';
import test from 'node:test';
import { toFacilityStation } from '../src/facilities.js';
import { fetchLatestFacilityObservation } from '../src/api/facilities.js';
import { fetchMapStations, fetchStationReadings } from '../src/api/stations.js';

const facility = {
  facility_id: 'sohoa:1', name: 'Hương Điền', kind: 'hydropower', station_id: null,
  latitude: 16.46, longitude: 107.42, specifications: { mndbt: '+58 m', socua: '0' },
};
function latest() {
  return {
    facility_id: 'sohoa:1', is_stale: false, data_age_minutes: 10,
    operation: {
      facility_id: 'sohoa:1', observed_at: '2026-10-14T13:00:00Z', time_status: 'valid',
      is_visible: true, is_deleted: false, gate_opening_text: '0/15',
      operation_start_text: null, operation_end_text: null,
      measurements: Object.fromEntries(['htl', 'hhl', 'qden', 'qdi', 'mucnuocsong', 'luuluongdieutiet']
        .map((field, index) => [field, { value: index === 0 ? 0 : index === 1 ? -0.03 : null, unit: null, unit_confirmed: false }])),
    },
  };
}

test('công trình giữ ID/thông số gốc, không tạo lại trạm sông hay quyền dự báo', () => {
  const station = toFacilityStation(facility);
  assert.equal(station.station_id, 'sohoa:1');
  assert.equal(station.specifications.mndbt, '+58 m');
  assert.equal(station.specifications.socua, '0');
  assert.equal(station.prediction_enabled, false);
  assert.equal(toFacilityStation({ ...facility, kind: 'river', station_id: '0074647403' }), null);
});

test('latest giữ 0/âm/NULL, chuỗi cửa mở; không nhận bản ghi xóa hoặc giờ lỗi', async (context) => {
  const payload = latest();
  context.mock.method(globalThis, 'fetch', async () => Response.json(payload));
  const observation = await fetchLatestFacilityObservation('sohoa:1');
  assert.equal(observation.measurements.htl.value, 0);
  assert.equal(observation.measurements.hhl.value, -0.03);
  assert.equal(observation.measurements.qden.value, null);
  assert.equal(observation.gate_opening_text, '0/15');
  payload.operation.is_deleted = true;
  await assert.rejects(fetchLatestFacilityObservation('sohoa:1'), /không hợp lệ/);
  payload.operation.is_deleted = false;
  payload.operation.time_status = 'invalid';
  await assert.rejects(fetchLatestFacilityObservation('sohoa:1'), /không hợp lệ/);
});

test('lỗi API công trình không làm mất danh mục trạm nước/mưa/gió', async (context) => {
  context.mock.method(globalThis, 'fetch', async (url) => url === '/api/v1/stations'
    ? Response.json({ stations: [{ station_id: '0074647403', name: 'Dã Viên' }] })
    : Response.json({}, { status: 503 }));
  const result = await fetchMapStations(new AbortController().signal);
  assert.equal(result.stations.length, 1);
  assert.equal(result.stations[0].station_id, '0074647403');
  assert.match(result.warnings[0], /danh sách công trình/);
});

test('đọc số đo công trình dùng đúng endpoint và 404 chưa có báo cáo là missing', async (context) => {
  const fetchMock = context.mock.method(globalThis, 'fetch', async () => Response.json({
    error: { code: 'OPERATION_NOT_FOUND' },
  }, { status: 404 }));
  const readings = await fetchStationReadings([toFacilityStation(facility)], new AbortController().signal);
  assert.equal(readings['sohoa:1'].status, 'missing');
  assert.equal(fetchMock.mock.calls[0].arguments[0], '/api/v1/facilities/sohoa%3A1/latest');
});
