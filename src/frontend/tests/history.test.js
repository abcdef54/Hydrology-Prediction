import assert from 'node:assert/strict';
import test from 'node:test';
import { fetchHourlyHistory } from '../src/api/history.js';
import { getHistoryFilterError, getHistoryHours, getVietnamDate, normalizeHistoryFilters } from '../src/history.js';

const filters = { startDate: '2026-10-14', endDate: '2026-10-14', stationType: 'all', stationId: '' };

function historyPage() {
  return {
    timezone: 'Asia/Ho_Chi_Minh', start_date: filters.startDate, end_date: filters.endDate,
    as_of: '2026-10-14T02:10:00Z', has_more: false, next_cursor: null,
    rows: [{
      date: '2026-10-14', station_id: '0074647403', station_type: 'water_level', name: 'Dã Viên', unit: 'mm',
      cells: Array.from({ length: 24 }, (_, hour) => ({
        hour, at: new Date(Date.parse('2026-10-13T17:00:00Z') + hour * 3600000).toISOString(),
        value: hour === 0 ? -25.5 : hour === 1 ? 0 : null,
        status: hour < 2 ? 'observed' : hour <= 9 ? 'missing' : 'future',
      })),
    }],
  };
}

test('ngày/giờ theo Việt Nam ở ranh giới UTC; ngày cũ/khoảng ngày có đủ 24 giờ', () => {
  assert.equal(getVietnamDate('2026-10-13T18:30:00Z'), '2026-10-14');
  assert.equal(getVietnamDate('2026-10-13T16:59:59Z'), '2026-10-13');
  const page = historyPage();
  assert.deepEqual(getHistoryHours(page), Array.from({ length: 10 }, (_, hour) => hour));
  assert.equal(getHistoryHours({ ...page, start_date: '2026-10-13', end_date: '2026-10-13' }).length, 24);
  assert.equal(getHistoryHours({ ...page, start_date: '2026-10-01' }).length, 24);
});

test('ngày không hợp lệ/tương lai/đảo khoảng bị chặn; một ngày không dùng ngày cuối cũ', () => {
  const draft = { ...filters, mode: 'day', endDate: '1900-01-01' };
  assert.equal(getHistoryFilterError(draft, '2026-10-14'), '');
  assert.equal(normalizeHistoryFilters(draft).endDate, '2026-10-14');
  for (const change of [
    { startDate: '' }, { startDate: '2026-02-30' }, { startDate: '2025-02-29' },
    { startDate: '2026-10-15' }, { startDate: '1899-12-31' },
    { mode: 'range', endDate: '2026-10-13' },
  ]) assert.notEqual(getHistoryFilterError({ ...draft, ...change }, '2026-10-14'), '');
  assert.equal(getHistoryFilterError({ ...draft, startDate: '2024-02-29' }, '2026-10-14'), '');
});

test('API giữ 0/âm/null, mã trạm có số 0 đầu và khoảng ngày khi phân trang', async (context) => {
  let requestedUrl;
  const page = historyPage();
  page.has_more = true;
  page.next_cursor = { date: '2026-10-14', station_id: '0074647403' };
  context.mock.method(globalThis, 'fetch', async (url) => { requestedUrl = url; return Response.json(page); });
  const cursor = { date: '2026-10-14', station_id: '0000000001' };
  const response = await fetchHourlyHistory({ ...filters, stationType: 'water_level', stationId: '0074647403' }, cursor);
  const parameters = new URL(requestedUrl, 'http://localhost').searchParams;
  assert.equal(parameters.get('station_id'), '0074647403');
  assert.equal(parameters.get('station_type'), 'water_level');
  assert.equal(parameters.get('start_date'), '2026-10-14');
  assert.equal(parameters.get('end_date'), '2026-10-14');
  assert.equal(parameters.get('cursor_date'), cursor.date);
  assert.equal(parameters.get('cursor_station_id'), cursor.station_id);
  assert.deepEqual(response.rows[0].cells.map((cell) => cell.value).slice(0, 3), [-25.5, 0, null]);
});

test('sai trạng thái/số ô hoặc cursor lặp không được hiển thị như dữ liệu thật', async (context) => {
  let page;
  context.mock.method(globalThis, 'fetch', async () => Response.json(page));
  for (const mutate of [
    (data) => { data.rows[0].cells[0].value = null; },
    (data) => { data.rows[0].cells[2].value = 0; },
    (data) => { data.rows[0].cells.pop(); },
    (data) => { data.rows = [null]; },
    (data) => { data.has_more = true; data.next_cursor = null; },
  ]) {
    page = historyPage(); mutate(page);
    await assert.rejects(fetchHourlyHistory(filters, null), /không đúng cấu trúc/);
  }
  page = historyPage(); page.has_more = true;
  page.next_cursor = { date: page.rows[0].date, station_id: page.rows[0].station_id };
  await assert.rejects(fetchHourlyHistory(filters, page.next_cursor), /không đúng cấu trúc/);
});
