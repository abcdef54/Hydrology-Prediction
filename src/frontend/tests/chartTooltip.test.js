import assert from 'node:assert/strict';
import test from 'node:test';
import { buildForecastTooltipRows, findNearestTooltipIndex } from '../src/chartTooltip.js';

const predictions = [
  { forecast_at: '2026-10-14T01:00:00Z', water_level_mm: 0 },
  { forecast_at: '2026-10-14T03:00:00Z', water_level_mm: -20 },
];
const actuals = [
  { timestamp: '2026-10-14T08:00:00+07:00', actual_water_level_mm: -10 },
  { timestamp: '2026-10-14T02:00:00Z', actual_water_level_mm: 30 },
  { timestamp: '2026-10-14T03:00:00Z', actual_water_level_mm: null },
];

test('ghép tooltip đúng thời điểm, giữ 0/âm/null và không thêm dự báo giữa horizon', () => {
  const rows = buildForecastTooltipRows(predictions, actuals);
  assert.equal(rows.length, 3);
  assert.deepEqual(rows.map(row => [row.predicted, row.actual, row.hasForecast]), [
    [0, -10, true], [null, 30, false], [-20, null, true],
  ]);
  assert.deepEqual(buildForecastTooltipRows(predictions, []).map(row => row.actual), [null, null]);
});

test('chọn mốc gần nhất, clamp hai đầu và hỗ trợ khoảng nhiều điểm', () => {
  const rows = buildForecastTooltipRows(predictions, actuals);
  const getX = timestamp => new Date(timestamp).getUTCHours() * 100;
  assert.equal(findNearestTooltipIndex(rows, -50, getX), 0);
  assert.equal(findNearestTooltipIndex(rows, 149, getX), 0);
  assert.equal(findNearestTooltipIndex(rows, 151, getX), 1);
  assert.equal(findNearestTooltipIndex(rows, 999, getX), 2);
  assert.equal(findNearestTooltipIndex([], 10, getX), -1);
  const many = Array.from({length:744}, (_, index) => ({timestamp:new Date(index*3600000).toISOString()}));
  assert.equal(findNearestTooltipIndex(many, 600.1, timestamp => Date.parse(timestamp)/3600000), 600);
});
