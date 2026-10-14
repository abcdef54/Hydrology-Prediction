/**
 * @param {object[]} predictions Các điểm dự báo thật, không thêm điểm giữa horizon.
 * @param {object[]} actuals Các mốc thực tế theo giờ, giữ null cho giờ thiếu.
 * @returns {object[]} Hợp các timestamp tăng dần, giá trị cùng mốc được ghép chính xác.
 */
export function buildForecastTooltipRows(predictions, actuals) {
  const rows = new Map();
  for (const point of actuals) {
    rows.set(Date.parse(point.timestamp), {
      timestamp: point.timestamp, predicted: null, actual: point.actual_water_level_mm,
      hasForecast: false,
    });
  }
  for (const point of predictions) {
    const time = Date.parse(point.forecast_at);
    rows.set(time, {
      ...(rows.get(time) ?? { actual: null }), timestamp: point.forecast_at,
      predicted: point.water_level_mm, hasForecast: true,
    });
  }
  return [...rows.entries()].sort(([first], [second]) => first - second).map(([, row]) => row);
}

/**
 * @param {object[]} rows Các mốc tăng dần theo thời gian.
 * @param {number} chartX Tọa độ ngang trong viewBox SVG, đã đổi từ tọa độ màn hình.
 * @param {function} getX Hàm timestamp -> tọa độ trên biểu đồ.
 * @returns {number} Index mốc gần nhất; -1 nếu rỗng. Không tạo giá trị nội suy.
 */
export function findNearestTooltipIndex(rows, chartX, getX) {
  if (!rows.length) return -1;
  let low = 0;
  let high = rows.length - 1;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (getX(rows[middle].timestamp) < chartX) low = middle + 1;
    else high = middle;
  }
  if (low > 0 && chartX - getX(rows[low - 1].timestamp) <= getX(rows[low].timestamp) - chartX) return low - 1;
  return low;
}
