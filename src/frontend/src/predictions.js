export const formatWaterLevel = (value) => new Intl.NumberFormat('vi-VN', { maximumFractionDigits: 2 }).format(value);

/**
 * @param {object[]} predictions Các horizon dự báo đã xác thực, đơn vị mm.
 * @param {object[]} actuals Trung bình giờ; chỉ ghép timestamp trùng tuyệt đối.
 * @returns {object[]} Bảng sai lệch (dự báo - thực tế) và sai số tuyệt đối, null nếu thiếu.
 */
export function buildComparisonRows(predictions, actuals = []) {
  const byTime = new Map(actuals.map((point) => [Date.parse(point.timestamp), point.actual_water_level_mm]));
  return predictions.map((point) => {
    const actual = byTime.get(Date.parse(point.forecast_at)) ?? null;
    const difference = actual === null ? null : point.water_level_mm - actual;
    return { ...point, actual, difference, absoluteError: difference === null ? null : Math.abs(difference) };
  });
}

/**
 * @param {object[]} actuals Chuỗi theo giờ, null là thiếu hoặc chưa có.
 * @returns {object[][]} Các đoạn liên tiếp có dữ liệu; không nối qua null và giữ giá trị 0/âm.
 */
export function splitActualSegments(actuals) {
  const segments = [];
  let segment = [];
  for (const point of actuals) {
    if (point.actual_water_level_mm === null) {
      if (segment.length) segments.push(segment);
      segment = [];
    } else segment.push(point);
  }
  if (segment.length) segments.push(segment);
  return segments;
}

/**
 * @param {number[]} values Các giá trị hữu hạn thực sự được vẽ, tính bằng mm.
 * @returns {[number, number]} Khoảng trục tung có lề, không ép về 0 hoặc cắt giá trị âm.
 */
export function getWaterLevelDomain(values) {
  const low = Math.min(...values);
  const high = Math.max(...values);
  const margin = Math.max((high - low) * 0.12, 1);
  return [low - margin, high + margin];
}
