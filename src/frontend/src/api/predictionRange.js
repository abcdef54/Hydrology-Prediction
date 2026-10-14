import { fetchApiJson } from './client.js';

/**
 * @param {string} stationId Trạm nước được phép dự báo.
 * @param {AbortSignal} signal Hủy khi đổi trạm/chọn một mốc.
 * @returns {Promise<object>} Horizon từ registry, không tải input hoặc chạy inference.
 * @throws {Error} Lỗi API hoặc metadata không hợp lệ.
 */
export async function fetchPredictionOptions(stationId, signal) {
  const data = await fetchApiJson(`/api/v1/stations/${encodeURIComponent(stationId)}/prediction-options`, signal);
  if (data?.station_id !== stationId || typeof data.model_version !== 'string'
    || !Array.isArray(data.forecast_horizons_minutes) || !data.forecast_horizons_minutes.length
    || !data.forecast_horizons_minutes.every(value => Number.isInteger(value) && value > 0)
    || new Set(data.forecast_horizons_minutes).size !== data.forecast_horizons_minutes.length) {
    throw new Error('Không đọc được cấu hình dự báo của trạm.');
  }
  return data;
}

/**
 * @param {object} request Một nhóm tối đa 24 giờ dự báo, có identity nếu nối nhóm trước.
 * @param {AbortSignal} signal Hủy chờ; nhóm đã chạy có thể vẫn hoàn tất và lưu trên server.
 * @returns {Promise<object>} Các điểm đúng mốc giờ cùng thực tế và thống kê tái sử dụng.
 * @throws {Error} Lỗi/quota/pipeline đổi hoặc response không khớp; không gửi overrides.
 */
export async function fetchRangeChunk(request, signal) {
  const data = await fetchApiJson('/api/v1/predictions/range', signal, {method:'POST', json:request, timeoutMs:180_000});
  const start=Date.parse(request.start), end=Date.parse(request.end);
  const count=(end-start)/3600000;
  if (data?.station_id !== request.station_id || data.unit !== 'mm'
    || Date.parse(data.start)!==start || Date.parse(data.end)!==end || data.horizon_minutes!==request.horizon_minutes
    || typeof data.model_version!=='string' || typeof data.pipeline_version!=='string'
    || request.model_version && (data.model_version!==request.model_version || data.pipeline_version!==request.pipeline_version)
    || !Array.isArray(data.points) || data.points.length!==count || count<1 || count>24) {
    throw new Error('Kết quả theo khoảng không khớp yêu cầu.');
  }
  for (let index=0;index<data.points.length;index++) {
    const point=data.points[index];
    if (!point || Date.parse(point.forecast_at)!==start+index*3600000
      || !['ready','insufficient_history','anchor_unavailable','future_anchor'].includes(point.status)
      || !Number.isInteger(point.actual_sample_count) || point.actual_sample_count<0
      || (point.actual_water_level_mm===null ? point.actual_sample_count!==0
        : !Number.isFinite(point.actual_water_level_mm) || point.actual_sample_count===0)
      || (point.status==='ready' ? !Number.isFinite(point.predicted_water_level_mm)
        || !/^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$/i.test(point.forecast_id)
        || Date.parse(point.anchor_observed_at)!==Date.parse(point.forecast_at)-request.horizon_minutes*60000
        || typeof point.reused!=='boolean'
        : point.predicted_water_level_mm!==null || point.forecast_id!==null)) {
      throw new Error('Dữ liệu theo giờ trong khoảng không hợp lệ.');
    }
  }
  const created=data.points.filter(point=>point.status==='ready'&&!point.reused).length;
  const reused=data.points.filter(point=>point.status==='ready'&&point.reused).length;
  if (data.created_count!==created || data.reused_count!==reused || data.skipped_count!==count-created-reused) {
    throw new Error('Thống kê dự báo không khớp dữ liệu theo giờ.');
  }
  return data;
}
