import { fetchApiJson } from './client.js';

const inputErrors = {
  MODEL_UNAVAILABLE: 'Mô hình của trạm chưa sẵn sàng hoặc chức năng dự báo chưa được bật.',
  INSUFFICIENT_HISTORY: 'Chưa đủ lịch sử hợp lệ để chuẩn bị đầu vào mô hình. Hãy chọn mốc khác.',
  MODEL_NOT_CONFIGURED: 'Trạm chưa có mô hình dự báo riêng.',
  PREDICTION_DISABLED: 'Trạm chưa được bật chức năng dự báo.',
  INVALID_REFERENCE_TIME: 'Mốc yêu cầu chưa hợp lệ hoặc nằm trong tương lai.',
  INPUT_UNAVAILABLE: 'Chưa thể chuẩn bị đầu vào tại mốc này. Hãy thử lại hoặc chọn mốc khác.',
};

/**
 * Đọc feature tại bước cuối; backend giữ và chuẩn bị toàn bộ chuỗi mô hình.
 * @param {string} stationId Mã trạm có quyền dự báo, giữ số 0 đầu.
 * @param {string|null} referenceTime Timestamp có timezone; null để lấy mới nhất.
 * @param {AbortSignal} [signal] Hủy khi đổi mốc/đóng bảng.
 * @returns {Promise<object>} Input gốc, metadata chuỗi, anchor thực tế và trạng thái sẵn sàng.
 * @throws {Error} Lỗi API an toàn hoặc response không hợp lệ; không trả dữ liệu giả.
 * @throws {DOMException} Khi caller hủy request.
 */
export async function fetchPredictionInputs(stationId, referenceTime, signal) {
  const query = new URLSearchParams();
  if (referenceTime) query.set('reference_time', referenceTime);
  let inputs;
  try {
    inputs = await fetchApiJson(`/api/v1/stations/${encodeURIComponent(stationId)}/prediction-inputs${query.size ? `?${query}` : ''}`, signal);
  } catch (error) {
    if (signal?.aborted) throw error;
    if (inputErrors[error.code]) throw new Error(inputErrors[error.code]);
    throw error;
  }
  const validTime = (value) => typeof value === 'string' && Number.isFinite(Date.parse(value));
  const validNames = (value) => Array.isArray(value) && value.every((name) => typeof name === 'string');
  if (
    inputs?.station_id !== stationId
    || typeof inputs.input_ready !== 'boolean' || typeof inputs.is_stale !== 'boolean'
    || !validTime(inputs.anchor_observed_at) || !validTime(inputs.hourly_reference_time)
    || inputs.requested_reference_time !== null && !validTime(inputs.requested_reference_time)
    || !Number.isInteger(inputs.sequence_length) || inputs.sequence_length < 1
    || !Number.isInteger(inputs.feature_count) || inputs.feature_count < 1
    || typeof inputs.model_version !== 'string' || typeof inputs.feature_schema_version !== 'string'
    || !Array.isArray(inputs.features) || inputs.features.length !== inputs.feature_count
    || !validNames(inputs.editable_features) || !validNames(inputs.missing_features)
    || !Array.isArray(inputs.forecast_horizons_minutes) || !inputs.forecast_horizons_minutes.length
    || !inputs.forecast_horizons_minutes.every((minutes) => Number.isInteger(minutes) && minutes > 0)
    || !inputs.features.every((feature) => feature && typeof feature.name === 'string' && feature.name.length
      && (feature.value === null || Number.isFinite(feature.value))
      && (feature.unit === null || typeof feature.unit === 'string')
      && (feature.source_station_id === null || typeof feature.source_station_id === 'string')
      && typeof feature.editable === 'boolean')
    || new Set(inputs.features.map((feature) => feature.name)).size !== inputs.feature_count
  ) throw new Error('Dữ liệu đầu vào từ backend không hợp lệ. Vui lòng thử lại.');
  return inputs;
}
