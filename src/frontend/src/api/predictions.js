import { fetchApiJson } from './client.js';

const validTime = (value) => typeof value === 'string' && Number.isFinite(Date.parse(value));
const validId = (value) => typeof value === 'string' && /^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$/i.test(value);

function validatePredictionPoints(result) {
  if (result?.unit !== 'mm' || !validTime(result.anchor_observed_at)
    || !Array.isArray(result.predictions) || !result.predictions.length
    || typeof result.model_version !== 'string' || typeof result.feature_schema_version !== 'string') throw new Error('Kết quả dự báo từ backend không hợp lệ.');
  let previousHorizon = 0;
  for (const point of result.predictions) {
    if (!point || !Number.isInteger(point.horizon_minutes) || point.horizon_minutes <= previousHorizon
      || !Number.isFinite(point.water_level_mm) || !validTime(point.forecast_at)
      || Date.parse(point.forecast_at) !== Date.parse(result.anchor_observed_at) + point.horizon_minutes * 60_000) {
      throw new Error('Mốc hoặc giá trị dự báo từ backend không hợp lệ.');
    }
    previousHorizon = point.horizon_minutes;
  }
}

/**
 * @param {string} stationId Trạm đã được bật dự báo và có input sẵn sàng.
 * @param {string} referenceTime Mốc có timezone, cố định theo input đang hiển thị, kể cả latest.
 * @param {AbortSignal} signal Hủy chờ khi quay lại/đóng bảng; không tự retry POST.
 * @param {object} [overrides] Giá trị khác gốc ở bước cuối, chỉ gửi trong request, không sửa quan trắc.
 * @param {string|null} [modelId] Mã mô hình dự báo cụ thể nếu chọn mô hình khác mặc định.
 * @param {string|null} [inputModelVersion] Phiên bản mô hình từ input để đối chiếu tính nhất quán.
 * @returns {Promise<object>} Dự báo gốc có ID; dự báo thử có metadata lưu trữ null, đơn vị mm.
 * @throws {Error} Lỗi mạng/HTTP hoặc kết quả không khớp dữ liệu đã gửi.
 */
export async function createPrediction(stationId, referenceTime, signal, overrides = {}, modelId = null, inputModelVersion = null) {
  const isScenario = Object.keys(overrides).length > 0;
  let result;
  try {
    result = await fetchApiJson('/api/v1/predictions', signal, {
      method: 'POST', json: { station_id: stationId, reference_time: referenceTime,
        ...(isScenario ? { overrides } : {}),
        ...(modelId ? { model_id: modelId } : {}) }, timeoutMs: 60_000,
    });
  } catch (error) {
    if (signal?.aborted) throw error;
    const messages = {
      INSUFFICIENT_HISTORY: 'Chưa đủ lịch sử để dự báo tại mốc này. Hãy quay lại và chọn mốc khác.',
      MODEL_UNAVAILABLE: 'Mô hình của trạm chưa sẵn sàng. Vui lòng thử lại sau.',
      PREDICTION_UNAVAILABLE: 'Chưa thể tạo dự báo từ dữ liệu hiện tại. Hãy thử lại hoặc chọn mốc khác.',
      INFERENCE_BUSY: 'Hệ thống đang xử lý một dự báo khác. Vui lòng đợi rồi thử lại.',
      INVALID_REFERENCE_TIME: 'Mốc dự báo không hợp lệ hoặc nằm trong tương lai.',
      INVALID_OVERRIDE: 'Giá trị chỉnh sửa chưa hợp lệ. Hãy quay lại kiểm tra đầu vào.',
    };
    throw new Error(messages[error.code] ?? error.message);
  }
  validatePredictionPoints(result);
  const validStorage = isScenario
    ? result.forecast_id === null && result.generated_at === null && result.pipeline_version === null
    : validId(result.forecast_id) && validTime(result.generated_at) && typeof result.pipeline_version === 'string';
  if (result.station_id !== stationId || !validStorage || typeof result.is_stale !== 'boolean'
    || !result.overrides_applied || Object.keys(result.overrides_applied).length !== Object.keys(overrides).length
    || Object.entries(overrides).some(([name, value]) => result.overrides_applied[name] !== value)
    || (result.model_id !== undefined && (typeof result.model_id !== 'string' || (modelId && result.model_id !== modelId)))
    || (result.model_name !== undefined && typeof result.model_name !== 'string')
    || (inputModelVersion && result.model_version !== inputModelVersion)) {
    throw new Error('Thông tin bản dự báo từ backend không hợp lệ.');
  }
  return result;
}

/**
 * @param {object} prediction Kết quả POST đã xác thực, dùng cố định forecast_id/model/anchor.
 * @param {AbortSignal} signal Hủy tải khi đóng kết quả.
 * @returns {Promise<object>} Dự báo và thực tế trung bình giờ, giữ null cho khoảng thiếu.
 * @throws {Error} Lỗi API hoặc detail không khớp bản dự báo đã chọn; không chạy inference.
 */
export async function fetchForecastDetail(prediction, signal) {
  const detail = await fetchApiJson(`/api/v1/forecasts/${encodeURIComponent(prediction.forecast_id)}`, signal);
  validatePredictionPoints(detail);
  if (['forecast_id', 'station_id', 'model_version', 'feature_schema_version', 'pipeline_version']
    .some((key) => detail[key] !== prediction[key])
    || Date.parse(detail.anchor_observed_at) !== Date.parse(prediction.anchor_observed_at)
    || detail.predictions.length !== prediction.predictions.length
    || detail.predictions.some((point, index) => point.horizon_minutes !== prediction.predictions[index].horizon_minutes
      || point.water_level_mm !== prediction.predictions[index].water_level_mm)) {
    throw new Error('Dữ liệu so sánh không khớp bản dự báo đang xem.');
  }
  validateActualWindow(detail, prediction);
  return detail;
}

function validateActualWindow(detail, prediction) {
  if (detail?.station_id !== prediction.station_id || detail.unit !== 'mm'
    || !validTime(detail.anchor_observed_at)
    || Date.parse(detail.anchor_observed_at) !== Date.parse(prediction.anchor_observed_at)
    || !validTime(detail.window_end)
    || Date.parse(detail.window_end) !== Date.parse(prediction.predictions.at(-1).forecast_at)
    || !Array.isArray(detail.actuals)) throw new Error('Dữ liệu thực tế không khớp trạm hoặc mốc dự báo.');
  const anchor = Date.parse(detail.anchor_observed_at);
  const expectedCount = Math.floor(prediction.predictions.at(-1).horizon_minutes / 60);
  if (detail.actuals.length !== expectedCount || !detail.actuals.every((point, index) => point
    && validTime(point.timestamp) && Date.parse(point.timestamp) === anchor + (index + 1) * 3_600_000
    && Number.isInteger(point.actual_sample_count) && point.actual_sample_count >= 0
    && (point.actual_water_level_mm === null ? point.actual_sample_count === 0
      : Number.isFinite(point.actual_water_level_mm) && point.actual_sample_count > 0))) {
    throw new Error('Dữ liệu thực tế theo giờ từ backend không hợp lệ.');
  }
}

/**
 * @param {object} prediction Response dự báo thử đã xác thực, có anchor và horizon thực tế.
 * @param {AbortSignal} signal Hủy GET khi quay lại/đóng kết quả.
 * @returns {Promise<object>} Thực tế cùng trạm/cửa sổ, không cần ID và không lưu kịch bản.
 * @throws {Error} Lỗi API hoặc dữ liệu không khớp đơn vị/anchor/cửa sổ đã chọn.
 */
export async function fetchScenarioActuals(prediction, signal) {
  const query = new URLSearchParams({
    anchor_observed_at: prediction.anchor_observed_at,
    horizon_minutes: String(prediction.predictions.at(-1).horizon_minutes),
  });
  const detail = await fetchApiJson(`/api/v1/stations/${encodeURIComponent(prediction.station_id)}/forecast-actuals?${query}`, signal);
  validateActualWindow(detail, prediction);
  return detail;
}
