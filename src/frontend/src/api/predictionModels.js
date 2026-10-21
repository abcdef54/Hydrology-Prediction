import { fetchApiJson } from './client.js';

const modelErrors = {
  MODEL_UNAVAILABLE: 'Mô hình của trạm chưa sẵn sàng hoặc chức năng dự báo chưa được bật.',
  MODEL_NOT_CONFIGURED: 'Trạm chưa có mô hình dự báo riêng.',
  PREDICTION_DISABLED: 'Trạm chưa được bật chức năng dự báo.',
};

/**
 * Đọc danh sách mô hình dự báo của trạm từ backend.
 * @param {string} stationId Mã trạm nước có quyền dự báo.
 * @param {AbortSignal} [signal] Signal để hủy request khi đổi trạm hoặc đóng bảng.
 * @returns {Promise<{station_id: string, default_model_id: string, models: Array<object>}>}
 * @throws {Error} Khi backend trả về lỗi hoặc dữ liệu danh mục không đúng schema.
 */
export async function fetchPredictionModels(stationId, signal) {
  let data;
  try {
    data = await fetchApiJson(`/api/v1/stations/${encodeURIComponent(stationId)}/prediction-models`, signal);
  } catch (error) {
    if (signal?.aborted) throw error;
    if (modelErrors[error.code]) throw new Error(modelErrors[error.code]);
    throw error;
  }

  if (
    data?.station_id !== stationId
    || typeof data.default_model_id !== 'string'
    || !Array.isArray(data.models)
    || data.models.length === 0
  ) {
    throw new Error('Dữ liệu danh mục mô hình từ backend không hợp lệ.');
  }

  const seenIds = new Set();
  let defaultCount = 0;

  for (const model of data.models) {
    if (
      !model
      || typeof model.model_id !== 'string' || !model.model_id
      || typeof model.name !== 'string' || !model.name
      || typeof model.model_version !== 'string'
      || !Array.isArray(model.rain_station_ids)
      || !model.rain_station_ids.every((id) => typeof id === 'string')
      || !Array.isArray(model.forecast_horizons_minutes)
      || !model.forecast_horizons_minutes.length
      || !model.forecast_horizons_minutes.every((m) => Number.isInteger(m) && m > 0)
      || !Number.isInteger(model.feature_count) || model.feature_count <= 0
      || !Number.isInteger(model.sequence_length) || model.sequence_length <= 0
      || typeof model.is_default !== 'boolean'
    ) {
      throw new Error('Thông tin mô hình từ backend không hợp lệ.');
    }

    if (seenIds.has(model.model_id)) {
      throw new Error('Danh sách mô hình chứa mã trùng lặp.');
    }
    seenIds.add(model.model_id);
    if (model.is_default) defaultCount += 1;
  }

  if (!seenIds.has(data.default_model_id) || defaultCount !== 1
    || !data.models.find((model) => model.model_id === data.default_model_id)?.is_default) {
    throw new Error('Mô hình mặc định không hợp lệ trong danh mục của trạm.');
  }

  return data;
}
