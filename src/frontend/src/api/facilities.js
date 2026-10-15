import { fetchApiJson } from './client.js';
import { facilityMetricLabels, toFacilityStation } from '../facilities.js';

/**
 * @param {AbortSignal} [signal] Hủy đợt tải metadata.
 * @returns {Promise<Array<object>>} Công trình độc lập; bỏ các điểm sông đã liên kết.
 * @throws {Error} Lỗi API hoặc metadata không hợp lệ.
 */
export async function fetchFacilities(signal) {
  const payload = await fetchApiJson('/api/v1/facilities', signal);
  if (!Array.isArray(payload?.facilities)) throw new Error('Backend không trả danh sách công trình hợp lệ.');
  const facilities = payload.facilities.map(toFacilityStation).filter(Boolean);
  if (new Set(facilities.map((item) => item.facility_id)).size !== facilities.length) {
    throw new Error('Danh sách công trình có mã trùng.');
  }
  return facilities;
}

/**
 * @param {string} facilityId ID công trình, giữ namespace sohoa:.
 * @param {AbortSignal} [signal] Hủy khi đóng box/chuyển lựa chọn.
 * @returns {Promise<object>} Metadata công trình ở cấu trúc dùng chung bản đồ.
 * @throws {Error} Lỗi API hoặc response sai ID/loại.
 */
export async function fetchFacilityDetail(facilityId, signal) {
  const payload = await fetchApiJson(`/api/v1/facilities/${encodeURIComponent(facilityId)}`, signal);
  const facility = toFacilityStation(payload);
  if (facility?.facility_id !== facilityId) throw new Error('Chi tiết công trình không khớp lựa chọn.');
  return facility;
}

/**
 * Đọc báo cáo vận hành mới nhất, giữ số 0/âm/thiếu và thời điểm đo gốc.
 * @param {string} facilityId Mã công trình chính xác từ metadata.
 * @param {AbortSignal} [signal] Hủy request khi caller không còn dùng kết quả.
 * @returns {Promise<object|null>} Số đo gốc và freshness; null nếu chưa có báo cáo hợp lệ.
 * @throws {Error} Lỗi API hoặc báo cáo sai ID/giờ/trạng thái/số đo.
 */
export async function fetchLatestFacilityObservation(facilityId, signal) {
  let payload;
  try {
    payload = await fetchApiJson(`/api/v1/facilities/${encodeURIComponent(facilityId)}/latest`, signal);
  } catch (error) {
    if (error.status === 404 && error.code === 'OPERATION_NOT_FOUND') return null;
    throw error;
  }
  const operation = payload?.operation;
  const validMetrics = operation && Object.keys(facilityMetricLabels).every((key) => {
    const metric = operation.measurements?.[key];
    return metric && (metric.value === null || Number.isFinite(metric.value))
      && (metric.unit === null || typeof metric.unit === 'string')
      && typeof metric.unit_confirmed === 'boolean';
  });
  if (payload?.facility_id !== facilityId || operation?.facility_id !== facilityId
    || !validMetrics || typeof operation.observed_at !== 'string'
    || !Number.isFinite(Date.parse(operation.observed_at))
    || operation.time_status !== 'valid' || operation.is_deleted !== false || operation.is_visible !== true
    || typeof payload.is_stale !== 'boolean'
    || !['gate_opening_text', 'operation_start_text', 'operation_end_text'].every((key) => (
      operation[key] === null || typeof operation[key] === 'string'
    ))) {
    throw new Error('Báo cáo vận hành từ backend không hợp lệ.');
  }
  return { ...operation, is_stale: payload.is_stale, data_age_minutes: payload.data_age_minutes };
}
