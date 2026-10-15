import { fetchApiJson } from './client.js';
import { fetchFacilities, fetchLatestFacilityObservation } from './facilities.js';

/**
 * Đọc danh sách trạm được bật và metadata vị trí từ backend.
 *
 * @param {AbortSignal} [signal] Hủy request khi component gọi API unmount.
 * @returns {Promise<Array<object>>} Các trạm đang được bật trong backend.
 * @throws {Error} Khi hết thời gian chờ, mất kết nối, HTTP lỗi hoặc sai cấu trúc JSON.
 * @throws {DOMException} Khi caller hủy request bằng signal.
 */
export async function fetchStations(signal) {
  const payload = await fetchApiJson('/api/v1/stations', signal);
  if (!Array.isArray(payload?.stations)) {
    throw new Error('Dữ liệu từ backend không có danh sách trạm hợp lệ.');
  }

  return payload.stations;
}

/**
 * Tải độc lập hai danh mục để lỗi một nguồn không làm mất bản đồ nguồn còn lại.
 * @param {AbortSignal} [signal] Hủy cả hai request khi caller unmount.
 * @returns {Promise<object>} stations và warnings về các danh mục tải thất bại.
 * @throws {Error} Cả hai danh mục thất bại hoặc mã hiển thị bị trùng.
 * @throws {DOMException} Khi caller hủy đợt tải.
 */
export async function fetchMapStations(signal) {
  const results = await Promise.allSettled([fetchStations(signal), fetchFacilities(signal)]);
  signal?.throwIfAborted();
  if (results.every((result) => result.status === 'rejected')) throw results[0].reason;
  const stations = results.flatMap((result) => result.status === 'fulfilled' ? result.value : []);
  if (new Set(stations.map((station) => station.station_id)).size !== stations.length) {
    throw new Error('Danh sách bản đồ có mã trùng.');
  }
  const warnings = results.flatMap((result, index) => result.status === 'rejected'
    ? [`Không tải được ${index === 0 ? 'danh sách trạm' : 'danh sách công trình'}: ${result.reason.message}`] : []);
  return { stations, warnings };
}

/**
 * Đọc metadata và các ngưỡng báo động mới nhất mỗi lần mở box thông tin.
 *
 * @param {string} stationId Mã trạm, giữ số 0 đầu.
 * @param {AbortSignal} [signal] Hủy request khi đóng box hoặc chuyển trạm.
 * @returns {Promise<object>} Chi tiết trạm; alarm_levels_m dùng mét, null là chưa cấu hình.
 * @throws {Error} Lỗi API hoặc response không khớp mã trạm/yêu cầu metadata.
 */
export async function fetchStationDetail(stationId, signal) {
  const detail = await fetchApiJson(`/api/v1/stations/${encodeURIComponent(stationId)}`, signal);
  if (detail?.station_id !== stationId || !detail.location || !detail.alarm_levels_m) {
    throw new Error('Thông tin chi tiết trạm từ backend không hợp lệ.');
  }
  return detail;
}

/**
 * Đọc lần đo mới nhất, giữ số 0, giá trị âm và đơn vị gốc từ backend.
 *
 * @param {string} stationId Mã trạm, giữ số 0 đầu.
 * @param {AbortSignal} [signal] Hủy request khi unmount.
 * @returns {Promise<object|null>} Quan trắc; null nếu chưa có lần đo hợp lệ.
 * @throws {Error} Lỗi API hoặc sai cấu trúc quan trắc.
 */
export async function fetchLatestObservation(stationId, signal) {
  let observation;
  try {
    observation = await fetchApiJson(`/api/v1/stations/${encodeURIComponent(stationId)}/latest`, signal);
  } catch (error) {
    if (error.status === 404 && error.code === 'OBSERVATION_NOT_FOUND') return null;
    throw error;
  }
  const isWind = observation?.station_type === 'wind';
  const validWind = isWind && ['ws', 'wd', 'wsg', 'wdg'].every((field) => {
    const measurement = observation.measurements?.[field];
    return measurement && (measurement.value === null || Number.isFinite(measurement.value))
      && (measurement.unit === null || typeof measurement.unit === 'string')
      && typeof measurement.unit_confirmed === 'boolean';
  }) && observation.value === observation.measurements.ws.value;
  if (
    observation?.station_id !== stationId
    || !(isWind ? validWind : Number.isFinite(observation.value))
    || !Number.isFinite(Date.parse(observation.observed_at))
    || !(observation.unit === null || typeof observation.unit === 'string')
  ) {
    throw new Error('Dữ liệu quan trắc từ backend không hợp lệ.');
  }
  return observation;
}

/**
 * Đọc số đo với tối đa 4 request đồng thời; lỗi một trạm không ảnh hưởng trạm khác.
 *
 * @param {Array<object>} stations Các trạm đã lấy từ API danh sách.
 * @param {AbortSignal} signal Hủy đợt tải, gồm request đang chạy và đang đợi.
 * @returns {Promise<object>} Bảng theo mã trạm: status, observation và error nếu có.
 * @throws {DOMException} Khi caller hủy đợt tải. Không tự retry khi bị giới hạn.
 */
export async function fetchStationReadings(stations, signal) {
  const readings = {};
  let nextIndex = 0;
  async function readNextStation() {
    while (nextIndex < stations.length) {
      signal.throwIfAborted();
      const station = stations[nextIndex++];
      try {
        const observation = station.facility_id
          ? await fetchLatestFacilityObservation(station.facility_id, signal)
          : await fetchLatestObservation(station.station_id, signal);
        readings[station.station_id] = { status: observation ? 'ready' : 'missing', observation };
      } catch (error) {
        if (signal.aborted) throw error;
        readings[station.station_id] = { status: 'error', observation: null, error: error.message };
      }
    }
  }
  await Promise.all(Array.from({ length: Math.min(4, stations.length) }, readNextStation));
  return readings;
}
