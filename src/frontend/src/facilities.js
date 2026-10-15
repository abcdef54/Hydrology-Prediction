export const facilityMetricLabels = {
  htl: 'Mực nước thượng lưu', hhl: 'Mực nước hạ lưu',
  qden: 'Lưu lượng đến', qdi: 'Lưu lượng ra',
  mucnuocsong: 'Mực nước sông', luuluongdieutiet: 'Lưu lượng điều tiết',
};

export const facilitySpecificationLabels = {
  mndbt: 'Mực nước dâng bình thường', mnc: 'Mực nước chết',
  nguongtran: 'Cao trình ngưỡng tràn', nguongxaday: 'Cao trình ngưỡng xả đáy',
  socua: 'Số cửa', authuyen: 'Số âu thuyền', khaudo: 'Khẩu độ cửa',
  baodongmuc1: 'Báo động I', baodongmuc2: 'Báo động II', baodongmuc3: 'Báo động III',
};

/**
 * Chuyển metadata công trình sang cấu trúc dùng chung của bản đồ/sidebar.
 * @param {object} facility Metadata API; station_id là liên kết trạm sông nếu có.
 * @returns {object|null} Công trình với ID nguyên vẹn; null cho điểm sông đã có marker.
 * @throws {Error} Metadata thiếu/sai kiểu hoặc loại công trình chưa hỗ trợ.
 */
export function toFacilityStation(facility) {
  if (!facility || typeof facility.facility_id !== 'string' || !facility.facility_id.trim()
    || typeof facility.name !== 'string' || !facility.name.trim()
    || !['hydropower', 'reservoir', 'dam', 'river'].includes(facility.kind)
    || !(facility.station_id === null || typeof facility.station_id === 'string')
    || !facility.specifications || typeof facility.specifications !== 'object'
    || Array.isArray(facility.specifications)
    || !Object.values(facility.specifications).every((value) => value === null || typeof value === 'string')
    || !(facility.latitude === null || Number.isFinite(facility.latitude))
    || !(facility.longitude === null || Number.isFinite(facility.longitude))) {
    throw new Error('Thông tin công trình từ backend không hợp lệ.');
  }
  if (facility.station_id !== null || facility.kind === 'river') return null;
  return {
    station_id: facility.facility_id, facility_id: facility.facility_id,
    name: facility.name, station_type: facility.kind, specifications: facility.specifications,
    location: { latitude: facility.latitude, longitude: facility.longitude, address: null },
    measurement_units: facility.measurement_units, measurement_unit: null,
    model_configured: false, prediction_enabled: false,
  };
}
