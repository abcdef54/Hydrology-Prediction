const appearances = {
  water_level: {
    label: 'Trạm nước', variant: 'water', order: 0,
    paths: ['M3 7c2-2 4 2 6 0s4 2 6 0 4 2 6 0M3 12c2-2 4 2 6 0s4 2 6 0 4 2 6 0M3 17c2-2 4 2 6 0s4 2 6 0 4 2 6 0'],
  },
  rainfall: {
    label: 'Trạm mưa', variant: 'rain', order: 1,
    paths: ['M7 14H6a4 4 0 0 1 0-8 6 6 0 0 1 11-1 4 4 0 0 1 1 8h-1M8 17l-1 3M13 16l-1 3M18 17l-1 3'],
  },
  wind: {
    label: 'Trạm gió', variant: 'other', order: 2,
    paths: ['M3 8h12a3 3 0 1 0-3-3M3 12h16a3 3 0 1 1-3 3M3 16h5a3 3 0 1 1-3 3'],
  },
  reservoir: {
    label: 'Hồ chứa', variant: 'reservoir', order: 3,
    paths: ['M3 5h18l-3 14H6L3 5ZM6 11c2-2 4 2 6 0s4 2 6 0M7 15c2-2 3 2 5 0s3 2 5 0'],
  },
  dam: {
    label: 'Đập', variant: 'dam', order: 4,
    paths: ['M7 3h10l3 18H4L7 3ZM6 9h12M5 15h14M12 3v6M9 9v6M15 9v6M12 15v6'],
  },
  hydropower: {
    label: 'Thủy điện', variant: 'hydropower', order: 5,
    paths: ['M13 2 5 13h6l-1 9 9-13h-6l1-7Z'],
  },
};

/**
 * @param {string} stationType Loại trạm do API cung cấp; không suy ra từ tên trạm.
 * @returns {object} Nhãn/icon dùng chung cho sidebar và marker, kể cả loại mới.
 */
export function getStationAppearance(stationType) {
  return Object.hasOwn(appearances, stationType) ? appearances[stationType] : {
    label: stationType ? `Trạm ${stationType}` : 'Trạm quan trắc',
    variant: 'other', order: 6,
    paths: ['M16 12a4 4 0 1 0-8 0 4 4 0 0 0 8 0M12 3v2M12 19v2M3 12h2M19 12h2'],
  };
}

/**
 * @param {object} station Metadata API; tọa độ theo độ thập phân.
 * @returns {[number, number]|null} [Kinh độ, vĩ độ] hợp lệ hoặc null, không tạo vị trí giả.
 */
export function getStationCoordinates(station) {
  const { latitude, longitude } = station.location ?? {};
  return Number.isFinite(latitude) && latitude >= -90 && latitude <= 90
    && Number.isFinite(longitude) && longitude >= -180 && longitude <= 180
    ? [longitude, latitude] : null;
}

/**
 * Chọn thời điểm mới nhất, ưu tiên trạm nước dù trạm mưa có quan trắc mới hơn.
 *
 * @param {Array<object>} stations Toàn bộ trạm, không phụ thuộc bộ lọc giao diện.
 * @param {object} readings Kết quả tải /latest theo mã trạm.
 * @returns {string|null} observed_at mới nhất của nước; fallback loại khác hoặc null.
 */
export function getLatestDataTime(stations, readings) {
  const available = stations.filter((station) => {
    const timestamp = readings[station.station_id]?.observation?.observed_at;
    return Number.isFinite(Date.parse(timestamp));
  });
  const waterStations = available.filter((station) => station.station_type === 'water_level');
  const candidates = waterStations.length ? waterStations : available;
  return candidates.reduce((latest, station) => {
    const timestamp = readings[station.station_id].observation.observed_at;
    return !latest || Date.parse(timestamp) > Date.parse(latest) ? timestamp : latest;
  }, null);
}

const dateFormatter = new Intl.DateTimeFormat('vi-VN', {
  timeZone: 'Asia/Ho_Chi_Minh', day: '2-digit', month: '2-digit', year: 'numeric',
  hour: '2-digit', minute: '2-digit', hourCycle: 'h23',
});

/**
 * @param {string} timestamp ISO 8601 hợp lệ có timezone, lấy từ observed_at.
 * @returns {string} Giờ Việt Nam, không phụ thuộc timezone trình duyệt.
 */
export function formatObservationTime(timestamp) {
  return dateFormatter.format(new Date(timestamp));
}
