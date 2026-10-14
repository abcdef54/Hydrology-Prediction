const calendarFormatter = new Intl.DateTimeFormat('en', {
  timeZone: 'Asia/Ho_Chi_Minh', year: 'numeric', month: '2-digit', day: '2-digit',
});

/**
 * @param {Date|string} [instant] Thời điểm tuyệt đối; mặc định hiện tại.
 * @returns {string} Ngày Việt Nam YYYY-MM-DD, không phụ thuộc timezone trình duyệt.
 */
export function getVietnamDate(instant = new Date()) {
  const parts = Object.fromEntries(calendarFormatter.formatToParts(new Date(instant))
    .map(({ type, value }) => [type, value]));
  return `${parts.year}-${parts.month}-${parts.day}`;
}

/**
 * @param {object} [station] Metadata trạm cần lọc sẵn; bỏ qua để chọn tất cả.
 * @returns {object} Bộ lọc mới mặc định hôm nay theo lịch Việt Nam.
 */
export function createHistoryFilters(station) {
  const today = getVietnamDate();
  return {
    stationType: station?.station_type ?? 'all', stationId: station?.station_id ?? '',
    mode: 'day', startDate: today, endDate: today,
  };
}

function isCalendarDate(value) {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
  const date = new Date(`${value}T00:00:00Z`);
  return Number.isFinite(date.getTime()) && date.toISOString().slice(0, 10) === value;
}

/**
 * @param {object} filters Bộ lọc đang sửa; day chỉ dùng startDate.
 * @param {string} [today] Ngày Việt Nam hiện tại, có thể truyền trong kiểm thử.
 * @returns {string} Lỗi ngày để hiển thị; chuỗi rỗng nếu hợp lệ.
 */
export function getHistoryFilterError(filters, today = getVietnamDate()) {
  const end = filters.mode === 'range' ? filters.endDate : filters.startDate;
  if (!isCalendarDate(filters.startDate) || !isCalendarDate(end)) return 'Vui lòng chọn ngày hợp lệ.';
  if (filters.startDate < '1900-01-01' || end < '1900-01-01') return 'Ngày phải từ 01/01/1900 trở đi.';
  if (end < filters.startDate) return 'Ngày kết thúc phải bằng hoặc sau ngày bắt đầu.';
  if (end > today) return 'Không thể xem ngày trong tương lai.';
  return '';
}

/**
 * @param {object} filters Bộ lọc đã kiểm tra ngày; mode day chỉ dùng startDate.
 * @returns {object} Khoảng ngày cố định để giữ nguyên khi phân trang, kể cả qua nửa đêm.
 */
export function normalizeHistoryFilters(filters) {
  return {
    stationType: filters.stationType, stationId: filters.stationId,
    startDate: filters.startDate,
    endDate: filters.mode === 'range' ? filters.endDate : filters.startDate,
  };
}

/**
 * @param {object} data Response API đã xác thực.
 * @returns {number[]} Ngày hôm nay đơn lẻ hiện đến giờ tại as_of; khoảng/past hiện đủ 24 giờ.
 */
export function getHistoryHours(data) {
  let lastHour = 23;
  if (data.start_date === data.end_date && data.start_date === getVietnamDate(data.as_of)) {
    lastHour = Number(new Intl.DateTimeFormat('en-GB', {
      timeZone: 'Asia/Ho_Chi_Minh', hour: '2-digit', hourCycle: 'h23',
    }).format(new Date(data.as_of)));
  }
  return Array.from({ length: lastHour + 1 }, (_, hour) => hour);
}

/**
 * @param {string} date Ngày YYYY-MM-DD đã xác thực.
 * @returns {string} Ngày trình bày dd/mm/yyyy.
 */
export function formatHistoryDate(date) {
  return date.split('-').reverse().join('/');
}
