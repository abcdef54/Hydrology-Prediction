import { fetchApiJson } from './client.js';

/**
 * Đọc một trang bảng trạm/ngày, kiểm tra ô thiếu/0 và cursor trước khi hiển thị.
 *
 * @param {object} filters Loại/mã trạm và startDate/endDate Việt Nam đã chuẩn hóa.
 * @param {object|null} cursor Cursor độc quyền do trang trước trả về; null là trang đầu.
 * @param {AbortSignal} [signal] Hủy request khi đóng bảng, đổi lọc hoặc đổi trang.
 * @returns {Promise<object>} Trang tối đa 50 dòng, mỗi dòng có đúng 24 ô mốc giờ.
 * @throws {Error} API lỗi, JSON sai hợp đồng hoặc cursor không tiến tới trang mới.
 * @throws {DOMException} Caller hủy request.
 */
export async function fetchHourlyHistory(filters, cursor, signal) {
  const parameters = new URLSearchParams({
    start_date: filters.startDate, end_date: filters.endDate, limit: '50',
  });
  if (filters.stationType !== 'all') parameters.set('station_type', filters.stationType);
  if (filters.stationId) parameters.set('station_id', filters.stationId);
  if (cursor) {
    parameters.set('cursor_date', cursor.date);
    parameters.set('cursor_station_id', cursor.station_id);
  }
  let data;
  try {
    data = await fetchApiJson(`/api/v1/observations/hourly?${parameters}`, signal);
  } catch (error) {
    if (error.status === 404) {
      throw new Error('Backend chưa có API lịch sử theo giờ. Vui lòng khởi động lại backend sau khi cập nhật code.');
    }
    throw error;
  }
  const invalid = () => new Error('Dữ liệu lịch sử từ backend không đúng cấu trúc yêu cầu.');
  if (
    data?.timezone !== 'Asia/Ho_Chi_Minh' || !Number.isFinite(Date.parse(data.as_of))
    || data.start_date !== filters.startDate || data.end_date !== filters.endDate
    || !Array.isArray(data.rows) || data.rows.length > 50 || typeof data.has_more !== 'boolean'
  ) throw invalid();

  const rowKeys = new Set();
  for (const row of data.rows) {
    if (!row) throw invalid();
    const key = `${row.date}/${row.station_id}`;
    if (
      typeof row.station_id !== 'string' || typeof row.name !== 'string'
      || typeof row.station_type !== 'string' || !(row.unit === null || typeof row.unit === 'string')
      || typeof row.date !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(row.date)
      || row.date < filters.startDate || row.date > filters.endDate
      || rowKeys.has(key) || !Array.isArray(row.cells) || row.cells.length !== 24
    ) throw invalid();
    rowKeys.add(key);
    for (const [hour, cell] of row.cells.entries()) {
      if (
        !cell || cell.hour !== hour || !Number.isFinite(Date.parse(cell.at))
        || !['observed', 'missing', 'future'].includes(cell.status)
        || (cell.status === 'observed' ? !Number.isFinite(cell.value) : cell.value !== null)
      ) throw invalid();
    }
  }
  if (data.has_more) {
    const lastRow = data.rows.at(-1);
    const next = data.next_cursor;
    if (!lastRow || next?.date !== lastRow.date || next?.station_id !== lastRow.station_id
      || (cursor && cursor.date === next.date && cursor.station_id === next.station_id)) throw invalid();
  } else if (data.next_cursor !== null) {
    throw invalid();
  }
  return data;
}
