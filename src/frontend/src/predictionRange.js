import { getVietnamDate } from './history.js';

/**
 * @param {string} value Ngày/giờ từ datetime-local, hiểu cố định theo giờ Việt Nam.
 * @returns {number} Epoch ms; NaN cho ngày giả hoặc không đúng mốc giờ.
 */
export function parseVietnamHour(value) {
  if (!/^\d{4}-\d{2}-\d{2}T\d{2}:00$/.test(value)) return NaN;
  const time=Date.parse(`${value}:00+07:00`);
  if (!Number.isFinite(time)) return NaN;
  const local=new Date(time+7*3600000).toISOString().slice(0,16);
  return local===value ? time : NaN;
}

/** @returns {object} Hôm nay 00:00 đến 00:00 hôm sau, theo Việt Nam, end không tính vào khoảng. */
export function defaultPredictionRange() {
  const day=getVietnamDate();
  const next=new Date(Date.parse(`${day}T00:00:00Z`)+86400000).toISOString().slice(0,10);
  return {start:`${day}T00:00`,end:`${next}T00:00`};
}

/**
 * @param {string} start Mốc đầu theo lịch Việt Nam.
 * @param {string} end Mốc cuối không thuộc khoảng, theo lịch Việt Nam.
 * @returns {string} Lỗi hoặc chuỗi rỗng; tối đa 31 ngày để giới hạn công việc.
 */
export function getPredictionRangeError(start,end) {
  const first=parseVietnamHour(start), last=parseVietnamHour(end);
  if (!Number.isFinite(first)||!Number.isFinite(last)) return 'Chọn ngày hợp lệ và giờ tròn (ví dụ 10:00).';
  if (start<'1900-01-01T00:00') return 'Ngày bắt đầu phải từ 01/01/1900.';
  if (first>=last) return 'Thời điểm kết thúc phải sau thời điểm bắt đầu.';
  if (last-first>31*86400000) return 'Mỗi lượt chọn tối đa 31 ngày.';
  return '';
}
