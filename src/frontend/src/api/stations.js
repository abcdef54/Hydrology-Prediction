/**
 * Đọc danh sách trạm để kiểm tra kết nối frontend → backend → PostgreSQL.
 *
 * @returns {Promise<Array<object>>} Các trạm đang được bật trong backend.
 * @throws {Error} Khi hết thời gian chờ, mất kết nối, HTTP lỗi hoặc sai cấu trúc JSON.
 */
export async function fetchStations() {
  let response;

  try {
    response = await fetch('/api/v1/stations', {
      headers: { Accept: 'application/json' },
      signal: AbortSignal.timeout(10_000),
    });
  } catch (error) {
    if (error.name === 'TimeoutError') {
      throw new Error('Backend phản hồi quá chậm. Vui lòng thử lại.');
    }

    throw new Error('Không thể kết nối. Kiểm tra mạng và thử lại.');
  }

  if (!response.ok) {
    if (response.status === 429) {
      const retryAfterSeconds = Number(response.headers.get('Retry-After'));
      const retryMessage = Number.isFinite(retryAfterSeconds) && retryAfterSeconds > 0
        ? `Vui lòng đợi ${Math.ceil(retryAfterSeconds)} giây rồi thử lại.`
        : 'Vui lòng đợi một lúc rồi thử lại.';
      throw new Error(`Đã vượt giới hạn yêu cầu. ${retryMessage}`);
    }

    throw new Error(
      `Chưa kết nối được backend (HTTP ${response.status}). Kiểm tra backend đang chạy tại cổng 8000 và PostgreSQL khả dụng.`,
    );
  }

  let payload;

  try {
    payload = await response.json();
  } catch {
    throw new Error('Backend trả về dữ liệu không phải JSON hợp lệ.');
  }

  if (!Array.isArray(payload?.stations)) {
    throw new Error('Dữ liệu từ backend không có danh sách trạm hợp lệ.');
  }

  return payload.stations;
}
