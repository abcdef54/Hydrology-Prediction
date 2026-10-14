/**
 * Đọc JSON từ API nội bộ với timeout và lỗi an toàn cho người dùng.
 *
 * @param {string} path Đường dẫn cùng origin, bắt đầu bằng /api/.
 * @param {AbortSignal} [signal] Signal của component gọi API.
 * @param {object} [options] method/json cho POST; timeoutMs mặc định 10 giây.
 * @returns {Promise<object>} Response JSON đã đọc.
 * @throws {Error} Lỗi mạng/timeout/HTTP/JSON; lỗi HTTP kèm status và code nếu có.
 * @throws {DOMException} Khi caller hủy request.
 */
export async function fetchApiJson(path, signal, { method = 'GET', json, timeoutMs = 10_000 } = {}) {
  let response;
  try {
    response = await fetch(path, {
      method,
      headers: { Accept: 'application/json', ...(json === undefined ? {} : { 'Content-Type': 'application/json' }) },
      ...(json === undefined ? {} : { body: JSON.stringify(json) }),
      signal: signal
        ? AbortSignal.any([signal, AbortSignal.timeout(timeoutMs)])
        : AbortSignal.timeout(timeoutMs),
    });
  } catch (error) {
    if (signal?.aborted) throw error;
    throw new Error(error.name === 'TimeoutError'
      ? 'Backend phản hồi quá chậm. Vui lòng thử lại.'
      : 'Không thể kết nối backend. Kiểm tra mạng và thử lại.');
  }

  let payload;
  try {
    payload = await response.json();
  } catch {
    throw new Error(`Backend trả về dữ liệu không phải JSON hợp lệ (HTTP ${response.status}).`);
  }

  if (!response.ok) {
    let message = `Không đọc được dữ liệu (HTTP ${response.status}).`;
    if (response.status === 429) {
      const seconds = Number(response.headers.get('Retry-After'));
      message = Number.isFinite(seconds) && seconds > 0
        ? `Đã vượt giới hạn yêu cầu. Thử lại sau ${Math.ceil(seconds)} giây.`
        : 'Đã vượt giới hạn yêu cầu. Vui lòng đợi rồi thử lại.';
    } else if (response.status === 503) {
      message = 'Dịch vụ dữ liệu tạm thời không khả dụng. Vui lòng thử lại.';
    }
    throw Object.assign(new Error(message), {
      status: response.status,
      code: payload?.error?.code,
      retryAfter: Number(response.headers.get('Retry-After')) || null,
    });
  }
  return payload;
}
