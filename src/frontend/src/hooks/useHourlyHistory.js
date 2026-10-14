import { useEffect, useState } from 'react';
import { fetchHourlyHistory } from '../api/history.js';

/**
 * @param {object} request Bộ lọc đã áp dụng, cursor và số lần retry; identity đổi khi tải trang mới.
 * @returns {{status: string, data: object|null, error: string}} Trạng thái của đúng request hiện tại.
 * Request cũ bị hủy và không ghi đè trang mới. Không tự retry hoặc tải hết các trang.
 */
export function useHourlyHistory(request) {
  const [result, setResult] = useState(null);
  useEffect(() => {
    const controller = new AbortController();
    async function load() {
      try {
        const data = await fetchHourlyHistory(request.filters, request.cursor, controller.signal);
        if (!controller.signal.aborted) setResult({ request, status: 'ready', data, error: '' });
      } catch (error) {
        if (!controller.signal.aborted) setResult({ request, status: 'error', data: null, error: error.message });
      }
    }
    load();
    return () => controller.abort();
  }, [request]);
  return result?.request === request ? result : { status: 'loading', data: null, error: '' };
}
