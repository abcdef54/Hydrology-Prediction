import { useEffect, useState } from 'react';
import { fetchPredictionModels } from '../api/predictionModels.js';

/**
 * Đọc danh sách mô hình dự báo của trạm, tự hủy request khi đổi trạm.
 * @param {string|null} stationId Mã trạm nước có quyền dự báo.
 * @returns {{status: 'idle'|'loading'|'ready'|'error', data: object|null, error: string}}
 */
export function usePredictionModels(stationId) {
  const [result, setResult] = useState(null);

  useEffect(() => {
    if (!stationId) return;
    const controller = new AbortController();
    async function load() {
      try {
        const data = await fetchPredictionModels(stationId, controller.signal);
        if (!controller.signal.aborted) {
          setResult({ stationId, status: 'ready', data, error: '' });
        }
      } catch (error) {
        if (!controller.signal.aborted) {
          setResult({ stationId, status: 'error', data: null, error: error.message });
        }
      }
    }
    load();
    return () => controller.abort();
  }, [stationId]);

  if (!stationId) return { status: 'idle', data: null, error: '' };
  return result?.stationId === stationId ? result : { status: 'loading', data: null, error: '' };
}
