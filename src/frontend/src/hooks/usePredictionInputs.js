import { useEffect, useState } from 'react';
import { fetchPredictionInputs } from '../api/predictionInputs.js';

/**
 * @param {object|null} selection Trạm, referenceTime hoặc null (latest); identity đổi để thử lại.
 * @returns {{status: string, data: object|null, error: string}} Chỉ kết quả của lựa chọn hiện tại.
 * Hủy request cũ, không polling và không chạy inference.
 */
export function usePredictionInputs(selection) {
  const [result, setResult] = useState(null);
  useEffect(() => {
    if (!selection) return;
    const controller = new AbortController();
    async function load() {
      try {
        const data = await fetchPredictionInputs(
          selection.station.station_id,
          selection.referenceTime,
          controller.signal,
          selection.modelId ?? null
        );
        if (!controller.signal.aborted) setResult({ selection, status: 'ready', data, error: '' });
      } catch (error) {
        if (!controller.signal.aborted) setResult({ selection, status: 'error', data: null, error: error.message });
      }
    }
    load();
    return () => controller.abort();
  }, [selection]);
  if (!selection) return { status: 'idle', data: null, error: '' };
  return result?.selection === selection ? result : { status: 'loading', data: null, error: '' };
}
