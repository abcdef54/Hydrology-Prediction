import { useEffect, useRef, useState } from 'react';
import { createPrediction, fetchForecastDetail } from '../api/predictions.js';

/**
 * @returns {object} state, run(selection, inputs), retryActuals(), cancel().
 * POST chỉ chạy từ thao tác người dùng, khóa click liên tiếp; GET thực tế có retry riêng.
 * Hủy/đóng ngăn response cũ cập nhật UI; hủy chờ không đảm bảo dừng inference trên server.
 */
export function usePrediction() {
  const [state, setState] = useState(null);
  const active = useRef(null);
  useEffect(() => () => active.current?.abort(), []);

  async function loadActuals(prediction, controller) {
    try {
      const detail = await fetchForecastDetail(prediction, controller.signal);
      if (!controller.signal.aborted) setState((current) => ({ ...current, detail, actualStatus: 'ready', actualError: '' }));
    } catch (error) {
      if (!controller.signal.aborted) setState((current) => ({ ...current, actualStatus: 'error', actualError: error.message }));
    }
  }

  async function run(selection, inputs) {
    if ((active.current && !active.current.signal.aborted) || !inputs?.input_ready || !selection) return;
    const controller = new AbortController();
    active.current = controller;
    const referenceTime = selection.referenceTime ?? inputs.hourly_reference_time;
    setState({ status: 'loading', selection, referenceTime, inputs, prediction: null, detail: null });
    try {
      const prediction = await createPrediction(selection.station.station_id, referenceTime, controller.signal);
      if (controller.signal.aborted) return;
      setState({ status: 'ready', selection, referenceTime, inputs, prediction, detail: null, actualStatus: 'loading' });
      await loadActuals(prediction, controller);
    } catch (error) {
      if (!controller.signal.aborted) setState({ status: 'error', selection, referenceTime, inputs, error: error.message });
    } finally {
      if (active.current === controller) active.current = null;
    }
  }

  async function retryActuals() {
    if (active.current || !state?.prediction) return;
    const controller = new AbortController();
    active.current = controller;
    setState((current) => ({ ...current, actualStatus: 'loading', actualError: '' }));
    await loadActuals(state.prediction, controller);
    if (active.current === controller) active.current = null;
  }

  function cancel() {
    active.current?.abort();
    active.current = null;
    setState(null);
  }
  return { state, run, retryActuals, cancel };
}
