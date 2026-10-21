import { useEffect, useRef, useState } from 'react';
import { createPrediction, fetchForecastDetail, fetchScenarioActuals } from '../api/predictions.js';

/**
 * @returns {object} state, run(selection, inputs, overrides), retryActuals(), cancel().
 * POST chỉ chạy từ thao tác người dùng, khóa click liên tiếp; GET thực tế có retry riêng.
 * Hủy/đóng ngăn response cũ cập nhật UI; hủy chờ không đảm bảo dừng inference trên server.
 */
export function usePrediction() {
  const [state, setState] = useState(null);
  const active = useRef(null);
  useEffect(() => () => active.current?.abort(), []);

  async function loadActuals(prediction, controller) {
    try {
      const detail = prediction.forecast_id
        ? await fetchForecastDetail(prediction, controller.signal)
        : await fetchScenarioActuals(prediction, controller.signal);
      if (!controller.signal.aborted) setState((current) => ({ ...current, detail, actualStatus: 'ready', actualError: '' }));
    } catch (error) {
      if (!controller.signal.aborted) setState((current) => ({ ...current, actualStatus: 'error', actualError: error.message }));
    }
  }

  async function run(selection, inputs, overrides = {}) {
    if ((active.current && !active.current.signal.aborted) || !inputs?.input_ready || !selection) return;
    const controller = new AbortController();
    active.current = controller;
    const referenceTime = selection.referenceTime ?? inputs.hourly_reference_time;
    const modelId = selection.modelId ?? inputs.model_id ?? null;
    setState({ status: 'loading', selection, referenceTime, inputs, overrides, prediction: null, detail: null });
    try {
      const prediction = await createPrediction(
        selection.station.station_id,
        referenceTime,
        controller.signal,
        overrides,
        modelId,
        inputs.model_version ?? null
      );
      if (controller.signal.aborted) return;
      setState({ status: 'ready', selection, referenceTime, inputs, overrides, prediction, detail: null,
        actualStatus: 'loading' });
      await loadActuals(prediction, controller);
    } catch (error) {
      if (!controller.signal.aborted) setState({ status: 'error', selection, referenceTime, inputs, overrides, error: error.message });
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
