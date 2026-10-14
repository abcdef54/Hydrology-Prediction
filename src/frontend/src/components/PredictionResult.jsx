import { useEffect, useRef } from 'react';
import ForecastChart from './ForecastChart.jsx';
import { formatObservationTime } from '../stations.js';
import { buildComparisonRows, formatWaterLevel } from '../predictions.js';

/**
 * @param {object} props state là lượt dự báo hiện tại; onBack giữ nguyên bảng lịch sử;
 * onRetry chỉ chạy lại POST khi người dùng yêu cầu, onRetryActuals chỉ đọc lại GET.
 * @returns {import('react').ReactElement} Kết quả theo một anchor, biểu đồ và bảng sai số mm.
 */
export default function PredictionResult({ state, onBack, onRetry, onRetryActuals }) {
  const headingRef = useRef(null);
  useEffect(() => { headingRef.current?.focus(); }, []);
  const prediction = state?.prediction;
  const actuals = state?.detail?.actuals ?? [];
  const rows = prediction ? buildComparisonRows(prediction.predictions, actuals) : [];
  const comparable = rows.filter((row) => row.actual !== null);
  const meanAbsoluteError = comparable.length
    ? comparable.reduce((sum, row) => sum + row.absoluteError, 0) / comparable.length : null;
  return (
    <section className="prediction-result" aria-labelledby="prediction-result-title">
      <div className="prediction-result-heading">
        <div><h3 id="prediction-result-title" ref={headingRef} tabIndex={-1}>Kết quả dự báo — {state.selection.station.name}</h3>
          <p>Mốc đã chọn: {formatObservationTime(state.referenceTime)}</p></div>
        <button type="button" onClick={onBack}>← Quay lại lịch sử</button>
      </div>
      {state.status === 'loading' && <p className="prediction-result-state" role="status">Đang dự báo và chuẩn bị biểu đồ…</p>}
      {state.status === 'error' && <div className="prediction-result-state" role="alert"><p>{state.error}</p>
        <button type="button" onClick={onRetry}>Thử lại dự báo</button></div>}
      {prediction && <>
        {Object.keys(state.overrides).length > 0 && <p className="prediction-result-meta">
          <strong>Dự báo thử từ đầu vào đã chỉnh sửa.</strong> Giá trị tự nhập và kết quả này không lưu vào database.
          Đường thực tế là số đo quan trắc để đối chiếu với kịch bản này.
        </p>}
        <p className="prediction-result-meta">Mốc đầu vào thực tế: <strong>{formatObservationTime(prediction.anchor_observed_at)}</strong>
          {' · '}Dự báo đến {formatObservationTime(prediction.predictions.at(-1).forecast_at)} · Đơn vị: mm</p>
        {Date.parse(state.referenceTime) !== Date.parse(prediction.anchor_observed_at)
          && <p className="prediction-inputs-warning">Mốc đầu vào thực tế khác mốc đã chọn; các thời điểm dự báo được tính từ mốc thực tế.</p>}
        {(prediction.is_stale || state.inputs.is_stale) && <p className="prediction-inputs-warning">Dữ liệu đầu vào đã cũ so với mốc tham chiếu.</p>}
        {state.actualStatus === 'loading' && <p role="status">Đang tải dữ liệu thực tế để so sánh…</p>}
        {state.actualStatus === 'error' && <div className="prediction-actual-error" role="alert">
          <p>Đã có dự báo, nhưng chưa tải được dữ liệu thực tế. {state.actualError}</p>
          <button type="button" onClick={onRetryActuals}>Tải lại dữ liệu thực tế</button></div>}
        {state.actualStatus === 'ready' && !actuals.some((point) => point.actual_water_level_mm !== null)
          && <p className="prediction-no-actuals">Chưa có dữ liệu thực tế sau mốc đầu vào. Biểu đồ hiện đường dự báo.</p>}
        <ForecastChart prediction={prediction} actuals={actuals} />
        <h4>So sánh tại các mốc dự báo</h4>
        <p>Sai lệch = dự báo − thực tế. Thiếu thực tế hiện “—”, không tính sai số.</p>
        <div className="prediction-comparison-scroll" tabIndex={0} role="region" aria-label="Bảng so sánh dự báo, cuộn ngang">
          <table className="prediction-comparison"><thead><tr>
            <th scope="col">Sau mốc đầu vào</th><th scope="col">Thời điểm</th><th scope="col">Dự báo (mm)</th>
            <th scope="col">Thực tế (mm)</th><th scope="col">Sai lệch (mm)</th><th scope="col">Sai số tuyệt đối (mm)</th>
          </tr></thead><tbody>{rows.map((row) => <tr key={row.horizon_minutes}>
            <th scope="row">{formatWaterLevel(row.horizon_minutes / 60)} giờ</th>
            <td>{formatObservationTime(row.forecast_at)}</td><td>{formatWaterLevel(row.water_level_mm)}</td>
            <td>{row.actual === null ? '—' : formatWaterLevel(row.actual)}</td>
            <td>{row.difference === null ? '—' : formatWaterLevel(row.difference)}</td>
            <td>{row.absoluteError === null ? '—' : formatWaterLevel(row.absoluteError)}</td>
          </tr>)}</tbody></table>
        </div>
        {meanAbsoluteError !== null && <p className="prediction-error-summary">Sai số tuyệt đối trung bình: <strong>{formatWaterLevel(meanAbsoluteError)} mm</strong>
          {' · '}Tính trên {comparable.length}/{rows.length} mốc có thực tế.</p>}
      </>}
    </section>
  );
}
