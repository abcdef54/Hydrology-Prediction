import { useState } from 'react';
import { usePredictionInputs } from '../hooks/usePredictionInputs.js';
import { formatObservationTime } from '../stations.js';
import { describePredictionFeature } from '../predictionInputs.js';

/**
 * @param {object} props stations dùng tra tên nguồn; eligibleStations là trạm nước được backend cho phép dự báo;
 * selection chứa trạm/mốc đang chọn; onSelect chọn mốc hoặc latest, onClear hủy lựa chọn.
 * @returns {import('react').ReactElement} Vùng auto-fill chỉ đọc, chưa chạy dự báo hoặc sửa input.
 */
export default function PredictionInputsPanel({ stations, eligibleStations, selection, onSelect, onClear, defaultStationId }) {
  const [latestStationId, setLatestStationId] = useState(defaultStationId ?? '');
  const { status, data, error } = usePredictionInputs(selection);
  const stationId = selection?.station.station_id ?? latestStationId;
  const latestStation = eligibleStations.find((station) => station.station_id === stationId)
    ?? (eligibleStations.length === 1 ? eligibleStations[0] : null);
  return (
    <section className="prediction-inputs-panel" aria-labelledby="prediction-inputs-title">
      <div className="prediction-inputs-toolbar">
        <div><h3 id="prediction-inputs-title">Đầu vào dự báo</h3>
          <p>Chọn ô giờ của trạm nước để tự điền đầu vào tại mốc đó.</p></div>
        <div className="prediction-inputs-actions">
          <label className="sr-only" htmlFor="prediction-latest-station">Trạm lấy đầu vào mới nhất</label>
          <select id="prediction-latest-station" value={latestStation?.station_id ?? ''}
            onChange={(event) => { setLatestStationId(event.target.value); onClear(); }}>
            <option value="" disabled>Chọn trạm nước</option>
            {eligibleStations.map((station) => <option key={station.station_id} value={station.station_id}>{station.name}</option>)}
          </select>
          <button type="button" disabled={!latestStation} onClick={() => onSelect(latestStation, null)}>Đầu vào mới nhất</button>
          {selection && <button type="button" onClick={onClear}>Bỏ chọn</button>}
        </div>
      </div>
      {!eligibleStations.length && <p>Chưa có trạm nước được bật dự báo và cấu hình mô hình.</p>}
      {selection && <div className="prediction-inputs-status" aria-live="polite">
        <strong>{selection.station.name}</strong> · Mốc yêu cầu: {selection.referenceTime ? formatObservationTime(selection.referenceTime) : 'Mới nhất'}
        {status === 'loading' && <p role="status">Đang chuẩn bị đầu vào…</p>}
        {status === 'error' && <div role="alert"><p>{error}</p>
          <button type="button" onClick={() => onSelect(selection.station, selection.referenceTime)}>Thử lại đầu vào</button></div>}
        {data && <>
          <p className={data.input_ready ? 'prediction-inputs-ready' : 'prediction-inputs-warning'}>
            {data.input_ready ? 'Đủ đầu vào để dự báo' : 'Chưa đủ đầu vào để dự báo'}
            {' · '}Mốc thực tế: {formatObservationTime(data.anchor_observed_at)}
          </p>
          {selection.referenceTime && Date.parse(selection.referenceTime) !== Date.parse(data.anchor_observed_at)
            && <p className="prediction-inputs-warning">Mốc thực tế khác mốc đã chọn; dự báo sẽ tính từ mốc thực tế.</p>}
          {data.is_stale && <p className="prediction-inputs-warning">Đầu vào đã cũ so với mốc tham chiếu.</p>}
          <details className="prediction-feature-details">
            <summary>Xem {data.feature_count} thông tin đầu vào · Thiếu {data.missing_features.length} giá trị</summary>
            <p>Thông tin tại mốc đầu vào, chỉ đọc. “Thiếu” không phải số 0; hệ thống xử lý dữ liệu thiếu khi chuẩn bị dự báo.</p>
            <p>Tổng số đo mưa được cộng từ các số đo trong khoảng ghi rõ; chưa xác định khoảng tích lũy của từng số đo nguồn.</p>
            <div className="prediction-feature-grid">
              {data.features.map((feature, index) => {
                const display = describePredictionFeature(feature, stations, index);
                return <div className="prediction-feature" key={feature.name}>
                  <label id={`prediction-label-${index}`} htmlFor={`prediction-feature-${index}`}>{display.label}</label>
                  <input id={`prediction-feature-${index}`} readOnly value={display.value} placeholder="Thiếu"
                    aria-describedby={`prediction-source-${index}`} />
                  <span id={`prediction-source-${index}`}>
                    {[display.unit, display.sourceName].filter(Boolean).join(' · ')}
                  </span>
                </div>;
              })}
            </div>
          </details>
        </>}
      </div>}
    </section>
  );
}
