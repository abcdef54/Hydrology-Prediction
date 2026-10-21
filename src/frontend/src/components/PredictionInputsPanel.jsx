import { useState } from 'react';
import { formatObservationTime } from '../stations.js';
import { buildInputOverrides, describePredictionFeature } from '../predictionInputs.js';
import PredictionRangeForm from './PredictionRangeForm.jsx';

/**
 * @param {object} props stations dùng tra tên nguồn; eligibleStations là trạm nước được backend cho phép dự báo;
 * selection chứa trạm/mốc đang chọn; onSelect chọn mốc hoặc latest, onClear hủy lựa chọn.
 * inputState là auto-fill gốc; onPredict nhận các overrides khác gốc.
 * @returns {import('react').ReactElement} Input có thể sửa theo quyền API, reset và kiểm tra lỗi.
 */
export default function PredictionInputsPanel({ stations, eligibleStations, selection, onSelect, onClear, defaultStationId,
  inputState, onPredict, predictButtonRef, onPredictRange, rangeButtonRef,
  activeModelId, onSelectModel, modelsData, modelsStatus, onStationChange }) {
  const [internalStationId, setInternalStationId] = useState(defaultStationId ?? '');
  const { status, data, error } = inputState;
  const [editState, setEditState] = useState(null);
  const drafts = editState?.inputs === data ? editState.values : {};
  const { overrides, errors } = buildInputOverrides(data, drafts);
  const hasErrors = Object.keys(errors).length > 0;
  function editFeature(name, value) {
    setEditState({ inputs: data, values: { ...drafts, [name]: value } });
  }
  const stationId = selection?.station.station_id ?? internalStationId;
  const latestStation = eligibleStations.find((station) => station.station_id === stationId)
    ?? (eligibleStations.length === 1 ? eligibleStations[0] : null);

  const availableModels = modelsData?.station_id === latestStation?.station_id ? (modelsData?.models ?? []) : [];
  const currentModelId = activeModelId ?? modelsData?.default_model_id ?? availableModels[0]?.model_id;
  const currentModel = availableModels.find((m) => m.model_id === currentModelId);

  return (
    <section className="prediction-inputs-panel" aria-labelledby="prediction-inputs-title">
      <div className="prediction-inputs-toolbar">
        <div><h3 id="prediction-inputs-title">Đầu vào dự báo</h3>
          <p>Chọn một ô giờ để tự điền đầu vào, hoặc bỏ chọn để xem theo khoảng.</p></div>
        <div className="prediction-inputs-actions">
          <label className="sr-only" htmlFor="prediction-latest-station">Trạm lấy đầu vào mới nhất</label>
          <select id="prediction-latest-station" value={latestStation?.station_id ?? ''}
            onChange={(event) => {
              const sid = event.target.value;
              setInternalStationId(sid);
              onStationChange?.(sid);
              onClear();
            }}>
            <option value="" disabled>Chọn trạm nước</option>
            {eligibleStations.map((station) => <option key={station.station_id} value={station.station_id}>{station.name}</option>)}
          </select>
          {eligibleStations.length > 0 && (
            <>
              <label className="sr-only" htmlFor="prediction-model-select">Mô hình dự báo</label>
              <select id="prediction-model-select"
                aria-label="Mô hình dự báo"
                value={currentModelId ?? ''}
                disabled={modelsStatus === 'loading' || !availableModels.length}
                onChange={(event) => {
                  setEditState(null);
                  onSelectModel?.(event.target.value);
                }}>
                {modelsStatus === 'loading' && <option value="">Đang đọc danh sách mô hình…</option>}
                {modelsStatus === 'error' && <option value="">Chưa tải được mô hình</option>}
                {availableModels.map((model) => (
                  <option key={model.model_id} value={model.model_id}>
                    {model.name}
                  </option>
                ))}
              </select>
            </>
          )}
          <button type="button" disabled={!latestStation || !currentModel}
            onClick={() => onSelect(latestStation, null)}>Đầu vào mới nhất</button>
          {selection && <button type="button" onClick={onClear}>Bỏ chọn</button>}
          {Object.keys(drafts).length > 0 && <button type="button" onClick={() => setEditState(null)}>Khôi phục giá trị gốc</button>}
          <button type="button" className="prediction-run-button" ref={predictButtonRef}
            disabled={status !== 'ready' || !data?.input_ready || !selection || hasErrors}
            onClick={() => onPredict(overrides)}>Dự báo</button>
        </div>
      </div>
      {!eligibleStations.length && <p>Chưa có trạm nước được bật dự báo và cấu hình mô hình.</p>}
      <PredictionRangeForm station={latestStation} enabled={!selection && Boolean(currentModel)} onRun={onPredictRange} buttonRef={rangeButtonRef}
        modelId={currentModelId} modelName={currentModel?.name} />
      {selection && <div className="prediction-inputs-status" aria-live="polite">
        <strong>{selection.station.name}</strong>
        {currentModel?.name && <> · Mô hình: <strong>{currentModel.name}</strong></>}
        {' · '}Mốc yêu cầu: {selection.referenceTime ? formatObservationTime(selection.referenceTime) : 'Mới nhất'}
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
            <p>Có thể sửa các ô được cho phép trước khi dự báo. Chỉnh sửa chỉ áp dụng tại mốc đầu vào, không tự đổi các ô liên quan.</p>
            <p>Giá trị tự nhập và kết quả dự báo thử không lưu vào database. Để trống số đo mưa hoặc giá trị trước đó nghĩa là thiếu, không phải số 0.</p>
            {Object.keys(overrides).length > 0 && <p className="prediction-inputs-warning">Đã thay đổi {Object.keys(overrides).length} giá trị · Sẽ chạy dự báo thử.</p>}
            {hasErrors && <p role="alert">Vui lòng sửa các ô chưa hợp lệ trước khi dự báo.</p>}
            <p>Tổng số đo mưa được cộng từ các số đo trong khoảng ghi rõ; chưa xác định khoảng tích lũy của từng số đo nguồn.</p>
            <div className="prediction-feature-grid">
              {data.features.map((feature, index) => {
                const display = describePredictionFeature(feature, stations, index);
                const editable = feature.editable && data.editable_features.includes(feature.name);
                const isFlag = feature.name.endsWith('_missing') || feature.name === 'flood_season';
                const value = drafts[feature.name] ?? String(feature.value ?? '');
                return <div className="prediction-feature" key={feature.name}>
                  <label id={`prediction-label-${index}`} htmlFor={`prediction-feature-${index}`}>{display.label}</label>
                  {editable && isFlag ? <select id={`prediction-feature-${index}`} value={value}
                    aria-describedby={`prediction-source-${index}`} aria-invalid={Boolean(errors[feature.name])}
                    onChange={(event) => editFeature(feature.name, event.target.value)}>
                    <option value="" disabled>Chọn trạng thái</option>
                    <option value="0">{feature.name === 'flood_season' ? 'Không' : 'Có dữ liệu'}</option>
                    <option value="1">{feature.name === 'flood_season' ? 'Có' : 'Thiếu dữ liệu'}</option>
                  </select> : <input id={`prediction-feature-${index}`} readOnly={!editable}
                    inputMode="decimal" value={editable ? value : display.value} placeholder="Thiếu"
                    aria-invalid={Boolean(errors[feature.name])}
                    onChange={(event) => editFeature(feature.name, event.target.value)}
                    aria-describedby={`prediction-source-${index}`} />}
                  <span id={`prediction-source-${index}`}>
                    {[display.unit, display.sourceName].filter(Boolean).join(' · ')}
                    {errors[feature.name] && <strong className="prediction-field-error">{errors[feature.name]}</strong>}
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
