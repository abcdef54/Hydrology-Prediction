import { useEffect, useMemo, useRef, useState } from 'react';
import { useHourlyHistory } from '../hooks/useHourlyHistory.js';
import { formatObservationTime, getStationAppearance } from '../stations.js';
import {
  createHistoryFilters, formatHistoryDate, getHistoryFilterError, getVietnamDate,
  normalizeHistoryFilters,
} from '../history.js';
import HourlyHistoryTable from './HourlyHistoryTable.jsx';
import PredictionInputsPanel from './PredictionInputsPanel.jsx';
import PredictionResult from './PredictionResult.jsx';
import { usePredictionInputs } from '../hooks/usePredictionInputs.js';
import { usePrediction } from '../hooks/usePrediction.js';
import { usePredictionRange } from '../hooks/usePredictionRange.js';
import PredictionRangeResult from './PredictionRangeResult.jsx';

/**
 * Bảng lịch sử độc lập với bộ lọc bản đồ; chỉ tải khi mở/áp dụng lọc/đổi trang.
 *
 * @param {object} props stations là metadata hiện có; station tùy chọn lọc sẵn; onClose đóng bảng.
 * @param {HTMLElement} props.opener Nút mở để khôi phục focus khi đóng bảng.
 * @returns {import('react').ReactElement} Native modal dialog có focus trap, Escape và phân trang.
 * Đóng bảng hủy request; mỗi lần mở mới mặc định hôm nay theo giờ Việt Nam.
 */
export default function HistoryDialog({ stations, station, opener, onClose }) {
  const dialogRef = useRef(null);
  const [draft, setDraft] = useState(() => createHistoryFilters(station));
  const [appliedFilters, setAppliedFilters] = useState(() => normalizeHistoryFilters(draft));
  const [cursors, setCursors] = useState([null]);
  const [retryCount, setRetryCount] = useState(0);
  const [formError, setFormError] = useState('');
  const [inputSelection, setInputSelection] = useState(null);
  const inputState = usePredictionInputs(inputSelection);
  const predictionFlow = usePrediction();
  const [showResult, setShowResult] = useState(false);
  const predictButtonRef = useRef(null);
  const rangeButtonRef = useRef(null);
  const rangeFlow = usePredictionRange();
  const [resultMode, setResultMode] = useState('single');
  const eligibleStations = stations.filter((item) => item.station_type === 'water_level'
    && item.prediction_enabled && item.model_configured);
  const request = useMemo(() => ({
    filters: appliedFilters, cursor: cursors.at(-1), retryCount,
  }), [appliedFilters, cursors, retryCount]);
  const { status, data, error } = useHourlyHistory(request);
  const isLoading = status === 'loading';
  const today = getVietnamDate();
  const stationTypes = [...new Set(stations.map((item) => item.station_type))]
    .sort((first, second) => getStationAppearance(first).order - getStationAppearance(second).order);
  const selectableStations = stations.filter((item) => draft.stationType === 'all' || item.station_type === draft.stationType);

  useEffect(() => {
    const dialog = dialogRef.current;
    dialog.showModal();
    return () => {
      dialog.close();
      if (opener?.isConnected) opener.focus({ preventScroll: true });
    };
  }, [opener]);

  function updateDraft(changes) {
    setDraft((previous) => ({ ...previous, ...changes }));
    setFormError('');
  }

  function applyFilters(event) {
    event.preventDefault();
    const validationError = getHistoryFilterError(draft);
    if (validationError) {
      setFormError(validationError);
      return;
    }
    setAppliedFilters(normalizeHistoryFilters(draft));
    setCursors([null]);
    setRetryCount(0);
    setInputSelection(null);
  }

  function selectInputs(selectedStation, referenceTime) {
    setInputSelection({ station: selectedStation, referenceTime });
  }

  function predict(overrides) {
    if (inputState.status !== 'ready' || !inputState.data?.input_ready || !inputSelection) return;
    predictionFlow.run(inputSelection, inputState.data, overrides);
    setResultMode('single');
    setShowResult(true);
  }

  function returnToHistory() {
    predictionFlow.cancel();
    rangeFlow.cancel();
    setShowResult(false);
    requestAnimationFrame(() => (resultMode === 'range' ? rangeButtonRef : predictButtonRef).current?.focus({ preventScroll: true }));
  }

  function predictRange(request) {
    rangeFlow.run(request);
    setResultMode('range');
    setShowResult(true);
  }

  return (
    <dialog ref={dialogRef} className="history-dialog" aria-labelledby="history-title"
      onCancel={(event) => { event.preventDefault(); onClose(); }}>
      <header className="history-header">
        <div>
          <p className="station-sidebar-eyebrow">DỮ LIỆU QUAN TRẮC</p>
          <h2 id="history-title">{showResult ? 'Dự báo mực nước' : 'Lịch sử quan trắc'}</h2>
          <p className="history-description">{showResult ? 'Dự báo và thực tế · Giờ Việt Nam · Đơn vị mm'
            : 'Mực nước trung bình giờ · Mưa cộng theo giờ · Giờ Việt Nam · “—”: thiếu · Ô nhạt: chưa tới giờ'}</p>
        </div>
        <button type="button" className="station-detail-close" onClick={onClose} aria-label="Đóng bảng lịch sử" autoFocus>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true"><path d="m6 6 12 12M18 6 6 18" /></svg>
        </button>
      </header>

      <div className="history-content" hidden={showResult}>
      <form className="history-filters" onSubmit={applyFilters} noValidate>
        <div className="history-filter-field"><label htmlFor="history-station-type">Loại trạm</label>
          <select id="history-station-type" value={draft.stationType} onChange={(event) => updateDraft({ stationType: event.target.value, stationId: '' })}>
            <option value="all">Tất cả loại trạm</option>
            {stationTypes.map((type) => <option key={type} value={type}>{getStationAppearance(type).label}</option>)}
          </select>
        </div>
        <div className="history-filter-field"><label htmlFor="history-station-id">Tên trạm</label>
          <select id="history-station-id" value={draft.stationId} onChange={(event) => updateDraft({ stationId: event.target.value })}>
            <option value="">Tất cả trạm</option>
            {selectableStations.map((item) => <option key={item.station_id} value={item.station_id}>{item.name}</option>)}
          </select>
        </div>
        <div className="history-filter-field"><label htmlFor="history-mode">Thời gian</label>
          <select id="history-mode" value={draft.mode} onChange={(event) => updateDraft({ mode: event.target.value })}>
            <option value="day">Một ngày</option><option value="range">Khoảng ngày</option>
          </select>
        </div>
        <div className="history-filter-field"><label htmlFor="history-start">{draft.mode === 'day' ? 'Ngày' : 'Từ ngày'}</label>
          <input id="history-start" type="date" value={draft.startDate} min="1900-01-01" max={today} required
            onChange={(event) => updateDraft({ startDate: event.target.value })} />
        </div>
        {draft.mode === 'range' && <div className="history-filter-field"><label htmlFor="history-end">Đến ngày</label>
          <input id="history-end" type="date" value={draft.endDate} min={draft.startDate || '1900-01-01'} max={today} required
            onChange={(event) => updateDraft({ endDate: event.target.value })} />
        </div>}
        <button type="submit" className="history-primary-button">Áp dụng</button>
        {formError && <p className="history-form-error" role="alert">{formError}</p>}
      </form>

      <PredictionInputsPanel stations={stations} eligibleStations={eligibleStations} selection={inputSelection}
        defaultStationId={station?.station_id} onSelect={selectInputs} onClear={() => setInputSelection(null)}
        inputState={inputState} onPredict={predict} predictButtonRef={predictButtonRef}
        onPredictRange={predictRange} rangeButtonRef={rangeButtonRef} />

      <div className="history-summary" aria-live="polite">
        <span>Đang xem: {formatHistoryDate(appliedFilters.startDate)}
          {appliedFilters.endDate !== appliedFilters.startDate && ` – ${formatHistoryDate(appliedFilters.endDate)}`}
        </span>
        {data && <span>Đọc dữ liệu lúc {formatObservationTime(data.as_of)}</span>}
      </div>

      {isLoading && <div className="history-state" role="status">Đang tải lịch sử quan trắc…</div>}
      {status === 'error' && <div className="history-state" role="alert">
        <p>{error}</p>
        <button type="button" className="history-primary-button" onClick={() => setRetryCount((count) => count + 1)}>Thử lại</button>
      </div>}
      {data && <HourlyHistoryTable data={data} eligibleStations={eligibleStations}
        selection={inputSelection} onSelectInputs={selectInputs} />}
      </div>

      <footer className="history-footer" hidden={showResult}>
        <div><strong>Trang {cursors.length}</strong>{data && ` · ${data.rows.length} dòng trạm/ngày`}</div>
        <p>Mực nước: trung bình trong (mốc giờ − 1 giờ, mốc giờ], gồm cả số đo tại mốc giờ. Mưa: tổng các số đo trong cùng khoảng giờ; chưa xác định khoảng tích lũy của từng số đo nguồn. Gió: số đo tại mốc giờ.</p>
        <nav aria-label="Phân trang lịch sử">
          <button type="button" disabled={isLoading || cursors.length === 1}
            onClick={() => setCursors((previous) => previous.slice(0, -1))}>Trang trước</button>
          <button type="button" disabled={isLoading || !data?.has_more}
            onClick={() => setCursors((previous) => [...previous, data.next_cursor])}>Trang tiếp</button>
        </nav>
      </footer>
      {showResult && resultMode === 'single' && predictionFlow.state && <PredictionResult state={predictionFlow.state}
        onBack={returnToHistory}
        onRetry={() => predictionFlow.run(predictionFlow.state.selection, predictionFlow.state.inputs, predictionFlow.state.overrides)}
        onRetryActuals={predictionFlow.retryActuals} />}
      {showResult && resultMode === 'range' && rangeFlow.state && <PredictionRangeResult state={rangeFlow.state}
        onBack={returnToHistory} onResume={rangeFlow.resume} />}
    </dialog>
  );
}
