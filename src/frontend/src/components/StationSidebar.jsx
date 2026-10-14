import StationIcon from './StationIcon.jsx';
import {
  formatObservationTime, getLatestDataTime, getStationAppearance, getStationCoordinates,
} from '../stations.js';

const valueFormatter = new Intl.NumberFormat('vi-VN', { maximumFractionDigits: 3 });

function StationReading({ reading }) {
  if (!reading) return <span className="station-reading-placeholder">Đang tải số đo…</span>;
  if (reading.status === 'missing') return <span className="station-reading-placeholder">Chưa có số đo</span>;
  if (reading.status === 'error') {
    return <span className="station-reading-error" title={reading.error}>Không tải được số đo</span>;
  }
  const observation = reading.observation;
  return (
    <>
      <span className="station-reading-value">
        {valueFormatter.format(observation.value)}
        <span className="station-reading-unit">{observation.unit ?? '(chưa rõ đơn vị)'}</span>
      </span>
      <span className={`station-reading-time${observation.is_stale ? ' station-reading-time--stale' : ''}`}>
        {observation.is_stale ? 'Dữ liệu cũ · ' : 'Đo lúc '}
        <time dateTime={observation.observed_at}>{formatObservationTime(observation.observed_at)}</time>
      </span>
    </>
  );
}

/**
 * Sidebar cuộn độc lập: bộ lọc áp dụng cho cả danh sách lẫn marker.
 *
 * @param {object} props Danh sách đầy đủ/đã lọc, số đo theo mã, trạng thái tải và callbacks.
 * @returns {import('react').ReactElement} Danh sách trạm có thể chọn bằng chuột/bàn phím.
 * Click chỉ chọn trạm để focus bản đồ; chưa mở detail hay chạy dự báo.
 */
export default function StationSidebar({
  stations, visibleStations, readings, stationType, onFilterChange,
  selectedStationId, onStationSelect, isLoading, isLoadingReadings, errorMessage, onRetry,
}) {
  const stationTypes = [...new Set(stations.map((station) => station.station_type))]
    .sort((first, second) => getStationAppearance(first).order - getStationAppearance(second).order
      || first.localeCompare(second));
  const latestTime = getLatestDataTime(stations, readings);
  const hasReadingErrors = Object.values(readings).some((reading) => reading.status === 'error');

  return (
    <aside id="station-sidebar" className="station-sidebar" aria-labelledby="station-sidebar-title">
      <header className="station-sidebar-header">
        <div className="station-sidebar-heading">
          <div>
            <p className="station-sidebar-eyebrow">QUAN TRẮC THỦY VĂN</p>
            <h1 id="station-sidebar-title">Trạm quan trắc</h1>
          </div>
          <span className="station-count" aria-label={`${stations.length} trạm`}>{stations.length}</span>
        </div>
        <p className="station-latest" aria-live="polite">
          Dữ liệu mới nhất
          <strong>
            {latestTime
              ? <time dateTime={latestTime}>{formatObservationTime(latestTime)} · giờ Việt Nam</time>
              : (isLoading || isLoadingReadings ? 'Đang tải…' : 'Chưa có dữ liệu')}
          </strong>
        </p>
        <label className="station-filter-label" htmlFor="station-type-filter">Loại trạm</label>
        <select
          id="station-type-filter" className="station-filter" value={stationType}
          onChange={(event) => onFilterChange(event.target.value)}
        >
          <option value="all">Tất cả trạm ({stations.length})</option>
          {stationTypes.map((type) => (
            <option key={type} value={type}>
              {getStationAppearance(type).label} ({stations.filter((station) => station.station_type === type).length})
            </option>
          ))}
        </select>
      </header>

      {(errorMessage || hasReadingErrors) && (
        <div className="station-sidebar-notice" role="status">
          <p>{errorMessage || 'Một số trạm chưa tải được số đo mới nhất.'}</p>
          <button type="button" onClick={onRetry} disabled={isLoading || isLoadingReadings}>Thử lại</button>
        </div>
      )}

      <div className="station-list-scroll" tabIndex={0} role="region" aria-label="Danh sách trạm, cuộn để xem thêm" aria-busy={isLoading}>
        {isLoading && stations.length === 0 && <p className="station-list-empty" role="status">Đang tải danh sách trạm…</p>}
        {!isLoading && !errorMessage && visibleStations.length === 0 && (
          <p className="station-list-empty">Chưa có trạm thuộc loại này.</p>
        )}
        <ul className="station-list">
          {visibleStations.map((station) => {
            const hasCoordinates = Boolean(getStationCoordinates(station));
            return (
              <li key={station.station_id}>
                <button
                  type="button" className="station-card"
                  aria-pressed={selectedStationId === station.station_id}
                  onClick={() => onStationSelect(station)} disabled={!hasCoordinates}
                  title={hasCoordinates ? `Xem vị trí ${station.name}` : 'Trạm chưa có tọa độ hợp lệ'}
                >
                  <span className="station-card-heading">
                    <StationIcon stationType={station.station_type} />
                    <span className="station-card-name">
                      <strong>{station.name}</strong>
                      <span>{getStationAppearance(station.station_type).label}</span>
                    </span>
                    <svg className="station-card-arrow" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true">
                      <path d="m9 6 6 6-6 6" />
                    </svg>
                  </span>
                  <span className="station-address">{station.location?.address || 'Chưa có địa chỉ'}</span>
                  <span className="station-reading">
                    <StationReading reading={readings[station.station_id]} />
                  </span>
                  {!hasCoordinates && <span className="station-reading-error">Chưa có tọa độ hợp lệ</span>}
                </button>
              </li>
            );
          })}
        </ul>
      </div>
      <footer className="station-sidebar-footer">
        {visibleStations.length} trạm hiển thị · Chọn trạm để xem vị trí
      </footer>
    </aside>
  );
}
