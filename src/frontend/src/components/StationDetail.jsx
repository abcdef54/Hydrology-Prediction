import { useEffect, useState } from 'react';
import { fetchLatestObservation, fetchStationDetail } from '../api/stations.js';
import { getStationAppearance, getStationCoordinates, formatObservationTime } from '../stations.js';
import { fetchFacilityDetail, fetchLatestFacilityObservation } from '../api/facilities.js';
import FacilitySpecifications from './FacilitySpecifications.jsx';
import StationIcon from './StationIcon.jsx';
import StationReading from './StationReading.jsx';

const levelFormatter = new Intl.NumberFormat('vi-VN', { maximumFractionDigits: 3 });

/**
 * Box cố định, tải chi tiết/latest khi mở; chuyển trạm/đóng box hủy request cũ.
 *
 * @param {object} props station là metadata đã chọn; onClose đóng box.
 * @returns {import('react').ReactElement} Vị trí, số đo và ngưỡng nước bằng mm.
 * Pan/zoom không đóng box hay gọi lại API. Không tự gán ngưỡng cho giá trị null.
 */
export default function StationDetail({ station, onClose }) {
  const [detail, setDetail] = useState(null);
  const [reading, setReading] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [detailError, setDetailError] = useState('');
  const [retryCount, setRetryCount] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    setDetail(null);
    setReading(null);
    setDetailError('');
    setIsLoading(true);

    async function load() {
      const [metadataResult, readingResult] = await Promise.allSettled([
        station.facility_id ? fetchFacilityDetail(station.facility_id, controller.signal)
          : fetchStationDetail(station.station_id, controller.signal),
        station.facility_id ? fetchLatestFacilityObservation(station.facility_id, controller.signal)
          : fetchLatestObservation(station.station_id, controller.signal),
      ]);
      if (controller.signal.aborted) return;
      if (metadataResult.status === 'fulfilled') setDetail(metadataResult.value);
      else setDetailError(metadataResult.reason.message);
      if (readingResult.status === 'fulfilled') {
        const observation = readingResult.value;
        setReading({ status: observation ? 'ready' : 'missing', observation });
      } else {
        setReading({ status: 'error', error: readingResult.reason.message });
      }
      setIsLoading(false);
    }
    load();
    return () => controller.abort();
  }, [station, retryCount]);

  const metadata = detail ?? station;
  const coordinates = getStationCoordinates(metadata);
  const hasError = Boolean(detailError) || reading?.status === 'error';

  return (
    <section id="station-detail-panel" className="station-detail" aria-labelledby="station-detail-title">
      <header className="station-detail-header">
        <StationIcon stationType={metadata.station_type} />
        <div className="station-detail-heading">
          <p>{getStationAppearance(metadata.station_type).label}</p>
          <h2 id="station-detail-title">{metadata.name}</h2>
        </div>
        <button type="button" className="station-detail-close" onClick={onClose} aria-label="Đóng thông tin trạm">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
            <path d="m6 6 12 12M18 6 6 18" />
          </svg>
        </button>
      </header>
      <div className="station-detail-content">
        <dl className="station-detail-location">
          <div><dt>Địa chỉ</dt><dd>{metadata.location?.address || 'Chưa có địa chỉ'}</dd></div>
          <div>
            <dt>Tọa độ · vĩ độ, kinh độ</dt>
            <dd>{coordinates
              ? `${coordinates[1].toFixed(6)} · ${coordinates[0].toFixed(6)}`
              : 'Chưa có tọa độ hợp lệ'}</dd>
          </div>
        </dl>
        <div className="station-detail-measurement" aria-live="polite">
          <h3>{metadata.facility_id ? 'Vận hành mới nhất' : metadata.station_type === 'water_level' ? 'Mực nước mới nhất' : 'Số đo mới nhất'}</h3>
          <div className="station-reading"><StationReading reading={reading} detailed={Boolean(metadata.facility_id)} /></div>
        </div>
        {metadata.facility_id && <>
          {reading?.status === 'ready' && <dl className="station-detail-location">
            {Number.isFinite(Date.parse(reading.observation.source_updated_at)) && <div>
              <dt>Cập nhật báo cáo</dt><dd>{formatObservationTime(reading.observation.source_updated_at)}</dd>
            </div>}
            {reading.observation.operation_start_text && <div>
              <dt>Bắt đầu vận hành · theo nguồn</dt><dd>{reading.observation.operation_start_text}</dd>
            </div>}
            {reading.observation.operation_end_text && <div>
              <dt>Kết thúc vận hành · theo nguồn</dt><dd>{reading.observation.operation_end_text}</dd>
            </div>}
          </dl>}
          <FacilitySpecifications specifications={detail?.specifications} isLoading={isLoading} error={detailError} />
        </>}
        {metadata.station_type === 'water_level' && (
          <div className="station-detail-alarms">
            <h3>Mức nước báo động</h3>
            <dl>
              {[1, 2, 3].map((level) => {
                const value = detail?.alarm_levels_m?.[`level_${level}`];
                let label = 'Chưa cấu hình';
                if (isLoading) label = 'Đang tải…';
                else if (detailError) label = 'Chưa tải được';
                else if (Number.isFinite(value)) label = `${levelFormatter.format(value * 1000)} mm`;
                return (
                  <div key={level}>
                    <dt>Báo động {level}</dt>
                    <dd>{label}</dd>
                  </div>
                );
              })}
            </dl>
          </div>
        )}
        {hasError && (
          <div className="station-detail-error" role="status">
            <p>{detailError || reading.error}</p>
            <button type="button" onClick={() => setRetryCount((count) => count + 1)} disabled={isLoading}>Thử lại</button>
          </div>
        )}
      </div>
    </section>
  );
}
