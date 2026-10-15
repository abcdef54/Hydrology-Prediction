import { formatObservationTime } from '../stations.js';
import FacilityReading from './FacilityReading.jsx';

const valueFormatter = new Intl.NumberFormat('vi-VN', { maximumFractionDigits: 3 });
const windLabels = {
  ws: 'Tốc độ gió', wd: 'Hướng gió', wsg: 'Tốc độ gió giật', wdg: 'Hướng gió giật',
};

/**
 * @param {object} props reading có status, observation và error; null là đang tải.
 * @returns {import('react').ReactElement} Số đo, đơn vị và giờ đo dùng chung cho sidebar/box.
 */
export default function StationReading({ reading, detailed = false }) {
  if (!reading) return <span className="station-reading-placeholder">Đang tải số đo…</span>;
  if (reading.status === 'missing') return <span className="station-reading-placeholder">Chưa có số đo</span>;
  if (reading.status === 'error') {
    return <span className="station-reading-error" title={reading.error}>Không tải được số đo</span>;
  }
  const observation = reading.observation;
  return (
    <>
      {observation.facility_id ? <FacilityReading observation={observation} detailed={detailed} />
        : observation.station_type === 'wind' ? (
        <>
          {Object.values(observation.measurements).some((field) => !field.unit_confirmed) && (
            <span className="station-reading-placeholder">Số đo gốc · Chưa xác minh đơn vị/thang đo</span>
          )}
          <span className="station-wind-readings">
            {Object.entries(windLabels).map(([field, label]) => {
              const measurement = observation.measurements[field];
              return <span className="station-wind-reading" key={field}>
                <span>{label}</span>
                <strong>{measurement.value === null ? '—' : valueFormatter.format(measurement.value)}
                  {measurement.unit_confirmed && measurement.unit ? ` ${measurement.unit}` : ''}
                </strong>
              </span>;
            })}
          </span>
        </>
      ) : <span className="station-reading-value">
        {valueFormatter.format(observation.value)}
        <span className="station-reading-unit">{observation.unit ?? '(chưa rõ đơn vị)'}</span>
      </span>}
      <span className={`station-reading-time${observation.is_stale ? ' station-reading-time--stale' : ''}`}>
        {observation.is_stale ? 'Dữ liệu cũ · ' : 'Đo lúc '}
        <time dateTime={observation.observed_at}>{formatObservationTime(observation.observed_at)}</time>
      </span>
    </>
  );
}
