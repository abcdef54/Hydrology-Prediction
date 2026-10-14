import { formatObservationTime } from '../stations.js';

const valueFormatter = new Intl.NumberFormat('vi-VN', { maximumFractionDigits: 3 });

/**
 * @param {object} props reading có status, observation và error; null là đang tải.
 * @returns {import('react').ReactElement} Số đo, đơn vị và giờ đo dùng chung cho sidebar/box.
 */
export default function StationReading({ reading }) {
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
