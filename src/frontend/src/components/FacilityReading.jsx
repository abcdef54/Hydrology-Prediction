import { facilityMetricLabels } from '../facilities.js';

const formatter = new Intl.NumberFormat('vi-VN', { maximumFractionDigits: 3 });

/**
 * @param {object} props observation là báo cáo đã xác thực; detailed mở đủ sáu chỉ tiêu.
 * @returns {import('react').ReactElement} Số gốc, đơn vị đã xác nhận nếu có và số cửa mở.
 * Không suy ra đơn vị/scale từ giá trị hoặc từ thông số cao trình của công trình.
 */
export default function FacilityReading({ observation, detailed = false }) {
  const fields = detailed ? Object.keys(facilityMetricLabels) : ['htl', 'hhl', 'qden', 'qdi'];
  return <>
    {Object.values(observation.measurements).some((metric) => !metric.unit_confirmed) && (
      <span className="station-reading-placeholder">Số đo gốc · Chưa xác minh đơn vị/thang đo</span>
    )}
    <span className="station-wind-readings">
      {fields.map((field) => {
        const metric = observation.measurements[field];
        return <span className="station-wind-reading" key={field}>
          <span>{facilityMetricLabels[field]}</span>
          <strong>{metric.value === null ? '—' : formatter.format(metric.value)}
            {metric.unit_confirmed && metric.unit ? ` ${metric.unit}` : ''}
          </strong>
        </span>;
      })}
      {observation.gate_opening_text !== null && <span className="station-wind-reading">
        <span>Cửa đang mở</span><strong>{observation.gate_opening_text}</strong>
      </span>}
    </span>
  </>;
}
