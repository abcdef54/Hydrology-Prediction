import { formatObservationTime } from '../stations.js';
import { formatWaterLevel } from '../predictions.js';

/**
 * @param {object} props control là lựa chọn tooltip; getX/getY là scale SVG.
 * @returns {import('react').ReactElement} Crosshair và lớp bắt hover/tap/phím dùng chung hai biểu đồ.
 */
export function ChartInteraction({ control, getX, getY, count }) {
  const { row } = control;
  return <g>
    {row && <g className="forecast-tooltip-selection" aria-hidden="true">
      <line x1={getX(row.timestamp)} x2={getX(row.timestamp)} y1="45" y2="300" className="forecast-crosshair" />
      {row.predicted !== null && <circle cx={getX(row.timestamp)} cy={getY(row.predicted)} r="6" className="forecast-tooltip-marker forecast-tooltip-marker--predicted" />}
      {row.actual !== null && <circle cx={getX(row.timestamp)} cy={getY(row.actual)} r="6" className="forecast-tooltip-marker forecast-tooltip-marker--actual" />}
    </g>}
    <rect x="80" y="45" width="840" height="255" fill="transparent" tabIndex={0}
      data-chart-interaction="true" role="slider" aria-label="Chọn thời điểm trên biểu đồ bằng chuột, chạm hoặc phím mũi tên"
      aria-valuemin={0} aria-valuemax={Math.max(count - 1, 0)} aria-valuenow={Math.max(control.index, 0)}
      aria-valuetext={row ? formatObservationTime(row.timestamp) : 'Chưa chọn thời điểm'}
      aria-describedby={row ? control.tooltipId : undefined} {...control.handlers} />
  </g>;
}

/**
 * @param {object} props control chứa row đã ghép đúng timestamp; svgRef/getX định vị hộp.
 * @returns {import('react').ReactElement|null} Tooltip mm, phân biệt thiếu thực tế và giờ không có horizon.
 * Sai lệch = dự báo - thực tế, chỉ tính khi cả hai có giá trị; giữ 0 và số âm.
 */
export default function ChartTooltip({ control, svgRef, getX }) {
  const { row, tooltipId } = control;
  if (!row) return null;
  const svg = svgRef.current;
  const figure = svg?.closest('figure');
  const matrix = svg?.getScreenCTM();
  let left = 8;
  if (svg && figure && matrix) {
    const point = svg.createSVGPoint();
    point.x = getX(row.timestamp);
    const screen = point.matrixTransform(matrix);
    const bounds = figure.getBoundingClientRect();
    left = Math.max(8, Math.min(bounds.width - 258, screen.x - bounds.left + 12));
  }
  const difference = row.predicted === null || row.actual === null ? null : row.predicted - row.actual;
  const value = number => number === null ? '—' : `${formatWaterLevel(number)} mm`;
  return <div id={tooltipId} className="forecast-tooltip" role="tooltip" style={{ left }}>
    <strong>{formatObservationTime(row.timestamp)}</strong>
    <dl>
      <div><dt>Dự báo</dt><dd>{row.predicted === null ? (row.hasForecast ? 'Chưa có dữ liệu' : 'Không có điểm dự báo') : value(row.predicted)}</dd></div>
      <div><dt>Thực tế (TB giờ)</dt><dd>{row.actual === null ? 'Chưa có dữ liệu' : value(row.actual)}</dd></div>
      <div><dt>Sai lệch</dt><dd>{difference === null ? '—' : `${difference > 0 ? '+' : ''}${value(difference)}`}</dd></div>
      <div><dt>Sai số tuyệt đối</dt><dd>{value(difference === null ? null : Math.abs(difference))}</dd></div>
    </dl>
  </div>;
}
