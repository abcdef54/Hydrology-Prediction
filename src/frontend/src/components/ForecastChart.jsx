import { useId } from 'react';
import { formatObservationTime } from '../stations.js';
import { formatWaterLevel, getWaterLevelDomain, splitActualSegments } from '../predictions.js';

const hourFormatter = new Intl.DateTimeFormat('vi-VN', { timeZone: 'Asia/Ho_Chi_Minh', hour: '2-digit', minute: '2-digit', hourCycle: 'h23' });

/**
 * @param {object} props prediction có anchor và horizon; actuals trung bình giờ, giữ null.
 * @returns {import('react').ReactElement} SVG co giãn, thời gian tuyến tính, điểm có tooltip và focus bàn phím.
 * Đường nối dự báo chỉ thể hiện xu hướng giữa các horizon; thực tế ngắt tại dữ liệu thiếu.
 */
export default function ForecastChart({ prediction, actuals = [] }) {
  const titleId = useId();
  const descriptionId = useId();
  const anchor = Date.parse(prediction.anchor_observed_at);
  const end = Date.parse(prediction.predictions.at(-1).forecast_at);
  const observed = actuals.filter((point) => point.actual_water_level_mm !== null);
  const [low, high] = getWaterLevelDomain([
    ...prediction.predictions.map((point) => point.water_level_mm), ...observed.map((point) => point.actual_water_level_mm),
  ]);
  const x = (timestamp) => 80 + (Date.parse(timestamp) - anchor) / (end - anchor) * 840;
  const y = (value) => 300 - (value - low) / (high - low) * 255;
  const line = (points, timeKey, valueKey) => points.map((point) => `${x(point[timeKey])},${y(point[valueKey])}`).join(' ');
  return (
    <figure className="forecast-chart">
      <div className="forecast-chart-legend"><span className="forecast-legend-predicted">● Dự báo</span>
        <span className="forecast-legend-actual">● Thực tế — trung bình giờ</span></div>
      <div className="forecast-chart-scroll" tabIndex={0} role="region" aria-label="Biểu đồ mực nước, cuộn ngang trên màn hình nhỏ">
        <svg viewBox="0 0 960 380" role="img" aria-labelledby={`${titleId} ${descriptionId}`}>
          <title id={titleId}>Dự báo và mực nước thực tế, đơn vị mm</title>
          <desc id={descriptionId}>Dự báo có {prediction.predictions.length} mốc. Thực tế có {observed.length} mốc theo giờ;
            ngắt đường tại giờ thiếu. Thời gian theo giờ Việt Nam. Bảng dưới biểu đồ cung cấp giá trị dự báo và sai số.</desc>
          <text x="16" y="25" className="forecast-axis-label">Mực nước (mm)</text>
          {Array.from({ length: 5 }, (_, index) => {
            const value = low + (high - low) * index / 4;
            return <g key={index}><line x1="80" x2="920" y1={y(value)} y2={y(value)} className="forecast-grid-line" />
              <text x="68" y={y(value) + 4} textAnchor="end">{formatWaterLevel(value)}</text></g>;
          })}
          <line x1="80" x2="80" y1="45" y2="300" className="forecast-anchor-line" />
          <text x="80" y="367">Mốc đầu vào</text>
          {prediction.predictions.map((point) => <g key={point.horizon_minutes}>
            <text x={x(point.forecast_at)} y="325" textAnchor="middle">+{formatWaterLevel(point.horizon_minutes / 60)}h</text>
            <text x={x(point.forecast_at)} y="346" textAnchor="middle">{hourFormatter.format(new Date(point.forecast_at))}</text>
          </g>)}
          {splitActualSegments(actuals).map((segment, index) => <polyline key={index}
            points={line(segment, 'timestamp', 'actual_water_level_mm')} className="forecast-line forecast-line--actual" />)}
          <polyline points={line(prediction.predictions, 'forecast_at', 'water_level_mm')} className="forecast-line forecast-line--predicted" />
          {observed.map((point) => <circle key={point.timestamp} cx={x(point.timestamp)} cy={y(point.actual_water_level_mm)} r="3.5"
            className="forecast-dot forecast-dot--actual" tabIndex={0}
            aria-label={`Thực tế ${formatObservationTime(point.timestamp)}: ${formatWaterLevel(point.actual_water_level_mm)} mm`}>
            <title>{`Thực tế ${formatObservationTime(point.timestamp)}: ${formatWaterLevel(point.actual_water_level_mm)} mm (${point.actual_sample_count} số đo)`}</title>
          </circle>)}
          {prediction.predictions.map((point) => <circle key={point.horizon_minutes} cx={x(point.forecast_at)} cy={y(point.water_level_mm)} r="5"
            className="forecast-dot forecast-dot--predicted" tabIndex={0}
            aria-label={`Dự báo ${formatObservationTime(point.forecast_at)}: ${formatWaterLevel(point.water_level_mm)} mm`}>
            <title>{`Dự báo ${formatObservationTime(point.forecast_at)}: ${formatWaterLevel(point.water_level_mm)} mm`}</title>
          </circle>)}
        </svg>
      </div>
      <figcaption><span className="forecast-mobile-hint">Cuộn ngang biểu đồ để xem đủ các mốc. </span>
        Giờ Việt Nam · Các đoạn nối giữa mốc dự báo chỉ thể hiện xu hướng, không phải dự báo bổ sung.
        Thực tế là trung bình từng giờ, khác số đo đúng mốc giờ trong bảng lịch sử.</figcaption>
    </figure>
  );
}
