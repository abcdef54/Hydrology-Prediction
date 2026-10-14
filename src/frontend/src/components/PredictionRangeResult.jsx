import { useEffect, useMemo, useRef } from 'react';
import { formatObservationTime } from '../stations.js';
import { formatWaterLevel, getWaterLevelDomain, splitActualSegments } from '../predictions.js';
import { useChartTooltip } from '../hooks/useChartTooltip.js';
import ChartTooltip, { ChartInteraction } from './ChartTooltip.jsx';

const timeFormatter = new Intl.DateTimeFormat('vi-VN', {timeZone:'Asia/Ho_Chi_Minh',day:'2-digit',month:'2-digit',hour:'2-digit',minute:'2-digit',hourCycle:'h23'});
const reasons = {insufficient_history:'Thiếu lịch sử đầu vào',anchor_unavailable:'Thiếu đúng mốc đầu vào',future_anchor:'Mốc đầu vào chưa tới'};

/**
 * @param {object} props state gồm toàn bộ điểm của các nhóm đã hoàn tất; onResume tiếp nhóm lỗi.
 * onBack hủy chờ và giữ lựa chọn thời gian; không xóa dự báo đã lưu.
 * @returns {import('react').ReactElement} Biểu đồ một horizon, tiến độ và bảng đối chiếu mm.
 */
export default function PredictionRangeResult({state,onBack,onResume}) {
  const heading = useRef(null);
  useEffect(()=>{heading.current?.focus();},[]);
  const {request,points} = state;
  const total = (Date.parse(request.end)-Date.parse(request.start))/3600000;
  const values = points.flatMap(point=>[point.predicted_water_level_mm,point.actual_water_level_mm]).filter(value=>value!==null);
  const [low,high] = values.length ? getWaterLevelDomain(values) : [0,1];
  const first=Date.parse(request.start), last=Date.parse(request.end)-3600000;
  const x = timestamp=>80+(Date.parse(timestamp)-first)/Math.max(last-first,3600000)*840;
  const y = value=>300-(value-low)/(high-low)*255;
  const segments = key=>splitActualSegments(points.map(point=>({...point,actual_water_level_mm:point[key]})));
  const comparable=points.filter(point=>point.predicted_water_level_mm!==null&&point.actual_water_level_mm!==null);
  const mae=comparable.length ? comparable.reduce((sum,point)=>sum+Math.abs(point.predicted_water_level_mm-point.actual_water_level_mm),0)/comparable.length : null;
  const svgRef = useRef(null);
  const tooltipRows = useMemo(() => points.map(point => ({ timestamp: point.forecast_at,
    predicted: point.predicted_water_level_mm, actual: point.actual_water_level_mm, hasForecast: true })), [points]);
  const tooltip = useChartTooltip(tooltipRows, svgRef, x);
  return <section className="prediction-result" aria-labelledby="range-result-title">
    <div className="prediction-result-heading"><div>
      <h3 id="range-result-title" ref={heading} tabIndex={-1}>Dự báo theo khoảng — {request.station.name}</h3>
      <p>{formatObservationTime(request.start)} đến trước {formatObservationTime(request.end)} · Dự báo trước {request.horizon/60} giờ · mm</p>
    </div><button type="button" onClick={onBack}>{state.status==='loading'||state.status==='waiting'?'Dừng và quay lại lịch sử':'← Quay lại lịch sử'}</button></div>
    <div className="prediction-result-meta" aria-live="polite">
      <p>Đã xử lý {points.length}/{total} mốc giờ · Dùng lại {state.reused} · Tạo mới {state.created} · Bỏ qua {state.skipped}</p>
      {(state.status==='loading'||state.status==='waiting')&&<p role="status">{state.status==='waiting'?`Đang chờ giới hạn yêu cầu: ${state.waitSeconds} giây…`:'Đang xử lý nhóm tiếp theo…'}</p>}
      <progress value={points.length} max={total} aria-label="Tiến độ dự báo theo khoảng"/>
    </div>
    {state.status==='error'&&<div role="alert"><p>{state.error} Các dự báo đã lưu vẫn được giữ.</p>
      {state.canResume!==false&&<button type="button" onClick={onResume}>Tiếp tục từ nhóm chưa hoàn tất</button>}</div>}
    <p>Mỗi điểm dự báo dùng dữ liệu đầu vào ở {request.horizon/60} giờ trước đó. Thực tế là trung bình giờ; không nối đường qua mốc thiếu.</p>
    {values.length>0 ? <figure className="forecast-chart">
      <div className="forecast-chart-legend"><span className="forecast-legend-predicted">● Dự báo trước {request.horizon/60} giờ</span><span className="forecast-legend-actual">● Thực tế — trung bình giờ</span></div>
      <div className="forecast-chart-scroll" tabIndex={0} role="region" aria-label="Biểu đồ theo khoảng, cuộn ngang" onScroll={tooltip.clear}>
        <svg ref={svgRef} viewBox="0 0 960 370" role="group" aria-label="Mực nước dự báo và thực tế theo khoảng, đơn vị mm">
          <text x="15" y="24">Mực nước (mm)</text>
          {Array.from({length:5},(_,index)=>{const value=low+(high-low)*index/4;return <g key={index}><line x1="80" x2="920" y1={y(value)} y2={y(value)} className="forecast-grid-line"/><text x="68" y={y(value)+4} textAnchor="end">{formatWaterLevel(value)}</text></g>;})}
          {Array.from({length:Math.min(total,6)},(_,index)=>{const count=Math.min(total,6);const time=first+Math.round(index/Math.max(count-1,1)*(total-1))*3600000;return <text key={index} x={x(new Date(time).toISOString())} y="330" textAnchor="middle">{timeFormatter.format(new Date(time))}</text>;})}
          {[['actual_water_level_mm','actual'],['predicted_water_level_mm','predicted']].map(([key,kind])=><g key={kind}>
            {segments(key).map((segment,index)=><polyline key={index} className={`forecast-line forecast-line--${kind}`}
              points={segment.map(point=>`${x(point.forecast_at)},${y(point.actual_water_level_mm)}`).join(' ')}/>)}
            {points.filter(point=>point[key]!==null).map(point=><circle key={point.forecast_at} cx={x(point.forecast_at)} cy={y(point[key])} r="2.5" className={`forecast-dot forecast-dot--${kind}`}>
              <title>{`${kind==='actual'?'Thực tế':'Dự báo'} ${formatObservationTime(point.forecast_at)}: ${formatWaterLevel(point[key])} mm`}</title></circle>)}
          </g>)}
          <ChartInteraction control={tooltip} getX={x} getY={y} count={tooltipRows.length} />
        </svg>
      </div><ChartTooltip control={tooltip} svgRef={svgRef} getX={x} /><figcaption>Giờ Việt Nam · Cuộn ngang trên màn hình nhỏ. Biểu đồ gồm tất cả mốc đã xử lý, không chỉ trang đầu.</figcaption>
    </figure> : <p>{state.status==='ready'?'Không có dự báo hoặc quan trắc trong khoảng đã chọn.':'Biểu đồ sẽ hiện khi có dữ liệu.'}</p>}
    {mae!==null&&<p className="prediction-error-summary">Sai số tuyệt đối trung bình: <strong>{formatWaterLevel(mae)} mm</strong> · {comparable.length} mốc có cả dự báo và thực tế.</p>}
    {points.length>0&&<div className="prediction-comparison-scroll prediction-range-table" tabIndex={0} role="region" aria-label="Bảng sai số theo khoảng, cuộn để xem thêm">
      <table className="prediction-comparison"><thead><tr><th scope="col">Thời điểm</th><th scope="col">Dự báo (mm)</th><th scope="col">Thực tế (mm)</th><th scope="col">Sai lệch (mm)</th><th scope="col">Sai số tuyệt đối (mm)</th><th scope="col">Trạng thái</th></tr></thead>
        <tbody>{points.map(point=>{const difference=point.predicted_water_level_mm===null||point.actual_water_level_mm===null?null:point.predicted_water_level_mm-point.actual_water_level_mm;return <tr key={point.forecast_at}>
          <th scope="row">{formatObservationTime(point.forecast_at)}</th><td>{point.predicted_water_level_mm===null?'—':formatWaterLevel(point.predicted_water_level_mm)}</td>
          <td>{point.actual_water_level_mm===null?'—':formatWaterLevel(point.actual_water_level_mm)}</td><td>{difference===null?'—':formatWaterLevel(difference)}</td><td>{difference===null?'—':formatWaterLevel(Math.abs(difference))}</td>
          <td>{point.status==='ready'?(point.reused?'Đã có':'Mới tạo'):reasons[point.status]}</td></tr>;})}</tbody></table>
    </div>}
  </section>;
}
