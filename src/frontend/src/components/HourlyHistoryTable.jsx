import StationIcon from './StationIcon.jsx';
import { formatObservationTime, getStationAppearance } from '../stations.js';
import { formatHistoryDate, getHistoryHours } from '../history.js';

const valueFormatter = new Intl.NumberFormat('vi-VN', { maximumFractionDigits: 3 });

/**
 * @param {object} props data là một trang API đã xác thực, có 24 ô cho từng trạm/ngày.
 * eligibleStations là metadata có quyền dự báo; selection là trạm/mốc đang chọn;
 * onSelectInputs nhận metadata trạm và timestamp ô, không tự chạy inference.
 * @returns {import('react').ReactElement} Bảng cuộn ngang/dọc, giữ đầu cột và thông tin trạm.
 * Ngày hôm nay đơn lẻ hiện đến giờ as_of; ô thiếu/future hiện — và có chú thích riêng.
 */
export default function HourlyHistoryTable({ data, eligibleStations = [], selection, onSelectInputs }) {
  const hours = getHistoryHours(data);
  return (
    <div className="history-table-scroll" tabIndex={0} role="region" aria-label="Bảng lịch sử, cuộn ngang để xem các giờ">
      <table className="history-table">
        <caption className="sr-only">Mực nước trung bình giờ kết thúc tại mốc giờ Việt Nam; mưa là tổng các số đo trong giờ; gió là số đo đúng giờ. Thiếu dữ liệu hiển thị dấu gạch ngang.</caption>
        <thead>
          <tr>
            <th scope="col" className="history-date-column">Ngày</th>
            <th scope="col" className="history-type-column">Loại trạm</th>
            <th scope="col" className="history-name-column">Tên trạm</th>
            <th scope="col" className="history-unit-column">Đơn vị</th>
            {hours.map((hour) => <th scope="col" key={hour}>{String(hour).padStart(2, '0')}:00</th>)}
          </tr>
        </thead>
        <tbody>
          {data.rows.map((row) => (
            <tr key={`${row.date}/${row.station_id}`}>
              <td className="history-date-column">{formatHistoryDate(row.date)}</td>
              <td className="history-type-column">{getStationAppearance(row.station_type).label}</td>
              <th scope="row" className="history-name-column">
                <span className="history-station-name"><StationIcon stationType={row.station_type} />{row.name}</span>
                {row.station_type === 'wind' && <small>Tốc độ gió · Số đo gốc</small>}
              </th>
              <td className="history-unit-column" title={row.unit ?? 'Chưa rõ đơn vị'}>{row.unit ?? '?'}</td>
              {hours.map((hour) => {
                const cell = row.cells[hour];
                const predictionStation = eligibleStations.find((station) => station.station_id === row.station_id);
                const canSelect = predictionStation && cell.status !== 'future';
                const isSelected = selection?.station.station_id === row.station_id
                  && selection.referenceTime === cell.at;
                const content = cell.status === 'observed' ? valueFormatter.format(cell.value) : '—';
                const description = cell.status === 'future' ? 'Chưa tới giờ'
                  : cell.status === 'missing' ? (row.station_type !== 'wind' ? 'Không có dữ liệu trong giờ kết thúc tại mốc này' : 'Không có số đo đúng mốc giờ')
                    : row.station_type === 'water_level' ? `Mực nước trung bình trong 1 giờ kết thúc lúc ${formatObservationTime(cell.at)}` : row.station_type === 'rainfall' ? `Tổng các số đo mưa trong 1 giờ kết thúc lúc ${formatObservationTime(cell.at)}` : `Đo lúc ${formatObservationTime(cell.at)}`;
                return (
                  <td key={hour} className={`history-cell history-cell--${cell.status}`} title={description}>
                    {canSelect ? <button type="button" className="history-input-cell"
                      aria-pressed={isSelected}
                      aria-label={`Chọn đầu vào ${row.name}, ${formatObservationTime(cell.at)}`}
                      onClick={() => onSelectInputs(predictionStation, cell.at)}>{content}</button>
                      : <span aria-label={description}>{content}</span>}
                  </td>
                );
              })}
            </tr>
          ))}
          {data.rows.length === 0 && (
            <tr><td className="history-empty" colSpan={hours.length + 4}>Không có trạm khớp bộ lọc đã chọn.</td></tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
