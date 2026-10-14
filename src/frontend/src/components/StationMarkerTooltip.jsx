import { getStationAppearance } from '../stations.js';
import StationReading from './StationReading.jsx';

/**
 * @param {object} props Metadata trạm và số đo đã tải cho sidebar; không gọi API.
 * @returns {import('react').ReactElement} Tooltip số đo gốc, đơn vị và giờ Việt Nam.
 */
export default function StationMarkerTooltip({ station, reading }) {
  return <div className="station-marker-tooltip-body">
    <p className="station-marker-tooltip-type">{getStationAppearance(station.station_type).label}</p>
    <strong>{station.name}</strong>
    <div className="station-reading"><StationReading reading={reading} /></div>
  </div>;
}
