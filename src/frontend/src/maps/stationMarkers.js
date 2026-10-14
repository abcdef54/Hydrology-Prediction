import { Marker } from 'maplibre-gl';
import { getStationAppearance, getStationCoordinates } from '../stations.js';

/**
 * Tạo marker tĩnh theo tọa độ API, ưu tiên hiển thị trạm nước hơn các loại khác.
 *
 * @param {import('maplibre-gl').Map} map Map instance đã được khởi tạo.
 * @param {Array<object>} stations Danh sách trạm, location dùng độ thập phân.
 * @returns {{markers: import('maplibre-gl').Marker[], skippedCount: number}}
 * Marker đã gắn vào map và số trạm bị bỏ qua do thiếu/sai tọa độ. Caller cần
 * remove từng marker khi unmount. Không gắn popup, click handler hoặc kéo marker.
 */
export function addStationMarkers(map, stations) {
  const markers = [];
  let skippedCount = 0;

  for (const station of stations) {
    const coordinates = getStationCoordinates(station);
    if (!coordinates) {
      skippedCount++;
      continue;
    }

    const { variant, label, paths } = getStationAppearance(station.station_type);
    const element = document.createElement('div');
    element.className = `station-marker station-marker--${variant}`;
    element.setAttribute('role', 'img');
    element.setAttribute('aria-label', `${label}: ${station.name}`);
    element.dataset.stationId = station.station_id;
    // Chỉ dùng SVG cố định trong mã nguồn; không đưa text từ API vào innerHTML.
    element.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths.map((path) => `<path d="${path}"/>`).join('')}</svg>`;

    const marker = new Marker({ element, anchor: 'center', draggable: false })
      .setLngLat(coordinates)
      .addTo(map);
    markers.push(marker);
  }

  return { markers, skippedCount };
}
