import { Marker } from 'maplibre-gl';
import { getStationAppearance, getStationCoordinates } from '../stations.js';

/**
 * Tạo marker có thể chọn bằng chuột/bàn phím, ưu tiên trạm nước.
 *
 * @param {import('maplibre-gl').Map} map Map instance đã được khởi tạo.
 * @param {Array<object>} stations Danh sách trạm, location dùng độ thập phân.
 * @param {(station: object) => void} onStationSelect Chọn trạm để mở box và focus map.
 * @returns {{markers: import('maplibre-gl').Marker[], skippedCount: number}}
 * Marker đã gắn vào map và số trạm bị bỏ qua do thiếu/sai tọa độ. Caller cần
 * remove từng marker khi unmount. Không kéo marker; drag map không được coi là click.
 */
export function addStationMarkers(map, stations, onStationSelect) {
  const markers = [];
  let skippedCount = 0;

  for (const station of stations) {
    const coordinates = getStationCoordinates(station);
    if (!coordinates) {
      skippedCount++;
      continue;
    }

    const { variant, label, paths } = getStationAppearance(station.station_type);
    const element = document.createElement('button');
    element.type = 'button';
    element.className = `station-marker station-map-marker station-marker--${variant}`;
    element.setAttribute('aria-label', `${label}: ${station.name}`);
    element.dataset.stationId = station.station_id;
    let pointerStart = null;
    element.addEventListener('pointerdown', (event) => {
      pointerStart = [event.clientX, event.clientY];
    });
    element.addEventListener('click', (event) => {
      event.stopPropagation();
      // Click bằng bàn phím có detail = 0, không phụ thuộc vị trí con trỏ trước đó.
      if (event.detail && pointerStart
        && Math.hypot(event.clientX - pointerStart[0], event.clientY - pointerStart[1]) >= 3) return;
      onStationSelect(station);
    });
    // Chỉ dùng SVG cố định trong mã nguồn; không đưa text từ API vào innerHTML.
    element.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths.map((path) => `<path d="${path}"/>`).join('')}</svg>`;

    const marker = new Marker({ element, anchor: 'center', draggable: false })
      .setLngLat(coordinates)
      .addTo(map);
    markers.push(marker);
  }

  return { markers, skippedCount };
}
