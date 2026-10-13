import { Marker } from 'maplibre-gl';

const markerIcons = {
  water_level: '<path d="M3 7c2-2 4 2 6 0s4 2 6 0 4 2 6 0M3 12c2-2 4 2 6 0s4 2 6 0 4 2 6 0M3 17c2-2 4 2 6 0s4 2 6 0 4 2 6 0"/>',
  rainfall: '<path d="M7 14H6a4 4 0 0 1 0-8 6 6 0 0 1 11-1 4 4 0 0 1 1 8h-1M8 17l-1 3M13 16l-1 3M18 17l-1 3"/>',
  wind: '<path d="M3 8h12a3 3 0 1 0-3-3M3 12h16a3 3 0 1 1-3 3M3 16h5a3 3 0 1 1-3 3"/>',
  other: '<circle cx="12" cy="12" r="4"/><path d="M12 3v2M12 19v2M3 12h2M19 12h2"/>',
};

const markerAppearance = {
  water_level: { variant: 'water', label: 'Trạm nước' },
  rainfall: { variant: 'rain', label: 'Trạm mưa' },
  wind: { variant: 'other', label: 'Trạm gió' },
};

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
    const { latitude, longitude } = station.location ?? {};
    if (
      !Number.isFinite(latitude) || latitude < -90 || latitude > 90
      || !Number.isFinite(longitude) || longitude < -180 || longitude > 180
    ) {
      skippedCount++;
      continue;
    }

    const type = station.station_type;
    const { variant, label } = markerAppearance[type] ?? { variant: 'other', label: 'Trạm quan trắc' };
    const element = document.createElement('div');
    element.className = `station-marker station-marker--${variant}`;
    element.setAttribute('role', 'img');
    element.setAttribute('aria-label', `${label}: ${station.name}`);
    element.dataset.stationId = station.station_id;
    // Chỉ dùng SVG cố định trong mã nguồn; không đưa text từ API vào innerHTML.
    element.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${markerIcons[type] ?? markerIcons.other}</svg>`;

    const marker = new Marker({ element, anchor: 'center', draggable: false })
      .setLngLat([longitude, latitude])
      .addTo(map);
    markers.push(marker);
  }

  return { markers, skippedCount };
}
