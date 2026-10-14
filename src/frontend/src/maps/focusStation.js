import { getStationCoordinates } from '../stations.js';

/**
 * Đưa trạm vào vùng bản đồ không bị sidebar che, tôn trọng reduced motion.
 *
 * @param {import('maplibre-gl').Map} map Bản đồ đã khởi tạo.
 * @param {object} station Metadata trạm, có tọa độ API theo độ thập phân.
 * @returns {boolean} Có thực hiện focus hay không; false nếu thiếu/sai tọa độ.
 * Sidebar nằm bên trái trên desktop, phía trên trên màn hình hẹp.
 */
export function focusStation(map, station) {
  const center = getStationCoordinates(station);
  if (!center) return false;

  const viewport = map.getContainer().getBoundingClientRect();
  const sidebar = document.getElementById('station-sidebar')?.getBoundingClientRect();
  const padding = { top: 24, bottom: 24, left: 24, right: 24 };
  if (sidebar && viewport.width >= 640) {
    padding.left = Math.min(sidebar.right - viewport.left + 24, viewport.width - 120);
  } else if (sidebar) {
    padding.top = Math.min(sidebar.bottom - viewport.top + 24, viewport.height - 120);
  }
  map.flyTo({
    center, zoom: Math.max(map.getZoom(), 12), padding,
    duration: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 0 : 1400,
  });
  return true;
}
