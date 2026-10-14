import { getStationCoordinates } from '../stations.js';

/**
 * Đưa trạm vào vùng bản đồ giữa các panel khi đủ chỗ, tôn trọng reduced motion.
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
  const detail = document.getElementById('station-detail-panel')?.getBoundingClientRect();
  const padding = { top: 24, bottom: 24, left: 24, right: 24 };
  if (sidebar && viewport.width >= 640) {
    padding.left = Math.min(sidebar.right - viewport.left + 24, viewport.width - 120);
  } else if (sidebar) {
    padding.top = Math.min(sidebar.bottom - viewport.top + 24, viewport.height - 120);
  }
  if (detail) {
    const rightSpace = viewport.right - detail.left + 24;
    const bottomSpace = viewport.bottom - detail.top + 24;
    if (viewport.width >= 640 && viewport.width - padding.left - rightSpace >= 120) {
      padding.right = rightSpace;
    } else if (viewport.width < 640 && viewport.height - padding.top - bottomSpace >= 64) {
      padding.bottom = bottomSpace;
    }
  }
  map.flyTo({
    center, zoom: Math.max(map.getZoom(), 12), padding,
    duration: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 0 : 1400,
  });
  return true;
}
