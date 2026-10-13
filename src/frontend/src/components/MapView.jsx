import { useEffect, useRef, useState } from 'react';
import { AttributionControl, Map, NavigationControl, setWorkerUrl } from 'maplibre-gl';
import workerUrl from 'maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url';
import { mapConfig } from '../mapConfig.js';
import { createMapStyle } from '../maps/mapStyle.js';
import { fetchStations } from '../api/stations.js';
import { addStationMarkers } from '../maps/stationMarkers.js';

setWorkerUrl(workerUrl);

/**
 * Hiển thị OpenMapTiles tại Huế, chỉ vẽ biên giới được gắn maritime = 0.
 *
 * @returns {import('react').ReactElement} Bản đồ vector, marker trạm và trạng thái tải/lỗi.
 * Request bị hủy; marker và tài nguyên WebGL được giải phóng khi unmount.
 */
export default function MapView() {
  const containerRef = useRef(null);
  const [isLoading, setIsLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState('');
  const [stationMessage, setStationMessage] = useState('Đang tải vị trí các trạm…');

  useEffect(() => {
    const controller = new AbortController();
    let map;
    let markers = [];

    /** @returns {Promise<void>} Hiển thị marker hoặc thông báo lỗi; giữ bản đồ nền. */
    async function loadStationMarkers() {
      try {
        const stations = await fetchStations(controller.signal);
        if (controller.signal.aborted) return;

        const markerResult = addStationMarkers(map, stations);
        markers = markerResult.markers;
        if (markerResult.skippedCount > 0) {
          setStationMessage(`${markerResult.skippedCount} trạm thiếu hoặc sai tọa độ nên chưa thể hiển thị.`);
        } else {
          setStationMessage(stations.length ? '' : 'Chưa có trạm được bật để hiển thị.');
        }
      } catch (error) {
        if (controller.signal.aborted) return;
        setStationMessage(`Không tải được vị trí các trạm. ${error.message}`);
      }
    }

    async function initializeMap() {
      try {
        // Lọc trước khi tạo map để style gốc không hiển thị dù chỉ trong lúc tải.
        const response = await fetch(mapConfig.styleUrl, {
          signal: AbortSignal.any([controller.signal, AbortSignal.timeout(15_000)]),
        });
        if (!response.ok) {
          throw new Error(`Không tải được cấu hình bản đồ (HTTP ${response.status}).`);
        }

        const style = createMapStyle(
          await response.json(),
          mapConfig.bounds,
          mapConfig.hiddenOffshoreLabelAreas,
        );
        if (controller.signal.aborted) return;

        map = new Map({
          container: containerRef.current,
          style,
          center: mapConfig.center,
          zoom: mapConfig.zoom,
          minZoom: mapConfig.minZoom,
          maxZoom: mapConfig.maxZoom,
          maxBounds: mapConfig.bounds,
          renderWorldCopies: false,
          dragRotate: false,
          pitchWithRotate: false,
          attributionControl: false,
        });
        map.touchZoomRotate.disableRotation();
        map.addControl(new NavigationControl({ showCompass: false }), 'top-left');
        map.addControl(new AttributionControl({ compact: false }), 'bottom-right');
        map.on('error', () => {
          setIsLoading(false);
          setErrorMessage('Không tải được một phần bản đồ. Kiểm tra kết nối Internet.');
        });
        map.once('load', () => setIsLoading(false));
        loadStationMarkers();
      } catch (error) {
        if (controller.signal.aborted) return;
        setIsLoading(false);
        setErrorMessage(
          error.name === 'TimeoutError'
            ? 'Nguồn bản đồ phản hồi quá chậm. Vui lòng tải lại trang.'
            : 'Không tải được bản đồ. Kiểm tra kết nối Internet và tải lại trang.',
        );
      }
    }

    initializeMap();
    return () => {
      controller.abort();
      markers.forEach((marker) => marker.remove());
      map?.remove();
    };
  }, []);

  return (
    <div className="relative h-full w-full">
      <div ref={containerRef} className="h-full w-full" />
      {(isLoading || errorMessage || stationMessage) && (
        <p
          role="status"
          className="absolute top-3 left-1/2 z-10 max-w-[75%] -translate-x-1/2 rounded bg-white px-3 py-2 text-sm text-slate-700 shadow"
        >
          {errorMessage || (isLoading ? 'Đang tải bản đồ…' : stationMessage)}
        </p>
      )}
    </div>
  );
}
