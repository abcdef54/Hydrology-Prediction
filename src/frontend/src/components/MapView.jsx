import { useEffect, useRef, useState } from 'react';
import { AttributionControl, Map, NavigationControl, setWorkerUrl } from 'maplibre-gl';
import workerUrl from 'maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url';
import { mapConfig } from '../mapConfig.js';
import { createMapStyle } from '../maps/mapStyle.js';
import { addStationMarkers } from '../maps/stationMarkers.js';
import { focusStation } from '../maps/focusStation.js';

setWorkerUrl(workerUrl);

/**
 * Hiển thị OpenMapTiles tại Huế, chỉ vẽ biên giới được gắn maritime = 0.
 *
 * @param {object} props stations đã lọc và selectedStation cần focus, hoặc null.
 * @returns {import('react').ReactElement} Bản đồ vector, marker trạm và trạng thái tải/lỗi.
 * Request bị hủy; marker và tài nguyên WebGL được giải phóng khi unmount.
 */
export default function MapView({ stations, selectedStation }) {
  const containerRef = useRef(null);
  const [mapInstance, setMapInstance] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState('');
  const [stationMessage, setStationMessage] = useState('');

  useEffect(() => {
    const controller = new AbortController();
    let map;

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
        map.addControl(new NavigationControl({ showCompass: false }), 'top-right');
        map.addControl(new AttributionControl({ compact: false }), 'bottom-right');
        map.on('error', () => {
          setIsLoading(false);
          setErrorMessage('Không tải được một phần bản đồ. Kiểm tra kết nối Internet.');
        });
        map.once('load', () => setIsLoading(false));
        setMapInstance(map);
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
      map?.remove();
    };
  }, []);

  useEffect(() => {
    if (!mapInstance) return;
    const { markers, skippedCount } = addStationMarkers(mapInstance, stations);
    setStationMessage(skippedCount
      ? `${skippedCount} trạm thiếu hoặc sai tọa độ nên chưa thể hiển thị trên bản đồ.` : '');
    return () => markers.forEach((marker) => marker.remove());
  }, [mapInstance, stations]);

  useEffect(() => {
    if (mapInstance && selectedStation) focusStation(mapInstance, selectedStation);
  }, [mapInstance, selectedStation]);

  return (
    <div className="relative h-full w-full">
      <div ref={containerRef} className="h-full w-full" />
      {(isLoading || errorMessage || stationMessage) && (
        <p
          role="status"
          className="map-status"
        >
          {errorMessage || (isLoading ? 'Đang tải bản đồ…' : stationMessage)}
        </p>
      )}
    </div>
  );
}
