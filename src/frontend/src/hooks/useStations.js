import { useEffect, useState } from 'react';
import { fetchStations, fetchStationReadings } from '../api/stations.js';

/**
 * Tải metadata một lần, làm mới số đo mỗi 10 phút khi tab đang hiển thị.
 *
 * @returns {object} stations/readings, trạng thái tải/lỗi và reload để thử lại.
 * Hủy request và lịch làm mới khi unmount; lỗi một trạm không làm mất bản đồ.
 */
export function useStations() {
  const [stations, setStations] = useState([]);
  const [readings, setReadings] = useState({});
  const [isLoading, setIsLoading] = useState(true);
  const [isLoadingReadings, setIsLoadingReadings] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [reloadCount, setReloadCount] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    let interval;
    let refreshing = false;
    setIsLoading(true);
    setErrorMessage('');

    async function refreshReadings(loadedStations) {
      if (refreshing || controller.signal.aborted) return;
      refreshing = true;
      setIsLoadingReadings(true);
      try {
        const latestReadings = await fetchStationReadings(loadedStations, controller.signal);
        if (!controller.signal.aborted) setReadings(latestReadings);
      } catch (error) {
        if (!controller.signal.aborted) setErrorMessage(error.message);
      } finally {
        refreshing = false;
        if (!controller.signal.aborted) setIsLoadingReadings(false);
      }
    }

    async function load() {
      try {
        const loadedStations = await fetchStations(controller.signal);
        if (controller.signal.aborted) return;
        setStations(loadedStations);
        setReadings({});
        setIsLoading(false);
        refreshReadings(loadedStations);
        interval = setInterval(() => {
          if (document.visibilityState === 'visible') refreshReadings(loadedStations);
        }, 600_000);
      } catch (error) {
        if (controller.signal.aborted) return;
        setErrorMessage(error.message);
        setIsLoading(false);
      }
    }
    load();
    return () => {
      controller.abort();
      clearInterval(interval);
    };
  }, [reloadCount]);

  return {
    stations, readings, isLoading, isLoadingReadings, errorMessage,
    reload: () => setReloadCount((count) => count + 1),
  };
}
