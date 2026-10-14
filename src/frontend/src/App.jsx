import { useMemo, useState } from 'react';
import MapView from './components/MapView.jsx';
import StationSidebar from './components/StationSidebar.jsx';
import { useStations } from './hooks/useStations.js';
import { getStationAppearance } from './stations.js';

export default function App() {
  const { stations, readings, isLoading, isLoadingReadings, errorMessage, reload } = useStations();
  const [stationType, setStationType] = useState('all');
  const [selectedStation, setSelectedStation] = useState(null);
  const orderedStations = useMemo(() => [...stations].sort((first, second) => (
    getStationAppearance(first.station_type).order - getStationAppearance(second.station_type).order
    || first.name.localeCompare(second.name, 'vi')
  )), [stations]);
  const visibleStations = useMemo(() => orderedStations.filter((station) => (
    stationType === 'all' || station.station_type === stationType
  )), [orderedStations, stationType]);

  function changeFilter(type) {
    setStationType(type);
    if (type !== 'all' && selectedStation?.station_type !== type) setSelectedStation(null);
  }

  return (
    <main aria-label="Bản đồ khu vực Huế, Việt Nam" className="relative h-dvh w-full overflow-hidden">
      <MapView stations={visibleStations} selectedStation={selectedStation} />
      <StationSidebar
        stations={orderedStations} visibleStations={visibleStations} readings={readings}
        stationType={stationType} onFilterChange={changeFilter}
        selectedStationId={selectedStation?.station_id}
        onStationSelect={(station) => setSelectedStation({ ...station })}
        isLoading={isLoading} isLoadingReadings={isLoadingReadings}
        errorMessage={errorMessage} onRetry={reload}
      />
    </main>
  );
}
