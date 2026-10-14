import { getStationAppearance } from '../stations.js';

/** @returns {import('react').ReactElement} Icon trang trí giống marker trên bản đồ. */
export default function StationIcon({ stationType }) {
  const { variant, paths } = getStationAppearance(stationType);
  return (
    <span className={`station-marker station-marker--${variant} station-card-icon`} aria-hidden="true">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        {paths.map((path) => <path key={path} d={path} />)}
      </svg>
    </span>
  );
}
