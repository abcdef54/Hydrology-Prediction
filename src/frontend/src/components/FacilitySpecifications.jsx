import { facilitySpecificationLabels } from '../facilities.js';

/**
 * @param {object} props specifications là văn bản gốc API; isLoading/error phân biệt thiếu và lỗi tải.
 * @returns {import('react').ReactElement} Thông số công trình; không tự chuyển đơn vị hoặc sửa số 0.
 */
export default function FacilitySpecifications({ specifications, isLoading, error }) {
  return <div className="station-detail-alarms">
    <h3>Thông số công trình · theo nguồn</h3>
    <dl>
      {Object.entries(facilitySpecificationLabels).map(([field, label]) => (
        <div key={field}>
          <dt>{label}</dt>
          <dd>{isLoading ? 'Đang tải…' : error ? 'Chưa tải được' : specifications?.[field] ?? '—'}</dd>
        </div>
      ))}
    </dl>
  </div>;
}
