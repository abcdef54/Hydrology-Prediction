/**
 * Chuyển tên feature thành lời mô tả; giữ nguyên khóa kỹ thuật trong dữ liệu API.
 * @param {object} feature Feature có name, value, unit, source_station_id.
 * @param {object[]} stations Metadata để tra tên trạm, không hiển thị mã trạm thay tên.
 * @param {number} index Vị trí dùng đặt nhãn an toàn cho feature mới chưa có mô tả.
 * @returns {{label: string, value: number|string, unit: string, sourceName?: string}} Nhãn ngắn và tên nguồn riêng.
 */
export function describePredictionFeature(feature, stations, index) {
  const station = stations.find((item) => item.station_id === feature.source_station_id);
  const waterLag = /^water_level_lag_(\d+)m$/.exec(feature.name);
  const rain = /^rain_\d+_(depth|missing|sum_(\d+)h)$/.exec(feature.name);
  let label = `Thông tin bổ sung ${index + 1}`;
  let flag = false;
  if (feature.name === 'water_level_depth') label = 'Mực nước đầu vào';
  else if (waterLag) {
    const minutes = Number(waterLag[1]);
    label = `Mực nước ${minutes % 60 === 0 ? `${minutes / 60} giờ` : `${minutes} phút`} trước`;
  } else if (feature.name === 'water_level_missing') {
    label = 'Tình trạng dữ liệu nước';
    flag = true;
  } else if (rain) {
    label = rain[1] === 'depth' ? 'Mưa đầu vào'
      : rain[1] === 'missing' ? 'Tình trạng dữ liệu mưa'
        : `Tổng số đo mưa ${rain[2]} giờ`;
    flag = rain[1] === 'missing';
  } else if (feature.name === 'flood_season') {
    return { label: 'Thuộc mùa lũ', value: feature.value === null ? '' : feature.value === 1 ? 'Có' : 'Không', unit: '' };
  }
  const value = feature.value === null ? ''
    : flag ? (feature.value === 1 ? 'Thiếu dữ liệu' : 'Có dữ liệu') : feature.value;
  const unit = flag || feature.unit === 'dimensionless' ? '' : feature.unit ?? 'Chưa rõ đơn vị';
  return { label, value, unit, sourceName: station?.name ?? (rain ? 'Trạm mưa' : '') };
}
