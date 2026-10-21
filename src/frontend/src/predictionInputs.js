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
  const forecastRain = /^weather_forecast_rain_(\d+)_(\d+)h(_missing)?$/.exec(feature.name);
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
  } else if (forecastRain) {
    label = forecastRain[3]
      ? `Tình trạng dự báo mưa ${forecastRain[1]}–${forecastRain[2]} giờ`
      : `Dự báo mưa ${forecastRain[1]}–${forecastRain[2]} giờ tới`;
    flag = Boolean(forecastRain[3]);
  } else if (feature.name === 'weather_forecast_age_hours') {
    label = 'Độ trễ bản tin dự báo';
  } else if (feature.name === 'flood_season') {
    return { label: 'Thuộc mùa lũ', value: feature.value === null ? '' : feature.value === 1 ? 'Có' : 'Không', unit: '' };
  }
  const value = feature.value === null ? ''
    : flag ? (feature.value === 1 ? 'Thiếu dữ liệu' : 'Có dữ liệu') : feature.value;
  const unit = flag || feature.unit === 'dimensionless' ? '' : feature.unit ?? 'Chưa rõ đơn vị';
  const defaultSource = feature.name.startsWith('weather_forecast_') ? 'Dự báo thời tiết' : (rain ? 'Trạm mưa' : '');
  return { label, value, unit, sourceName: station?.name ?? defaultSource };
}

/**
 * @param {object} inputs Response auto-fill gốc, gồm features và editable_features.
 * @param {object} drafts Các giá trị chuỗi đang sửa theo tên feature, chỉ ở bộ nhớ frontend.
 * @returns {{overrides: object, errors: object}} Chỉ gửi giá trị khác gốc; null là thiếu, không phải 0.
 * Kiểm tra hữu hạn/float32, cờ 0/1 và mực nước bắt buộc; backend vẫn kiểm tra lại.
 */
export function buildInputOverrides(inputs, drafts) {
  const overrides = {};
  const errors = {};
  for (const feature of inputs?.features ?? []) {
    if (!feature.editable || !inputs.editable_features.includes(feature.name)
      || !Object.hasOwn(drafts, feature.name)) continue;
    const text = drafts[feature.name].trim();
    const isFlag = feature.name.endsWith('_missing') || feature.name === 'flood_season';
    const numeric = /^[+-]?(?:\d+(?:[.,]\d*)?|[.,]\d+)(?:[eE][+-]?\d+)?$/.test(text);
    const value = text === '' ? null : Number(text.replace(',', '.'));
    if (text !== '' && (!numeric || !Number.isFinite(value) || Math.abs(value) > 3.4028234663852886e38)) {
      errors[feature.name] = 'Nhập một số hợp lệ trong phạm vi cho phép.';
    } else if (isFlag && value !== 0 && value !== 1) {
      errors[feature.name] = 'Vui lòng chọn một trạng thái.';
    } else if (feature.name === 'water_level_depth' && value === null) {
      errors[feature.name] = 'Mực nước đầu vào không được để trống.';
    } else if (value !== feature.value) overrides[feature.name] = value;
  }
  return { overrides, errors };
}
