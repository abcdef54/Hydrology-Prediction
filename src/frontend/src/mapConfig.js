// MapLibre dùng thứ tự [kinh độ, vĩ độ] cho tâm nhìn và các góc của bounds.
export const mapConfig = {
  center: [107.571361, 16.458944],
  zoom: 12,
  // Khung nhìn phục vụ UI, không phải dữ liệu đường biên hành chính.
  bounds: [[101.5, 7.5], [118, 24]],
  styleUrl: 'https://tiles.openfreemap.org/styles/liberty',
  minZoom: 6,
  maxZoom: 19,
  // Vùng ẩn nhãn ngoài khơi phục vụ UI; không biểu diễn đường biên/chủ quyền.
  // Bao các cụm đảo nhỏ ngoài khơi, tránh đất liền và đảo lớn/đảo ven bờ.
  hiddenOffshoreLabelAreas: {
    type: 'MultiPolygon',
    coordinates: [
      [[[110, 15], [115.5, 15], [115.5, 18], [110, 18], [110, 15]]],
      [[[110, 7.5], [116.8, 7.5], [116.8, 13.5], [110, 13.5], [110, 7.5]]],
      [[[117.3, 14.7], [118, 14.7], [118, 15.7], [117.3, 15.7], [117.3, 14.7]]],
      [[[116.3, 20.3], [117.1, 20.3], [117.1, 21.1], [116.3, 21.1], [116.3, 20.3]]],
    ],
  },
};
