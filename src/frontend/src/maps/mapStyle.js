/**
 * Nhận diện nhãn địa hình dưới biển đang được nguồn đưa vào water_name.
 *
 * @returns {Array} Biểu thức boolean cho các loại nước biển có tên chứa thuật
 * ngữ núi ngầm/sống núi/rãnh biển. Chỉ dùng trong water_name, không lọc tên đất liền.
 */
function createUnderseaLabelMatch() {
  const terms = [
    '海山', '海脊', '海谷', '海沟', '海溝', '海盆', '海岭', '海嶺', '海丘',
    'seamount', 'ridge', 'trough', 'trench', 'guyot', 'knoll',
  ];
  const names = ['name', 'name:en', 'name_en', 'name:latin', 'name:nonlatin'];
  return [
    'all',
    ['match', ['get', 'class'], ['ocean', 'sea', 'bay', 'strait'], true, false],
    [
      'let',
      'underseaName',
      ['downcase', ['concat', ...names.map((name) => ['coalesce', ['get', name], ''])]],
      ['any', ...terms.map((term) => ['in', term, ['var', 'underseaName']])],
    ],
  ];
}

/**
 * Áp quy tắc biên giới và nhãn địa danh lên style OpenMapTiles trước khi hiển thị.
 *
 * @param {object} style Style JSON phiên bản 8, chứa các lớp source-layer boundary.
 * @param {number[][]} bounds Góc tây nam/đông bắc theo [kinh độ, vĩ độ].
 * @param {object} hiddenLabelAreas GeoJSON MultiPolygon của các cụm đảo ngoài
 * khơi cần ẩn nhãn; chỉ là vùng lọc giao diện, không phải đường biên địa lý.
 * @returns {object} Style mới, giữ bộ lọc gốc và yêu cầu maritime === 0 cho mọi
 * lớp boundary. Feature thiếu maritime cũng bị loại. Nhãn Biển Đông/Vịnh Bắc Bộ
 * được thay trọn vẹn bằng tiếng Việt; nhãn trong vùng ngoài khơi được ẩn.
 * Nhãn địa hình dưới biển được ẩn cả ngoài vùng đảo. Các nhãn đất liền, hồ và
 * sông ngoài vùng lọc giữ nguyên, kể cả địa danh Trung Quốc.
 * Nguồn vector chỉ yêu cầu các tiles giao với khung bounds.
 * @throws {Error} Nếu style không đúng cấu trúc hoặc không có lớp boundary.
 */
export function createMapStyle(style, bounds, hiddenLabelAreas) {
  if (style?.version !== 8 || !style.sources || !Array.isArray(style.layers)) {
    throw new Error('Nguồn bản đồ không trả về style vector hợp lệ.');
  }

  if (!style.layers.some((layer) => layer['source-layer'] === 'boundary')) {
    throw new Error('Không xác định được lớp biên giới để ẩn đường biên trên biển.');
  }

  const sourceBounds = [...bounds[0], ...bounds[1]];
  const sources = Object.fromEntries(
    Object.entries(style.sources).map(([name, source]) => [
      name,
      source.type === 'vector' ? { ...source, bounds: sourceBounds } : source,
    ]),
  );

  const layers = style.layers.map((layer) => {
    if (layer['source-layer'] === 'boundary') {
      const landOnlyFilter = ['==', ['get', 'maritime'], 0];
      return {
        ...layer,
        filter: layer.filter ? ['all', layer.filter, landOnlyFilter] : landOnlyFilter,
      };
    }

    if (layer.type !== 'symbol' || layer.layout?.['text-field'] === undefined) {
      return layer;
    }

    let textField = ['case', ['within', hiddenLabelAreas], '', layer.layout['text-field']];

    if (layer['source-layer'] === 'water_name') {
      const englishName = ['coalesce', ['get', 'name:en'], ['get', 'name_en'], ''];
      textField = [
        'case',
        ['all', ['==', ['get', 'class'], 'sea'], ['==', englishName, 'South China Sea']],
        'Biển Đông',
        [
          'all',
          ['==', ['get', 'class'], 'bay'],
          ['match', englishName, ['Gulf of Tonkin', 'Gulf of Tokin'], true, false],
        ],
        'Vịnh Bắc Bộ',
        createUnderseaLabelMatch(),
        '',
        textField,
      ];
    }

    return { ...layer, layout: { ...layer.layout, 'text-field': textField } };
  });

  return { ...style, sources, layers };
}
