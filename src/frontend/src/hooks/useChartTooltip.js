import { useEffect, useId, useState } from 'react';
import { findNearestTooltipIndex } from '../chartTooltip.js';

/**
 * @param {object[]} rows Timestamp/value pairs tăng dần, chỉ các mốc dữ liệu thật.
 * @param {object} svgRef Ref SVG để đổi tọa độ chính xác kể cả co giãn, letterbox và cuộn.
 * @param {function} getX Timestamp -> tọa độ viewBox.
 * @returns {object} Row đang chọn, ID tooltip và handlers cho hover/tap/focus/phím.
 * Touch chọn khi pointer-up, không chặn pan/cuộn; mouse rời vùng vẽ đóng tooltip.
 */
export function useChartTooltip(rows, svgRef, getX) {
  const tooltipId = useId();
  const [selectedTime, setSelectedTime] = useState(null);
  const [pinned, setPinned] = useState(false);
  const index = rows.findIndex(row => Date.parse(row.timestamp) === selectedTime);
  const row = rows[index] ?? null;

  function selectIndex(nextIndex, keepVisible = false) {
    if (!rows[nextIndex]) return;
    setSelectedTime(Date.parse(rows[nextIndex].timestamp));
    setPinned(keepVisible);
  }

  function selectPointer(event, keepVisible) {
    const svg = svgRef.current;
    const matrix = svg?.getScreenCTM();
    if (!matrix) return;
    const point = svg.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    const local = point.matrixTransform(matrix.inverse());
    selectIndex(findNearestTooltipIndex(rows, local.x, getX), keepVisible);
  }

  function clear() {
    setSelectedTime(null);
    setPinned(false);
  }

  useEffect(() => {
    const onResize = () => setSelectedTime(null);
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  function onKeyDown(event) {
    if (event.key === 'Escape') {
      event.preventDefault();
      event.stopPropagation();
      clear();
      return;
    }
    const nextIndex = event.key === 'ArrowRight' ? Math.min(rows.length - 1, Math.max(index, 0) + 1)
      : event.key === 'ArrowLeft' ? Math.max(0, index - 1)
        : event.key === 'Home' ? 0 : event.key === 'End' ? rows.length - 1 : null;
    if (nextIndex === null) return;
    event.preventDefault();
    selectIndex(nextIndex, true);
  }

  return {
    row, index, tooltipId, clear,
    selectTime: timestamp => { setSelectedTime(Date.parse(timestamp)); setPinned(true); },
    handlers: {
      onPointerMove: event => { if (event.pointerType === 'mouse') selectPointer(event, false); },
      onPointerUp: event => selectPointer(event, event.pointerType !== 'mouse'),
      onPointerLeave: event => { if (event.pointerType === 'mouse' && !pinned) clear(); },
      onPointerCancel: clear,
      onFocus: () => { if (!row) selectIndex(0, true); },
      onBlur: clear,
      onKeyDown,
    },
  };
}
