import { useEffect, useState } from 'react';
import { fetchPredictionOptions } from '../api/predictionRange.js';
import { defaultPredictionRange, getPredictionRangeError, parseVietnamHour } from '../predictionRange.js';

/**
 * @param {object} props station là trạm chọn; onRun nhận trạm, [start,end) UTC và horizon;
 * buttonRef phục hồi focus khi quay lại. Chế độ này không tải/hiện/sửa feature.
 * @returns {import('react').ReactElement} Chọn thời gian Việt Nam và một horizon từ API.
 */
export default function PredictionRangeForm({station,onRun,buttonRef,enabled,modelId,modelName}) {
  const [dates,setDates]=useState(defaultPredictionRange);
  const [horizon,setHorizon]=useState(null);
  const [result,setResult]=useState(null);
  const [retry,setRetry]=useState(0);
  const id=station?.station_id;
  useEffect(()=>{
    if(!id||!enabled)return;
    const controller=new AbortController();
    fetchPredictionOptions(id,controller.signal,modelId??null).then(data=>{
      if(!controller.signal.aborted)setResult({id,modelId:modelId??null,data});
    }).catch(error=>{if(!controller.signal.aborted)setResult({id,modelId:modelId??null,error:error.message});});
    return()=>controller.abort();
  },[id,modelId,retry,enabled]);
  const data=result&&result.id===id&&result.modelId===(modelId??null) ? result.data : null;
  const error=result&&result.id===id&&result.modelId===(modelId??null) ? result.error : '';
  const selectedHorizon=data?.forecast_horizons_minutes.includes(horizon)?horizon:data?.forecast_horizons_minutes[0];
  const validation=getPredictionRangeError(dates.start,dates.end);
  return <div className="prediction-range-form" hidden={!enabled}>
    <h4>Dự báo và xem theo khoảng thời gian</h4>
    <p>Mỗi điểm dùng dữ liệu gốc tại mốc đầu vào tương ứng. Kết quả đã có sẽ được dùng lại; còn thiếu sẽ chạy và lưu.</p>
    <div className="prediction-range-fields">
      <div><label htmlFor="range-start">Bắt đầu (giờ Việt Nam)</label>
        <input id="range-start" type="datetime-local" step="3600" value={dates.start} onChange={e=>setDates({...dates,start:e.target.value})}/></div>
      <div><label htmlFor="range-end">Kết thúc (không bao gồm)</label>
        <input id="range-end" type="datetime-local" step="3600" value={dates.end} onChange={e=>setDates({...dates,end:e.target.value})}/></div>
      <div><label htmlFor="range-horizon">Dự báo trước</label><select id="range-horizon" disabled={!data}
        value={selectedHorizon??''} onChange={e=>setHorizon(Number(e.target.value))}>
        {!data&&<option value="">Đang đọc cấu hình…</option>}
        {data?.forecast_horizons_minutes.map(minutes=><option key={minutes} value={minutes}>{minutes/60} giờ</option>)}
      </select></div>
      <button type="button" ref={buttonRef} disabled={!station||!data||Boolean(validation)}
        onClick={()=>onRun({station,start:new Date(parseVietnamHour(dates.start)).toISOString(),
          end:new Date(parseVietnamHour(dates.end)).toISOString(),horizon:selectedHorizon,
          modelId:modelId??null,modelName:modelName??null})}>Dự báo theo khoảng</button>
    </div>
    {validation&&<p role="alert">{validation}</p>}
    {error&&<div role="alert">{error} <button type="button" onClick={()=>{setResult(null);setRetry(value=>value+1);}}>Thử lại cấu hình</button></div>}
    <p>Tối đa 31 ngày mỗi lượt. Ví dụ 00:00 ngày 1 đến 00:00 ngày 2 gồm 24 mốc giờ của ngày 1.</p>
  </div>;
}
