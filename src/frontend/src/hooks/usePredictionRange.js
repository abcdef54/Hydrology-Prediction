import { useEffect, useRef, useState } from 'react';
import { fetchRangeChunk } from '../api/predictionRange.js';

function waitOneSecond(signal) {
  return new Promise((resolve,reject)=>{
    const abort=()=>{clearTimeout(timer);reject(signal.reason);};
    const timer=setTimeout(()=>{signal.removeEventListener('abort',abort);resolve();},1000);
    signal.addEventListener('abort',abort,{once:true});
    if(signal.aborted)abort();
  });
}

/**
 * @returns {object} state/run/resume/cancel: lần lượt xử lý nhóm 24 giờ, không song song GPU.
 * Mỗi nhóm pin model/pipeline của nhóm đầu. Retry nhóm lỗi dùng lại DB/cache;
 * dừng chỉ hủy chờ và các nhóm chưa gửi, không xóa kết quả đã commit.
 */
export function usePredictionRange() {
  const [state,setState]=useState(null);
  const active=useRef(null);
  useEffect(()=>()=>active.current?.abort(),[]);
  async function run(request, previous=null) {
    if (active.current) return;
    const controller=new AbortController();active.current=controller;
    let current=previous ? {...previous,status:'loading',error:''}
      : {request,status:'loading',points:[],created:0,reused:0,skipped:0,identity:null,error:''};
    setState(current);
    try {
      while(Date.parse(request.start)+current.points.length*3600000<Date.parse(request.end)) {
        const start=Date.parse(request.start)+current.points.length*3600000;
        const end=Math.min(start+24*3600000,Date.parse(request.end));
        let chunk;
        while(!chunk) {
          try {
            chunk=await fetchRangeChunk({station_id:request.station.station_id,
              start:new Date(start).toISOString(),end:new Date(end).toISOString(),horizon_minutes:request.horizon,
              ...(current.identity??{})},controller.signal);
          } catch(error) {
            if(controller.signal.aborted)throw error;
            if(error.status!==429&&error.code!=='INFERENCE_BUSY')throw error;
            const delay=Math.min(120,Math.max(error.retryAfter??60,error.code==='INFERENCE_BUSY'?5:1));
            for(let remaining=Math.ceil(delay);remaining>0;remaining--) {
              setState({...current,status:'waiting',waitSeconds:remaining});
              await waitOneSecond(controller.signal);
            }
            setState(current);
          }
        }
        if (controller.signal.aborted) return;
        current={...current,points:[...current.points,...chunk.points],created:current.created+chunk.created_count,
          reused:current.reused+chunk.reused_count,skipped:current.skipped+chunk.skipped_count,
          identity:{model_version:chunk.model_version,pipeline_version:chunk.pipeline_version}};
        setState(current);
      }
      if (!controller.signal.aborted) setState({...current,status:'ready'});
    } catch(error) {
      if (!controller.signal.aborted) setState({...current,status:'error',canResume:error.code!=='PIPELINE_CHANGED',
        error:error.code==='PIPELINE_CHANGED'?'Mô hình đã thay đổi. Quay lại để bắt đầu khoảng mới.':error.message});
    } finally {if(active.current===controller) active.current=null;}
  }
  function cancel(){active.current?.abort();active.current=null;setState(null);}
  return {state,run,resume:()=>state&&run(state.request,state),cancel};
}
