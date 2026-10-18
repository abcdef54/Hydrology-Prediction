"""Plot the two long +24h error runs with one-day context on both sides."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/large_error_events_2026-10-15'
TZ='Asia/Ho_Chi_Minh'


def main():
    p=pd.read_csv(OUT/'forecast_context.csv')
    p['forecast_at']=pd.to_datetime(p.forecast_at,utc=True).dt.tz_convert(TZ)
    p=p[p.horizon_minutes.eq(1440)].set_index('forecast_at').sort_index()
    cases=[('underprediction','2026-09-12 10:00','2026-09-13 11:00','Dự báo thiếu',26),
           ('overprediction','2026-09-13 22:00','2026-09-14 13:00','Dự báo thừa',16)]
    for slug,start,end,label,n in cases:
        a=pd.Timestamp(start,tz=TZ);b=pd.Timestamp(end,tz=TZ)
        lo=a-pd.Timedelta(days=1);hi=b+pd.Timedelta(days=1)
        grid=pd.date_range(lo,hi,freq='h')
        g=p.reindex(grid)
        run=g.loc[a:b]
        paired = run.dropna(subset=['actual', 'water_level_mm'])
        model_mae = paired.ae.mean()
        assert len(run)==n and run.ae.gt(200).all()
        assert (run.error.lt(-200) if slug=='underprediction' else run.error.gt(200)).all()
        fig,axes=plt.subplots(2,1,figsize=(15,8),sharex=True,gridspec_kw={'height_ratios':[2,1]})
        ax,err=axes
        ax.plot(g.index,g.actual,color='#ce4525',lw=2,label='Thực tế (trung bình giờ)')
        ax.plot(g.index,g.water_level_mm,color='#1677b8',lw=1.8,label='Dự báo trước 24 giờ')
        ax.scatter(run.index,run.water_level_mm,color='#1677b8',s=15,zorder=3)
        err.plot(g.index,g.error,color='#1677b8',lw=1.6)
        err.axhline(0,color='#454545',lw=.8)
        for value in [-200,200]:err.axhline(value,color='#ce4525',ls='--',lw=1.3)
        for panel in axes:
            panel.axvspan(a,b,color='#ffc857',alpha=.23)
            panel.axvline(a,color='#9a6a00',ls=':',lw=1)
            panel.axvline(b,color='#9a6a00',ls=':',lw=1)
            panel.grid(alpha=.2);panel.set_xlim(lo,hi)
        ax.set_ylabel('Mực nước (mm)');ax.set_ylim(250,1350)
        err.set_ylabel('Sai lệch (mm)\nDự báo − thực tế');err.set_ylim(-550,550)
        err.set_xlabel('Thời điểm được dự báo · Giờ Việt Nam (UTC+7)')
        ax.legend(loc='upper left')
        err.legend(handles=[Line2D([0],[0],color='#1677b8',label='Sai lệch của mô hình'),Line2D([0],[0],color='#ce4525',ls='--',label='Ngưỡng ±200 mm'),Patch(facecolor='#ffc857',alpha=.35,label='Chuỗi sai số tuyệt đối >200 mm')],loc='upper left',ncol=2,fontsize=9)
        peak=run.ae.idxmax();row=run.loc[peak]
        ax.annotate(f'Sai số lớn nhất trong vùng tô vàng\n{peak:%d/%m %H:%M}\nDự báo: {row.water_level_mm:.2f} mm\nThực tế: {row.actual:.2f} mm\nSai lệch: {row.error:+.2f} mm',
            xy=(peak,row.water_level_mm),xytext=(.64,.95),textcoords='axes fraction',ha='left',va='top',fontsize=10,
            bbox=dict(boxstyle='round,pad=.5',fc='white',ec='#cccccc',alpha=.95),arrowprops=dict(arrowstyle='->',color='#555555'))
        err.xaxis.set_major_locator(mdates.HourLocator(byhour=[0,6,12,18],tz=a.tzinfo))
        err.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m\n%H:%M',tz=a.tzinfo))
        fig.suptitle(f'Dã Viên | {label} trước 24 giờ | {a:%d/%m/%Y %H:%M} – {b:%d/%m/%Y %H:%M}\nVùng tô vàng: {len(paired)} mốc liên tiếp | Sai số tuyệt đối trung bình: {model_mae:.2f} mm | Hiển thị thêm 1 ngày trước và sau',fontsize=14)
        fig.tight_layout(rect=(0,0,1,.93))
        for ext in ['png','pdf','svg']:fig.savefig(OUT/f'september_{slug}_padding_1day.{ext}',dpi=180)
        plt.close(fig)
        g.to_csv(OUT/f'september_{slug}_padding_1day.csv',index_label='target_vietnam')
        print(slug,'window',lo,hi,'highlight',n,'max_error',row.error,'model_mae',model_mae)


if __name__=='__main__':main()
