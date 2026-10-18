"""Plot observed monthly station rain sums, complete-day rain means and Da Vien water.

Missing rain is never treated as dry. Fixed geographic groups require all member
stations to have >=90% observed readings in a month; network uses eligible stations.
"""
import json
from pathlib import Path
import calendar
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/monthly_rain_water_2025-11_to_2026-10-15'
WEST=['750062','750007','750003','750011']
SOUTH=['750010','228005','750066']
TZ='Asia/Ho_Chi_Minh'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=Path('/tmp/hydrology-monthly-rain')
    metadata=json.loads((source/'manifest.json').read_text())
    cutoff=pd.Timestamp(metadata['cutoff_utc']).tz_convert(TZ)
    start=pd.Timestamp('2025-11-01',tz=TZ)
    grid=pd.date_range(start,cutoff,freq='h')
    hours=pd.Series(1,index=grid).groupby(grid.strftime('%Y-%m')).sum()
    days=pd.read_csv(source/'daily_station_rain.csv',dtype={'station_id':str})
    days['month']=days.day.str[:7]
    stations=pd.read_csv(source/'stations.csv',dtype={'id':str})
    agg=days.groupby(['station_id','month']).agg(valid_records=('valid_records','sum'),observed_sum=('observed_sum',lambda s:s.sum(min_count=1)),complete_hours=('complete_hours','sum')).reset_index()
    daily=days[days.complete_hours.eq(24)].groupby(['station_id','month']).agg(complete_days=('day','size'),mean_complete_day_rain=('complete_hour_sum','mean')).reset_index()
    idx=pd.MultiIndex.from_product([stations.id,hours.index],names=['station_id','month'])
    agg=agg.set_index(['station_id','month']).reindex(idx).reset_index().merge(daily,how='left',on=['station_id','month'])
    agg['expected_records']=agg.month.map(hours)*6
    agg['coverage_pct']=agg.valid_records.fillna(0)/agg.expected_records*100
    agg['eligible']=agg.coverage_pct.ge(90)
    agg=agg.merge(stations[['id','name']],left_on='station_id',right_on='id',how='left')
    agg.to_csv(OUT/'station_month_rain.csv',index=False)
    rows=[]
    for month in hours.index:
        for name,ids in [('A Luoi (4 model stations)',WEST),('South (3 stations)',SOUTH),('Network (eligible stations)',stations.id.tolist())]:
            part=agg[agg.month.eq(month)&agg.station_id.isin(ids)]
            good=part[part.eligible]
            usable=not good.empty and (name.startswith('Network') or len(good)==len(ids))
            rows.append(dict(month=month,group=name,eligible_stations=len(good),total_stations=len(ids),coverage_pct=part.coverage_pct.mean(),mean_observed_monthly_sum=good.observed_sum.mean() if usable else np.nan,mean_complete_day_rain=good.mean_complete_day_rain.mean() if usable else np.nan,complete_days_per_station=good.complete_days.mean() if usable else np.nan,partial_month=(month==cutoff.strftime('%Y-%m'))))
    monthly=pd.DataFrame(rows);monthly.to_csv(OUT/'monthly_rain.csv',index=False)
    # Versioned PostgreSQL snapshot created earlier at the identical hourly cutoff.
    rawpath=Path('/tmp/hydrology-training-snapshot-2026-10-15/water_level_observations.csv.gz')
    water=pd.read_csv(rawpath)
    water['time_point']=pd.to_datetime(water.time_point,utc=True)
    b=water.set_index('time_point').depth.resample('1h',closed='right',label='right')
    hourly=b.mean().where(b.count().eq(6));hourly=hourly[(hourly.index>=start)&(hourly.index<=cutoff)]
    local=hourly.index.tz_convert(TZ)
    wm=hourly.groupby(local.strftime('%Y-%m')).agg(['mean','count','min','max']).reindex(hours.index)
    wm['expected_hours']=hours;wm['coverage_pct']=wm['count']/hours*100
    wm.index.name='month';wm.to_csv(OUT/'monthly_water_da_vien.csv')
    labels=[pd.Timestamp(m+'-01').strftime('%m/%Y')+('*' if m==cutoff.strftime('%Y-%m') else '') for m in hours.index]
    x=np.arange(len(labels));colors=['#367b40','#8a4cad','#1777b5'];groupnames=monthly.group.unique()
    fig,axes=plt.subplots(4,1,figsize=(15,14),sharex=True,gridspec_kw={'height_ratios':[1.2,1.1,1.2,.8]})
    for i,(name,color) in enumerate(zip(groupnames,colors)):
        part=monthly[monthly.group.eq(name)].set_index('month').reindex(hours.index)
        axes[0].bar(x+(i-1)*.24,part.mean_observed_monthly_sum,width=.23,color=color,label=name)
        axes[1].plot(x,part.mean_complete_day_rain,color=color,marker='o',label=name)
        axes[3].plot(x,part.coverage_pct,color=color,marker='o',label=name)
    axes[0].set_ylabel('Mean monthly source sum\nper eligible station (stored mm)')
    axes[1].set_ylabel('Mean complete-day source sum\nper eligible station (stored mm)')
    axes[2].plot(x,wm['mean'],color='#ce4525',marker='o',lw=2,label='Da Vien: mean of complete hourly means')
    for xx,yy in zip(x,wm['mean']):
        if pd.notna(yy):axes[2].annotate(f'{yy:.1f}',(xx,yy),xytext=(0,9),textcoords='offset points',ha='center',fontsize=9)
    axes[2].set_ylabel('Mean water level\n(stored mm)');axes[2].margins(y=.12)
    axes[3].plot(x,wm.coverage_pct,color='#ce4525',marker='s',label='Da Vien water')
    axes[3].set_ylabel('Coverage (%)');axes[3].set_ylim(0,108)
    for ax in axes:
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(loc='upper left',fontsize=9,ncol=2) if ax in (axes[2], axes[3]) else None
    axes[-1].set_xticks(x,labels);axes[-1].set_xlabel('* October is partial; dates in Vietnam time (UTC+7)')
    fig.suptitle(f'Monthly rain observations and Da Vien water level | Nov 2025 – {cutoff:%d Oct %Y %H:%M}\nRain: observed source sums, not independently verified accumulation; missing is not zero',fontsize=14)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='upper center', bbox_to_anchor=(.5,.94), ncol=3, frameon=False)
    fig.tight_layout(rect=(0,.04,1,.91))
    fig.text(.05,.015,'Fixed groups require all stations ≥90% coverage. Network station membership can vary by month.\nDaily rain uses only complete 24-hour days. November water starts mid-month; October totals are not full-month totals.',fontsize=10)
    for ext in ['png','pdf','svg']:fig.savefig(OUT/f'monthly_rain_and_water.{ext}',dpi=170)
    plt.close(fig)
    # Preserve all 55 station histories as a heatmap, with missing shown distinctly.
    hm=agg.pivot(index='name',columns='month',values='mean_complete_day_rain').reindex(columns=hours.index).sort_index()
    cmap=plt.get_cmap('YlGnBu').copy();cmap.set_bad('#d9d9d9')
    fig,ax=plt.subplots(figsize=(13,17));im=ax.imshow(np.ma.masked_invalid(hm.to_numpy()),aspect='auto',cmap=cmap)
    ax.set_xticks(x,labels,rotation=45,ha='right');ax.set_yticks(np.arange(len(hm)),hm.index,fontsize=8)
    ax.set_title('All rain stations: mean source sum per complete day\nGrey = no complete days; station monthly coverage is in CSV')
    fig.colorbar(im,ax=ax,label='Stored mm per complete day (source-sum convention)',shrink=.5)
    fig.tight_layout();fig.savefig(OUT/'all_stations_daily_rain_heatmap.png',dpi=170);plt.close(fig)
    def table(headers,rows):return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)
    table_rows=[]
    for month in hours.index:
        rr=monthly[monthly.month.eq(month)].set_index('group')
        fmt=lambda v:'—' if pd.isna(v) else f'{v:.2f}'
        table_rows.append([month,*[fmt(rr.loc[g,'mean_observed_monthly_sum']) for g in groupnames],fmt(wm.loc[month,'mean']),f'{wm.loc[month,"coverage_pct"]:.1f}%',str(rr.loc[groupnames[-1],'eligible_stations'])+'/55'])
    report=f'''# Mưa theo tháng và mực nước Dã Viên

Khoảng yêu cầu: **01/11/2025 đến {cutoff:%d/%m/%Y %H:%M}**, giờ Việt Nam. PostgreSQL chỉ đọc; không huấn luyện hay đổi model/dataset trong bước này.

## Ý nghĩa số liệu

- Panel 1: tổng các số đo mưa theo tháng tại từng trạm, sau đó trung bình giữa các trạm đủ độ phủ. Không cộng lượng mưa giữa trạm, không nội suy thành mưa lưu vực.
- Panel 2: trung bình tổng số đo nguồn của những ngày đủ 24 giờ × 6 giá trị/giờ ở từng trạm, rồi trung bình giữa các trạm đủ độ phủ tháng. Giúp so tháng khác độ dài hoặc tháng 10 còn dở; vẫn có thiên lệch nếu mất dữ liệu đúng ngày mưa.
- Độ phủ: số giá trị hợp lệ so với lịch 10 phút toàn tháng, hoặc phần tháng đã trôi qua. Trạm được dùng cho mưa tổng hợp phải đạt ≥90%. Nhóm cố định chỉ hiện khi tất cả thành viên đủ điều kiện; toàn mạng dùng các trạm đủ điều kiện nên thành phần có thể đổi.
- A Lưới gồm Hồng Thái, Nhâm, TT A Lưới, Hồ A Lá. Phía nam gồm Giang Hải, Cảng Chân Mây, Đỉnh Bạch Mã. Không mặc định các nhóm này là lưu vực đóng góp của Dã Viên.
- Quy tắc giờ (t−1h,t]; số mưa là tổng bản ghi nguồn theo quy tắc dataset hiện tại. Khoảng tích lũy nguồn chưa xác minh; chưa coi đây là tổng mưa khí tượng đã kiểm định.
- Mực nước tháng là trung bình các giá trị trung bình giờ có đủ 6 số đo, không lấy trực tiếp trung bình tất cả bản ghi 10 phút. Có thêm min/max trung bình giờ trong CSV để kiểm tra.
- Tháng 11/2025: Dã Viên bắt đầu có dữ liệu từ giữa tháng; không đại diện trọn tháng. Tháng 10/2026: chỉ đến thời điểm hiện tại. Mưa tổng tháng 10 không so trực tiếp với tổng trọn tháng trước.
- Không lấy thiếu làm 0. Bảng hiện — khi nhóm không đủ điều kiện; xem CSV từng trạm và heatmap để không che mất dữ liệu còn lại.

## Tổng hợp

Đơn vị các giá trị là mm theo metadata dự án; xem các giới hạn diễn giải nêu trên.

'''
    report+=table(['Tháng','Mưa TB tổng tháng/trạm: A Lưới','Mưa TB tổng tháng/trạm: phía nam','Mưa TB tổng tháng/trạm: toàn mạng','Mực nước TB Dã Viên','Độ phủ nước','Trạm mưa mạng đủ điều kiện'],table_rows)
    report+='\n\n## Mean complete-day source sums\n\n'+table(['Month',*groupnames],[[month,*[('—' if pd.isna(v) else f'{v:.2f}') for v in monthly[monthly.month.eq(month)].set_index('group').reindex(groupnames).mean_complete_day_rain]] for month in hours.index])
    report+='''

## Sử dụng để cân nhắc train/validation/test

Chưa chốt lại split ở bước này. Mưa trung bình tháng giúp thấy tính mùa và khác biệt không gian, nhưng không đủ để chọn tập: còn cần phân phối mực nước, tốc độ tăng/giảm và các đợt cực trị. Cần giữ chia theo thời gian, purge theo horizon tối đa và tránh để một đợt mưa/lũ sát ranh giới làm đánh giá quá lạc quan. Tập validation nên có cả ổn định và biến động mạnh; tập test cuối cần giữ chưa dùng để chọn trọng số/penalty. Dữ liệu tháng 9–10 đã được xem nhiều nên kết quả trên đó là thăm dò.

## Figure

![Mưa và mực nước tháng](monthly_rain_and_water.png)

[PDF](monthly_rain_and_water.pdf) · [SVG](monthly_rain_and_water.svg)

![Toàn bộ 55 trạm](all_stations_daily_rain_heatmap.png)

## Dữ liệu

[Mưa theo tháng](monthly_rain.csv) · [Mưa từng trạm và độ phủ](station_month_rain.csv) · [Mực nước tháng](monthly_water_da_vien.csv)

Script: `research/plot_monthly_rain_water.py`. Snapshot mưa: `/tmp/hydrology-monthly-rain`; nước: `/tmp/hydrology-training-snapshot-2026-10-15/water_level_observations.csv.gz` (cùng mốc cutoff giờ).
'''
    (OUT/'phan_tich_theo_thang.md').write_text(report)
    (OUT/'manifest.json').write_text(json.dumps(dict(metadata,coverage_min_pct=90,groups={'west':WEST,'south':SOUTH},water_source=str(rawpath),water_aggregation='mean of hourly means with exactly six valid readings'),ensure_ascii=False,indent=2))
    print('WATER\n'+wm.round(2).to_string())
    print('RAIN\n'+monthly.round(2).to_string(index=False))
    print('OUTPUT',OUT)


if __name__=='__main__':main()
