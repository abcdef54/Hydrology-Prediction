"""Create a combined histogram and Vietnamese Markdown from exported error statistics."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


def table(headers, rows):
    return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(str(v) for v in row)+' |' for row in rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    args=parser.parse_args();out=args.folder
    stats=pd.read_csv(out/'statistics.csv');p=pd.read_csv(out/'forecast_points.csv');bins=pd.read_csv(out/'histogram_counts.csv');meta=json.loads((out/'manifest.json').read_text())
    valid=p.dropna(subset=['actual']);start=pd.Timestamp(meta['start_local']);cutoff=pd.Timestamp(meta['requested_cutoff_local']);end=cutoff-pd.Timedelta(seconds=1)
    title=f'{start:%d/%m/%Y} – {end:%d/%m/%Y}'
    edges=np.r_[bins.left_mm.unique(),bins.right_mm.max()];edges.sort()
    ymax=bins['count'].max()
    fig,axes=plt.subplots(2,3,figsize=(18,10),sharex=True,sharey=True)
    for ax,r in zip(axes.flat,stats.itertuples()):
        e=valid.loc[valid.horizon_minutes.eq(r.horizon_hours*60),'abs_error']
        counts,_=np.histogram(e,bins=edges);assert counts.sum()==r.valid_points
        ax.hist(e,bins=edges,color='#1677b8',edgecolor='white',linewidth=.6)
        ax.axvline(200,color='#ce4525',linestyle='--',linewidth=1.4)
        ax.axvline(r.mae_mm,color='#308047',linewidth=1.4)
        ax.set(title=f'+{r.horizon_hours} hours',xlim=(0,edges[-1]),ylim=(0,ymax*1.08),xticks=np.arange(0,edges[-1]+1,100))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True));ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        ax.text(.97,.95,f'n = {r.valid_points:,}\nMAE = {r.mae_mm:.2f} mm\nMin = {r.min_mm:.2f} mm\nMax = {r.max_mm:.2f} mm\n>200 mm: {r.above_200} ({r.above_200_pct:.2f}%)',transform=ax.transAxes,ha='right',va='top',fontsize=10)
    ax=axes.flat[5];ax.axis('off')
    first=pd.to_datetime(valid.forecast_at,utc=True).min().tz_convert('Asia/Ho_Chi_Minh');last=pd.to_datetime(valid.forecast_at,utc=True).max().tz_convert('Asia/Ho_Chi_Minh')
    ax.text(.08,.85,f'Da Vien water-level forecasts\n\nRequested: {title}\nScored targets (Vietnam time):\n{first:%d %b %Y %H:%M} – {last:%d %b %Y %H:%M}\n\nBin width: 25 mm\nAll panels use the same axis scales.\nIncomplete observed hours excluded.\nSaved forecasts may be retrospective.',transform=ax.transAxes,va='top',fontsize=12,linespacing=1.5)
    fig.suptitle(f'Absolute forecast error distributions | {title}',fontsize=18)
    fig.legend(handles=[Patch(facecolor='#1677b8',label='Sample count'),Line2D([0],[0],color='#308047',label='MAE'),Line2D([0],[0],color='#ce4525',linestyle='--',label='200 mm threshold')],loc='upper center',bbox_to_anchor=(.5,.945),ncol=3,frameon=False,fontsize=12)
    fig.supxlabel('Absolute error |forecast − observed| (mm)',fontsize=13);fig.supylabel('Number of samples',fontsize=13)
    fig.tight_layout(rect=(.02,.03,1,.90))
    for ext in ['png','svg','pdf']:fig.savefig(out/f'absolute_errors_all_horizons.{ext}',dpi=180)
    plt.close(fig)
    primary=[];partitions=[];coverage=[]
    for r in stats.itertuples():
        primary.append([f'+{r.horizon_hours} giờ',r.valid_points,r.above_200,f'{r.above_200_pct:.2f}%',f'{r.min_mm:.2f}',f'{r.max_mm:.2f}',f'{r.mae_mm:.2f}'])
        partitions.append([f'+{r.horizon_hours} giờ',r.le_mae,f'{r.le_mae_pct:.2f}%',r.gt_mae,f'{r.gt_mae_pct:.2f}%'])
        coverage.append([f'+{r.horizon_hours} giờ',r.saved_points,r.excluded_truth,r.first_target,r.last_target])
    n=int(stats.valid_points.sum());bad=int(stats.above_200.sum())
    md=f'''# Phân tích sai số dự báo Dã Viên: {title}

## Phạm vi và phương pháp

- Lọc theo **thời điểm được dự báo**, từ {start:%d/%m/%Y %H:%M} đến trước {cutoff:%d/%m/%Y %H:%M}, giờ Việt Nam (UTC+7).
- Thời điểm thực sự có thể chấm điểm: **{first:%d/%m/%Y %H:%M} – {last:%d/%m/%Y %H:%M}**. Phạm vi từng horizon có thể khác nhau; xem bảng độ phủ.
- Thực tế là trung bình mực nước trong **(t−1 giờ, t]**; chỉ chấm giờ có đủ **6 số đo hợp lệ**. Thiếu không thay bằng 0.
- Sai số tuyệt đối = **|dự báo − thực tế|**, đơn vị **mm**. MAE là trung bình sai số tuyệt đối.
- Điều kiện vượt ngưỡng là **>200 mm**. Phân chia ≤ MAE và > MAE dùng MAE riêng từng horizon, trước khi làm tròn.
- Tổng hợp có **{n:,} điểm dự báo**, **{bad} điểm ({100*bad/n:.2f}%)** sai số >200 mm. Một thời điểm có thể được tính ở nhiều horizon; đây không phải số sự kiện độc lập.
- Dùng cùng snapshot PostgreSQL ngày 15/10/2026, cùng model và pipeline như báo cáo trước. Dự báo lưu có thể được tạo hồi cứu.
- **Tháng 03/2026 và tháng 11/2025 nằm trong khoảng train theo manifest dataset local** (train kết thúc 25/06/2026). Chưa xác minh tuyệt đối nguồn train của checkpoint; không coi thống kê này là đánh giá ngoài mẫu hoặc so trực tiếp với tháng 09–10 như cùng điều kiện.

## Thống kê theo horizon

'''
    md+=table(['Horizon','Số mẫu hợp lệ','Số mẫu >200 mm','Tỷ lệ >200 mm','Min (mm)','Max (mm)','MAE (mm)'],primary)
    md+='\n\n## Phân bố so với MAE riêng của từng horizon\n\n'+table(['Horizon','Số mẫu ≤ MAE','Tỷ lệ ≤ MAE','Số mẫu > MAE','Tỷ lệ > MAE'],partitions)
    md+='\n\n## Độ phủ dự báo và điểm bị loại\n\n'+table(['Horizon','Dự báo đã lưu','Loại vì thực tế thiếu/không đủ 6 số đo','Mốc đầu có thực tế','Mốc cuối có thực tế'],coverage)
    daily=pd.read_csv(out/'daily_statistics.csv');top=daily[daily.above_200.gt(0)].sort_values(['above_200','day'],ascending=[False,True]).head(10)
    md+='\n\n## Các ngày có nhiều sai số >200 mm\n\n'
    md+=table(['Ngày','Số điểm','Số điểm >200 mm','Tỷ lệ >200 mm','MAE (mm)'],[[r.day,r.total,r.above_200,f'{r.above_200_pct:.2f}%',f'{r.mae_mm:.2f}'] for r in top.itertuples()]) if not top.empty else 'Không có điểm dự báo nào có sai số tuyệt đối >200 mm trong khoảng được chấm.'
    md+='\n\n## Nhận xét\n\n'
    best=stats.loc[stats.mae_mm.idxmin()];worst=stats.loc[stats.mae_mm.idxmax()]
    md+=f'- MAE thấp nhất ở +{int(best.horizon_hours)} giờ: **{best.mae_mm:.2f} mm**; cao nhất ở +{int(worst.horizon_hours)} giờ: **{worst.mae_mm:.2f} mm**.\n'
    if start.year==2025 and start.month==11:md+='- Không có dự báo cho toàn bộ đầu tháng 11 trong snapshot; thống kê chỉ phản ánh đoạn cuối tháng có dự báo được lưu. Không tính thời gian không có dự báo là sai số bằng 0.\n'
    md+='- Sự khác biệt giữa tháng có thể liên quan chế độ mực nước, thời tiết và việc trùng giai đoạn train; các bảng không chứng minh nguyên nhân.\n'
    md+='''
## Figure tổng hợp

Histogram dùng bin **25 mm**, cùng thang trục. Trục x là sai số tuyệt đối; trục y là số mẫu. Bin dạng [trái, phải), bin cuối gồm biên phải. Legend chung thể hiện số mẫu, MAE và ngưỡng 200 mm; mỗi panel ghi n, MAE, Min, Max.

![Phân bố sai số theo horizon](absolute_errors_all_horizons.png)

[Tải PDF](absolute_errors_all_horizons.pdf) · [Tải SVG](absolute_errors_all_horizons.svg)

## Dữ liệu đối chiếu

[Thống kê horizon](statistics.csv) · [Số mẫu từng bin](histogram_counts.csv) · [Thống kê ngày](daily_statistics.csv) · [Dự báo và sai số](forecast_points.csv) · [Metadata](manifest.json)

Chạy lại: dùng `research/plot_forecast_error_histograms.py` với `--start-date`, `--end-date`, `--output-dir`, sau đó `research/summarize_error_histograms.py <output-dir>`.
'''
    (out/'phan_tich_sai_so.md').write_text(md)
    (out/'report.md').write_text(md)
    print('Created combined PNG/SVG/PDF and Markdown:',out)


if __name__=='__main__':main()
