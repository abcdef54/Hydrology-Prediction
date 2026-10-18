"""Write descriptive tables and plots for rain regime diagnostics."""
from pathlib import Path
import pandas as pd,numpy as np,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def write_report(out):
    p=pd.read_csv(out/'forecast_points.csv')
    for col in ['forecast_at','anchor_observed_at']:p[col]=pd.to_datetime(p[col],utc=True)
    valid=p.dropna(subset=['actual','anchor_level','target24_west','target24_south']).copy()
    valid=valid[valid.scope.ne('earlier_exploratory')]
    w,s=valid.target24_west,valid.target24_south
    valid['absolute_regime']=np.select([(w<=10)&(s>=20),(w>=20)&(s<=10),(w>=20)&(s>=20),(w<=10)&(s<=10)],['South high / West low','West high / South low','Both high','Both low'],default='Intermediate')
    valid['relative_regime']=np.select([(s>=20)&(s>=3*w),(w>=20)&(w>=3*s),(w>=20)&(s>=20)],['South >= 3x West','West >= 3x South','Both high, balanced'],default='Other')
    rows=[]
    for kind in ['absolute_regime','relative_regime']:
     for (h,label),f in valid.groupby(['horizon_minutes',kind]):
      rows.append(dict(definition=kind,horizon=h//60,regime=label,n=len(f),days=f.forecast_at.dt.tz_convert('Asia/Ho_Chi_Minh').dt.date.nunique(),mae=f.abs_error.mean(),bias=f.error.mean(),persistence_mae=f.persistence_abs_error.mean()))
    metrics=pd.DataFrame(rows);metrics.to_csv(out/'recent_metrics.csv',index=False)
    print(metrics[metrics.horizon==24].round(2).to_string(index=False))
    fig,axes=plt.subplots(1,2,figsize=(13,4.8))
    for ax,kind in zip(axes,['absolute_regime','relative_regime']):
     f=metrics[metrics.definition.eq(kind)]
     for name,part in f.groupby('regime'):
      ax.plot(part.horizon,part.mae,marker='o',label=name)
     ax.set(xlabel='Forecast horizon (hours)',ylabel='MAE (mm)',title=kind.replace('_',' '),xticks=[1,3,6,12,24]);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Da Vien saved model: anchors from 2026-08-13; rain in 24h ending at target')
    fig.tight_layout();fig.savefig(out/'mae_by_rain_regime.png',dpi=160);plt.close(fig)

    def table(frame):
     f=frame.copy()
     for col in f.select_dtypes('number'):f[col]=f[col].round(2)
     return '| '+' | '.join(f.columns)+' |\n|'+'|'.join(['---']*len(f.columns))+'|\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in f.itertuples(index=False,name=None))
    summary=pd.read_csv(out/'regime_metrics.csv')
    parts=['# Phân tích sai số Dã Viên theo phân bố mưa',
    '## Dữ liệu và cách tính',
    '- 39.650 điểm dự báo đã lưu, 7.930 anchors, cùng một model_version và pipeline_version; không train lại hoặc thay đổi database.',
    '- Nhóm A Lưới: Hồng Thái 750062, Nhâm 750007, TT A Lưới 750003, Hồ A Lá 750011. Nhóm phía nam: Giang Hải 750010, Cảng Chân Mây 228005, Đỉnh Bạch Mã 750066. Đây là nhóm vị trí, không phải phân vùng lưu vực đã xác minh. Không đại diện toàn bộ thượng nguồn.',
    '- Tổng 24 giờ tại từng trạm, sau đó lấy trung bình giữa các trạm trong nhóm (không cộng dồn giữa trạm). Giữ cách cộng số đo nguồn hiện tại; chưa xác minh khoảng tích lũy nguồn. Chỉ giữ cửa sổ đủ 6 giá trị hợp lệ mỗi giờ, đủ 24 giờ tại tất cả 7 trạm; thực tế và mực nước anchor cũng yêu cầu đủ 6 giá trị/giờ. Thiếu không thay bằng 0.',
    '- Cao >=20, thấp <=10 theo tổng số đo nguồn 24 giờ. Ngưỡng phân tích, không phải phân loại mưa khí tượng. Trung gian giữ riêng. Phân tích tương đối dùng vùng trội >=20 và >=3 lần vùng kia. Kiểm tra độ nhạy high=50/low=10 và high=20/low=5 trong regime_metrics.csv.',
    '- known: mưa 24 giờ kết thúc tại anchor, đã có thể quan trắc. target24: mưa 24 giờ kết thúc tại mốc dự báo, dùng hồi cứu; với +24h là đúng khoảng tương lai (anchor,target]. Với horizon ngắn hơn cửa sổ chứa cả trước và sau anchor.',
    '- Bảng chính giới hạn anchors từ 2026-08-13 07:00 UTC, gồm giai đoạn test theo manifest local và dữ liệu sau manifest đến 2026-10-15. Phân tích toàn bộ lịch sử chỉ mang tính khám phá; split local không chứng minh tuyệt đối provenance huấn luyện checkpoint. Dự báo được lưu có thể là backfill, không coi generated_at là thời điểm phát vận hành thực.',
    '- Các cửa sổ giờ chồng lấn. Số ngày không phải số đợt mưa độc lập. Không có kết luận nhân quả hay kiểm định ý nghĩa thống kê từ các mẫu giờ này.',
    '## Kết quả 24 giờ: ngưỡng tuyệt đối, anchors từ 13/08',table(metrics[(metrics.definition=='absolute_regime')&(metrics.horizon==24)]),
    '## Kết quả 24 giờ: tương phản giữa hai vùng',table(metrics[(metrics.definition=='relative_regime')&(metrics.horizon==24)]),
    '## Phân biệt mưa đã biết và mưa tương lai: riêng sau manifest',table(summary[(summary.high==20)&(summary.low==10)&(summary.scope=='after_dataset')&(summary.horizon_hours==24)][['phase','regime','n','days','mae','bias','persistence_mae']]),
    '## Các trường hợp cụ thể',
    'Thời gian dưới đây là giờ Việt Nam. Bias = dự báo - thực tế. Mưa tính 24 giờ kết thúc tại mốc dự báo.',
    ]
    cases=[('2026-10-08T00:00:00+07:00','Ví dụ ảnh 08/10'),('2026-10-13T09:00:00+07:00','Ví dụ ảnh 13/10'),('2026-10-14T12:00:00+07:00','Phía nam cao, dự báo thừa'),('2026-09-14T08:00:00+07:00','A Lưới cao, phía nam thấp'),('2026-09-13T06:00:00+07:00','Cả hai cao, dự báo thiếu')]
    selected=[]
    for dt,label in cases:
     f=valid[valid.horizon_minutes.eq(1440)&valid.forecast_at.eq(pd.Timestamp(dt))]
     if f.empty: continue
     assert len(f)==1,(dt,len(f))
     r=f.iloc[0];selected.append({'case':label,'target_VN':str(r.forecast_at.tz_convert('Asia/Ho_Chi_Minh')),'west_24h':r.target24_west,'south_24h':r.target24_south,'predicted':r.water_level_mm,'actual':r.actual,'bias':r.error})
    parts+=[table(pd.DataFrame(selected)),'## Đọc kết quả',
    '- Hai vùng cùng cao có MAE 24h lớn và thiên hướng dự báo thiếu. A Lưới cao/phía nam thấp có các trường hợp dự báo thừa lớn. Phía nam cao/A Lưới thấp có cả thừa và thiếu; không thể suy rộng từ hai hình.',
    '- Theo ngưỡng 20/10, ảnh 08/10 thuộc cả hai cao (A Lưới 27,2, phía nam 289,13); ảnh 13/10 thuộc trung gian (15,55 và 95). Theo tương phản >=3x, cả hai đều phía nam trội. Cửa sổ 8 giờ trong trao đổi trước không tương đương 24 giờ ở đây.',
    '- Model có cải thiện đáng kể ở horizon ngắn nhưng lợi ích so với persistence ở +24h phụ thuộc nhóm. Không có bằng chứng từ phân tích này rằng thay trạm chắc chắn giúp; cần ablation trên dữ liệu tách thời gian.',
    '- Bước tiếp theo: xác minh lưu vực/đơn vị số đo mưa, đánh giá theo đợt độc lập, đối chiếu hồ và triều, rồi so sánh model cũ với bộ nguồn đa vùng trên cùng tập. Mưa tương lai chỉ dùng giải thích hồi cứu, không đưa vào training như quan trắc đã biết.',
    '## Tái lập',
    '`python research/analyze_rain_regime_errors.py --input-dir /tmp/hydrology-rain-regimes --output-dir reports/rain_regime_errors_2026-10-15`',
    'Xuất lại snapshot: `python research/export_rain_regime_snapshot.py` (truy vấn chỉ đọc, chạy từ root repo). Raw export hiện nằm ngoài Git ở /tmp/hydrology-rain-regimes. forecast_points.csv và hourly_group_rain.csv giữ dữ liệu tổng hợp để kiểm tra các bảng; manifest.json giữ cấu hình. Không xuất URL hay thông tin đăng nhập PostgreSQL.']
    (out/'report.md').write_text('\n\n'.join(parts))
