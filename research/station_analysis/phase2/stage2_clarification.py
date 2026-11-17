"""Focused eight-question audit of existing Stage 2 artifacts, without rerunning EDA."""
from pathlib import Path

import numpy as np
import pandas as pd

from .station_selection import evaluate_single_station_candidates

DV = '0074647403'
PAIRS = ['750062', '750007', '750041', '750030', '750014', '750066']


def paired_stats(x, y, mask=None):
    pair = pd.concat([x.rename('x'), y.rename('y')], axis=1)
    if mask is not None:
        pair = pair.loc[mask]
    pair = pair.dropna()
    r = pair.x.corr(pair.y) if len(pair) >= 3 and pair.x.std() > 0 and pair.y.std() > 0 else np.nan
    return len(pair), r


def contiguous_count(mask):
    values = mask.fillna(False).astype(bool)
    return int((values & ~values.shift(1, fill_value=False)).sum())


def audit_clarification(*, output, rainfall, targets, stations, candidates, combinations,
                        clusters, episodes, cutoff, snapshot, table):
    """Compute only diagnostics required to answer the user's eight audit questions."""
    output = Path(output)
    audit_dir = output / 'tables' / 'clarification'
    audit_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    def save(name, frame):
        path = audit_dir / f'{name}.csv'
        frame.to_csv(path, index=False)
        saved.append(path)
        return frame

    def corr(x, y, mask=None):
        return paired_stats(x, y, mask)[1]

    index = rainfall.index
    boundary = pd.Timestamp(cutoff).tz_convert('UTC')
    masks = {'Train đã purge': index + pd.Timedelta(hours=12) <= boundary,
             'Đánh giá thăm dò': index > boundary,
             'Toàn kỳ': np.ones(len(index), dtype=bool)}
    target = targets[DV]
    rain6 = rainfall.rolling(6, min_periods=6).sum()
    old_boundary = pd.Timestamp('2026-08-01T00:00:00Z')
    dv_candidates = candidates[candidates.water_station_id.astype(str).eq(DV)].set_index('rain_station_id')
    # Reproduce the documented OLD grid/cutoff using the same frozen input.
    # The current selector purges 12h, hence +12h restores the historical unpurged anchor boundary.
    rain_meta = stations[stations.station_type.eq('rainfall')]
    water_meta = stations[stations.station_id.astype(str).eq(DV)]
    distances = pd.DataFrame([dv_candidates.distance_km.to_dict()], index=[DV])
    old = evaluate_single_station_candidates(
        {w: rainfall.rolling(w, min_periods=w).sum() for w in (1, 3, 6, 12, 24)},
        {DV: target}, water_meta, rain_meta, distances,
        dv_candidates.reset_index()[['rain_station_id', 'elevation_m']].rename(columns={'rain_station_id':'station_id'}),
        train_end_utc=str(old_boundary + pd.Timedelta(hours=12)),
        windows=(1, 3, 6, 12, 24), lags=tuple(range(0, 37, 2)),
    ).set_index('rain_station_id')
    historical = {
        '750007': (.4037, .4037), '750062': (.3931, .3931),
        '750041': (.1842, .1596), '750014': (.1621, .0991),
        '750033': (.1512, .1332), '750030': (.1905, .1905),
    }
    rows = []
    for sid, (draft, csv) in historical.items():
        current = dv_candidates.loc[sid]
        previous = old.loc[sid]
        rows.append({'station_id': sid, 'name': current.rain_station_name,
                     'draft_report_r': draft, 'historical_csv_r_as_reported': csv,
                     'recomputed_old_r': previous.best_corr_dh12,
                     'old_window_h': previous.best_window_dh12, 'old_lag_h': previous.best_lag_dh12,
                     'current_csv_r': current.best_corr_dh12,
                     'current_window_h': current.best_window_dh12, 'current_lag_h': current.best_lag_dh12,
                     'full_period_r_at_old_configuration': corr(
                         rainfall[sid].rolling(int(previous.best_window_dh12), min_periods=int(previous.best_window_dh12)).sum().shift(int(previous.best_lag_dh12)),
                         target.dh_p12h)})
    reconciliation = save('01_report_csv_recomputed', pd.DataFrame(rows))

    # Independently recompute all current Dã Viên rows, at the configurations saved in CSV.
    mismatches = []
    scope_rows = []
    for sid, row in dv_candidates.iterrows():
        w, lag = int(row.best_window_dh12), int(row.best_lag_dh12)
        x = rainfall[sid].rolling(w, min_periods=w).sum().shift(lag)
        n, r = paired_stats(x, target.dh_p12h, masks['Train đã purge'])
        if not np.isclose(round(r, 4), row.best_corr_dh12, equal_nan=True):
            mismatches.append(sid)
        scope_rows.append({'calculation':'Candidate CSV at saved optimum', 'station_id':sid,
                           'scope':'Train đã purge', 'first_anchor_utc':str(index[masks['Train đã purge']].min()),
                           'last_anchor_utc':str(index[masks['Train đã purge']].max()),
                           'window_h':w, 'lag_h':lag, 'target':'dh_p12h', 'sample_n':n, 'pearson_r':r})
    for sid in PAIRS:
        for label, mask in masks.items():
            for column in ['water_level_mm', 'dh_p6h', 'dh_p12h']:
                n, r = paired_stats(rain6[sid], target[column], mask)
                scope_rows.append({'calculation':'Fixed rain W6 L0 / LOO baseline before removal',
                                   'station_id':sid, 'scope':label,
                                   'first_anchor_utc':str(index[mask].min()), 'last_anchor_utc':str(index[mask].max()),
                                   'window_h':6, 'lag_h':0, 'target':column, 'sample_n':n, 'pearson_r':r})
    for row in combinations.itertuples():
        for label, n in [('Train đã purge',row.train_n),('Đánh giá thăm dò',row.evaluation_n)]:
            mask=masks[label]
            scope_rows.append({'calculation':row.subset_name,'station_id':row.station_ids,
                               'scope':label,'first_anchor_utc':str(index[mask].min()),
                               'last_anchor_utc':str(index[mask].max()),'window_h':6,'lag_h':0,
                               'target':'dh_p12h','sample_n':n,
                               'pearson_r':row.train_pearson_r if label=='Train đã purge' else row.test_pearson_r})
    scope = save('02_calculation_scope', pd.DataFrame(scope_rows))

    event_rows = []
    for sid in ['750062', '750007', '750041', '750030']:
        baseline_n, baseline_r = paired_stats(rain6[sid], target.dh_p12h)
        for number, (start, end) in enumerate(episodes, 1):
            keep = (index < start) | (index > end)
            n, r = paired_stats(rain6[sid], target.dh_p12h, keep)
            event_rows.append({'station_id':sid, 'removed_episode':number,
                               'start_utc':str(start), 'end_utc':str(end),
                               'duration_hours':int((end-start)/pd.Timedelta(hours=1))+1,
                               'baseline_n':baseline_n, 'baseline_r':baseline_r, 'remaining_n':n,
                               'loo_r':r, 'delta_r':r-baseline_r})
    loo_detail = save('03_loo_per_removed_episode', pd.DataFrame(event_rows))
    influential = loo_detail.loc[loo_detail.groupby('station_id').loo_r.idxmin()]
    number = int(influential.loc[influential.station_id.eq('750062'), 'removed_episode'].iloc[0])
    start, end = episodes[number-1]
    event_mask = (index >= start) & (index <= end)
    ep_stats = []
    for sid in PAIRS:
        ep_stats.append({'station_id':sid, 'event':number, 'observed_rain_total_mm':rainfall.loc[event_mask,sid].sum(min_count=1),
                         'rain_valid_hours':int(rainfall.loc[event_mask,sid].notna().sum()),
                         'max_rain6_mm':rain6.loc[event_mask,sid].max(),
                         'r_with_dh12_inside_event':corr(rain6[sid], target.dh_p12h, event_mask)})
    event_rain = save('04_influential_event_rain',pd.DataFrame(ep_stats))

    distribution_rows, counts = [], []
    for label, mask in list(masks.items())[:2]:
        observed = rainfall.loc[mask]
        network_wet = observed.gt(0).any(axis=1) & observed.notna().any(axis=1)
        h = target.loc[mask]
        rapid = h.dh_p12h.abs().ge(200)
        rapid_rise = h.dh_p12h.ge(200)
        counts.append({'scope':label, 'grid_hours':int(mask.sum()),
                       'network_observed_hours':int(observed.notna().any(axis=1).sum()),
                       'network_wet_hours':int(network_wet.sum()),
                       'hong_thai_wet_hours':int(observed['750062'].gt(0).sum()),
                       'bach_ma_750041_wet_hours':int(observed['750041'].gt(0).sum()),
                       'positive_dh12_hours':int(h.dh_p12h.gt(0).sum()),
                       'abs_dh12_ge200_hours':int(rapid.sum()),
                       'dh12_ge200_hours':int(rapid_rise.sum()),
                       'rapid_rise_contiguous_runs':contiguous_count(rapid_rise),
                       'flagged_episodes_intersecting_period':sum(bool(((index>=a)&(index<=b)&mask).any()) for a,b in episodes),
                       'episodes_with_abs_dh12_ge200_in_period':sum(bool(((index>=a)&(index<=b)&mask&target.dh_p12h.abs().ge(200).to_numpy()).any()) for a,b in episodes)})
        series = {f'rain1_{sid}':rainfall[sid] for sid in PAIRS}
        series.update({f'rain6_{sid}':rain6[sid] for sid in PAIRS})
        series.update({column:target[column] for column in ['dh_p6h','dh_p12h']})
        for name, value in series.items():
            s = value.loc[mask].dropna()
            distribution_rows.append({'scope':label, 'variable':name, 'valid_n':len(s),
                                      'mean':s.mean(), 'median':s.median(), 'std':s.std(),
                                      'p90':s.quantile(.9), 'p99':s.quantile(.99),
                                      'min':s.min(), 'max':s.max()})
    counts = save('05_period_event_counts',pd.DataFrame(counts))
    distributions = save('06_train_evaluation_distributions',pd.DataFrame(distribution_rows))

    required = sorted({sid.strip() for ids in combinations.station_ids for sid in ids.split(',')})
    common = rain6[required].notna().all(axis=1) & target.dh_p12h.notna()
    train = masks['Train đã purge'] & common
    evaluation = masks['Đánh giá thăm dò'] & common
    # Replay the existing Ridge objective analytically to audit its saved metrics.
    prediction = {}
    for row in combinations.itertuples():
        ids = [sid.strip() for sid in row.station_ids.split(',')]
        x = rain6.loc[train,ids].to_numpy()
        y = target.loc[train,'dh_p12h'].to_numpy()
        xm, ym = x.mean(axis=0), y.mean()
        xc, yc = x-xm, y-ym
        beta = np.linalg.solve(xc.T@xc + np.eye(len(ids)), xc.T@yc)
        pred = (rain6[ids]-xm).dot(beta)+ym
        if not np.isclose(round(corr(pred,target.dh_p12h,evaluation),4),row.test_pearson_r):
            raise ValueError(f'Ridge replay disagrees: {row.subset_name}')
        prediction[row.subset_name] = pred
    signals = {f'Rain6 {sid}':rain6[sid] for sid in PAIRS}
    signals.update(prediction)
    conditional = []
    for label, mask in list(masks.items())[:2]:
        for condition, cond in {
            'All':np.ones(len(index),dtype=bool),
            'Flood months Sep-Dec':index.tz_convert('Asia/Ho_Chi_Minh').month.isin([9,10,11,12]),
            'Observed past rise dh_m3h > 0':target.dh_m3h.gt(0),
            'Retrospective abs future dh12 >=200':target.dh_p12h.abs().ge(200),
        }.items():
            for name, signal in signals.items():
                use = mask & cond & (common if name in prediction else np.ones(len(index),dtype=bool))
                n,r = paired_stats(signal,target.dh_p12h,use)
                conditional.append({'scope':label,'condition':condition,'signal':name,'n':n,'r':r})
    conditional = save('07_conditional_diagnostics',pd.DataFrame(conditional))

    lag_rows = []
    for num,(a,b) in enumerate(episodes,1):
        for sid in ['750062','750007']:
            for col,horizon in [('water_level_mm',0),('dh_p6h',6),('dh_p12h',12)]:
                # Require the rain window AND target to lie inside the interval.
                mask=(index>=a+pd.Timedelta(hours=5))&(index+pd.Timedelta(hours=horizon)<=b)
                choices=[]
                for lag in range(0,49,2):
                    use=mask&(index-pd.Timedelta(hours=lag+5)>=a)
                    n,r=paired_stats(rain6[sid].shift(lag),target[col],use)
                    if n>=24 and np.isfinite(r): choices.append((r,lag,n))
                if choices:
                    r,lag,n=max(choices)
                    lag_rows.append({'episode':num,'station_id':sid,'target':col,'window_h':6,
                                     'best_lag_h':lag,'r':r,'n':n})
    event_lags = save('08_within_event_lag_diagnostics',pd.DataFrame(lag_rows))
    lag_scope_rows=[]
    for label, mask in masks.items():
        for sid in ['750062','750007']:
            for col in ['water_level_mm','dh_p6h','dh_p12h']:
                values=[]
                for lag in range(0,73,2):
                    n,r=paired_stats(rain6[sid].shift(lag),target[col],mask)
                    if np.isfinite(r): values.append((r,lag,n))
                r,lag,n=max(values)
                n18,r18=paired_stats(rain6[sid].shift(18),target[col],mask)
                lag_scope_rows.append({'scope':label,'station_id':sid,'target':col,
                                       'window_h':6,'best_lag_h':lag,'best_r':r,'n_at_best':n,
                                       'r_at_lag18':r18,'n_at_lag18':n18})
    lag_scopes=save('12_fixed_window_lag_by_scope',pd.DataFrame(lag_scope_rows))
    identity = stations[stations.station_id.astype(str).isin(['750041','750030','750066'])][['station_id','name','latitude','longitude']].merge(
        clusters[['station_id','elevation_m','cluster_id','cluster_name']],on='station_id',how='left')
    identity = save('09_station_identity',identity)
    if clusters.station_id.duplicated().any() or set(clusters.station_id) != set(rainfall.columns):
        raise ValueError('Cluster membership must contain every rain station exactly once')
    membership = clusters.groupby(['cluster_id','cluster_name']).agg(
        n=('station_id','size'), station_ids=('station_id',lambda s:', '.join(sorted(s.astype(str)))),
        median_elevation_m=('elevation_m','median')).reset_index()
    membership = save('10_cluster_membership',membership)

    # One useful additional figure: the influential interval and its observed rain.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True)
    target.loc[start:end,'water_level_mm'].plot(ax=axes[0],color='navy')
    axes[0].set_ylabel('Water level (mm)')
    rain6.loc[start:end,['750062','750007','750041']].plot(ax=axes[1])
    axes[1].set_ylabel('Past 6h rain (mm)')
    axes[1].set_xlabel('UTC')
    fig.suptitle(f'Flagged episode {number}: influential interval in full-period LOO')
    fig.tight_layout()
    figure=output/'figures/08_clarification_influential_episode.png'
    fig.savefig(figure,dpi=150)
    plt.close(fig)

    # Render the answers from measured values; no event attribution is invented.
    selected = conditional[conditional.signal.isin(['Rain6 750062','Rain6 750007','Rain6 750066',
        'Model B (2 stations): Hồng Thái + Nhâm',
        'Pair (Upstream + Ridge): Hồng Thái + Bạch Mã',
        '3-Station Arm: Hồng Thái + Bạch Mã + Phú Bài'])]
    summary_lags = event_lags.groupby(['station_id','target']).agg(
        event_n=('episode','size'),min_lag=('best_lag_h','min'),max_lag=('best_lag_h','max'),
        unique_lags=('best_lag_h',lambda s:', '.join(map(str,sorted(s.unique()))))).reset_index() if len(event_lags) else pd.DataFrame()
    e=target.loc[start:end]
    gaps=[(episodes[i+1][0]-episodes[i][1])/pd.Timedelta(hours=1) for i in range(len(episodes)-1)]
    names=stations.set_index('station_id').name.to_dict()
    absent=[]
    for sid, frame in targets.items():
        if sid not in set(candidates.water_station_id.astype(str)):
            h=frame.water_level_mm.dropna()
            absent.append({'station_id':sid,'name':names.get(sid,sid),'first_water_utc':str(h.index.min()),
                           'valid_train_dh12':int(frame.loc[masks['Train đã purge'],'dh_p12h'].notna().sum()),
                           'valid_full_water':len(h)})
    full_dist=distributions[distributions.variable.isin(['dh_p6h','dh_p12h','rain6_750062','rain6_750066'])]
    current_loo=pd.read_csv(output/'tables/08_leave_one_out_stability.csv',dtype={'rain_station_id':str})
    errors=pd.DataFrame([
        ['Bảng nháp khác CSV','Đã xác nhận','Bỏ bảng chép tay; đối chiếu sáu trạm trong mục 1.'],
        ['Thiếu purge và cutoff lệch timezone','Đã xác nhận và sửa trước audit này','Train hiện kết thúc 12h trước cutoff UTC+7; không tiếp tục dùng 6.224h như phạm vi mới.'],
        ['Tổ hợp khác anchor và collinearity dùng toàn kỳ','Đã xác nhận và sửa trước audit này','Bảng tổ hợp hiện dùng common anchors và train-only correlation.'],
        ['Tổng mưa train 2.140 / 2.052mm và giải thích tâm bão','Không khớp artifact','Dùng bảng tổng mưa thực tính; không suy ra tâm bão từ tổng cả giai đoạn.'],
        ['r=0,7588 gắn với W6 L18','Sai cấu hình','Đây từng là W24 L12 trên grid cũ; bảng hiện tại tìm cả W48 và ghi riêng từng scope.'],
        ['5/6 luôn làm r lệch dưới 0,0007','Sai','Sai khác tùy trạm và tập anchor; xem bảng sensitivity.'],
        ['LOO score cao chứng minh ổn định vật lý','Không được dữ liệu hỗ trợ','Đọc cả LOO min và ảnh hưởng từng đợt; chưa phải event-independent validation.'],
        ['Ba trạm là tối ưu; Bạch Mã có thể thay mã tùy ý','Chưa có bằng chứng / sai định danh','Giữ giả thuyết; phân biệt 750041, 750030, 750066.'],
    ],columns=['issue','status','resolution'])
    save('11_confirmed_errors',errors)
    text=f'''# Stage 2 — Giải đáp và kiểm chứng tám nhóm câu hỏi nghiên cứu

Báo cáo này trả lời trực tiếp tám nhóm câu hỏi sau review. Các diagnostic được tính lại từ artifacts đóng băng `{Path(snapshot).name}`; không tải thêm dữ liệu, không chạy lại toàn bộ EDA, không thay đổi training dataset và không huấn luyện mô hình dự báo. Phép Ridge hiện có được tái tính bằng nghiệm của cùng hàm mục tiêu chỉ để kiểm chứng bảng và phân tích phần dư theo điều kiện; không chọn lại feature từ kết quả đánh giá.

Bản báo cáo và CSV trước lượt bổ sung được lưu ở [audit_history](audit_history/). Các con số nháp lịch sử trong câu hỏi được giữ riêng với kết quả tái tính, vì không còn đủ artifact gốc để truy ngược chính xác từng con số nháp đến một lần chạy cụ thể.

## 1. Vì sao Markdown Report khác CSV? Số nào đúng?

Code tạo báo cáo cũ dùng bảng Markdown viết cứng, trong khi CSV được tạo từ DataFrame ứng viên. Vì thế khác biệt không chứng minh hai pipeline có chủ ý dùng target khác nhau. Target tuyển chọn là ΔH+12h(t)=H(t+12h)−H(t); mưa là tổng quá khứ ở window và lag ghi trong từng hàng. Bản nháp không lưu đủ provenance để xác nhận riêng 0,1842 hay 0,1621 đã bị lấy từ horizon, phạm vi hay lần chạy nào. Không nên khẳng định nguyên nhân cụ thể đó khi chưa có bằng chứng.

Bảng sau đối chiếu số nháp và CSV lịch sử được nêu trong review, kết quả tái tính đúng cấu hình cũ (cutoff 01/08/2026 00:00 UTC, chưa purge, W=1/3/6/12/24h, L=0..36h bước 2h), cùng kết quả CSV hiện tại sau sửa cutoff/purge và mở grid tới W48/L72. Cột toàn kỳ dùng đúng W/L tối ưu cũ, không dùng một cấu hình khác để so.

{table(reconciliation)}

Các hàng Nhâm, Hồng Thái và VQG Bạch Mã cho phép kiểm tra những trường hợp không lệch r trong bản nháp, trong khi Bạch Mã 750041, Phú Bài và VP có khác biệt. Thủy Thanh, Hồng Vân và Vỹ Dạ cũng từng bị bỏ khỏi top 10 của bảng nháp; bảng master hiện lấy top 10 từ CSV động. Giá trị đúng phải gắn với cấu hình và phạm vi tương ứng, không có một r duy nhất đúng cho mọi phép tính.

Đối soát độc lập tất cả {len(dv_candidates)} hàng Dã Viên hiện tại tại W/L được lưu: **{len(mismatches)} hàng sai khác** khi làm tròn r tới bốn chữ số. [Bảng phạm vi và số mẫu thực](tables/clarification/02_calculation_scope.csv) lưu kết quả tái tính từng hàng. Không có dấu hiệu nhầm station ID trong phép đối soát này; nhầm tên Bạch Mã trong diễn giải vẫn phải phân biệt như mục 7.

## 2. Sự kiện nào làm LOO gần biến mất? Có thật sự ổn định không?

LOO được tính trên **toàn kỳ**, bằng cách loại anchor nằm trong một khoảng sự kiện rồi tính Pearson giữa R6(t), lag 0 và ΔH+12h trên phần còn lại, không phải chỉ trên những giờ lũ còn lại. Kết quả chi tiết từng lần loại xác nhận:

{table(influential)}

Đối với Hồng Thái, khoảng có ảnh hưởng lớn nhất là **episode {number}**, từ **{start}** đến **{end}**, kéo dài **{int((end-start)/pd.Timedelta(hours=1))+1} giờ** tính cả hai đầu. Trong khoảng này, mực nước Dã Viên đạt {e.water_level_mm.max():.2f}mm, ΔH+12h lớn nhất {e.dh_p12h.max():.2f}mm và nhỏ nhất {e.dh_p12h.min():.2f}mm. Tổng mưa và cực đại 6h tại từng trạm được tính trực tiếp:

{table(event_rain)}

![Mực nước và mưa trong khoảng chi phối LOO](figures/08_clarification_influential_episode.png)

Thuật toán trong `water_relationship.extract_flood_episodes` dùng ngưỡng H(t)≥Q90 toàn kỳ (Q90={target.water_level_mm.quantile(.9):.2f}mm) **hoặc** |ΔH+12h|≥200mm; kết thúc đợt khi đã có ít nhất 12 giờ kể từ anchor được đánh dấu cuối. Nó tạo **{len(episodes)} khoảng**, không chồng lấn theo anchor; khoảng cách nhỏ nhất giữa hai đầu đợt kế tiếp là **{min(gaps):.0f} giờ**. Đó chưa phải {len(episodes)} trận lũ độc lập về thủy văn: target và rolling rain ở mép đợt vẫn có thể chạm thời gian vừa loại, các đợt ngắn có thể phản ánh cùng chế độ nền. Điều kiện dùng ΔH tương lai cũng khiến đây là phép phân loại hồi cứu.

Công thức đúng trong code là `max(0, 1 − std(r_LOO) / max(abs(mean(r_LOO)), 0.05))`, dùng độ lệch chuẩn với ddof=0. Vì đa số lần loại làm tương quan thay đổi ít, trung bình và độ lệch chuẩn vẫn có thể tạo score cao dù một lần loại làm r giảm gần 0. Score không kiểm tra trực tiếp mức ảnh hưởng tệ nhất.

{table(current_loo[['rain_station_id','loo_mean_r','loo_std_r','loo_min_r','loo_max_r','stability_score']])}

Kết luận phù hợp là mối tương quan toàn kỳ **nhạy mạnh với ít nhất một khoảng sự kiện**. Chưa chứng minh quan hệ ổn định vật lý qua các trận lũ độc lập. [Tất cả lần loại](tables/clarification/03_loo_per_removed_episode.csv) và [danh mục đợt](tables/14_water_episodes.csv) cho phép kiểm tra lại.

## 3. Vì sao train r khoảng 0,4 nhưng đánh giá gần 0?

Cutoff hiện tại là **{pd.Timestamp(cutoff)}**, tức **{boundary}**. Train chỉ nhận anchor t+12h≤cutoff; giai đoạn đánh giá nhận t>cutoff; 12 anchor ở giữa được purge. Mốc giờ và số cặp Ridge chính xác nằm trong [bảng tổ hợp](tables/10_multistation_combinations.csv). Không còn overlap nhãn +12h ở ranh giới này. Cùng preprocessing, quy tắc missing và một bộ hệ số Ridge fit trên train được áp dụng cho cả hai phía. Rolling rain ở đầu giai đoạn đánh giá được phép dùng quan trắc quá khứ phía train vì thông tin đó đã nằm trước anchor; đây không phải nhãn tương lai.

Các số 0,4049/0,0278 trước đây thuộc cutoff/grid anchor cũ và tổ hợp chưa dùng common anchors. Không so trực tiếp chúng với số mới mà bỏ qua thay đổi protocol. Giai đoạn sau cutoff đã được xem nhiều lần, nên gọi là **đánh giá thăm dò**, không phải independent final test.

{table(counts)}

`network_wet_hours` là số giờ có ít nhất một trạm mưa đo mưa >0; đây không phải mưa trung bình lưu vực. `rapid_rise_contiguous_runs` đếm các chuỗi giờ liên tiếp có ΔH+12h≥200mm, khác với thuật toán episode. `episodes_with_abs_dh12_ge200_in_period` đếm các episode có ít nhất một giờ biến động mạnh trong phần thời gian đó. Không thể gọi số này là số trận “lũ lớn” theo báo động khi chưa có ngưỡng Dã Viên đã xác minh.

{table(full_dist)}

[Bảng phân phối đầy đủ](tables/clarification/06_train_evaluation_distributions.csv) có mean, median, std, P90, P99, min/max cho mưa 1h/6h tại sáu trạm và ΔH+6h/+12h. Dữ liệu đánh giá vẫn có giờ mưa và biến động mạnh, nên không thể giải thích r gần 0 bằng việc “không có mưa lũ”. Khác biệt cường độ/phân phối và ảnh hưởng lớn của episode {number} trên train là bằng chứng về hạn chế tổng quát hóa; chúng chưa đủ để phân rã nguyên nhân thành thay đổi quan hệ vật lý, điều tiết hồ hay triều.

Bảng dưới tách mùa lũ, trạng thái đang dâng theo quá khứ và nhóm biến động mạnh trong tương lai. Nó trả lời định lượng câu hỏi liệu r còn gần 0 khi xét riêng điều kiện. Các hàng rain là tương quan của tín hiệu mưa, còn các hàng tổ hợp là r giữa dự báo Ridge đã có và ΔH thực tế; không được nhập hai loại thành một chỉ số.

{table(selected)}

Nhóm “Retrospective abs future dh12” chỉ dùng để chẩn đoán sau khi biết kết quả, không dùng làm cổng chọn feature hoặc mô hình lúc vận hành. Các nhóm ít mẫu có r không ổn định. Kết quả điều kiện khác r toàn kỳ không chứng minh một thay đổi nhân quả hoặc bảo đảm hiệu quả ở sự kiện chưa thấy.

## 4. Tín hiệu nhiều trạm được kết hợp thế nào?

Không cộng hoặc lấy trung bình mưa các trạm. Code dùng Ridge với intercept: dự báo ΔH+12h bằng `intercept + Σ beta_i × R6_i(t)`, alpha=1. Mọi trạm cùng W=6h và lag=0. Không Z-score input; giữ đơn vị mm là lựa chọn triển khai hiện tại, không phải yêu cầu vật lý bắt buộc. Ridge nhạy với thang đo input, nên chưa chuẩn hóa là một giới hạn cần ghi nhận nếu sau này so với thiết kế khác.

Mọi tổ hợp trong lượt chốt dùng cùng anchor đầy đủ trên hợp tất cả trạm tham gia: {int(combinations.train_n.iloc[0])} train / {int(combinations.evaluation_n.iloc[0])} đánh giá. Đa cộng tuyến là Pearson giữa các R6_i trên cùng tập train đó. Thứ tự bảng dựa trên train; không chọn cấu hình bằng test diagnostic.

{table(combinations[['subset_name','max_collinearity','train_pearson_r','test_pearson_r','test_r2']])}

Đây là predictive performance của một phép hồi quy mưa-only rất đơn giản, không phải correlation của tổng mưa, cũng không phải kết quả checkpoint ResidualLSTM Model B. Giảm collinearity chưa chứng minh tăng thông tin dự báo có ích ngoài lịch sử mực nước. **Không có bằng chứng từ bảng này để kết luận Hồng Thái+Bạch Mã+Phú Bài tối ưu.** Cần ablation có water-only baseline, cùng anchors và kiểm định ngoài train; Stage 4 chưa được thực hiện ở lượt audit này.

## 5. Lag 18h có ý nghĩa gì?

Code tính mưa tại lag L bằng `rain_W.shift(L)` trên lưới giờ đầy đủ. Tại anchor t, đặc trưng chứa mưa trong **(t−L−W, t−L]**. L dương nghĩa mưa kết thúc sớm hơn anchor, không phải dịch mưa tương lai về quá khứ. “Best lag” chỉ là L có Pearson lớn nhất trên grid, không phải đo trực tiếp thời gian tập trung dòng chảy.

Nhận định cũ “18h, r=0,518” chưa xác định đầy đủ scope và không được giữ như một số liệu đã kiểm chứng. Giá trị 0,7588 từng thuộc bảng train W24/L12, không được ghép với W6/L18. Bảng scope ghi tách hẳn window/lag/target. Nếu cần xác nhận giá trị tại lag18 trên cùng cấu hình, phải dùng đúng window và scope trong bảng đối soát, không đọc r cực đại của cấu hình khác.

Bảng sau tái tính riêng W6 trên train, đánh giá và toàn kỳ cho H(t), ΔH+6h và ΔH+12h; có cả r tại lag18 và cực đại trong grid. Nó cho thấy trực tiếp việc đổi target hoặc scope thay đổi kết quả ra sao.

{table(lag_scopes)}

Riêng Hồng Thái với W6, toàn kỳ đạt đỉnh tại lag {int(lag_scopes.loc[(lag_scopes.scope=='Toàn kỳ') & (lag_scopes.station_id=='750062') & (lag_scopes.target=='water_level_mm'),'best_lag_h'].iloc[0])}h, r={lag_scopes.loc[(lag_scopes.scope=='Toàn kỳ') & (lag_scopes.station_id=='750062') & (lag_scopes.target=='water_level_mm'),'best_r'].iloc[0]:.4f}; r tại lag18 là {lag_scopes.loc[(lag_scopes.scope=='Toàn kỳ') & (lag_scopes.station_id=='750062') & (lag_scopes.target=='water_level_mm'),'r_at_lag18'].iloc[0]:.4f}. Như vậy 18h không phải cực đại toàn kỳ trong grid đã kiểm tra. Khi dùng ΔH+12h trên train, cực đại W6 chuyển về lag0. Điều này trực tiếp bác bỏ cách hiểu một lag duy nhất áp dụng cho mọi target và phạm vi.

Các diagnostic theo từng khoảng sự kiện dưới đây tìm lag trên W6, grid 0..48h bước 2h, chỉ giữ ít nhất 24 cặp có cửa sổ mưa và target cùng nằm trong sự kiện. Chúng là diagnostic để kiểm tra giả thuyết “cùng một lag”, không dùng chọn feature. Không phải tất cả các episode đều dài đủ để tham gia; thay đổi sample count theo lag cũng là một giới hạn.

{table(summary_lags)}

[Từng sự kiện, target và lag](tables/clarification/08_within_event_lag_diagnostics.csv) thể hiện những lag khác nhau; không có căn cứ áp một travel time cố định cho mọi trận. Điều tiết hồ, trạng thái nước ban đầu và triều là cơ chế có thể ảnh hưởng về mặt giả thuyết, nhưng dữ liệu audit chưa cô lập chúng bằng chuỗi xả hồ/triều và mô hình nhân quả. Vì vậy không gán một khoảng truyền lũ 4–24h như kết quả đã đo.

## 6. Có dùng cùng phạm vi cho tuyển chọn và LOO không?

Không. Tuyển chọn tối ưu W/L dùng train đã purge, còn LOO dùng toàn kỳ, W6/L0 và ΔH+12h. Bảng thống nhất dưới đây lưu cả số mẫu thực sau loại thiếu, khác với số giờ của lưới. Các hàng full-period ở hai trạm Hồng Thái/Nhâm chính là baseline LOO; không kỳ vọng chúng bằng r train.

{table(scope[(scope.station_id.isin(['750062','750007'])) & ((scope.calculation.eq('Candidate CSV at saved optimum')) | scope.target.eq('dh_p12h'))])}

[Bảng thống nhất đầy đủ](tables/clarification/02_calculation_scope.csv) bao gồm cả H(t), ΔH+6h, ΔH+12h. Kiểm tra `pipeline.run_rainfall` và `station_selection.evaluate_single_station_candidates` cho thấy bảng xếp hạng được tạo từ targets và rain accumulations trên train; `loo_df` không được truyền vào hàm xếp hạng. Vì vậy baseline LOO không thay r tuyển chọn trong code. LOO vẫn có thể ảnh hưởng cách diễn giải bằng lời, nên report không dùng nó để tuyên bố tính ổn định hoặc thay đổi lựa chọn sau khi xem test.

## 7. Station identity và clustering có chính xác không?

Đối chiếu station_id trong catalog đóng băng với cluster CSV cho ba trạm:

{table(identity)}

Các trạm có tên gần nhau nhưng tọa độ/cao độ khác nhau. Cao độ ở đây là pixel FABDEM/Copernicus được pipeline lấy tại tọa độ, không phải cao độ mốc trạm. Không thể chuyển bằng chứng của Đỉnh Bạch Mã 750066 sang trạm Bạch Mã 750041 đang dùng trong một tổ hợp.

Clustering tạo {clusters.cluster_id.nunique()} nhóm bằng Ward với Euclidean distance trên vector `[10×latitude, 10×longitude, log(1+DEM_elevation), correlation_to_station_1, …, correlation_to_station_55]`. Profile correlation dùng toàn kỳ; NaN được thay bằng 0. Không sử dụng ma trận khoảng cách km để fit Ward và không standardize vector sau khi ghép. Do đó không có các trọng số vật lý đã hiệu chỉnh giữa tương quan, địa lý và DEM; phần kinh/vĩ độ có hệ số 10, DEM dùng log, còn 55 chiều correlation có trọng số riêng 1 theo Euclidean squared distance. Đây là phân cụm thăm dò, chưa được kiểm định độ bền khi đổi cách scale.

{table(membership)}

Đối soát hiện tại có {clusters.cluster_name.nunique()} tên khác nhau cho {clusters.cluster_id.nunique()} cluster ID và mỗi station_id xuất hiện đúng một lần. Nhãn cụm được suy ra sau clustering từ trung vị cao độ và vị trí trung bình, không tham gia fit. Nhãn trùng giữa hai ID trong bản cũ là lỗi gán tên/diễn giải, không chứng minh hai cụm có cùng thành viên hoặc station mapping sai. Nhãn hiện tại là nhãn địa lý suy luận, không xác nhận lưu vực Tả Trạch/Hữu Trạch. Không dùng tên cụm làm ground truth lưu vực.

Có {len(targets)} chuỗi nước trong EDA, nhưng bảng candidates có {candidates.water_station_id.nunique()} trạm. Các trạm bị thiếu trong bảng:

{table(pd.DataFrame(absent))}

Hàm tuyển chọn bỏ trạm có dưới 100 nhãn ΔH+12h hợp lệ trong train. Đây là kiểm tra dữ liệu sẵn có, không chứng minh ngày lắp đặt thiết bị. Những ngày đầu quan trắc thực tế trong bảng có thể được đối soát trực tiếp; không tiếp tục dùng nhận định “mới từ tháng 8/2026” nếu không khớp timestamps.

## 8. Phân loại lại mức độ tin cậy của các kết luận

| Nhận định | Phân loại | Căn cứ và giới hạn | Cần làm nếu muốn xác minh thêm |
| --- | --- | --- | --- |
| Mưa A Lưới có liên hệ với ΔH Dã Viên +6h/+12h | Supported trong phạm vi train đã xét | Có tương quan train trực tiếp; hiệu quả dự báo bổ sung và tổng quát hóa chưa được chứng minh. | Ablation so với water-only trên các đợt chưa thấy. |
| Quan hệ ổn định qua các trận lũ độc lập | Unsupported | LOO min gần 0 khi bỏ một khoảng, các episode chưa độc lập về cửa sổ/động lực học. | Chia sự kiện độc lập, purge cửa sổ, nhiều mùa lũ. |
| Bạch Mã có predictive information bổ sung ngoài Hồng Thái/Nhâm | Tentative | Rain correlation thấp hơn là khác biệt tín hiệu, không phải kiểm định incremental skill; từng mã trạm phải được xét riêng. | Đo thay đổi lỗi khi thêm từng trạm, cùng anchor và baseline. |
| Hồng Thái+Bạch Mã+Phú Bài là tốt nhất | Unsupported | Ridge không chứng minh ưu thế và chưa có đối chứng đủ cho khẳng định tối ưu. | Giữ như ứng viên, thực hiện ablation sau Stage 3. |
| Lag0 và W6/W12 phù hợp forecasting | Tentative cho toàn mạng; Supported như ứng viên tại một số cặp Dã Viên | Một số cặp đạt cực đại train ở cấu hình này, nhiều cặp khác có lag/window khác; chọn cực đại nhiều lần có selection bias. | Đóng băng cấu hình trên train và kiểm tra hiệu quả ngoài train. |

Supported ở đây luôn gắn phạm vi đo, không mang nghĩa nhân quả hay hiệu quả production. Không thay Model B từ audit này.

## Các lỗi đã xác nhận và kết luận phải điều chỉnh

{table(errors)}

Quy ước interval rain vẫn là giả định của dự án nếu chưa có tài liệu nhà cung cấp. Các giới hạn forecast as-of từ Stage 1 vẫn giữ nguyên. Audit này dừng tại kiểm chứng Stage 2, chưa thực hiện Stage 3/4 hoặc thay đổi training datasets.

[Báo cáo master](report.md) · [Bảng audit](tables/clarification/) · [Dấu vết chạy](clarification_manifest.json)
'''
    return text, saved, figure


def main():
    """Regenerate reports and targeted diagnostics from saved artifacts only."""
    import argparse
    import hashlib
    import json
    from datetime import datetime, timezone
    from .config import load_config
    from .rainfall_preprocessing import preprocess_canonical_water_levels
    from .stage2_report import write_stage2_report

    parser=argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('config.yaml'))
    args=parser.parse_args()
    config=load_config(args.config)
    output=Path(config['output_dir'])/'stage2_rainfall'
    inventory=Path(config['work_dir'])/'inventory'
    work=Path(config['work_dir'])/'rainfall'
    snapshot=Path(json.loads((inventory/'current_snapshot.json').read_text())['directory'])
    stations=pd.read_parquet(snapshot/'stations.parquet')
    rainfall=pd.read_parquet(work/'tables/hourly_rainfall_standard.parquet')
    water=stations[stations.station_type.eq('water_level')]
    _,targets=preprocess_canonical_water_levels(
        pd.read_parquet(inventory/'tables/observation_availability.parquet'),water,rainfall.index)
    names=['01_station_rainfall_summary','02_seasonal_rainfall_summary','03_spatial_distance_decay',
           '04_rainfall_clusters','05_independent_rainfall_events','06_single_station_candidates',
           '07_conditional_relationships','08_leave_one_out_stability','09_policy_sensitivity',
           '10_multistation_combinations','11_station_exclusion_list']
    tables=[pd.read_csv(output/'tables'/f'{name}.csv',dtype={
        'station_id':str,'rain_station_id':str,'water_station_id':str}) for name in names]
    episode_frame=pd.read_csv(output/'tables/14_water_episodes.csv')
    episodes=[(pd.Timestamp(r.start_utc),pd.Timestamp(r.end_utc)) for r in episode_frame.itertuples()]
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    input_paths=[work/'tables/hourly_rainfall_standard.parquet',snapshot/'stations.parquet',
                 inventory/'tables/observation_availability.parquet',output/'tables/14_water_episodes.csv']
    input_paths += [output/'tables'/f'{name}.csv' for name in names]
    input_hashes={str(p):sha(p) for p in input_paths}
    code={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
    write_stage2_report(*tables,[],output,anchors=rainfall.index,
                        cutoff=config['future_analysis']['grouping_end'],rainfall=rainfall,
                        targets=targets,episodes=episodes,snapshot=snapshot,stations=stations)
    outputs=[output/'report.md',output/'stage2_clarification_report.md']
    outputs+=sorted((output/'tables/clarification').glob('*.csv'))
    outputs+=[output/'figures/08_clarification_influential_episode.png']
    record={'stage':'stage2_targeted_clarification_audit','completed_at_utc':datetime.now(timezone.utc).isoformat(),
            'snapshot_directory':str(snapshot),'snapshot_manifest_sha256':sha(snapshot/'manifest.json'),
            'configuration':config,'code_sha256':code,'input_sha256':input_hashes,
            'output_sha256':{str(p.relative_to(output)):sha(p) for p in outputs},
            'not_executed':['full_eda','stage3','stage4','model_training','dataset_changes']}
    (output/'clarification_manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    # Preserve original run metadata; explicitly record this subsequent report-only audit.
    path=output/'run_manifest.json'
    manifest=json.loads(path.read_text())
    manifest['latest_clarification_audit']='clarification_manifest.json'
    manifest['output_sha256']={str(p.relative_to(output)):sha(p) for p in output.rglob('*')
        if p.is_file() and 'audit_history' not in p.parts and p.suffix in ('.csv','.md','.png')}
    manifest['tables_count']=len(list((output/'tables').rglob('*.csv')))
    manifest['figures_count']=len(list((output/'figures').glob('*.png')))
    path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(f'Targeted audit complete: {output / "stage2_clarification_report.md"}')


if __name__=='__main__':
    main()
