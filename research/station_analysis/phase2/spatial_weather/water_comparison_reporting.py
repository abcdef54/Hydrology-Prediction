"""Readable, paired old/new-grid comparisons from existing Stage 3B results."""
import numpy as np
import pandas as pd

from .analysis import WINDOWS

BASELINE = 'ecmwf_ifs:16.274164:107.264960'
KEYS = ['water_station_id', 'scope', 'horizon_h', 'feature']


def paired_water_comparisons(relationships):
    """Retain signs and join each grid to the exact same station/target/scope/window."""
    frame = relationships[
        relationships.condition.eq('all') & relationships.horizon_h.isin([6, 12])
    ].copy()
    if frame.duplicated(KEYS + ['grid_id']).any():
        raise ValueError('Duplicate station/grid comparison rows')
    if frame.groupby(KEYS).n.nunique().gt(1).any():
        raise ValueError('Grid comparisons do not have matching anchor counts')
    reference = frame[frame.grid_id.eq(BASELINE)][KEYS + ['n', 'pearson_r', 'spearman_r']]
    reference = reference.rename(columns={
        'n': 'baseline_n', 'pearson_r': 'baseline_pearson_r',
        'spearman_r': 'baseline_spearman_r'})
    paired = frame.merge(reference, on=KEYS, how='left', validate='many_to_one')
    if paired.baseline_n.isna().any():
        raise ValueError('An old-grid reference is missing')
    paired['delta_pearson_r'] = paired.pearson_r - paired.baseline_pearson_r
    paired['delta_spearman_r'] = paired.spearman_r - paired.baseline_spearman_r
    paired['delta_abs_pearson_r'] = paired.pearson_r.abs() - paired.baseline_pearson_r.abs()
    paired['delta_abs_spearman_r'] = paired.spearman_r.abs() - paired.baseline_spearman_r.abs()
    return paired


def _number(value):
    return f'{value:.4f}' if pd.notna(value) and np.isfinite(value) else '—'


def _table(headers, rows):
    def safe(value):
        return str(value).replace('|', '/').replace('\n', ' ')
    return '\n'.join([
        '| ' + ' | '.join(map(safe, headers)) + ' |',
        '| ' + ' | '.join(['---'] * len(headers)) + ' |',
        *['| ' + ' | '.join(map(safe, row)) + ' |' for row in rows],
    ])


def water_comparison_section(relationships, partial, grids, stations, arms, table_dir):
    """No new selection: expose all windows/grids and label retrospective examples."""
    paired = paired_water_comparisons(relationships)
    paired.to_csv(table_dir / 'paired_water_grid_comparisons.csv', index=False)
    names = grids.set_index('grid_id').grid_name.to_dict()
    water = stations[stations.station_type.eq('water_level')]
    station_names = water.set_index('station_id').name.to_dict()
    order = [BASELINE] + [g for g in grids.grid_id if g != BASELINE]
    windows = {f'rain_{a}_{b}h_mm': f'{a}–{b}' for a, b in [(0, 3), (3, 6), (6, 12), (12, 24)]}
    sections = ['''### 7.1. Đối chiếu đầy đủ ô A Lưới và các ô mới tại cùng trạm nước

Các bảng dưới trả lời trực tiếp câu hỏi: với cùng một trạm nước, dự báo mưa từ ô mới có tương quan với mức nước thay đổi sau +6/+12 giờ khác ô A Lưới cũ hay không? Mỗi hàng giữ cùng trạm, horizon, khoảng mưa, phạm vi thời gian và số cặp hợp lệ trên cả bảy ô. Các giá trị dự báo tại một mốc đều lấy từ cùng chu kỳ khởi tạo. Những mốc có nhãn vượt ranh giới thời gian đã bị loại khỏi phần huấn luyện như các phần trên.

**Cách đọc:** mỗi ô số liệu ghi **Pearson / Spearman**, giữ nguyên dấu. Pearson mô tả liên hệ tuyến tính; Spearman mô tả liên hệ theo thứ hạng. Ví dụ `0,1401 / −0,0927` cho biết hai phép đo chưa đồng thuận về chiều liên hệ. Số âm không tự động nghĩa là dữ liệu vô ích; số dương lớn hơn cũng không đồng nghĩa mô hình sẽ dự báo tốt hơn. Dấu chấm trong bảng là dấu thập phân.

Cột A Lưới là đối chứng cũ. Khoảng mưa tính tương đối từ mốc phát dự báo; nhãn nước là ΔH = H(t+h) − H(t). Tất cả bốn khoảng mưa được trình bày, không chọn riêng khoảng có hệ số đẹp nhất. Các nhãn “ô gần trạm” chỉ nói đến ô gần nhất trong bảy ô đã thu thập, chưa xác minh cùng lưu vực hoặc ô API chọn tại tọa độ chính xác của trạm.

[Bảng đối chiếu máy đọc được](tables/paired_water_grid_comparisons.csv) còn lưu chênh lệch hệ số có dấu và chênh lệch độ lớn so với A Lưới, số mẫu, nguồn và các hệ số gốc. Đây là cùng bộ số liệu của bảng quan hệ đầy đủ, không phải một lượt tuyển chọn đặc trưng mới.
''']
    for station in water.itertuples():
        local = arms[arms.water_station_id.eq(station.station_id) & arms.arm.eq('W2')].iloc[0]
        sections.append(f'#### {station.name} — mã {station.station_id}\n\nÔ gần nhất trong tập đã thu thập là **{names[local.grid_ids]}**, cách trạm khoảng **{local.local_proxy_distance_km:.1f} km**.\n')
        for scope, title in [('train', 'Phần huấn luyện'), ('evaluation', 'Phần đánh giá thăm dò')]:
            part = paired[paired.water_station_id.eq(station.station_id) & paired.scope.eq(scope)]
            rows = []
            for horizon in [6, 12]:
                for feature in WINDOWS:
                    group = part[part.horizon_h.eq(horizon) & part.feature.eq(feature)].set_index('grid_id')
                    if len(group) != len(order):
                        raise ValueError(f'Incomplete grid table for {station.station_id}/{scope}/{horizon}/{feature}')
                    rows.append([f'+{horizon}h', windows[feature], int(group.n.iloc[0]), *[
                        f'{_number(group.loc[g, "pearson_r"])} / {_number(group.loc[g, "spearman_r"])}' for g in order]])
            sections.append(f'**{title}.**\n\n' + _table(
                ['Nhãn nước', 'Mưa dự báo (giờ)', 'N', *[names[g] + (' (cũ)' if g == BASELINE else '') for g in order]], rows) + '\n')
            if part.n.max() == 0:
                sections.append('Không có cặp hợp lệ trong phạm vi này; dấu “—” không phải hệ số tương quan bằng 0. Không dùng dữ liệu đánh giá để tạo thay kết quả huấn luyện.\n')
    sections.append('### 7.2. Những dấu hiệu đáng chú ý và giới hạn của chúng\n')
    train = paired[paired.scope.eq('train') & paired.n.ge(100)]
    evaluation = paired[paired.scope.eq('evaluation')]
    def matching(frame, row):
        mask = pd.Series(True, index=frame.index)
        for key in ['water_station_id', 'horizon_h', 'feature', 'grid_id']:
            mask &= frame[key].eq(row[key])
        result = frame.loc[mask]
        return result.iloc[0] if len(result) else None
    strongest = train.loc[train.pearson_r.idxmax()]
    check = matching(evaluation, strongest)
    strongest_rank = train.loc[train.spearman_r.idxmax()]
    check_rank = matching(evaluation, strongest_rank)
    sections.append(
        f'**Đối chiếu trường hợp có Pearson lớn nhất trên phần huấn luyện.** Trong các hàng có ít nhất 100 cặp, '
        f'{strongest.water_station_name}, ΔH +{int(strongest.horizon_h)}h, mưa {windows[strongest.feature]} giờ tại ô '
        f'{names[strongest.grid_id]} có Pearson **{_number(strongest.pearson_r)}**, nhưng Spearman chỉ **{_number(strongest.spearman_r)}** '
        f'(N={int(strongest.n)}). Với đúng tổ hợp đó trên phần đánh giá, hai hệ số lần lượt là '
        f'**{_number(check.pearson_r)} / {_number(check.spearman_r)}**, N={int(check.n)}. '
        'Sự khác nhau giữa Pearson và Spearman là dấu hiệu cần kiểm tra theo từng đợt và phân phối biến thiên; '
        'chưa thể kết luận nguyên nhân là một trận cực trị hay một quá trình thủy văn cụ thể.\n')
    sections.append(
        f'**Một số liên hệ theo thứ hạng khá rõ nhưng không đồng thuận với liên hệ tuyến tính.** '
        f'Tại {strongest_rank.water_station_name}, ΔH +{int(strongest_rank.horizon_h)}h, '
        f'mưa {windows[strongest_rank.feature]} giờ tại ô {names[strongest_rank.grid_id]}, '
        f'Pearson/Spearman trên huấn luyện là **{_number(strongest_rank.pearson_r)} / {_number(strongest_rank.spearman_r)}** '
        f'(N={int(strongest_rank.n)}); trên đánh giá là '
        f'**{_number(check_rank.pearson_r)} / {_number(check_rank.spearman_r)}** (N={int(check_rank.n)}). '
        'Vì vậy, chỉ đọc một loại hệ số sẽ bỏ qua sự khác biệt quan trọng. Đây là ví dụ được chọn bằng phần huấn luyện, '
        'không phải bằng chứng một mô hình phi tuyến chắc chắn sẽ khai thác được tín hiệu này.\n')
    residual = partial[partial.control_set.eq('plus_existing_a_luoi')]
    fit = residual[residual.scope.eq('train') & residual.n.ge(100)]
    top = fit.loc[fit.partial_pearson_r.idxmax()]
    follow = matching(residual[residual.scope.eq('evaluation')], top)
    sections.append(
        f'**Có một ứng viên bổ sung đáng chú ý sau khi đã xét ô A Lưới.** '
        f'Tại {station_names[top.water_station_id]}, ΔH +{int(top.horizon_h)}h, mưa {windows[top.feature]} giờ từ ô '
        f'{names[top.grid_id]}, tương quan riêng phần Pearson/Spearman là '
        f'**{_number(top.partial_pearson_r)} / {_number(top.partial_spearman_r)}** trên huấn luyện (N={int(top.n)}), '
        f'và **{_number(follow.partial_pearson_r)} / {_number(follow.partial_spearman_r)}** trên đánh giá (N={int(follow.n)}). '
        'Pearson còn tín hiệu, nhưng Spearman không xác nhận sự ổn định về chiều liên hệ. '
        'Nên giữ đây là một giả thuyết để kiểm chứng bằng ablation, chưa khẳng định tín hiệu mạnh và bền vững.\n')
    # This fixed illustration was requested before inspecting the new report tables.
    dv = paired[paired.water_station_id.eq('0074647403') & paired.feature.eq('rain_6_12h_mm') & paired.horizon_h.eq(12)]
    dv_rows = []
    for gid in order:
        a = dv[dv.grid_id.eq(gid) & dv.scope.eq('train')].iloc[0]
        b = dv[dv.grid_id.eq(gid) & dv.scope.eq('evaluation')].iloc[0]
        dv_rows.append([names[gid], int(a.n), _number(a.pearson_r), _number(a.spearman_r), int(b.n), _number(b.pearson_r), _number(b.spearman_r)])
    sections.append('**Tại Dã Viên, ô mới có thể cho Pearson lớn hơn nhưng chưa có sự xác nhận đồng nhất.** '
                    'Bảng sau giữ cố định ΔH +12h và mưa dự báo 6–12 giờ, đúng ví dụ đã thảo luận trước khi bổ sung báo cáo. '
                    'Cần đọc đồng thời hai hệ số và hai giai đoạn; không chọn ô chỉ vì một hệ số huấn luyện lớn hơn A Lưới.\n\n' +
                    _table(['Ô mưa', 'N huấn luyện', 'Pearson huấn luyện', 'Spearman huấn luyện',
                            'N đánh giá', 'Pearson đánh giá', 'Spearman đánh giá'], dv_rows) + '\n')
    sections.append('**Chưa xác nhận một ô mới vượt ô cũ một cách ổn định cho toàn bộ trạm.** '
                    'Các ví dụ cực đại ở trên là thống kê mô tả sau khi xem nhiều trạm, ô và cửa sổ, có thể lạc quan do nhiều phép so sánh. '
                    'Ngưỡng 100 cặp chỉ tránh diễn giải cực đại từ mẫu quá nhỏ, không phải ngưỡng ý nghĩa thống kê. '
                    'Các cửa sổ chồng lấn và các trạm cùng chịu một đợt mưa không phải những mẫu độc lập. '
                    'Chưa có kiểm định theo sự kiện hoặc khoảng tin cậy để gọi một chênh lệch là có ý nghĩa. '
                    'Phần đánh giá chỉ dùng kiểm tra hồi cứu các tổ hợp đã chỉ ra trên train; không dùng để sửa danh sách W0–W4. '
                    'Bước tiếp theo có cơ sở nhất vẫn là ablation trên cùng anchor, cùng đầu vào nước/mưa và cùng cách chia thời gian.\n')
    return '\n\n'.join(sections)
