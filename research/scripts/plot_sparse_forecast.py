"""Observed level vs a forecast issued only every H hours (e.g. +6 h at 00/06/12/18 VN), drawn at its valid time.

Usage: plot_sparse_forecast.py PREDS.parquet H PHASE OUT.png [--station NAME] [--label TEXT]
PREDS.parquet: research/scripts/member_ensemble.py output (anchor, split, h, y, pred, err).
PHASE: the issue hours kept are the Vietnam-time hours with hour % H == PHASE.
Top: 31/07 -> 12/11 (val, test, new). Below: zoom windows on the storm / flood events. Errors > 200 mm are ringed.
"""
import argparse

import matplotlib

matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('preds'); ap.add_argument('h', type=int); ap.add_argument('phase', type=int); ap.add_argument('out')
ap.add_argument('--station', default='Dã Viên'); ap.add_argument('--label', default='models/da_vien/v1')
A = ap.parse_args()
TZ = 'Asia/Ho_Chi_Minh'
OBS, FC, CRIT = '#2a78d6', '#eb6834', '#d03b3b'          # categorical slots 1-2 (validated), status critical
INK, INK2, GRID, SURF, SHADE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb', '#efeeea'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.titlecolor': INK, 'figure.facecolor': SURF, 'axes.facecolor': SURF})

D = pd.read_parquet(A.preds)
D = D[(D.h == A.h) & D.split.isin(['train', 'validation', 'purged', 'test', 'new'])].copy()
D['issue'] = D.anchor.dt.tz_convert(TZ)
D['valid'] = D.issue + pd.Timedelta(hours=A.h)
# observed hourly level at the valid times of the hourly forecasts (y of every issue time = level at issue + h)
obs = D.set_index('valid').y.sort_index()
obs = obs[~obs.index.duplicated()].asfreq('h')
S = D[D.issue.dt.hour % A.h == A.phase].set_index('valid').sort_index()
out = S.split != 'train'
hours = ', '.join(f'{x:02d}' for x in range(24) if x % A.h == A.phase)
n_err = int((S.err.abs()[out] > 200).sum())


def style(ax):
    ax.grid(axis='y', color=GRID, lw=.6)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def draw(ax, lo, hi, zoom):
    o = obs[lo:hi]; s = S[lo:hi]
    ax.plot(o.index, o.values, color=OBS, lw=1.2 if zoom else .8, label='Mực nước thực tế', zorder=2)
    if zoom:
        ax.vlines(s.index, s.y, s.pred, color=INK2, lw=.8, alpha=.6, zorder=1)
    # line through consecutive forecasts only (a gap longer than one issue step breaks it)
    line = s.pred.reindex(pd.date_range(s.index.min(), s.index.max(), freq=f'{A.h}h')) if len(s) else s.pred
    ax.plot(line.index, line.values, color=FC, lw=1.2 if zoom else .8, zorder=3)
    ax.scatter(s.index, s.pred, s=22 if zoom else 6, color=FC, edgecolor=SURF, linewidth=.6 if zoom else 0,
               label=f'Dự báo +{A.h} h (phát hành {hours} giờ VN)', zorder=3)
    bad = s[s.err.abs() > 200]
    if len(bad):
        ax.scatter(bad.index, bad.pred, s=110 if zoom else 60, facecolor='none', edgecolor=CRIT, linewidth=1.6,
                   label='Sai > 200 mm', zorder=4)
        if zoom:
            for t, r in bad.iterrows():
                ax.annotate(f'{r.err:+.0f} mm', (t, r.pred), xytext=(6, 6 if r.err > 0 else -12),
                            textcoords='offset points', fontsize=8, color=INK)
    style(ax)


fig = plt.figure(figsize=(14, 10.5))
gs = fig.add_gridspec(3, 3, height_ratios=[1.25, 1, 1], hspace=.45, wspace=.18)
ax0 = fig.add_subplot(gs[0, :])
lo, hi = pd.Timestamp('2026-07-31', tz=TZ), S.index.max()
for sp, g in S.groupby('split'):
    if sp in ('validation', 'new'):
        ax0.axvspan(g.index.min(), g.index.max(), color=SHADE, zorder=0, lw=0)
    if sp in ('validation', 'test', 'new'):
        ax0.text(g.index.min() + (g.index.max() - g.index.min()) / 2, 1.0, {'validation': 'val', 'test': 'test', 'new': 'mới (15/10–12/11)'}[sp],
                 transform=ax0.get_xaxis_transform(), ha='center', va='bottom', color=INK2, fontsize=8)
draw(ax0, lo, hi, False)
ax0.set_ylabel('mm'); ax0.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m', tz=TZ))
ax0.legend(loc='upper left', frameon=False, fontsize=8)
ax0.set_title(f'{A.station}: dự báo +{A.h} h chỉ phát hành mỗi {A.h} giờ ({hours} giờ VN), {A.label}. '
              f'Lỗi > 200 mm ngoài train: {n_err} / {int(out.sum())} lần dự báo', loc='left', fontsize=10, pad=16)
events = [('Bão 11–13/09 (val)', '2026-09-10 12:00', '2026-09-14 12:00'), ('20–22/09 (test)', '2026-09-20 06:00', '2026-09-22 06:00'),
          ('05–07/10 (test)', '2026-10-05 06:00', '2026-10-07 06:00'), ('12–14/10 (test)', '2026-10-12 06:00', '2026-10-14 06:00'),
          ('28–30/10 (mới)', '2026-10-28 00:00', '2026-10-30 00:00'), ('Lũ 09–11/11 (mới)', '2026-11-09 00:00', '2026-11-11 18:00')]
for k, (title, a, b) in enumerate(events):
    ax = fig.add_subplot(gs[1 + k // 3, k % 3])
    a, b = pd.Timestamp(a, tz=TZ), pd.Timestamp(b, tz=TZ)
    draw(ax, a, b, True)
    s = S[a:b]
    e = s.err.abs().max() if len(s) else np.nan
    ax.set_title(f'{title}: sai lớn nhất {e:.0f} mm', loc='left', fontsize=9)
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12], tz=TZ))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m\n%H h', tz=TZ))
    ax.tick_params(labelsize=7.5)
fig.text(.01, .005, 'Chấm và đường cam: các giá trị dự báo đặt tại giờ hiệu lực (giờ phát hành + {} h), nối liền các lần phát hành liên tiếp; vạch xám: sai số so với thực tế. '
         'Giờ VN.'.format(A.h), fontsize=8, color=INK2)
fig.savefig(A.out, dpi=130, bbox_inches='tight')
print(A.out, 'errors out of train:', n_err)
