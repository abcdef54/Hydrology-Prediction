"""Observed level vs model forecast at one horizon over the whole period (train -> 12/11/2026).

Usage: plot_station_forecast.py PREDS.parquet EPOCH OUT.png [--label L] [--horizon H] [--station NAME] [--note TEXT]
PREDS.parquet: per-epoch predictions (columns epoch, split, anchor, now, y, pred; split in train / validation / purged /
test / new; other splits are dropped). The forecast is drawn at its valid time (anchor + H). Times in Vietnam time.
"""
import argparse

import matplotlib

matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('preds'); ap.add_argument('epoch', type=int); ap.add_argument('out')
ap.add_argument('--label'); ap.add_argument('--horizon', type=int, default=3); ap.add_argument('--station', default='Dã Viên')
ap.add_argument('--events', help='zoom windows "title|start|end;title|start|end" (VN time); default: 4 reference events')
ap.add_argument('--note', default='Khoảng trống 02/2026 và 06/2026: hai đoạn sensor đứng đã xóa (invalid_ranges).')
A = ap.parse_args()
PREDS, EPOCH, OUT, H = A.preds, A.epoch, A.out, A.horizon
LABEL = A.label or f'epoch {EPOCH}'
TZ = 'Asia/Ho_Chi_Minh'
OBS, FC = '#2a78d6', '#eb6834'                       # categorical slots 1-2 (validated, light surface)
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
SPLIT_NAMES = {'train': 'train', 'validation': 'val', 'purged': '', 'test': 'test', 'new': 'mới (15/10–12/11)'}
SPLIT_SHADE = {'train': None, 'validation': '#efeeea', 'test': None, 'new': '#efeeea'}

P = pd.read_parquet(PREDS)
P = P[(P.epoch == EPOCH) & P.split.isin(list(SPLIT_NAMES))].copy()
P['valid'] = (pd.to_datetime(P.anchor, utc=True) + pd.Timedelta(hours=H)).dt.tz_convert(TZ)
P = P.sort_values('valid').set_index('valid')
P['err'] = P.pred - P.y
full = pd.date_range(P.index.min(), P.index.max(), freq='h')
obs = P.y.reindex(full)                               # gaps stay gaps (excluded / missing hours)
fc = P.pred.reindex(full)
err = P.err.reindex(full)

plt.rcParams.update({'font.size': 9, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.titlecolor': INK, 'figure.facecolor': SURF, 'axes.facecolor': SURF})


def shade_splits(ax, label=False):
    for sp, g in P.groupby('split'):
        if SPLIT_SHADE.get(sp):
            ax.axvspan(g.index.min(), g.index.max(), color=SPLIT_SHADE[sp], zorder=0, lw=0)
        if label and SPLIT_NAMES.get(sp):
            ax.text(g.index.min() + (g.index.max() - g.index.min()) / 2, 1.0, SPLIT_NAMES[sp], transform=ax.get_xaxis_transform(),
                    ha='center', va='bottom', color=INK2, fontsize=8)


def style(ax):
    ax.grid(axis='y', color=GRID, lw=.6)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


# --- zoom windows: the three storm/flood events + a strong-tide test week
EVENTS = [('Lũ 15–21/11/2025 (train)', '2025-11-15', '2025-11-22'),
          ('Bão 10–14/09/2026 (val)', '2026-09-10', '2026-09-14 12:00'),
          ('Triều mạnh 26/09–02/10 (test)', '2026-09-26', '2026-10-02'),
          ('Lũ 07–12/11/2026 (mới)', '2026-11-07', '2026-11-12')]

if A.events:
    EVENTS = [tuple(e.split('|')) for e in A.events.split(';')]
zrows = (len(EVENTS) + 1) // 2
fig = plt.figure(figsize=(16, 6.6 + 3.2 * zrows))
gs = fig.add_gridspec(2 + zrows, 2, height_ratios=[2.2, 1.0] + [1.7] * zrows, hspace=.55, wspace=.12)
ax0 = fig.add_subplot(gs[0, :])
ax1 = fig.add_subplot(gs[1, :], sharex=ax0)
shade_splits(ax0, label=True)
ax0.plot(full, obs, color=OBS, lw=.6, label='Mực nước thực tế')
ax0.plot(full, fc, color=FC, lw=.6, alpha=.85, label=f'Dự báo +{H} h ({LABEL})')
ax0.set_ylabel('Mực nước (mm)')
for h in ax0.legend(loc='upper right', frameon=False, ncol=2).get_lines():
    h.set_linewidth(2.5)
ax0.set_title(f'{A.station}: mực nước thực tế và dự báo +{H} h, toàn bộ giai đoạn (dự báo vẽ tại giờ hiệu lực)', loc='left',
              fontsize=11, pad=16)
style(ax0)

shade_splits(ax1)
ax1.plot(full, err, color=INK2, lw=.5)
for v in (200, -200):
    ax1.axhline(v, color='#b3261e', lw=.8, ls='--')
ax1.text(full[-1], 205, '±200 mm', color='#b3261e', fontsize=8, va='bottom', ha='right')
big = err[err.abs() > 200]
ax1.scatter(big.index, big, s=14, color='#b3261e', zorder=3, edgecolor=SURF, lw=.8)
ax1.set_ylabel('Lỗi = dự báo − thực tế (mm)')
ax1.set_ylim(-max(500, err.abs().max() * 1.05), max(500, err.abs().max() * 1.05))
style(ax1)
ax1.xaxis.set_major_locator(mdates.MonthLocator(tz=P.index.tz))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y', tz=P.index.tz))

for i, (title, a, b) in enumerate(EVENTS):
    ax = fig.add_subplot(gs[2 + i // 2, i % 2])
    a, b = pd.Timestamp(a, tz=TZ), pd.Timestamp(b, tz=TZ)
    o, f, e = obs[a:b], fc[a:b], err[a:b]
    ax.plot(o.index, o, color=OBS, lw=1.6)
    ax.plot(f.index, f, color=FC, lw=1.6)
    bad = e[e.abs() > 200]
    ax.scatter(bad.index, f.reindex(bad.index), s=30, color='#b3261e', zorder=3, edgecolor=SURF, lw=1)
    n200, mae = int((e.abs() > 200).sum()), float(e.abs().mean())
    ax.set_title(f'{title} · lỗi > 200 mm: {n200} · MAE {mae:.0f} mm', loc='left', fontsize=9.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m', tz=P.index.tz))
    ax.xaxis.set_major_locator(mdates.DayLocator(tz=P.index.tz))
    if i % 2 == 0:
        ax.set_ylabel('mm')
    style(ax)

stats = []
for sp in ['train', 'validation', 'purged', 'test', 'new']:
    g = P[P.split == sp]
    if len(g):
        stats.append(f'{sp}: {int((g.err.abs() > 200).sum())} lỗi, MAE {g.err.abs().mean():.1f}')
fig.text(.01, .005, 'Lỗi > 200 mm và MAE theo tập: ' + ' · '.join(stats) + '. Chấm đỏ = lỗi > 200 mm.\n' +
         A.note, color=INK2, fontsize=8.5)
fig.savefig(OUT, dpi=130, bbox_inches='tight', facecolor=SURF)
print(OUT)
