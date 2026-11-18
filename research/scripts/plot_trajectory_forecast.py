"""Observed level vs multi-horizon forecast trajectories (+1..+H h) issued every H hours.

Usage: plot_trajectory_forecast.py SCORES.npz SAMPLES.parquet OUT.png [--epoch N | --last 10] [--phase 0] [--label TEXT]
SCORES.npz: research score output (pred [epochs, samples, horizons], y, split, anchor ns UTC, epochs, horizons);
SAMPLES.parquet: the dataset's samples (for the level at issue time, water_now_mm).
Each issue time t with VN hour % H == PHASE draws one trajectory: observed level at t -> forecasts t+1..t+H.
Top: 31/07 -> 12/11; below: zoom windows on storm / flood events. Forecasts with |error| > 200 mm are ringed.
"""
import argparse

import matplotlib

matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('scores'); ap.add_argument('samples'); ap.add_argument('out')
ap.add_argument('--epoch', type=int); ap.add_argument('--last', type=int, default=10)
ap.add_argument('--phase', type=int, default=0); ap.add_argument('--label', default='')
ap.add_argument('--station', default='Dã Viên')
A = ap.parse_args()
TZ = 'Asia/Ho_Chi_Minh'
OBS, FC, CRIT = '#2a78d6', '#eb6834', '#d03b3b'          # categorical slots 1-2 (validated), status critical
INK, INK2, GRID, SURF, SHADE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb', '#efeeea'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.titlecolor': INK, 'figure.facecolor': SURF, 'axes.facecolor': SURF})

z = np.load(A.scores)
hz = list(z['horizons']); H = max(hz)
if hz != list(range(1, H + 1)):
    raise ValueError(f'needs horizons 1..H, got {hz}')
if A.epoch is not None:
    pred = z['pred'][list(z['epochs']).index(A.epoch)]; which = f'epoch {A.epoch}'
else:
    pred = z['pred'][-A.last:].mean(0); which = f'trung bình {A.last} epoch cuối'
issue = pd.to_datetime(z['anchor'], utc=True).tz_convert(TZ)
smp = pd.read_parquet(A.samples)
now = smp.set_index(pd.to_datetime(smp.anchor_utc, utc=True)).water_now_mm
now = now[~now.index.duplicated()].reindex(issue.tz_convert('UTC')).to_numpy()
split = z['split']; Y = z['y']
# observed hourly level from the +1 h targets (level at issue + 1 h)
obs = pd.Series(Y[:, 0], index=issue + pd.Timedelta(hours=1)).sort_index()
obs = obs[~obs.index.duplicated()].asfreq('h')
keep = (issue.hour % H == A.phase) & np.isin(split, ['train', 'validation', 'purged', 'test', 'new'])
idx = np.flatnonzero(keep)
out_train = np.isin(split[idx], ['validation', 'purged', 'test', 'new'])
err = pred[idx] - Y[idx]
n_err = int((np.abs(err[out_train]) > 200).any(axis=1).sum())
n_pts = int((np.abs(err[out_train]) > 200).sum())
hours = ', '.join(f'{x:02d}' for x in range(24) if x % H == A.phase)


def style(ax):
    ax.grid(axis='y', color=GRID, lw=.6)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def draw(ax, lo, hi, zoom):
    o = obs[lo:hi]
    ax.plot(o.index, o.values, color=OBS, lw=1.3 if zoom else .8, label='Mực nước thực tế', zorder=2)
    first = True
    for i in idx:
        t0 = issue[i]
        if t0 < lo or t0 > hi:
            continue
        tt = [t0] + [t0 + pd.Timedelta(hours=h) for h in hz]
        vv = [now[i]] + list(pred[i])
        ax.plot(tt, vv, color=FC, lw=1.1 if zoom else .7, zorder=3, label=f'Dự báo +1…+{H} h' if first else None)
        if zoom:
            ax.scatter(tt[1:], vv[1:], s=10, color=FC, zorder=3)
            ax.scatter([t0], [now[i]], s=14, color=SURF, edgecolor=FC, linewidth=1, zorder=4,
                       label='Thời điểm phát hành' if first else None)
        bad = np.abs(pred[i] - Y[i]) > 200
        if bad.any():
            ax.scatter(np.array(tt[1:])[bad], pred[i][bad], s=90 if zoom else 45, facecolor='none', edgecolor=CRIT,
                       linewidth=1.5, zorder=5, label='Sai > 200 mm' if not getattr(ax, '_bad', False) else None)
            ax._bad = True
        first = False
    style(ax)


fig = plt.figure(figsize=(14, 10.5))
gs = fig.add_gridspec(3, 3, height_ratios=[1.25, 1, 1], hspace=.45, wspace=.18)
ax0 = fig.add_subplot(gs[0, :])
lo, hi = pd.Timestamp('2026-07-31', tz=TZ), issue[idx].max() + pd.Timedelta(hours=H)
for sp, name, shade in (('validation', 'val', True), ('test', 'test', False), ('new', 'mới (15/10–12/11)', True)):
    m = split == sp
    a, b = issue[m].min(), issue[m].max()
    if shade:
        ax0.axvspan(a, b, color=SHADE, zorder=0, lw=0)
    ax0.text(a + (b - a) / 2, 1.0, name, transform=ax0.get_xaxis_transform(), ha='center', va='bottom', color=INK2, fontsize=8)
draw(ax0, lo, hi, False)
ax0.set_ylabel('mm'); ax0.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m', tz=TZ))
ax0.legend(loc='upper left', frameon=False, fontsize=8)
ax0.set_title(f'{A.station}: dự báo +1…+{H} h phát hành mỗi {H} giờ ({hours} giờ VN), {which}{A.label}. '
              f'Ngoài train: {n_err} lần phát hành có điểm sai > 200 mm ({n_pts} điểm)', loc='left', fontsize=10, pad=16)
events = [('Bão 11–13/09 (val)', '2026-09-10 12:00', '2026-09-14 12:00'), ('20–22/09 (test)', '2026-09-20 06:00', '2026-09-22 06:00'),
          ('05–07/10 (test)', '2026-10-05 06:00', '2026-10-07 18:00'), ('12–14/10 (test)', '2026-10-12 06:00', '2026-10-14 06:00'),
          ('28–30/10 (mới)', '2026-10-28 00:00', '2026-10-30 00:00'), ('Lũ 09–11/11 (mới)', '2026-11-09 00:00', '2026-11-11 18:00')]
for k, (title, a, b) in enumerate(events):
    ax = fig.add_subplot(gs[1 + k // 3, k % 3])
    a, b = pd.Timestamp(a, tz=TZ), pd.Timestamp(b, tz=TZ)
    draw(ax, a, b, True)
    sel = [i for i in idx if a <= issue[i] <= b]
    mx = np.abs(pred[sel] - Y[sel]).max() if sel else np.nan
    ax.set_title(f'{title}: sai lớn nhất {mx:.0f} mm', loc='left', fontsize=9)
    ax.set_xlim(a, b)
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12], tz=TZ))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m\n%H h', tz=TZ))
    ax.tick_params(labelsize=7.5)
    if k == 0:
        ax.legend(loc='upper left', frameon=False, fontsize=7)
fig.text(.01, .005, f'Mỗi đoạn cam: một lần phát hành, nối mực nước đo lúc phát hành (chấm rỗng) với các dự báo +1…+{H} h. Giờ VN.',
         fontsize=8, color=INK2)
fig.savefig(A.out, dpi=130, bbox_inches='tight')
print(A.out, 'issues with an error > 200 mm out of train:', n_err, 'points:', n_pts)
