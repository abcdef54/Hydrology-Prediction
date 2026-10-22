"""Summarize saved forecast absolute errors over a Vietnam-local date range.

Use the existing read-only forecast/observation snapshot. Water truth is
mean of six valid readings in (t-1h,t]. Saved forecasts may be retrospective.
"""
import argparse
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path('/tmp/hydrology-rain-regimes'))
    parser.add_argument('--start-date', default='2026-09-01')
    parser.add_argument('--end-date', default=None, help='Inclusive local date; default today')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    now = pd.Timestamp.now(tz='Asia/Ho_Chi_Minh')
    start = pd.Timestamp(args.start_date, tz='Asia/Ho_Chi_Minh')
    end_day = pd.Timestamp(args.end_date, tz='Asia/Ho_Chi_Minh') if args.end_date else now.normalize()
    cutoff = min(end_day + pd.Timedelta(days=1), now)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    p = pd.read_csv(args.input_dir / 'forecasts.csv')
    p['forecast_at'] = pd.to_datetime(p.forecast_at, utc=True)
    p['anchor_observed_at'] = pd.to_datetime(p.anchor_observed_at, utc=True)
    p = p[(p.forecast_at >= start) & (p.forecast_at < cutoff)].copy()
    obs = pd.read_csv(args.input_dir / 'observations.csv', dtype={'station_id': str})
    obs = obs[obs.station_id.eq('0074647403')].copy()
    obs['observed_at'] = pd.to_datetime(obs.observed_at, utc=True)
    bins = obs.set_index('observed_at').depth.resample('1h', closed='right', label='right')
    truth = bins.mean().where(bins.count().eq(6))
    p['actual'] = p.forecast_at.map(truth)
    p['error'] = p.water_level_mm - p.actual
    p['abs_error'] = p.error.abs()
    p['local_target'] = p.forecast_at.dt.tz_convert('Asia/Ho_Chi_Minh')
    valid = p.dropna(subset=['actual']).copy()
    assert not valid.duplicated(['forecast_at', 'horizon_minutes']).any()
    assert ((valid.forecast_at-valid.anchor_observed_at).dt.total_seconds() == valid.horizon_minutes*60).all()
    bin_width = 25
    upper = max(550, int(np.ceil(valid.abs_error.max()/bin_width))*bin_width)
    edges = np.arange(0, upper+bin_width, bin_width)
    ymax = max(np.histogram(g.abs_error, bins=edges)[0].max() for _,g in valid.groupby('horizon_minutes'))
    stats, counts = [], []
    for minutes, all_points in p.groupby('horizon_minutes'):
        g = all_points.dropna(subset=['actual'])
        e = g.abs_error
        mean = e.mean()
        n = len(e)
        h = int(minutes//60)
        row = dict(horizon_hours=h, saved_points=len(all_points), valid_points=n,
            excluded_truth=len(all_points)-n, above_200=int(e.gt(200).sum()),
            above_200_pct=e.gt(200).mean()*100, min_mm=e.min(), max_mm=e.max(),
            mae_mm=mean, le_mae=int(e.le(mean).sum()), le_mae_pct=e.le(mean).mean()*100,
            gt_mae=int(e.gt(mean).sum()), gt_mae_pct=e.gt(mean).mean()*100,
            bias_mm=g.error.mean(), first_target=str(g.local_target.min()), last_target=str(g.local_target.max()))
        stats.append(row)
        hist,_ = np.histogram(e,bins=edges)
        assert hist.sum()==n
        assert row['le_mae']+row['gt_mae']==n
        counts.extend(dict(horizon_hours=h,left_mm=int(a),right_mm=int(b),count=int(c)) for a,b,c in zip(edges[:-1],edges[1:],hist))
        fig, ax = plt.subplots(figsize=(10,5))
        ax.hist(e,bins=edges,color='#1677b8',edgecolor='white',linewidth=.8)
        ax.axvline(200,color='#ce4525',linestyle='--',label='200 mm')
        ax.axvline(mean,color='#308047',label=f'MAE = {mean:.2f} mm')
        ax.set(title=f'Da Vien | +{h}h | {start:%d %b %Y} – {g.local_target.max():%d %b %Y %H:%M} (VN)',
            xlabel='Absolute error |forecast − observed| (mm)',ylabel='Number of samples',
            xlim=(0,upper),ylim=(0,ymax+max(3,ymax*.05)),xticks=np.arange(0,upper+1,50))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True));ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        ax.legend();ax.text(.98,.76,f'n = {n}\nBin width = 25 mm\nError >200 mm: {row["above_200"]} ({row["above_200_pct"]:.2f}%)',transform=ax.transAxes,ha='right',va='top')
        fig.tight_layout();fig.savefig(out/f'absolute_error_{h:02d}h.png',dpi=180);fig.savefig(out/f'absolute_error_{h:02d}h.svg');plt.close(fig)
    stats = pd.DataFrame(stats)
    stats.to_csv(out/'statistics.csv',index=False)
    pd.DataFrame(counts).to_csv(out/'histogram_counts.csv',index=False)
    p.to_csv(out/'forecast_points.csv',index=False)
    valid['day'] = valid.local_target.dt.strftime('%Y-%m-%d')
    daily = valid.assign(over_200=valid.abs_error.gt(200)).groupby('day').agg(total=('error','size'),above_200=('over_200','sum'),mae_mm=('abs_error','mean'))
    daily['above_200_pct'] = daily.above_200/daily.total*100
    daily.to_csv(out/'daily_statistics.csv')
    metadata = dict(start_local=str(start),requested_cutoff_local=str(cutoff),
        latest_observation_local=str(obs.observed_at.max().tz_convert('Asia/Ho_Chi_Minh')),
        latest_scored_target_local=str(valid.local_target.max()),
        model_versions=p.model_version.unique().tolist(), pipeline_versions=p.pipeline_version.unique().tolist(),
        truth='Mean of exactly six valid observations in (t-1h,t]; incomplete hours excluded',
        error='absolute difference in mm; strict >200; <= MAE uses unrounded horizon-specific mean',
        bin_width_mm=25, bin_rule='[left,right), last bin includes right edge',
        caveats=['Saved forecasts can be backfilled, not necessarily operational forecasts',
                 'Hourly samples overlap; no independent-event significance claim'])
    (out/'manifest.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    def table(df):
        df=df.copy()
        for col in df.select_dtypes('number'): df[col]=df[col].round(2)
        return '| '+' | '.join(df.columns)+' |\n|'+'|'.join(['---']*len(df.columns))+'|\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in df.itertuples(index=False,name=None))
    report='# Sai số tuyệt đối Dã Viên từ 01/09/2026 đến hôm nay\n\n'+json.dumps(metadata,ensure_ascii=False,indent=2)+'\n\n'+table(stats)+'\n\n## Ngày có nhiều điểm sai số >200 mm\n\n'+table(daily.sort_values('above_200',ascending=False).head(10).reset_index())
    (out/'report.md').write_text(report)
    print(stats.round(2).to_string(index=False))
    print('TOP_DAYS\n'+daily.sort_values('above_200',ascending=False).head(8).round(2).to_string())
    print(json.dumps(metadata,ensure_ascii=False))


if __name__=='__main__':
    main()
