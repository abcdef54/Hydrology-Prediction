"""Analyze saved forecasts by observed rain patterns; never trains or writes to DB.

Inputs: forecasts.csv, observations.csv and stations.csv exported from a read-only
repeatable-read database transaction. Rain sums preserve current pipeline semantics;
these groups are geographic proxies, not delineated contributing catchments.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

WEST = ['750062', '750007', '750003', '750011']
SOUTH = ['750010', '228005', '750066']


def regime(west, south, high, low):
    return np.select(
        [(west <= low) & (south >= high), (west >= high) & (south <= low),
         (west >= high) & (south >= high), (west <= low) & (south <= low)],
        ['south_high_west_low', 'west_high_south_low', 'both_high', 'both_low'],
        default='intermediate',
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    obs = pd.read_csv(args.input_dir / 'observations.csv', dtype={'station_id': str})
    obs['observed_at'] = pd.to_datetime(obs.observed_at, utc=True)
    forecasts = pd.read_csv(args.input_dir / 'forecasts.csv')
    for col in ['anchor_observed_at', 'forecast_at', 'generated_at']:
        forecasts[col] = pd.to_datetime(forecasts[col], utc=True)
    assert not forecasts.duplicated(['anchor_observed_at', 'horizon_minutes']).any()
    assert ((forecasts.forecast_at - forecasts.anchor_observed_at).dt.total_seconds()
            == forecasts.horizon_minutes * 60).all()
    grid = pd.date_range(obs.observed_at.min().floor('h'), obs.observed_at.max().ceil('h'), freq='h')
    rain = pd.DataFrame(index=grid)
    coverage = pd.DataFrame(index=grid)
    for sid in WEST + SOUTH:
        series = obs.loc[obs.station_id.eq(sid)].set_index('observed_at').depth
        grouped = series.resample('1h', closed='right', label='right')
        rain[sid] = grouped.sum(min_count=1).reindex(grid)
        coverage[sid] = grouped.count().reindex(grid).fillna(0)
    # Require all expected 144 ten-minute readings at every group station.
    rolling = rain.rolling(24, min_periods=24).sum()
    complete = coverage.eq(6).rolling(24, min_periods=24).sum().eq(24)
    groups = pd.DataFrame(index=grid)
    for name, ids in [('west', WEST), ('south', SOUTH)]:
        groups[name] = rolling[ids].mean(axis=1).where(complete[ids].all(axis=1))
    water = obs.loc[obs.station_id.eq('0074647403')].set_index('observed_at').depth
    water_bins = water.resample('1h', closed='right', label='right')
    truth = water_bins.mean().where(water_bins.count().eq(6))
    points = forecasts.copy()
    points['actual'] = points.forecast_at.map(truth)
    points['anchor_level'] = points.anchor_observed_at.map(truth)
    points['error'] = points.water_level_mm - points.actual
    points['abs_error'] = points.error.abs()
    points['persistence_abs_error'] = (points.anchor_level - points.actual).abs()
    for phase, timecol in [('known', 'anchor_observed_at'), ('target24', 'forecast_at')]:
        for name in ['west', 'south']:
            points[f'{phase}_{name}'] = points[timecol].map(groups[name])
    points['scope'] = np.select(
        [points.anchor_observed_at >= pd.Timestamp('2026-09-30T05:00Z'),
         points.anchor_observed_at >= pd.Timestamp('2026-08-13T07:00Z')],
        ['after_dataset', 'historical_test'], default='earlier_exploratory',
    )
    # after_dataset is after the local manifest, not a claim of verified training provenance.
    summaries = []
    for high, low in [(20, 10), (50, 10), (20, 5)]:
        for phase in ['known', 'target24']:
            valid = points.dropna(subset=['actual', 'anchor_level', f'{phase}_west', f'{phase}_south']).copy()
            valid['regime'] = regime(valid[f'{phase}_west'], valid[f'{phase}_south'], high, low)
            for scope in ['all', 'after_dataset', 'historical_test']:
                selected = valid if scope == 'all' else valid.loc[valid.scope.eq(scope)]
                for (horizon, label), part in selected.groupby(['horizon_minutes', 'regime']):
                    # Distinct local target days help expose overlapping hourly samples.
                    days = part.forecast_at.dt.tz_convert('Asia/Ho_Chi_Minh').dt.date.nunique()
                    summaries.append(dict(high=high, low=low, phase=phase, scope=scope,
                        horizon_hours=horizon // 60, regime=label, n=len(part), days=days,
                        mae=part.abs_error.mean(), rmse=np.sqrt(np.mean(part.error**2)),
                        bias=part.error.mean(), median_ae=part.abs_error.median(),
                        p90_ae=part.abs_error.quantile(.9),
                        persistence_mae=part.persistence_abs_error.mean(),
                        underprediction_pct=part.error.lt(0).mean()*100))
    summary = pd.DataFrame(summaries)
    points.to_csv(args.output_dir / 'forecast_points.csv', index=False)
    summary.to_csv(args.output_dir / 'regime_metrics.csv', index=False)
    groups.to_csv(args.output_dir / 'hourly_group_rain.csv', index_label='timestamp_utc')
    examples = points.dropna(subset=['actual','target24_west','target24_south']).copy()
    examples['regime'] = regime(examples.target24_west, examples.target24_south, 20, 10)
    examples['local_target'] = examples.forecast_at.dt.tz_convert('Asia/Ho_Chi_Minh')
    examples['local_anchor'] = examples.anchor_observed_at.dt.tz_convert('Asia/Ho_Chi_Minh')
    examples['local_day'] = examples.local_target.dt.date
    # One largest-error example per target day/regime, then five different days.
    examples = examples.loc[examples.horizon_minutes.eq(1440)].sort_values('abs_error', ascending=False)
    examples = examples.drop_duplicates(['regime','local_day']).groupby('regime').head(5)
    examples.to_csv(args.output_dir / 'examples_24h.csv',index=False)
    metadata = dict(west_station_ids=WEST, south_station_ids=SOUTH,
        model_versions=points.model_version.unique().tolist(),
        pipeline_versions=points.pipeline_version.unique().tolist(),
        forecast_points=len(points), anchors=points.anchor_observed_at.nunique(),
        first_anchor=str(points.anchor_observed_at.min()), last_anchor=str(points.anchor_observed_at.max()),
        complete_truth_points=int(points.actual.notna().sum()),
        rules='24h source sums averaged over stations; every hour must contain exactly 6 valid readings at every group station; right-closed bins; missing is not zero',
        phases={'known':'24h ending at forecast anchor; available observations',
                'target24':'24h ending at target; retrospective, includes future relative to anchor'},
        caveats=['Geographic groups, not verified catchments', 'Source accumulation interval unverified',
                 'Hourly windows overlap; days are not independent storms',
                 'Saved forecasts can be retrospective; generated_at retained',
                 'Local dataset manifest split dates do not prove checkpoint training provenance'])
    (args.output_dir / 'manifest.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    from rain_regime_report import write_report
    write_report(args.output_dir)
    print(summary.loc[(summary.high.eq(20)) & summary.low.eq(10) & summary.horizon_hours.eq(24)].round(2).to_string(index=False))
    print('SENSITIVITY_AFTER_DATASET_TARGET24')
    print(summary.loc[summary.scope.eq('after_dataset') & summary.phase.eq('target24') & summary.horizon_hours.eq(24)].round(2).to_string(index=False))


if __name__ == '__main__':
    main()
