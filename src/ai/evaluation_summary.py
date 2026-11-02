"""Physical-unit station metrics and explicitly equal-station averages."""
import numpy as np
import pandas as pd


def station_error_table(observed, predicted, station_ids, station_mapping, target_columns):
    observed, predicted = np.asarray(observed), np.asarray(predicted)
    station_ids = np.asarray(station_ids)
    if (observed.shape != predicted.shape or observed.ndim != 2 or len(observed) == 0
            or observed.shape != (len(station_ids), len(target_columns))
            or not np.isfinite(observed).all() or not np.isfinite(predicted).all()):
        raise ValueError('Expected finite, aligned [samples,horizons] predictions and station IDs')
    rows = []
    for sid in sorted(set(station_ids)):
        error = predicted[station_ids == sid]-observed[station_ids == sid]
        row = dict(station_id=sid, name=station_mapping.get(sid, {}).get('name', sid), n=len(error))
        for i, column in enumerate(target_columns):
            minutes = int(column.rsplit('_', 1)[-1][:-1])
            hours = f'{minutes/60:g}'
            row[f'rmse_h{hours}_mm'] = float(np.sqrt(np.mean(error[:, i]**2)))
            row[f'mae_h{hours}_mm'] = float(np.mean(abs(error[:, i])))
        row['rmse_all_mm'] = float(np.sqrt(np.mean(error**2)))
        row['mae_all_mm'] = float(np.mean(abs(error)))
        rows.append(row)
    table = pd.DataFrame(rows).set_index('station_id')
    metrics = [c for c in table if c.startswith(('rmse_', 'mae_'))]
    table.loc['mean_across_stations', 'name'] = 'Mean across stations (equal weight)'
    table.loc['mean_across_stations', 'n'] = len(observed)
    table.loc['mean_across_stations', metrics] = table.loc[table.index != 'mean_across_stations', metrics].mean()
    table['n'] = table['n'].astype(int)
    return table
