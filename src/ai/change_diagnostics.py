"""Streaming per-station/horizon diagnostics; no per-batch device-to-host copy."""
import torch
import pandas as pd


class ChangeDiagnostics:
    fields = ('n', 'sse', 'sae', 'persistence_sse', 'normalized_sse', 'large_errors',
              'actual_abs_change', 'pred_abs_change', 'extreme_n', 'extreme_sse',
              'extreme_actual_abs', 'extreme_pred_abs', 'rise_n', 'rise_bias_sum',
              'rise_shortfall_sum', 'rise_under_n', 'predicted_rise_n', 'false_alarm_n',
              'true_positive_n', 'objective_sum')

    def __init__(self, station_count, horizons, device, threshold=200.):
        self.values = torch.zeros(station_count, horizons, len(self.fields), device=device, dtype=torch.float64)
        self.threshold = threshold

    @torch.no_grad()
    def update(self, predictions, targets, persistence, scales, ids, objective_elements=None):
        e = (predictions.double()-targets.double())*scales
        actual = (targets.double()-persistence.double())*scales
        pred = (predictions.double()-persistence.double())*scales
        extreme, rise, alarm = actual.abs() > self.threshold, actual > self.threshold, pred > self.threshold
        parts = (torch.ones_like(e), e.square(), e.abs(), actual.square(),
                 (predictions.double()-targets.double()).square(), (e.abs() > self.threshold),
                 actual.abs(), pred.abs(), extreme, e.square()*extreme, actual.abs()*extreme,
                 pred.abs()*extreme, rise, e*rise, (-e).clamp_min(0)*rise, (rise & (e < 0)),
                 alarm, alarm & ~rise, alarm & rise,
                 e.new_zeros(e.shape) if objective_elements is None else objective_elements)
        self.values.index_add_(0, ids, torch.stack([v.double() for v in parts], dim=-1))

    def table(self, mapping, target_cols):
        rows = []
        values = self.values.cpu().numpy()
        def ratio(a, b):
            return float(a/b) if b else float('nan')
        normalized_total = values[:, :, self.fields.index('normalized_sse')].sum()
        objective_total = values[:, :, self.fields.index('objective_sum')].sum()
        for sid, meta in mapping.items():
            for h, target in enumerate(target_cols):
                d = dict(zip(self.fields, values[meta['station_index'], h]))
                n = d['n']
                if not n:
                    continue
                rows.append(dict(station_id=sid, station=meta.get('name', sid), target=target,
                    **d, rmse_mm=ratio(d['sse'], n)**.5, mae_mm=ratio(d['sae'], n),
                    persistence_rmse_mm=ratio(d['persistence_sse'], n)**.5,
                    persistence_skill=1-ratio(d['sse'], d['persistence_sse']),
                    actual_abs_change_mm=ratio(d['actual_abs_change'], n),
                    correction_abs_mm=ratio(d['pred_abs_change'], n),
                    rise_false_positive_rate=ratio(d['false_alarm_n'], n-d['rise_n']),
                    extreme_rmse_mm=ratio(d['extreme_sse'], d['extreme_n'])**.5,
                    extreme_actual_abs_mm=ratio(d['extreme_actual_abs'], d['extreme_n']),
                    extreme_pred_abs_mm=ratio(d['extreme_pred_abs'], d['extreme_n']),
                    rise_bias_mm=ratio(d['rise_bias_sum'], d['rise_n']),
                    rise_shortfall_mm=ratio(d['rise_shortfall_sum'], d['rise_n']),
                    rise_under_rate=ratio(d['rise_under_n'], d['rise_n']),
                    rise_recall=ratio(d['true_positive_n'], d['rise_n']),
                    rise_false_alarm_ratio=ratio(d['false_alarm_n'], d['predicted_rise_n']),
                    large_error_pct=100*ratio(d['large_errors'], n),
                    normalized_sse_share=ratio(d['normalized_sse'], normalized_total),
                    objective_share=ratio(d['objective_sum'], objective_total)))
        return pd.DataFrame(rows)
