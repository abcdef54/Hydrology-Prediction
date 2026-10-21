"""Subsample the existing hourly pool at fixed local clock times, without refitting.

Predictors retain their physical definitions: sum_24h still means 24 hours.
Labels are exact future hourly observations, never means of future 12/24h blocks.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd

from research.build_output24_dataset import read_frame
from research.build_rain_station_dataset import DEFAULT_POOL


def sample_grid(frame, step_hours):
    local = frame.index.tz_convert("Asia/Ho_Chi_Minh")
    return frame.loc[(local.hour % step_hours == 0) & (local.minute == 0)].copy()


def build(source_dir, output_dir, step_hours, target_hours):
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refuse to overwrite {output_dir}")
    if step_hours not in (12, 24):
        raise ValueError("Use a 12h or 24h observation step")
    if not target_hours or len(set(target_hours)) != len(target_hours) or min(target_hours) < 1:
        raise ValueError("Provide distinct positive target hours")
    spec = json.loads((source_dir / "manifest.json").read_text())
    if spec.get("resolution_minutes") != 60:
        raise ValueError("Source must be hourly")
    source = read_frame(source_dir / "hourly.csv")
    features = spec["feature_columns"]
    targets = [f"target_water_level_plus_{h * 60}m" for h in sorted(target_hours)]
    for h, target in zip(sorted(target_hours), targets):
        expected = source.water_level_depth.reindex(source.index + pd.Timedelta(hours=h)).to_numpy()
        np.testing.assert_allclose(source[target], expected, rtol=0, atol=1e-8, equal_nan=True)
    combined = sample_grid(source[features + targets], step_hours)
    splits, coverage = {}, {}
    for split in ("train", "val", "test"):
        original = read_frame(source_dir / f"{split}.csv")
        sampled = sample_grid(original[features + targets], step_hours)
        if len(sampled) < 2 or not sampled.index.to_series().diff().dropna().eq(pd.Timedelta(hours=step_hours)).all():
            raise ValueError(f"{split}: insufficient/irregular sampled grid")
        pd.testing.assert_frame_equal(sampled, combined.reindex(sampled.index))
        splits[split] = sampled
        coverage[split] = dict(rows=len(sampled), complete_targets=int(sampled[targets].notna().all(axis=1).sum()),
                               start_utc=str(sampled.index.min()), end_utc=str(sampled.index.max()))
    for left, right in (("train", "val"), ("val", "test")):
        if splits[left].index.max() + pd.Timedelta(hours=max(target_hours)) >= splits[right].index.min():
            raise ValueError("Target purge is not preserved")
    result = {
        **spec, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_dataset": str(source_dir), "resolution_minutes": step_hours * 60,
        "experiment": "temporal_sampling", "target_columns": targets, "output_size": len(targets),
        "sequence_length": 96 // step_hours + 1,
        "sequence_history_span_hours": 96,
        "sampling_policy": "Keep source rows at 00:00/12:00 local for 12h; 00:00 local for 24h",
        "predictor_policy": "Exact parent values at input time; rolling windows stay physical hours",
        "splits": coverage,
    }
    result.pop("common_target_columns", None)
    output_dir.mkdir(parents=True)
    combined.to_csv(output_dir / "hourly.csv", index_label="timestamp_utc")
    for split, frame in splits.items():
        frame.to_csv(output_dir / f"{split}.csv", index_label="timestamp_utc")
    provenance = source_dir / "weather_provenance.csv.gz"
    if provenance.exists():
        frame = pd.read_csv(provenance)
        times = pd.to_datetime(frame.timestamp_utc, utc=True)
        frame.loc[times.isin(combined.index)].to_csv(output_dir / provenance.name, index=False)
    (output_dir / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(output_dir=str(output_dir), step_hours=step_hours,
                          target_hours=sorted(target_hours), splits=coverage), indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--step-hours", type=int, choices=(12, 24), required=True)
    parser.add_argument("--target-hours", type=int, nargs="+", required=True)
    args = parser.parse_args()
    build(args.source_dir, args.output_dir, args.step_hours, args.target_hours)


if __name__ == "__main__":
    main()
