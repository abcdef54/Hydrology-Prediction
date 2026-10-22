"""Build a reusable rainfall pool while preserving B labels, splits and predictors.

Run python -m research.scripts.build_rain_station_dataset. New stations use the existing
offline hourly export; source datasets and checkpoints are never overwritten.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from research.scripts.build_output24_dataset import DEFAULT_OUTPUT, TARGETS, read_frame
from src.ai.hourly_dataset import DEFAULT_POOL
from src.ai.rain_stations import (
    DEFAULT_COMBINATION, RAIN_ALIASES, resolve_rain_stations, station_alias,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SNAPSHOT = ROOT / "reports/rain_regions_h12_h24_20261019"


def build(source_dir=DEFAULT_OUTPUT, snapshot_dir=DEFAULT_SNAPSHOT,
          output_dir=DEFAULT_POOL, rain_stations=DEFAULT_COMBINATION):
    source_dir, snapshot_dir, output_dir = map(Path, (source_dir, snapshot_dir, output_dir))
    if output_dir.exists():
        raise FileExistsError(f"Refuse to overwrite: {output_dir}")
    spec = json.loads((source_dir / "manifest.json").read_text())
    if spec.get("variant") != "mask" or spec["target_columns"] != TARGETS:
        raise ValueError("Use the mask O24 dataset as parent")
    catalog = pd.read_csv(snapshot_dir / "station_catalog.csv", dtype={"id": str})
    catalog["id"] = catalog.id.str.lstrip("0")
    if not catalog.id.is_unique:
        raise ValueError("Duplicate catalog station IDs")
    aliases = dict(RAIN_ALIASES)
    for row in catalog.itertuples():
        alias = station_alias(row.name)
        if alias in aliases and aliases[alias] != row.id:
            raise ValueError(f"Ambiguous station alias: {alias}")
        aliases[alias] = row.id
    ids = resolve_rain_stations(rain_stations, catalog.id, aliases)
    catalog = catalog.set_index("id")
    raw = pd.read_csv(snapshot_dir / "hourly_rain.csv.gz", dtype={"station_id": str})
    raw["station_id"] = raw.station_id.str.lstrip("0")
    raw = raw.loc[raw.station_id.isin(ids)].copy()
    raw["timestamp_utc"] = pd.to_datetime(raw.timestamp_utc, utc=True)
    if raw.duplicated(["station_id", "timestamp_utc"]).any():
        raise ValueError("Duplicate hourly rain records")
    if set(raw.station_id) != set(ids):
        raise ValueError("Rain observations missing for requested stations")
    raw["rain_mm"] = pd.to_numeric(raw.rain_mm, errors="raise")
    invalid = raw.n_valid.le(0) | raw.n_invalid.gt(0) | raw.rain_mm.lt(0)
    raw.loc[invalid | ~np.isfinite(raw.rain_mm), "rain_mm"] = np.nan

    source = read_frame(source_dir / "hourly.csv")
    blocks = {}
    for sid in ids:
        prefix = f"rain_{sid}_"
        existing = [c for c in spec["feature_columns"] if c.startswith(prefix)]
        if existing:
            # Preserve B exactly, including its original early-window history.
            blocks[sid] = source[existing].copy()
            continue
        series = raw.loc[raw.station_id.eq(sid)].set_index("timestamp_utc").rain_mm.sort_index()
        grid = pd.date_range(min(series.index.min(), source.index.min()),
                             max(series.index.max(), source.index.max()), freq="1h")
        series = series.reindex(grid)
        block = pd.DataFrame({prefix + "depth": series})
        for hours in (3, 6, 12, 24):
            block[f"{prefix}sum_{hours}h"] = series.rolling(hours, min_periods=hours).sum()
        block[prefix + "missing"] = series.isna().astype("int8")
        blocks[sid] = block.reindex(source.index)
    prefixes = tuple(f"rain_{sid}_" for sid in ids)
    original_features = [c for c in spec["feature_columns"]
                         if not c.startswith("rain_") or c.startswith(prefixes)]
    added = [block for block in blocks.values() if not set(block).issubset(source.columns)]
    features = original_features + [c for block in added for c in block]
    combined = pd.concat([source[original_features], *added, source[TARGETS]], axis=1)
    splits, coverage = {}, {}
    for split in ("train", "val", "test"):
        original = read_frame(source_dir / f"{split}.csv")
        if not original.index.isin(combined.index).all():
            raise ValueError("Parent split falls outside hourly snapshot")
        frame = combined.reindex(original.index).copy()
        preserved = [c for c in frame if c in original]
        frame[preserved] = original[preserved]
        pd.testing.assert_frame_equal(frame[preserved], original[preserved])
        splits[split] = frame
        coverage[split] = {
            "rows": len(frame), "complete_24h": int(frame[TARGETS].notna().all(axis=1).sum()),
            "start_utc": str(frame.index.min()), "end_utc": str(frame.index.max()),
            "rain_hour_coverage": {sid: float(frame[f"rain_{sid}_depth"].notna().mean()) for sid in ids},
        }
    for left, right in (("train", "val"), ("val", "test")):
        if splits[left].index.max() + pd.Timedelta(hours=24) >= splits[right].index.min():
            raise ValueError("Parent 24-hour purge is not preserved")
    result = {
        **spec, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_dataset": str(source_dir.resolve()), "rain_snapshot": str(snapshot_dir.resolve()),
        "feature_columns": features, "input_size": len(features), "rain_station_ids": ids,
        "rain_station_aliases": {alias: sid for alias, sid in aliases.items() if sid in ids},
        "rain_stations": {sid: catalog.loc[sid].to_dict() for sid in ids}, "splits": coverage,
        "rain_aggregation": "Source sum in (t-1h,t]; full-window rolling 3/6/12/24h; missing kept NaN",
        "rain_quality_policy": "Keep partial valid hours as in B; reject invalid/negative records; existing B features unchanged",
        "matching_policy": "Exact parent rows and labels; never filter samples by selected rain availability",
    }
    output_dir.mkdir(parents=True)
    combined.to_csv(output_dir / "hourly.csv", index_label="timestamp_utc")
    for split, frame in splits.items():
        frame.to_csv(output_dir / f"{split}.csv", index_label="timestamp_utc")
    provenance = source_dir / "weather_provenance.csv.gz"
    if provenance.exists():
        shutil.copyfile(provenance, output_dir / provenance.name)
    (output_dir / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output_dir": str(output_dir), "rain_station_ids": ids,
                      "input_size_pool": len(features), "splits": coverage}, ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--rain-stations", default=DEFAULT_COMBINATION,
                        help="Aliases/IDs joined by '+', or all for the entire catalog")
    args = parser.parse_args()
    build(args.source_dir, args.snapshot_dir, args.output_dir, args.rain_stations)


if __name__ == "__main__":
    main()
