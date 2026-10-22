"""Named station experiments must preserve the cohort and train-only scaling."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from research.scripts.build_output24_dataset import TARGETS
from research.scripts.build_rain_station_dataset import build
from research.scripts.output_horizon_experiment import make_model, parser, prepare_data
from src.ai.comparison import load_candidate
from src.ai.rain_stations import (
    DEFAULT_COMBINATION, RAIN_ALIASES, combination_name, resolve_rain_stations,
)


COMBINATIONS = [
    "hong_thai+nham",
    "hong_thai+nham+bach_ma+phu_bai",
    "hong_thai+bach_ma+phu_bai+vp",
    "hong_thai+bach_ma+binh_dien+vp",
    DEFAULT_COMBINATION,
]


class RainCombinationTests(unittest.TestCase):
    def setUp(self):
        self.temp = self.enterContext(tempfile.TemporaryDirectory())
        root = Path(self.temp)
        self.parent, self.snapshot, self.output = root / "parent", root / "rain", root / "pool"
        self.parent.mkdir()
        self.snapshot.mkdir()
        times = pd.date_range("2026-01-01", periods=240, freq="1h", tz="UTC", name="timestamp_utc")
        values = np.arange(len(times), dtype=float)
        frame = pd.DataFrame({"water_level_depth": values + 100}, index=times)
        for sid in ["750062", "750007"]:
            for suffix in ["depth", "sum_3h", "sum_6h", "sum_12h", "sum_24h"]:
                frame[f"rain_{sid}_{suffix}"] = values % 7
            frame[f"rain_{sid}_missing"] = 0
        frame["flood_season"] = 0
        frame["weather_forecast_rain_0_3h"] = np.where(values % 4 == 0, np.nan, values % 9)
        features = list(frame)
        for h, target in enumerate(TARGETS, 1):
            frame[target] = values + 100 + h
        frame.to_csv(self.parent / "hourly.csv")
        for split, start in [("train", 0), ("val", 80), ("test", 160)]:
            frame.iloc[start:start + 48].to_csv(self.parent / f"{split}.csv")
        spec = dict(variant="mask", experiment="O5_O24", target_columns=TARGETS,
                    feature_columns=features, rain_station_ids=["750062", "750007"])
        (self.parent / "manifest.json").write_text(json.dumps(spec))
        catalog, records = [], []
        for alias, sid in RAIN_ALIASES.items():
            catalog.append(dict(id=sid, name=alias, latitude=16.3, longitude=107.5))
            records.extend(dict(station_id=sid, timestamp_utc=t, rain_mm=float(i % 5),
                                n_valid=6, n_invalid=0) for i, t in enumerate(times))
        pd.DataFrame(catalog).to_csv(self.snapshot / "station_catalog.csv", index=False)
        raw = pd.DataFrame(records)
        # Invalid/absent rainfall must not remove label anchors or become zero rain.
        raw.loc[(raw.station_id == "750014") & (raw.timestamp_utc == times[90]), "n_invalid"] = 1
        self.raw = raw
        raw.to_csv(self.snapshot / "hourly_rain.csv.gz", index=False)
        with redirect_stdout(io.StringIO()):
            self.spec = build(self.parent, self.snapshot, self.output)

    def test_resolution_errors_and_no_four_station_limit(self):
        ids = list(RAIN_ALIASES.values())
        self.assertEqual(resolve_rain_stations("Hồng Thái+VP", ids), ["750062", "750033"])
        self.assertEqual(len(resolve_rain_stations(DEFAULT_COMBINATION, ids)), 6)
        for selection in ["", "hong_thai+", "unknown", "hong_thai+750062"]:
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                resolve_rain_stations(selection, ids)
        with self.assertRaises(ValueError):
            resolve_rain_stations("phu_bai", ["750062"])
        self.assertEqual(combination_name(ids), combination_name(ids[::-1]))

    def test_builder_keeps_parent_values_splits_and_past_only_rain(self):
        for split in ["hourly", "train", "val", "test"]:
            parent = pd.read_csv(self.parent / f"{split}.csv")
            pool = pd.read_csv(self.output / f"{split}.csv")
            pd.testing.assert_frame_equal(pool[parent.columns], parent)
        pool = pd.read_csv(self.output / "hourly.csv")
        self.assertEqual(pool.rain_750041_sum_24h.iloc[30], sum(float(i % 5) for i in range(7, 31)))
        self.assertTrue(np.isnan(pool.rain_750014_depth.iloc[90]))
        self.assertEqual(pool.rain_750014_missing.iloc[90], 1)
        self.assertTrue(np.isnan(pool.rain_750014_sum_24h.iloc[110]))
        self.assertTrue(np.isfinite(pool.rain_750014_sum_24h.iloc[114]))
        with self.assertRaises(FileExistsError):
            build(self.parent, self.snapshot, self.output)

    def test_combinations_preserve_cohort_scaling_and_model_reload(self):
        torch.set_num_threads(2)
        _, baseline, parent_saved = prepare_data(self.parent, 5, 4)
        for combination in COMBINATIONS:
            frames, datasets, saved = prepare_data(self.output, 5, 4, combination)
            expected_ids = resolve_rain_stations(combination, self.spec["rain_station_ids"])
            self.assertEqual(set(saved["rain_station_ids"]), set(expected_ids))
            if combination == "hong_thai+nham":
                self.assertEqual(saved["feature_columns"], parent_saved["feature_columns"])
                np.testing.assert_array_equal(datasets['train'].X, baseline['train'].X)
            for split, dataset in datasets.items():
                np.testing.assert_array_equal(dataset.end_indices, baseline[split].end_indices)
                np.testing.assert_array_equal(dataset.y, baseline[split].y)
                self.assertIs(dataset.feature_scaler, datasets['train'].feature_scaler)
            expected_mean = frames['train'][saved['feature_columns']].mean().to_numpy()
            np.testing.assert_allclose(saved['feature_scaler'].mean_, expected_mean)
            args = parser().parse_args(['--outputs', '5', '--device', 'cpu',
                '--lstm-hidden-size', '4', '--lstm-num-layers', '2'])
            saved['train_method'] = 'residual_lstm'
            model = make_model(saved, args, 'cpu')
            x, y = datasets['train'][0]
            prediction = model(x[None])
            loss = (prediction - y[None]).square().mean()
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            model.eval()
            checkpoint = Path(self.temp) / 'test.pt'
            torch.save(model.state_dict(), checkpoint)
            reloaded = load_candidate(checkpoint.read_bytes(), saved, 'cpu')
            torch.testing.assert_close(model(x[None]), reloaded(x[None]))

    def test_standard_train_entry_point_accepts_explicit_combinations(self):
        from src.ai.train import build_sequence_trainer, parser as train_parser, validate_args

        paths = set()
        for combination in COMBINATIONS:
            args = train_parser.parse_args([
                '--mlflow-ex-name', 'test', '--train-method', 'residual_lstm', '--horizon', '1h',
                '--dataset-dir', str(self.output), '--rain-stations', combination, '--outputs', '5',
                '--device', 'cpu', '--lstm-seq-len', '4', '--batch-size', '4',
                '--lstm-hidden-size', '4', '--lstm-num-layers', '2',
            ])
            validate_args(args)
            trainer = build_sequence_trainer(args)
            self.assertEqual(trainer.train_dataloader.dataset.num_targets, 5)
            self.assertEqual(len(trainer.train_dataloader.dataset.rain_station_ids), combination.count('+') + 1)
            paths.add(trainer.checkpoint_path)
        self.assertEqual(len(paths), len(COMBINATIONS))

    def test_output_subset_never_promotes_other_targets_to_features(self):
        from src.ai.utils import get_dataloader

        for outputs in (5, 24):
            loaders = get_dataloader('1h', 4, dataset_dir=self.output, seq_len=4,
                                     num_worker=0, pin_memory=False, outputs=outputs)
            for loader in loaders:
                self.assertEqual(loader.dataset.num_targets, outputs)
                self.assertFalse(any(column.startswith('target_')
                                     for column in loader.dataset.feature_cols))


if __name__ == '__main__':
    unittest.main()
