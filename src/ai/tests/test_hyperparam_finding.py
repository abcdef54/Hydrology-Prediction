"""Small synthetic searches exercise fitting, persistence, resume and pruning."""
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import optuna
import pandas as pd

from src.ai import hyperparam_finding as search
from src.ai.metrics import extrema_timing, paired_error_stats


class AlwaysPrune(optuna.pruners.BasePruner):
    def prune(self, study, trial):
        return True


class HyperparameterSearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / 'dataset'
        self.dataset.mkdir()
        t = np.arange(440)
        water = 400 + 110 * np.sin(t / 15) + .1 * t
        frame = pd.DataFrame(dict(
            timestamp_utc=pd.date_range('2026-01-01', periods=len(t), freq='h', tz='UTC'),
            water_level_depth=water, rain=np.maximum(np.sin(t / 15), 0)))
        for hour, name in enumerate(search.TARGETS, 1):
            frame[name] = pd.Series(water).shift(-hour)
        frame.iloc[:260].to_csv(self.dataset / 'train.csv', index=False)
        frame.iloc[285:385].to_csv(self.dataset / 'val.csv', index=False)
        # Optimization must succeed even when test cannot be parsed.
        (self.dataset / 'test.csv').write_text('TEST MUST NOT BE READ')
        (self.dataset / 'manifest.json').write_text(json.dumps(dict(
            variant='mask', feature_columns=['water_level_depth', 'rain'])))

    def arguments(self, method='xgb', mode='grid', budget=1, extra=()):
        space = {'max_depth': [2, 3], 'learning_rate': [.1]} if method in search.TREES else {
            'lstm_hidden_size': [4], 'lstm_num_layers': [1], 'lstm_drop_out': [0.],
            'lstm_input_dropout': [0.], 'lstm_head_dropout': [0.], 'batch_size': [32],
            'lstm_lr': [.001, .01]}
        path = self.root / f'{method}_space.json'
        path.write_text(json.dumps(space))
        return search.parser().parse_args([
            '--search-method', mode, '--train-method', method, '--dataset-dir', str(self.dataset),
            '--output-dir', str(self.root / method), '--search-space', str(path),
            '--sequence-length', '4', '--n-trials', str(budget), '--n-estimators', '3',
            '--num-epochs', '2', '--early-stopping', '2', '--n-jobs', '1', *extra])

    def run_search(self, args):
        with redirect_stdout(io.StringIO()):
            search.run(args)
        storage = optuna.storages.RDBStorage(f'sqlite:///{args.output_dir}/study.sqlite3')
        self.addCleanup(storage.engine.dispose)
        self.addCleanup(storage.remove_session)
        return optuna.load_study(study_name='hydrology', storage=storage)

    def test_count_dominates_rmse_and_focus_excludes_short_horizons(self):
        self.assertLess(search.objective_value(1, 1e30), search.objective_value(2, 0))
        self.assertLess(search.objective_value(2, 10), search.objective_value(2, 11))
        self.assertEqual(search.objective_value(100, 8, 'rmse'), 8)
        metrics = {}
        for h in [1, 3, 6, 12, 24]:
            metrics[f'horizon/target_water_level_plus_{h*60}m/above_200'] = h
            metrics[f'horizon/target_water_level_plus_{h*60}m/rmse_mm'] = h
        focus = search.focus_metrics(metrics, [12, 24])
        self.assertEqual(focus['above_200'], 36)
        self.assertAlmostEqual(focus['rmse_mm'], np.sqrt((12**2 + 24**2) / 2))
        for n, r in [(float('nan'), 1), (1, float('inf')), (-1, 2), (1.5, 2)]:
            with self.assertRaises(ValueError):
                search.objective_value(n, r)

    def test_expanded_spaces_and_default_budgets(self):
        for method in sorted(search.TREES | search.NEURALS):
            for strategy in ['grid', 'tpe']:
                space = search.default_space(method, strategy)
                search.validate_space(space, method, strategy)

                if strategy == 'grid':
                    combinations = np.prod([len(values) for values in space.values()])
                    self.assertEqual(combinations, 2016 if method in search.TREES else 1440)
                    continue

                # Exercise TPE beyond its random startup, without training or files.
                sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=2)
                study = optuna.create_study(sampler=sampler)

                def objective(trial):
                    params = search.suggest(trial, space)
                    for key, value in params.items():
                        search.check_parameter(key, value, method)
                    return sum(float(value) for value in params.values())

                study.optimize(objective, n_trials=5)

                if method in search.TREES:
                    self.assertEqual(space['max_depth']['high'], 16)
                    self.assertEqual(space['learning_rate']['low'], .001)
                    self.assertEqual(space['reg_lambda']['high'], 1000.)
                    self.assertTrue(any(trial.params['max_depth'] > 6 for trial in study.trials))
                else:
                    self.assertEqual(space['lstm_hidden_size']['high'], 256)
                    self.assertEqual(space['lstm_num_layers']['low'], 1)
                    self.assertTrue(any(trial.params['lstm_hidden_size'] > 64 for trial in study.trials))

        args = search.parser().parse_args([
            '--search-method', 'tpe', '--train-method', 'xgb',
            '--output-dir', str(self.root / 'not_created'),
        ])
        self.assertEqual(search.DEFAULT_TPE_TRIALS, 100)
        self.assertEqual(args.n_estimators, 5000)
        self.assertEqual(args.tpe_startup_trials, 20)

    def test_sequence_search_bounds_fixed_override_and_custom_choices(self):
        args = search.parser().parse_args([
            '--search-method', 'tpe', '--train-method', 'xgb',
            '--output-dir', str(self.root / 'not_created'),
        ])
        space, warmup = search.resolve_search_space(args)
        self.assertEqual(space['sequence_length'], dict(type='int', low=12, high=168))
        self.assertEqual(warmup, 168)

        args.sequence_length = 96
        space, warmup = search.resolve_search_space(args)
        self.assertNotIn('sequence_length', space)
        self.assertEqual(warmup, 96)

        args.sequence_length = None
        args.sequence_lengths = [12, 48, 168]
        space, warmup = search.resolve_search_space(args)
        self.assertEqual(space['sequence_length'], [12, 48, 168])
        self.assertEqual(warmup, 168)

        for choices in [[11, 168], [12, 169], [12, 12], []]:
            args.sequence_lengths = choices
            with self.subTest(choices=choices), self.assertRaises(ValueError):
                search.resolve_search_space(args)

    def test_shorter_windows_preserve_anchors_labels_scalers_and_past_only_inputs(self):
        for method in sorted(search.TREES | search.NEURALS):
            args = self.arguments(method)
            args.sequence_length = 24
            frames, reference, saved = search.load_search_data(args)
            shorter, pre = search.data_at_sequence(reference, saved, 12, method)
            self.assertEqual(pre['sequence_length'], 12)
            self.assertEqual(pre['cohort_sequence_length'], 24)

            for split in ['train', 'val']:
                if method in search.TREES:
                    manual = search.sequence_table(
                        frames[split], saved['feature_columns'], saved['target_columns'], 12,
                    )
                    positions = np.searchsorted(manual['indices'], reference[split]['indices'])
                    np.testing.assert_array_equal(shorter[split]['indices'], reference[split]['indices'])
                    np.testing.assert_array_equal(shorter[split]['X'], manual['X'][positions])
                    np.testing.assert_array_equal(shorter[split]['y'], reference[split]['y'])
                    self.assertTrue(np.shares_memory(shorter[split]['X'], reference[split]['X']))
                else:
                    np.testing.assert_array_equal(shorter[split].end_indices, reference[split].end_indices)
                    np.testing.assert_array_equal(shorter[split][0][0], reference[split][0][0][-12:])
                    np.testing.assert_array_equal(shorter[split][0][1], reference[split][0][1])
                    self.assertIs(shorter[split].feature_scaler, reference[split].feature_scaler)
                    self.assertIs(shorter[split].target_scaler, reference[split].target_scaler)
                    self.assertEqual(reference[split].seq_len, 24)

    def test_sequence_grid_fits_each_length_and_saves_matching_artifacts(self):
        import joblib

        for method in ['xgb', 'lgbm', 'residual_lstm']:
            with self.subTest(method=method):
                args = self.arguments(method, budget=2)
                args.sequence_length = None
                space = json.loads(args.search_space.read_text())
                space['sequence_length'] = [12, 24]

                if method in search.TREES:
                    space['max_depth'] = [2]
                else:
                    space['lstm_lr'] = [.001]

                args.search_space.write_text(json.dumps(space))
                study = self.run_search(args)
                self.assertEqual({trial.params['sequence_length'] for trial in study.trials}, {12, 24})
                self.assertTrue(all(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials))
                config = json.loads((args.output_dir / 'search_config.json').read_text())
                self.assertEqual(config['population'], dict(train=237, val=77))
                self.assertEqual(config['cohort_sequence_length'], 24)

                for trial in study.trials:
                    folder = Path(trial.user_attrs['artifact_dir'])
                    if method in search.TREES:
                        saved = json.loads((folder / 'tree_model.json').read_text())
                    else:
                        saved = joblib.load(folder / 'model_best.preprocessing.joblib')
                    self.assertEqual(saved['sequence_length'], trial.params['sequence_length'])
                    self.assertEqual(saved['cohort_sequence_length'], 24)

                # Explicit test scoring retains the study cohort even if the
                # winning/selected artifact uses only twelve input hours.
                if method == 'residual_lstm':
                    selected = next(trial for trial in study.trials if trial.params['sequence_length'] == 12)
                    folder = Path(selected.user_attrs['artifact_dir'])
                    pd.read_csv(self.dataset / 'val.csv').to_csv(self.dataset / 'test.csv', index=False)
                    reference_args = search.copy(args)
                    reference_args.sequence_length = 24
                    _, _, reference_saved = search.load_search_data(reference_args)

                    def compare_cohort(output, model, pre, frames, datasets, options):
                        self.assertEqual(pre['sequence_length'], 12)
                        self.assertEqual(len(datasets['test']), 77)
                        self.assertEqual(datasets['test'].end_indices[0], 23)
                        self.assertTrue(frames['test']['water_level_depth'].notna().all())
                        return {}, {}

                    with patch('research.scripts.output_horizon_experiment.write_comparisons', side_effect=compare_cohort):
                        search.evaluate_best(args, folder, args.output_dir, reference_saved)

    def test_low_learning_rate_gets_longer_patience_and_comparable_pruning_steps(self):
        from types import SimpleNamespace

        args = self.arguments()
        args.early_stopping = 50
        args.n_estimators = 6000
        _, data, saved = search.load_search_data(args)
        folder = self.root / 'slow_tree'
        folder.mkdir()
        trial = optuna.create_study().ask()
        steps = []

        def no_fit(method, train, val, index, options, output, on_iteration):
            self.assertEqual(options['early_stopping'], 2500)
            self.assertEqual(options['n_estimators'], 6000)
            metrics = dict(above_200=0, rmse_mm=10.)
            for raw_step in [1, 49, 50, 100, 150]:
                on_iteration(raw_step, SimpleNamespace(best_metrics=metrics))
            return dict(
                kind='persistence', path=None, best_round=0,
                validation_metrics=metrics, history=[dict(round=150)],
            )

        with patch.object(search, 'train_horizon', side_effect=no_fit):
            with patch.object(search, 'maybe_prune', side_effect=lambda trial, value, step: steps.append(step)):
                with redirect_stdout(io.StringIO()):
                    search.train_tree(trial, args, {'learning_rate': .001}, data, saved, folder)

        self.assertEqual(steps, [1, 2, 3])
        self.assertEqual(search.tree_progress_scale(.1), 1.)
        self.assertEqual(search.tree_progress_scale(.001), .02)

    def test_boundaries_are_reported_using_validation_only_without_new_files(self):
        distribution = optuna.distributions.FloatDistribution(.001, .3, log=True)
        study = optuna.create_study()
        for score, value in enumerate([.001, .0011, .0012]):
            study.add_trial(optuna.trial.create_trial(
                value=float(score), params={'learning_rate': value},
                distributions={'learning_rate': distribution},
            ))

        space = {'learning_rate': dict(type='float', low=.001, high=.3, log=True)}
        before = set(self.root.rglob('*'))
        messages = search.boundary_messages(study, space)
        self.assertEqual(len(messages), 1)
        self.assertIn('3/3 top trials near lower bound', messages[0])
        self.assertEqual(before, set(self.root.rglob('*')))

    def test_space_rejects_inactive_keys_and_invalid_distributions(self):
        for space, method in [({'unknown_parameter': [24]}, 'xgb'),
                              ({'lstm_drop_out': [1.]}, 'lstm'),
                              ({'lstm_correction_penalty': [.1]}, 'lstm'),
                              ({'num_leaves': [4]}, 'xgb'),
                              ({'min_child_weight': [2.]}, 'lgbm'),
                              ({'lstm_lr': dict(type='float', low=.001, high=.1, log=True, step=1)}, 'lstm')]:
            with self.subTest(space=space), self.assertRaises(ValueError):
                search.validate_space(space, method, 'tpe')
        search.validate_space({'lstm_correction_penalty': [-.1, 0., .1]}, 'residual_lstm', 'grid')

    def test_tree_grid_resume_reload_and_no_test_access(self):
        for method in ['xgb', 'lgbm']:
            with self.subTest(method=method):
                first = self.run_search(self.arguments(method))
                self.assertEqual(len(first.trials), 1)
                resumed = self.run_search(self.arguments(method, budget=10, extra=['--resume']))
                self.assertEqual(len(resumed.trials), 2)  # full grid cap, not 10
                self.assertTrue(all(t.state == optuna.trial.TrialState.COMPLETE for t in resumed.trials))
                self.assertEqual(len({t.params['max_depth'] for t in resumed.trials}), 2)
                chosen = json.loads((self.root / method / 'best_trial.json').read_text())
                native = search.TreeSequenceForecast(chosen['artifact_dir'])
                frame = search.read_frame(self.dataset / 'val.csv').reset_index()
                _, values = native.predict_frame(frame)
                self.assertEqual(values.shape, (97, 5))
                with self.assertRaises(FileExistsError):
                    self.run_search(self.arguments(method))
                args = self.arguments(method, extra=['--resume'])
                args.seed = 7
                with self.assertRaisesRegex(ValueError, 'configuration differs'):
                    self.run_search(args)

    def test_neural_real_fit_and_saved_model_reload(self):
        from src.ai.comparison import load_candidate
        import joblib
        import torch
        args = self.arguments('residual_lstm')
        study = self.run_search(args)
        self.assertEqual(study.best_trial.state, optuna.trial.TrialState.COMPLETE)
        folder = Path(study.best_trial.user_attrs['artifact_dir'])
        saved = joblib.load(folder / 'model_best.preprocessing.joblib')
        model = load_candidate((folder / 'model_best.pt').read_bytes(), saved, 'cpu')
        _, data, _ = search.load_search_data(args)
        with torch.inference_mode():
            pred = model(data['val'][0][0].unsqueeze(0))
        self.assertEqual(tuple(pred.shape), (1, 5))
        self.assertTrue(torch.isfinite(pred).all())
        self.assertEqual(len(pd.read_csv(folder / 'validation.csv')), 3)

    def test_tpe_pruning_interrupts_neural_and_both_boosters(self):
        for method in ['residual_lstm', 'xgb', 'lgbm']:
            with self.subTest(method=method), patch.object(optuna.pruners, 'MedianPruner', return_value=AlwaysPrune()):
                args = self.arguments(method, mode='tpe')
                study = self.run_search(args)
                self.assertEqual(study.trials[0].state, optuna.trial.TrialState.PRUNED)
                self.assertTrue(study.trials[0].intermediate_values)
                self.assertFalse((args.output_dir / 'best_trial.json').exists())
                status = json.loads((args.output_dir / 'trial_0000' / 'result.json').read_text())
                self.assertEqual(status['status'], 'PRUNED')

    def test_median_pruner_stops_bad_trial_after_reference_trial(self):
        study = optuna.create_study(pruner=optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=0))
        study.optimize(lambda trial: (trial.report(1., 1), 1.)[-1], n_trials=1)
        trial = study.ask()
        with self.assertRaises(optuna.TrialPruned):
            search.maybe_prune(trial, 100., 1)

    def test_check_only_writes_nothing(self):
        args = self.arguments(extra=['--check-only'])
        with redirect_stdout(io.StringIO()):
            search.run(args)
        self.assertFalse(args.output_dir.exists())

    def test_shared_metrics_preserve_threshold_and_empty_extrema(self):
        frame = pd.DataFrame(dict(candidate_error_mm=[200, 201, -199], serving_error_mm=[201, 199, -201]))
        stats = paired_error_stats(frame)
        self.assertEqual((stats['fixed'], stats['new'], stats['both']), (2, 1, 0))
        points = pd.DataFrame(dict(horizon_minutes=[720] * 4,
            target_utc=pd.date_range('2026-01-01', periods=4, freq='h', tz='UTC'),
            actual_mm=[0.] * 4, serving_mm=[0.] * 4, candidate_mm=[0.] * 4))
        detail, summary = extrema_timing(points)
        self.assertTrue(detail.empty)
        self.assertTrue(summary.empty)
        points = pd.DataFrame(dict(horizon_minutes=[1440] * 9,
            target_utc=pd.date_range('2026-01-01', periods=9, freq='h', tz='UTC'),
            actual_mm=[0, 0, 0, 100, 0, 0, 0, 0, 0],
            serving_mm=[0, 0, 0, 0, 100, 0, 0, 0, 0], candidate_mm=[0.] * 9))
        detail, summary = extrema_timing(points)
        late = summary.loc[summary.model.eq('B') & summary.kind.eq('peak')].iloc[0]
        self.assertEqual(late.mean_lag_hours, 1.)
        missed = summary.loc[summary.model.eq('candidate') & summary.kind.eq('peak')].iloc[0]
        self.assertEqual(missed.n_missed, 1)


if __name__ == '__main__':
    unittest.main()
