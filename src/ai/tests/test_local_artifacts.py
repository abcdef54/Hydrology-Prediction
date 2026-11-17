"""The training result must remain usable locally without MLflow model upload."""

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock, patch

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.ai.trainers import LSTMTrainer


class LocalArtifactTests(TestCase):
    def test_existing_artifacts_block_training_before_initial_checkpoint(self):
        for artifact in ("model_best.pt", "model_best.preprocessing.joblib", "model_best.dynamics"):
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                target = root / artifact
                if target.suffix == '.dynamics':
                    target.mkdir()
                    (target / 'preserve').write_bytes(b'old snapshot')
                else:
                    target.write_bytes(b'old artifact')
                trainer = object.__new__(LSTMTrainer)
                trainer.checkpoint_path = root / 'model_best.pt'
                trainer.print_settings = Mock()
                trainer.prepare_checkpoint = Mock()
                with self.assertRaisesRegex(FileExistsError, 'new --checkpoint-dir'):
                    trainer.train()
                trainer.prepare_checkpoint.assert_not_called()
                self.assertFalse((root / 'model_best.training.lock').exists())
                preserved = target / 'preserve' if target.is_dir() else target
                self.assertIn(preserved.read_bytes(), (b'old artifact', b'old snapshot'))

    def test_existing_lock_blocks_concurrent_run_without_deleting_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock = root / 'model_best.training.lock'
            lock.write_text('another run')
            trainer = object.__new__(LSTMTrainer)
            trainer.checkpoint_path = root / 'model_best.pt'
            trainer.prepare_checkpoint = Mock()
            with self.assertRaisesRegex(FileExistsError, 'locked'):
                trainer.train()
            trainer.prepare_checkpoint.assert_not_called()
            self.assertEqual(lock.read_text(), 'another run')

    def test_training_saves_local_scalers_without_uploading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = object.__new__(LSTMTrainer)
            trainer.model = torch.nn.Linear(1, 1)
            trainer.checkpoint_path = root / "model_best.pt"
            feature_scaler = StandardScaler().fit([[1.], [3.]])
            target_scaler = StandardScaler().fit([[2.], [6.]])
            trainer.train_dataloader = SimpleNamespace(dataset=SimpleNamespace(
                feature_scaler=feature_scaler, target_scaler=target_scaler,
                feature_cols=["water_level_depth"], target_cols=["target_water_level_plus_60m"],
                seq_len=4, rain_station_ids=[],
            ))
            trainer.feature_set = "water_level"
            trainer.checkpoint_name = "test_model"
            trainer.horizon = "1h"
            trainer.best_checkpoint_epoch = 1
            trainer.best_val_loss = .1
            trainer.dynamics = None
            trainer.skip_test = False
            trainer.print_settings = Mock()
            trainer.prepare_checkpoint = Mock()
            trainer.train_epochs = Mock(return_value=1)

            def restore():
                torch.save(trainer.model.state_dict(), trainer.checkpoint_path)

            trainer.restore_best_checkpoint = Mock(side_effect=restore)
            trainer.test = Mock(return_value=pd.DataFrame([{"mae": 1.}], index=["target"]))
            # A model upload would reproduce the original post-training failure.
            with patch("mlflow.pytorch.log_model", side_effect=AssertionError("No model upload")), \
                 patch("src.ai.trainers.mlflow.log_metrics") as log_metrics, \
                 patch("src.ai.trainers.mlflow.log_table"):
                self.assertIs(trainer.train(), trainer.model)
                self.assertEqual(log_metrics.call_count, 2)
            saved = joblib.load(root / "model_best.preprocessing.joblib")
            np.testing.assert_array_equal(saved["feature_scaler"].mean_, feature_scaler.mean_)
            np.testing.assert_array_equal(saved["target_scaler"].scale_, target_scaler.scale_)
            self.assertEqual(saved["feature_columns"], ["water_level_depth"])
            self.assertEqual(saved["sequence_length"], 4)
            restored = torch.nn.Linear(1, 1)
            restored.load_state_dict(torch.load(trainer.checkpoint_path, weights_only=True))
            torch.testing.assert_close(restored.weight, trainer.model.weight)


if __name__ == "__main__":
    main()
