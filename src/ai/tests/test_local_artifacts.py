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
