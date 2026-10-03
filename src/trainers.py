"""Training and evaluation for the sequence forecasting models."""

from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.metrics import evaluate_all
from src.models import LSTM, MeanEmbeddingForecastLSTMWithAdapter, ResidualLSTM


class LSTMTrainer:
    """Share the epoch loop and checkpoint selection across sequence models."""

    def __init__(
        self,
        model: nn.Module,
        horizon: str,
        num_epochs: int,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        test_dataloader: DataLoader,
        loss_func: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        early_stopping: int,
        gradient_accumulation: int = 1,
        device: str | torch.device = "cpu",
        checkpoint_dir: str | Path = "model",
    ) -> None:
        if gradient_accumulation < 1:
            raise ValueError("gradient_accumulation must be at least 1")

        self.model = model.to(device)
        self.horizon = horizon
        self.num_epochs = num_epochs
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.gradient_accumulation = gradient_accumulation
        self.device = torch.device(device)

        self.feature_set = getattr(train_dataloader.dataset, "feature_set", "all")
        checkpoint_name = type(model).__name__
        if self.feature_set != "all":
            checkpoint_name += f"_{self.feature_set}"
        self.checkpoint_path = Path(checkpoint_dir) / horizon / f"{checkpoint_name}_best.pt"

        self.best_val_loss = float("inf")
        self.best_training_val_loss = float("inf")
        self.best_checkpoint_epoch: int | None = None
        self.epochs_without_improvement = 0

    def model_settings(self) -> dict[str, object]:
        return {}

    def prepare_checkpoint(self) -> None:
        """Allow a child trainer to provide a baseline before epoch one."""

    def training_loss(
        self,
        predictions: torch.Tensor,
        inputs: torch.Tensor,
        data_loss: torch.Tensor,
    ) -> torch.Tensor:
        return data_loss

    def train(self) -> nn.Module:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.print_settings()
        self.prepare_checkpoint()

        for epoch in tqdm(range(1, self.num_epochs + 1), desc=f"Training {type(self.model).__name__}"):
            train_loss = self.train_epoch(epoch)
            val_loss = self.evaluate_loss(self.eval_dataloader)
            print(f"Epoch {epoch}/{self.num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
            self.record_validation(epoch, val_loss)
            if self.early_stopping and self.epochs_without_improvement >= self.early_stopping:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

        self.restore_best_checkpoint()
        self.test()
        return self.model

    def print_settings(self) -> None:
        dataset = self.train_dataloader.dataset
        trainable_parameters = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        print("\n" + "=" * 50)
        print("Training settings")
        print(f"Model: {type(self.model).__name__}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Feature set: {self.feature_set} ({dataset.num_features} inputs)")
        if dataset.num_features <= 12:
            print(f"Selected features: {dataset.feature_cols}")
        print(f"Output size: {dataset.num_targets}")
        print(f"Sequence length: {dataset.seq_len}")
        print(f"Trainable parameters: {trainable_parameters:,}")
        print(f"Learning rate: {self.optimizer.defaults['lr']}")
        print(f"Weight decay: {self.optimizer.defaults['weight_decay']}")
        print(f"Gradient accumulation: {self.gradient_accumulation}")
        split_sizes = (
            len(dataset),
            len(self.eval_dataloader.dataset),
            len(self.test_dataloader.dataset),
        )
        print(f"Train/validation/test sequences: {split_sizes[0]}/{split_sizes[1]}/{split_sizes[2]}")
        for setting, value in self.model_settings().items():
            print(f"{setting}: {value}")
        print(f"Best checkpoint: {self.checkpoint_path}")
        print("=" * 50 + "\n")

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        batch_losses = []
        for batch_number, (inputs, targets) in enumerate(
            tqdm(self.train_dataloader, desc=f"Epoch {epoch}/{self.num_epochs}", leave=False)
        ):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            if batch_number % self.gradient_accumulation == 0:
                self.optimizer.zero_grad()

            predictions = self.model(inputs)
            data_loss = self.loss_func(predictions, targets)
            objective = self.training_loss(predictions, inputs, data_loss)
            (objective / self.gradient_accumulation).backward()
            batch_losses.append(data_loss.item())

            last_batch = batch_number + 1 == len(self.train_dataloader)
            if (batch_number + 1) % self.gradient_accumulation == 0 or last_batch:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()
        return sum(batch_losses) / max(1, len(batch_losses))

    def evaluate_loss(self, dataloader: DataLoader) -> float:
        self.model.eval()
        loss_sum = 0.0
        element_count = 0
        with torch.inference_mode():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                loss_sum += self.loss_func(self.model(inputs), targets).item() * targets.numel()
                element_count += targets.numel()
        if element_count == 0:
            raise ValueError("Evaluation dataset has no eligible sequences")
        return loss_sum / element_count

    def record_validation(self, epoch: int, val_loss: float) -> None:
        if val_loss < self.best_training_val_loss:
            self.best_training_val_loss = val_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_checkpoint_epoch = epoch
            torch.save(self.model.state_dict(), self.checkpoint_path)
            print(f"New best val loss: {val_loss:.4f} (saving to {self.checkpoint_path})")
        elif self.best_checkpoint_epoch == 0 and self.epochs_without_improvement == 0:
            print("Validation improved, but persistence is still the best checkpoint.")

    def restore_best_checkpoint(self) -> None:
        if self.best_checkpoint_epoch is None:
            raise ValueError("No checkpoint was selected; train for at least one epoch")
        if self.best_checkpoint_epoch == 0:
            print("Best checkpoint: initial persistence baseline (no trained epoch beat it).")
        else:
            print(f"Best checkpoint: epoch {self.best_checkpoint_epoch}.")
        print(f"\nLoading best checkpoint from {self.checkpoint_path} for testing...")
        state_dict = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)

    def test(self) -> pd.DataFrame:
        self.model.eval()
        predictions = []
        targets = []
        with torch.inference_mode():
            for inputs, batch_targets in tqdm(self.test_dataloader, desc="Testing LSTM"):
                inputs = inputs.to(self.device)
                predictions.append(self.model(inputs).cpu())
                targets.append(batch_targets)
        if not targets:
            raise ValueError("Test dataset has no eligible sequences")

        predicted_scaled = torch.cat(predictions).numpy()
        target_scaled = torch.cat(targets).numpy()
        mse = ((predicted_scaled - target_scaled) ** 2).mean()
        print(f"Test Loss (MSE): {mse:.4f}")

        dataset = self.test_dataloader.dataset
        predicted = dataset.target_scaler.inverse_transform(predicted_scaled)
        observed = dataset.target_scaler.inverse_transform(target_scaled)
        summaries = []
        for target_index, target_name in enumerate(dataset.target_cols):
            metrics = evaluate_all(
                observed[:, target_index],
                predicted[:, target_index],
                timestamps=dataset.timestamps,
            )
            metrics["target"] = target_name
            summaries.append(metrics)
        summary = pd.DataFrame(summaries).set_index("target")
        displayed_metrics = (
            "rmse", "mae", "nse", "kge", "flood_f1", "flood_precision",
            "flood_recall", "flood_accuracy", "peak_absolute_error",
            "peak_timing_error_minutes",
        )
        print("\n" + "=" * 45 + " TEST EVALUATION " + "=" * 45)
        print(summary[[name for name in displayed_metrics if name in summary]].to_string())
        print("=" * 107 + "\n")
        return summary


class VanillaLSTMTrainer(LSTMTrainer):
    def __init__(self, model: LSTM, **kwargs) -> None:
        if type(model) is not LSTM:
            raise TypeError("VanillaLSTMTrainer requires an LSTM model")
        super().__init__(model=model, **kwargs)

    def model_settings(self) -> dict[str, object]:
        return {
            "Layers": self.model.num_layers,
            "Hidden size": self.model.hidden_size,
            "LSTM dropout": self.model.dropout,
            "Input dropout": self.model.input_dropout,
            "Head dropout": self.model.head_dropout,
        }


class ResidualLSTMTrainer(LSTMTrainer):
    def __init__(self, model: ResidualLSTM, correction_penalty: float = 0.0, **kwargs) -> None:
        if not isinstance(model, ResidualLSTM):
            raise TypeError("ResidualLSTMTrainer requires a ResidualLSTM model")
        if correction_penalty < 0:
            raise ValueError("Correction penalty must be non-negative")
        super().__init__(model=model, **kwargs)
        self.correction_penalty = correction_penalty

    def model_settings(self) -> dict[str, object]:
        return {
            "Layers": self.model.num_layers,
            "Hidden size": self.model.hidden_size,
            "LSTM dropout": self.model.dropout,
            "Input dropout": self.model.input_dropout,
            "Head dropout": self.model.head_dropout,
            "Correction penalty": self.correction_penalty,
        }

    def prepare_checkpoint(self) -> None:
        squared_error = torch.zeros(self.model.output_size, dtype=torch.float64)
        row_count = 0
        self.model.eval()
        with torch.inference_mode():
            for inputs, targets in self.eval_dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                squared_error += (self.model.persistence(inputs) - targets).square().sum(dim=0).cpu().double()
                row_count += len(targets)
        if row_count == 0:
            raise ValueError("Validation dataset has no eligible persistence samples")
        baseline_mse = squared_error / row_count
        print(f"Validation persistence MSE: {baseline_mse.mean():.4f}")
        for name, mse in zip(self.eval_dataloader.dataset.target_cols, baseline_mse):
            print(f"  {name}: {mse:.4f}")
        self.best_val_loss = baseline_mse.mean().item()
        self.best_checkpoint_epoch = 0
        torch.save(self.model.state_dict(), self.checkpoint_path)
        print(f"Saved initial persistence checkpoint to {self.checkpoint_path}")

    def training_loss(
        self,
        predictions: torch.Tensor,
        inputs: torch.Tensor,
        data_loss: torch.Tensor,
    ) -> torch.Tensor:
        correction = predictions - self.model.persistence(inputs)
        return data_loss + self.correction_penalty * correction.square().mean()


class MEFLSTMAdapterTrainer(LSTMTrainer):
    def __init__(self, model: MeanEmbeddingForecastLSTMWithAdapter, **kwargs) -> None:
        if not isinstance(model, MeanEmbeddingForecastLSTMWithAdapter):
            raise TypeError("MEFLSTMAdapterTrainer requires a MEF LSTM adapter model")
        super().__init__(model=model, **kwargs)

    def model_settings(self) -> dict[str, object]:
        return {
            "Hidden size": self.model.backbone.hindcast_lstm.hidden_size,
            "Backbone": "frozen",
            "Future forecast inputs": "unavailable (zero embedding)",
        }
