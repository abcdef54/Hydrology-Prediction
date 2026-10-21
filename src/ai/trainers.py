"""Training and evaluation for the sequence forecasting models."""

import math
from pathlib import Path
from typing import Protocol
import joblib
import mlflow
import pandas as pd
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.ai.losses import HorizonWeightedMSE, HorizonWeightedTailLoss, validate_correction_penalty
from src.ai.metrics import evaluate_all
from src.ai.models import LSTM, MeanEmbeddingForecastLSTMWithAdapter, ResidualLSTM


class SequenceTrainer(Protocol):
    def train(self) -> nn.Module:
        ...

    def test(self) -> pd.DataFrame:
        ...

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
        checkpoint_metric: str = "val_loss",
        validation_threshold_mm: float = 200.0,
        gradient_clip: float = 1.0,
    ) -> None:
        if gradient_accumulation < 1:
            raise ValueError("gradient_accumulation must be at least 1")
        if not math.isfinite(gradient_clip) or gradient_clip <= 0:
            raise ValueError("gradient_clip must be finite and positive")
        self.gradient_clip = gradient_clip

        if checkpoint_metric not in ("val_loss", "large_error_rate", "rmse"):
            raise ValueError("Unknown checkpoint metric")
        if not math.isfinite(validation_threshold_mm) or validation_threshold_mm <= 0:
            raise ValueError("Validation error threshold must be finite and positive")
        self.checkpoint_metric = checkpoint_metric
        self.validation_threshold_mm = float(validation_threshold_mm)
        self.best_checkpoint_score = (float("inf"), float("inf"))
        self.last_validation_metrics = {}
        self.best_validation_metrics = {}
        self.model = model.to(device)
        self.horizon = horizon
        self.num_epochs = num_epochs
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.loss_func = loss_func.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.gradient_accumulation = gradient_accumulation
        self.device = torch.device(device)

        self.feature_set = getattr(train_dataloader.dataset, "feature_set", "all")
        self.checkpoint_name = type(model).__name__
        if isinstance(model, MeanEmbeddingForecastLSTMWithAdapter) and any(
            parameter.requires_grad for parameter in model.backbone.parameters()
        ):
            self.checkpoint_name += "_unfrozen"
        if self.feature_set != "all":
            self.checkpoint_name += f"_{self.feature_set}"
        self.checkpoint_path = Path(checkpoint_dir) / horizon / f"{self.checkpoint_name}_best.pt"

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

        epoch = self.train_epochs()
        self.restore_best_checkpoint()
        self.save_preprocessing()
        results = self.test()
        mlflow.log_metrics(
            {
                "epochs_completed": epoch,
                "best_checkpoint_epoch": self.best_checkpoint_epoch,
                "best_val_loss": self.best_val_loss
            }
        )

        best_metrics = {key.replace("val/", "best_val/", 1): value
                        for key, value in getattr(self, "best_validation_metrics", {}).items()}
        if best_metrics:
            mlflow.log_metrics(best_metrics)
        run_name = f"{self.checkpoint_name}_{self.horizon}_{self.feature_set}"
        metrics = {}
        for target_name, target_metrics in results.iterrows():
            for metric_name, value in target_metrics.items():
                numeric_value = float(value)
                if math.isfinite(numeric_value):
                    metrics[f"test/{target_name}/{metric_name}"] = numeric_value
        mlflow.log_metrics(metrics)
        mlflow.log_table(results.reset_index(), f"{run_name}_test_results.json")
        return self.model

    def train_epochs(self) -> int:
        return self.run_epochs(start_epoch=1, num_epochs=self.num_epochs)

    def run_epochs(self, start_epoch: int, num_epochs: int, stage: int | None = None) -> int:
        if num_epochs < 1:
            raise ValueError("Training requires at least one epoch")
        final_epoch = start_epoch + num_epochs - 1
        for epoch in tqdm(range(start_epoch, final_epoch + 1), desc=f"Training {type(self.model).__name__}"):
            train_loss = self.train_epoch(epoch)
            val_loss = self.evaluate_loss(self.eval_dataloader)
            print(f"Epoch {epoch}/{final_epoch} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
            metrics = {"train_loss": train_loss, "val_loss": val_loss}
            metrics.update(getattr(self, "last_validation_metrics", {}))
            if self.last_validation_metrics:
                print(f"Validation >{self.validation_threshold_mm:g} mm: "
                      f"{self.last_validation_metrics['val/large_error_count']:.0f} "
                      f"({self.last_validation_metrics['val/large_error_pct']:.2f}%); "
                      f"RMSE: {self.last_validation_metrics['val/rmse_mm']:.2f} mm")
            if stage is not None:
                metrics["training_stage"] = stage
            mlflow.log_metrics(metrics, step=epoch)
            self.record_validation(epoch, val_loss)
            if self.early_stopping and self.epochs_without_improvement >= self.early_stopping:
                print(f"Early stopping triggered after epoch {epoch}.")
                break
        return epoch

    def save_preprocessing(self) -> Path:
        """Keep preprocessing beside the local checkpoint; no model upload to MLflow."""
        dataset = self.train_dataloader.dataset
        preprocessing_path = self.checkpoint_path.with_suffix(".preprocessing.joblib")
        preprocessing_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "feature_scaler": dataset.feature_scaler,
                "target_scaler": dataset.target_scaler,
                "feature_columns": dataset.feature_cols,
                "target_columns": dataset.target_cols,
                "sequence_length": dataset.seq_len,
                "resolution_minutes": getattr(dataset, "resolution_minutes", 60),
                "missing_scaled_value": getattr(dataset.feature_scaler, "hydrology_normalized_missing_value_", 0.0),
                "raw_missing_value": getattr(dataset.feature_scaler, "hydrology_raw_missing_value_", None),
                "rain_station_ids": getattr(dataset, "rain_station_ids", []),
                "rain_station_combination": getattr(dataset, "rain_station_combination", ""),
            },
            preprocessing_path,
        )
        print(f"Local preprocessing: {preprocessing_path}")
        return preprocessing_path

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
        print(f"Rain station IDs: {getattr(dataset, 'rain_station_ids', [])}")
        for row in getattr(dataset, "rain_station_ranking", []):
            print(f"  Train correlation: {row['station_id']} r={row['pearson_r']:.6f} (n={row['sample_count']})")
        if dataset.num_features <= 12:
            print(f"Selected features: {dataset.feature_cols}")
        print(f"Output size: {dataset.num_targets}")
        print(f"Sequence length: {dataset.seq_len}")
        print(f"Trainable parameters: {trainable_parameters:,}")
        print(f"Learning rate: {self.optimizer.defaults['lr']}")
        print(f"Weight decay: {self.optimizer.defaults['weight_decay']}")
        print(f"Gradient accumulation: {self.gradient_accumulation}")
        print(f"Loss: {type(self.loss_func).__name__}")
        print(f"Checkpoint selection: {self.checkpoint_metric}; error threshold: {self.validation_threshold_mm:g} mm")
        if isinstance(self.loss_func, HorizonWeightedTailLoss):
            print(f"Tail penalty: {self.loss_func.strength}; transition: {self.loss_func.transition_mm:g} mm")
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
            tqdm(self.train_dataloader, desc=f"Epoch {epoch}", leave=False)
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
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=getattr(self, "gradient_clip", 1.0))
                self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()
        return sum(batch_losses) / max(1, len(batch_losses))

    def evaluate_loss(self, dataloader: DataLoader, predictor=None) -> float:
        """Score the same validation objective for a learned model or persistence."""
        self.model.eval()
        predictor = self.model if predictor is None else predictor
        loss_sum, element_count, row_count = 0.0, 0, 0
        squared = None
        physical_squared = None
        physical_bad = None
        base_sum, tail_sum = 0.0, 0.0
        scaler = getattr(dataloader.dataset, "target_scaler", None)
        threshold = getattr(self, "validation_threshold_mm", 200.0)
        with torch.inference_mode():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                predictions = predictor(inputs)
                loss = self.loss_func(predictions, targets)
                if not torch.isfinite(loss):
                    raise ValueError("Validation loss is non-finite")
                loss_sum += loss.item() * targets.numel()
                element_count += targets.numel()
                row_count += len(targets)
                errors = predictions.double() - targets.double()
                batch_squared = errors.square().sum(dim=0).cpu()
                squared = batch_squared if squared is None else squared + batch_squared
                if isinstance(self.loss_func, HorizonWeightedTailLoss):
                    base, tail = self.loss_func.components(predictions, targets)
                    base_sum += base.item() * targets.numel()
                    tail_sum += tail.item() * targets.numel()
                if scaler is not None:
                    scales = torch.as_tensor(scaler.scale_, device=self.device, dtype=torch.float64)
                    error_mm = errors * scales
                    sq = error_mm.square().sum(dim=0).cpu()
                    bad = (error_mm.abs() > threshold).sum(dim=0).cpu()
                    physical_squared = sq if physical_squared is None else physical_squared + sq
                    physical_bad = bad if physical_bad is None else physical_bad + bad
        if element_count == 0:
            raise ValueError("Evaluation dataset has no eligible sequences")
        self.last_validation_mse = squared / row_count
        self.last_validation_metrics = {}
        if physical_squared is not None:
            self.last_validation_metrics = {
                "val/large_error_count": int(physical_bad.sum()),
                "val/large_error_pct": 100 * physical_bad.sum().item() / element_count,
                "val/rmse_mm": math.sqrt(physical_squared.sum().item() / element_count),
            }
            for name, sq, bad in zip(dataloader.dataset.target_cols, physical_squared, physical_bad):
                self.last_validation_metrics[f"val/{name}/rmse_mm"] = math.sqrt(sq.item() / row_count)
                self.last_validation_metrics[f"val/{name}/large_error_pct"] = 100 * bad.item() / row_count
        elif getattr(self, "checkpoint_metric", "val_loss") in ("large_error_rate", "rmse"):
            raise ValueError("Physical-unit checkpoint selection requires the train-fitted target scaler")
        if isinstance(self.loss_func, HorizonWeightedTailLoss):
            self.last_validation_metrics.update({"val/base_weighted_mse": base_sum / element_count,
                                                  "val/tail_penalty": tail_sum / element_count})
        return loss_sum / element_count

    def validation_score(self, val_loss):
        if getattr(self, "checkpoint_metric", "val_loss") == "rmse":
            return (self.last_validation_metrics["val/rmse_mm"],)
        if getattr(self, "checkpoint_metric", "val_loss") == "large_error_rate":
            metrics = self.last_validation_metrics
            return (metrics["val/large_error_count"], metrics["val/rmse_mm"])
        return (val_loss,)

    def record_validation(self, epoch: int, val_loss: float) -> None:
        score = self.validation_score(val_loss)
        if getattr(self, "checkpoint_metric", "val_loss") in ("large_error_rate", "rmse"):
            improved = score < self.best_checkpoint_score
            self.epochs_without_improvement = 0 if improved else self.epochs_without_improvement + 1
            self.best_training_val_loss = min(self.best_training_val_loss, val_loss)
        else:
            if val_loss < self.best_training_val_loss:
                self.best_training_val_loss = val_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            improved = val_loss < self.best_val_loss
        if improved:
            self.best_val_loss = val_loss
            self.best_checkpoint_score = score
            self.best_validation_metrics = dict(getattr(self, "last_validation_metrics", {}))
            self.best_checkpoint_epoch = epoch
            torch.save(self.model.state_dict(), self.checkpoint_path)
            print(f"New best validation score: {score} (saving to {self.checkpoint_path})")
        elif self.best_checkpoint_epoch == 0 and self.epochs_without_improvement == 0:
            print("Validation improved, but persistence is still the best checkpoint.")

    def restore_best_checkpoint(self, purpose: str = "testing") -> None:
        if self.best_checkpoint_epoch is None:
            raise ValueError("No checkpoint was selected; train for at least one epoch")
        if self.best_checkpoint_epoch == 0:
            print("Best checkpoint: initial persistence baseline (no trained epoch beat it).")
        else:
            print(f"Best checkpoint: epoch {self.best_checkpoint_epoch}.")
        print(f"\nLoading best checkpoint from {self.checkpoint_path} for {purpose}...")
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
        mlflow.log_metric("test_loss", float(mse))
        mlflow.set_tag("test_loss_scale", "scaled_target_units")
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
        validate_correction_penalty(correction_penalty, kwargs["loss_func"], model.output_size)
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
        baseline_loss = self.evaluate_loss(self.eval_dataloader, predictor=self.model.persistence)
        print(f"Validation persistence loss ({type(self.loss_func).__name__}): {baseline_loss:.4f}")
        for name, mse in zip(self.eval_dataloader.dataset.target_cols, self.last_validation_mse):
            print(f"  {name}: {mse:.4f}")
        self.best_val_loss = baseline_loss
        self.best_checkpoint_score = self.validation_score(baseline_loss)
        self.best_validation_metrics = dict(self.last_validation_metrics)
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
    def __init__(
        self, model: MeanEmbeddingForecastLSTMWithAdapter,
        two_stage: bool = False, finetune_epochs: int = 30,
        backbone_lr: float = 1e-5, adapter_lr: float = 1e-4,
        finetune_backbone: str = "forecast", **kwargs,
    ) -> None:
        if not isinstance(model, MeanEmbeddingForecastLSTMWithAdapter):
            raise TypeError("MEFLSTMAdapterTrainer requires a MEF LSTM adapter model")
        if finetune_epochs < 1 or not all(
            math.isfinite(rate) and rate > 0 for rate in (backbone_lr, adapter_lr)
        ):
            raise ValueError("Fine-tuning epochs and learning rates must be positive")
        if finetune_backbone not in ("forecast", "all"):
            raise ValueError("finetune_backbone must be forecast or all")
        self.two_stage = two_stage
        self.finetune_epochs = finetune_epochs
        self.backbone_lr = backbone_lr
        self.adapter_lr = adapter_lr
        self.finetune_backbone = finetune_backbone
        if two_stage:
            model.freeze_backbone()
        super().__init__(model=model, **kwargs)
        if two_stage:
            self.checkpoint_name += "_two_stage"
            self.checkpoint_path = self.checkpoint_path.with_name(f"{self.checkpoint_name}_best.pt")

    def train_epochs(self) -> int:
        if not self.two_stage:
            return super().train_epochs()

        print("Stage 1: frozen backbone; training adapter and output head.")
        stage_one_epoch = self.run_epochs(1, self.num_epochs, stage=1)
        self.restore_best_checkpoint(purpose="stage-two fine tuning")
        mlflow.log_metrics({"stage_one_epochs": stage_one_epoch, "stage_one_best_val_loss": self.best_val_loss})

        # Only recurrent modules used by the adapter's forward pass need gradients.
        self.model.freeze_backbone()
        backbone_parameters = list(self.model.backbone.forecast_lstm.parameters())
        if self.finetune_backbone == "all":
            backbone_parameters += list(self.model.backbone.hindcast_lstm.parameters())
        for parameter in backbone_parameters:
            parameter.requires_grad = True
        adapter_parameters = list(self.model.hindcast_adapter.parameters()) + list(self.model.output_head.parameters())
        weight_decay = self.optimizer.defaults["weight_decay"]
        self.optimizer = torch.optim.AdamW(
            [{"params": backbone_parameters, "lr": self.backbone_lr},
             {"params": adapter_parameters, "lr": self.adapter_lr}],
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)
        self.epochs_without_improvement = 0
        self.best_training_val_loss = self.best_val_loss
        print(
            f"Stage 2: {self.finetune_backbone} recurrent backbone; "
            f"backbone LR={self.backbone_lr}, adapter/head LR={self.adapter_lr}. "
            f"Validation score to beat: {self.best_val_loss:.4f}"
        )
        final_epoch = self.run_epochs(stage_one_epoch + 1, self.finetune_epochs, stage=2)
        best_stage = 1 if self.best_checkpoint_epoch <= stage_one_epoch else 2
        mlflow.set_tag("best_checkpoint_stage", best_stage)
        mlflow.log_metric("stage_two_epochs", final_epoch - stage_one_epoch)
        print(f"Best checkpoint belongs to stage {best_stage}.")
        return final_epoch

    def model_settings(self) -> dict[str, object]:
        backbone_trainable = any(
            parameter.requires_grad for parameter in self.model.backbone.parameters()
        )
        settings = {
            "Hidden size": self.model.backbone.hindcast_lstm.hidden_size,
            "Backbone": "unfrozen" if backbone_trainable else "frozen",
            "Input dropout": self.model.input_dropout,
            "Head dropout": self.model.head_dropout,
            "Future forecast inputs": "unavailable (zero embedding)",
        }
        if self.two_stage:
            settings.update({
                "Training stages": 2,
                "Fine-tuning epochs": self.finetune_epochs,
                "Fine-tuning backbone": self.finetune_backbone,
                "Fine-tuning backbone LR": self.backbone_lr,
                "Fine-tuning adapter/head LR": self.adapter_lr,
            })
        return settings
