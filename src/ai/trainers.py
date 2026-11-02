"""Training and evaluation for the sequence forecasting models."""

import math
import json
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
from src.ai.change_objective import ChangeRiskLoss
from src.ai.change_diagnostics import ChangeDiagnostics
from src.ai.metrics import evaluate_all
from src.ai.models import LSTM, MeanEmbeddingForecastLSTMWithAdapter, ResidualLSTM
from src.ai.residual_sequence import ResidualSequence
from src.ai.sequence_checkpoint import model_spec, save_model_metadata
from src.ai.evaluation_summary import station_error_table


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
        dynamics=None,
        skip_test: bool = False,
    ) -> None:
        if gradient_accumulation < 1:
            raise ValueError("gradient_accumulation must be at least 1")
        if not math.isfinite(gradient_clip) or gradient_clip <= 0:
            raise ValueError("gradient_clip must be finite and positive")
        self.gradient_clip = gradient_clip
        self.dynamics = dynamics
        self.skip_test = skip_test

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
        if self.dynamics is not None:
            self.dynamics.run(self, 0)

        epoch = self.train_epochs()
        self.restore_best_checkpoint()
        self.save_preprocessing()
        results = None if self.skip_test else self.test()
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
        if self.skip_test:
            mlflow.set_tag('test/status', 'skipped_by_request')
            return self.model
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
            epoch_lr = self.optimizer.param_groups[0]['lr'] if isinstance(self.optimizer, Optimizer) else None
            train_loss = self.train_epoch(epoch)
            val_loss = self.evaluate_loss(self.eval_dataloader)
            print(f"Epoch {epoch}/{final_epoch} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
            metrics = {"train_loss": train_loss, "val_loss": val_loss}
            metrics.update(getattr(self, 'last_train_components', {}))
            if epoch_lr is not None:
                metrics['learning_rate_epoch'] = epoch_lr
            metrics.update(getattr(self, "last_validation_metrics", {}))
            if self.last_validation_metrics:
                print(f"Validation >{self.validation_threshold_mm:g} mm: "
                      f"{self.last_validation_metrics['val/large_error_count']:.0f} "
                      f"({self.last_validation_metrics['val/large_error_pct']:.2f}%); "
                      f"RMSE: {self.last_validation_metrics['val/rmse_mm']:.2f} mm"
                      + (f"; MAE: {self.last_validation_metrics['val/mae_mm']:.2f} mm"
                         if 'val/mae_mm' in self.last_validation_metrics else ''))
            if stage is not None:
                metrics["training_stage"] = stage
            mlflow.log_metrics(metrics, step=epoch)
            self.record_validation(epoch, val_loss)
            if getattr(self, 'dynamics', None) is not None:
                stopping = bool(self.early_stopping and self.epochs_without_improvement >= self.early_stopping)
                if epoch % self.dynamics.every == 0 or epoch == final_epoch or stopping:
                    self.dynamics.run(self, epoch)
            if self.early_stopping and self.epochs_without_improvement >= self.early_stopping:
                print(f"Early stopping triggered after epoch {epoch}.")
                break
        return epoch

    def save_preprocessing(self, checkpoint=None, *, announce=True) -> Path:
        """Keep preprocessing beside the local checkpoint; no model upload to MLflow."""
        dataset = self.train_dataloader.dataset
        preprocessing_path = Path(checkpoint or self.checkpoint_path).with_suffix(".preprocessing.joblib")
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
                "model_spec": model_spec(self.model),
                "regional": getattr(dataset, 'regional_metadata', None),
                "forecast_history_policy": getattr(self.model, 'forecast_history_policy', 'all_timesteps'),
            },
            preprocessing_path,
        )
        if announce:
            print(f"Local preprocessing: {preprocessing_path}")
        return preprocessing_path

    def save_checkpoint(self, checkpoint=None):
        checkpoint = Path(checkpoint or self.checkpoint_path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), checkpoint)
        save_model_metadata(self.model, checkpoint)
        if isinstance(self.loss_func, ChangeRiskLoss):
            spec = self.loss_func.specification()
            if getattr(self.model, 'event_aware', False):
                spec['rise_classification'] = dict(loss='BCEWithLogitsLoss',
                    weight=self.rise_classification_weight, positive_weight=self.rise_positive_weight,
                    threshold_mm=self.loss_func.threshold_mm, comparison='strict_greater_than')
            metadata = getattr(self.train_dataloader.dataset, 'regional_metadata', {})
            spec['training_provenance'] = {key: metadata[key] for key in
                ('dataset_path', 'dataset_manifest_sha256', 'fold', 'station_mapping',
                 'target_horizons', 'training_selection') if key in metadata}
            checkpoint.with_suffix('.loss.json').write_text(json.dumps(spec, indent=2))

    def prepare_batch(self, batch, dataset, *, need_physical=True, need_mean=True):
        if isinstance(batch, dict):
            inputs = {'x': batch['x'].to(self.device, non_blocking=True)}
            if getattr(self.model, 'num_stations', 0):
                inputs['station_ids'] = batch['station_ids'].to(self.device, non_blocking=True)
                if isinstance(self.model, ResidualSequence) and getattr(dataset, 'station_ids_validated', False):
                    inputs['_validate_station_ids'] = False
            scale = batch['target_scale'].to(self.device, non_blocking=True) if need_physical else None
            mean = batch['target_mean'].to(self.device, non_blocking=True) if need_mean else None
            return inputs, batch['y'].to(self.device, non_blocking=True), scale, mean
        inputs, targets = batch
        scaler = getattr(dataset, 'target_scaler', None)
        scale = None if scaler is None else torch.as_tensor(scaler.scale_, device=self.device, dtype=torch.float64)
        mean = None if scaler is None else torch.as_tensor(getattr(scaler, 'mean_', torch.zeros_like(scale)), device=self.device, dtype=torch.float64)
        return inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True), scale, mean

    @staticmethod
    def predict_batch(predictor, inputs):
        return predictor(**inputs) if isinstance(inputs, dict) else predictor(inputs)

    def data_loss(self, predictions, targets, scales, inputs=None, *, apply_rise_weight=True):
        if isinstance(self.loss_func, ChangeRiskLoss):
            if not isinstance(inputs, dict) or 'station_ids' not in inputs:
                raise ValueError('Change loss requires station IDs')
            baseline = self.predict_batch(self.model.persistence, inputs)
            return self.loss_func(predictions, targets, baseline, inputs['station_ids'],
                                  apply_rise_weight=apply_rise_weight)
        if isinstance(self.loss_func, HorizonWeightedTailLoss):
            return self.loss_func(predictions, targets, target_scales=scales)
        return self.loss_func(predictions, targets)

    def rise_classification_loss(self, logits, targets, scales, inputs):
        baseline = self.predict_batch(self.model.persistence, inputs)
        labels = (((targets-baseline.detach())*scales) > self.loss_func.threshold_mm).to(logits.dtype)
        pos = torch.as_tensor(self.rise_positive_weight, dtype=logits.dtype, device=logits.device)
        cells = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos, reduction='none')
        return self.loss_func.reduce_per_horizon(cells)

    def new_change_diagnostics(self, dataset):
        if (not getattr(dataset, 'regional_metadata', None) or not hasattr(self.model, 'persistence')
                or not getattr(self.model, 'num_stations', 0)):
            return None
        return ChangeDiagnostics(len(dataset.regional_metadata['station_mapping']),
            dataset.num_targets, self.device,
            getattr(self.loss_func, 'threshold_mm', self.validation_threshold_mm))

    def update_change_diagnostics(self, meter, predictions, targets, scales, inputs, *, apply_rise_weight=True):
        if meter is None:
            return
        baseline = self.predict_batch(self.model.persistence, inputs)
        elements = None
        if isinstance(self.loss_func, ChangeRiskLoss):
            elements = self.loss_func.elements(predictions, targets, baseline, inputs['station_ids'],
                                               apply_rise_weight=apply_rise_weight)
            elements = elements*self.loss_func.weights.to(elements)
        meter.update(predictions, targets, baseline, scales, inputs['station_ids'], elements)

    def save_change_diagnostics(self, meter, dataset, prefix):
        if meter is None:
            return
        table = meter.table(dataset.regional_metadata['station_mapping'], dataset.target_cols)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(self.checkpoint_path.with_suffix(f'.{prefix}_change_metrics.csv'), index=False)
        metrics = {}
        fields = ('rmse_mm', 'mae_mm', 'persistence_skill', 'extreme_rmse_mm',
                  'extreme_pred_abs_mm', 'extreme_actual_abs_mm', 'rise_bias_mm',
                  'rise_under_rate', 'rise_recall', 'rise_false_alarm_ratio',
                  'normalized_sse_share', 'objective_share', 'large_errors',
                  'correction_abs_mm', 'actual_abs_change_mm', 'rise_false_positive_rate')
        for _, row in table.iterrows():
            for field in fields:
                value = float(row[field])
                if math.isfinite(value):
                    metrics[f'{prefix}/{row.station_id}/{row.target}/{field}'] = value
        # Integrate with existing epoch-level MLflow logging; retain readable CSV.
        if prefix == 'val':
            self.last_validation_metrics.update(metrics)
        elif mlflow.active_run():
            mlflow.log_metrics(metrics)
        if prefix in ('val', 'test'):
            print(f'Change diagnostics ({prefix}):')
            print(table[['station', 'target', 'rmse_mm', 'persistence_skill',
                'extreme_actual_abs_mm', 'extreme_pred_abs_mm', 'rise_recall',
                'rise_false_alarm_ratio']].to_string(index=False, float_format=lambda x: f'{x:.3f}'))

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
        regression_losses, classification_losses = [], []
        meter = self.new_change_diagnostics(self.train_dataloader.dataset)
        for batch_number, batch in enumerate(
            tqdm(self.train_dataloader, desc=f"Epoch {epoch}", leave=False)
        ):
            inputs, targets, scales, _ = self.prepare_batch(batch, self.train_dataloader.dataset,
                need_physical=isinstance(self.loss_func, (HorizonWeightedTailLoss, ChangeRiskLoss)) or meter is not None, need_mean=False)
            if batch_number % self.gradient_accumulation == 0:
                self.optimizer.zero_grad()

            if getattr(self.model, 'event_aware', False):
                predictions, aux = self.model(**inputs, return_diagnostics=True)
            else:
                predictions, aux = self.predict_batch(self.model, inputs), None
            data_loss = self.data_loss(predictions, targets, scales, inputs)
            objective = self.training_loss(predictions, inputs, data_loss)
            if aux is not None:
                classification = self.rise_classification_loss(aux['rise_logits'], targets, scales, inputs)
                objective = objective + self.rise_classification_weight*classification
                regression_losses.append(data_loss.detach())
                classification_losses.append(classification.detach())
            (objective / self.gradient_accumulation).backward()
            batch_losses.append(objective.detach())
            with torch.no_grad():
                self.update_change_diagnostics(meter, predictions.detach(), targets, scales, inputs)

            last_batch = batch_number + 1 == len(self.train_dataloader)
            if (batch_number + 1) % self.gradient_accumulation == 0 or last_batch:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=getattr(self, "gradient_clip", 1.0))
                self.optimizer.step()

        self.save_change_diagnostics(meter, self.train_dataloader.dataset, 'train')
        if self.scheduler is not None:
            self.scheduler.step()
        self.last_train_components = {}
        if regression_losses:
            self.last_train_components = {'train/regression_loss': torch.stack(regression_losses).double().mean().item(),
                                          'train/rise_classification_loss': torch.stack(classification_losses).double().mean().item()}
        return torch.stack(batch_losses).double().mean().item() if batch_losses else 0.

    def evaluate_loss(self, dataloader: DataLoader, predictor=None) -> float:
        """Score the same validation objective for a learned model or persistence."""
        self.model.eval()
        predictor = self.model if predictor is None else predictor
        loss_sum, element_count, row_count = 0.0, 0, 0
        squared = None
        meter = self.new_change_diagnostics(dataloader.dataset)
        physical_squared = None
        physical_absolute = None
        physical_bad = None
        base_sum, tail_sum = 0.0, 0.0
        classification_sum, classification_n = 0., 0
        scaler = getattr(dataloader.dataset, "target_scaler", None)
        threshold = getattr(self, "validation_threshold_mm", 200.0)
        with torch.inference_mode():
            for batch in dataloader:
                inputs, targets, scales, _ = self.prepare_batch(batch, dataloader.dataset, need_mean=False)
                if predictor is self.model and getattr(self.model, 'event_aware', False):
                    predictions, aux = self.model(**inputs, return_diagnostics=True)
                    classification_sum += self.rise_classification_loss(aux['rise_logits'], targets, scales, inputs).double()*len(targets)
                    classification_n += len(targets)
                else:
                    predictions = self.predict_batch(predictor, inputs)
                loss = self.data_loss(predictions, targets, scales, inputs, apply_rise_weight=False)
                self.update_change_diagnostics(meter, predictions, targets, scales, inputs, apply_rise_weight=False)
                loss_sum = loss_sum+loss.double()*targets.numel()
                element_count += targets.numel()
                row_count += len(targets)
                errors = predictions.double() - targets.double()
                batch_squared = errors.square().sum(dim=0)
                squared = batch_squared if squared is None else squared + batch_squared
                if isinstance(self.loss_func, HorizonWeightedTailLoss):
                    base, tail = self.loss_func.components(predictions, targets, target_scales=scales)
                    base_sum = base_sum+base.double()*targets.numel()
                    tail_sum = tail_sum+tail.double()*targets.numel()
                if scales is not None:
                    error_mm = errors * scales
                    sq = error_mm.square().sum(dim=0)
                    absolute = error_mm.abs().sum(dim=0)
                    bad = (error_mm.abs() > threshold).sum(dim=0)
                    physical_squared = sq if physical_squared is None else physical_squared + sq
                    physical_absolute = absolute if physical_absolute is None else physical_absolute + absolute
                    physical_bad = bad if physical_bad is None else physical_bad + bad
        if element_count == 0:
            raise ValueError("Evaluation dataset has no eligible sequences")
        if not torch.isfinite(loss_sum):
            raise ValueError('Validation loss is non-finite')
        self.last_validation_mse = squared.cpu() / row_count
        self.last_validation_metrics = {}
        if classification_n:
            self.last_validation_metrics['val/rise_classification_loss'] = (classification_sum/classification_n).item()
        if physical_squared is not None:
            physical_squared, physical_bad = physical_squared.cpu(), physical_bad.cpu()
            physical_absolute = physical_absolute.cpu()
            self.last_validation_metrics.update({
                "val/large_error_count": int(physical_bad.sum()),
                "val/large_error_pct": 100 * physical_bad.sum().item() / element_count,
                "val/rmse_mm": math.sqrt(physical_squared.sum().item() / element_count),
                "val/mae_mm": physical_absolute.sum().item() / element_count,
            })
            for name, sq, absolute, bad in zip(dataloader.dataset.target_cols, physical_squared, physical_absolute, physical_bad):
                self.last_validation_metrics[f"val/{name}/rmse_mm"] = math.sqrt(sq.item() / row_count)
                self.last_validation_metrics[f"val/{name}/mae_mm"] = absolute.item() / row_count
                self.last_validation_metrics[f"val/{name}/large_error_pct"] = 100 * bad.item() / row_count
        elif getattr(self, "checkpoint_metric", "val_loss") in ("large_error_rate", "rmse"):
            raise ValueError("Physical-unit checkpoint selection requires the train-fitted target scaler")
        if isinstance(self.loss_func, HorizonWeightedTailLoss):
            self.last_validation_metrics.update({"val/base_weighted_mse": base_sum.item() / element_count,
                                                  "val/tail_penalty": tail_sum.item() / element_count})
        self.save_change_diagnostics(meter, dataloader.dataset, 'val')
        return loss_sum.item() / element_count

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
            self.save_checkpoint()
            print(f"New best validation score: {score} (saving to {self.checkpoint_path})")
        elif self.best_checkpoint_epoch == 0 and self.epochs_without_improvement == 0:
            initial = ("the initial event-aware model" if getattr(self.model, 'event_aware', False) else
                       "the initial direct model" if getattr(self.model, 'prediction_mode', 'residual') == 'direct' else "persistence")
            print(f"Validation improved, but {initial} is still the best checkpoint.")

    def restore_best_checkpoint(self, purpose: str = "testing") -> None:
        if self.best_checkpoint_epoch is None:
            raise ValueError("No checkpoint was selected; train for at least one epoch")
        if self.best_checkpoint_epoch == 0:
            initial = ("initial event-aware model" if getattr(self.model, 'event_aware', False) else
                       "initial direct model" if getattr(self.model, 'prediction_mode', 'residual') == 'direct' else "initial persistence baseline")
            print(f"Best checkpoint: {initial} (no trained epoch beat it).")
        else:
            print(f"Best checkpoint: epoch {self.best_checkpoint_epoch}.")
        print(f"\nLoading best checkpoint from {self.checkpoint_path} for {purpose}...")
        state_dict = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)

    def test(self) -> pd.DataFrame:
        self.model.eval()
        predictions = []
        targets = []
        predicted_physical, actual_physical, event_outputs = [], [], []
        meter = self.new_change_diagnostics(self.test_dataloader.dataset)
        with torch.inference_mode():
            for batch in tqdm(self.test_dataloader, desc="Testing sequence model"):
                inputs, batch_targets, scale, mean = self.prepare_batch(batch, self.test_dataloader.dataset)
                if getattr(self.model, 'event_aware', False):
                    output, aux = self.model(**inputs, return_diagnostics=True)
                    event_outputs.append(torch.stack([aux['rise_logits'], aux['rise_probability'],
                        aux['base_delta_normalized']*scale, aux['amplitude_normalized']*scale,
                        aux['correction_normalized']*scale], dim=-1).cpu())
                else:
                    output = self.predict_batch(self.model, inputs)
                self.update_change_diagnostics(meter, output, batch_targets, scale, inputs, apply_rise_weight=False)
                predictions.append(output.cpu())
                targets.append(batch_targets.cpu())
                if isinstance(batch, dict):
                    predicted_physical.append((output.double()*scale.double()+mean.double()).cpu())
                    actual_physical.append((batch_targets.double()*scale.double()+mean.double()).cpu())
        self.save_change_diagnostics(meter, self.test_dataloader.dataset, 'test')
        if not targets:
            raise ValueError("Test dataset has no eligible sequences")

        predicted_scaled = torch.cat(predictions).numpy()
        target_scaled = torch.cat(targets).numpy()
        mse = ((predicted_scaled - target_scaled) ** 2).mean()
        mlflow.log_metric("test_loss", float(mse))
        mlflow.set_tag("test_loss_scale", "scaled_target_units")
        print(f"Test Loss (MSE): {mse:.4f}")

        dataset = self.test_dataloader.dataset
        predicted = (torch.cat(predicted_physical).numpy() if predicted_physical
                     else dataset.target_scaler.inverse_transform(predicted_scaled))
        observed = (torch.cat(actual_physical).numpy() if actual_physical
                    else dataset.target_scaler.inverse_transform(target_scaled))
        if event_outputs:
            meta = dataset.view.samples
            horizons = dataset.target_horizons
            event = torch.cat(event_outputs).numpy().reshape(-1, 5)
            report = pd.DataFrame(dict(station_id=np.repeat(meta.station_id.to_numpy(),len(horizons)),
                anchor_utc=np.repeat(meta.anchor_utc.to_numpy(),len(horizons)),
                horizon_hours=np.tile(horizons,len(meta)), predicted_mm=predicted.ravel(),
                actual_mm=observed.ravel(), current_mm=np.repeat(meta.water_now_mm.to_numpy(),len(horizons))))
            for index, name in enumerate(('rise_logit','rise_probability','base_delta_mm','amplitude_mm','correction_mm')):
                report[name] = event[:, index]
            report.to_parquet(self.checkpoint_path.with_suffix('.test_predictions.parquet'), index=False)
        summaries = []
        for target_index, target_name in enumerate(dataset.target_cols):
            if getattr(dataset, 'regional_metadata', None):
                station_ids = dataset.view.samples.station_id.to_numpy()
                for sid in sorted(set(station_ids)):
                    mask = station_ids == sid
                    metrics = evaluate_all(observed[mask, target_index], predicted[mask, target_index],
                                           timestamps=dataset.timestamps.loc[mask])
                    metrics['target'] = f'{sid}/{target_name}'
                    summaries.append(metrics)
                error = predicted[:, target_index]-observed[:, target_index]
                summaries.append(dict(target=f'pooled/{target_name}', rmse=float((error**2).mean()**.5),
                    mae=float(abs(error).mean()), n200=int((abs(error) > self.validation_threshold_mm).sum())))
                continue
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
        if getattr(dataset, 'regional_metadata', None):
            station_table = station_error_table(observed, predicted, dataset.view.samples.station_id.to_numpy(),
                dataset.regional_metadata['station_mapping'], dataset.target_cols)
            print('RMSE / MAE in mm for every station; all = both horizons pooled within that station.')
            displayed = station_table.reset_index().drop(columns='station_id').set_index('name')
            displayed.index.name = 'Station'
            print(displayed.to_string(float_format=lambda value: f'{value:.3f}'))
            print('Mean across stations = arithmetic mean of station metrics, with equal station weights.')
            print('Pooled metrics below use all prediction errors together and are different from that mean.')
            pooled = summary.loc[summary.index.str.startswith('pooled/'), ['rmse', 'mae']].copy()
            pooled.index = [f"+{int(name.rsplit('_', 1)[-1][:-1])/60:g}h" for name in pooled.index]
            print(pooled.to_string())
            for target_name in dataset.target_cols:
                hours = f"{int(target_name.rsplit('_', 1)[-1][:-1])/60:g}"
                summary.loc[f'mean_across_stations/{target_name}', ['rmse', 'mae']] = [
                    station_table.loc['mean_across_stations', f'rmse_h{hours}_mm'],
                    station_table.loc['mean_across_stations', f'mae_h{hours}_mm']]
            summary.loc['mean_across_stations/all_horizons', ['rmse', 'mae']] = [
                station_table.loc['mean_across_stations', 'rmse_all_mm'],
                station_table.loc['mean_across_stations', 'mae_all_mm']]
            self.station_test_metrics = station_table
            output = self.checkpoint_path.with_suffix('.test_station_metrics.csv')
            station_table.to_csv(output)
            summary.to_csv(self.checkpoint_path.with_suffix('.test_metrics.csv'))
            print(f'Saved station metrics: {output}')
        else:
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
    def __init__(self, model: ResidualLSTM, correction_penalty: float = 0.0,
                 rise_classification_weight: float = 0., rise_positive_weight: float = 1., **kwargs) -> None:
        if not isinstance(model, (ResidualLSTM, ResidualSequence)):
            raise TypeError("ResidualLSTMTrainer requires a ResidualLSTM model")
        validate_correction_penalty(correction_penalty, kwargs["loss_func"], model.output_size)
        super().__init__(model=model, **kwargs)
        self.correction_penalty = correction_penalty
        self.rise_classification_weight = rise_classification_weight
        self.rise_positive_weight = rise_positive_weight

    def model_settings(self) -> dict[str, object]:
        settings = {
            "Layers": self.model.num_layers,
            "Hidden size": self.model.hidden_size,
            "LSTM dropout": self.model.dropout,
            "Input dropout": self.model.input_dropout,
            "Head dropout": self.model.head_dropout,
            "Correction penalty": self.correction_penalty,
        }
        if isinstance(self.model, ResidualSequence):
            settings['Encoder dropout'] = settings.pop('LSTM dropout')
            settings['Architecture'] = self.model.architecture_config['architecture']
            settings['Prediction mode'] = self.model.prediction_mode
            settings['Fusion'] = 'late' if self.model.late_fusion else 'early'
            settings['Station count'] = self.model.num_stations
            if self.model.head_mode != 'shared':
                settings['Head sharing'] = self.model.head_mode
            if self.model.head_mode == 'station-mlp':
                settings['Station MLP hidden layers'] = 3
                settings['Station MLP hidden size'] = self.model.architecture_config['station_mlp_hidden_size']
            if self.model.horizon_decoder_mode != 'none':
                settings['Horizon decoder'] = self.model.horizon_decoder_mode
            settings['TCN channels'] = self.model.architecture_config['tcn_channels']
            settings['CNN channels'] = self.model.architecture_config['cnn_channels']
            settings['Forecast MLP hidden size'] = self.model.architecture_config['forecast_hidden_size']
            if 'transformer_heads' in self.model.architecture_config:
                settings['Attention heads'] = self.model.architecture_config['transformer_heads']
                settings['Transformer feed-forward size'] = self.model.architecture_config['transformer_ff_size']
        return settings

    def prepare_checkpoint(self) -> None:
        baseline_loss = self.evaluate_loss(self.eval_dataloader, predictor=self.model.persistence)
        print(f"Validation persistence loss ({type(self.loss_func).__name__}): {baseline_loss:.4f}")
        for name, mse in zip(self.eval_dataloader.dataset.target_cols, self.last_validation_mse):
            print(f"  {name}: {mse:.4f}")
        initial_model = getattr(self.model, 'prediction_mode', 'residual') == 'direct' or getattr(self.model, 'event_aware', False)
        initial_loss = self.evaluate_loss(self.eval_dataloader) if initial_model else baseline_loss
        if initial_model:
            print(f"Initial model validation loss ({type(self.loss_func).__name__}): {initial_loss:.4f}")
        self.best_val_loss = initial_loss
        self.best_checkpoint_score = self.validation_score(initial_loss)
        self.best_validation_metrics = dict(self.last_validation_metrics)
        self.best_checkpoint_epoch = 0
        self.save_checkpoint()
        print(f"Saved initial {'model' if initial_model else 'persistence'} checkpoint to {self.checkpoint_path}")

    def training_loss(
        self,
        predictions: torch.Tensor,
        inputs: torch.Tensor,
        data_loss: torch.Tensor,
    ) -> torch.Tensor:
        if self.correction_penalty == 0:
            return data_loss
        correction = predictions - self.predict_batch(self.model.persistence, inputs)
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
