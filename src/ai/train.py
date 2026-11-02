"""Command-line entry point for hydrology model training."""

import os
import math
import json
import sys
import warnings
from pathlib import Path
from dataclasses import asdict

# Set this before importing PyTorch so CUDA LSTM kernels can be deterministic.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:2")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import mlflow
import mlflow.xgboost
import mlflow.lightgbm
import pandas as pd
import torch
from torch import nn
from torch import optim

from src.ai.change_objective import from_training_dataset
from src.ai.losses import HorizonWeightedMSE, HorizonWeightedTailLoss
from src.ai.lr_schedules import warmup_cosine
from src.ai.comparison import (
    DEFAULT_REGISTRY, mlflow_comparison_metrics, run_post_training_comparison,
)
from src.ai.metrics import evaluate_all
from src.ai.models import (
    LightGBM, LightGBMSettings, LSTM, MeanEmbeddingForecastLSTMWithAdapter,
    ResidualLSTM, XGBoost, XGBoostSettings,
)
from src.ai.trainers import LSTMTrainer, MEFLSTMAdapterTrainer, ResidualLSTMTrainer, VanillaLSTMTrainer
from src.ai.residual_sequence import NEW_METHODS, residual_model
from src.ai.regional_training_data import regional_loaders, regional_persistence
from src.ai.utils import (
    FEATURE_SETS, dataset_directory, get_dataloader, load_dataset,
    load_mef_backbone, normalize_feature_set, set_seed,
)


MLFLOW_TRACKING_URI = "http://localhost:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--mlflow-ex-name", type=str.strip, required=True,
    help="MLflow experiment name, for example 30min or 1h",
)
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible training")
parser.add_argument("--horizon", required=True, choices=["10min", "30min", "1h", "12h", "24h"])
parser.add_argument("--dataset-dir", type=Path,
                    help="Directory containing train.csv, val.csv and test.csv; overrides the horizon default")
parser.add_argument("--outputs", type=int, choices=(5, 24),
                    help="Sequence targets: 5 common horizons or all 24 hourly targets; omitted keeps dataset targets")
parser.add_argument("--target-horizons", type=int, nargs="+",
                    help="Exact forecast horizons in physical hours, e.g. 12 24 or 24")
parser.add_argument(
    "--num-rain-station", type=int,
    help="Keep the top N rain stations ranked on train (6h rain / 12h water change); omitted keeps all",
)
parser.add_argument(
    "--rain-stations",
    help="Exact named combination, e.g. hong_thai+vp+bach_ma+phu_bai; no fixed count limit",
)
parser.add_argument(
    "--checkpoint-dir", type=Path,
    help="Optional checkpoint root; explicit rain selections also get a station-specific subdirectory",
)
parser.add_argument("--comparison-station-id", default="0074647403",
                    help="App registry station to compare after sequence training (default Da Vien)")
parser.add_argument("--comparison-registry", type=Path, default=DEFAULT_REGISTRY,
                    help="App model registry used for automatic post-training comparison")
parser.add_argument("--comparison-model-id", default="model_b",
                    help="Reference model ID for post-training comparison (default: model_b)")
parser.add_argument("--train-method", required=True, choices=["xgb", "lgbm", "lstm", "residual_lstm", "vanilla_lstm", "lstm_adapter", *NEW_METHODS],
                    help='lstm is a residual baseline; vanilla_lstm retains the previous non-residual training option')
parser.add_argument('--rain-station', dest='rain_stations', help='Alias of --rain-stations')
parser.add_argument('--water-station', help='Train one station from a regional pool; omitted uses all stations with embeddings')
parser.add_argument('--regional-fold', default=None, choices=['fixed', 'rainy', 'winter', 'spring', 'summer'],
                    help='Optional override for legacy research datasets; normally use the split saved in the dataset')
parser.add_argument('--time-features', default='none', choices=['none', 'hour', 'day_of_year', 'all'])
def boolean(value):
    if value.lower() not in ('true', 'false'):
        raise argparse.ArgumentTypeError('Use true or false')
    return value.lower() == 'true'
parser.add_argument('--use-weather-forecast', type=boolean, default=None)
parser.add_argument('--mlflow-tracking-uri', help='Override the default MLflow tracking server')
parser.add_argument('--tcn-channels', type=int, nargs='+', default=[32]*5)
parser.add_argument('--tcn-kernel-size', type=int, default=3)
parser.add_argument('--cnn-channels', type=int, nargs='+', default=[32, 32])
parser.add_argument('--cnn-kernel-size', type=int, default=3)
parser.add_argument('--forecast-hidden-size', type=int, default=32)
parser.add_argument('--station-embedding-size', type=int, default=8)
parser.add_argument('--transformer-heads', type=int, default=4,
                    help='Attention heads; must divide --lstm-hidden-size for Transformer models')
parser.add_argument('--transformer-ff-size', type=int, default=128,
                    help='Transformer feed-forward width; hidden size/layers/dropout use --lstm-* flags')
parser.add_argument('--head-mode', choices=['shared', 'station', 'station-mlp'], default='shared',
                    help='Regional correction head: shared, per-station linear, or per-station 3-layer MLP')
parser.add_argument('--prediction-mode', choices=['residual', 'direct'], default='residual',
                    help='Predict a correction to persistence or the normalized water level directly')
parser.add_argument('--event-aware', action='store_true', help='Enable supervised rise probability and positive correction heads')
parser.add_argument('--rise-classification-weight', type=float, default=0.1)
parser.add_argument('--rise-positive-weight', type=float, default=1.0)
parser.add_argument('--rise-event-weight', type=float, default=0., help='Extra regression weight for observed rises > threshold')
parser.add_argument('--station-mlp-hidden-size', type=int, default=32,
                    help='Hidden width of each of the three station MLP layers (default: 32)')
parser.add_argument('--horizon-decoder', choices=['none', 'transformer'], default='none',
                    help='Optional Transformer horizon-query decoder for regional late-fusion station heads')
parser.add_argument('--diagnose-change-dynamics', action='store_true',
                    help='Save epoch snapshots, eval-mode train/validation regimes and validation predictions')
parser.add_argument('--dynamics-every', type=int, default=1, help='Diagnostic epoch interval; also evaluate epoch 0 and final epoch')
parser.add_argument('--diagnose-station-gradients', action='store_true',
                    help='Probe encoder gradients on fixed, train-only stratified samples; requires dynamics')
parser.add_argument('--gradient-probe-samples', type=int, default=32, help='Maximum fixed train anchors per station for gradient diagnostics')
parser.add_argument('--lr-scheduler', choices=['step', 'warmup-cosine'], default='step',
                    help='Sequence learning-rate schedule; step preserves the existing schedule')
parser.add_argument('--lr-warmup-epochs', type=int, default=3)
parser.add_argument('--lr-min', type=float, default=1e-6)
parser.add_argument('--skip-test', action='store_true', help='Do not run test or post-training model comparison')
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--num-worker", type=int, default=0)
parser.add_argument("--num-epochs", type=int, default=100)
parser.add_argument("--early-stopping", type=int, default=10)
parser.add_argument("--gradient-accumulation", type=int, default=1)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--pretrained-model", help="MEF checkpoint state_dict path for lstm_adapter")
parser.add_argument("--pretrained-config", help="MEF YAML configuration path for lstm_adapter")
parser.add_argument(
    "--mef-unfreeze-backbone", action="store_true",
    help="Train the pretrained MEF backbone as well as the adapter and output head",
)
parser.add_argument("--mef-two-stage", action="store_true",
                    help="Train frozen MEF first, then fine-tune from its best checkpoint")
parser.add_argument("--mef-finetune-epochs", type=int, default=30,
                    help="Additional stage-two epochs; --num-epochs controls stage one")
parser.add_argument("--mef-finetune-backbone", choices=["forecast", "all"], default="forecast",
                    help="Recurrent MEF modules to unfreeze in stage two")
parser.add_argument("--mef-backbone-lr", type=float, default=1e-5,
                    help="Stage-two learning rate for unfrozen recurrent weights")
parser.add_argument("--mef-adapter-lr", type=float, default=1e-4,
                    help="Stage-two learning rate for the adapter and output head")
parser.add_argument("--lstm-output-size", type=int, default=3)
parser.add_argument("--lstm-hidden-size", '--hidden-size', type=int, default=128)
parser.add_argument("--lstm-num-layers", '--num-layers', type=int, default=2)
parser.add_argument("--lstm-drop-out", '--dropout', type=float, default=0.2)
parser.add_argument("--lstm-input-dropout", type=float, default=0.0)
parser.add_argument("--lstm-head-dropout", type=float, default=0.0)
parser.add_argument(
    "--lstm-correction-penalty", type=float, default=0.0,
    help="Residual squared penalty; negative values reward corrections (bounded MSE only)",
)
parser.add_argument(
    "--lstm-horizon-weights", type=float, nargs="+",
    help="Positive MSE weights in dataset target-column order; default equal weights",
)
parser.add_argument("--lstm-tail-penalty", type=float, default=0.0,
                    help="Strength of physical-unit softplus penalty near the error threshold; 0 disables")
parser.add_argument("--lstm-tail-threshold-mm", type=float, default=200.0)
parser.add_argument("--lstm-tail-transition-mm", type=float, default=20.0)
parser.add_argument("--lstm-checkpoint-metric", choices=["val_loss", "large_error_rate"], default="val_loss",
                    help="Select checkpoint/early stop by loss or large-error count then pooled RMSE on validation")
parser.add_argument(
    "--feature-set", "--lstm-feature-set", dest="feature_set",
    type=normalize_feature_set, choices=FEATURE_SETS, default="all",
    help="Input groups for tree and LSTM models; water is an alias for water_level",
)
parser.add_argument("--lstm-lr", type=float, default=1e-4)
parser.add_argument("--lstm-weight-decay", type=float, default=0.01)
parser.add_argument("--lstm-seq-len", type=int)

xgb_defaults = XGBoostSettings()
xgb_args = parser.add_argument_group("XGBoost settings")
xgb_args.add_argument("--xgb-n-estimators", type=int, default=xgb_defaults.n_estimators)
xgb_args.add_argument("--xgb-learning-rate", type=float, default=xgb_defaults.learning_rate)
xgb_args.add_argument("--xgb-max-depth", type=int, default=xgb_defaults.max_depth)
xgb_args.add_argument("--xgb-min-child-weight", type=float, default=xgb_defaults.min_child_weight)
xgb_args.add_argument("--xgb-subsample", type=float, default=xgb_defaults.subsample)
xgb_args.add_argument("--xgb-colsample-bytree", type=float, default=xgb_defaults.colsample_bytree)
xgb_args.add_argument("--xgb-reg-alpha", type=float, default=xgb_defaults.reg_alpha)
xgb_args.add_argument("--xgb-reg-lambda", type=float, default=xgb_defaults.reg_lambda)
xgb_args.add_argument("--xgb-early-stopping-rounds", type=int, default=xgb_defaults.early_stopping_rounds,
                      help="Set to 0 to disable early stopping")
xgb_args.add_argument("--xgb-n-jobs", type=int, default=xgb_defaults.n_jobs)

lgbm_defaults = LightGBMSettings()
lgbm_args = parser.add_argument_group("LightGBM settings")
lgbm_args.add_argument("--lgbm-n-estimators", type=int, default=lgbm_defaults.n_estimators)
lgbm_args.add_argument("--lgbm-learning-rate", type=float, default=lgbm_defaults.learning_rate)
lgbm_args.add_argument("--lgbm-num-leaves", type=int, default=lgbm_defaults.num_leaves)
lgbm_args.add_argument("--lgbm-max-depth", type=int, default=lgbm_defaults.max_depth)
lgbm_args.add_argument("--lgbm-subsample", type=float, default=lgbm_defaults.subsample)
lgbm_args.add_argument("--lgbm-subsample-freq", type=int, default=lgbm_defaults.subsample_freq)
lgbm_args.add_argument("--lgbm-colsample-bytree", type=float, default=lgbm_defaults.colsample_bytree)
lgbm_args.add_argument("--lgbm-reg-alpha", type=float, default=lgbm_defaults.reg_alpha)
lgbm_args.add_argument("--lgbm-reg-lambda", type=float, default=lgbm_defaults.reg_lambda)
lgbm_args.add_argument("--lgbm-early-stopping-rounds", type=int, default=lgbm_defaults.early_stopping_rounds,
                       help="Set to 0 to disable early stopping")
lgbm_args.add_argument("--lgbm-n-jobs", type=int, default=lgbm_defaults.n_jobs)
lgbm_args.add_argument("--lgbm-verbose", type=int, default=lgbm_defaults.verbose)


parser.add_argument('--change-loss', choices=['baseline', 'station-balanced', 'event-weighted', 'asymmetric', 'combined', 'large-error-count', 'large-error-primary'], default='baseline')
parser.add_argument('--change-scale-floor-mm', type=float, default=200.)
parser.add_argument('--change-event-weight', type=float, default=1.)
parser.add_argument('--change-under-weight', type=float, default=1.)
parser.add_argument('--change-threshold-mm', type=float, default=200.)
parser.add_argument('--large-error-penalty', type=float, default=0.2,
                    help='Strength of the smooth prediction-error count penalty')
parser.add_argument('--large-error-threshold-mm', type=float, default=200.)
parser.add_argument('--large-error-transition-mm', type=float, default=20.,
                    help='Smoothing width in mm around the large-error threshold')


def is_regional_dataset(args):
    return args.dataset_dir is not None and (args.dataset_dir/'samples.parquet').is_file()


def validate_args(args: argparse.Namespace) -> None:
    regional = is_regional_dataset(args)
    if args.event_aware and (not regional or args.water_station is not None or args.train_method != 'transformer-late-fusion'
                             or args.prediction_mode != 'residual' or args.horizon_decoder != 'none'
                             or args.change_loss != 'station-balanced'):
        parser.error('--event-aware requires station-conditioned residual transformer-late-fusion and station-balanced loss')
    if (not math.isfinite(args.rise_classification_weight) or args.rise_classification_weight < 0
            or not math.isfinite(args.rise_positive_weight) or args.rise_positive_weight <= 0
            or not math.isfinite(args.rise_event_weight) or args.rise_event_weight < 0):
        parser.error('Rise weights must be finite; classification/event weights nonnegative and positive weight positive')
    if args.rise_event_weight and args.change_loss != 'station-balanced':
        parser.error('--rise-event-weight requires station-balanced loss')
    if args.lr_scheduler == 'warmup-cosine':
        if args.train_method in ('xgb', 'lgbm') or args.mef_two_stage:
            parser.error('--lr-scheduler warmup-cosine requires single-stage sequence training')
        if args.lr_warmup_epochs < 1 or args.num_epochs < args.lr_warmup_epochs + 2:
            parser.error('Warmup-cosine requires at least one warmup and two decay epochs')
        if (not math.isfinite(args.lr_min) or not math.isfinite(args.lstm_lr)
                or args.lr_min < 0 or args.lstm_lr <= 0 or args.lr_min > args.lstm_lr):
            parser.error('Warmup-cosine requires 0 <= --lr-min <= a positive --lstm-lr')
    if args.prediction_mode == 'direct' and (not regional or args.water_station is not None):
        parser.error('--prediction-mode direct currently requires a regional station-conditioned model')
    if args.prediction_mode == 'direct' and args.lstm_correction_penalty != 0:
        parser.error('--prediction-mode direct requires --lstm-correction-penalty 0')
    if args.skip_test and args.train_method in ('xgb','lgbm'):
        parser.error('--skip-test is currently supported for sequence models only')
    if args.head_mode != 'shared' and (not regional or args.water_station is not None):
        parser.error('Station heads require regional station conditioning')
    if args.head_mode == 'station-mlp' and args.train_method not in (*NEW_METHODS, 'lstm', 'residual_lstm'):
        parser.error('--head-mode station-mlp requires a supported residual sequence model')
    if args.station_mlp_hidden_size < 1:
        parser.error('--station-mlp-hidden-size must be positive')
    if args.head_mode != 'station-mlp' and args.station_mlp_hidden_size != 32:
        parser.error('--station-mlp-hidden-size requires --head-mode station-mlp')
    if args.horizon_decoder == 'transformer' and (args.head_mode != 'station' or not args.train_method.endswith('-late-fusion')):
        parser.error('--horizon-decoder transformer requires late fusion and --head-mode station')
    if args.diagnose_change_dynamics:
        if not regional or args.water_station is not None:
            parser.error('--diagnose-change-dynamics requires regional station conditioning')
        if args.lstm_tail_penalty or args.lstm_correction_penalty:
            parser.error('Dynamics diagnostics require zero tail/correction penalties')
    if args.dynamics_every < 1 or args.gradient_probe_samples < 2:
        parser.error('Diagnostic interval must be positive and probe samples must be at least 2')
    if args.diagnose_station_gradients and not args.diagnose_change_dynamics:
        parser.error('--diagnose-station-gradients requires --diagnose-change-dynamics')
    if args.change_loss in ('station-balanced', 'large-error-count', 'large-error-primary') and (args.change_event_weight or args.change_under_weight):
        warnings.warn(f'{args.change_loss} ignores event/under weights; use combined to activate them. '
                      'Set both weights to 0 for the B1 control.', UserWarning)
    if args.change_loss != 'baseline':
        if not regional or args.water_station is not None:
            parser.error('--change-loss requires regional station-conditioned training')
        if args.lstm_tail_penalty != 0 or args.lstm_correction_penalty != 0:
            parser.error('For change-loss ablations, tail and correction penalties must be zero')
    if args.target_horizons is not None:
        if args.outputs is not None:
            parser.error("Use --target-horizons or --outputs, not both")
        if args.train_method in {"xgb", "lgbm"}:
            parser.error("--target-horizons currently supports sequence models only")
        if len(set(args.target_horizons)) != len(args.target_horizons) or min(args.target_horizons) < 1:
            parser.error("--target-horizons requires distinct positive hours")
    if args.outputs is not None and args.train_method in {"xgb", "lgbm"}:
        parser.error("--outputs currently supports sequence models only")
    if args.rain_stations is not None:
        if args.num_rain_station is not None:
            parser.error("Use --rain-stations or --num-rain-station, not both")
        if args.train_method in {"xgb", "lgbm"}:
            parser.error("--rain-stations currently supports sequence models only")
        if args.feature_set != "all" and "rain" not in args.feature_set.split("+"):
            parser.error("--rain-stations requires rain in --feature-set")
    if args.num_rain_station is not None:
        if args.num_rain_station < 1:
            parser.error("--num-rain-station must be positive")
        if args.train_method in {"xgb", "lgbm"}:
            parser.error("--num-rain-station currently supports sequence models only")
        if args.feature_set != "all" and "rain" not in args.feature_set.split("+"):
            parser.error("--num-rain-station requires rain in --feature-set")
    if args.checkpoint_dir is not None and args.train_method in {"xgb", "lgbm"}:
        parser.error("--checkpoint-dir currently supports sequence models only")
    if regional:
        if args.regional_fold is not None:
            available = json.loads((args.dataset_dir/'normalizers.json').read_text())
            if args.regional_fold not in available:
                parser.error(f'Dataset has splits {list(available)}; omit --regional-fold to use its saved fixed split')
        if args.train_method not in (*NEW_METHODS, 'lstm', 'residual_lstm'):
            parser.error('Regional datasets require a residual sequence model')
        if args.outputs is not None:
            parser.error('For regional datasets, select hours with --target-horizons')
        if args.target_horizons is not None:
            args.target_horizons = sorted(args.target_horizons)
        if args.lstm_seq_len is not None and args.lstm_seq_len < 1:
            parser.error('--lstm-seq-len must be positive')
        if (args.target_horizons is not None and args.lstm_horizon_weights is not None
                and len(args.lstm_horizon_weights) != len(args.target_horizons)):
            parser.error(f'{len(args.target_horizons)} target horizons require {len(args.target_horizons)} horizon weights; use --lstm-horizon-weights '+ ' '.join(['1']*len(args.target_horizons)))
        if args.num_rain_station is not None:
            parser.error('Select explicit shared rain stations for regional training')
        if args.horizon != '1h':
            parser.error('Regional hourly datasets require --horizon 1h')
        if args.feature_set != 'all' and not {'water_level', 'flood_season'}.issubset(args.feature_set.split('+')):
            parser.error('Regional feature sets retain water_level and flood_season; select rain/weather with their groups or dedicated flags')
    elif args.water_station is not None:
        parser.error('--water-station requires a regional dataset pool')
    elif args.time_features != 'none':
        parser.error('--time-features requires a regional dataset with stored cyclic features')
    if args.dataset_dir is not None and not regional:
        for split in ("train", "val", "test"):
            if not (args.dataset_dir / f"{split}.csv").is_file():
                parser.error(f"Dataset split not found: {args.dataset_dir / (split + '.csv')}")
    if not args.mlflow_ex_name:
        parser.error("--mlflow-ex-name must not be empty")
    if args.mef_unfreeze_backbone and args.train_method != "lstm_adapter":
        parser.error("--mef-unfreeze-backbone requires --train-method lstm_adapter")
    if args.mef_two_stage:
        if args.train_method != "lstm_adapter":
            parser.error("--mef-two-stage requires --train-method lstm_adapter")
        if args.mef_unfreeze_backbone:
            parser.error("--mef-two-stage starts frozen; omit --mef-unfreeze-backbone")
        if args.num_epochs < 1 or args.mef_finetune_epochs < 1:
            parser.error("Both training stages require at least one epoch")
        if not (0 < args.mef_backbone_lr < float("inf") and 0 < args.mef_adapter_lr < float("inf")):
            parser.error("MEF fine-tuning learning rates must be finite and positive")
    if args.lstm_horizon_weights is not None:
        if args.train_method in {"xgb", "lgbm"}:
            parser.error("--lstm-horizon-weights requires a sequence model")
        if not all(math.isfinite(w) and w > 0 for w in args.lstm_horizon_weights):
            parser.error("--lstm-horizon-weights must be finite and positive")
    if not math.isfinite(args.lstm_tail_penalty) or args.lstm_tail_penalty < 0:
        parser.error("--lstm-tail-penalty must be finite and nonnegative")
    if not all(math.isfinite(v) and v > 0 for v in (args.lstm_tail_threshold_mm, args.lstm_tail_transition_mm)):
        parser.error("Tail threshold and transition must be finite and positive")
    if args.train_method in {"xgb", "lgbm"} and (args.lstm_tail_penalty or args.lstm_checkpoint_metric != "val_loss"):
        parser.error("Tail loss and checkpoint metric require a sequence model")
    if not math.isfinite(args.lstm_correction_penalty) or args.lstm_correction_penalty <= -1:
        parser.error("--lstm-correction-penalty must be finite and > -1; weighted MSE may require a stricter bound")
    if args.lstm_correction_penalty and args.train_method not in (*NEW_METHODS, 'lstm', 'residual_lstm') and not regional:
        parser.error("--lstm-correction-penalty requires --train-method residual_lstm")
    if args.train_method in (*NEW_METHODS, 'lstm', 'residual_lstm'):
        groups = args.feature_set.split("+")
        if args.feature_set != "all" and "water_level" not in groups:
            parser.error("residual_lstm requires water_level in --feature-set")
    if args.train_method == "lstm_adapter":
        if not args.pretrained_model or not args.pretrained_config:
            parser.error("lstm_adapter requires --pretrained-model and --pretrained-config")
        for path in (args.pretrained_model, args.pretrained_config):
            if not Path(path).is_file():
                parser.error(f"MEF file not found: {path}")

    if any(n < 1 for n in [*args.tcn_channels, *args.cnn_channels, args.tcn_kernel_size, args.cnn_kernel_size,
                           args.forecast_hidden_size, args.station_embedding_size, args.lstm_hidden_size, args.lstm_num_layers]):
        parser.error('Architecture sizes and kernel sizes must be positive')
    if args.train_method.removesuffix('-late-fusion') == 'transformer' or args.horizon_decoder == 'transformer':
        if args.transformer_heads < 1 or args.transformer_ff_size < 1:
            parser.error('Transformer heads and feed-forward size must be positive')
        if args.lstm_hidden_size % args.transformer_heads:
            parser.error('--lstm-hidden-size must be divisible by --transformer-heads')

    if args.train_method == "xgb":
        if args.xgb_n_estimators < 1 or args.xgb_early_stopping_rounds < 0:
            parser.error("XGBoost estimators must be positive and early stopping must be non-negative")
        if args.xgb_learning_rate <= 0 or args.xgb_max_depth < 0 or args.xgb_min_child_weight < 0:
            parser.error("XGBoost learning rate must be positive; depth and child weight must be non-negative")
        proportions = (args.xgb_subsample, args.xgb_colsample_bytree)
        regularizers = (args.xgb_reg_alpha, args.xgb_reg_lambda)
    elif args.train_method == "lgbm":
        if args.lgbm_n_estimators < 1 or args.lgbm_early_stopping_rounds < 0:
            parser.error("LightGBM estimators must be positive and early stopping must be non-negative")
        if args.lgbm_learning_rate <= 0 or args.lgbm_num_leaves < 2:
            parser.error("LightGBM learning rate must be positive and num leaves must be at least 2")
        if args.lgbm_max_depth != -1 and args.lgbm_max_depth < 1:
            parser.error("LightGBM max depth must be -1 or positive")
        if args.lgbm_subsample_freq < 0:
            parser.error("LightGBM subsample frequency must be non-negative")
        proportions = (args.lgbm_subsample, args.lgbm_colsample_bytree)
        regularizers = (args.lgbm_reg_alpha, args.lgbm_reg_lambda)
    else:
        return

    if any(not 0 < value <= 1 for value in proportions):
        parser.error("Tree sampling fractions must be in (0, 1]")
    if any(value < 0 for value in regularizers):
        parser.error("Tree regularization values must be non-negative")


def evaluate_tabular(model: XGBoost | LightGBM, test_df: pd.DataFrame) -> pd.DataFrame:
    predictions = model.predict(test_df)
    observed = test_df[model.target_columns].to_numpy(dtype=float)
    predicted = predictions[model.target_columns].to_numpy(dtype=float)
    test_loss = float(((predicted - observed) ** 2).mean())
    mlflow.log_metric("test_loss", test_loss)
    mlflow.set_tag("test_loss_scale", "original_target_units")
    print(f"Test Loss (MSE, original target units): {test_loss:.4f}")
    timestamps = test_df["timestamp_utc"] if "timestamp_utc" in test_df.columns else None
    summaries = []
    for target_name in model.target_columns:
        metrics = evaluate_all(test_df[target_name], predictions[target_name], timestamps=timestamps)
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


def log_dataset_source(args: argparse.Namespace) -> None:
    directory = dataset_directory(args.horizon, args.dataset_dir).resolve()
    mlflow.log_param("dataset_dir", str(directory))
    manifest = directory / "manifest.json"
    if manifest.is_file():
        mlflow.log_artifact(str(manifest), artifact_path="dataset")


def train_tabular(args: argparse.Namespace) -> None:
    train_df, eval_df, test_df = load_dataset(args.horizon, args.dataset_dir)
    # The hourly sequence files retain missing-target rows to preserve cadence.
    # Trees need complete labels, but missing predictors must remain available.
    targets = [column for column in train_df if column.startswith("target_")]
    train_df, eval_df, test_df = [frame.dropna(subset=targets).copy()
                                 for frame in (train_df, eval_df, test_df)]
    if any(frame.empty for frame in (train_df, eval_df, test_df)):
        raise ValueError("Each dataset split must contain complete target rows")
    if args.train_method == "xgb":
        settings = XGBoostSettings(
            n_estimators=args.xgb_n_estimators,
            learning_rate=args.xgb_learning_rate,
            max_depth=args.xgb_max_depth,
            min_child_weight=args.xgb_min_child_weight,
            subsample=args.xgb_subsample,
            colsample_bytree=args.xgb_colsample_bytree,
            reg_alpha=args.xgb_reg_alpha,
            reg_lambda=args.xgb_reg_lambda,
            early_stopping_rounds=args.xgb_early_stopping_rounds,
            n_jobs=args.xgb_n_jobs,
        )
        model = XGBoost(settings=settings, feature_set=args.feature_set)
    else:
        settings = LightGBMSettings(
            n_estimators=args.lgbm_n_estimators,
            learning_rate=args.lgbm_learning_rate,
            num_leaves=args.lgbm_num_leaves,
            max_depth=args.lgbm_max_depth,
            subsample=args.lgbm_subsample,
            subsample_freq=args.lgbm_subsample_freq,
            colsample_bytree=args.lgbm_colsample_bytree,
            reg_alpha=args.lgbm_reg_alpha,
            reg_lambda=args.lgbm_reg_lambda,
            early_stopping_rounds=args.lgbm_early_stopping_rounds,
            n_jobs=args.lgbm_n_jobs,
            verbose=args.lgbm_verbose,
        )
        model = LightGBM(settings=settings, feature_set=args.feature_set)

    run_name = f"{args.train_method}_{args.horizon}_{args.feature_set}"
    with mlflow.start_run(run_name=run_name):
        log_dataset_source(args)
        mlflow.log_params(
            {
                "train_method": args.train_method,
                "horizon": args.horizon,
                "feature_set": args.feature_set,
                "seed": args.seed,
                "train_size": len(train_df),
                "eval_size": len(eval_df),
                "test_size": len(test_df),
                **asdict(model.settings)
            }
        )
        model.train(train_df, eval_df, random_state=args.seed)

        mlflow.log_param("input_size", len(model.feature_columns))
        mlflow.log_dict({"feature_columns": model.feature_columns}, "features.json")

        summary = evaluate_tabular(model, test_df)

        metrics = {}
        for target_name, target_metrics in summary.iterrows():
            for metric_name, value in target_metrics.items():
                numeric_value = float(value)
                if math.isfinite(numeric_value):
                    metrics[f"test/{target_name}/{metric_name}"] = numeric_value

        mlflow.log_metrics(metrics)
        mlflow.log_table(summary.reset_index(), f"{run_name}_test_results.json")

        for target_name, estimator in model.models.items():
            model_name = f"{args.train_method}_{target_name}"

            if args.train_method == "xgb":
                model_info = mlflow.xgboost.log_model(estimator, name=model_name)
            else:
                model_info = mlflow.lightgbm.log_model(estimator, name=model_name)

            mlflow.set_tag(f"model_uri/{target_name}", model_info.model_uri)


def build_sequence_trainer(args: argparse.Namespace) -> LSTMTrainer:
    regional = is_regional_dataset(args)
    if args.change_loss != 'baseline':
        if not regional or args.water_station is not None:
            raise ValueError('Change loss currently requires regional station-conditioned training')
        if args.lstm_tail_penalty != 0 or args.lstm_correction_penalty != 0:
            raise ValueError('Isolate change loss: set tail and correction penalties to zero')
    default_sequence_lengths = {"10min": 144, "30min": 48, "1h": 24, "12h": 9, "24h": 5}
    sequence_length = (
        default_sequence_lengths[args.horizon]
        if args.lstm_seq_len is None
        else args.lstm_seq_len
    )
    includes_current_level = (
        args.feature_set == "all"
        or "water_level" in args.feature_set.split("+")
    )
    if regional:
        groups = args.feature_set.split('+')
        rain = (args.rain_stations or 'nham+bach_ma+ho_hoa_my+hong_thai') if args.feature_set == 'all' or 'rain' in groups else 'none'
        weather = (args.feature_set == 'all' or 'weather_forecast' in groups) if args.use_weather_forecast is None else args.use_weather_forecast
        train_loader, eval_loader, test_loader = regional_loaders(args.dataset_dir,
            batch_size=args.batch_size, num_workers=args.num_worker, seed=args.seed,
            fold=args.regional_fold, rain_stations=rain,
            use_weather_forecast=weather,
            time_features=args.time_features, station_id=args.water_station,
            sequence_length=args.lstm_seq_len, target_horizons=args.target_horizons)
    else:
        train_loader, eval_loader, test_loader = get_dataloader(
        args.horizon,
        args.batch_size,
        args.num_worker,
        pin_memory=True,
        drop_last=True,
        train_shuffle=True,
        eval_shuffle=False,
        test_shuffle=False,
        persistent_workers=False,
        seq_len=sequence_length,
        seed=args.seed,
        require_current_level=includes_current_level,
        feature_set=args.feature_set,
        dataset_dir=args.dataset_dir,
        num_rain_station=args.num_rain_station,
        rain_stations=args.rain_stations,
        outputs=args.outputs,
        target_horizons=args.target_horizons,
        use_weather_forecast=args.use_weather_forecast,
        )
    input_size = train_loader.dataset.num_features
    output_size = train_loader.dataset.num_targets
    if args.lstm_horizon_weights is not None and len(args.lstm_horizon_weights) != output_size:
        raise ValueError(
            f"Expected {output_size} horizon weights for targets "
            f"{train_loader.dataset.target_cols}; got {len(args.lstm_horizon_weights)}"
        )

    if args.train_method == "lstm_adapter":
        model = MeanEmbeddingForecastLSTMWithAdapter(
            load_mef_backbone(args.pretrained_config, args.pretrained_model),
            input_size,
            output_size,
            freeze_backbone=not args.mef_unfreeze_backbone,
            input_dropout=args.lstm_input_dropout,
            head_dropout=args.lstm_head_dropout,
        )
        parameters = (parameter for parameter in model.parameters() if parameter.requires_grad)
        trainer_class = MEFLSTMAdapterTrainer
    else:
        model_options = dict(
            input_size=input_size,
            hidden_size=args.lstm_hidden_size,
            num_layers=args.lstm_num_layers,
            dropout=args.lstm_drop_out,
            input_dropout=args.lstm_input_dropout,
            head_dropout=args.lstm_head_dropout,
            output_size=output_size,
        )
        if regional or args.train_method in (*NEW_METHODS, 'lstm'):
            dataset = train_loader.dataset
            level_index = dataset.feature_cols.index('water_level_depth')
            weather = [i for i, col in enumerate(dataset.feature_cols) if col.startswith('weather_forecast_')]
            masks = [i for i in weather if dataset.feature_cols[i].endswith('_missing')]
            if regional:
                scale, offset, station_count = regional_persistence(dataset, regional=args.water_station is None)
                mask_values = [1.]*len(masks)
            else:
                fs, ts = dataset.feature_scaler, dataset.target_scaler
                scale, offset = (fs.scale_[level_index]/ts.scale_).tolist(), ((fs.mean_[level_index]-ts.mean_)/ts.scale_).tolist()
                station_count = 0
                mask_values = ((1.-fs.mean_[masks])/fs.scale_[masks]).tolist()
            model = residual_model('lstm' if args.train_method == 'residual_lstm' else args.train_method,
                **model_options, level_feature_index=level_index, persistence_scale=scale, persistence_offset=offset,
                weather_feature_indices=weather, weather_mask_indices=masks, weather_mask_values=mask_values,
                forecast_history_policy='anchor_only', num_stations=station_count,
                station_embedding_size=args.station_embedding_size, forecast_hidden_size=args.forecast_hidden_size,
                tcn_channels=args.tcn_channels, tcn_kernel_size=args.tcn_kernel_size,
                cnn_channels=args.cnn_channels, cnn_kernel_size=args.cnn_kernel_size,
                transformer_heads=args.transformer_heads, transformer_ff_size=args.transformer_ff_size,
                head_mode=args.head_mode, horizon_decoder=args.horizon_decoder,
                station_mlp_hidden_size=args.station_mlp_hidden_size, prediction_mode=args.prediction_mode,
                event_aware=args.event_aware)
            trainer_class = ResidualLSTMTrainer
        elif args.train_method == 'residual_lstm':
            dataset = train_loader.dataset
            level_index = dataset.feature_cols.index("water_level_depth")
            level_mean = dataset.feature_scaler.mean_[level_index]
            level_scale = dataset.feature_scaler.scale_[level_index]
            target_mean = dataset.target_scaler.mean_
            target_scale = dataset.target_scaler.scale_
            model = ResidualLSTM(
                **model_options,
                level_feature_index=level_index,
                persistence_scale=(level_scale / target_scale).tolist(),
                persistence_offset=((level_mean - target_mean) / target_scale).tolist(),
            )
            trainer_class = ResidualLSTMTrainer
        else:
            model = LSTM(**model_options)
            trainer_class = VanillaLSTMTrainer
        parameters = model.parameters()

    optimizer = optim.AdamW(parameters, lr=args.lstm_lr, weight_decay=args.lstm_weight_decay)
    scheduler = (optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
                 if args.lr_scheduler == 'step' else
                 warmup_cosine(optimizer, total_epochs=args.num_epochs,
                               warmup_epochs=args.lr_warmup_epochs, min_lr=args.lr_min))
    loss_func = (HorizonWeightedMSE(args.lstm_horizon_weights)
                 if args.lstm_horizon_weights is not None else nn.MSELoss())
    if args.lstm_tail_penalty > 0:
        loss_func = HorizonWeightedTailLoss(
            args.lstm_horizon_weights or [1.0] * output_size,
            [1.]*output_size if regional else train_loader.dataset.target_scaler.scale_, strength=args.lstm_tail_penalty,
            threshold_mm=args.lstm_tail_threshold_mm, transition_mm=args.lstm_tail_transition_mm)
    if args.change_loss != 'baseline':
        loss_func = from_training_dataset(train_loader.dataset,
            weights=args.lstm_horizon_weights or [1.]*output_size, mode=args.change_loss,
            scale_floor_mm=args.change_scale_floor_mm, event_weight=args.change_event_weight,
            under_weight=args.change_under_weight, threshold_mm=args.change_threshold_mm,
            large_error_penalty=args.large_error_penalty,
            large_error_threshold_mm=args.large_error_threshold_mm,
            large_error_transition_mm=args.large_error_transition_mm, rise_event_weight=args.rise_event_weight)
    trainer_options = dict(
        model=model,
        horizon=args.horizon,
        num_epochs=args.num_epochs,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        test_dataloader=test_loader,
        loss_func=loss_func,
        checkpoint_metric=args.lstm_checkpoint_metric,
        validation_threshold_mm=args.lstm_tail_threshold_mm,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping=args.early_stopping,
        gradient_accumulation=args.gradient_accumulation,
        device=args.device,
        skip_test=args.skip_test,
    )
    checkpoint_root = args.checkpoint_dir
    if checkpoint_root is None:
        checkpoint_root = (Path("model") / args.dataset_dir.resolve().name
                           if args.dataset_dir is not None else Path("model"))
    if regional or args.train_method in (*NEW_METHODS, 'lstm'):
        checkpoint_root /= args.train_method
    if args.num_rain_station is not None or args.rain_stations is not None:
        checkpoint_root /= "rain_" + "-".join(sorted(train_loader.dataset.rain_station_ids))
    if regional:
        checkpoint_root /= f"seq{train_loader.dataset.seq_len}"
    if args.head_mode != 'shared':
        head_folder = f'head_{args.head_mode}'
        if args.head_mode == 'station-mlp' and args.station_mlp_hidden_size != 32:
            head_folder += f'_width{args.station_mlp_hidden_size}'
        checkpoint_root /= head_folder
    if args.prediction_mode != 'residual':
        checkpoint_root /= f'prediction_{args.prediction_mode}'
    if args.event_aware:
        checkpoint_root /= f'event_aware_alpha{args.rise_classification_weight:g}_pos{args.rise_positive_weight:g}'
    if args.rise_event_weight:
        checkpoint_root /= f'rise_weight{args.rise_event_weight:g}'
    if args.horizon_decoder != 'none':
        checkpoint_root /= f'decoder_{args.horizon_decoder}'
    if args.outputs is not None:
        checkpoint_root /= f"o{args.outputs}"
    if args.target_horizons is not None:
        checkpoint_root /= "targets_" + "-".join(str(h) for h in sorted(args.target_horizons)) + "h"
    if args.lstm_tail_penalty > 0 or args.lstm_checkpoint_metric != "val_loss":
        checkpoint_root /= (f"tail_{args.lstm_tail_penalty:g}_threshold_{args.lstm_tail_threshold_mm:g}"
                            f"_transition_{args.lstm_tail_transition_mm:g}_{args.lstm_checkpoint_metric}")
    if args.change_loss != 'baseline':
        checkpoint_root /= f'change_{args.change_loss}_floor{args.change_scale_floor_mm:g}_event{args.change_event_weight:g}_under{args.change_under_weight:g}_threshold{args.change_threshold_mm:g}'
        if args.change_loss in ('large-error-count', 'large-error-primary'):
            checkpoint_root /= f'large_error_weight{args.large_error_penalty:g}_threshold{args.large_error_threshold_mm:g}_transition{args.large_error_transition_mm:g}'
    trainer_options["checkpoint_dir"] = checkpoint_root
    if args.diagnose_change_dynamics:
        from src.ai.change_dynamics import ChangeDynamics
        trainer_options['dynamics'] = ChangeDynamics(every=args.dynamics_every,
            gradients=args.diagnose_station_gradients, probe_samples=args.gradient_probe_samples)
    if trainer_class is ResidualLSTMTrainer:
        return ResidualLSTMTrainer(**trainer_options, correction_penalty=args.lstm_correction_penalty,
            rise_classification_weight=args.rise_classification_weight if args.event_aware else 0.,
            rise_positive_weight=args.rise_positive_weight)
    if trainer_class is MEFLSTMAdapterTrainer:
        return MEFLSTMAdapterTrainer(
            **trainer_options, two_stage=args.mef_two_stage,
            finetune_epochs=args.mef_finetune_epochs,
            backbone_lr=args.mef_backbone_lr, adapter_lr=args.mef_adapter_lr,
            finetune_backbone=args.mef_finetune_backbone,
        )
    return trainer_class(**trainer_options)


def main() -> None:
    args = parser.parse_args()
    validate_args(args)
    set_seed(args.seed)
    print(f"Seed: {args.seed}")
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_ex_name)
    print(f"MLflow experiment: {args.mlflow_ex_name}")
    if args.train_method in ("xgb", "lgbm"):
        train_tabular(args)
    else:
        trainer = build_sequence_trainer(args)
        dataset = trainer.train_dataloader.dataset
        run_name = f"{args.train_method}_{args.horizon}_{args.feature_set}"
        if args.num_rain_station is not None or args.rain_stations is not None:
            run_name += "_rain_" + "-".join(sorted(dataset.rain_station_ids))
        if args.target_horizons is not None:
            run_name += "_targets_" + "-".join(str(h) for h in sorted(args.target_horizons)) + "h"
        if args.train_method == "lstm_adapter":
            backbone_mode = "two_stage" if args.mef_two_stage else (
                "unfrozen" if args.mef_unfreeze_backbone else "frozen"
            )
            run_name += f"_{backbone_mode}"

        with mlflow.start_run(run_name=run_name) as active_run:
            log_dataset_source(args)
            mlflow.log_params(
                {
                    "train_method": args.train_method,
                    "horizon": args.horizon,
                    "feature_set": args.feature_set,
                    "rain_station_ids": dataset.rain_station_ids,
                    "num_rain_station": len(dataset.rain_station_ids),
                    "rain_station_selection": ("explicit_combination" if args.rain_stations is not None
                                               else "train_correlation_top_n" if args.num_rain_station is not None
                                               else "all_available"),
                    "rain_station_combination": args.rain_stations or "",
                    "target_horizons_hours": args.target_horizons or "dataset",
                    "checkpoint_path": str(trainer.checkpoint_path),
                    "change_loss": args.change_loss,
                    "change_scale_floor_mm": args.change_scale_floor_mm,
                    "change_event_weight": args.change_event_weight,
                    "change_under_weight": args.change_under_weight,
                    "change_threshold_mm": args.change_threshold_mm,
                    "large_error_penalty": args.large_error_penalty if args.change_loss in ('large-error-count', 'large-error-primary') else 0.,
                    "large_error_threshold_mm": args.large_error_threshold_mm,
                    "large_error_transition_mm": args.large_error_transition_mm,
                    "seed": args.seed,
                    "batch_size": args.batch_size,
                    "num_epochs": args.num_epochs,
                    "early_stopping": args.early_stopping,
                    "gradient_accumulation": args.gradient_accumulation,
                    "sequence_length": dataset.seq_len,
                    "input_size": dataset.num_features,
                    "output_size": dataset.num_targets,
                    "learning_rate": trainer.optimizer.defaults["lr"],
                    "lr_scheduler": args.lr_scheduler,
                    "lr_warmup_epochs": args.lr_warmup_epochs if args.lr_scheduler == 'warmup-cosine' else 0,
                    "lr_min": args.lr_min if args.lr_scheduler == 'warmup-cosine' else 0.,
                    "weight_decay": trainer.optimizer.defaults["weight_decay"],
                    "device": str(trainer.device),
                    "loss_function": type(trainer.loss_func).__name__,
                    "horizon_weights": args.lstm_horizon_weights or [1.0] * dataset.num_targets,
                    "tail_penalty": args.lstm_tail_penalty,
                    "tail_threshold_mm": args.lstm_tail_threshold_mm,
                    "tail_transition_mm": args.lstm_tail_transition_mm,
                    "checkpoint_metric": args.lstm_checkpoint_metric,
                    "time_features": args.time_features,
                    "head_mode": args.head_mode,
                    "prediction_mode": args.prediction_mode,
                    "event_aware": args.event_aware,
                    "rise_classification_weight": args.rise_classification_weight if args.event_aware else 0.,
                    "rise_positive_weight": args.rise_positive_weight,
                    "rise_event_weight": args.rise_event_weight,
                    "horizon_decoder": args.horizon_decoder,
                    "diagnose_change_dynamics": args.diagnose_change_dynamics,
                    "dynamics_every": args.dynamics_every,
                    "diagnose_station_gradients": args.diagnose_station_gradients,
                    "gradient_probe_samples": args.gradient_probe_samples,
                    "skip_test": args.skip_test,
                    "change_event_weight_effective": args.change_event_weight if args.change_loss in ('event-weighted','combined') else 0.,
                    "change_under_weight_effective": args.change_under_weight if args.change_loss in ('asymmetric','combined') else 0.,
                    "target_columns": dataset.target_cols,
                    **trainer.model_settings()
                }
            )
            mlflow.log_dict({"feature_columns": dataset.feature_cols}, "features.json")
            if dataset.rain_station_ranking:
                mlflow.log_dict({"scope": "train", "rain_window_hours": 6,
                                 "water_change_horizon_hours": 12,
                                 "stations": dataset.rain_station_ranking}, "rain_station_ranking.json")
            trainer.train()
            if args.skip_test:
                mlflow.set_tag('comparison/status', 'skipped_with_test')
                return
            if is_regional_dataset(args):
                mlflow.set_tag('comparison/status', 'regional_evaluation_in_test_metrics')
                mlflow.set_tag('evaluation/scope', 'post_cutoff_exploratory')
                return
            comparison = run_post_training_comparison(
                trainer, dataset_directory(args.horizon, args.dataset_dir) / "test.csv",
                active_run.info.run_id, station_id=args.comparison_station_id,
                registry_path=args.comparison_registry,
                model_id=args.comparison_model_id,
            )
            mlflow.set_tags({"comparison/status": comparison["status"],
                             "comparison/reference_model_id": args.comparison_model_id,
                             "comparison/report_dir": str(comparison["output_dir"])})
            if comparison["status"] == "completed":
                mlflow.log_metrics(mlflow_comparison_metrics(comparison))


if __name__ == "__main__":
    main()
