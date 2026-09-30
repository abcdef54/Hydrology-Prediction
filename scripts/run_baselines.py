"""Run leakage-aware temporal baselines for +1h, +3h, and +6h targets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analysis_config import AnalysisPaths, FORECAST_HORIZONS_MINUTES, RANDOM_SEED
from analysis_utils import contiguous_events, find_training_table, metric_bundle, parse_utc, read_csv


EXCLUDED_FEATURES = {"timestamp_utc"} | {
    f"target_water_level_plus_{minutes}m" for minutes in FORECAST_HORIZONS_MINUTES
}


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    local_time = frame["timestamp_utc"].dt.tz_convert("Asia/Bangkok")
    enhanced = frame.copy()
    enhanced["hour_sin"] = np.sin(2 * np.pi * local_time.dt.hour / 24)
    enhanced["hour_cos"] = np.cos(2 * np.pi * local_time.dt.hour / 24)
    enhanced["day_of_year_sin"] = np.sin(2 * np.pi * local_time.dt.dayofyear / 365.25)
    enhanced["day_of_year_cos"] = np.cos(2 * np.pi * local_time.dt.dayofyear / 365.25)
    return enhanced


def add_all_rain_station_features(
    paths: AnalysisPaths,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Add issue-time and past-only features for every available rain station."""
    observations = read_csv(paths.data / "rain_observations.csv")
    observations["timestamp_utc"] = parse_utc(observations["time_point"]).dt.floor("10min")
    rain_matrix = observations.pivot_table(
        index="timestamp_utc",
        columns="station_id",
        values="depth",
        aggfunc="mean",
    ).sort_index()
    rain_matrix = rain_matrix.reindex(pd.DatetimeIndex(frame["timestamp_utc"]))
    rain_matrix.index = frame.index

    feature_blocks = {
        "current": rain_matrix,
        "lag_1h": rain_matrix.shift(6),
        "lag_3h": rain_matrix.shift(18),
        "sum_past_6h": rain_matrix.shift(1).rolling(36, min_periods=36).sum(),
    }
    enhanced = frame.copy()
    for feature_name, feature_values in feature_blocks.items():
        feature_values.columns = [
            f"rain_station_{station_id}_{feature_name}"
            for station_id in feature_values.columns
        ]
        enhanced = pd.concat([enhanced, feature_values], axis=1)
    return enhanced


def temporal_partitions(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end = int(len(frame) * 0.70)
    validation_end = int(len(frame) * 0.85)
    return frame.iloc[:train_end], frame.iloc[train_end:validation_end], frame.iloc[validation_end:]


def ridge_model() -> TransformedTargetRegressor:
    regressor = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=10.0)),
        ]
    )
    return TransformedTargetRegressor(regressor=regressor, transformer=StandardScaler())


def gradient_boosting_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=0.06,
        max_iter=250,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=RANDOM_SEED,
    )


def peak_metrics(test: pd.DataFrame, observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    if not valid.any():
        return {"peak_absolute_error": None, "peak_timing_error_minutes": None}
    timestamps = test.loc[valid, "timestamp_utc"].reset_index(drop=True)
    actual = observed[valid]
    forecast = predicted[valid]
    actual_peak_index = int(np.argmax(actual))
    predicted_peak_index = int(np.argmax(forecast))
    timing_error = abs((timestamps.iloc[predicted_peak_index] - timestamps.iloc[actual_peak_index]).total_seconds() / 60)
    return {
        "peak_absolute_error": float(abs(forecast[predicted_peak_index] - actual[actual_peak_index])),
        "peak_timing_error_minutes": float(timing_error),
    }


def event_counts_by_split(frame: pd.DataFrame, threshold: float) -> dict[str, int]:
    counts = {}
    for name, partition in zip(("train", "validation", "test"), temporal_partitions(frame)):
        events = contiguous_events(
            partition["timestamp_utc"], partition["water_level_depth"], threshold, pd.Timedelta(hours=6)
        )
        counts[name] = len(events)
    return counts


def evaluate_model(
    model_name: str,
    test: pd.DataFrame,
    observed: np.ndarray,
    predicted: np.ndarray,
    high_water_threshold: float,
) -> list[dict[str, object]]:
    rows = []
    for subset_name, subset_mask in (
        ("all_test", np.ones(len(test), dtype=bool)),
        ("high_water_test", test["water_level_depth"].to_numpy() > high_water_threshold),
    ):
        metrics = metric_bundle(observed[subset_mask], predicted[subset_mask])
        rows.append({"model": model_name, "subset": subset_name, **metrics})
    rows[0].update(peak_metrics(test, observed, predicted))
    return rows


def run(paths: AnalysisPaths) -> None:
    paths.create_output_directories()
    frame = read_csv(find_training_table(paths.data, paths.reports))
    frame["timestamp_utc"] = parse_utc(frame["timestamp_utc"])
    frame = add_time_features(frame)
    frame = frame.sort_values("timestamp_utc").reset_index(drop=True)
    frame = add_all_rain_station_features(paths, frame)
    train, validation, test = temporal_partitions(frame)
    all_feature_columns = [
        column for column in frame.select_dtypes(include=[np.number]).columns
        if column not in EXCLUDED_FEATURES
    ]
    selected_station_feature_columns = [
        column
        for column in all_feature_columns
        if not column.startswith("rain_station_")
    ]
    split_rows = []
    metric_rows = []
    prediction_rows = []
    for horizon_minutes in FORECAST_HORIZONS_MINUTES:
        target_column = f"target_water_level_plus_{horizon_minutes}m"
        horizon = pd.Timedelta(minutes=horizon_minutes)
        train_threshold = float(train["water_level_depth"].quantile(0.90))
        event_counts = event_counts_by_split(frame, train_threshold)
        for split_name, partition in zip(("train", "validation", "test"), (train, validation, test)):
            split_rows.append(
                {
                    "horizon_minutes": horizon_minutes,
                    "split": split_name,
                    "start": partition["timestamp_utc"].min(),
                    "end": partition["timestamp_utc"].max(),
                    "rows": len(partition),
                    "valid_targets": int(partition[target_column].notna().sum()),
                    "boundary_purge_minutes": horizon_minutes if split_name in {"train", "validation"} else 0,
                    "high_water_events_using_train_p90": event_counts[split_name],
                }
            )
        train_valid = train[target_column].notna() & (
            train["timestamp_utc"] + horizon <= train["timestamp_utc"].max()
        )
        test_valid = test[target_column].notna()
        y_train = train.loc[train_valid, target_column].to_numpy()
        test_evaluation = test.loc[test_valid].reset_index(drop=True)
        observed = test_evaluation[target_column].to_numpy()
        baseline_predictions = {
            "persistence": test_evaluation["water_level_depth"].to_numpy(),
            "seasonal_24h": test_evaluation["water_level_depth"].shift(144).to_numpy(),
        }
        model_variants = (
            ("ridge_selected_stations", ridge_model(), selected_station_feature_columns),
            (
                "hist_gradient_boosting_selected_stations",
                gradient_boosting_model(),
                selected_station_feature_columns,
            ),
            ("ridge_all_rain_stations", ridge_model(), all_feature_columns),
            (
                "hist_gradient_boosting_all_rain_stations",
                gradient_boosting_model(),
                all_feature_columns,
            ),
        )
        for model_name, model, model_feature_columns in model_variants:
            x_train = train.loc[train_valid, model_feature_columns]
            x_test = test.loc[test_valid, model_feature_columns]
            model.fit(x_train, y_train)
            baseline_predictions[model_name] = model.predict(x_test)
        for model_name, predicted in baseline_predictions.items():
            for metric_row in evaluate_model(model_name, test_evaluation, observed, predicted, train_threshold):
                metric_rows.append({"horizon_minutes": horizon_minutes, **metric_row})
            for timestamp, actual, prediction in zip(test_evaluation["timestamp_utc"], observed, predicted):
                prediction_rows.append(
                    {
                        "timestamp_utc": timestamp,
                        "horizon_minutes": horizon_minutes,
                        "model": model_name,
                        "observed": actual,
                        "predicted": prediction,
                    }
                )
    pd.DataFrame(split_rows).to_csv(paths.reports / "temporal_split_summary.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(paths.reports / "baseline_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(paths.reports / "baseline_predictions.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=AnalysisPaths().data)
    parser.add_argument("--reports-dir", type=Path, default=AnalysisPaths().reports)
    parser.add_argument("--figures-dir", type=Path, default=AnalysisPaths().figures)
    args = parser.parse_args()
    run(AnalysisPaths(args.data_dir, args.reports_dir, args.figures_dir))


if __name__ == "__main__":
    main()
