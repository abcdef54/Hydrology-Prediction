"""Compare a local candidate with the app registry on identical hourly test samples.

No fitting, model uploads, registry changes or database access. Each model uses
its own feature order/scalers and the intersection of eligible sequence anchors.
"""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from src.ai.dataset import HydrologyDataset
from src.ai.lstm import LSTM, ResidualLSTM


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = ROOT / "src/backend/prediction/model_configs.json"
DEFAULT_OUTPUT = ROOT / "reports/model_comparisons"


def evaluation_dataset(frame, saved):
    return HydrologyDataset(
        frame, feature_cols=saved["feature_columns"], target_cols=saved["target_columns"],
        feature_scaler=saved["feature_scaler"], target_scaler=saved["target_scaler"],
        seq_len=saved["sequence_length"], require_current_level=True,
    )


def comparison_observations_source(test_csv):
    """Follow dataset ancestry to the explicit serving-observation source.

    Derived datasets can add/remove predictors without carrying the original
    observation columns. Never fill absent serving inputs with zero or refit
    its scalers; use its original snapshot and validate timestamps/labels.
    """
    test_csv = Path(test_csv)
    directory = test_csv.parent.resolve()
    visited = set()
    while (directory / "manifest.json").is_file():
        if directory in visited:
            raise ValueError("Cycle in dataset parent metadata")
        visited.add(directory)
        spec = json.loads((directory / "manifest.json").read_text())
        observations_dir = spec.get("comparison_observations_dir")
        if observations_dir:
            path = (directory / observations_dir / test_csv.name).resolve()
            expected = spec.get("source_split_sha256", {}).get(test_csv.name)
            return path, expected
        parent = spec.get("parent_dataset")
        if not parent:
            break
        directory = (directory / parent).resolve()
    return None, None


def predict(model, dataset, indices, device, batch_size):
    positions = np.searchsorted(dataset.end_indices, indices)
    loader = DataLoader(Subset(dataset, positions.tolist()), batch_size=batch_size,
                        shuffle=False, drop_last=False)
    model.eval()
    predictions = []
    with torch.inference_mode():
        for inputs, _ in loader:
            predictions.append(model(inputs.to(device)).cpu().numpy())
    scaled = np.concatenate(predictions)
    if scaled.shape != (len(indices), dataset.num_targets) or not np.isfinite(scaled).all():
        raise ValueError("Invalid prediction shape or non-finite model predictions")
    return dataset.target_scaler.inverse_transform(scaled)


def paired_metrics(frame, candidate, candidate_saved, serving, serving_saved,
                   device="cpu", batch_size=128, serving_frame=None):
    """Return per-horizon metrics, aggregate summaries, paired predictions and coverage."""
    timestamps = pd.to_datetime(frame.timestamp_utc, utc=True)
    steps = timestamps.diff().dropna()
    if (not timestamps.is_unique or not timestamps.is_monotonic_increasing
            or steps.empty or steps.nunique() != 1
            or steps.iloc[0] not in [pd.Timedelta(hours=h) for h in (1, 12, 24)]):
        raise ValueError("Comparison requires an ordered, unique, complete hourly grid or regular 12h/24h grid")
    common_targets = [c for c in serving_saved["target_columns"] if c in candidate_saved["target_columns"]]
    if not common_targets:
        raise ValueError("Candidate and serving model have no shared horizons")
    if any(not re.fullmatch(r"target_water_level_plus_\d+m", c) for c in common_targets):
        raise ValueError("Unsupported target naming; expected water-level horizons in minutes")
    new_ds = evaluation_dataset(frame, candidate_saved)
    serving_frame = frame if serving_frame is None else serving_frame
    serving_times = pd.DatetimeIndex(pd.to_datetime(serving_frame.timestamp_utc, utc=True))
    serving_steps = serving_times.to_series().diff().dropna()
    if (not serving_times.is_unique or not serving_times.is_monotonic_increasing
            or serving_steps.empty or serving_steps.nunique() != 1):
        raise ValueError("Serving observations need a regular ordered time grid")
    source_positions = serving_times.get_indexer(timestamps)
    if (source_positions < 0).any():
        raise ValueError("Serving observations must contain all candidate test timestamps")
    for target in common_targets:
        if not np.allclose(frame[target].to_numpy(), serving_frame[target].iloc[source_positions].to_numpy(),
                           rtol=0, atol=1e-9, equal_nan=True):
            raise ValueError("Candidate and serving observations have different target values")
    app_ds = evaluation_dataset(serving_frame, serving_saved)
    candidate_times = pd.DatetimeIndex(timestamps)
    shared_times = candidate_times[new_ds.end_indices].intersection(serving_times[app_ds.end_indices]).sort_values()
    indices = candidate_times.get_indexer(shared_times)
    serving_indices = serving_times.get_indexer(shared_times)
    if not len(indices):
        raise ValueError("No common eligible test anchors after sequence warmup/target filtering")
    new_values = predict(candidate, new_ds, indices, device, batch_size)
    app_values = predict(serving, app_ds, serving_indices, device, batch_size)
    frames, rows = [], []
    for target in common_targets:
        minutes = int(target.rsplit("_", 1)[-1][:-1])
        actual = frame[target].to_numpy(dtype=np.float64)[indices]
        part = pd.DataFrame({
            "anchor_utc": timestamps.iloc[indices].to_numpy(),
            "target_utc": (timestamps.iloc[indices] + pd.Timedelta(minutes=minutes)).to_numpy(),
            "horizon_minutes": minutes, "actual_mm": actual,
            "candidate_mm": new_values[:, candidate_saved["target_columns"].index(target)],
            "serving_mm": app_values[:, serving_saved["target_columns"].index(target)],
        })
        for name in ("candidate", "serving"):
            error = part[f"{name}_mm"].to_numpy(dtype=np.float64) - actual
            if not np.isfinite(error).all():
                raise ValueError("Non-finite paired evaluation values")
            part[f"{name}_error_mm"] = error
            rows.append(dict(horizon_minutes=minutes, horizon_hours=minutes / 60,
                             model=name, n=len(indices), mae_mm=float(np.abs(error).mean()),
                             rmse_mm=float(np.sqrt(np.mean(error ** 2))),
                             above_200=int((np.abs(error) > 200).sum()),
                             above_200_pct=float((np.abs(error) > 200).mean() * 100)))
        frames.append(part)
    metrics = pd.DataFrame(rows)
    summaries = {}
    points = pd.concat(frames, ignore_index=True)
    for name in ("candidate", "serving"):
        group = metrics[metrics.model.eq(name)]
        errors = points[f"{name}_error_mm"].to_numpy()
        summaries[name] = dict(
            mean_horizon_rmse_mm=float(group.rmse_mm.mean()),
            pooled_rmse_mm=float(np.sqrt(np.mean(errors ** 2))),
            mae_mm=float(np.abs(errors).mean()), n_predictions=len(errors),
            above_200=int((np.abs(errors) > 200).sum()),
            above_200_pct=float((np.abs(errors) > 200).mean() * 100),
        )
    delta_rmse = summaries["candidate"]["mean_horizon_rmse_mm"] - summaries["serving"]["mean_horizon_rmse_mm"]
    delta_tail = summaries["candidate"]["above_200_pct"] - summaries["serving"]["above_200_pct"]
    if delta_rmse <= 0 and delta_tail <= 0 and (delta_rmse < 0 or delta_tail < 0):
        assessment = "Model mới tốt hơn theo tổng hợp RMSE và tỷ lệ lỗi lớn; vẫn cần xem từng horizon."
    elif delta_rmse >= 0 and delta_tail >= 0 and (delta_rmse > 0 or delta_tail > 0):
        assessment = "Model mới kém hơn theo tổng hợp RMSE và tỷ lệ lỗi lớn; vẫn cần xem từng horizon."
    elif delta_rmse == 0 and delta_tail == 0:
        assessment = "Hai model bằng nhau theo hai chỉ số tổng hợp."
    else:
        assessment = "Kết quả đánh đổi: một chỉ số tổng hợp tốt hơn, chỉ số kia kém hơn."
    summaries.update(delta_mean_rmse_mm=delta_rmse, delta_above_200_percentage_points=delta_tail,
                     assessment=assessment, automatic_promotion=False)
    coverage = dict(total_test_rows=len(frame), candidate_eligible=len(new_ds),
                    serving_eligible=len(app_ds), common_anchors=len(indices),
                    candidate_step_hours=steps.iloc[0].total_seconds() / 3600,
                    serving_step_hours=serving_steps.iloc[0].total_seconds() / 3600,
                    first_anchor_vietnam=str(timestamps.iloc[indices[0]].tz_convert("Asia/Ho_Chi_Minh")),
                    last_anchor_vietnam=str(timestamps.iloc[indices[-1]].tz_convert("Asia/Ho_Chi_Minh")),
                    common_targets=common_targets,
                    candidate_only_targets=[c for c in candidate_saved["target_columns"] if c not in common_targets],
                    serving_only_targets=[c for c in serving_saved["target_columns"] if c not in common_targets])
    return metrics, summaries, points, coverage


def load_candidate(checkpoint_bytes, saved, device):
    """Reconstruct existing LSTM/ResidualLSTM architecture from state for standalone use."""
    state = torch.load(BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    layers = [k for k in state if re.fullmatch(r"lstm.weight_ih_l\d+", k)]
    options = dict(input_size=len(saved["feature_columns"]), output_size=len(saved["target_columns"]),
                   hidden_size=state["lstm.weight_hh_l0"].shape[1], num_layers=len(layers), dropout=0.)
    direct_architecture,residual_architecture=LSTM,ResidualLSTM
    if 'attention_score.weight' in state:
        from src.ai.models import AttentionLSTM1,AttentionLSTM2,AttentionLSTM3
        direct_architecture=AttentionLSTM1
        residual_architecture=AttentionLSTM2
        if 'horizon_heads.0.0.weight' in state:
            residual_architecture=AttentionLSTM3
            options['head_dim']=state['horizon_heads.0.0.weight'].shape[0]
    if "persistence_scale" in state:
        index = saved["feature_columns"].index("water_level_depth")
        fs, ts = saved["feature_scaler"], saved["target_scaler"]
        history_policy = saved.get('forecast_history_policy', 'all_timesteps')
        options['forecast_history_policy'] = history_policy
        if history_policy == 'anchor_only':
            weather_indices = [i for i, c in enumerate(saved['feature_columns'])
                               if c.startswith('weather_forecast_')]
            mask_indices = [i for i in weather_indices
                            if saved['feature_columns'][i].endswith('_missing')]
            options.update(weather_feature_indices=weather_indices,
                           weather_mask_indices=mask_indices,
                           weather_mask_values=((1. - fs.mean_[mask_indices]) / fs.scale_[mask_indices]).tolist())
        model = residual_architecture(**options, level_feature_index=index,
                             persistence_scale=(fs.scale_[index] / ts.scale_).tolist(),
                             persistence_offset=((fs.mean_[index] - ts.mean_) / ts.scale_).tolist())
        for key in ("persistence_scale", "persistence_offset"):
            torch.testing.assert_close(state[key], getattr(model, key))
    else:
        model = direct_architecture(**options)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def write_report(out, metrics, summary, coverage):
    lines = ["# So sánh model mới với model cấu hình trong app", "",
             f"Mốc đầu vào: **{coverage['first_anchor_vietnam']} → {coverage['last_anchor_vietnam']}**.",
             f"Số mốc đầu vào chung: **{coverage['common_anchors']}**. Đơn vị mực nước/sai số: **mm**.", "",
             "| Horizon | Số mẫu | RMSE app | RMSE mới | Mới − app | >200 mm app | >200 mm mới |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for minutes, group in metrics.groupby("horizon_minutes", sort=True):
        group = group.set_index("model")
        app, new = group.loc["serving"], group.loc["candidate"]
        lines.append(f"| +{minutes / 60:g} giờ | {int(new.n)} | {app.rmse_mm:.2f} | {new.rmse_mm:.2f} | {new.rmse_mm-app.rmse_mm:+.2f} | {int(app.above_200)} ({app.above_200_pct:.2f}%) | {int(new.above_200)} ({new.above_200_pct:.2f}%) |")
    lines += ["", "## Tổng hợp", "",
              "| Model | Trung bình RMSE các horizon | RMSE gộp mọi dự báo | MAE | Tỷ lệ >200 mm |",
              "|---|---:|---:|---:|---:|"]
    for name, label in (("serving", "App"), ("candidate", "Mới")):
        row = summary[name]
        lines.append(f"| {label} | {row['mean_horizon_rmse_mm']:.2f} | {row['pooled_rmse_mm']:.2f} | {row['mae_mm']:.2f} | {row['above_200_pct']:.2f}% |")
    lines += ["", summary["assessment"], "",
              "Trung bình RMSE = trung bình cộng RMSE của các horizon chung; RMSE gộp = căn trung bình bình phương tất cả sai số. Đây là hai phép tính khác nhau.",
              "Ngưỡng lỗi lớn là |dự báo − thực tế| >200 mm, không bao gồm đúng 200 mm. Tỷ lệ tính trên số dự báo có thực tế tại từng horizon; một anchor có nhiều horizon là nhiều dự báo.",
              "Mỗi model dùng đúng scaler/feature order riêng, không fit lại. So sánh trên giao các mẫu hợp lệ sau warmup và lọc nhãn; báo cáo độ phủ trong manifest.json. Actual lấy trực tiếp từ cùng test.csv.",
              "Model app được đọc từ registry local lúc chạy đánh giá, có model version và checksum trong manifest; không truy vấn model đang nằm trong bộ nhớ của tiến trình API.",
              "Các model có thể khác dữ liệu train/loss; so sánh này không cô lập tác động của riêng số trạm. Test đã dùng để xem nhiều lần là đánh giá khám phá; không tự động chọn hoặc thay model phục vụ.",
              "Horizon chỉ có ở một model không được trộn vào chỉ số so sánh; xem danh sách trong manifest. Các phép tổng hợp chỉ dùng horizon chung.", "",
              "[Metric CSV](metrics.csv) · [Dự báo từng mẫu](paired_predictions.csv) · [Metadata](manifest.json)", ""]
    (out / "comparison.md").write_text("\n".join(lines))


def compare_checkpoint(checkpoint, test_csv, out, station_id="0074647403",
                       registry_path=DEFAULT_REGISTRY, device="cpu", batch_size=128, candidate=None,
                       candidate_run_id=None):
    """Save an immutable comparison snapshot; candidate may be the just-trained model."""
    from src.backend.prediction.config import StationModelRegistry
    from src.backend.prediction.inference import ResidualLSTMRuntime

    checkpoint, test_csv, out = Path(checkpoint), Path(test_csv), Path(out)
    if out.exists():
        raise FileExistsError("Comparison output must be a new directory")
    model_bytes = checkpoint.read_bytes()
    preprocessing_path = checkpoint.with_suffix(".preprocessing.joblib")
    preprocessing_bytes = preprocessing_path.read_bytes()
    saved = joblib.load(BytesIO(preprocessing_bytes))
    test_bytes = test_csv.read_bytes()
    frame = pd.read_csv(BytesIO(test_bytes))
    registry_bytes = Path(registry_path).read_bytes()
    # Load the same captured registry bytes used for its checksum below.
    entries = json.loads(registry_bytes)
    match = [e for e in entries if e["station_id"] == station_id]
    if len(match) != 1:
        raise ValueError(f"Expected exactly one app model for station {station_id}")
    config = StationModelRegistry.from_json(registry_path, artifact_root=ROOT).get(station_id)
    if Path(registry_path).read_bytes() != registry_bytes:
        raise ValueError("Registry changed while comparison was loading")
    runtime = ResidualLSTMRuntime(config, device=str(device))
    serving_saved = dict(feature_columns=list(config.feature_columns),
                         target_columns=[f"target_water_level_plus_{h}m" for h in config.forecast_horizons_minutes],
                         sequence_length=config.sequence_length,
                         feature_scaler=runtime.preprocessor.feature_scaler, target_scaler=runtime.target_scaler)
    if candidate is None:
        candidate = load_candidate(model_bytes, saved, device)
    else:
        state = torch.load(BytesIO(model_bytes), map_location="cpu", weights_only=True)
        actual_state = candidate.state_dict()
        if state.keys() != actual_state.keys() or any(not torch.equal(v, actual_state[k].detach().cpu()) for k, v in state.items()):
            raise ValueError("Local checkpoint differs from just-trained model; refusing misleading provenance")
    serving_frame = None
    observations_sha = None
    observations_file, expected = comparison_observations_source(test_csv)
    if observations_file is not None:
        observations_bytes = observations_file.read_bytes()
        observations_sha = sha256(observations_bytes).hexdigest()
        if expected and observations_sha != expected:
            raise ValueError("Original serving observations changed since dataset construction")
        serving_frame = pd.read_csv(BytesIO(observations_bytes))
    metrics, summary, points, coverage = paired_metrics(
        frame, candidate, saved, runtime.model, serving_saved, device, batch_size,
        serving_frame=serving_frame)
    coverage["serving_observations_sha256"] = observations_sha
    out.mkdir(parents=True)
    # Keep the evaluated candidate even if the next trial overwrites its checkpoint.
    (out / "candidate.pt").write_bytes(model_bytes)
    (out / "candidate.preprocessing.joblib").write_bytes(preprocessing_bytes)
    metrics.to_csv(out / "metrics.csv", index=False)
    points.to_csv(out / "paired_predictions.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    manifest = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), station_id=station_id,
                    candidate_run_id=candidate_run_id,
                    candidate_checkpoint=str(checkpoint.resolve()), candidate_sha256=sha256(model_bytes).hexdigest(),
                    candidate_preprocessing_sha256=sha256(preprocessing_bytes).hexdigest(),
                    serving_model_version=config.model_version, serving_checkpoint=str(config.model_path),
                    serving_sha256=runtime.artifact_hashes[0], serving_preprocessing_sha256=runtime.artifact_hashes[1],
                    registry_sha256=sha256(registry_bytes).hexdigest(), test_csv=str(test_csv.resolve()),
                    test_csv_sha256=sha256(test_bytes).hexdigest(), device=str(device), coverage=coverage,
                    candidate_feature_columns=saved["feature_columns"], serving_feature_columns=serving_saved["feature_columns"],
                    candidate_sequence_length=saved["sequence_length"], serving_sequence_length=config.sequence_length,
                    threshold_mm=200, threshold_comparison="strict greater than", automatic_promotion=False)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    write_report(out, metrics, summary, coverage)
    print("\nSo sánh model mới với app (cùng mẫu, đơn vị mm):")
    print(metrics.round(3).to_string(index=False))
    print(summary["assessment"])
    print(f"Báo cáo: {out / 'comparison.md'}")
    return dict(status="completed", output_dir=out, metrics=metrics, summary=summary)


def run_post_training_comparison(trainer, test_csv, run_id, station_id="0074647403",
                                 registry_path=DEFAULT_REGISTRY, output_root=DEFAULT_OUTPUT):
    """A comparison failure is explicit but does not invalidate completed training."""
    out = Path(output_root) / f"{run_id}_{uuid4().hex[:8]}"
    try:
        return compare_checkpoint(trainer.checkpoint_path, test_csv, out, station_id,
                                  registry_path, trainer.device, candidate=trainer.model,
                                  candidate_run_id=run_id)
    except Exception as exc:
        out.mkdir(parents=True, exist_ok=True)
        error = dict(status="failed", error_type=type(exc).__name__, error=str(exc),
                     candidate_checkpoint=str(trainer.checkpoint_path), automatic_promotion=False)
        (out / "comparison_error.json").write_text(json.dumps(error, ensure_ascii=False, indent=2) + "\n")
        print(f"Đã train xong nhưng so sánh chưa hoàn tất: {type(exc).__name__}: {exc}")
        print(f"Chi tiết: {out / 'comparison_error.json'}")
        return dict(status="failed", output_dir=out, metrics=None, summary=None)


def mlflow_comparison_metrics(result):
    values = {}
    for row in result["metrics"].itertuples(index=False):
        prefix = f"comparison/{row.model}/h{row.horizon_minutes}m"
        values.update({f"{prefix}/rmse_mm": row.rmse_mm,
                       f"{prefix}/above_200_pct": row.above_200_pct,
                       f"{prefix}/above_200_count": row.above_200})
    for name in ("candidate", "serving"):
        values[f"comparison/{name}/mean_horizon_rmse_mm"] = result["summary"][name]["mean_horizon_rmse_mm"]
        values[f"comparison/{name}/above_200_pct"] = result["summary"][name]["above_200_pct"]
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--station-id", default="0074647403")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    compare_checkpoint(args.checkpoint, args.dataset_dir / "test.csv", args.output_dir,
                       args.station_id, args.registry, args.device)


if __name__ == "__main__":
    main()
