"""Local, three-model validation reports generated from paired predictions."""

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_NAMES = {"legacy": "Production cũ", "model_b": "Model B", "candidate": "Model mới"}


def markdown_table(frame):
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.2f}" if np.isfinite(value) else "—"
        return str(value).replace("|", "\\|")

    lines = ["| " + " | ".join(map(str, frame.columns)) + " |",
             "| " + " | ".join("---" for _ in frame.columns) + " |"]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |"
                 for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def write_comparison_report(directory, points, metadata):
    """Write local artifacts only; callers must supply the same anchors/truth."""
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"Refuse to overwrite report {directory}")
    if (points.empty or sorted(points.horizon_hours.unique()) != [1, 3, 6, 12, 24]
            or len(points) != points.anchor_utc.nunique() * 5 * 3
            or not np.isfinite(points[["actual_mm", "predicted_mm", "current_mm"]].to_numpy()).all()):
        raise ValueError("Require finite paired predictions on all five horizons")
    keys = ["anchor_utc", "horizon_hours"]
    if points.duplicated(keys + ["model"]).any():
        raise ValueError("Duplicate model/anchor/horizon comparison points")
    for _, group in points.groupby(keys):
        if set(group.model) != set(MODEL_NAMES):
            raise ValueError("Every anchor/horizon requires all three models")
        if group.actual_mm.nunique() != 1 or group.current_mm.nunique() != 1:
            raise ValueError("Models must share observed truth and current water level")
    rows = []
    for (model, horizon), group in points.groupby(["model", "horizon_hours"]):
        error = group.predicted_mm - group.actual_mm
        delta = group.predicted_mm - group.current_mm
        actual_delta = group.actual_mm - group.current_mm
        fast = actual_delta.abs() > 200
        rows.append(dict(
            model=model, horizon_hours=int(horizon), n=len(group),
            rmse_mm=float(np.sqrt(np.mean(error ** 2))), mae_mm=float(error.abs().mean()),
            n200=int(error.abs().gt(200).sum()), pct200=float(100 * error.abs().gt(200).mean()),
            bias_mm=float(error.mean()), mean_change_mm=float(delta.mean()),
            median_change_mm=float(delta.median()), min_change_mm=float(delta.min()),
            max_change_mm=float(delta.max()), positive_change_pct=float(100 * delta.gt(0).mean()),
            mean_abs_change_mm=float(delta.abs().mean()),
            actual_mean_abs_change_mm=float(actual_delta.abs().mean()), fast_n=int(fast.sum()),
            fast_pred_abs_mm=float(delta[fast].abs().mean()) if fast.any() else None,
            fast_actual_abs_mm=float(actual_delta[fast].abs().mean()) if fast.any() else None,
        ))
    metrics = pd.DataFrame(rows)
    summary = []
    for model in MODEL_NAMES:
        group = points[points.model.eq(model)]
        error = group.predicted_mm - group.actual_mm
        summary.append(dict(model=model, n=len(group), rmse_mm=float(np.sqrt(np.mean(error ** 2))),
                            mae_mm=float(error.abs().mean()), n200=int(error.abs().gt(200).sum()),
                            pct200=float(100 * error.abs().gt(200).mean())))
    summary = pd.DataFrame(summary).set_index("model")
    coverage = {
        "common_anchors": int(points.anchor_utc.nunique()),
        "first_anchor_vietnam": str(pd.Timestamp(points.anchor_utc.min()).tz_convert("Asia/Ho_Chi_Minh")),
        "last_anchor_vietnam": str(pd.Timestamp(points.anchor_utc.max()).tz_convert("Asia/Ho_Chi_Minh")),
        "horizon_hours": sorted(points.horizon_hours.unique().tolist()),
    }
    manifest = {**metadata, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "evaluation_scope": "validation_only", "independent_test": False,
                "coverage": coverage, "summary": summary.to_dict(orient="index")}
    directory.mkdir(parents=True)
    metrics.to_csv(directory / "metrics.csv", index=False)
    points.to_csv(directory / "predictions.csv.gz", index=False)
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    display = summary.reset_index().replace({"model": MODEL_NAMES}).rename(columns={
        "model": "Model", "n": "Số dự báo", "rmse_mm": "RMSE mm", "mae_mm": "MAE mm",
        "n200": "Lỗi >200 mm", "pct200": "Tỷ lệ %",
    })
    lines = ["# So sánh production cũ, Model B và model train mở rộng", "",
             f"Variant: **{metadata.get('variant', '—')}** · seed **{metadata.get('seed', '—')}** · "
             f"checkpoint epoch **{metadata.get('best_checkpoint_epoch') if metadata.get('best_checkpoint_epoch') is not None else 'không có metadata'}**.",
             f"Tiêu chí checkpoint: **{metadata.get('training_settings', {}).get('checkpoint_metric', 'không có metadata')}**.",
             f"Dataset: `{metadata['dataset_dir']}`.",
             f"Checkpoint mới: `{metadata['checkpoint_path']}`.", "",
             f"**{coverage['common_anchors']} anchor chung**, {coverage['first_anchor_vietnam']} → "
             f"{coverage['last_anchor_vietnam']}. Đơn vị mm; >200 dùng điều kiện nghiêm ngặt.", "",
             "Đây là **validation mới = test cũ**, đã dùng chọn checkpoint/early stopping và xem nhiều lần.",
             "Không còn test độc lập. Model mới được train thêm dữ liệu; so sánh này không cô lập tác dụng",
             "của sampler nếu đối chiếu trực tiếp với B cũ. Muốn đo sampler phải so uniform/event-balanced",
             "trên cùng train mở rộng và các seed cố định. Mỗi model dùng đúng scaler và feature order riêng.", "",
             "## Tổng hợp cả năm horizon", "", markdown_table(display), "",
             "## Theo từng horizon", "",
             "| Horizon | RMSE cũ | RMSE B | RMSE mới | N200 cũ | N200 B | N200 mới |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for horizon in coverage["horizon_hours"]:
        group = metrics[metrics.horizon_hours.eq(horizon)].set_index("model")
        lines.append(f"| +{horizon}h | " + " | ".join(
            [f"{group.loc[model, 'rmse_mm']:.2f}" for model in MODEL_NAMES]
            + [str(int(group.loc[model, 'n200'])) for model in MODEL_NAMES]) + " |")
    lines += ["", "## Chênh lệch model mới so với tham chiếu", ""]
    for reference in ("legacy", "model_b"):
        difference = summary.loc["candidate"] - summary.loc[reference]
        lines.append(f"- So với {MODEL_NAMES[reference]}: ΔRMSE **{difference.rmse_mm:+.2f} mm**, "
                     f"ΔN200 **{difference.n200:+.0f}**. Số âm là tốt hơn.")
    lines += ["", "## Độ lớn thay đổi dự báo so với thực tế", "",
              "ΔH = H dự báo(t+h) − H quan trắc(t), không phải sai số. Nhóm mạnh được chọn theo",
              "|ΔH thực tế| >200 mm, dùng cùng nhóm cho ba model; không dùng nhóm này để chuyển model online.", "",
              "| Horizon | TB độ lớn thực tế | Cũ | B | Mới | N mạnh | Thực tế nhóm mạnh | Cũ nhóm mạnh | B nhóm mạnh | Mới nhóm mạnh |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for horizon in coverage["horizon_hours"]:
        group = metrics[metrics.horizon_hours.eq(horizon)].set_index("model")
        first = group.loc["candidate"]
        values = [f"+{horizon}h", f"{first.actual_mean_abs_change_mm:.2f}"]
        values += [f"{group.loc[model, 'mean_abs_change_mm']:.2f}" for model in MODEL_NAMES]
        values += [str(int(first.fast_n)), f"{first.fast_actual_abs_mm:.2f}" if first.fast_n else "—"]
        values += [f"{group.loc[model, 'fast_pred_abs_mm']:.2f}" if first.fast_n else "—" for model in MODEL_NAMES]
        lines.append("| " + " | ".join(values) + " |")
    lines += ["", "[Metrics đầy đủ, gồm phân bố ΔH](metrics.csv) · "
              "[Dự báo từng mốc](predictions.csv.gz) · [Metadata/model/scaler](manifest.json)", ""]
    report = directory / "comparison.md"
    report.write_text("\n".join(lines))
    return report
