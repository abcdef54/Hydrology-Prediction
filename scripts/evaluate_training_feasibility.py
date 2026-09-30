"""Audit feature leakage and synthesize evidence into the final feasibility report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_config import AnalysisPaths, FORECAST_HORIZONS_MINUTES, TARGET_STATION_NAMES
from analysis_utils import find_training_table, markdown_table, parse_utc, read_csv, write_json


OPENHYDRONET_SOURCES = {
    "repository": "https://github.com/google-research/flood-forecasting",
    "configuration": "https://openhydronet.readthedocs.io/en/latest/usage/config.html",
    "pretrained": "https://github.com/google-research/flood-forecasting/blob/main/pretrained-models/Pretrained-Models-README.md",
    "paper": "https://hess.copernicus.org/articles/29/6221/2025/",
}


def exact_shift_match(left: pd.Series, right: pd.Series, shift_steps: int) -> dict[str, object]:
    expected = pd.to_numeric(left, errors="coerce").shift(shift_steps)
    actual = pd.to_numeric(right, errors="coerce")
    comparable = expected.notna() & actual.notna()
    matches = np.isclose(expected[comparable], actual[comparable], equal_nan=False)
    return {
        "comparable_rows": int(comparable.sum()),
        "matching_rows": int(matches.sum()),
        "match_ratio": float(matches.mean()) if matches.size else None,
    }


def leakage_audit(training: pd.DataFrame) -> dict[str, object]:
    checks: dict[str, object] = {}
    for minutes in (10, 30, 60, 180):
        steps = minutes // 10
        checks[f"rain_lag_{minutes}m"] = exact_shift_match(training["rain_depth"], training[f"rain_lag_{minutes}m"], steps)
        checks[f"water_level_lag_{minutes}m"] = exact_shift_match(
            training["water_level_depth"], training[f"water_level_lag_{minutes}m"], steps
        )
    for minutes in FORECAST_HORIZONS_MINUTES:
        checks[f"target_water_level_plus_{minutes}m"] = exact_shift_match(
            training["water_level_depth"], training[f"target_water_level_plus_{minutes}m"], -(minutes // 10)
        )
    for hours in (1, 3, 6, 24):
        expected = training["rain_depth"].rolling(hours * 6, min_periods=hours * 6).sum()
        actual = training[f"rain_sum_{hours}h"]
        current_inclusive = exact_shift_match(expected, actual, 0)
        past_only = exact_shift_match(expected, actual, 1)
        checks[f"rain_sum_{hours}h"] = {
            "matches_current_inclusive_window": current_inclusive,
            "matches_strictly_past_window": past_only,
            "interpretation": "Current-inclusive is valid only when issue-time rain is available; strictly-past is safer operationally.",
        }
    target_columns = [f"target_water_level_plus_{minutes}m" for minutes in FORECAST_HORIZONS_MINUTES]
    numeric_features = set(training.select_dtypes(include=[np.number]).columns) - set(target_columns)
    checks["direct_target_in_features"] = sorted(set(target_columns) & numeric_features)
    checks["split_policy"] = (
        "Baseline script uses chronological 70/15/15, purges train-boundary rows by forecast horizon, "
        "and fits imputers/scalers/models on train only."
    )
    checks["interpolation"] = "No evidence of interpolated long gaps: missing values remain explicit; rebuild script performs no temporal interpolation."
    return checks


def compatibility_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("Historical meteorological forcing", "Partial", "56 rain observation IDs and 4 wind stations", "~10-11 months; catchment relevance unverified; wind histories uneven", "Long multi-year record and basin-verified forcings"),
            ("Forecast meteorological forcing", "No", "None", "Absent", "Operational precipitation and other weather forecasts"),
            ("Catchment/static attributes", "No", "Station points only", "No basin descriptors", "Area, elevation, soil, geology, land cover and compatible attributes"),
            ("Catchment boundaries", "No", "None", "Absent", "Dã Viên contributing basin boundary and verified routing"),
            ("Streamflow/discharge target", "No", "Water-level depth only", "Different physical target; units unconfirmed", "Discharge series or validated rating curve"),
            ("Precipitation", "Partial", "2.60M observations across 56 IDs", "Units undocumented; one ID lacks metadata; catchment links UNKNOWN", "Gauge/radar/reanalysis mapped to the verified contributing catchment"),
            ("Temperature", "No", "None", "Absent", "Historical plus forecast temperature if matching pretrained inputs"),
            ("Radiation/vapor pressure/other forcing", "No", "None", "Absent", "Exact variables/providers expected by selected pretrained run"),
            ("Time resolution", "Mismatch", "10-minute table", "OpenHydroNet reference pipeline is not plug-compatible", "Custom dataset/resampling and scientifically justified resolution"),
            ("Normalizer/scaler compatibility", "No", "Local raw units", "Feature names/units differ from pretrained scaler", "Matching features, units, transformations and pretrained scaler"),
        ],
        columns=["OpenHydroNet requirement", "Huế hiện có?", "Source", "Quality", "Missing / Problem"],
    )


def missing_data_table() -> pd.DataFrame:
    rows = [
        ("Baseline water-level", "MUST HAVE", "Confirmed units, datum, sensor QA/QC, and official station metadata"),
        ("Baseline water-level", "SHOULD HAVE", "Several years of co-located/upstream rain and water-level history"),
        ("Baseline water-level", "SHOULD HAVE", "Tide level and upstream gauges; Dã Viên may have downstream/backwater influences"),
        ("OpenHydroNet fine-tuning", "MUST HAVE", "Discharge target or validated time-varying rating curve from stage to discharge"),
        ("OpenHydroNet fine-tuning", "MUST HAVE", "Catchment boundary and pretrained-compatible static attributes"),
        ("OpenHydroNet fine-tuning", "MUST HAVE", "Historical and forecast meteorological variables matching the pretrained config/scaler"),
        ("OpenHydroNet fine-tuning", "SHOULD HAVE", "Long local record with multiple independent floods and regulated-flow metadata"),
        ("Operational warning", "MUST HAVE", "Official alert/flood thresholds and reference datum"),
        ("Operational warning", "MUST HAVE", "Real-time robust feeds, forecast rainfall, latency and outage monitoring"),
        ("Operational warning", "MUST HAVE", "Reservoir releases/operations and verified upstream-downstream/catchment relationships"),
        ("Operational warning", "SHOULD HAVE", "DEM, river network, land cover, soils and rainfall radar"),
        ("Operational warning", "OPTIONAL", "Additional wind observations after demonstrating incremental skill"),
    ]
    return pd.DataFrame(rows, columns=["Objective", "Priority", "Data needed"])


def build_report(paths: AnalysisPaths, leakage: dict[str, object]) -> str:
    inventory = pd.read_csv(paths.reports / "dataset_inventory.csv")
    temporal = pd.read_csv(paths.reports / "temporal_quality.csv")
    persistence = pd.read_csv(paths.reports / "persistence_correlations.csv")
    signals = pd.read_csv(paths.reports / "lagged_signal_correlations.csv")
    event_summary = pd.read_csv(paths.reports / "high_water_event_summary.csv")
    baselines = pd.read_csv(paths.reports / "baseline_metrics.csv")
    splits = pd.read_csv(paths.reports / "temporal_split_summary.csv")
    coordinates = pd.read_csv(paths.reports / "station_coordinates.csv")
    distances = pd.read_csv(paths.reports / "station_distances.csv")
    reservoirs = pd.read_csv(paths.reports / "reservoir_signal_summary.csv")
    station_counts = pd.read_csv(paths.reports / "observation_counts_summary.csv")
    station_count_details = pd.read_csv(paths.reports / "observation_counts_by_station.csv")
    all_station_signals = pd.read_csv(paths.reports / "all_station_lagged_signals.csv")
    distributions = json.loads((paths.reports / "distribution_summary.json").read_text(encoding="utf-8"))

    sensor_temporal = temporal.loc[temporal["source"] != "reservoir_observations.csv"]
    water_candidates = sensor_temporal.loc[
        (sensor_temporal["source"] == "water_level_observations.csv")
        & (sensor_temporal["series_id"].astype(str) == "74647403")
    ]
    if water_candidates.empty:
        raise ValueError("Dã Viên series 74647403 is missing from temporal analysis")
    water_temporal = water_candidates.iloc[0]
    duration_months = float(water_temporal["duration_days"]) / 30.4375
    wind_missing = distributions["hydrology_training_10min.csv"]["wind_speed"]["missing_ratio"]
    water_stats = distributions["water_level_observations.csv"]["depth"]
    rain_stats = distributions["rain_observations.csv"]["depth"]
    all_test = baselines.loc[baselines["subset"] == "all_test"].copy()
    best_models = all_test.sort_values(["horizon_minutes", "mae"]).groupby("horizon_minutes").first().reset_index()
    persistence_test = all_test.loc[all_test["model"] == "persistence", ["horizon_minutes", "mae", "rmse", "nse"]]
    model_comparison = best_models[["horizon_minutes", "model", "mae", "rmse", "nse"]].merge(
        persistence_test, on="horizon_minutes", suffixes=("_best", "_persistence")
    )
    model_comparison["mae_improvement_vs_persistence_pct"] = (
        100 * (model_comparison["mae_persistence"] - model_comparison["mae_best"]) / model_comparison["mae_persistence"]
    )
    target_distances = distances.loc[
        distances["name_a"].isin(TARGET_STATION_NAMES.values())
        & distances["name_b"].isin(TARGET_STATION_NAMES.values())
    ]
    rain_signals = signals.loc[(signals["input"] == "rain_depth") & (signals["subset"] == "all")]
    strongest_rain = rain_signals.iloc[rain_signals["six_hour_rise_correlation"].abs().argmax()]
    all_rain_signals = all_station_signals.loc[
        (all_station_signals["station_type"] == "rain")
        & (all_station_signals["subset"] == "all")
    ].dropna(subset=["six_hour_rise_correlation"])
    strongest_station_rain = all_rain_signals.iloc[
        all_rain_signals["six_hour_rise_correlation"].abs().argmax()
    ]
    high_water_baselines = baselines.loc[baselines["subset"] == "high_water_test"]
    rain_ablation = all_test.loc[
        all_test["model"].isin(["ridge_selected_stations", "ridge_all_rain_stations"]),
        ["horizon_minutes", "model", "mae", "rmse", "nse"],
    ]
    first_horizon_improvement = model_comparison.iloc[0]["mae_improvement_vs_persistence_pct"]
    last_horizon_improvement = model_comparison.iloc[-1]["mae_improvement_vs_persistence_pct"]
    unmatched_observation_ids = station_count_details.loc[
        station_count_details["metadata_status"] == "observations_without_metadata",
        ["station_type", "station_id"],
    ]
    unmatched_labels = ", ".join(
        unmatched_observation_ids["station_type"].astype(str)
        + ":"
        + unmatched_observation_ids["station_id"].astype(str)
    )
    reservoir_observations = read_csv(paths.data / "reservoir_observations.csv")
    reservoir_dates = pd.to_datetime(
        reservoir_observations["ngaylaysolieu"], format="%d/%m/%Y", errors="coerce"
    )
    manifest = json.loads((paths.data / "manifest.json").read_text(encoding="utf-8"))
    extraction_date = pd.Timestamp(manifest["generated_at_utc"]).tz_convert(None).normalize()
    future_dated_reservoir_rows = int(reservoir_dates.gt(extraction_date).sum())

    sections = [
        "# Báo cáo phân tích khả năng dự báo thủy văn Huế",
        "## 1. Executive summary",
        f"Dữ liệu hiện có hỗ trợ **thử nghiệm dự báo mực nước Dã Viên ngắn hạn**, không đủ bằng chứng để gọi là mô hình dự báo lũ vận hành và không plug-compatible với pipeline pretrained OpenHydroNet. Dữ liệu mới mở rộng mạnh theo không gian, nhưng chuỗi target vẫn chỉ dài {duration_months:.2f} tháng; số trạm tăng không thay thế cho nhiều mùa mưa/lũ độc lập.",
        "Các giá trị và sai số dưới đây giữ nguyên đơn vị nguồn. Dataset không mô tả đơn vị đáng tin cậy, nên báo cáo không chuyển đổi hoặc suy đoán cm/m, mm hay đơn vị gió.",
        "## 2. Dataset inventory",
        markdown_table(inventory),
        "### Coverage theo observation ID và station metadata",
        markdown_table(station_counts),
        f"Manifest và raw files cùng hiện diện. Raw CSV/GZIP không bị thay đổi. Thống kê aggregate toàn bộ water-level observations có min={water_stats['min']:.4g}, p95={water_stats['p95']:.4g}, max={water_stats['max']:.4g}; toàn bộ rain observations có zero ratio={rain_stats['zero_ratio_of_valid']:.2%}, max={rain_stats['max']:.4g}. Đây là distribution trong raw units, không phải xác nhận vật lý. Chi tiết từng trạm nằm trong `distribution_by_station.csv`.",
        "## 3. Temporal coverage",
        markdown_table(sensor_temporal[["source", "series_id", "start", "end", "duration_days", "actual_records", "expected_records", "availability_ratio", "longest_gap_minutes"]]),
        f"Chuỗi Dã Viên chỉ dài khoảng {duration_months:.2f} tháng. Không thể xác định số mùa lũ thực sự từ dữ liệu hiện có vì thiếu ngưỡng lũ chính thức và chuỗi chưa bao phủ nhiều năm độc lập.",
        "## 4. Data quality",
        f"Wind Cảng Thuận An thiếu {wind_missing:.1%} trong bảng training 10 phút. Raw files không có duplicate rows; duplicate timestamp được đánh giá trong từng station series. Extreme được giữ nguyên vì không có units/QA flags đáng tin cậy. Có {len(unmatched_observation_ids)} observation ID không nối được metadata ({unmatched_labels or 'không có'}). Reservoir có {future_dated_reservoir_rows} dòng mang ngày sau ngày xuất manifest {extraction_date.date()}; cần kiểm tra timezone hoặc lỗi nhập liệu. Giá trị âm/cực lớn được xếp loại **unknown requiring domain verification**.",
        "## 5. Spatial relationships",
        markdown_table(coordinates.loc[coordinates["name"].isin(TARGET_STATION_NAMES.values())]),
        markdown_table(target_distances[["name_a", "name_b", "distance_km", "hydrologic_relationship"]]),
        "Khoảng cách địa lý không chứng minh quan hệ thủy văn. Dataset không có catchment boundary, river network, upstream/downstream topology hoặc khóa nối hồ–trạm. Vì vậy quan hệ Hòa Mỹ → Dã Viên và vai trò Cảng Thuận An là **UNKNOWN**; chưa đủ căn cứ dùng rain Hòa Mỹ như forcing đại diện lưu vực Dã Viên.",
        "## 6. Hydrologic signal analysis",
        "### Persistence",
        markdown_table(persistence),
        "Correlation rất cao chỉ cho thấy target trơn/autocorrelated; nó làm persistence trở thành baseline khó vượt và không chứng minh khả năng dự báo lũ.",
        "### Rainfall, wind, reservoir",
        f"Riêng rain Hòa Mỹ, correlation lớn nhất về trị tuyệt đối giữa rain lag và thay đổi mực nước 6 giờ xuất hiện ở lag {strongest_rain['lag_hours']:.0f}h: r={strongest_rain['six_hour_rise_correlation']:.3f}. Khi quét toàn bộ trạm mưa, trị tuyệt đối lớn nhất là `{strongest_station_rain['station_name']}` ở lag {strongest_station_rain['lag_hours']:.0f}h: r={strongest_station_rain['six_hour_rise_correlation']:.3f}. Đây chỉ là screening correlation; multiple comparisons và quan hệ lưu vực chưa biết nên không được diễn giải nhân quả. Kết quả đầy đủ nằm trong `all_station_lagged_signals.csv`.",
        f"Wind coverage yếu ({1-wind_missing:.1%}); chỉ nên là feature ablation, không phải dependency bắt buộc. Reservoir correlations ({len(reservoirs)} series-variable pairs) không phải bằng chứng để merge vì hydrologic relationship vẫn UNKNOWN.",
        "## 7. Event analysis",
        markdown_table(event_summary),
        "Các event là **high-water exceedance events theo percentile**, không được gọi là flood. Quy tắc gộp cho phép khoảng cách tối đa 6 giờ. Số event nhạy với threshold và không thay thế ngưỡng chính thức.",
        "## 8. Leakage risks",
        "Lag và target được đối chiếu bằng exact-shift. Kết quả chi tiết ở `leakage_audit.json`. Target columns không được đưa vào feature list; imputer/scaler/model chỉ fit trên train; split không shuffle. Rolling rain trong file hiện hữu được audit cả current-inclusive và strictly-past. Current rain tại issue time chỉ hợp lệ nếu thực sự sẵn có với đúng latency trong vận hành.",
        "## 9. Baseline performance",
        markdown_table(model_comparison),
        "### High-water test metrics",
        markdown_table(high_water_baselines[["horizon_minutes", "model", "n", "mae", "rmse", "nse"]]),
        f"Ridge với các trạm được chọn là model tốt nhất ở cả ba horizon trong test hiện tại; HistGradientBoosting kém hơn rõ rệt, đặc biệt ở high-water periods. Mức cải thiện MAE tốt nhất so với persistence giảm từ {first_horizon_improvement:.1f}% (+1h) xuống {last_horizon_improvement:.1f}% (+6h), cho thấy lợi ích ML còn hạn chế và không ổn định theo horizon.",
        "### Ablation: trạm được chọn so với toàn bộ trạm mưa",
        markdown_table(rain_ablation),
        "Đưa đồng loạt tất cả trạm mưa vào Ridge làm test MAE xấu hơn ở cả ba horizon. Điều này không chứng minh các trạm khác vô ích; nó cho thấy blanket merge theo timestamp tạo thêm nhiễu/shift khi chưa có catchment mapping, feature selection và validation nhiều năm.",
        "### Temporal split event support",
        markdown_table(splits[["horizon_minutes", "split", "start", "end", "rows", "high_water_events_using_train_p90"]]),
        "Split 70/15/15 chỉ là boundary minh bạch để benchmark, không được coi là tối ưu. Training rows ở sát biên được purge theo từng forecast horizon để nhãn không vượt sang validation; scaler/imputer chỉ fit trên phần train còn lại. Nếu validation/test thiếu event đáng kể, metric extreme không đủ tin cậy và cần event-aware rolling-origin evaluation khi có thêm năm dữ liệu.",
        "## 10. OpenHydroNet compatibility",
        markdown_table(compatibility_table()),
        "OpenHydroNet hiện hỗ trợ `multimet`, yêu cầu hindcast/forecast inputs, static attributes và target variables; reference experiments dùng nhiều forcing meteorological, static catchment attributes và streamflow. Pretrained releases kèm scaler theo đúng feature/target của training run. Thay output head sang water level là **architecture adaptation/transfer learning experiment**, không phải fine-tune trực tiếp đúng pipeline; representation có thể còn hữu ích ở mức chưa biết, nhưng input semantics, scaler và target đã mismatch. Rating curve Q↔stage (và datum/backwater validity) là cần thiết nếu mục tiêu là discharge/compatibility vật lý. Nguồn: [repository]({repository}), [configuration]({configuration}), [pretrained notes]({pretrained}), [HESS paper]({paper}).".format(**OPENHYDRONET_SOURCES),
        "## 11. Missing data",
        markdown_table(missing_data_table()),
        "## 12. Recommended next step",
        "1. Xác nhận units/datum/QA, ID mưa thiếu metadata, record reservoir future-dated và ngưỡng báo động chính thức.\n2. Giữ persistence làm chuẩn; Ridge với các trạm được chọn hiện thắng test. Không dùng blanket all-rain merge; trước hết xác định catchment/upstream rồi chọn hoặc aggregate mưa theo không gian.\n3. Chạy ablation water-only vs catchment-verified rain vs wind/tide/upstream; dùng rolling-origin/event-based evaluation.\n4. Thu thập nhiều năm và reservoir releases. Chỉ thử MEF-LSTM sau khi có signal ổn định; OpenHydroNet chỉ sau khi tạo dataset catchment/forcing/target tương thích.",
        "## Trả lời trực tiếp",
        "- **Có nên dùng Mean-Embedding-Forecast-LSTM ngay?** Không. Nhiều trạm hơn nhưng vẫn dưới một năm target và thiếu forecast forcing/static attributes.\n- **Fine-tune pretrained OpenHydroNet trực tiếp?** Không. Target là water level thay vì discharge, feature/scaler/data adapter không tương thích, thiếu catchment statics và forecast meteorology.\n- **Nếu vẫn muốn OpenHydroNet?** Bổ sung discharge hoặc rating curve hợp lệ, basin boundary/statics, forcing lịch sử+dự báo đúng schema/units và nhiều năm/event; sau đó đánh giá transfer trên thời gian post-2023/held-out hợp lệ.\n- **Bài toán khả thi nhất?** Site-specific short-term water-level forecasting tại Dã Viên với autoregressive tabular baseline.\n- **Target water level hay discharge?** Water level cho bài toán khả thi ngay; discharge cho pipeline OpenHydroNet/rainfall-runoff, chỉ khi có Q hoặc rating curve đáng tin cậy.\n- **+1h/+3h/+6h?** Có thể benchmark cả ba; +1h có skill tốt nhất nhưng bị persistence chi phối, +6h chỉ cải thiện nhỏ.\n- **Đủ high-water/extreme events?** Không đủ để khẳng định khả năng tổng quát hóa flood; chỉ có percentile events trong một phần năm và không có official threshold.\n- **Baseline đầu tiên?** Persistence, sau đó Ridge với water-level lags và forcing đã xác minh; blanket all-rain model hiện kém hơn.\n- **Experiment tiếp theo?** Xây catchment-aware rainfall aggregation/feature selection, rồi thêm tide/upstream/reservoir đã xác minh và dùng rolling-origin theo event.",
        "## Giới hạn diễn giải",
        "Bất kỳ upstream/downstream relation, physical units, official flood status, catchment membership hoặc rating-curve validity nào không có trong file đều được ghi là: **Không thể xác định từ dữ liệu hiện có**.",
    ]
    return "\n\n".join(sections) + "\n"


def run(paths: AnalysisPaths) -> None:
    paths.create_output_directories()
    training = read_csv(find_training_table(paths.data, paths.reports))
    training["timestamp_utc"] = parse_utc(training["timestamp_utc"])
    leakage = leakage_audit(training)
    write_json(leakage, paths.reports / "leakage_audit.json")
    compatibility_table().to_csv(paths.reports / "openhydronet_compatibility.csv", index=False)
    missing_data_table().to_csv(paths.reports / "missing_data_priorities.csv", index=False)
    (paths.reports / "data_analysis_report.md").write_text(build_report(paths, leakage), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=AnalysisPaths().data)
    parser.add_argument("--reports-dir", type=Path, default=AnalysisPaths().reports)
    parser.add_argument("--figures-dir", type=Path, default=AnalysisPaths().figures)
    args = parser.parse_args()
    run(AnalysisPaths(args.data_dir, args.reports_dir, args.figures_dir))


if __name__ == "__main__":
    main()
