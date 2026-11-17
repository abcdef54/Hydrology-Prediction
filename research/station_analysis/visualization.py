"""Publication-friendly static EDA figures using cached tables and clipped DEMs."""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .statistics import exact_change

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def _catalog(metadata):
    data = metadata.copy()
    if "station_id" not in data:
        data["station_id"] = data["id"] if "id" in data else data.index.astype(str)
    data["station_id"] = data["station_id"].astype(str)
    return data.set_index("station_id", drop=False)


def _labels(metadata):
    return [f"{i + 1:02d} {str(row.get('name', station_id))[:52]}"
            for i, (station_id, row) in enumerate(metadata.iterrows())]


def _save(figure, path):
    figure.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def _boxplot(axis, arrays, labels, color):
    values = [array if len(array) else np.array([np.nan]) for array in arrays]
    boxes = axis.boxplot(values, vert=False, tick_labels=labels,
                         showfliers=False, patch_artist=True, widths=0.55)
    for box in boxes["boxes"]:
        box.set_facecolor(color)
        box.set_alpha(0.7)
    axis.invert_yaxis()
    axis.grid(axis="x", alpha=0.2)
    axis.tick_params(axis="y", labelsize=8)


def _plot_boundary(axis, path, crs):
    """Draw available polygon outlines only; administrative borders are contextual."""
    from pyproj import Transformer
    from shapely.geometry import shape
    from shapely.ops import transform
    if not path or not Path(path).is_file():
        return
    document = json.loads(Path(path).read_text())
    features = document.get("features", [{"geometry": document}])
    project = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    for feature in features:
        geometry = transform(project, shape(feature["geometry"]))
        polygons = list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]
        for polygon in polygons:
            if polygon.geom_type == "Polygon":
                x, y = polygon.exterior.xy
                axis.plot(x, y, color="#333333", linewidth=0.6, alpha=0.65)


def _label_map_points(axis, points):
    """Offset crowded station numbers without introducing a plotting dependency."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.transforms import Bbox

    axis.figure.canvas.draw()
    renderer = axis.figure.canvas.get_renderer()
    font = FontProperties(size=8, weight="bold")
    pixel_scale = axis.figure.dpi / 72
    placed = []
    candidates = [(5, 5), (5, -15), (-18, 5), (-18, -15),
                  (16, 14), (-30, 14), (16, -26), (-30, -26),
                  (0, 28), (0, -37), (31, 0), (-43, 0)]
    for x, y, label in points:
        px, py = axis.transData.transform((x, y))
        width, height, _ = renderer.get_text_width_height_descent(label, font, False)
        choices = []
        for order, (dx, dy) in enumerate(candidates):
            bounds = Bbox.from_bounds(px + dx * pixel_scale - 2,
                                      py + dy * pixel_scale - 2, width + 4, height + 4)
            overlaps = sum(bounds.overlaps(previous) for previous in placed)
            choices.append((overlaps, order, dx, dy, bounds))
        _, _, dx, dy, bounds = min(choices, key=lambda item: item[:2])
        placed.append(bounds)
        axis.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points",
                      fontsize=8, fontweight="bold", ha="left", va="bottom",
                      bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.8},
                      arrowprops={"arrowstyle": "-", "color": "#555555", "linewidth": 0.45})


def plot_results(hourly, metadata, tables, dem_paths, config) -> list[Path]:
    """Create figures without fitting models, interpolating observations or web calls.

    The station order follows the metadata and includes stations with no data.
    DEM metadata/elevation is never interpreted as a station's water-level zero.
    """
    output = Path(config["output_dir"]) / "figures"
    output.mkdir(parents=True, exist_ok=True)
    stations = _catalog(metadata)
    station_ids = stations.index.tolist()
    labels = _labels(stations)
    frame = hourly.reindex(columns=station_ids).replace([np.inf, -np.inf], np.nan)
    frame = frame.copy()
    frame.index = frame.index.tz_convert(config.get("timezone", "Asia/Ho_Chi_Minh"))
    paths = []
    count = len(stations)
    height = max(6, count * 0.35)

    monthly = tables.get("monthly_statistics", pd.DataFrame())
    if not monthly.empty:
        coverage = (1 - monthly.pivot(index="station_id", columns="month", values="missing_rate"))
        coverage = coverage.reindex(station_ids)
        figure, axis = plt.subplots(figsize=(12, height))
        raster = axis.imshow(coverage.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="YlGnBu")
        axis.set_xticks(range(len(coverage.columns)), coverage.columns, rotation=45, ha="right")
        axis.set_yticks(range(count), labels)
        axis.set_title("Độ phủ quan trắc theo giờ, từng tháng — 0% bao gồm trạm chưa có dữ liệu")
        figure.colorbar(raster, ax=axis, label="Tỷ lệ giờ có giá trị hữu hạn")
        paths.append(_save(figure, output / "01_monthly_coverage.png"))

    columns = 3
    rows = math.ceil(count / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(16, rows * 2.2), sharex=True, squeeze=False)
    for i, (station_id, label) in enumerate(zip(station_ids, labels)):
        axis = axes.flat[i]
        series = frame[station_id]
        axis.plot(series.index, series, linewidth=0.5, color="#27668a")
        if not series.notna().any():
            axis.text(0.5, 0.5, "Chưa có quan trắc", transform=axis.transAxes,
                      ha="center", va="center", color="#666666")
        axis.set_title(label, loc="left", fontsize=8)
        axis.set_ylabel("mm", fontsize=8)
        axis.grid(alpha=0.2)
        axis.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
        axis.tick_params(labelsize=7)
    for axis in list(axes.flat)[count:]:
        axis.set_visible(False)
    figure.suptitle("Chuỗi mực nước/ngập gốc theo giờ — giữ khoảng thiếu; mỗi trạm có trục y riêng", y=1.005)
    figure.tight_layout()
    paths.append(_save(figure, output / "02_station_time_series.png"))

    figure, axis = plt.subplots(figsize=(12, height))
    centered = []
    for station_id in station_ids:
        values = frame[station_id].dropna()
        centered.append((values - values.median()).to_numpy())
    _boxplot(axis, centered, labels, "#75adc3")
    axis.axvline(0, color="#777777", linewidth=0.7)
    axis.set_xlabel("Giá trị trừ median riêng từng trạm (mm); whiskers 1,5 IQR, ẩn điểm ngoại lệ")
    axis.set_title("Phân phối đã loại offset — không giả định các trạm có chung mốc cao độ")
    paths.append(_save(figure, output / "03_centered_distributions.png"))

    figure, axes = plt.subplots(1, 2, figsize=(17, height), sharey=True)
    threshold = config.get("rapid_threshold_mm", 200)
    for axis, horizon, color in zip(axes, [6, 12], ["#74a993", "#bd9275"]):
        arrays = [exact_change(frame[station_id], horizon).dropna().to_numpy()
                  for station_id in station_ids]
        _boxplot(axis, arrays, labels, color)
        for signed in (-threshold, threshold):
            axis.axvline(signed, linestyle="--", color="#b13c3c", linewidth=0.9)
        axis.axvline(0, color="#777777", linewidth=0.6)
        axis.set_title(f"ΔH sau +{horizon}h — đúng timestamp, không nội suy")
        axis.set_xlabel(f"mm; vạch đỏ ±{threshold:g} mm; boxplot ẩn điểm ngoại lệ")
    # invert_yaxis twice would cancel the inversion on shared axes.
    axes[0].set_ylim(count + 0.5, 0.5)
    figure.tight_layout()
    paths.append(_save(figure, output / "04_horizon_changes.png"))

    correlation = tables.get("correlations", pd.DataFrame())
    if not correlation.empty:
        matrix = pd.DataFrame(np.nan, index=station_ids, columns=station_ids)
        subset = correlation.loc[(correlation.kind == "change_6h") & (correlation.lag_hours == 0)
                                 & (correlation.paired_hours >= config.get("min_pair_hours", 168))]
        for row in subset.itertuples():
            matrix.loc[str(row.station_a), str(row.station_b)] = row.pearson_r
            matrix.loc[str(row.station_b), str(row.station_a)] = row.pearson_r
        for station_id in station_ids:
            delta = exact_change(frame[station_id], 6).dropna()
            if len(delta) >= config.get("min_pair_hours", 168) and delta.nunique() > 1:
                matrix.loc[station_id, station_id] = 1
        figure, axis = plt.subplots(figsize=(12, 10))
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("#dddddd")
        raster = axis.imshow(np.ma.masked_invalid(matrix.to_numpy()), cmap=cmap, vmin=-1, vmax=1)
        axis.set_xticks(range(count), [f"{i + 1:02d}" for i in range(count)])
        axis.set_yticks(range(count), labels, fontsize=8)
        for left in range(count):
            for right in range(count):
                value = matrix.iloc[left, right]
                if np.isfinite(value):
                    axis.text(right, left, f"{value:.1f}", ha="center", va="center", fontsize=6,
                              color="white" if abs(value) > 0.65 else "black")
        axis.set_title("Tương quan ΔH +6h trên giờ chung — ô xám: thiếu overlap hoặc chuỗi hằng")
        figure.colorbar(raster, ax=axis, label="Pearson r; tương quan không chứng minh ảnh hưởng nhân quả")
        figure.tight_layout()
        paths.append(_save(figure, output / "05_change_correlation.png"))

    fabdem = dem_paths.get("FABDEM") or dem_paths.get("fabdem")
    if fabdem and Path(fabdem).is_file():
        import rasterio
        from rasterio.enums import Resampling
        from pyproj import Transformer
        with rasterio.open(fabdem) as source:
            ratio = min(1.0, 1200 / max(source.height, source.width))
            terrain = source.read(1, out_shape=(max(1, round(source.height * ratio)),
                                               max(1, round(source.width * ratio))),
                                  masked=True, resampling=Resampling.average)
            bounds = source.bounds
            projection = Transformer.from_crs("EPSG:4326", source.crs, always_xy=True)
            figure, axis = plt.subplots(figsize=(12, 11))
            finite = terrain.compressed()
            vmax = max(30, float(np.percentile(finite, 97))) if len(finite) else 100
            raster = axis.imshow(terrain, extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                                 cmap="terrain", vmin=0, vmax=vmax)
            _plot_boundary(axis, config.get("boundary_file"), source.crs)
            station_points = []
            for i, (_, row) in enumerate(stations.iterrows()):
                if pd.isna(row.get("longitude")) or pd.isna(row.get("latitude")):
                    continue
                x, y = projection.transform(row.longitude, row.latitude)
                axis.scatter(x, y, marker="o", s=24, facecolor="#bb2433", edgecolor="white", linewidth=0.6)
                station_points.append((x, y, f"{i + 1:02d}"))
            geographic = source.crs.is_geographic
            axis.set_xlabel("Kinh độ" if geographic else f"Easting ({source.crs})")
            axis.set_ylabel("Vĩ độ" if geographic else "Northing (m)")
            axis.set_title("FABDEM Huế và các vị trí trạm — số thứ tự theo bảng metadata")
            figure.colorbar(raster, ax=axis, label="Cao độ địa hình (m, EGM2008); không phải mốc quan trắc")
            _label_map_points(axis, station_points)
            paths.append(_save(figure, output / "06_station_terrain_map.png"))

    terrain = tables.get("terrain", tables.get("terrain_statistics", pd.DataFrame()))
    if not terrain.empty:
        available = sorted(terrain.radius_m.unique())
        radius = min(available, key=lambda value: abs(value - 200))
        chosen = terrain.loc[terrain.radius_m == radius].copy()
        figure, axis = plt.subplots(figsize=(12, height))
        colors = {"fabdem": "#327b87", "copernicus": "#b68c50"}
        for i, name in enumerate(chosen.dem_source.unique()):
            rows = chosen.loc[chosen.dem_source == name].set_index("station_id").reindex(station_ids)
            y = np.arange(count) + (i - 0.5) * 0.26
            median = rows.median_m.to_numpy()
            lower = np.maximum(0, median - rows.p10_m.to_numpy())
            upper = np.maximum(0, rows.p90_m.to_numpy() - median)
            axis.errorbar(median, y, xerr=np.vstack([lower, upper]), fmt="o", markersize=3,
                          color=colors.get(str(name).lower()), label=str(name), capsize=2, alpha=0.85)
        axis.set_yticks(range(count), labels, fontsize=8)
        axis.invert_yaxis()
        axis.set_xlabel("Median và P10–P90 cao độ địa hình (m, EGM2008)")
        axis.set_title(f"Đối chiếu địa hình quanh trạm, bán kính {radius:g} m — không phải cao độ đáy sông")
        axis.legend()
        axis.grid(axis="x", alpha=0.2)
        paths.append(_save(figure, output / "07_terrain_comparison.png"))

    seasonal = tables.get("seasonal_statistics", pd.DataFrame())
    summary = tables.get("station_statistics", pd.DataFrame())
    if not seasonal.empty and not summary.empty:
        figure, axes = plt.subplots(1, 2, figsize=(17, height), sharey=True)
        flood_label = "Tháng " + ", ".join(str(month) for month in config.get("flood_months", [9, 10, 11, 12]))
        for offset, name, label, color in [(-0.16, "flood_months", flood_label, "#b36e4f"),
                                           (0.16, "other_months", "Các tháng còn lại", "#468ca8")]:
            subset = seasonal.loc[seasonal.season == name].set_index("station_id").reindex(station_ids)
            axes[0].barh(np.arange(count) + offset, subset.p95_mm - subset.p05_mm,
                         height=0.28, label=label, color=color)
        subset = summary.set_index("station_id").reindex(station_ids)
        axes[1].barh(np.arange(count) - 0.16, subset.strict_water_windows, height=0.28,
                     label="Lịch sử 100% giờ có nước", color="#468ca8")
        axes[1].barh(np.arange(count) + 0.16, subset.history80_water_windows, height=0.28,
                     label="Lịch sử ≥80%, anchor/targets đủ", color="#a9bdc0")
        axes[0].set_yticks(range(count), labels, fontsize=8)
        axes[0].invert_yaxis()
        axes[0].set_title("Độ rộng P95–P5 theo mùa (mm)")
        axes[1].set_title(f"Cửa sổ nước {config.get('sequence_length', 96)}h, targets +6h/+12h")
        axes[1].set_xlabel("Số windows; chưa kiểm tra mưa/forecast, không phải mẫu train cuối cùng")
        for axis in axes:
            axis.legend(fontsize=8)
            axis.grid(axis="x", alpha=0.2)
        figure.tight_layout()
        paths.append(_save(figure, output / "08_seasonality_and_windows.png"))
    return paths
