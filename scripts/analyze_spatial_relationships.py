"""Extract coordinates and compute geographic—not hydrologic—proximity."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from analysis_config import AnalysisPaths, METADATA_FILES, TARGET_STATION_NAMES
from analysis_utils import haversine_km, parse_geojson_point, read_csv


def coordinate_inventory(paths: AnalysisPaths) -> pd.DataFrame:
    station_records: list[dict[str, object]] = []
    for station_type, filename in METADATA_FILES.items():
        frame = read_csv(paths.data / filename)
        for _, row in frame.iterrows():
            if station_type == "reservoir":
                longitude = pd.to_numeric(row.get("kinhdo"), errors="coerce")
                latitude = pd.to_numeric(row.get("vido"), errors="coerce")
                name = row.get("tencongtrinh")
                identifier = row.get("id")
            else:
                point = parse_geojson_point(row.get("geom"))
                longitude, latitude = point if point else (None, None)
                name = row.get("name")
                identifier = row.get("code", row.get("id"))
            station_records.append(
                {
                    "type": station_type,
                    "id": identifier,
                    "name": name,
                    "longitude": longitude,
                    "latitude": latitude,
                    "coordinates_available": pd.notna(longitude) and pd.notna(latitude),
                }
            )
    return pd.DataFrame(station_records)


def pairwise_distances(stations: pd.DataFrame) -> pd.DataFrame:
    valid = stations.loc[stations["coordinates_available"]].reset_index(drop=True)
    rows = []
    for index_a, index_b in combinations(range(len(valid)), 2):
        station_a = valid.iloc[index_a]
        station_b = valid.iloc[index_b]
        if station_a["type"] == station_b["type"]:
            continue
        rows.append(
            {
                "type_a": station_a["type"],
                "name_a": station_a["name"],
                "type_b": station_b["type"],
                "name_b": station_b["name"],
                "distance_km": haversine_km(
                    station_a["longitude"], station_a["latitude"],
                    station_b["longitude"], station_b["latitude"],
                ),
                "hydrologic_relationship": "UNKNOWN",
            }
        )
    return pd.DataFrame(rows).sort_values("distance_km")


def plot_station_map(paths: AnalysisPaths, stations: pd.DataFrame) -> None:
    valid = stations.loc[stations["coordinates_available"]]
    colors = {"rain": "#2878b5", "water_level": "#c82423", "wind": "#56a64b", "reservoir": "#7c4aa5"}
    figure, axis = plt.subplots(figsize=(10, 9))
    for station_type, group in valid.groupby("type"):
        axis.scatter(group["longitude"], group["latitude"], label=station_type, alpha=0.8, color=colors[station_type])
    target_names = set(TARGET_STATION_NAMES.values())
    for _, station in valid.loc[valid["name"].isin(target_names)].iterrows():
        axis.annotate(station["name"], (station["longitude"], station["latitude"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.set_title("Sensor and reservoir coordinates (geographic proximity only)")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(paths.figures / "station_reservoir_map.png", dpi=160)
    plt.close(figure)


def run(paths: AnalysisPaths) -> None:
    paths.create_output_directories()
    stations = coordinate_inventory(paths)
    distances = pairwise_distances(stations)
    stations.to_csv(paths.reports / "station_coordinates.csv", index=False)
    distances.to_csv(paths.reports / "station_distances.csv", index=False)
    plot_station_map(paths, stations)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=AnalysisPaths().data)
    parser.add_argument("--reports-dir", type=Path, default=AnalysisPaths().reports)
    parser.add_argument("--figures-dir", type=Path, default=AnalysisPaths().figures)
    args = parser.parse_args()
    run(AnalysisPaths(args.data_dir, args.reports_dir, args.figures_dir))


if __name__ == "__main__":
    main()
