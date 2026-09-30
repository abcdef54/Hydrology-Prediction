"""Classify Hue stations relative to Dã Viên using HydroBASINS/HydroRIVERS topology."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Point

from analysis_config import AnalysisPaths
from analysis_utils import parse_geojson_point, read_csv, write_json


WGS84 = "EPSG:4326"
HUE_PROJECTED_CRS = "EPSG:32648"
DA_VIEN_NAME = "Trạm Dã Viên (Sông Hương)"
HYDRORIVERS_LAYER = "HydroRIVERS_v10_as"
DEFAULT_BASIN_LEVEL = 12
DEFAULT_MAXIMUM_RIVER_SNAP_METERS = 5_000.0


@dataclass(frozen=True)
class HydrographyPaths:
    basin_path: Path
    river_gdb_path: Path


def normalized_station_name(name: object) -> str:
    return " ".join(str(name).casefold().split())


def point_from_geojson(value: object) -> Point | None:
    coordinates = parse_geojson_point(value)
    return Point(*coordinates) if coordinates else None


def metadata_station_records(paths: AnalysisPaths) -> pd.DataFrame:
    """Read every requested station table into one coordinate inventory."""
    definitions = (
        ("rain", "rain_stations.csv", "code", "name"),
        ("water_level", "water_level_stations.csv", "code", "name"),
        ("wind", "wind_stations.csv", "id", "name"),
    )
    records: list[dict[str, object]] = []
    coordinate_references: dict[str, dict[str, object]] = {}
    for station_type, filename, identifier_column, name_column in definitions:
        stations = read_csv(paths.data / filename)
        for _, station in stations.iterrows():
            geometry = point_from_geojson(station.get("geom"))
            record = {
                "station_id": str(station[identifier_column]),
                "station_name": station[name_column],
                "type": station_type,
                "coordinate_source": filename,
                "geometry": geometry,
            }
            records.append(record)
            if geometry is not None:
                coordinate_references[normalized_station_name(station[name_column])] = record

    reservoirs = read_csv(paths.data / "reservoirs.csv")
    for _, reservoir in reservoirs.iterrows():
        longitude = pd.to_numeric(reservoir.get("kinhdo"), errors="coerce")
        latitude = pd.to_numeric(reservoir.get("vido"), errors="coerce")
        geometry = (
            Point(float(longitude), float(latitude))
            if pd.notna(longitude) and pd.notna(latitude)
            else None
        )
        coordinate_source = "reservoirs.csv"
        if geometry is None:
            reference = coordinate_references.get(
                normalized_station_name(reservoir["tencongtrinh"])
            )
            if reference:
                geometry = reference["geometry"]
                coordinate_source = (
                    f"name_match:{reference['type']}:{reference['station_id']}"
                )
        records.append(
            {
                "station_id": str(reservoir["id"]),
                "station_name": reservoir["tencongtrinh"],
                "type": "reservoir",
                "coordinate_source": coordinate_source,
                "geometry": geometry,
            }
        )
    return pd.DataFrame(records)


def station_geodataframe(paths: AnalysisPaths) -> gpd.GeoDataFrame:
    records = metadata_station_records(paths)
    return gpd.GeoDataFrame(records, geometry="geometry", crs=WGS84)


def discover_hydrography_paths(
    data_directory: Path,
    basin_level: int,
) -> HydrographyPaths:
    basin_matches = sorted(
        data_directory.glob(f"**/hybas_*_lev{basin_level:02d}_v1c.shp")
    )
    river_matches = sorted(
        path
        for path in data_directory.glob("**/*.gdb")
        if path.is_dir()
        and "hydrorivers" in path.name.casefold()
        and (path / "gdb").exists()
    )
    if len(basin_matches) != 1:
        raise FileNotFoundError(
            f"Expected one HydroBASINS level {basin_level} shapefile, found {basin_matches}"
        )
    if len(river_matches) != 1:
        raise FileNotFoundError(
            f"Expected one HydroRIVERS FileGDB directory, found {river_matches}"
        )
    return HydrographyPaths(basin_matches[0], river_matches[0])


def expanded_bounds(
    stations: gpd.GeoDataFrame,
    padding_degrees: float = 0.25,
) -> tuple[float, float, float, float]:
    valid_stations = stations.loc[stations.geometry.notna()]
    minimum_x, minimum_y, maximum_x, maximum_y = valid_stations.total_bounds
    return (
        minimum_x - padding_degrees,
        minimum_y - padding_degrees,
        maximum_x + padding_degrees,
        maximum_y + padding_degrees,
    )


def assign_basin_ids(
    stations: gpd.GeoDataFrame,
    local_basins: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    valid = stations.loc[stations.geometry.notna()].copy()
    joined = gpd.sjoin(
        valid,
        local_basins[["HYBAS_ID", "geometry"]],
        how="left",
        predicate="within",
    )
    joined = joined.loc[~joined.index.duplicated(keep="first")]
    missing_indices = joined.index[joined["HYBAS_ID"].isna()]
    if len(missing_indices):
        boundary_matches = gpd.sjoin(
            valid.loc[missing_indices],
            local_basins[["HYBAS_ID", "geometry"]],
            how="left",
            predicate="intersects",
        )
        boundary_matches = boundary_matches.loc[
            ~boundary_matches.index.duplicated(keep="first")
        ]
        joined.loc[boundary_matches.index, "HYBAS_ID"] = boundary_matches["HYBAS_ID"]
    output = stations.copy()
    output["basin_id"] = pd.Series(pd.NA, index=output.index, dtype="Int64")
    output.loc[joined.index, "basin_id"] = joined["HYBAS_ID"].astype("Int64")
    return output


def assign_nearest_river_reaches(
    stations: gpd.GeoDataFrame,
    local_rivers: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    valid = stations.loc[stations.geometry.notna()].to_crs(HUE_PROJECTED_CRS)
    rivers_projected = local_rivers.to_crs(HUE_PROJECTED_CRS)
    nearest = gpd.sjoin_nearest(
        valid,
        rivers_projected[["HYRIV_ID", "geometry"]],
        how="left",
        distance_col="river_snap_distance_m",
    )
    nearest = nearest.loc[~nearest.index.duplicated(keep="first")]
    output = stations.copy()
    output["river_reach_id"] = pd.Series(pd.NA, index=output.index, dtype="Int64")
    output["river_snap_distance_m"] = pd.Series(
        float("nan"), index=output.index, dtype="float64"
    )
    output.loc[nearest.index, "river_reach_id"] = nearest["HYRIV_ID"].astype("Int64")
    output.loc[nearest.index, "river_snap_distance_m"] = nearest[
        "river_snap_distance_m"
    ]
    return output


def upstream_ids(
    identifiers: Iterable[int],
    next_downstream: Iterable[int],
    target_identifier: int,
) -> set[int]:
    direct_upstream: dict[int, list[int]] = defaultdict(list)
    for identifier, downstream_identifier in zip(identifiers, next_downstream):
        direct_upstream[int(downstream_identifier)].append(int(identifier))
    visited = {int(target_identifier)}
    queue = deque([int(target_identifier)])
    while queue:
        current = queue.popleft()
        for upstream_identifier in direct_upstream.get(current, []):
            if upstream_identifier not in visited:
                visited.add(upstream_identifier)
                queue.append(upstream_identifier)
    return visited


def downstream_ids(
    identifiers: Iterable[int],
    next_downstream: Iterable[int],
    target_identifier: int,
) -> set[int]:
    downstream_lookup = {
        int(identifier): int(downstream_identifier)
        for identifier, downstream_identifier in zip(identifiers, next_downstream)
    }
    visited: set[int] = set()
    current = int(target_identifier)
    while current in downstream_lookup:
        downstream_identifier = downstream_lookup[current]
        if downstream_identifier == 0 or downstream_identifier in visited:
            break
        visited.add(downstream_identifier)
        current = downstream_identifier
    return visited


def classify_station_relations(
    stations: gpd.GeoDataFrame,
    da_vien_index: int,
    upstream_basin_ids: set[int],
    downstream_basin_ids: set[int],
    upstream_reach_ids: set[int],
    downstream_reach_ids: set[int],
    maximum_snap_distance_m: float,
) -> gpd.GeoDataFrame:
    target_basin_id = int(stations.loc[da_vien_index, "basin_id"])
    target_reach_id = int(stations.loc[da_vien_index, "river_reach_id"])
    relations: list[str] = []
    for station_index, station in stations.iterrows():
        if station_index == da_vien_index:
            relations.append("same_catchment")
            continue
        basin_id = int(station["basin_id"]) if pd.notna(station["basin_id"]) else None
        reach_id = (
            int(station["river_reach_id"])
            if pd.notna(station["river_reach_id"])
            else None
        )
        reliable_reach_snap = (
            reach_id is not None
            and pd.notna(station["river_snap_distance_m"])
            and station["river_snap_distance_m"] <= maximum_snap_distance_m
        )
        if basin_id == target_basin_id:
            if reliable_reach_snap and reach_id in upstream_reach_ids - {target_reach_id}:
                relations.append("upstream")
            elif reliable_reach_snap and reach_id in downstream_reach_ids:
                relations.append("downstream")
            else:
                relations.append("same_catchment")
        elif basin_id in upstream_basin_ids:
            relations.append("upstream")
        elif basin_id in downstream_basin_ids:
            relations.append("downstream")
        else:
            relations.append("outside_catchment")
    output = stations.copy()
    output["relation_to_da_vien"] = relations
    return output


def plot_topology_map(
    paths: AnalysisPaths,
    stations: gpd.GeoDataFrame,
    upstream_basins: gpd.GeoDataFrame,
    local_rivers: gpd.GeoDataFrame,
    upstream_rivers: gpd.GeoDataFrame,
    da_vien_index: int,
) -> None:
    catchment = upstream_basins.dissolve()
    minimum_x, minimum_y, maximum_x, maximum_y = catchment.total_bounds
    x_padding = max((maximum_x - minimum_x) * 0.08, 0.03)
    y_padding = max((maximum_y - minimum_y) * 0.08, 0.03)

    figure, axis = plt.subplots(figsize=(13, 12))
    catchment.plot(
        ax=axis,
        facecolor="#dceef8",
        edgecolor="#176b87",
        linewidth=1.5,
        label="Dã Viên upstream catchment",
    )
    upstream_basins.boundary.plot(ax=axis, color="#73a9c2", linewidth=0.45, alpha=0.65)
    local_rivers.plot(ax=axis, color="#a9b7c0", linewidth=0.45, alpha=0.55)
    upstream_rivers.plot(ax=axis, color="#176bba", linewidth=1.0, alpha=0.9)

    styles = {
        "rain": ("^", "#1f77b4", 42),
        "water_level": ("s", "#d62728", 46),
        "reservoir": ("D", "#7b3294", 46),
        "wind": ("x", "#2ca02c", 42),
    }
    for station_type, type_stations in stations.loc[stations.geometry.notna()].groupby("type"):
        marker, color, size = styles[station_type]
        plot_options = {
            "ax": axis,
            "marker": marker,
            "color": color,
            "markersize": size,
            "linewidth": 0.9,
            "label": station_type,
            "zorder": 4,
        }
        if marker != "x":
            plot_options["edgecolor"] = "white"
        type_stations.plot(
            **plot_options,
        )
    da_vien = stations.loc[[da_vien_index]]
    da_vien.plot(
        ax=axis,
        marker="*",
        color="black",
        edgecolor="white",
        linewidth=0.8,
        markersize=220,
        label="Dã Viên",
        zorder=6,
    )
    axis.annotate(
        "Dã Viên",
        (da_vien.geometry.iloc[0].x, da_vien.geometry.iloc[0].y),
        xytext=(7, 7),
        textcoords="offset points",
        fontsize=9,
        fontweight="bold",
    )
    axis.set_xlim(minimum_x - x_padding, maximum_x + x_padding)
    axis.set_ylim(minimum_y - y_padding, maximum_y + y_padding)
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.set_title("Dã Viên upstream catchment from HydroBASINS/HydroRIVERS topology")
    axis.legend(loc="best")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(paths.figures / "da_vien_upstream_catchment.png", dpi=180)
    plt.close(figure)


def candidate_model_inputs(classified_stations: gpd.GeoDataFrame) -> pd.DataFrame:
    relation = classified_stations["relation_to_da_vien"]
    station_type = classified_stations["type"]
    is_target_water_level = (
        (station_type == "water_level")
        & (classified_stations["station_name"] == DA_VIEN_NAME)
    )
    candidates = classified_stations.loc[
        ((station_type == "rain") & relation.isin(["upstream", "same_catchment"]))
        | ((station_type == "water_level") & (relation == "upstream"))
        | ((station_type == "reservoir") & (relation == "upstream"))
        | ((station_type == "wind") & relation.isin(["upstream", "same_catchment"]))
        | is_target_water_level
    ].copy()
    candidates["candidate_reason"] = "topology_upstream"
    candidates.loc[
        (candidates["type"] == "rain")
        & (candidates["relation_to_da_vien"] == "same_catchment"),
        "candidate_reason",
    ] = "rain_inside_target_level12_basin"
    candidates.loc[is_target_water_level, "candidate_reason"] = "autoregressive_target_lags"
    return candidates.sort_values(["type", "relation_to_da_vien", "station_name"])


def run(
    paths: AnalysisPaths,
    basin_level: int = DEFAULT_BASIN_LEVEL,
    maximum_snap_distance_m: float = DEFAULT_MAXIMUM_RIVER_SNAP_METERS,
) -> dict[str, object]:
    paths.create_output_directories()
    hydrography_paths = discover_hydrography_paths(paths.data, basin_level)
    stations = station_geodataframe(paths)
    local_bounds = expanded_bounds(stations)
    local_basins = gpd.read_file(hydrography_paths.basin_path, bbox=local_bounds)
    local_rivers = gpd.read_file(
        hydrography_paths.river_gdb_path,
        layer=HYDRORIVERS_LAYER,
        bbox=local_bounds,
    )
    stations = assign_basin_ids(stations, local_basins)
    stations = assign_nearest_river_reaches(stations, local_rivers)

    da_vien_matches = stations.index[
        (stations["type"] == "water_level")
        & (stations["station_name"] == DA_VIEN_NAME)
    ]
    if len(da_vien_matches) != 1:
        raise ValueError(f"Expected one Dã Viên water-level station, found {len(da_vien_matches)}")
    da_vien_index = int(da_vien_matches[0])
    da_vien_basin_id = int(stations.loc[da_vien_index, "basin_id"])
    da_vien_reach_id = int(stations.loc[da_vien_index, "river_reach_id"])

    target_basin = local_basins.loc[local_basins["HYBAS_ID"] == da_vien_basin_id]
    target_reach = local_rivers.loc[local_rivers["HYRIV_ID"] == da_vien_reach_id]
    if target_basin.empty or target_reach.empty:
        raise ValueError("Could not resolve Dã Viên basin or river topology")
    main_basin_id = int(target_basin.iloc[0]["MAIN_BAS"])
    main_river_id = int(target_reach.iloc[0]["MAIN_RIV"])
    basin_topology = gpd.read_file(
        hydrography_paths.basin_path,
        where=f"MAIN_BAS = {main_basin_id}",
    )
    river_topology = gpd.read_file(
        hydrography_paths.river_gdb_path,
        layer=HYDRORIVERS_LAYER,
        where=f"MAIN_RIV = {main_river_id}",
    )

    upstream_basin_identifiers = upstream_ids(
        basin_topology["HYBAS_ID"], basin_topology["NEXT_DOWN"], da_vien_basin_id
    )
    downstream_basin_identifiers = downstream_ids(
        basin_topology["HYBAS_ID"], basin_topology["NEXT_DOWN"], da_vien_basin_id
    )
    upstream_reach_identifiers = upstream_ids(
        river_topology["HYRIV_ID"], river_topology["NEXT_DOWN"], da_vien_reach_id
    )
    downstream_reach_identifiers = downstream_ids(
        river_topology["HYRIV_ID"], river_topology["NEXT_DOWN"], da_vien_reach_id
    )
    classified = classify_station_relations(
        stations,
        da_vien_index,
        upstream_basin_identifiers,
        downstream_basin_identifiers,
        upstream_reach_identifiers,
        downstream_reach_identifiers,
        maximum_snap_distance_m,
    )

    output_columns = [
        "station_id",
        "station_name",
        "type",
        "basin_id",
        "river_reach_id",
        "relation_to_da_vien",
        "river_snap_distance_m",
        "coordinate_source",
    ]
    classified[output_columns].to_csv(
        paths.reports / "station_hydrologic_relations.csv", index=False
    )
    candidates = candidate_model_inputs(classified)
    candidates[[*output_columns, "candidate_reason"]].to_csv(
        paths.reports / "da_vien_candidate_input_stations.csv", index=False
    )

    upstream_basins = basin_topology.loc[
        basin_topology["HYBAS_ID"].isin(upstream_basin_identifiers)
    ]
    upstream_rivers = river_topology.loc[
        river_topology["HYRIV_ID"].isin(upstream_reach_identifiers)
    ]
    upstream_basins.to_file(
        paths.reports / "da_vien_upstream_basins.geojson", driver="GeoJSON"
    )
    upstream_rivers.to_file(
        paths.reports / "da_vien_upstream_rivers.geojson", driver="GeoJSON"
    )
    plot_topology_map(
        paths,
        classified,
        upstream_basins,
        local_rivers,
        upstream_rivers,
        da_vien_index,
    )

    upstream_counts = (
        classified.loc[classified["relation_to_da_vien"] == "upstream"]
        .groupby("type")
        .size()
        .to_dict()
    )
    stations_in_upstream_catchment = classified.loc[
        classified["basin_id"].isin(upstream_basin_identifiers)
    ]
    rain_stations_in_upstream_catchment = int(
        (stations_in_upstream_catchment["type"] == "rain").sum()
    )
    summary = {
        "hydrobasins_level": basin_level,
        "da_vien_basin_id": da_vien_basin_id,
        "da_vien_river_reach_id": da_vien_reach_id,
        "da_vien_river_snap_distance_m": float(
            classified.loc[da_vien_index, "river_snap_distance_m"]
        ),
        "upstream_basin_count_including_target": len(upstream_basin_identifiers),
        "upstream_polygon_sub_area_sum_km2": float(upstream_basins["SUB_AREA"].sum()),
        "da_vien_hydrobasins_reported_up_area_km2": float(
            target_basin.iloc[0]["UP_AREA"]
        ),
        "upstream_river_reach_count_including_target": len(upstream_reach_identifiers),
        "rain_stations_in_upstream_catchment": rain_stations_in_upstream_catchment,
        "strictly_upstream_rain_stations": int(upstream_counts.get("rain", 0)),
        "upstream_water_level_stations": int(upstream_counts.get("water_level", 0)),
        "upstream_reservoirs": int(upstream_counts.get("reservoir", 0)),
        "maximum_river_snap_distance_used_for_same_basin_classification_m": maximum_snap_distance_m,
        "classification_rule": (
            "upstream/downstream follows NEXT_DOWN topology; same_catchment means the "
            "same HydroBASINS level-12 unit where reach topology cannot resolve a direction"
        ),
        "candidate_input_stations": candidates[
            [
                "station_id",
                "station_name",
                "type",
                "relation_to_da_vien",
                "candidate_reason",
            ]
        ].to_dict(orient="records"),
    }
    write_json(summary, paths.reports / "da_vien_topology_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=AnalysisPaths().data)
    parser.add_argument("--reports-dir", type=Path, default=AnalysisPaths().reports)
    parser.add_argument("--figures-dir", type=Path, default=AnalysisPaths().figures)
    parser.add_argument("--basin-level", type=int, default=DEFAULT_BASIN_LEVEL)
    parser.add_argument(
        "--maximum-river-snap-meters",
        type=float,
        default=DEFAULT_MAXIMUM_RIVER_SNAP_METERS,
    )
    args = parser.parse_args()
    summary = run(
        AnalysisPaths(args.data_dir, args.reports_dir, args.figures_dir),
        basin_level=args.basin_level,
        maximum_snap_distance_m=args.maximum_river_snap_meters,
    )
    print(f"Dã Viên basin ID: {summary['da_vien_basin_id']}")
    print(f"Dã Viên HydroRIVERS reach ID: {summary['da_vien_river_reach_id']}")
    print(
        "Rain stations inside upstream catchment: "
        f"{summary['rain_stations_in_upstream_catchment']}"
    )
    print(
        "Strictly upstream rain stations: "
        f"{summary['strictly_upstream_rain_stations']}"
    )
    print(f"Upstream water-level stations: {summary['upstream_water_level_stations']}")
    print(f"Upstream reservoirs: {summary['upstream_reservoirs']}")
    print("Candidate model input stations:")
    for station in summary["candidate_input_stations"]:
        print(
            f"- [{station['type']}] {station['station_name']} "
            f"({station['station_id']}): {station['relation_to_da_vien']}"
        )


if __name__ == "__main__":
    main()
