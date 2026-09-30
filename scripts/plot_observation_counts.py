"""Count and plot raw observations for every sensor station and reservoir."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from analysis_config import AnalysisPaths
from analysis_utils import read_csv


@dataclass(frozen=True)
class StationCountSource:
    station_type: str
    metadata_filename: str
    observations_filename: str
    metadata_id_column: str
    observation_id_column: str
    station_name_column: str


COUNT_SOURCES = (
    StationCountSource(
        station_type="rain",
        metadata_filename="rain_stations.csv",
        observations_filename="rain_observations.csv",
        metadata_id_column="code",
        observation_id_column="station_id",
        station_name_column="name",
    ),
    StationCountSource(
        station_type="water_level",
        metadata_filename="water_level_stations.csv",
        observations_filename="water_level_observations.csv",
        metadata_id_column="code",
        observation_id_column="station_id",
        station_name_column="name",
    ),
    StationCountSource(
        station_type="wind",
        metadata_filename="wind_stations.csv",
        observations_filename="wind_observations.csv",
        metadata_id_column="id",
        observation_id_column="sid",
        station_name_column="name",
    ),
    StationCountSource(
        station_type="reservoir",
        metadata_filename="reservoirs.csv",
        observations_filename="reservoir_observations.csv",
        metadata_id_column="id",
        observation_id_column="tencongtrinh",
        station_name_column="tencongtrinh",
    ),
)


def count_observations_for_source(
    paths: AnalysisPaths,
    source: StationCountSource,
) -> pd.DataFrame:
    """Return every metadata station, including stations with no observations."""
    stations = read_csv(paths.data / source.metadata_filename)
    observations = read_csv(paths.data / source.observations_filename)

    station_catalog = stations[
        [source.metadata_id_column, source.station_name_column]
    ].rename(
        columns={
            source.metadata_id_column: "station_id",
            source.station_name_column: "station_name",
        }
    )
    observation_counts = (
        observations.groupby(source.observation_id_column, dropna=False)
        .size()
        .rename("observation_count")
        .reset_index()
        .rename(columns={source.observation_id_column: "station_id"})
    )
    counts = station_catalog.merge(
        observation_counts,
        on="station_id",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    counts["station_name"] = counts["station_name"].fillna(
        "Không có trong station metadata"
    )
    counts["observation_count"] = (
        counts["observation_count"].fillna(0).astype("int64")
    )
    counts["station_type"] = source.station_type
    counts["metadata_status"] = counts["_merge"].map(
        {
            "left_only": "metadata_only_no_observations",
            "right_only": "observations_without_metadata",
            "both": "matched",
        }
    )
    return counts.drop(columns="_merge")[
        [
            "station_type",
            "station_id",
            "station_name",
            "observation_count",
            "metadata_status",
        ]
    ]


def plot_station_counts(
    paths: AnalysisPaths,
    station_type: str,
    counts: pd.DataFrame,
) -> None:
    sorted_counts = counts.sort_values(
        ["observation_count", "station_name"], ascending=[True, True]
    ).reset_index(drop=True)
    figure_height = max(4.5, 0.28 * len(sorted_counts) + 1.8)
    figure, axis = plt.subplots(figsize=(12, figure_height))
    colors = [
        "#2673a2" if count > 0 else "#c8c8c8"
        for count in sorted_counts["observation_count"]
    ]
    bars = axis.barh(
        sorted_counts["station_name"],
        sorted_counts["observation_count"],
        color=colors,
    )
    positive_counts = sorted_counts.loc[
        sorted_counts["observation_count"] > 0, "observation_count"
    ]
    annotation_offset = max(float(positive_counts.max()) * 0.008, 1) if not positive_counts.empty else 1
    for bar, count in zip(bars, sorted_counts["observation_count"]):
        if count > 0:
            axis.text(
                bar.get_width() + annotation_offset,
                bar.get_y() + bar.get_height() / 2,
                f"{count:,}",
                va="center",
                fontsize=8,
            )
    observed_station_count = int(sorted_counts["observation_count"].gt(0).sum())
    axis.set_title(
        f"{station_type}: số quan sát theo trạm "
        f"({observed_station_count}/{len(sorted_counts)} trạm có dữ liệu)"
    )
    axis.set_xlabel("Số quan sát raw")
    axis.set_ylabel("Trạm / công trình")
    axis.grid(axis="x", alpha=0.25)
    maximum_count = int(sorted_counts["observation_count"].max())
    axis.set_xlim(0, maximum_count * 1.14 if maximum_count else 1)
    figure.tight_layout()
    figure.savefig(
        paths.figures / f"observation_counts_{station_type}.png",
        dpi=170,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_source_summary(paths: AnalysisPaths, summary: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(summary["station_type"], summary["total_observations"], color="#2673a2")
    axes[0].set_title("Tổng số quan sát raw theo loại")
    axes[0].set_ylabel("Số quan sát")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.25)

    positions = range(len(summary))
    axes[1].bar(
        positions,
        summary["total_stations"],
        label="Tổng ID/công trình sau đối chiếu",
        color="#c8c8c8",
    )
    axes[1].bar(
        positions,
        summary["stations_with_observations"],
        label="Trạm có quan sát",
        color="#3a923a",
    )
    axes[1].set_xticks(list(positions), summary["station_type"], rotation=20)
    axes[1].set_title("Mức phủ trạm")
    axes[1].set_ylabel("Số trạm / công trình")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(paths.figures / "observation_counts_summary.png", dpi=170)
    plt.close(figure)


def run(paths: AnalysisPaths) -> None:
    paths.create_output_directories()
    count_frames = [
        count_observations_for_source(paths, source) for source in COUNT_SOURCES
    ]
    all_counts = pd.concat(count_frames, ignore_index=True)
    all_counts.to_csv(
        paths.reports / "observation_counts_by_station.csv", index=False
    )

    summary = (
        all_counts.groupby("station_type", sort=False)
        .agg(
            total_stations=("station_id", "size"),
            stations_with_observations=(
                "observation_count",
                lambda values: int(values.gt(0).sum()),
            ),
            stations_without_observations=(
                "observation_count",
                lambda values: int(values.eq(0).sum()),
            ),
            total_observations=("observation_count", "sum"),
            minimum_observations=("observation_count", "min"),
            median_observations=("observation_count", "median"),
            maximum_observations=("observation_count", "max"),
        )
        .reset_index()
    )
    summary.to_csv(paths.reports / "observation_counts_summary.csv", index=False)

    for station_type, type_counts in all_counts.groupby("station_type", sort=False):
        plot_station_counts(paths, station_type, type_counts)
    plot_source_summary(paths, summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=AnalysisPaths().data)
    parser.add_argument("--reports-dir", type=Path, default=AnalysisPaths().reports)
    parser.add_argument("--figures-dir", type=Path, default=AnalysisPaths().figures)
    args = parser.parse_args()
    run(AnalysisPaths(args.data_dir, args.reports_dir, args.figures_dir))


if __name__ == "__main__":
    main()
