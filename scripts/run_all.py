"""Run the complete reproducible hydrology audit in dependency order."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from analysis_config import AnalysisPaths
from analysis_utils import resolve_csv_path
from analyze_hydrologic_signal import run as run_hydrologic_signal
from analyze_spatial_relationships import run as run_spatial_analysis
from analyze_temporal_quality import run as run_temporal_analysis
from build_training_dataset import build_training_table
from evaluate_training_feasibility import run as run_feasibility_analysis
from inspect_datasets import run as run_dataset_inspection
from plot_observation_counts import run as run_observation_count_analysis
from run_baselines import run as run_baseline_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=AnalysisPaths().data)
    parser.add_argument("--reports-dir", type=Path, default=AnalysisPaths().reports)
    parser.add_argument("--figures-dir", type=Path, default=AnalysisPaths().figures)
    args = parser.parse_args()
    paths = AnalysisPaths(args.data_dir, args.reports_dir, args.figures_dir)
    paths.create_output_directories()
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "hydrology-matplotlib"))

    try:
        resolve_csv_path(paths.data / "hydrology_training_10min.csv")
    except FileNotFoundError:
        training_output = paths.reports / "hydrology_training_10min.csv"
        print(f"[Building derived training table: {training_output}]")
        build_training_table(paths.data).to_csv(training_output, index=False)

    stages = (
        ("Inspecting schemas and distributions", run_dataset_inspection),
        ("Counting observations by station", run_observation_count_analysis),
        ("Analyzing temporal quality", run_temporal_analysis),
        ("Analyzing spatial proximity", run_spatial_analysis),
        ("Analyzing hydrologic signals and events", run_hydrologic_signal),
        ("Running temporal baselines", run_baseline_experiments),
        ("Auditing leakage and writing final report", run_feasibility_analysis),
    )
    for message, stage in stages:
        print(f"[{message}]")
        stage(paths)
    print(f"Report: {paths.reports / 'data_analysis_report.md'}")


if __name__ == "__main__":
    main()
