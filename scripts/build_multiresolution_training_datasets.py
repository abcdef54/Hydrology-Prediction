"""Build comparable 30-minute and 1-hour Dã Viên training datasets."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from analysis_config import AnalysisPaths
from analysis_utils import write_json
from build_training_dataset_v2 import (
    DEFAULT_RESERVOIR_FORWARD_FILL_MINUTES,
    TemporalResolution,
    build_report,
    build_training_dataset,
    output_companion_paths,
    validate_output_paths,
)


@dataclass(frozen=True)
class DatasetJob:
    """Output path and physical-time feature specification for one dataset."""

    output_filename: str
    temporal: TemporalResolution


DATASET_JOBS = (
    DatasetJob(
        output_filename="hydrology_training_30min.csv",
        temporal=TemporalResolution(
            bin_minutes=30,
            water_level_lag_minutes=(30, 60, 180),
            forecast_horizon_minutes=(60, 180, 360, 720),
        ),
    ),
    DatasetJob(
        output_filename="hydrology_training_1h.csv",
        temporal=TemporalResolution(
            bin_minutes=60,
            water_level_lag_minutes=(60, 180),
            forecast_horizon_minutes=(60, 180, 360, 720, 1_440),
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    default_paths = AnalysisPaths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=default_paths.data)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=default_paths.reports / "da_vien_candidate_input_stations.csv",
    )
    parser.add_argument(
        "--relations",
        type=Path,
        default=default_paths.reports / "station_hydrologic_relations.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_paths.reports,
    )
    parser.add_argument("--reservoir-timezone", default="Asia/Bangkok")
    parser.add_argument(
        "--reservoir-ffill-limit-minutes",
        type=int,
        default=DEFAULT_RESERVOIR_FORWARD_FILL_MINUTES,
        help=(
            "Maximum past-only reservoir forward-fill age in physical minutes "
            f"(default: {DEFAULT_RESERVOIR_FORWARD_FILL_MINUTES})."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing 30-minute and 1-hour derived outputs.",
    )
    return parser.parse_args()


def validate_jobs(args: argparse.Namespace) -> None:
    for job in DATASET_JOBS:
        job.temporal.validate()
        job.temporal.steps_for_minutes(args.reservoir_ffill_limit_minutes)
        output_path = args.output_dir / job.output_filename
        validate_output_paths(
            output_path,
            output_companion_paths(output_path),
            args.data_dir,
            args.force,
        )


def build_job(args: argparse.Namespace, job: DatasetJob) -> dict[str, object]:
    output_path = args.output_dir / job.output_filename
    companion_paths = output_companion_paths(output_path)
    training_table, catalog, missingness, leakage_audit = build_training_dataset(
        data_directory=args.data_dir,
        candidate_path=args.candidates,
        relation_path=args.relations,
        reservoir_timezone=args.reservoir_timezone,
        reservoir_forward_fill_minutes=args.reservoir_ffill_limit_minutes,
        temporal=job.temporal,
    )
    report_text = build_report(
        training_table,
        catalog,
        missingness,
        leakage_audit,
        args.candidates,
        args.relations,
        args.reservoir_timezone,
        args.reservoir_ffill_limit_minutes,
        job.temporal,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    training_table.to_csv(output_path, index=False)
    missingness.to_csv(companion_paths["missingness"], index=False)
    write_json(leakage_audit, companion_paths["leakage_audit"])
    companion_paths["report"].write_text(report_text, encoding="utf-8")

    target_count = len(job.temporal.forecast_horizon_minutes)
    predictor_count = len(training_table.columns) - target_count - 1
    return {
        "output": output_path,
        "report": companion_paths["report"],
        "rows": len(training_table),
        "predictors": predictor_count,
        "targets": target_count,
        "leakage_checks": leakage_audit["check_count"],
    }


def main() -> None:
    args = parse_args()
    validate_jobs(args)
    for job in DATASET_JOBS:
        summary = build_job(args, job)
        print(
            f"Wrote {summary['output']} | rows={summary['rows']:,} | "
            f"predictors={summary['predictors']:,} | "
            f"targets={summary['targets']} | "
            f"leakage_checks={summary['leakage_checks']}"
        )
        print(f"Wrote {summary['report']}")


if __name__ == "__main__":
    main()
