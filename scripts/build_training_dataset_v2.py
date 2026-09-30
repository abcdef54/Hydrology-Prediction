"""Build the topology-filtered, leakage-safe Dã Viên training dataset."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

from analysis_config import AnalysisPaths
from analysis_utils import markdown_table, parse_utc, resolve_csv_path, write_json


DA_VIEN_NAME = "Trạm Dã Viên (Sông Hương)"
EXPECTED_RAIN_STATION_COUNT = 14
EXPECTED_RESERVOIR_COUNT = 4
RAIN_ROLLING_HOURS = (1, 3, 6, 24)
WATER_LEVEL_LAG_MINUTES = (10, 30, 60, 180)
RESERVOIR_LAG_MINUTES = (60, 180)
DEFAULT_RESERVOIR_FORWARD_FILL_MINUTES = 60
FORECAST_HORIZONS_MINUTES = (60, 180, 360)
RESERVOIR_VALUE_COLUMNS = (
    "htl",
    "hhl",
    "qden",
    "qdi",
    "socuamo",
    "mucnuocsong",
    "luuluongdieutiet",
)
WIND_VALUE_COLUMNS = ("ws", "wsg", "wd", "wdg")
WIND_STATION_NAMES = ("Cảng Thuận An", "Cảng Tư Hiền")


@dataclass(frozen=True)
class TemporalResolution:
    """Physical-time configuration for one regularly binned dataset."""

    bin_minutes: int
    water_level_lag_minutes: tuple[int, ...]
    forecast_horizon_minutes: tuple[int, ...]

    @property
    def frequency(self) -> str:
        return f"{self.bin_minutes}min"

    def steps_for_minutes(self, minutes: int) -> int:
        if minutes <= 0 or minutes % self.bin_minutes != 0:
            raise ValueError(
                f"{minutes} minutes is not representable by "
                f"{self.bin_minutes}-minute bins"
            )
        return minutes // self.bin_minutes

    def steps_for_hours(self, hours: int) -> int:
        return self.steps_for_minutes(hours * 60)

    def validate(self) -> None:
        if self.bin_minutes <= 0:
            raise ValueError("bin_minutes must be positive")
        for minutes in (
            *self.water_level_lag_minutes,
            *RESERVOIR_LAG_MINUTES,
            *self.forecast_horizon_minutes,
            DEFAULT_RESERVOIR_FORWARD_FILL_MINUTES,
        ):
            self.steps_for_minutes(minutes)
        for hours in RAIN_ROLLING_HOURS:
            self.steps_for_hours(hours)


DEFAULT_TEMPORAL_RESOLUTION = TemporalResolution(
    bin_minutes=10,
    water_level_lag_minutes=WATER_LEVEL_LAG_MINUTES,
    forecast_horizon_minutes=FORECAST_HORIZONS_MINUTES,
)


@dataclass
class FeatureCatalog:
    """Track generated columns and source-station usage for reporting."""

    groups: dict[str, list[str]] = field(default_factory=dict)
    station_usage: list[dict[str, object]] = field(default_factory=list)
    reservoir_variables: dict[str, list[str]] = field(default_factory=dict)

    def add_group(self, group_name: str, columns: Iterable[str]) -> None:
        self.groups.setdefault(group_name, []).extend(columns)


def read_selected_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read only the raw columns needed by this builder."""
    return pd.read_csv(
        resolve_csv_path(path),
        usecols=columns,
        encoding="utf-8-sig",
        low_memory=False,
    )


def safe_identifier(identifier: object) -> str:
    """Return a stable identifier that is safe to embed in a column name."""
    return re.sub(r"[^0-9A-Za-z]+", "_", str(identifier)).strip("_")


def require_columns(frame: pd.DataFrame, required: Iterable[str], source: Path) -> None:
    missing_columns = sorted(set(required).difference(frame.columns))
    if missing_columns:
        raise ValueError(f"{source} is missing required columns: {missing_columns}")


def load_topology_candidates(
    candidate_path: Path,
    relation_path: Path,
) -> pd.DataFrame:
    """Load and validate the topology-approved input station list."""
    candidates = pd.read_csv(
        candidate_path,
        dtype={"station_id": "string"},
        encoding="utf-8-sig",
    )
    require_columns(
        candidates,
        (
            "station_id",
            "station_name",
            "type",
            "relation_to_da_vien",
            "candidate_reason",
        ),
        candidate_path,
    )
    candidates["station_id"] = candidates["station_id"].str.strip()

    rain_candidates = candidates.loc[candidates["type"].eq("rain")]
    reservoir_candidates = candidates.loc[candidates["type"].eq("reservoir")]
    da_vien_candidates = candidates.loc[
        candidates["type"].eq("water_level")
        & candidates["station_name"].eq(DA_VIEN_NAME)
    ]

    if len(rain_candidates) != EXPECTED_RAIN_STATION_COUNT:
        raise ValueError(
            "Topology candidate validation failed: expected "
            f"{EXPECTED_RAIN_STATION_COUNT} rain stations, found {len(rain_candidates)}"
        )
    if len(reservoir_candidates) != EXPECTED_RESERVOIR_COUNT:
        raise ValueError(
            "Topology candidate validation failed: expected "
            f"{EXPECTED_RESERVOIR_COUNT} reservoirs, found {len(reservoir_candidates)}"
        )
    if len(da_vien_candidates) != 1:
        raise ValueError(
            f"Expected exactly one {DA_VIEN_NAME!r} candidate, found {len(da_vien_candidates)}"
        )
    if candidates["station_id"].isna().any():
        raise ValueError("Topology candidate file contains a missing station_id")
    if candidates.duplicated(["type", "station_id"]).any():
        raise ValueError("Topology candidate file contains duplicate type/station_id pairs")

    relations = pd.read_csv(
        relation_path,
        dtype={"station_id": "string"},
        encoding="utf-8-sig",
    )
    require_columns(
        relations,
        ("station_id", "type", "relation_to_da_vien"),
        relation_path,
    )
    relations["station_id"] = relations["station_id"].str.strip()
    if relations.duplicated(["type", "station_id"]).any():
        raise ValueError(
            "Hydrologic relation file contains duplicate type/station_id pairs"
        )

    topology_check = candidates.merge(
        relations[["type", "station_id", "relation_to_da_vien"]].rename(
            columns={"relation_to_da_vien": "relation_from_full_topology"}
        ),
        on=["type", "station_id"],
        how="left",
        validate="one_to_one",
    )
    missing_relations = topology_check["relation_from_full_topology"].isna()
    if missing_relations.any():
        missing_keys = topology_check.loc[
            missing_relations,
            ["type", "station_id"],
        ].to_dict(orient="records")
        raise ValueError(
            "Candidate stations missing from station_hydrologic_relations.csv: "
            f"{missing_keys}"
        )
    relation_mismatches = topology_check[
        "relation_to_da_vien"
    ].ne(topology_check["relation_from_full_topology"])
    if relation_mismatches.any():
        mismatched_rows = topology_check.loc[
            relation_mismatches,
            [
                "type",
                "station_id",
                "relation_to_da_vien",
                "relation_from_full_topology",
            ],
        ].to_dict(orient="records")
        raise ValueError(f"Topology relation mismatch: {mismatched_rows}")

    invalid_rain_relations = rain_candidates.loc[
        ~rain_candidates["relation_to_da_vien"].isin(
            {"upstream", "same_catchment"}
        )
    ]
    if not invalid_rain_relations.empty:
        raise ValueError(
            "Rain candidates must be upstream or in the Dã Viên level-12 "
            "catchment"
        )
    if not reservoir_candidates["relation_to_da_vien"].eq("upstream").all():
        raise ValueError("Every reservoir candidate must be topology-upstream")

    return candidates


def load_requested_wind_stations(
    data_directory: Path,
    relation_path: Path,
) -> pd.DataFrame:
    """Load the two wind stations explicitly requested for the Dã Viên model."""
    station_path = data_directory / "wind_stations.csv"
    stations = read_selected_columns(station_path, ["id", "name"])
    stations["id"] = stations["id"].astype("string").str.strip()
    selected_stations = stations.loc[
        stations["name"].isin(WIND_STATION_NAMES),
        ["id", "name"],
    ].rename(columns={"id": "station_id", "name": "station_name"})

    selected_names = set(selected_stations["station_name"])
    missing_names = sorted(set(WIND_STATION_NAMES).difference(selected_names))
    if missing_names:
        raise ValueError(f"Requested wind stations not found: {missing_names}")
    if len(selected_stations) != len(WIND_STATION_NAMES):
        raise ValueError("Requested wind station names must each resolve exactly once")

    relations = pd.read_csv(
        relation_path,
        dtype={"station_id": "string"},
        encoding="utf-8-sig",
    )
    wind_relations = relations.loc[
        relations["type"].eq("wind"),
        ["station_id", "relation_to_da_vien"],
    ]
    selected_stations = selected_stations.merge(
        wind_relations,
        on="station_id",
        how="left",
        validate="one_to_one",
    )
    if selected_stations["relation_to_da_vien"].isna().any():
        raise ValueError("Requested wind station is missing from topology relations")
    return selected_stations


def aggregate_station_values(
    observations: pd.DataFrame,
    station_id_column: str,
    value_column: str,
    aggregation: str,
) -> pd.DataFrame:
    """Aggregate station observations into precomputed UTC bins."""
    valid_observations = observations.dropna(
        subset=["timestamp_bin", station_id_column]
    )
    grouped_values = valid_observations.groupby(
        ["timestamp_bin", station_id_column],
        sort=True,
    )[value_column]

    if aggregation == "sum":
        aggregated = grouped_values.sum(min_count=1)
    elif aggregation == "mean":
        aggregated = grouped_values.mean()
    elif aggregation == "last":
        aggregated = grouped_values.last()
    else:
        raise ValueError(f"Unsupported aggregation: {aggregation}")

    return aggregated.unstack(station_id_column)


def load_da_vien_water_level(
    data_directory: Path,
    da_vien_candidate: pd.Series,
    temporal: TemporalResolution,
) -> tuple[pd.Series, int]:
    """Return the observed Dã Viên water level on a regular UTC timeline."""
    source_path = data_directory / "water_level_observations.csv"
    observations = read_selected_columns(
        source_path,
        ["station_id", "depth", "time_point"],
    )
    observations["station_id"] = observations["station_id"].astype("string")
    station_observations = observations.loc[
        observations["station_id"].eq(da_vien_candidate["station_id"])
    ].copy()
    if station_observations.empty:
        raise ValueError(
            f"No water-level observations found for {da_vien_candidate['station_name']}"
        )

    station_observations["timestamp_bin"] = (
        parse_utc(station_observations["time_point"]).dt.floor(temporal.frequency)
    )
    station_observations["depth"] = pd.to_numeric(
        station_observations["depth"],
        errors="coerce",
    )
    binned_values = (
        station_observations.dropna(subset=["timestamp_bin"])
        .groupby("timestamp_bin")["depth"]
        .mean()
        .sort_index()
    )
    if binned_values.empty:
        raise ValueError("Dã Viên has no valid timestamped water-level values")

    timeline = pd.date_range(
        start=binned_values.index.min(),
        end=binned_values.index.max(),
        freq=temporal.frequency,
        tz="UTC",
        name="timestamp_utc",
    )
    return binned_values.reindex(timeline).rename("water_level_depth"), len(
        station_observations
    )


def build_water_level_features(
    water_level: pd.Series,
    da_vien_candidate: pd.Series,
    raw_observation_count: int,
    temporal: TemporalResolution,
    catalog: FeatureCatalog,
) -> pd.DataFrame:
    water_features = pd.DataFrame(index=water_level.index)
    water_features["water_level_depth"] = water_level

    lag_columns: list[str] = []
    for lag_minutes in temporal.water_level_lag_minutes:
        lag_column = f"water_level_lag_{lag_minutes}m"
        water_features[lag_column] = water_level.shift(
            temporal.steps_for_minutes(lag_minutes)
        )
        lag_columns.append(lag_column)

    missing_column = "water_level_missing"
    water_features[missing_column] = water_level.isna().astype("int8")
    catalog.add_group("water_level_current", ["water_level_depth"])
    catalog.add_group("water_level_lags", lag_columns)
    catalog.add_group("missing_flags", [missing_column])
    catalog.station_usage.append(
        {
            "type": "water_level",
            "station_id": da_vien_candidate["station_id"],
            "station_name": da_vien_candidate["station_name"],
            "relation_to_da_vien": da_vien_candidate["relation_to_da_vien"],
            "raw_observations_selected": raw_observation_count,
            "observed_bins_in_timeline": int(water_level.notna().sum()),
            "source_variables": "depth",
            "predictor_columns": len(water_features.columns),
        }
    )
    return water_features


def build_rain_features(
    data_directory: Path,
    rain_candidates: pd.DataFrame,
    timeline: pd.DatetimeIndex,
    temporal: TemporalResolution,
    catalog: FeatureCatalog,
) -> pd.DataFrame:
    source_path = data_directory / "rain_observations.csv"
    observations = read_selected_columns(
        source_path,
        ["station_id", "depth", "time_point"],
    )
    observations["station_id"] = observations["station_id"].astype("string")
    candidate_ids = rain_candidates["station_id"].tolist()
    observations = observations.loc[
        observations["station_id"].isin(candidate_ids)
    ].copy()
    observations["timestamp_bin"] = (
        parse_utc(observations["time_point"]).dt.floor(temporal.frequency)
    )
    observations["depth"] = pd.to_numeric(observations["depth"], errors="coerce")

    station_matrix = aggregate_station_values(
        observations,
        station_id_column="station_id",
        value_column="depth",
        aggregation="sum",
    ).reindex(index=timeline, columns=candidate_ids)

    rain_features = pd.DataFrame(index=timeline)
    current_columns: list[str] = []
    rolling_columns: list[str] = []
    missing_columns: list[str] = []

    for candidate in rain_candidates.itertuples(index=False):
        station_id = str(candidate.station_id)
        station_key = safe_identifier(station_id)
        current_column = f"rain_{station_key}_depth"
        current_values = station_matrix[station_id]
        rain_features[current_column] = current_values
        current_columns.append(current_column)

        for rolling_hours in RAIN_ROLLING_HOURS:
            window_steps = temporal.steps_for_hours(rolling_hours)
            rolling_column = f"rain_{station_key}_sum_{rolling_hours}h"
            rain_features[rolling_column] = current_values.rolling(
                window=window_steps,
                min_periods=window_steps,
            ).sum()
            rolling_columns.append(rolling_column)

        missing_column = f"rain_{station_key}_missing"
        rain_features[missing_column] = current_values.isna().astype("int8")
        missing_columns.append(missing_column)
        raw_station_count = int(observations["station_id"].eq(station_id).sum())
        if raw_station_count == 0:
            raise ValueError(f"Rain candidate {station_id} has no observations")
        catalog.station_usage.append(
            {
                "type": "rain",
                "station_id": station_id,
                "station_name": candidate.station_name,
                "relation_to_da_vien": candidate.relation_to_da_vien,
                "raw_observations_selected": raw_station_count,
                "observed_bins_in_timeline": int(
                    current_values.notna().sum()
                ),
                "source_variables": "depth",
                "predictor_columns": 1 + len(RAIN_ROLLING_HOURS) + 1,
            }
        )

    catalog.add_group("rain_current", current_columns)
    catalog.add_group("rain_rolling", rolling_columns)
    catalog.add_group("missing_flags", missing_columns)
    return rain_features


def parse_reservoir_timestamps(
    observations: pd.DataFrame,
    reservoir_timezone: str,
    temporal: TemporalResolution,
) -> pd.Series:
    """Parse timezone-naive reservoir timestamps using an explicit assumption."""
    local_timestamp_text = (
        observations["ngaylaysolieu"].astype("string").str.strip()
        + " "
        + observations["gio"].astype("string").str.strip()
    )
    local_timestamps = pd.to_datetime(
        local_timestamp_text,
        format="%d/%m/%Y %H:%M",
        errors="coerce",
    )
    return (
        local_timestamps.dt.tz_localize(
            reservoir_timezone,
            ambiguous="NaT",
            nonexistent="NaT",
        )
        .dt.tz_convert("UTC")
        .dt.floor(temporal.frequency)
    )


def build_reservoir_features(
    data_directory: Path,
    reservoir_candidates: pd.DataFrame,
    timeline: pd.DatetimeIndex,
    reservoir_timezone: str,
    forward_fill_minutes: int,
    temporal: TemporalResolution,
    catalog: FeatureCatalog,
) -> pd.DataFrame:
    source_path = data_directory / "reservoir_observations.csv"
    required_columns = [
        "tencongtrinh",
        "ngaylaysolieu",
        "gio",
        *RESERVOIR_VALUE_COLUMNS,
    ]
    observations = read_selected_columns(source_path, required_columns)
    observations["tencongtrinh"] = observations["tencongtrinh"].astype("string")
    candidate_ids = reservoir_candidates["station_id"].tolist()
    observations = observations.loc[
        observations["tencongtrinh"].isin(candidate_ids)
    ].copy()
    observations["timestamp_bin"] = parse_reservoir_timestamps(
        observations,
        reservoir_timezone,
        temporal,
    )
    for value_column in RESERVOIR_VALUE_COLUMNS:
        observations[value_column] = pd.to_numeric(
            observations[value_column],
            errors="coerce",
        )

    reservoir_features = pd.DataFrame(index=timeline)
    current_columns: list[str] = []
    lag_columns: list[str] = []
    missing_columns: list[str] = []
    forward_fill_flag_columns: list[str] = []
    forward_fill_steps = temporal.steps_for_minutes(forward_fill_minutes)

    for candidate in reservoir_candidates.itertuples(index=False):
        station_id = str(candidate.station_id)
        station_key = safe_identifier(station_id)
        station_observations = observations.loc[
            observations["tencongtrinh"].eq(station_id)
        ].copy()
        if station_observations.empty:
            raise ValueError(f"Reservoir candidate {station_id} has no observations")

        binned_values_by_variable: dict[str, pd.Series] = {}
        for value_column in RESERVOIR_VALUE_COLUMNS:
            binned_values = (
                station_observations.dropna(subset=["timestamp_bin"])
                .groupby("timestamp_bin")[value_column]
                .mean()
                .reindex(timeline)
            )
            if binned_values.notna().any():
                binned_values_by_variable[value_column] = binned_values

        usable_variables = list(binned_values_by_variable)
        if not usable_variables:
            raise ValueError(
                f"Reservoir candidate {station_id} has no values on the target timeline"
            )
        catalog.reservoir_variables[station_id] = usable_variables
        generated_column_count = 0

        for value_column in usable_variables:
            current_column = f"reservoir_{station_key}_{value_column}"
            observed_values = binned_values_by_variable[value_column]
            filled_values = observed_values.ffill(limit=forward_fill_steps)
            reservoir_features[current_column] = filled_values
            current_columns.append(current_column)
            generated_column_count += 1

            for lag_minutes in RESERVOIR_LAG_MINUTES:
                lag_column = f"{current_column}_lag_{lag_minutes}m"
                reservoir_features[lag_column] = filled_values.shift(
                    temporal.steps_for_minutes(lag_minutes)
                )
                lag_columns.append(lag_column)
                generated_column_count += 1

            missing_column = f"{current_column}_source_missing"
            reservoir_features[missing_column] = (
                observed_values.isna().astype("int8")
            )
            missing_columns.append(missing_column)
            generated_column_count += 1

            forward_fill_flag_column = f"{current_column}_forward_filled"
            reservoir_features[forward_fill_flag_column] = (
                observed_values.isna() & filled_values.notna()
            ).astype("int8")
            forward_fill_flag_columns.append(forward_fill_flag_column)
            generated_column_count += 1

        catalog.station_usage.append(
            {
                "type": "reservoir",
                "station_id": station_id,
                "station_name": candidate.station_name,
                "relation_to_da_vien": candidate.relation_to_da_vien,
                "raw_observations_selected": len(station_observations),
                "observed_bins_in_timeline": int(
                    pd.DataFrame(binned_values_by_variable).notna().any(axis=1).sum()
                ),
                "source_variables": ", ".join(usable_variables),
                "predictor_columns": generated_column_count,
            }
        )

    catalog.add_group("reservoir_current", current_columns)
    catalog.add_group("reservoir_lags", lag_columns)
    catalog.add_group("missing_flags", missing_columns)
    catalog.add_group(
        "reservoir_forward_fill_flags",
        forward_fill_flag_columns,
    )
    return reservoir_features


def build_wind_features(
    data_directory: Path,
    wind_candidates: pd.DataFrame,
    timeline: pd.DatetimeIndex,
    temporal: TemporalResolution,
    catalog: FeatureCatalog,
) -> pd.DataFrame:
    if wind_candidates.empty:
        return pd.DataFrame(index=timeline)

    source_path = data_directory / "wind_observations.csv"
    observations = read_selected_columns(
        source_path,
        ["sid", "time_point", *WIND_VALUE_COLUMNS],
    )
    observations["sid"] = observations["sid"].astype("string")
    candidate_ids = wind_candidates["station_id"].tolist()
    observations = observations.loc[observations["sid"].isin(candidate_ids)].copy()
    observations["timestamp_exact"] = parse_utc(observations["time_point"])
    observations["timestamp_bin"] = observations["timestamp_exact"].dt.floor(
        temporal.frequency
    )
    for value_column in WIND_VALUE_COLUMNS:
        observations[value_column] = pd.to_numeric(
            observations[value_column],
            errors="coerce",
        )
    observations = observations.sort_values("timestamp_exact")

    wind_features = pd.DataFrame(index=timeline)
    current_columns: list[str] = []
    missing_columns: list[str] = []
    station_availability: list[pd.Series] = []

    for candidate in wind_candidates.itertuples(index=False):
        station_id = str(candidate.station_id)
        station_key = safe_identifier(station_id)
        station_observations = observations.loc[observations["sid"].eq(station_id)]
        if station_observations.empty:
            raise ValueError(f"Wind candidate {station_id} has no observations")

        generated_column_count = 0
        station_current_columns: list[str] = []
        for value_column in WIND_VALUE_COLUMNS:
            aggregation = "mean" if value_column in {"ws", "wsg"} else "last"
            station_matrix = aggregate_station_values(
                station_observations,
                station_id_column="sid",
                value_column=value_column,
                aggregation=aggregation,
            )
            binned_values = station_matrix[station_id].reindex(timeline)
            current_column = f"wind_{station_key}_{value_column}"
            wind_features[current_column] = binned_values
            current_columns.append(current_column)
            station_current_columns.append(current_column)

            missing_column = f"{current_column}_missing"
            wind_features[missing_column] = binned_values.isna().astype("int8")
            missing_columns.append(missing_column)
            generated_column_count += 2

        station_availability.append(
            wind_features[station_current_columns].notna().any(axis="columns")
        )

        catalog.station_usage.append(
            {
                "type": "wind",
                "station_id": station_id,
                "station_name": candidate.station_name,
                "relation_to_da_vien": candidate.relation_to_da_vien,
                "raw_observations_selected": len(station_observations),
                "observed_bins_in_timeline": int(
                    station_observations.loc[
                        station_observations["timestamp_bin"].isin(timeline),
                        "timestamp_bin",
                    ].nunique()
                ),
                "source_variables": ", ".join(WIND_VALUE_COLUMNS),
                "predictor_columns": generated_column_count,
            }
        )

    available_station_count = (
        pd.concat(station_availability, axis="columns", sort=False)
        .sum(axis="columns")
        .astype("int8")
    )
    network_indicator_columns = [
        "wind_network_available_station_count",
        "wind_network_missing",
    ]
    wind_features["wind_network_available_station_count"] = available_station_count
    wind_features["wind_network_missing"] = available_station_count.eq(0).astype(
        "int8"
    )

    catalog.add_group("wind_current", current_columns)
    catalog.add_group("missing_flags", missing_columns)
    catalog.add_group("wind_network_indicators", network_indicator_columns)
    return wind_features


def add_forecast_targets(
    training_table: pd.DataFrame,
    temporal: TemporalResolution,
    catalog: FeatureCatalog,
) -> None:
    target_columns: list[str] = []
    for horizon_minutes in temporal.forecast_horizon_minutes:
        target_column = f"target_water_level_plus_{horizon_minutes}m"
        training_table[target_column] = training_table["water_level_depth"].shift(
            -temporal.steps_for_minutes(horizon_minutes)
        )
        target_columns.append(target_column)
    catalog.add_group("targets", target_columns)


def values_match_with_missing(
    actual: pd.Series,
    expected: pd.Series,
) -> bool:
    matching_values = actual.eq(expected) | (actual.isna() & expected.isna())
    return bool(matching_values.all())


def audit_temporal_features(
    training_table: pd.DataFrame,
    catalog: FeatureCatalog,
    reservoir_forward_fill_minutes: int,
    temporal: TemporalResolution,
) -> dict[str, object]:
    """Verify every lag, rolling feature, and future target against its source."""
    checks: dict[str, bool] = {}

    for lag_minutes in temporal.water_level_lag_minutes:
        lag_column = f"water_level_lag_{lag_minutes}m"
        checks[lag_column] = values_match_with_missing(
            training_table[lag_column],
            training_table["water_level_depth"].shift(
                temporal.steps_for_minutes(lag_minutes)
            ),
        )

    for current_column in catalog.groups.get("rain_current", []):
        station_prefix = current_column.removesuffix("_depth")
        for rolling_hours in RAIN_ROLLING_HOURS:
            rolling_column = f"{station_prefix}_sum_{rolling_hours}h"
            expected_values = training_table[current_column].rolling(
                window=temporal.steps_for_hours(rolling_hours),
                min_periods=temporal.steps_for_hours(rolling_hours),
            ).sum()
            checks[rolling_column] = values_match_with_missing(
                training_table[rolling_column],
                expected_values,
            )

    for lag_column in catalog.groups.get("reservoir_lags", []):
        match = re.fullmatch(r"(.+)_lag_(\d+)m", lag_column)
        if match is None:
            checks[lag_column] = False
            continue
        source_column, lag_minutes_text = match.groups()
        checks[lag_column] = values_match_with_missing(
            training_table[lag_column],
            training_table[source_column].shift(
                temporal.steps_for_minutes(int(lag_minutes_text))
            ),
        )

    reservoir_fill_steps = temporal.steps_for_minutes(
        reservoir_forward_fill_minutes
    )
    for fill_flag_column in catalog.groups.get(
        "reservoir_forward_fill_flags",
        [],
    ):
        current_column = fill_flag_column.removesuffix("_forward_filled")
        source_missing_column = f"{current_column}_source_missing"
        source_missing = training_table[source_missing_column].eq(1)
        reconstructed_observations = training_table[current_column].mask(
            source_missing
        )
        expected_filled_values = reconstructed_observations.ffill(
            limit=reservoir_fill_steps
        )
        checks[f"{current_column}_bounded_forward_fill"] = (
            values_match_with_missing(
                training_table[current_column],
                expected_filled_values,
            )
        )
        expected_fill_flag = (
            source_missing & training_table[current_column].notna()
        ).astype("int8")
        checks[fill_flag_column] = training_table[fill_flag_column].equals(
            expected_fill_flag
        )

    for horizon_minutes in temporal.forecast_horizon_minutes:
        target_column = f"target_water_level_plus_{horizon_minutes}m"
        checks[target_column] = values_match_with_missing(
            training_table[target_column],
            training_table["water_level_depth"].shift(
                -temporal.steps_for_minutes(horizon_minutes)
            ),
        )

    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    return {
        "all_checks_passed": not failed_checks,
        "check_count": len(checks),
        "failed_checks": failed_checks,
        "checks": checks,
        "policy": {
            "lags": "source shifted backward; no future source values",
            "bin_minutes": temporal.bin_minutes,
            "rain_rolling": "current-inclusive window ending at t with full window required",
            "targets": "Dã Viên water level shifted from the requested future horizon",
            "reservoir_forward_fill": (
                "past-only and capped at "
                f"{reservoir_forward_fill_minutes} minutes"
            ),
            "other_forward_fill": "not used",
            "interpolation": "not used",
        },
    }


def summarize_missingness(
    training_table: pd.DataFrame,
    catalog: FeatureCatalog,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_name, group_columns in catalog.groups.items():
        existing_columns = [
            column for column in group_columns if column in training_table.columns
        ]
        if not existing_columns:
            continue
        missing_rates = training_table[existing_columns].isna().mean()
        rows.append(
            {
                "feature_group": group_name,
                "column_count": len(existing_columns),
                "mean_missing_rate": float(missing_rates.mean()),
                "min_missing_rate": float(missing_rates.min()),
                "max_missing_rate": float(missing_rates.max()),
            }
        )
    return pd.DataFrame(rows)


def build_report(
    training_table: pd.DataFrame,
    catalog: FeatureCatalog,
    missingness: pd.DataFrame,
    leakage_audit: dict[str, object],
    candidate_path: Path,
    relation_path: Path,
    reservoir_timezone: str,
    reservoir_forward_fill_minutes: int,
    temporal: TemporalResolution,
) -> str:
    resolution_label = (
        "1h" if temporal.bin_minutes == 60 else f"{temporal.bin_minutes}min"
    )
    target_columns = catalog.groups["targets"]
    predictor_columns = [
        column
        for column in training_table.columns
        if column not in {"timestamp_utc", *target_columns}
    ]
    station_usage = pd.DataFrame(catalog.station_usage).sort_values(
        ["type", "station_id"]
    )
    station_counts = station_usage.groupby("type").size().to_dict()
    wind_network_missing_rate = float(
        training_table["wind_network_missing"].mean()
    )
    coverage_rows = []
    for source_name, group_name in (
        ("rain", "rain_current"),
        ("water_level", "water_level_current"),
        ("reservoir_after_bounded_fill", "reservoir_current"),
    ):
        group_columns = catalog.groups[group_name]
        coverage_rows.append(
            {
                "source": source_name,
                "column_count": len(group_columns),
                "mean_coverage_rate": float(
                    training_table[group_columns].notna().mean().mean()
                ),
                "min_coverage_rate": float(
                    training_table[group_columns].notna().mean().min()
                ),
                "max_coverage_rate": float(
                    training_table[group_columns].notna().mean().max()
                ),
            }
        )
    coverage_rows.append(
        {
            "source": "wind_network_either_station",
            "column_count": len(catalog.groups["wind_current"]),
            "mean_coverage_rate": 1 - wind_network_missing_rate,
            "min_coverage_rate": 1 - wind_network_missing_rate,
            "max_coverage_rate": 1 - wind_network_missing_rate,
        }
    )
    coverage = pd.DataFrame(coverage_rows)
    for column in (
        "mean_coverage_rate",
        "min_coverage_rate",
        "max_coverage_rate",
    ):
        coverage[column] = coverage[column].map(
            lambda value: f"{100 * value:.4f}%"
        )
    reservoir_rows = []
    reservoir_names = station_usage.loc[
        station_usage["type"].eq("reservoir"),
        ["station_id", "station_name"],
    ].set_index("station_id")["station_name"]
    for station_id, variables in catalog.reservoir_variables.items():
        reservoir_rows.append(
            {
                "station_id": station_id,
                "station_name": reservoir_names.get(station_id, ""),
                "raw_variables_kept": ", ".join(variables),
            }
        )

    missingness_display = missingness.copy()
    for column in (
        "mean_missing_rate",
        "min_missing_rate",
        "max_missing_rate",
    ):
        missingness_display[column] = missingness_display[column].map(
            lambda value: f"{100 * value:.4f}%"
        )

    lines = [
        f"# Báo cáo hydrology_training_{resolution_label}",
        "",
        "## Tổng quan",
        "",
        f"- Số dòng: {len(training_table):,}",
        f"- Số predictor features: {len(predictor_columns):,}",
        f"- Số target: {len(target_columns)}",
        f"- Tổng số cột, gồm timestamp: {len(training_table.columns):,}",
        f"- Khoảng thời gian UTC: {training_table['timestamp_utc'].min()} đến "
        f"{training_table['timestamp_utc'].max()}",
        f"- Timeline: đều {temporal.frequency}, lấy theo coverage của mực nước Dã Viên",
        f"- Topology candidates: {candidate_path}",
        f"- Full topology relations cross-check: {relation_path}",
        f"- Trạm mưa được dùng: {station_counts.get('rain', 0)}",
        f"- Trạm mực nước được dùng: {station_counts.get('water_level', 0)}",
        f"- Hồ chứa được dùng: {station_counts.get('reservoir', 0)}",
        f"- Trạm gió được dùng: {station_counts.get('wind', 0)}",
        "",
        "Reservoir được forward-fill chỉ từ quá khứ và tối đa "
        f"{reservoir_forward_fill_minutes} phút; không nội suy. Rain, water level "
        "và wind không được forward-fill. Source-missing và forward-filled flags "
        "được giữ riêng cho reservoir.",
        "",
        f"- Wind network missing rate (chỉ missing khi cả hai trạm cùng thiếu): "
        f"{100 * wind_network_missing_rate:.4f}%",
        "",
        "## Missing rate theo nhóm feature",
        "",
        markdown_table(missingness_display),
        "",
        "## Coverage theo nguồn",
        "",
        markdown_table(coverage),
        "",
        "## Station thực sự được sử dụng",
        "",
        markdown_table(station_usage),
        "",
        "## Biến hồ chứa được giữ",
        "",
        markdown_table(pd.DataFrame(reservoir_rows)),
        "",
        "Các tên htl, hhl, qden, qdi, mucnuocsong được giữ đúng theo raw data. "
        "Builder không suy đoán tên đầy đủ, unit hoặc đổi đơn vị. Các cột hồ chứa "
        "không có bất kỳ giá trị nào giao với timeline Dã Viên bị loại để tránh "
        "feature 100% missing.",
        "",
        "Timestamp hồ chứa không có timezone trong raw data. Builder diễn giải theo "
        f"timezone cấu hình {reservoir_timezone}, sau đó đổi sang UTC. Có thể thay "
        "đổi bằng tham số --reservoir-timezone nếu metadata chính thức xác nhận khác.",
        f" Forward-fill hồ chứa bị chặn sau {reservoir_forward_fill_minutes} phút "
        "và có flag riêng; giới hạn có thể đổi bằng "
        "--reservoir-ffill-limit-minutes.",
        "",
        "## Quy tắc tổng hợp và feature engineering",
        "",
        f"- Rain depth: cộng các bản ghi trong cùng bin "
        f"{temporal.bin_minutes} phút; rolling 1h, 3h, "
        "6h, 24h gồm thời điểm t và chỉ dùng [t-window+1, t]. Cửa sổ thiếu bất kỳ "
        "bin nào sẽ là missing.",
        f"- Dã Viên water level: trung bình trong bin {temporal.bin_minutes} phút; "
        f"lag {', '.join(f'{value}m' for value in temporal.water_level_lag_minutes)}.",
        f"- Reservoir: trung bình raw values trong bin {temporal.bin_minutes} phút; "
        "forward-fill từ "
        f"quá khứ tối đa {reservoir_forward_fill_minutes} phút; lag 60m và 180m "
        "được tạo sau bước fill. Source-missing và forward-filled flags cho biết "
        "nguồn gốc từng giá trị.",
        "- Wind: ws và wsg lấy trung bình trong bin; wd và wdg lấy quan sát cuối "
        "cùng trong bin. Dùng riêng Cảng Thuận An và Cảng Tư Hiền theo yêu cầu; "
        "wind_network_missing chỉ bằng 1 khi cả hai trạm cùng thiếu, và "
        "wind_network_available_station_count cho biết có 0, 1 hay 2 trạm.",
        "- Hai trạm gió này được topology phân loại outside_catchment nhưng được "
        "dùng theo chỉ định mới của người dùng, thay cho VPTT PCTTHUE.",
        "- Targets: water level Dã Viên tại "
        + ", ".join(
            f"+{minutes // 60}h"
            for minutes in temporal.forecast_horizon_minutes
        )
        + ".",
        "",
        "## Leakage audit",
        "",
        f"- Số phép kiểm tra: {leakage_audit['check_count']}",
        f"- Tất cả kiểm tra đạt: {leakage_audit['all_checks_passed']}",
        f"- Kiểm tra lỗi: {leakage_audit['failed_checks']}",
        "",
        "Dataset giữ các dòng có target missing ở cuối chuỗi và các khoảng mất dữ "
        "liệu. Khi huấn luyện từng horizon, chỉ loại các dòng target tương ứng bị "
        "missing sau khi đã chia train/validation theo thời gian.",
        "",
    ]
    return "\n".join(lines)


def build_training_dataset(
    data_directory: Path,
    candidate_path: Path,
    relation_path: Path,
    reservoir_timezone: str,
    reservoir_forward_fill_minutes: int,
    temporal: TemporalResolution = DEFAULT_TEMPORAL_RESOLUTION,
) -> tuple[pd.DataFrame, FeatureCatalog, pd.DataFrame, dict[str, object]]:
    temporal.validate()
    candidates = load_topology_candidates(candidate_path, relation_path)
    da_vien_candidate = candidates.loc[
        candidates["type"].eq("water_level")
        & candidates["station_name"].eq(DA_VIEN_NAME)
    ].iloc[0]
    rain_candidates = candidates.loc[candidates["type"].eq("rain")].copy()
    reservoir_candidates = candidates.loc[
        candidates["type"].eq("reservoir")
    ].copy()
    wind_candidates = load_requested_wind_stations(
        data_directory,
        relation_path,
    )

    water_level, water_observation_count = load_da_vien_water_level(
        data_directory,
        da_vien_candidate,
        temporal,
    )
    timeline = water_level.index
    catalog = FeatureCatalog()

    feature_frames = [
        build_rain_features(
            data_directory,
            rain_candidates,
            timeline,
            temporal,
            catalog,
        ),
        build_water_level_features(
            water_level,
            da_vien_candidate,
            water_observation_count,
            temporal,
            catalog,
        ),
        build_reservoir_features(
            data_directory,
            reservoir_candidates,
            timeline,
            reservoir_timezone,
            reservoir_forward_fill_minutes,
            temporal,
            catalog,
        ),
        build_wind_features(
            data_directory,
            wind_candidates,
            timeline,
            temporal,
            catalog,
        ),
    ]
    training_table = pd.concat(feature_frames, axis="columns").copy()
    add_forecast_targets(training_table, temporal, catalog)
    training_table = training_table.reset_index()

    expected_timestamps = pd.date_range(
        training_table["timestamp_utc"].iloc[0],
        training_table["timestamp_utc"].iloc[-1],
        freq=temporal.frequency,
        tz="UTC",
    )
    if not training_table["timestamp_utc"].reset_index(drop=True).equals(
        pd.Series(expected_timestamps, name="timestamp_utc")
    ):
        raise AssertionError("Output timeline is not a continuous 10-minute UTC index")

    leakage_audit = audit_temporal_features(
        training_table,
        catalog,
        reservoir_forward_fill_minutes,
        temporal,
    )
    if not leakage_audit["all_checks_passed"]:
        raise AssertionError(
            f"Temporal leakage audit failed: {leakage_audit['failed_checks']}"
        )
    missingness = summarize_missingness(training_table, catalog)
    return training_table, catalog, missingness, leakage_audit


def output_companion_paths(output_path: Path) -> dict[str, Path]:
    return {
        "report": output_path.with_name(f"{output_path.stem}_report.md"),
        "missingness": output_path.with_name(
            f"{output_path.stem}_missingness.csv"
        ),
        "leakage_audit": output_path.with_name(
            f"{output_path.stem}_leakage_audit.json"
        ),
    }


def validate_output_paths(
    output_path: Path,
    companion_paths: dict[str, Path],
    data_directory: Path,
    force: bool,
) -> None:
    raw_directory = data_directory.resolve()
    all_output_paths = [output_path, *companion_paths.values()]
    for path in all_output_paths:
        resolved_path = path.resolve()
        if resolved_path == raw_directory or raw_directory in resolved_path.parents:
            raise ValueError(f"Refusing to write a derived output into data/: {path}")
    existing_paths = [path for path in all_output_paths if path.exists()]
    if existing_paths and not force:
        formatted_paths = ", ".join(map(str, existing_paths))
        raise FileExistsError(
            f"Derived output already exists: {formatted_paths}. Pass --force to replace."
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
        "--output",
        type=Path,
        default=default_paths.reports / "hydrology_training_10min_v2.csv",
    )
    parser.add_argument(
        "--reservoir-timezone",
        default="Asia/Bangkok",
        help=(
            "Timezone of timezone-naive reservoir ngaylaysolieu/gio values "
            "(default: Asia/Bangkok)."
        ),
    )
    parser.add_argument(
        "--reservoir-ffill-limit-minutes",
        type=int,
        default=DEFAULT_RESERVOIR_FORWARD_FILL_MINUTES,
        help=(
            "Maximum past-only reservoir forward-fill age in minutes "
            f"(default: {DEFAULT_RESERVOIR_FORWARD_FILL_MINUTES})."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing derived outputs. Raw files are never overwritten.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.reservoir_ffill_limit_minutes <= 0
        or args.reservoir_ffill_limit_minutes % 10 != 0
    ):
        raise ValueError(
            "--reservoir-ffill-limit-minutes must be a positive multiple of 10"
        )
    companion_paths = output_companion_paths(args.output)
    validate_output_paths(
        args.output,
        companion_paths,
        args.data_dir,
        args.force,
    )

    training_table, catalog, missingness, leakage_audit = build_training_dataset(
        data_directory=args.data_dir,
        candidate_path=args.candidates,
        relation_path=args.relations,
        reservoir_timezone=args.reservoir_timezone,
        reservoir_forward_fill_minutes=args.reservoir_ffill_limit_minutes,
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
        DEFAULT_TEMPORAL_RESOLUTION,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    training_table.to_csv(args.output, index=False)
    missingness.to_csv(companion_paths["missingness"], index=False)
    write_json(leakage_audit, companion_paths["leakage_audit"])
    companion_paths["report"].write_text(report_text, encoding="utf-8")

    target_columns = catalog.groups["targets"]
    predictor_count = len(training_table.columns) - len(target_columns) - 1
    print(f"Wrote {args.output}")
    print(f"Wrote {companion_paths['report']}")
    print(f"Rows: {len(training_table):,}")
    print(f"Predictor features: {predictor_count:,}")
    print(f"Leakage audit passed: {leakage_audit['all_checks_passed']}")


if __name__ == "__main__":
    main()
