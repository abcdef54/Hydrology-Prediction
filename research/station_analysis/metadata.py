"""Enrich station identities without inventing sensor datums or alert levels.

The checked-in YAML records source provenance and provisional geographic groups.
No network requests occur here, making repeated analyses reproducible and offline.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import yaml

DEFAULT_SOURCES = Path(__file__).with_name("metadata_sources.yaml")


def load_sources(source_config: Path | None = None) -> dict:
    """Load curated sources and reject malformed station IDs or source references."""
    path = Path(source_config) if source_config else DEFAULT_SOURCES
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Unsupported station metadata source schema")
    for station_id, entry in data.get("stations", {}).items():
        if not isinstance(station_id, str) or not isinstance(entry, dict):
            raise ValueError("Metadata station IDs must be quoted strings")
        if any(key not in data["sources"] for key in entry.get("source_ids", [])):
            raise ValueError(f"Unknown metadata source for {station_id}")
    return data


def _text(value) -> str:
    return "" if value is None or pd.isna(value) else str(value)


def enrich_stations(stations: pd.DataFrame, source_config: Path | None = None) -> pd.DataFrame:
    """Return source-preserving metadata plus explicit hypotheses and confidence.

    All original fields remain unchanged, including coordinates and database alert
    values. The aliases and added fields are analysis metadata, not DB updates.
    A flood-tower code supports instrument classification but does not establish
    that the raw depth is referenced to the local road surface.
    """
    sources = load_sources(source_config)
    result = stations.copy()
    if "station_id" not in result:
        if "id" in result:
            result["station_id"] = result["id"].astype(str)
        elif result.index.name in {"station_id", "id"}:
            result["station_id"] = result.index.astype(str)
        else:
            raise ValueError("Station metadata requires station_id or id")
    if result["station_id"].duplicated().any():
        raise ValueError("Station IDs must be unique")
    records = []
    for _, row in result.iterrows():
        station_id = str(row["station_id"])
        entry = sources.get("stations", {}).get(station_id, {})
        subtype = _text(row.get("station_subtype"))
        flood_tower = subtype.startswith("flood_")
        source_ids = entry.get("source_ids", ["flood_towers"] if flood_tower else [])
        links = [sources["sources"][key]["url"] for key in source_ids]
        alarms_present = any(
            pd.notna(row.get(f"alarm_level_{level}_m")) for level in (1, 2, 3)
        )
        records.append({
            "hydrological_type": entry.get(
                "hydrological_type", "inundation_tower_provisional" if flood_tower else "unknown"
            ),
            "static_group_hypothesis": entry.get("static_group_hypothesis", "unclassified"),
            "river_or_basin": entry.get("river_or_basin", _text(row.get("river_name")) or "unknown"),
            "classification_basis": (
                "registry_subtype_and_curated_geographic_context" if entry
                else "registry_subtype_only" if subtype else "unverified"
            ),
            "classification_confidence": entry.get(
                "classification_confidence", "medium" if flood_tower else "unknown"
            ),
            "measured_quantity": "unknown",
            "measured_quantity_hypothesis": entry.get(
                "measured_quantity_hypothesis", "local_inundation_depth_or_sensor_relative_level"
                if flood_tower else "unknown"
            ),
            "measurement_unit_basis": "database_project_convention_not_sensor_calibration",
            "water_level_datum": "unknown",
            "station_zero_elevation_m": np.nan,
            "station_zero_vertical_datum": "unknown",
            "coordinate_basis": "database_registry_not_independently_surveyed",
            "alarm_basis": "database_value_applicability_unverified" if alarms_present else "not_available",
            "alarm_source_station": "unknown" if alarms_present else "not_available",
            "alarm_source": "hydrology.stations" if alarms_present else "not_available",
            "tidal_evidence": entry.get("tidal_evidence", "unknown_at_sensor"),
            "tidal_confidence": entry.get("tidal_confidence", "unknown"),
            "geography_notes": entry.get("geography_notes", "No curated station context available."),
            "public_source_urls": json.dumps(links, ensure_ascii=False),
            "public_source_ids": ";".join(source_ids),
            "metadata_reviewed_at": sources["reviewed_at"],
            "verification_status": "quantity_datum_and_alert_applicability_need_verification",
        })
    extra = pd.DataFrame(records, index=result.index)
    collisions = set(extra.columns) & set(result.columns)
    if collisions:
        # Idempotent enrichment: recompute only this module's derived fields.
        result = result.drop(columns=sorted(collisions))
    return pd.concat([result, extra], axis=1)
