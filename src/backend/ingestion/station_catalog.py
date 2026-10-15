"""Import source station metadata explicitly; never start observation ingestion."""

import argparse
import json
import math
from pathlib import Path
from threading import Event

from dotenv import load_dotenv
import sqlalchemy as sa

from src.backend.db.models import Station
from src.backend.db.session import build_engine
from src.backend.ingestion.worker import HueSourceClient, SourceError, SyncSettings, write_report

SOURCES = {
    "water_level_station": ("water_level", "water_level_depth", "water_station_type"),
    "rain_water_stations": ("rainfall", "rain_water_depth", "waterStationType"),
}
SOURCE_FIELDS = ("address", "latitude", "longitude", "station_subtype", "station_subtype_label")


def normalize_station(record, collection):
    """Validate map metadata and preserve source codes including leading zeros.

    Args:
        record: One API station object; personal/system fields are discarded.
        collection: Supported water/rain metadata collection.

    Returns:
        Station values; new stations are disabled and prediction is disabled.
        Rain uses project-approved mm; new water units remain unconfirmed.

    Raises:
        SourceError: Invalid identity, name, coordinates, address or subtype.
    """
    try:
        if not isinstance(record, dict):
            raise ValueError()
        station_type, observations, subtype_field = SOURCES[collection]
        code, name = record["code"], record["name"]
        if not isinstance(code, str) or not code.strip() or code != code.strip():
            raise ValueError()
        if not isinstance(name, str) or not name.strip():
            raise ValueError()
        geom = record.get("geom")
        if not isinstance(geom, dict) or geom.get("type") != "Point":
            raise ValueError()
        longitude, latitude = geom["coordinates"]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in (longitude, latitude)
        ):
            raise ValueError()
        if not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
            raise ValueError()
        address = record.get("address")
        if address is not None and not isinstance(address, str):
            raise ValueError()
        subtype = record.get(subtype_field)
        if isinstance(subtype, str):
            subtype = json.loads(subtype) if subtype.strip().startswith("{") else None
        if subtype is not None and not isinstance(subtype, dict):
            raise ValueError()
        subtype = subtype or {}
        if any(
            value is not None and not isinstance(value, str)
            for value in (subtype.get("name"), subtype.get("desc"))
        ):
            raise ValueError()
    except (KeyError, TypeError, ValueError):
        raise SourceError("Source station metadata failed validation.") from None
    return {
        "id": code,
        "name": name.strip(),
        "station_type": station_type,
        "address": address.strip() or None if address is not None else None,
        "latitude": float(latitude),
        "longitude": float(longitude),
        "station_subtype": subtype.get("name"),
        "station_subtype_label": subtype.get("desc"),
        "enabled": False,
        "prediction_enabled": False,
        "source_name": "hue_dcu",
        "source_station_id": code,
        "observation_collection": observations,
        "measurement_unit": "mm",
    }


def fetch_station_catalog(client):
    """Read every metadata page, stopping only at an empty page.

    Args:
        client: Authenticated Hue source client with bounded timeouts/retries.

    Returns:
        Validated water/rain metadata; all IDs are unique across both collections.

    Raises:
        SourceError: Bad payload, duplicate identity, pagination stall or empty catalog.
    """
    stations, seen = [], set()
    for collection in SOURCES:
        found = 0
        for page in range(1, 101):
            payload = client.request(
                "GET", f"/items/{collection}", params={"limit": 100, "page": page, "sort": "id"}
            )
            rows = payload.get("data")
            if not isinstance(rows, list):
                raise SourceError("Source station list is not an array.")
            if not rows:
                break
            for record in rows:
                station = normalize_station(record, collection)
                if station["id"] in seen:
                    raise SourceError("Duplicate station code; refusing ambiguous source mapping.")
                seen.add(station["id"])
                stations.append(station)
                found += 1
        else:
            raise SourceError("Station pagination exceeded its bound.")
        if not found:
            raise SourceError("Source station collection is empty.")
    return stations


def import_station_catalog(engine, stations, *, dry_run=True):
    """Atomically add metadata and update source-managed fields only when changed.

    Args:
        engine: Application PostgreSQL engine, owned by caller.
        stations: Validated catalog from fetch_station_catalog.
        dry_run: True previews changes without writing any database rows.

    Returns:
        Counts and IDs for additions/updates; existing names, units, alarms, river
        names, enabled flags and prediction permissions are retained.

    Raises:
        SourceError: A code is already assigned to another source/type.
        sqlalchemy.exc.SQLAlchemyError: Database failure; all writes roll back.
    """
    report = {"dry_run": dry_run, "inserted": [], "updated": [], "unchanged": []}
    with engine.begin() as connection:
        # Serialize catalog imports without blocking observation ingestion.
        connection.execute(sa.text("SELECT pg_advisory_xact_lock(74647403, 2)"))
        existing = {
            row["id"]: row
            for row in connection.execute(
                sa.select(Station.__table__).where(Station.id.in_([row["id"] for row in stations]))
            ).mappings()
        }
        for station in stations:
            current = existing.get(station["id"])
            if current is None:
                report["inserted"].append(station["id"])
                if not dry_run:
                    connection.execute(sa.insert(Station).values(**station))
                continue
            if any(
                current[key] != station[key]
                for key in (
                    "station_type",
                    "source_name",
                    "source_station_id",
                    "observation_collection",
                )
            ):
                raise SourceError("Station code conflicts with existing source mapping.")
            changes = {
                key: station[key]
                for key in SOURCE_FIELDS
                if station[key] is not None and station[key] != current[key]
            }
            report["updated" if changes else "unchanged"].append(station["id"])
            if changes and not dry_run:
                connection.execute(
                    sa.update(Station).where(Station.id == station["id"]).values(**changes)
                )
    report["counts"] = {key: len(report[key]) for key in ("inserted", "updated", "unchanged")}
    return report


def main():
    """Run an explicit metadata import; default is dry-run, --apply commits changes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--report", type=Path, default=Path("reports/station_catalog_import.json"))
    arguments = parser.parse_args()
    load_dotenv()
    engine = build_engine()
    client = HueSourceClient(SyncSettings.from_env(), Event())
    try:
        client.login()
        stations = fetch_station_catalog(client)
        report = import_station_catalog(engine, stations, dry_run=not arguments.apply)
        report["station_types"] = {
            kind: sum(row["station_type"] == kind for row in stations)
            for kind in ("water_level", "rainfall")
        }
        write_report(arguments.report, report)
        print(
            json.dumps(
                {
                    "dry_run": report["dry_run"],
                    **report["counts"],
                    "station_types": report["station_types"],
                },
                ensure_ascii=False,
            )
        )
    except (SourceError, sa.exc.SQLAlchemyError, ValueError) as error:
        parser.exit(1, f"Station catalog import failed; error_type={type(error).__name__}\n")
    finally:
        client.session.close()
        engine.dispose()


if __name__ == "__main__":
    main()
