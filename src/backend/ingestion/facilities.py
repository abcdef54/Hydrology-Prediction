"""Validate SOHOA operational reports without guessing units, dates or station identity."""

from datetime import datetime, timezone
import math
import re
from zoneinfo import ZoneInfo

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert
from src.backend.db.models import Facility, FacilityOperation, Station
from src.backend.ingestion.worker import SourceError

VIETNAM = ZoneInfo("Asia/Ho_Chi_Minh")
METRICS = ("htl", "hhl", "qden", "qdi", "mucnuocsong", "luuluongdieutiet")
SPECIFICATIONS = (
    "mndbt",
    "mnc",
    "nguongtran",
    "nguongxaday",
    "socua",
    "authuyen",
    "khaudo",
    "baodongmuc1",
    "baodongmuc2",
    "baodongmuc3",
)
RIVER_LINKS = {
    16: ("0074647403", "Trạm Dã Viên (Sông Hương)"),
    17: ("0074647404", "Trạm Tứ Phú (Sông Bồ)"),
    19: ("0074647402", "Cảng cá Thuận An"),
    20: ("0074648201", "Cảng Tư Hiền"),
}


def text(value):
    if value is None:
        return ""
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        raise SourceError("Invalid facility text field.")
    return str(value).strip()


def positive_id(value):
    raw = text(value)
    if not re.fullmatch(r"[1-9]\d*", raw):
        raise SourceError("Invalid facility source identity.")
    return int(raw)


def number(value):
    raw = text(value)
    if not raw:
        return None
    if not re.fullmatch(r"[+-]?\d+(?:[.,]\d+)?", raw):
        raise SourceError("Invalid operational numeric value; no unit/scale conversion applied.")
    value = float(raw.replace(",", "."))
    if not math.isfinite(value):
        raise SourceError("Nonfinite operational reading.")
    return value


def normalize_facility(record):
    """Keep source specifications and link only explicitly verified duplicate river IDs.

    Args:
        record: SOHOA metadata containing id, type, name and nullable coordinates.
    Returns:
        Facility metadata; linked river stations reuse their existing DCU identity.
    Raises:
        SourceError: Invalid metadata or a river not matching the reviewed mapping.
    """
    identifier = positive_id(record.get("id"))
    kind = {"1": "hydropower", "2": "reservoir", "3": "dam", "4": "river"}.get(
        text(record.get("loaicongtrinhid"))
    )
    name = text(record.get("tencongtrinh"))
    if not kind or not name:
        raise SourceError("Invalid facility type/name.")
    station_id = None
    if kind == "river":
        mapping = RIVER_LINKS.get(identifier)
        if mapping is None or " ".join(name.casefold().split()) != " ".join(
            mapping[1].casefold().split()
        ):
            raise SourceError("Unreviewed river source mapping; refusing duplicate station.")
        station_id = mapping[0]
    latitude, longitude = number(record.get("vido")), number(record.get("kinhdo"))
    if (latitude is None) != (longitude is None) or (
        latitude is not None and not (-90 <= latitude <= 90 and -180 <= longitude <= 180)
    ):
        raise SourceError("Invalid facility coordinates.")
    return {
        "id": f"sohoa:{identifier}",
        "source_id": identifier,
        "source_name": "hue_sohoa",
        "name": name,
        "kind": kind,
        "kind_label": text(record.get("tenloaicongtrinhid")),
        "station_id": station_id,
        "latitude": latitude,
        "longitude": longitude,
        "specifications": {field: text(record.get(field)) or None for field in SPECIFICATIONS},
        "measurement_units": dict.fromkeys(METRICS),
        "sync_enabled": True,
    }


def parse_observation_time(date_text, hour_text):
    """Interpret only explicit day/month/year and hour:minute as Vietnam wall time.

    Args:
        date_text: Original day/month/year observation date, not update date.
        hour_text: Explicit hour:minute; missing/ambiguous hours are not repaired.
    Returns:
        Aware UTC datetime, or None for invalid/ambiguous original time text.
    """
    if not re.fullmatch(r"\d{1,2}:\d{1,2}", hour_text):
        return None
    try:
        parsed = datetime.strptime(f"{date_text} {hour_text}", "%d/%m/%Y %H:%M")
    except ValueError:
        return None
    return parsed.replace(tzinfo=VIETNAM).astimezone(timezone.utc)


def parse_operation(record):
    """Retain one source report, including tombstones and invalid/unknown time text.

    Args:
        record: Original SOHOA report; source users/notes/attachments are excluded.
    Returns:
        Typed finite raw metrics, status and UTC times; invalid observation time
        is NULL with time_status=invalid, never replaced by update/receipt time.
    Raises:
        SourceError: Invalid identity or numeric field; caller rolls back its batch.
    """
    date = text(record.get("ngaylaysolieu"))
    hour = text(record.get("gio"))
    observed = parse_observation_time(date, hour)
    updated_text = text(record.get("ngaycapnhat"))
    try:
        updated = (
            datetime.strptime(updated_text, "%m/%d/%Y %I:%M:%S %p")
            .replace(tzinfo=VIETNAM)
            .astimezone(timezone.utc)
        )
    except ValueError:
        updated = None
    return {
        "source_record_id": positive_id(record.get("id")),
        "facility_id": f"sohoa:{positive_id(record.get('tencongtrinh'))}",
        "function_code": text(record.get("chucnang")),
        "observed_at": observed,
        "time_status": "valid" if observed else "invalid",
        "observation_date_text": date,
        "observation_hour_text": hour,
        "source_updated_at": updated,
        "source_updated_text": updated_text,
        "source_status": text(record.get("trangthai")) or None,
        "source_status_label": text(record.get("tentrangthai")) or None,
        **{field: number(record.get(field)) for field in METRICS},
        "gate_opening_text": text(record.get("socuamo")) or None,
        "operation_start_text": text(record.get("thoigianbatdau")) or None,
        "operation_end_text": text(record.get("thoigianketthuc")) or None,
    }


def upsert_operations(connection, rows, *, snapshot=False):
    """Insert older snapshots without overwrite; apply live revisions/deletions by source ID.

    Args:
        connection: Caller-owned transaction using the shared ingestion writer lock.
        rows: Validated reports with unique source IDs, including deleted records.
        snapshot: SQLite only inserts; live API is authoritative for status/values.
    Returns:
        Inserted/updated/unchanged counts; received_at is never overwritten.
    Raises:
        SourceError: A source ID is reassigned to a different facility.
        sqlalchemy.exc.SQLAlchemyError: Constraint/storage failure, batch rolls back.
    """
    counts = {"inserted": 0, "updated": 0, "unchanged": 0}
    for offset in range(0, len(rows), 500):
        batch = rows[offset : offset + 500]
        existing = {
            row["source_record_id"]: row
            for row in connection.execute(
                sa.select(FacilityOperation.__table__).where(
                    FacilityOperation.source_record_id.in_(
                        [row["source_record_id"] for row in batch]
                    )
                )
            ).mappings()
        }
        if any(
            row["source_record_id"] in existing
            and existing[row["source_record_id"]]["facility_id"] != row["facility_id"]
            for row in batch
        ):
            raise SourceError("Operational source ID changed facility identity.")
        statement = insert(FacilityOperation).values(batch)
        if snapshot:
            statement = statement.on_conflict_do_nothing(
                index_elements=[FacilityOperation.source_record_id]
            )
        else:
            fields = [
                field for field in batch[0] if field not in ("source_record_id", "facility_id")
            ]
            statement = statement.on_conflict_do_update(
                index_elements=[FacilityOperation.source_record_id],
                set_={field: statement.excluded[field] for field in fields},
                where=sa.or_(
                    *(
                        FacilityOperation.__table__.c[field].is_distinct_from(
                            statement.excluded[field]
                        )
                        for field in fields
                    )
                ),
            )
        changed = set(connection.scalars(statement.returning(FacilityOperation.source_record_id)))
        counts["inserted"] += sum(identifier not in existing for identifier in changed)
        counts["updated"] += sum(identifier in existing for identifier in changed)
        counts["unchanged"] += len(batch) - len(changed)
    return counts


def upsert_facilities(connection, rows):
    """Save catalog while retaining configured units, sync flags and full-sync timestamps.

    Args:
        connection: Caller-owned transaction.
        rows: Reviewed source metadata; river links must refer to real DCU water stations.
    Returns:
        None. No existing station/model/observation is modified.
    Raises:
        SourceError: A linked station is absent or belongs to a different source/type.
        sqlalchemy.exc.SQLAlchemyError: Metadata write failure.
    """
    links = [row["station_id"] for row in rows if row["station_id"]]
    valid_links = set(
        connection.scalars(
            sa.select(Station.id).where(
                Station.id.in_(links),
                Station.station_type == "water_level",
                Station.source_name == "hue_dcu",
            )
        )
    )
    if set(links) != valid_links:
        raise SourceError("River mapping does not match existing DCU stations.")
    existing = {
        row["id"]: row
        for row in connection.execute(
            sa.select(Facility.__table__).where(Facility.id.in_([row["id"] for row in rows]))
        ).mappings()
    }
    for row in rows:
        current = existing.get(row["id"])
        if current and any(
            current[key] != row[key] for key in ("source_id", "source_name", "kind", "station_id")
        ):
            raise SourceError("Facility identity changed; refusing unreviewed remapping.")
        statement = insert(Facility).values(row)
        fields = ("name", "kind_label", "latitude", "longitude", "specifications")
        statement = statement.on_conflict_do_update(
            index_elements=[Facility.id],
            set_={field: statement.excluded[field] for field in fields},
            where=sa.or_(
                *(
                    Facility.__table__.c[field].is_distinct_from(statement.excluded[field])
                    for field in fields
                )
            ),
        )
        connection.execute(statement)
