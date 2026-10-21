"""Resolve named rainfall combinations without a fixed station-count limit."""

import re
import unicodedata


RAIN_ALIASES = {
    "hong_thai": "750062",
    "nham": "750007",
    "bach_ma": "750041",
    "phu_bai": "750014",
    "vp": "750033",
    "binh_dien": "750061",
}
DEFAULT_COMBINATION = "+".join(RAIN_ALIASES)


def station_alias(name):
    """Make an ASCII alias; full catalog names remain available for expansion."""
    name = unicodedata.normalize("NFKD", name.lower().replace("đ", "d"))
    name = "".join(char for char in name if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", "_", name).strip("_")


def available_rain_ids(columns):
    return list(dict.fromkeys(
        column.split("_", 2)[1]
        for column in columns
        if column.startswith("rain_") and len(column.split("_", 2)) == 3
    ))


def resolve_rain_stations(selection, available_ids, aliases=None):
    """Accept aliases/IDs joined by '+', rejecting duplicates and absent sources."""
    available_ids = list(available_ids)
    if selection is None or selection.strip().lower() == "all":
        if not available_ids:
            raise ValueError("No rain stations available")
        return available_ids
    lookup = {**RAIN_ALIASES, **(aliases or {})}
    selected = []
    for token in selection.split("+"):
        token = station_alias(token.strip())
        sid = lookup.get(token, token if token.isdecimal() else None)
        if sid is None or sid not in available_ids:
            valid = sorted(name for name, code in lookup.items() if code in available_ids)
            raise ValueError(
                f"Unknown/unavailable rain station: {token!r}; "
                f"names: {', '.join(valid)}; IDs: {', '.join(available_ids)}"
            )
        if sid in selected:
            raise ValueError(f"Duplicate rain station: {token} ({sid})")
        selected.append(sid)
    return selected


def select_rain_columns(columns, station_ids):
    """Filter entire station blocks, retaining original feature order."""
    available = available_rain_ids(columns)
    if not station_ids or len(set(station_ids)) != len(station_ids):
        raise ValueError("Provide distinct rain stations")
    unknown = set(station_ids) - set(available)
    if unknown:
        raise ValueError(f"Rain stations absent from features: {sorted(unknown)}")
    prefixes = tuple(f"rain_{sid}_" for sid in station_ids)
    return [column for column in columns
            if not column.startswith("rain_") or column.startswith(prefixes)]


def combination_name(station_ids):
    """Stable name independent of CLI order, with numeric fallback for new sources."""
    names = {sid: name for name, sid in RAIN_ALIASES.items()}
    return "+".join(names.get(sid, sid) for sid in sorted(station_ids))
