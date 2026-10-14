"""Read exact clock-hour observations for a paged station/day history table."""

from datetime import date, datetime, time, timedelta, timezone
from typing import Annotated
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Query, Request
import sqlalchemy as sa

from src.backend.api.schemas import (
    ErrorResponse,
    HourlyHistoryCell,
    HourlyHistoryCursor,
    HourlyHistoryQuery,
    HourlyHistoryResponse,
    HourlyHistoryRow,
)
from src.backend.db.models import Observation, Station

router = APIRouter(prefix="/api/v1", tags=["Observations"])
VIETNAM_TIME = ZoneInfo("Asia/Ho_Chi_Minh")


def select_station_days(
    stations: list[dict], start: date, end: date, query: HourlyHistoryQuery
) -> list[tuple[date, dict]]:
    """Build only the requested page plus one row, including days without readings.

    Args:
        stations: Enabled, filtered metadata ordered by ascending station ID.
        start: Inclusive first local date.
        end: Inclusive last local date, no later than today.
        query: Page size and optional exclusive (date, station ID) cursor.

    Returns:
        At most limit + 1 station/day pairs, newest date first then station ID.
        No calendar-wide array is generated, even for a multi-year date range.
    """
    if not stations:
        return []
    selected = []
    day = query.cursor_date or end
    while day >= start:
        for station in stations:
            if day == query.cursor_date and station["id"] <= query.cursor_station_id:
                continue
            selected.append((day, station))
            if len(selected) > query.limit:
                return selected
        if day == start:
            break
        day -= timedelta(days=1)
    return selected


def clock_hours(day: date) -> list[datetime]:
    """Return the 24 scheduled local hours as timezone-aware UTC timestamps.

    Args:
        day: Calendar date in Asia/Ho_Chi_Minh.

    Returns:
        Exactly 24 instants, corresponding to 00:00 through 23:00 Vietnam time.
    """
    return [
        datetime.combine(day, time(hour=hour), VIETNAM_TIME).astimezone(timezone.utc)
        for hour in range(24)
    ]


@router.get(
    "/observations/hourly",
    response_model=HourlyHistoryResponse,
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def get_hourly_history(
    request: Request, query: Annotated[HourlyHistoryQuery, Query()]
) -> HourlyHistoryResponse:
    """Read a station/day table using exact clock-hour lookups, without aggregation.

    Args:
        request: Request providing the database engine; this endpoint only reads.
        query: Inclusive Vietnam dates, type/ID/name filters and row pagination.
            Both dates absent means today; one date means that single day.

    Returns:
        Rows sorted by descending date, ascending station ID; each has 24 cells.
        A missing raw reading or explicit missing value is null/missing. Future
        hours are null/future. Zero and negative water levels remain unchanged.
        next_cursor is exclusive and must be reused with the same date/filter set.
        Values retain station units; rain cells are not hourly accumulated rainfall.

    Raises:
        HTTPException: 422 for future dates or a cursor outside the date range.
        sqlalchemy.exc.SQLAlchemyError: Database failures handled as safe 503s.
    """
    now = datetime.now(timezone.utc)
    today = now.astimezone(VIETNAM_TIME).date()
    start = query.start_date or query.end_date or today
    end = query.end_date or query.start_date or today
    if end > today:
        raise HTTPException(
            status_code=422,
            detail={"code": "INVALID_REQUEST", "message": "Dates must not be in the future."},
        )
    if query.cursor_date and not start <= query.cursor_date <= end:
        raise HTTPException(
            status_code=422,
            detail={"code": "INVALID_REQUEST", "message": "Cursor date is outside the range."},
        )

    statement = sa.select(
        Station.id, Station.name, Station.station_type, Station.measurement_unit
    ).where(Station.enabled.is_(True))
    if query.station_type is not None:
        statement = statement.where(Station.station_type == query.station_type)
    if query.station_id is not None:
        statement = statement.where(Station.id == query.station_id)
    if query.station_name is not None:
        statement = statement.where(Station.name.icontains(query.station_name, autoescape=True))

    with request.app.state.database_engine.connect() as connection:
        # C collation matches Python's ID comparison used by the exclusive cursor.
        stations = connection.execute(statement.order_by(Station.id.collate("C"))).mappings().all()
        candidates = select_station_days(stations, start, end, query)
        page = candidates[: query.limit]
        # Composite primary-key lookups fetch only the page's scheduled instants.
        # 10/20/30/40/50-minute records are never fetched or rounded to a clock hour.
        keys = [
            (station["id"], at) for day, station in page for at in clock_hours(day) if at <= now
        ]
        observations = []
        if keys:
            observations = (
                connection.execute(
                    sa.select(
                        Observation.station_id,
                        Observation.observed_at,
                        Observation.depth,
                        Observation.quality_status,
                    ).where(sa.tuple_(Observation.station_id, Observation.observed_at).in_(keys))
                )
                .mappings()
                .all()
            )

    readings = {(row["station_id"], row["observed_at"]): row for row in observations}
    rows = []
    for day, station in page:
        cells = []
        for hour, at in enumerate(clock_hours(day)):
            observation = readings.get((station["id"], at))
            value = None
            status = "missing"
            if at > now:
                status = "future"
            elif (
                observation
                and observation["quality_status"] == "observed"
                and observation["depth"] is not None
            ):
                value = observation["depth"]
                status = "observed"
            cells.append(HourlyHistoryCell(hour=hour, at=at, value=value, status=status))
        rows.append(
            HourlyHistoryRow(
                date=day,
                station_id=station["id"],
                name=station["name"],
                station_type=station["station_type"],
                unit=station["measurement_unit"],
                cells=cells,
            )
        )

    has_more = len(candidates) > query.limit
    next_cursor = None
    if has_more:
        day, station = page[-1]
        next_cursor = HourlyHistoryCursor(date=day, station_id=station["id"])
    return HourlyHistoryResponse(
        start_date=start,
        end_date=end,
        as_of=now,
        rows=rows,
        has_more=has_more,
        next_cursor=next_cursor,
    )
