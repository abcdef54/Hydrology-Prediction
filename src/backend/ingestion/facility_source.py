"""Read SOHOA catalogs/reports with bounded paging and private token configuration."""

from dataclasses import dataclass, field
from datetime import date
import json
import os
from threading import Event

import requests
from src.backend.ingestion.worker import PROJECT_ROOT, SourceError, SyncCancelled

CATALOG_SERVICE = "5PKmlC8ZioLCs3OThNEN5A=="
OPERATIONS_SERVICE = "FBuJEaybjjCbAHM+znEEsA=="


@dataclass(frozen=True)
class FacilitySourceSettings:
    base_url: str
    token: str = field(repr=False)
    page_size: int = 1000

    @classmethod
    def from_env(cls):
        """Read server env; local fallback reads the existing private Postman environment.

        Returns:
            Settings with token excluded from repr/logs; no network requests.
        Raises:
            SourceError: Missing credentials or invalid local configuration.
        """
        url = os.environ.get("SOHOA_API_BASE_URL")
        token = os.environ.get("SOHOA_API_TOKEN")
        if not token:
            path = PROJECT_ROOT / "data/HueNgapLut.postman_environment.json"
            try:
                values = {
                    row["key"]: row.get("value", "")
                    for row in json.loads(path.read_text())["values"]
                }
            except (OSError, ValueError, KeyError, TypeError):
                raise SourceError("Configure SOHOA_API_TOKEN on the server.") from None
            token = values.get("sohoa_token")
            url = url or values.get("sohoa_base_url")
        if not token or not isinstance(token, str):
            raise SourceError("Configure SOHOA_API_TOKEN on the server.")
        url = (url or "https://sohoa.hue.gov.vn").rstrip("/")
        if not url.startswith(("https://", "http://")):
            raise SourceError("Invalid SOHOA API base URL.")
        return cls(url, token)


class FacilitySourceClient:
    """Token-authenticated read POSTs; never log request headers or source payloads."""

    def __init__(self, settings, stop=None, session=None):
        self.settings = settings
        self.stop = stop if stop is not None else Event()
        self.session = session if session is not None else requests.Session()
        self.session.headers.update({"token": settings.token, "Content-Type": "application/json"})

    def request_page(self, service, parameters, page):
        """Read one source page with three bounded retries for transient failures.

        Args:
            service: Published collection service identifier, not a credential.
            parameters: Source filters, including both date bounds and all statuses.
            page: One-based source page number.
        Returns:
            Array and totalrow; source business errors are never treated as empty data.
        Raises:
            SourceError: Auth/transport/JSON/schema/provider failure without secrets.
            SyncCancelled: Worker shutdown requested.
        """
        for attempt in range(3):
            if self.stop.is_set():
                raise SyncCancelled()
            try:
                response = self.session.post(
                    self.settings.base_url + "/data/danhsach",
                    json={
                        "serviceid": service,
                        "thamso": parameters,
                        "bangcon": "false",
                        "page": str(page),
                        "perpage": str(self.settings.page_size),
                    },
                    timeout=(10, 30),
                )
            except requests.RequestException:
                if attempt == 2:
                    raise SourceError("SOHOA transport failed after three attempts.") from None
                if self.stop.wait(1 + attempt):
                    raise SyncCancelled()
                continue
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == 2:
                    raise SourceError("SOHOA transient HTTP failure.", response.status_code)
                if self.stop.wait(min(30, 1 + attempt)):
                    raise SyncCancelled()
                continue
            if response.status_code != 200:
                raise SourceError("SOHOA request rejected.", response.status_code)
            try:
                payload = response.json()
            except ValueError:
                raise SourceError("SOHOA response is not JSON.") from None
            if not isinstance(payload, dict) or payload.get("code") not in (0, "0"):
                raise SourceError("SOHOA provider returned a business error.")
            total = payload.get("totalrow")
            if isinstance(total, str) and total.isdecimal():
                total = int(total)
            rows = payload.get("data")
            if type(total) is not int or total < 0 or not isinstance(rows, list):
                raise SourceError("SOHOA page shape/count is invalid.")
            return rows, total

    def read_all(self, service, parameters):
        """Require stable totalrow and unique IDs across every page; do not truncate silently.

        Args:
            service: Metadata or report service identifier.
            parameters: Fixed filters reused unchanged for every page.
        Returns:
            Complete snapshot with unique source IDs, including deleted reports.
        Raises:
            SourceError: Moving count, duplicate ID, premature empty page or page limit.
            SyncCancelled: Shutdown requested.
        """
        rows, seen, expected = [], set(), None
        for page in range(1, 2001):
            batch, total = self.request_page(service, parameters, page)
            if expected is None:
                expected = total
            if total != expected:
                raise SourceError("SOHOA source changed during paging; rerun snapshot.")
            for record in batch:
                if (
                    not isinstance(record, dict)
                    or type(record.get("id")) is not int
                    or record["id"] <= 0
                ):
                    raise SourceError("Invalid SOHOA record identity.")
                if record["id"] in seen:
                    raise SourceError("Duplicate SOHOA ID during paging; rerun snapshot.")
                seen.add(record["id"])
                rows.append(record)
            if len(rows) == expected:
                return rows
            if not batch or len(rows) > expected:
                raise SourceError("SOHOA totalrow does not match pages.")
        raise SourceError("SOHOA pagination exceeded its bound.")

    def catalog(self):
        rows = self.read_all(CATALOG_SERVICE, {"tukhoa": "", "loaicongtrinhid": "0"})
        if not rows:
            raise SourceError("SOHOA catalog is empty; refusing incomplete metadata.")
        return rows

    def operations(self, start: date, end: date):
        """Read inclusive Vietnam calendar dates, all functions/statuses and tombstones.

        Args:
            start/end: Inclusive source calendar bounds; source dates use Vietnam time.
        Returns:
            Complete reports with original fields; source user fields are stripped
            by parse_operation before persistence or API exposure.
        Raises:
            SourceError, SyncCancelled: Auth/paging/transport failure or shutdown.
        """
        return self.read_all(
            OPERATIONS_SERVICE,
            {
                "tukhoa": "",
                "chucnang": "0",
                "loaicongtrinh": "0",
                "tencongtrinh": "0",
                "trangthai": "0",
                "tungaylaysolieu": start.strftime("%d/%m/%Y"),
                "denngaylaysolieu": end.strftime("%d/%m/%Y"),
            },
        )
