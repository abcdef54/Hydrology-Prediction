"""Worker entrypoint isolates SOHOA failures from already committed DCU sync."""

from contextlib import redirect_stdout
import io
from unittest import TestCase
from unittest.mock import MagicMock, patch
from src.backend.ingestion.worker import main, SourceError, SyncSettings


class FacilityWorkerTests(TestCase):
    def run_once(self, facility_error=None):
        patches = {
            "load_dotenv": None,
            "SyncSettings.from_env": SyncSettings("https://source.invalid", "test", "test"),
            "build_engine": MagicMock(),
            "LatestObservationCache.from_env": MagicMock(),
            "HueSourceClient": MagicMock(),
            "sync_once": {"status": "committed", "stations": []},
            "write_report": None,
        }
        for name, result in patches.items():
            self.enterContext(patch(f"src.backend.ingestion.worker.{name}", return_value=result))
        self.enterContext(patch("src.backend.ingestion.worker.signal.signal"))
        self.enterContext(patch("sys.argv", ["worker", "--once"]))
        facility = self.enterContext(
            patch(
                "src.backend.ingestion.facility_import.sync_facilities",
                return_value={"status": "committed", "facilities": 19},
                side_effect=facility_error,
            )
        )
        output = io.StringIO()
        with redirect_stdout(output):
            status = main()
        return status, facility, output.getvalue()

    def test_entrypoint_runs_facility_sync_after_dcu_commit(self):
        status, facility, output = self.run_once()
        self.assertEqual(status, 0)
        facility.assert_called_once()
        self.assertIn('"facilities": 19', output)

    def test_facility_failure_keeps_dcu_commit_and_hides_source_error_text(self):
        status, facility, output = self.run_once(SourceError("private-token"))
        self.assertEqual(status, 1)
        self.assertIn('"status": "committed"', output)
        self.assertIn('"error_type": "SourceError"', output)
        self.assertNotIn("private-token", output)
