"""API-owned worker lifetime and suspend-aware scheduling without real database writes."""

from datetime import datetime, timedelta, timezone
import subprocess
from threading import Event
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.backend.ingestion.startup import SyncWorkerProcess
from src.backend.ingestion.worker import wait_for_next_cycle, watch_api_parent
from src.backend.main import create_app
from src.backend.prediction.config import StationModelRegistry


class SyncStartupTests(TestCase):
    def test_child_owned_start_is_idempotent_and_second_manager_cannot_duplicate(self):
        with TemporaryDirectory() as directory, patch(
            "src.backend.ingestion.startup.subprocess.Popen"
        ) as spawn:
            child = spawn.return_value
            child.poll.return_value = None
            first = SyncWorkerProcess(Path(directory))
            second = SyncWorkerProcess(Path(directory))
            try:
                self.assertTrue(first.start())
                self.assertTrue(first.start())
                self.assertFalse(second.start())
                spawn.assert_called_once()
                self.assertEqual(spawn.call_args.args[0][1:], ["-u", "-m", "src.backend.ingestion"])
                self.assertIn("HYDROLOGY_SYNC_PARENT_PID", spawn.call_args.kwargs["env"])
                self.assertTrue(spawn.call_args.kwargs["pass_fds"])
                second.stop()
                child.terminate.assert_not_called()
            finally:
                first.stop()
            child.terminate.assert_called_once()
            child.wait.assert_called_once_with(timeout=40)

    def test_spawn_failure_releases_lock_and_timeout_escalates_only_owned_child(self):
        with TemporaryDirectory() as directory:
            owner = SyncWorkerProcess(Path(directory))
            with patch(
                "src.backend.ingestion.startup.subprocess.Popen", side_effect=OSError("private")
            ):
                self.assertFalse(owner.start())
                self.assertIsNone(owner.lock_file)
            with patch("src.backend.ingestion.startup.subprocess.Popen") as spawn:
                child = spawn.return_value
                child.poll.return_value = None
                child.wait.side_effect = [subprocess.TimeoutExpired("worker", 40), 0]
                self.assertTrue(owner.start())
                owner.stop()
                child.kill.assert_called_once()
                self.assertIsNone(owner.lock_file)

    def test_api_lifespan_starts_and_stops_owned_worker_once(self):
        app = create_app(engine=MagicMock(), registry=StationModelRegistry([]), sync_autostart=True)
        with patch("src.backend.main.SyncWorkerProcess") as manager:
            worker = manager.return_value
            worker.start.return_value = True
            with TestClient(app):
                self.assertEqual(app.state.sync_startup_status, "requested")
                worker.start.assert_called_once_with()
                worker.stop.assert_not_called()
            worker.stop.assert_called_once_with()

    def test_worker_failure_does_not_block_api_and_injected_engine_defaults_off(self):
        app = create_app(engine=MagicMock(), registry=StationModelRegistry([]), sync_autostart=True)
        with patch("src.backend.main.SyncWorkerProcess") as manager:
            manager.return_value.start.return_value = False
            with TestClient(app) as client:
                self.assertEqual(client.get("/openapi.json").status_code, 200)
                self.assertEqual(app.state.sync_startup_status, "unavailable")
        with patch.dict("os.environ", {"HYDROLOGY_SYNC_AUTOSTART": "true"}):
            app = create_app(engine=MagicMock(), registry=StationModelRegistry([]))
            with patch("src.backend.main.SyncWorkerProcess") as manager:
                with TestClient(app):
                    self.assertEqual(app.state.sync_startup_status, "disabled")
                manager.assert_not_called()

    def test_worker_stops_if_api_parent_disappears(self):
        stop, wake = Event(), Event()
        with patch("src.backend.ingestion.worker.os.getppid", return_value=1):
            watch_api_parent(99999, stop, wake)
        self.assertTrue(stop.is_set())
        self.assertTrue(wake.is_set())


class SyncScheduleTests(TestCase):
    def setUp(self):
        self.now = datetime(2026, 10, 14, 6, tzinfo=timezone.utc)
        self.stop = Event()
        self.wake = MagicMock(spec=Event)
        self.wake.wait.return_value = False

    def test_elapsed_wall_deadline_after_suspend_runs_without_waiting(self):
        with patch("src.backend.ingestion.worker.datetime") as clock:
            clock.now.return_value = self.now + timedelta(hours=1)
            wait_for_next_cycle(self.stop, self.wake, self.now + timedelta(minutes=11))
        self.wake.wait.assert_not_called()

    def test_wall_time_is_checked_in_short_waits_and_wake_is_consumed(self):
        with patch("src.backend.ingestion.worker.datetime") as clock:
            clock.now.side_effect = [self.now, self.now + timedelta(hours=1)]
            wait_for_next_cycle(self.stop, self.wake, self.now + timedelta(minutes=11))
        self.wake.wait.assert_called_once_with(5.0)
        self.wake.reset_mock()
        self.wake.wait.return_value = True
        with patch("src.backend.ingestion.worker.datetime") as clock:
            clock.now.return_value = self.now
            wait_for_next_cycle(self.stop, self.wake, self.now + timedelta(minutes=11))
        self.wake.clear.assert_called_once()

    def test_shutdown_skips_wait(self):
        self.stop.set()
        wait_for_next_cycle(self.stop, self.wake, self.now + timedelta(minutes=11))
        self.wake.wait.assert_not_called()
