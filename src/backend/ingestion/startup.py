"""Own one local ingestion child for the lifetime of the API process."""

import fcntl
import logging
import os
from pathlib import Path
import subprocess
import sys

logger = logging.getLogger(__name__)


class SyncWorkerProcess:
    """Manage an API-owned worker; no systemd service is started or enabled.

    Args:
        project_root: Project working directory containing src and private .env.

    A project-local advisory file lock prevents two API instances from spawning
    duplicate local workers. The child inherits the lock descriptor so it remains
    held until the child exits, even if its parent is killed. Worker parent watching
    stops the child after unexpected API exit. No credentials enter argv or logs.
    """

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.process: subprocess.Popen | None = None
        self.lock_file = None

    def start(self) -> bool:
        """Acquire ownership and launch the worker with the API's Python interpreter.

        Returns:
            True when this manager owns a child. False if another API owns the
            project lock or process creation fails. Existing database reads can
            still be served. No separate shell or detached system service is used.
        """
        if self.process is not None:
            return self.process.poll() is None
        try:
            directory = self.project_root / "reports"
            directory.mkdir(parents=True, exist_ok=True)
            self.lock_file = (directory / "hydrology_sync_worker.lock").open("a")
            fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            environment = {**os.environ, "HYDROLOGY_SYNC_PARENT_PID": str(os.getpid())}
            self.process = subprocess.Popen(
                [sys.executable, "-u", "-m", "src.backend.ingestion"],
                cwd=self.project_root,
                env=environment,
                pass_fds=(self.lock_file.fileno(),),
            )
        except BlockingIOError:
            self._release_lock()
            logger.warning("An API-owned ingestion worker already exists for this project.")
            return False
        except OSError as error:
            self._release_lock()
            logger.warning("Ingestion startup failed; error_type=%s", type(error).__name__)
            return False
        logger.info("API-owned ingestion worker started; pid=%s", self.process.pid)
        return True

    def stop(self) -> None:
        """Stop only the child owned by this API, then release its project lock.

        Returns:
            None. SIGTERM permits the worker to close/rollback active work; after
            forty seconds it is killed and reaped. An existing worker owned by
            another API or a manually started service is never stopped here.
        """
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=40)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            self.process = None
        self._release_lock()

    def _release_lock(self) -> None:
        if self.lock_file is not None:
            self.lock_file.close()
            self.lock_file = None
