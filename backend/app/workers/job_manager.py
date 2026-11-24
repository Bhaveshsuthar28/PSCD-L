"""Bounded background job queue, worker lifecycle, cancellation, and crash recovery."""

from __future__ import annotations

import queue
import threading

from backend.app.core.config import Settings
from backend.app.core.exceptions import (
    ConflictError,
    QueueFullError,
    ServiceUnavailableError,
)
from backend.app.core.logging import get_logger
from backend.app.database.connection import DatabaseManager
from backend.app.database.models import TERMINAL_JOB_STATUSES
from backend.app.database.repositories import JobRepository
from backend.app.integrations.trinetra_engine import ManagedInferenceEngine
from backend.app.services.storage_service import StorageService
from backend.app.workers.processing_worker import ProcessingWorker
from backend.app.workers.progress_manager import ProgressManager


logger = get_logger("workers.job_manager")


class JobManager:
    """Manages bounded job queue, worker thread(s), cooperative cancellation, and recovery."""

    def __init__(
        self,
        settings: Settings,
        db_manager: DatabaseManager,
        storage: StorageService,
        managed_engine: ManagedInferenceEngine,
        progress_manager: ProgressManager,
    ) -> None:
        self.settings = settings
        self.db_manager = db_manager
        self.storage = storage
        self.managed_engine = managed_engine
        self.progress_manager = progress_manager
        self.worker = ProcessingWorker(
            settings=settings,
            db_manager=db_manager,
            storage=storage,
            managed_engine=managed_engine,
            progress_manager=progress_manager,
        )

        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=settings.max_queue_size)
        self._lock = threading.RLock()
        self._tracked_jobs: set[str] = set()
        self._cancel_flags: dict[str, threading.Event] = {}
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._started = False

    def is_running(self) -> bool:
        with self._lock:
            return self._started and any(t.is_alive() for t in self._threads)

    def recover_interrupted_jobs(self) -> int:
        """
        Mark any jobs left active by a previous crash/restart as FAILED.
        Never marks interrupted jobs as COMPLETED.
        """
        recovered = 0
        with self.db_manager.session() as db:
            repo = JobRepository(db)
            interrupted = repo.list_interrupted_active()
            for job in interrupted:
                prior_status = job.status
                repo.transition_status(
                    job,
                    "FAILED",
                    stage="INTERRUPTED",
                    error_code="INTERRUPTED_BY_RESTART",
                    error_message=(
                        f"Job was interrupted while in state '{prior_status}' "
                        "due to application restart or worker shutdown."
                    ),
                )
                repo.add_log(
                    job.id,
                    f"Recovered interrupted job from state {prior_status} -> FAILED.",
                    level="WARNING",
                    stage="INTERRUPTED",
                )
                recovered += 1
        if recovered:
            logger.warning(
                "Recovered %d interrupted processing job(s) as FAILED",
                recovered,
                extra={"operation": "job_recovery"},
            )
        return recovered

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._stop_event.clear()
            self.recover_interrupted_jobs()
            self._threads = []
            for idx in range(self.settings.worker_count):
                t = threading.Thread(
                    target=self._worker_loop,
                    name=f"trinetra-worker-{idx}",
                    daemon=True,
                )
                t.start()
                self._threads.append(t)
            self._started = True
            logger.info(
                "JobManager started with %d worker(s)",
                len(self._threads),
                extra={"operation": "worker_start"},
            )

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is None:
                self._queue.task_done()
                break

            job_id = item
            try:
                with self._lock:
                    cancel_event = self._cancel_flags.setdefault(job_id, threading.Event())
                self.worker.process_job(job_id, cancel_event)
            except Exception as exc:
                logger.exception("Unhandled worker loop error for job %s: %s", job_id, exc)
            finally:
                with self._lock:
                    self._tracked_jobs.discard(job_id)
                    self._cancel_flags.pop(job_id, None)
                self._queue.task_done()

    def enqueue(self, job_id: str) -> None:
        with self._lock:
            if not self.is_running():
                raise ServiceUnavailableError(
                    "Background processing worker is not running.",
                    error_code="WORKER_UNAVAILABLE",
                )
            if job_id in self._tracked_jobs:
                raise ConflictError(
                    f"Job '{job_id}' is already queued or executing.",
                    error_code="DUPLICATE_JOB_EXECUTION",
                )
            cancel_event = threading.Event()
            try:
                self._queue.put_nowait(job_id)
            except queue.Full as exc:
                raise QueueFullError(
                    "Processing queue is at capacity. Please retry later.",
                    details={"max_queue_size": self.settings.max_queue_size},
                ) from exc
            self._tracked_jobs.add(job_id)
            self._cancel_flags[job_id] = cancel_event

    def request_cancel(self, job_id: str) -> dict[str, str]:
        with self._lock:
            cancel_event = self._cancel_flags.get(job_id)
            if cancel_event is not None:
                cancel_event.set()

        with self.db_manager.session() as db:
            repo = JobRepository(db)
            job = repo.get(job_id)
            if job is None:
                raise ConflictError("Job not found.", error_code="JOB_NOT_FOUND")
            if job.status in TERMINAL_JOB_STATUSES:
                raise ConflictError(
                    f"Cannot cancel job in terminal state '{job.status}'.",
                    error_code="JOB_ALREADY_TERMINAL",
                )
            if job.status == "QUEUED":
                repo.transition_status(
                    job,
                    "CANCELLED",
                    stage="CANCELLED",
                    error_code="CANCELLED",
                    error_message="Job cancelled before processing started.",
                )
                repo.add_log(job_id, "Job cancelled while queued.", level="WARNING", stage="CANCELLED")
                self.progress_manager.publish(
                    job_id,
                    "cancelled",
                    {"status": "CANCELLED", "stage": "CANCELLED", "message": "Cancelled while queued."},
                )
                return {
                    "job_id": job_id,
                    "status": "CANCELLED",
                    "current_stage": "CANCELLED",
                    "message": "Queued job cancelled.",
                }

            if job.status != "CANCEL_REQUESTED":
                repo.transition_status(job, "CANCEL_REQUESTED", stage="CANCEL_REQUESTED")
                repo.add_log(
                    job_id,
                    "Cooperative cancellation requested.",
                    level="WARNING",
                    stage="CANCEL_REQUESTED",
                )
                self.progress_manager.publish(
                    job_id,
                    "stage_changed",
                    {"status": "CANCEL_REQUESTED", "stage": "CANCEL_REQUESTED"},
                )
            return {
                "job_id": job_id,
                "status": job.status,
                "current_stage": job.current_stage,
                "message": "Cooperative cancellation requested; job will stop at next checkpoint.",
            }

    def shutdown(self, timeout_seconds: float = 10.0) -> None:
        with self._lock:
            if not self._started:
                return
            self._stop_event.set()
            for ev in self._cancel_flags.values():
                ev.set()
            threads = list(self._threads)
            for _ in threads:
                try:
                    self._queue.put_nowait(None)
                except queue.Full:
                    pass

        for t in threads:
            t.join(timeout=timeout_seconds)

        with self._lock:
            self._threads.clear()
            self._started = False

        self.progress_manager.shutdown()
        self.managed_engine.unload()
        logger.info("JobManager shutdown complete", extra={"operation": "worker_shutdown"})
