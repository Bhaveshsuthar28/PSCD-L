"""Background processing worker that executes a single job end-to-end."""

from __future__ import annotations

import mimetypes
from pathlib import Path
import threading
import time
from typing import Any

import cv2
import torch

from backend.app.core.config import Settings
from backend.app.core.exceptions import ModelInitializationError
from backend.app.core.logging import get_logger, job_id_ctx, video_id_ctx
from backend.app.database.connection import DatabaseManager
from backend.app.database.models import Event, Evidence, ProcessingJob, Video
from backend.app.database.repositories import (
    EventRepository,
    EvidenceRepository,
    JobRepository,
)
from backend.app.integrations.trinetra_engine import ManagedInferenceEngine
from backend.app.integrations.video_runner_adapter import VideoRunnerAdapter
from backend.app.services.storage_service import StorageService
from backend.app.utils.serialization import safe_float, sanitize_for_json
from backend.app.workers.progress_manager import ProgressManager
from src.inference.video_runner import VideoProcessingCancelled


logger = get_logger("workers.processing")


EVIDENCE_KEY_TO_TYPE = {
    "original": "original_frame",
    "annotated": "annotated_frame",
    "probability": "probability_map",
    "metadata": "metadata",
}


class ProcessingWorker:
    """Executes queued TRINETRA processing jobs safely with Task 2 integration."""

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
        self.adapter = VideoRunnerAdapter(settings, managed_engine)

    @staticmethod
    def _build_event_description(raw: dict[str, Any]) -> str:
        if raw.get("was_confirmed"):
            if raw.get("review_required"):
                return (
                    "Persistent scene-change event (historically confirmed, "
                    "flagged for human review due to post-confirmation degradation)."
                )
            return "Persistent scene-change event (visual change only; does not establish object identity or ownership)."
        return "Candidate visual scene-change track."

    @staticmethod
    def _image_dimensions(path: Path) -> tuple[int | None, int | None]:
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            return None, None
        img = cv2.imread(str(path))
        if img is None:
            return None, None
        h, w = img.shape[:2]
        return int(w), int(h)

    @staticmethod
    def _mime_type_for_path(path: Path) -> str:
        guessed, _ = mimetypes.guess_type(str(path))
        if guessed:
            return guessed
        ext = path.suffix.lower()
        if ext == ".json":
            return "application/json"
        if ext == ".csv":
            return "text/csv"
        return "application/octet-stream"

    def _sanitize_summary_for_api(self, summary: dict[str, Any], output_dir: Path) -> dict[str, Any]:
        """Replace absolute internal filesystem paths in run summary with safe relative names."""
        cleaned = sanitize_for_json(summary)
        # Never expose absolute filesystem path of video or output directory.
        cleaned.pop("video", None)
        cleaned.pop("summary_file", None)
        event_files = cleaned.get("event_files")
        if isinstance(event_files, dict):
            safe_event_files = {}
            for fmt, raw_path in event_files.items():
                if isinstance(raw_path, str):
                    p = Path(raw_path)
                    try:
                        safe_event_files[fmt] = self.storage.to_relative_path(p)
                    except Exception:
                        safe_event_files[fmt] = p.name
            cleaned["event_files"] = safe_event_files
        cleaned["output_directory"] = self.storage.to_relative_path(output_dir)
        return cleaned

    def process_job(self, job_id: str, cancel_event: threading.Event) -> None:
        job_token = job_id_ctx.set(job_id)
        vid_token = None
        acquired_slot = False

        try:
            # 1. Load job & transition QUEUED -> VALIDATING
            with self.db_manager.session() as db:
                job_repo = JobRepository(db)
                job = job_repo.get(job_id)
                if job is None:
                    logger.warning("Job %s not found when worker picked it up", job_id)
                    return

                vid_token = video_id_ctx.set(job.video_id)

                if job.status in {"CANCEL_REQUESTED", "CANCELLED"} or cancel_event.is_set():
                    if job.status != "CANCELLED":
                        job_repo.transition_status(job, "CANCELLED", stage="CANCELLED")
                        job_repo.add_log(job_id, "Job cancelled before execution started.", stage="CANCELLED")
                    self.progress_manager.publish(
                        job_id,
                        "cancelled",
                        {"status": "CANCELLED", "stage": "CANCELLED", "message": "Job cancelled before start."},
                    )
                    return

                if job.status != "QUEUED":
                    logger.info("Skipping job %s with status %s", job_id, job.status)
                    return

                job_repo.transition_status(job, "VALIDATING", stage="VALIDATING")
                job_repo.add_log(job_id, "Validating video file and job configuration.", stage="VALIDATING")

                video = db.get(Video, job.video_id)
                if video is None or video.upload_status == "DELETED":
                    raise FileNotFoundError("Associated video record is missing or deleted.")

                video_path = self.storage.resolve_relative_path(video.relative_storage_path)
                if not video_path.is_file():
                    raise FileNotFoundError("Uploaded video file is missing from storage.")

                output_dir = self.storage.resolve_relative_path(job.output_directory)
                output_dir.mkdir(parents=True, exist_ok=True)

                job_params = {
                    "video_id": job.video_id,
                    "video_path": video_path,
                    "output_dir": output_dir,
                    "reference_seconds": job.reference_seconds,
                    "start_seconds": job.start_seconds,
                    "end_seconds": job.end_seconds,
                    "sampling_interval_seconds": job.sampling_interval_seconds,
                    "configuration_snapshot": dict(job.configuration_snapshot or {}),
                }

            self.progress_manager.publish(
                job_id,
                "stage_changed",
                {"status": "VALIDATING", "stage": "VALIDATING"},
            )

            if cancel_event.is_set():
                raise VideoProcessingCancelled("Cancelled during validation.")

            # 2. Acquire GPU concurrency slot and transition VALIDATING -> INITIALIZING
            self.managed_engine.acquire_execution_slot()
            acquired_slot = True

            with self.db_manager.session() as db:
                job_repo = JobRepository(db)
                job = job_repo.get(job_id)
                if job is None:
                    return
                if cancel_event.is_set() or job.status == "CANCEL_REQUESTED":
                    raise VideoProcessingCancelled("Cancelled prior to model initialization.")
                job_repo.transition_status(job, "INITIALIZING", stage="LOADING_MODEL")
                job_repo.add_log(job_id, "Initializing TRINETRA inference engine.", stage="LOADING_MODEL")

            self.progress_manager.publish(
                job_id,
                "stage_changed",
                {"status": "INITIALIZING", "stage": "LOADING_MODEL"},
            )

            # Initialize/verify frozen model outside DB lock
            self.managed_engine.acquire_engine()

            if cancel_event.is_set():
                raise VideoProcessingCancelled("Cancelled after model initialization.")

            with self.db_manager.session() as db:
                job_repo = JobRepository(db)
                job = job_repo.get(job_id)
                if job is None:
                    return
                if job.status == "CANCEL_REQUESTED":
                    raise VideoProcessingCancelled("Cancelled before frame reading.")
                job_repo.transition_status(job, "PROCESSING", stage="READING_VIDEO")
                job_repo.add_log(job_id, "Starting Task 2 temporal video processing.", stage="READING_VIDEO")

            self.progress_manager.publish(
                job_id,
                "stage_changed",
                {"status": "PROCESSING", "stage": "READING_VIDEO"},
            )

            # 3. Execute Task 2 VideoRunner with throttled DB progress + real-time SSE
            run_start = time.perf_counter()
            last_db_persist = 0.0
            last_persisted_frames = 0

            def on_progress(info: dict[str, Any]) -> None:
                nonlocal last_db_persist, last_persisted_frames
                now = time.perf_counter()
                elapsed = now - run_start
                processed = int(info.get("processed_frames", 0))
                total = info.get("total_frames")
                fps = (processed / elapsed) if elapsed > 0 and processed > 0 else None

                pct: float | None = None
                eta: float | None = None
                if isinstance(total, int) and total > 0:
                    pct = round(min(99.0, (processed / total) * 100.0), 2)
                    if fps and fps > 0 and processed >= 2 and total > processed:
                        eta = round((total - processed) / fps, 2)

                progress_payload = {
                    "status": "CANCEL_REQUESTED" if cancel_event.is_set() else "PROCESSING",
                    "stage": info.get("stage", "PROCESSING_FRAMES"),
                    "processed_frames": processed,
                    "total_frames": total,
                    "progress_percentage": pct,
                    "current_video_timestamp_seconds": safe_float(info.get("timestamp_seconds")),
                    "processing_fps": round(fps, 3) if fps is not None else None,
                    "elapsed_seconds": round(elapsed, 3),
                    "estimated_remaining_seconds": eta,
                    "active_tracks_count": int(info.get("active_tracks", 0)),
                    "confirmed_events_count": int(info.get("confirmed_events", 0)),
                    "review_required_events_count": int(info.get("review_required_events", 0)),
                    "total_candidates_count": int(info.get("total_candidates", 0)),
                }

                self.progress_manager.publish(job_id, "progress", progress_payload)

                should_persist = (
                    (now - last_db_persist) >= self.settings.progress_db_persist_interval_seconds
                    or (processed - last_persisted_frames) >= self.settings.progress_db_persist_frame_batch
                )
                if should_persist:
                    last_db_persist = now
                    last_persisted_frames = processed
                    self._persist_progress(job_id, progress_payload, job_params["start_seconds"])

            def on_event(evt: dict[str, Any]) -> None:
                evt_type = evt.get("type", "event_confirmed")
                self.progress_manager.publish(job_id, evt_type, evt)

            result = self.adapter.execute(
                video_path=job_params["video_path"],
                output_dir=job_params["output_dir"],
                reference_seconds=job_params["reference_seconds"],
                start_seconds=job_params["start_seconds"],
                end_seconds=job_params["end_seconds"],
                sampling_interval_seconds=job_params["sampling_interval_seconds"],
                configuration_snapshot=job_params["configuration_snapshot"],
                progress_callback=on_progress,
                cancel_check=cancel_event.is_set,
                event_callback=on_event,
            )

            # Check cooperative cancellation before finalizing as COMPLETED.
            if cancel_event.is_set():
                raise VideoProcessingCancelled("Cancelled prior to finalization.")

            # 4. Persist events, evidence, and run summary transactionally
            with self.db_manager.session() as db:
                job_repo = JobRepository(db)
                job = job_repo.get(job_id)
                if job is None:
                    return
                if job.status == "CANCEL_REQUESTED" or cancel_event.is_set():
                    raise VideoProcessingCancelled("Cancelled prior to finalization.")

                job_repo.transition_status(job, "FINALIZING", stage="FINALIZING_EVENTS")
                self.progress_manager.publish(
                    job_id,
                    "stage_changed",
                    {"status": "FINALIZING", "stage": "FINALIZING_EVENTS"},
                )

                summary = result["summary"]
                raw_events: list[dict[str, Any]] = result["events"]
                evidence_errors: list[dict[str, Any]] = result.get("evidence_errors", [])

                self._persist_events_and_evidence(
                    db=db,
                    job=job,
                    raw_events=raw_events,
                    summary=summary,
                    evidence_errors=evidence_errors,
                )

                processed_frames = int(summary.get("processed_frames", 0))
                elapsed = float(result["elapsed_seconds"])
                fps = (processed_frames / elapsed) if elapsed > 0 and processed_frames > 0 else None

                job.processed_frames = processed_frames
                job.total_frames = processed_frames
                job.processed_seconds = float(
                    max(0.0, float(summary.get("processing_end_seconds", job.start_seconds)) - job.start_seconds)
                )
                job.processing_fps = round(fps, 4) if fps is not None else None
                job.elapsed_seconds = round(elapsed, 4)
                job.estimated_remaining_seconds = 0.0
                job.active_tracks_count = 0
                job.total_candidates_count = int(summary.get("total_candidates", 0))
                job.confirmed_events_count = int(summary.get("confirmed_event_count", 0))
                job.review_required_events_count = int(summary.get("review_required_event_count", 0))
                job.summary_snapshot = self._sanitize_summary_for_api(
                    summary, job_params["output_dir"]
                )

                if evidence_errors:
                    job.evidence_generation_status = "PARTIAL_FAILED"
                    job.evidence_generation_error = "; ".join(
                        f"track {item.get('track_id')}: {item.get('error')}"
                        for item in evidence_errors
                    )[:1000]
                else:
                    job.evidence_generation_status = "COMPLETED"
                    job.evidence_generation_error = None

                job_repo.transition_status(job, "COMPLETED", stage="COMPLETED")
                job_repo.add_log(
                    job_id,
                    f"Processing completed: {len(raw_events)} tracks, "
                    f"{job.confirmed_events_count} confirmed, "
                    f"{job.review_required_events_count} review-required.",
                    stage="COMPLETED",
                )

            self.progress_manager.publish(
                job_id,
                "completed",
                {
                    "status": "COMPLETED",
                    "stage": "COMPLETED",
                    "progress_percentage": 100.0,
                    "processed_frames": processed_frames,
                    "total_frames": processed_frames,
                    "confirmed_events_count": int(summary.get("confirmed_event_count", 0)),
                    "review_required_events_count": int(summary.get("review_required_event_count", 0)),
                    "elapsed_seconds": round(elapsed, 4),
                },
            )
            logger.info("Job %s completed successfully", job_id, extra={"operation": "job_complete"})

        except VideoProcessingCancelled as exc:
            self._handle_cancellation(job_id, str(exc))
        except torch.cuda.OutOfMemoryError as exc:
            self.managed_engine.clear_reference()
            self._handle_failure(
                job_id,
                error_code="CUDA_OUT_OF_MEMORY",
                error_message="GPU ran out of memory during inference.",
                exc=exc,
            )
        except ModelInitializationError as exc:
            self._handle_failure(
                job_id,
                error_code=exc.error_code,
                error_message=exc.message,
                exc=exc,
            )
        except FileNotFoundError as exc:
            self._handle_failure(
                job_id,
                error_code="INPUT_OR_CHECKPOINT_MISSING",
                error_message=str(exc),
                exc=exc,
            )
        except Exception as exc:
            self._handle_failure(
                job_id,
                error_code="PROCESSING_FAILED",
                error_message=f"Processing failed: {type(exc).__name__}",
                exc=exc,
            )
        finally:
            if acquired_slot:
                self.managed_engine.release_execution_slot()
            if vid_token is not None:
                video_id_ctx.reset(vid_token)
            job_id_ctx.reset(job_token)

    def _persist_progress(self, job_id: str, payload: dict[str, Any], start_seconds: float) -> None:
        try:
            with self.db_manager.session() as db:
                job = JobRepository(db).get(job_id)
                if job is None or job.status in {"COMPLETED", "FAILED", "CANCELLED"}:
                    return
                job.current_stage = str(payload.get("stage", job.current_stage))
                job.processed_frames = int(payload.get("processed_frames", job.processed_frames))
                if payload.get("total_frames") is not None:
                    job.total_frames = int(payload["total_frames"])
                job.progress_percentage = payload.get("progress_percentage")
                ts = payload.get("current_video_timestamp_seconds")
                if ts is not None:
                    job.current_video_timestamp_seconds = float(ts)
                    job.processed_seconds = max(0.0, float(ts) - start_seconds)
                job.processing_fps = payload.get("processing_fps")
                job.elapsed_seconds = payload.get("elapsed_seconds")
                job.estimated_remaining_seconds = payload.get("estimated_remaining_seconds")
                job.active_tracks_count = int(payload.get("active_tracks_count", 0))
                job.confirmed_events_count = int(payload.get("confirmed_events_count", 0))
                job.review_required_events_count = int(payload.get("review_required_events_count", 0))
                job.total_candidates_count = int(payload.get("total_candidates_count", 0))
        except Exception as exc:
            logger.warning("Failed to persist throttled progress for job %s: %s", job_id, exc)

    def _persist_events_and_evidence(
        self,
        *,
        db,
        job: ProcessingJob,
        raw_events: list[dict[str, Any]],
        summary: dict[str, Any],
        evidence_errors: list[dict[str, Any]],
    ) -> None:
        event_repo = EventRepository(db)
        evidence_repo = EvidenceRepository(db)

        for raw in raw_events:
            track_id = int(raw["track_id"])
            timeline = sanitize_for_json(raw.get("evidence") or [])
            latest_bbox = None
            if isinstance(timeline, list):
                for entry in reversed(timeline):
                    if isinstance(entry, dict) and entry.get("bbox_xywh"):
                        latest_bbox = entry["bbox_xywh"]
                        break

            event_model = Event(
                job_id=job.id,
                video_id=job.video_id,
                track_id=track_id,
                event_state=str(raw["state"]),
                first_seen_seconds=float(raw["first_seen_seconds"]),
                last_seen_seconds=float(raw["last_seen_seconds"]),
                duration_seconds=float(raw["duration_seconds"]),
                observations=int(raw.get("observations", 0)),
                missed_observations=int(raw.get("missed_observations", 0)),
                total_missed_observations=int(raw.get("total_missed_observations", 0)),
                detection_consistency=safe_float(raw.get("detection_consistency")) or 0.0,
                center_deviation_pixels=safe_float(raw.get("center_deviation_pixels")) or 0.0,
                bbox_stability=safe_float(raw.get("bbox_stability")) or 0.0,
                motion_samples=int(raw.get("motion_samples", 0)),
                motion_median_ratio=safe_float(raw.get("motion_median_ratio")),
                motion_p90_ratio=safe_float(raw.get("motion_p90_ratio")),
                settling_motion_samples=int(raw.get("settling_motion_samples", 0)),
                settling_motion_median_ratio=safe_float(raw.get("settling_motion_median_ratio")),
                settling_motion_p90_ratio=safe_float(raw.get("settling_motion_p90_ratio")),
                confidence=safe_float(raw.get("confidence")) or 0.0,
                confidence_type=str(
                    raw.get("confidence_kind", "heuristic_score_not_probability")
                ),
                was_confirmed=bool(raw.get("was_confirmed", False)),
                confirmation_seconds=safe_float(raw.get("confirmation_seconds")),
                review_required=bool(raw.get("review_required", False)),
                post_confirmation_degraded=bool(raw.get("post_confirmation_degraded", False)),
                degradation_first_seen_seconds=safe_float(
                    raw.get("degradation_first_seen_seconds")
                ),
                post_confirmation_degradation_reasons=list(
                    raw.get("post_confirmation_degradation_reasons") or []
                ),
                boundary_candidate=bool(raw.get("boundary_candidate", False)),
                latest_bbox_xywh=latest_bbox,
                state_history=sanitize_for_json(raw.get("state_history") or []),
                evidence_timeline=timeline,
                description=self._build_event_description(raw),
            )
            db.add(event_model)
            db.flush()

            for path_bundle in raw.get("evidence_paths") or []:
                if not isinstance(path_bundle, dict):
                    continue
                for key, raw_path in path_bundle.items():
                    if not raw_path or key not in EVIDENCE_KEY_TO_TYPE:
                        continue
                    file_path = Path(raw_path).resolve()
                    if not file_path.is_file():
                        evidence_errors.append(
                            {"track_id": track_id, "error": f"Missing evidence file for {key}"}
                        )
                        continue
                    rel_path = self.storage.to_relative_path(file_path)
                    if evidence_repo.get_by_job_and_relpath(job.id, rel_path) is not None:
                        continue
                    w, h = self._image_dimensions(file_path)
                    ev = Evidence(
                        event_id=event_model.id,
                        job_id=job.id,
                        evidence_type=EVIDENCE_KEY_TO_TYPE[key],
                        relative_storage_path=rel_path,
                        original_filename=file_path.name,
                        mime_type=self._mime_type_for_path(file_path),
                        file_size_bytes=file_path.stat().st_size,
                        sha256=self.storage.compute_file_sha256(file_path),
                        timestamp_seconds=event_model.confirmation_seconds,
                        width=w,
                        height=h,
                        bbox_xywh=latest_bbox,
                    )
                    evidence_repo.create(ev)

        # Register job-level exported reports (events.json, events.csv) if present
        event_files = summary.get("event_files") or {}
        if isinstance(event_files, dict):
            for _, raw_path in event_files.items():
                if not raw_path:
                    continue
                file_path = Path(raw_path).resolve()
                if not file_path.is_file():
                    continue
                rel_path = self.storage.to_relative_path(file_path)
                if evidence_repo.get_by_job_and_relpath(job.id, rel_path) is not None:
                    continue
                evidence_repo.create(
                    Evidence(
                        event_id=None,
                        job_id=job.id,
                        evidence_type="exported_report",
                        relative_storage_path=rel_path,
                        original_filename=file_path.name,
                        mime_type=self._mime_type_for_path(file_path),
                        file_size_bytes=file_path.stat().st_size,
                        sha256=self.storage.compute_file_sha256(file_path),
                    )
                )

    def _handle_cancellation(self, job_id: str, reason: str) -> None:
        logger.info("Job %s cancelled: %s", job_id, reason, extra={"operation": "job_cancel"})
        try:
            with self.db_manager.session() as db:
                repo = JobRepository(db)
                job = repo.get(job_id)
                if job is not None and job.status not in {"COMPLETED", "FAILED", "CANCELLED"}:
                    repo.transition_status(
                        job,
                        "CANCELLED",
                        stage="CANCELLED",
                        error_code="CANCELLED",
                        error_message=reason,
                    )
                    repo.add_log(job_id, reason, level="WARNING", stage="CANCELLED")
        finally:
            self.progress_manager.publish(
                job_id,
                "cancelled",
                {"status": "CANCELLED", "stage": "CANCELLED", "message": reason},
            )

    def _handle_failure(
        self,
        job_id: str,
        *,
        error_code: str,
        error_message: str,
        exc: Exception,
    ) -> None:
        logger.exception(
            "Job %s failed (%s): %s",
            job_id,
            error_code,
            exc,
            extra={"operation": "job_fail", "error_code": error_code},
        )
        try:
            with self.db_manager.session() as db:
                repo = JobRepository(db)
                job = repo.get(job_id)
                if job is not None and job.status not in {"COMPLETED", "FAILED", "CANCELLED"}:
                    repo.transition_status(
                        job,
                        "FAILED",
                        stage="FAILED",
                        error_code=error_code,
                        error_message=error_message,
                    )
                    repo.add_log(
                        job_id,
                        error_message,
                        level="ERROR",
                        stage="FAILED",
                        details={"error_code": error_code, "exception_type": type(exc).__name__},
                    )
        finally:
            self.progress_manager.publish(
                job_id,
                "failed",
                {
                    "status": "FAILED",
                    "stage": "FAILED",
                    "error_code": error_code,
                    "error_message": error_message,
                },
            )
