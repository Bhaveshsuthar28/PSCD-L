"""Processing job creation, validation, listing, cancellation, and summary service."""

from __future__ import annotations

import uuid

from backend.app.core.config import Settings
from backend.app.core.exceptions import (
    ConflictError,
    ResourceNotFoundError,
    ServiceUnavailableError,
    ValidationError,
)
from backend.app.core.logging import get_logger
from backend.app.core.security import validate_processing_configuration, validate_uuid
from backend.app.database.connection import DatabaseManager
from backend.app.database.models import ProcessingJob
from backend.app.database.repositories import JobRepository, VideoRepository
from backend.app.schemas.processing import ProcessingJobCreateRequest
from backend.app.services.storage_service import StorageService
from backend.app.utils.pagination import build_pagination_meta, normalize_pagination
from backend.app.workers.job_manager import JobManager


logger = get_logger("services.processing")


class ProcessingService:
    def __init__(
        self,
        settings: Settings,
        db_manager: DatabaseManager,
        storage: StorageService,
        job_manager: JobManager,
    ) -> None:
        self.settings = settings
        self.db_manager = db_manager
        self.storage = storage
        self.job_manager = job_manager

    def create_job(self, req: ProcessingJobCreateRequest) -> ProcessingJob:
        video_id = validate_uuid(req.video_id, "video_id")

        if (
            req.sampling_interval_seconds < self.settings.min_sampling_interval_seconds
            or req.sampling_interval_seconds > self.settings.max_sampling_interval_seconds
        ):
            raise ValidationError(
                f"sampling_interval_seconds must be between "
                f"{self.settings.min_sampling_interval_seconds} and "
                f"{self.settings.max_sampling_interval_seconds}.",
                error_code="INVALID_SAMPLING_INTERVAL",
            )

        validated_config = validate_processing_configuration(req.configuration)

        if not self.job_manager.is_running():
            raise ServiceUnavailableError(
                "Background processing worker is unavailable.",
                error_code="WORKER_UNAVAILABLE",
            )

        self.storage.ensure_free_space()

        with self.db_manager.session() as db:
            video = VideoRepository(db).get(video_id)
            if video is None:
                raise ResourceNotFoundError(
                    f"Video '{video_id}' not found.",
                    error_code="VIDEO_NOT_FOUND",
                )

            if req.reference_seconds >= video.duration_seconds:
                raise ValidationError(
                    f"reference_seconds ({req.reference_seconds}) must be less than video duration ({video.duration_seconds:.3f}s).",
                    error_code="INVALID_TIMESTAMP_WINDOW",
                )

            if req.start_seconds >= video.duration_seconds:
                raise ValidationError(
                    f"start_seconds ({req.start_seconds}) must be less than video duration ({video.duration_seconds:.3f}s).",
                    error_code="INVALID_TIMESTAMP_WINDOW",
                )

            if req.end_seconds is not None and req.end_seconds > video.duration_seconds:
                raise ValidationError(
                    f"end_seconds ({req.end_seconds}) exceeds video duration ({video.duration_seconds:.3f}s).",
                    error_code="INVALID_TIMESTAMP_WINDOW",
                )

            job_repo = JobRepository(db)
            duplicate = job_repo.find_active_duplicate(
                video_id=video_id,
                reference_seconds=req.reference_seconds,
                start_seconds=req.start_seconds,
                end_seconds=req.end_seconds,
                sampling_interval_seconds=req.sampling_interval_seconds,
            )
            if duplicate is not None:
                raise ConflictError(
                    "An identical processing job for this video and window is already queued or active.",
                    error_code="DUPLICATE_JOB_SUBMISSION",
                    details={"existing_job_id": duplicate.id},
                )

            job_id = str(uuid.uuid4())
            output_dir = self.storage.create_job_output_dir(job_id)
            rel_output_dir = self.storage.to_relative_path(output_dir)

            job = ProcessingJob(
                id=job_id,
                video_id=video_id,
                status="QUEUED",
                current_stage="QUEUED",
                progress_percentage=0.0,
                reference_seconds=float(req.reference_seconds),
                start_seconds=float(req.start_seconds),
                end_seconds=float(req.end_seconds) if req.end_seconds is not None else None,
                sampling_interval_seconds=float(req.sampling_interval_seconds),
                configuration_snapshot=validated_config,
                output_directory=rel_output_dir,
            )
            job_repo.create(job)
            job_repo.add_log(job_id, "Job created and queued for background processing.", stage="QUEUED")

        # Enqueue strictly AFTER the DB transaction commits so the worker thread
        # always sees the committed ProcessingJob row.
        try:
            self.job_manager.enqueue(job_id)
        except Exception:
            with self.db_manager.session() as cleanup_db:
                orphan = JobRepository(cleanup_db).get(job_id)
                if orphan is not None and orphan.status == "QUEUED":
                    cleanup_db.delete(orphan)
            raise

        logger.info(
            "Created processing job %s for video %s",
            job_id,
            video_id,
            extra={"job_id": job_id, "video_id": video_id, "operation": "job_create"},
        )
        return job

    def list_jobs(
        self,
        *,
        video_id: str | None = None,
        status: str | None = None,
        page: int = 1,
        page_size: int = 20,
        sort_order: str = "desc",
    ) -> dict:
        page, page_size, offset = normalize_pagination(page, page_size)
        canonical_video_id = validate_uuid(video_id, "video_id") if video_id else None
        if sort_order not in {"asc", "desc"}:
            raise ValidationError("sort_order must be 'asc' or 'desc'.")
        with self.db_manager.session() as db:
            items, total = JobRepository(db).list_paginated(
                offset=offset,
                limit=page_size,
                video_id=canonical_video_id,
                status=status,
                sort_order=sort_order,
            )
            return {
                "items": items,
                "pagination": build_pagination_meta(total, page, page_size),
            }

    def get_job(self, job_id: str) -> ProcessingJob:
        canonical_id = validate_uuid(job_id, "job_id")
        with self.db_manager.session() as db:
            job = JobRepository(db).get(canonical_id)
            if job is None:
                raise ResourceNotFoundError(
                    f"Processing job '{canonical_id}' not found.",
                    error_code="JOB_NOT_FOUND",
                )
            return job

    def cancel_job(self, job_id: str) -> dict[str, str]:
        canonical_id = validate_uuid(job_id, "job_id")
        # Verify existence first so 404 is returned rather than 409.
        self.get_job(canonical_id)
        return self.job_manager.request_cancel(canonical_id)

    def get_logs(self, job_id: str, *, limit: int = 200, offset: int = 0) -> dict:
        canonical_id = validate_uuid(job_id, "job_id")
        if limit < 1 or limit > 1000 or offset < 0:
            raise ValidationError("Invalid log pagination bounds.", error_code="INVALID_PAGINATION")
        with self.db_manager.session() as db:
            repo = JobRepository(db)
            if repo.get(canonical_id) is None:
                raise ResourceNotFoundError(
                    f"Processing job '{canonical_id}' not found.",
                    error_code="JOB_NOT_FOUND",
                )
            items, total = repo.get_logs(canonical_id, limit=limit, offset=offset)
            return {"job_id": canonical_id, "items": items, "total_items": total}

    def get_summary(self, job_id: str) -> dict:
        job = self.get_job(job_id)
        return {
            "job_id": job.id,
            "video_id": job.video_id,
            "status": job.status,
            "summary": job.summary_snapshot,
        }
