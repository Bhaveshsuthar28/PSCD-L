"""Video ingestion, validation, metadata retrieval, and safe deletion service."""

from __future__ import annotations

from pathlib import Path
import uuid

from fastapi import UploadFile

from backend.app.core.config import Settings
from backend.app.core.exceptions import (
    ConflictError,
    DuplicateVideoError,
    ResourceNotFoundError,
)
from backend.app.core.logging import get_logger
from backend.app.core.security import sanitize_filename, validate_uuid
from backend.app.database.connection import DatabaseManager
from backend.app.database.models import Video
from backend.app.database.repositories import VideoRepository
from backend.app.services.storage_service import StorageService
from backend.app.utils.file_validation import (
    stream_upload_to_temp,
    validate_container_signature,
    validate_video_extension,
)
from backend.app.utils.pagination import build_pagination_meta, normalize_pagination
from backend.app.utils.video_metadata import extract_and_validate_video_metadata


logger = get_logger("services.video")


class VideoService:
    """Coordinates untrusted video uploads, validation, persistence, and safe deletion."""

    def __init__(
        self,
        settings: Settings,
        db_manager: DatabaseManager,
        storage: StorageService,
    ) -> None:
        self.settings = settings
        self.db_manager = db_manager
        self.storage = storage

    async def ingest_upload(self, file: UploadFile) -> Video:
        original_filename = sanitize_filename(file.filename)
        extension = validate_video_extension(
            original_filename,
            self.settings.allowed_video_extensions,
        )

        self.storage.ensure_free_space()

        video_id = str(uuid.uuid4())
        temp_path = self.storage.temp_dir / f"{video_id}{extension}.part"
        stored_filename = f"{video_id}{extension}"
        final_path = self.storage.resolve_upload_target(stored_filename)
        moved_to_final = False

        try:
            file_size, sha256 = await stream_upload_to_temp(
                file,
                temp_path,
                max_size_bytes=self.settings.max_upload_size_bytes,
                chunk_size_bytes=self.settings.upload_chunk_size_bytes,
            )

            validate_container_signature(temp_path, extension)

            metadata = extract_and_validate_video_metadata(
                temp_path,
                extension=extension,
                max_duration_seconds=self.settings.max_video_duration_seconds,
                max_width=self.settings.max_video_width,
                max_height=self.settings.max_video_height,
            )

            with self.db_manager.session() as db:
                repo = VideoRepository(db)
                existing = repo.get_by_sha256(sha256)
                if existing is not None:
                    raise DuplicateVideoError(
                        "An identical video has already been uploaded.",
                        details={
                            "existing_video_id": existing.id,
                            "sha256": sha256,
                        },
                    )

                self.storage.atomic_move(temp_path, final_path)
                moved_to_final = True
                rel_path = self.storage.to_relative_path(final_path)

                video = Video(
                    id=video_id,
                    original_filename=original_filename,
                    stored_filename=stored_filename,
                    relative_storage_path=rel_path,
                    file_size_bytes=file_size,
                    sha256=sha256,
                    duration_seconds=metadata.duration_seconds,
                    fps=metadata.fps,
                    frame_count=metadata.frame_count,
                    width=metadata.width,
                    height=metadata.height,
                    codec=metadata.codec,
                    format=metadata.format,
                    upload_status="UPLOADED",
                )
                repo.create(video)
                logger.info(
                    "Video ingested successfully",
                    extra={
                        "video_id": video_id,
                        "operation": "video_upload",
                    },
                )
                return video

        except Exception as exc:
            if temp_path.exists():
                self.storage.safe_unlink(temp_path)
            if moved_to_final and final_path.exists():
                self.storage.safe_unlink(final_path)
            logger.warning(
                "Video upload rejected: %s",
                exc,
                extra={
                    "video_id": video_id,
                    "operation": "video_upload",
                    "error_code": getattr(exc, "error_code", "UPLOAD_FAILED"),
                },
            )
            raise
        finally:
            await file.close()

    def list_videos(
        self,
        *,
        page: int = 1,
        page_size: int = 20,
        sort_by: str = "created_at",
        sort_order: str = "desc",
    ) -> dict:
        page, page_size, offset = normalize_pagination(page, page_size)
        with self.db_manager.session() as db:
            repo = VideoRepository(db)
            items, total = repo.list_paginated(
                offset=offset,
                limit=page_size,
                sort_by=sort_by,
                sort_order=sort_order,
            )
            return {
                "items": items,
                "pagination": build_pagination_meta(total, page, page_size),
            }

    def get_video(self, video_id: str) -> Video:
        canonical_id = validate_uuid(video_id, "video_id")
        with self.db_manager.session() as db:
            video = VideoRepository(db).get(canonical_id)
            if video is None:
                raise ResourceNotFoundError(
                    f"Video '{canonical_id}' not found.",
                    error_code="VIDEO_NOT_FOUND",
                )
            return video

    def resolve_video_file(self, video: Video) -> Path:
        path = self.storage.resolve_relative_path(video.relative_storage_path)
        if not path.is_file():
            raise ResourceNotFoundError(
                "Video media file is missing from storage.",
                error_code="VIDEO_FILE_MISSING",
            )
        return path

    def delete_video(self, video_id: str) -> dict:
        canonical_id = validate_uuid(video_id, "video_id")
        with self.db_manager.session() as db:
            repo = VideoRepository(db)
            video = repo.get(canonical_id)
            if video is None:
                raise ResourceNotFoundError(
                    f"Video '{canonical_id}' not found.",
                    error_code="VIDEO_NOT_FOUND",
                )

            if repo.has_active_jobs(canonical_id):
                raise ConflictError(
                    "Cannot delete video while processing jobs are queued or active.",
                    error_code="VIDEO_IN_USE",
                )

            other_refs = repo.count_storage_references(
                video.relative_storage_path, exclude_video_id=canonical_id
            )
            file_removed = False
            if other_refs == 0:
                target = self.storage.resolve_relative_path(video.relative_storage_path)
                if target.exists():
                    file_removed = self.storage.safe_unlink(target)
                    if not file_removed:
                        raise ConflictError(
                            "Failed to delete video file from storage.",
                            error_code="FILE_DELETE_FAILED",
                        )

            job_count = repo.count_jobs(canonical_id)
            if job_count > 0:
                # Preserve historical job/event/evidence integrity per retention policy.
                video.upload_status = "DELETED"
                db.flush()
                logger.info(
                    "Video soft-deleted while preserving historical job/event records",
                    extra={"video_id": canonical_id, "operation": "video_delete"},
                )
                return {
                    "success": True,
                    "video_id": canonical_id,
                    "file_removed": file_removed,
                    "record_deleted": False,
                    "historical_records_preserved": True,
                    "message": "Video file removed; historical job and event records preserved.",
                }

            repo.delete_record(video)
            logger.info(
                "Video record and file deleted",
                extra={"video_id": canonical_id, "operation": "video_delete"},
            )
            return {
                "success": True,
                "video_id": canonical_id,
                "file_removed": file_removed,
                "record_deleted": True,
                "historical_records_preserved": False,
                "message": "Video deleted successfully.",
            }
