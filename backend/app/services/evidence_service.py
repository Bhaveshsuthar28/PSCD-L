"""Secure evidence lookup and file-serving service."""

from __future__ import annotations

from pathlib import Path

from backend.app.core.exceptions import ResourceNotFoundError, ValidationError
from backend.app.core.security import validate_uuid
from backend.app.database.connection import DatabaseManager
from backend.app.database.models import EVIDENCE_TYPES, Evidence
from backend.app.database.repositories import EventRepository, EvidenceRepository
from backend.app.schemas.evidence import EvidenceResponse
from backend.app.services.storage_service import StorageService
from backend.app.utils.pagination import build_pagination_meta, normalize_pagination


class EvidenceService:
    def __init__(
        self,
        db_manager: DatabaseManager,
        storage: StorageService,
        api_prefix: str = "/api/v1",
    ) -> None:
        self.db_manager = db_manager
        self.storage = storage
        self.api_prefix = api_prefix

    def to_response(self, ev: Evidence, *, video_id: str | None = None) -> EvidenceResponse:
        return EvidenceResponse(
            id=ev.id,
            event_id=ev.event_id,
            job_id=ev.job_id,
            video_id=video_id,
            evidence_type=ev.evidence_type,
            original_filename=ev.original_filename,
            mime_type=ev.mime_type,
            file_size_bytes=ev.file_size_bytes,
            sha256=ev.sha256,
            timestamp_seconds=ev.timestamp_seconds,
            width=ev.width,
            height=ev.height,
            bbox_xywh=ev.bbox_xywh,
            download_url=f"{self.api_prefix}/evidence/{ev.id}/download",
            created_at=ev.created_at,
        )

    def list_paginated(
        self,
        *,
        video_id: str | None = None,
        job_id: str | None = None,
        event_id: str | None = None,
        evidence_type: str | None = None,
        page: int = 1,
        page_size: int = 24,
    ) -> dict:
        page, page_size, offset = normalize_pagination(page, page_size, max_page_size=100)
        canonical_video = validate_uuid(video_id, "video_id") if video_id else None
        canonical_job = validate_uuid(job_id, "job_id") if job_id else None
        canonical_event = validate_uuid(event_id, "event_id") if event_id else None
        if evidence_type is not None and evidence_type not in EVIDENCE_TYPES:
            raise ValidationError(
                f"Unsupported evidence_type '{evidence_type}'. Allowed: {list(EVIDENCE_TYPES)}",
                error_code="INVALID_EVIDENCE_TYPE",
            )
        with self.db_manager.session() as db:
            rows, total = EvidenceRepository(db).list_paginated(
                offset=offset,
                limit=page_size,
                video_id=canonical_video,
                job_id=canonical_job,
                event_id=canonical_event,
                evidence_type=evidence_type,
            )
            items = [self.to_response(ev, video_id=vid) for ev, vid in rows]
            return {
                "items": items,
                "pagination": build_pagination_meta(total, page, page_size),
            }

    def list_for_event(self, event_id: str) -> dict:
        canonical_event_id = validate_uuid(event_id, "event_id")
        with self.db_manager.session() as db:
            if EventRepository(db).get(canonical_event_id) is None:
                raise ResourceNotFoundError(
                    f"Event '{canonical_event_id}' not found.",
                    error_code="EVENT_NOT_FOUND",
                )
            items = EvidenceRepository(db).list_for_event(canonical_event_id)
            serialized = [self.to_response(item) for item in items]
            return {
                "event_id": canonical_event_id,
                "items": serialized,
                "count": len(serialized),
            }

    def get_evidence(self, evidence_id: str) -> EvidenceResponse:
        canonical_id = validate_uuid(evidence_id, "evidence_id")
        with self.db_manager.session() as db:
            item = EvidenceRepository(db).get(canonical_id)
            if item is None:
                raise ResourceNotFoundError(
                    f"Evidence '{canonical_id}' not found.",
                    error_code="EVIDENCE_NOT_FOUND",
                )
            return self.to_response(item)

    def resolve_evidence_file(self, evidence_id: str) -> tuple[Evidence, Path]:
        canonical_id = validate_uuid(evidence_id, "evidence_id")
        with self.db_manager.session() as db:
            item = EvidenceRepository(db).get(canonical_id)
            if item is None:
                raise ResourceNotFoundError(
                    f"Evidence '{canonical_id}' not found.",
                    error_code="EVIDENCE_NOT_FOUND",
                )

            resolved = self.storage.resolve_relative_path(item.relative_storage_path)
            if not resolved.is_file():
                raise ResourceNotFoundError(
                    "Evidence file does not exist on storage.",
                    error_code="EVIDENCE_FILE_MISSING",
                )
            return item, resolved

    def resolve_event_evidence_by_type(
        self,
        event_id: str,
        evidence_type: str,
    ) -> tuple[Evidence, Path]:
        canonical_event_id = validate_uuid(event_id, "event_id")
        if evidence_type not in EVIDENCE_TYPES:
            raise ValidationError(
                f"Unsupported evidence_type '{evidence_type}'. Allowed: {list(EVIDENCE_TYPES)}",
                error_code="INVALID_EVIDENCE_TYPE",
            )
        with self.db_manager.session() as db:
            if EventRepository(db).get(canonical_event_id) is None:
                raise ResourceNotFoundError(
                    f"Event '{canonical_event_id}' not found.",
                    error_code="EVENT_NOT_FOUND",
                )
            items = EvidenceRepository(db).list_for_event_and_type(
                canonical_event_id, evidence_type
            )
            if not items:
                raise ResourceNotFoundError(
                    f"No evidence of type '{evidence_type}' found for event '{canonical_event_id}'.",
                    error_code="EVIDENCE_NOT_FOUND",
                )
            item = items[0]
            resolved = self.storage.resolve_relative_path(item.relative_storage_path)
            if not resolved.is_file():
                raise ResourceNotFoundError(
                    "Evidence file does not exist on storage.",
                    error_code="EVIDENCE_FILE_MISSING",
                )
            return item, resolved
