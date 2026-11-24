"""Event querying, filtering, and detail retrieval service."""

from __future__ import annotations

from backend.app.core.exceptions import ResourceNotFoundError, ValidationError
from backend.app.core.security import validate_uuid
from backend.app.database.connection import DatabaseManager
from backend.app.database.models import Event
from backend.app.database.repositories import EventRepository, EvidenceRepository, JobRepository, VideoRepository
from backend.app.schemas.evidence import EvidenceResponse
from backend.app.utils.pagination import build_pagination_meta, normalize_pagination


ALLOWED_EVENT_SORTS = {
    "created_at",
    "confidence",
    "duration_seconds",
    "first_seen_seconds",
    "confirmation_seconds",
    "track_id",
}


class EventService:
    def __init__(self, db_manager: DatabaseManager, api_prefix: str = "/api/v1") -> None:
        self.db_manager = db_manager
        self.api_prefix = api_prefix

    def _serialize_evidence(self, ev) -> EvidenceResponse:
        return EvidenceResponse(
            id=ev.id,
            event_id=ev.event_id,
            job_id=ev.job_id,
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

    def list_events(
        self,
        *,
        video_id: str | None = None,
        job_id: str | None = None,
        event_state: str | None = None,
        was_confirmed: bool | None = None,
        review_required: bool | None = None,
        post_confirmation_degraded: bool | None = None,
        min_confidence: float | None = None,
        start_time_seconds: float | None = None,
        end_time_seconds: float | None = None,
        min_duration_seconds: float | None = None,
        sort_by: str = "created_at",
        sort_order: str = "desc",
        page: int = 1,
        page_size: int = 20,
        verify_video_exists: bool = False,
        verify_job_exists: bool = False,
    ) -> dict:
        page, page_size, offset = normalize_pagination(page, page_size)

        if sort_by not in ALLOWED_EVENT_SORTS:
            raise ValidationError(
                f"Unsupported sort_by '{sort_by}'. Allowed: {sorted(ALLOWED_EVENT_SORTS)}",
                error_code="INVALID_SORT_PARAMETER",
            )
        if sort_order not in {"asc", "desc"}:
            raise ValidationError("sort_order must be 'asc' or 'desc'.", error_code="INVALID_SORT_PARAMETER")
        if min_confidence is not None and not (0.0 <= min_confidence <= 1.0):
            raise ValidationError("min_confidence must be in [0.0, 1.0].", error_code="INVALID_FILTER")
        if min_duration_seconds is not None and min_duration_seconds < 0:
            raise ValidationError("min_duration_seconds cannot be negative.", error_code="INVALID_FILTER")
        if (
            start_time_seconds is not None
            and end_time_seconds is not None
            and end_time_seconds < start_time_seconds
        ):
            raise ValidationError("end_time_seconds cannot precede start_time_seconds.", error_code="INVALID_FILTER")

        canonical_video = validate_uuid(video_id, "video_id") if video_id is not None else None
        canonical_job = validate_uuid(job_id, "job_id") if job_id is not None else None

        with self.db_manager.session() as db:
            if verify_video_exists and canonical_video is not None:
                if VideoRepository(db).get(canonical_video, include_deleted=True) is None:
                    raise ResourceNotFoundError(
                        f"Video '{canonical_video}' not found.",
                        error_code="VIDEO_NOT_FOUND",
                    )
            if verify_job_exists and canonical_job is not None:
                if JobRepository(db).get(canonical_job) is None:
                    raise ResourceNotFoundError(
                        f"Processing job '{canonical_job}' not found.",
                        error_code="JOB_NOT_FOUND",
                    )

            items, total = EventRepository(db).list_filtered(
                offset=offset,
                limit=page_size,
                video_id=canonical_video,
                job_id=canonical_job,
                event_state=event_state,
                was_confirmed=was_confirmed,
                review_required=review_required,
                post_confirmation_degraded=post_confirmation_degraded,
                min_confidence=min_confidence,
                start_time_seconds=start_time_seconds,
                end_time_seconds=end_time_seconds,
                min_duration_seconds=min_duration_seconds,
                sort_by=sort_by,
                sort_order=sort_order,
            )
            return {
                "items": items,
                "pagination": build_pagination_meta(total, page, page_size),
            }

    def get_event_detail(self, event_id: str) -> dict:
        canonical_id = validate_uuid(event_id, "event_id")
        with self.db_manager.session() as db:
            event: Event | None = EventRepository(db).get(canonical_id)
            if event is None:
                raise ResourceNotFoundError(
                    f"Event '{canonical_id}' not found.",
                    error_code="EVENT_NOT_FOUND",
                )
            ev_items = EvidenceRepository(db).list_for_event(canonical_id)
            serialized_evidence = [self._serialize_evidence(item) for item in ev_items]
            return {
                "id": event.id,
                "job_id": event.job_id,
                "video_id": event.video_id,
                "track_id": event.track_id,
                "event_state": event.event_state,
                "first_seen_seconds": event.first_seen_seconds,
                "last_seen_seconds": event.last_seen_seconds,
                "duration_seconds": event.duration_seconds,
                "observations": event.observations,
                "missed_observations": event.missed_observations,
                "total_missed_observations": event.total_missed_observations,
                "detection_consistency": event.detection_consistency,
                "center_deviation_pixels": event.center_deviation_pixels,
                "bbox_stability": event.bbox_stability,
                "motion_samples": event.motion_samples,
                "motion_median_ratio": event.motion_median_ratio,
                "motion_p90_ratio": event.motion_p90_ratio,
                "settling_motion_samples": event.settling_motion_samples,
                "settling_motion_median_ratio": event.settling_motion_median_ratio,
                "settling_motion_p90_ratio": event.settling_motion_p90_ratio,
                "confidence": event.confidence,
                "confidence_type": event.confidence_type,
                "was_confirmed": event.was_confirmed,
                "confirmation_seconds": event.confirmation_seconds,
                "review_required": event.review_required,
                "post_confirmation_degraded": event.post_confirmation_degraded,
                "degradation_first_seen_seconds": event.degradation_first_seen_seconds,
                "post_confirmation_degradation_reasons": event.post_confirmation_degradation_reasons,
                "boundary_candidate": event.boundary_candidate,
                "latest_bbox_xywh": event.latest_bbox_xywh,
                "description": event.description,
                "created_at": event.created_at,
                "updated_at": event.updated_at,
                "state_history": event.state_history,
                "evidence_timeline": event.evidence_timeline,
                "evidence_items": serialized_evidence,
            }
