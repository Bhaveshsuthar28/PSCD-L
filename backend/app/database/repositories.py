"""Repository classes for database operations."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import asc, case, desc, func, select
from sqlalchemy.orm import Session

from backend.app.core.exceptions import ConflictError
from backend.app.database.models import (
    ACTIVE_JOB_STATUSES,
    LEGAL_JOB_TRANSITIONS,
    Event,
    Evidence,
    JobLog,
    ProcessingJob,
    SystemSetting,
    Video,
)
from backend.app.utils.serialization import sanitize_for_json, utc_now


class VideoRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, video: Video) -> Video:
        self.db.add(video)
        self.db.flush()
        return video

    def get(self, video_id: str, *, include_deleted: bool = False) -> Video | None:
        video = self.db.get(Video, video_id)
        if video is None:
            return None
        if not include_deleted and video.upload_status == "DELETED":
            return None
        return video

    def get_by_sha256(self, sha256: str, *, include_deleted: bool = False) -> Video | None:
        stmt = select(Video).where(Video.sha256 == sha256)
        if not include_deleted:
            stmt = stmt.where(Video.upload_status != "DELETED")
        return self.db.scalars(stmt).first()

    def count_storage_references(self, relative_storage_path: str, exclude_video_id: str | None = None) -> int:
        stmt = select(func.count(Video.id)).where(
            Video.relative_storage_path == relative_storage_path,
            Video.upload_status != "DELETED",
        )
        if exclude_video_id:
            stmt = stmt.where(Video.id != exclude_video_id)
        return int(self.db.scalar(stmt) or 0)

    def list_paginated(
        self,
        *,
        offset: int,
        limit: int,
        include_deleted: bool = False,
        sort_by: str = "created_at",
        sort_order: str = "desc",
    ) -> tuple[list[Video], int]:
        base = select(Video)
        count_stmt = select(func.count(Video.id))
        if not include_deleted:
            base = base.where(Video.upload_status != "DELETED")
            count_stmt = count_stmt.where(Video.upload_status != "DELETED")

        sort_columns = {
            "created_at": Video.created_at,
            "original_filename": Video.original_filename,
            "file_size_bytes": Video.file_size_bytes,
            "duration_seconds": Video.duration_seconds,
        }
        col = sort_columns.get(sort_by, Video.created_at)
        order_fn = asc if sort_order == "asc" else desc
        stmt = base.order_by(order_fn(col), desc(Video.id)).offset(offset).limit(limit)

        items = list(self.db.scalars(stmt).all())
        total = int(self.db.scalar(count_stmt) or 0)
        return items, total

    def has_active_jobs(self, video_id: str) -> bool:
        stmt = select(func.count(ProcessingJob.id)).where(
            ProcessingJob.video_id == video_id,
            ProcessingJob.status.in_(ACTIVE_JOB_STATUSES),
        )
        return int(self.db.scalar(stmt) or 0) > 0

    def count_jobs(self, video_id: str) -> int:
        stmt = select(func.count(ProcessingJob.id)).where(ProcessingJob.video_id == video_id)
        return int(self.db.scalar(stmt) or 0)

    def delete_record(self, video: Video) -> None:
        self.db.delete(video)
        self.db.flush()


class JobRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, job: ProcessingJob) -> ProcessingJob:
        self.db.add(job)
        self.db.flush()
        return job

    def get(self, job_id: str) -> ProcessingJob | None:
        return self.db.get(ProcessingJob, job_id)

    def find_active_duplicate(
        self,
        *,
        video_id: str,
        reference_seconds: float,
        start_seconds: float,
        end_seconds: float | None,
        sampling_interval_seconds: float,
    ) -> ProcessingJob | None:
        stmt = select(ProcessingJob).where(
            ProcessingJob.video_id == video_id,
            ProcessingJob.status.in_(ACTIVE_JOB_STATUSES),
            ProcessingJob.reference_seconds == reference_seconds,
            ProcessingJob.start_seconds == start_seconds,
            ProcessingJob.sampling_interval_seconds == sampling_interval_seconds,
        )
        if end_seconds is None:
            stmt = stmt.where(ProcessingJob.end_seconds.is_(None))
        else:
            stmt = stmt.where(ProcessingJob.end_seconds == end_seconds)
        return self.db.scalars(stmt).first()

    def count_active(self) -> int:
        stmt = select(func.count(ProcessingJob.id)).where(
            ProcessingJob.status.in_(ACTIVE_JOB_STATUSES)
        )
        return int(self.db.scalar(stmt) or 0)

    def list_paginated(
        self,
        *,
        offset: int,
        limit: int,
        video_id: str | None = None,
        status: str | None = None,
        sort_order: str = "desc",
    ) -> tuple[list[ProcessingJob], int]:
        base = select(ProcessingJob)
        count_stmt = select(func.count(ProcessingJob.id))
        if video_id:
            base = base.where(ProcessingJob.video_id == video_id)
            count_stmt = count_stmt.where(ProcessingJob.video_id == video_id)
        if status:
            base = base.where(ProcessingJob.status == status)
            count_stmt = count_stmt.where(ProcessingJob.status == status)

        order_fn = asc if sort_order == "asc" else desc
        stmt = base.order_by(order_fn(ProcessingJob.created_at), desc(ProcessingJob.id)).offset(offset).limit(limit)
        items = list(self.db.scalars(stmt).all())
        total = int(self.db.scalar(count_stmt) or 0)
        return items, total

    def transition_status(
        self,
        job: ProcessingJob,
        new_status: str,
        *,
        stage: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> ProcessingJob:
        if job.status == new_status:
            if stage is not None:
                job.current_stage = stage
            job.updated_at = utc_now()
            self.db.flush()
            return job

        allowed = LEGAL_JOB_TRANSITIONS.get(job.status, set())
        if new_status not in allowed:
            raise ConflictError(
                f"Illegal job state transition: {job.status} -> {new_status}",
                error_code="ILLEGAL_JOB_STATE_TRANSITION",
            )

        now = utc_now()
        job.status = new_status
        job.current_stage = stage or new_status
        job.updated_at = now
        if new_status in {"VALIDATING", "INITIALIZING", "PROCESSING"} and job.started_at is None:
            job.started_at = now
        if new_status in {"COMPLETED", "FAILED", "CANCELLED"}:
            job.completed_at = now
        if new_status == "COMPLETED":
            job.progress_percentage = 100.0
        if error_code is not None:
            job.error_code = error_code
        if error_message is not None:
            job.error_message = error_message
        self.db.flush()
        return job

    def list_interrupted_active(self) -> list[ProcessingJob]:
        stmt = select(ProcessingJob).where(ProcessingJob.status.in_(ACTIVE_JOB_STATUSES))
        return list(self.db.scalars(stmt).all())

    def add_log(
        self,
        job_id: str,
        message: str,
        *,
        level: str = "INFO",
        stage: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> JobLog:
        entry = JobLog(
            job_id=job_id,
            level=level,
            stage=stage,
            message=message,
            details=sanitize_for_json(details) if details is not None else None,
        )
        self.db.add(entry)
        self.db.flush()
        return entry

    def get_logs(self, job_id: str, *, limit: int = 200, offset: int = 0) -> tuple[list[JobLog], int]:
        base = select(JobLog).where(JobLog.job_id == job_id)
        total = int(
            self.db.scalar(select(func.count(JobLog.id)).where(JobLog.job_id == job_id)) or 0
        )
        items = list(
            self.db.scalars(base.order_by(asc(JobLog.id)).offset(offset).limit(limit)).all()
        )
        return items, total


class EventRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create_many(self, events: list[Event]) -> list[Event]:
        self.db.add_all(events)
        self.db.flush()
        return events

    def get(self, event_id: str) -> Event | None:
        return self.db.get(Event, event_id)

    def get_by_job_and_track(self, job_id: str, track_id: int) -> Event | None:
        stmt = select(Event).where(Event.job_id == job_id, Event.track_id == track_id)
        return self.db.scalars(stmt).first()

    def list_filtered(
        self,
        *,
        offset: int,
        limit: int,
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
    ) -> tuple[list[Event], int]:
        conditions = []
        if video_id is not None:
            conditions.append(Event.video_id == video_id)
        if job_id is not None:
            conditions.append(Event.job_id == job_id)
        if event_state is not None:
            conditions.append(Event.event_state == event_state)
        if was_confirmed is not None:
            conditions.append(Event.was_confirmed.is_(was_confirmed))
        if review_required is not None:
            conditions.append(Event.review_required.is_(review_required))
        if post_confirmation_degraded is not None:
            conditions.append(Event.post_confirmation_degraded.is_(post_confirmation_degraded))
        if min_confidence is not None:
            conditions.append(Event.confidence >= min_confidence)
        if start_time_seconds is not None:
            conditions.append(Event.last_seen_seconds >= start_time_seconds)
        if end_time_seconds is not None:
            conditions.append(Event.first_seen_seconds <= end_time_seconds)
        if min_duration_seconds is not None:
            conditions.append(Event.duration_seconds >= min_duration_seconds)

        sort_columns = {
            "created_at": Event.created_at,
            "confidence": Event.confidence,
            "duration_seconds": Event.duration_seconds,
            "first_seen_seconds": Event.first_seen_seconds,
            "confirmation_seconds": Event.confirmation_seconds,
            "track_id": Event.track_id,
        }
        col = sort_columns.get(sort_by, Event.created_at)
        order_fn = asc if sort_order == "asc" else desc

        stmt = select(Event).where(*conditions).order_by(order_fn(col), asc(Event.track_id), asc(Event.id)).offset(offset).limit(limit)
        count_stmt = select(func.count(Event.id)).where(*conditions)

        items = list(self.db.scalars(stmt).all())
        total = int(self.db.scalar(count_stmt) or 0)
        return items, total

    def list_all_for_job(self, job_id: str) -> list[Event]:
        stmt = select(Event).where(Event.job_id == job_id).order_by(asc(Event.track_id), asc(Event.id))
        return list(self.db.scalars(stmt).all())


class EvidenceRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, item: Evidence) -> Evidence:
        self.db.add(item)
        self.db.flush()
        return item

    def get(self, evidence_id: str) -> Evidence | None:
        return self.db.get(Evidence, evidence_id)

    def get_by_job_and_relpath(self, job_id: str, relpath: str) -> Evidence | None:
        stmt = select(Evidence).where(
            Evidence.job_id == job_id,
            Evidence.relative_storage_path == relpath,
        )
        return self.db.scalars(stmt).first()

    def list_for_event(self, event_id: str) -> list[Evidence]:
        stmt = (
            select(Evidence)
            .where(Evidence.event_id == event_id)
            .order_by(asc(Evidence.created_at), asc(Evidence.evidence_type))
        )
        return list(self.db.scalars(stmt).all())

    def list_for_event_and_type(self, event_id: str, evidence_type: str) -> list[Evidence]:
        stmt = (
            select(Evidence)
            .where(Evidence.event_id == event_id, Evidence.evidence_type == evidence_type)
            .order_by(desc(Evidence.created_at))
        )
        return list(self.db.scalars(stmt).all())

    def list_for_job(self, job_id: str) -> list[Evidence]:
        stmt = (
            select(Evidence)
            .where(Evidence.job_id == job_id)
            .order_by(asc(Evidence.created_at), asc(Evidence.id))
        )
        return list(self.db.scalars(stmt).all())

    def list_paginated(
        self,
        *,
        offset: int,
        limit: int,
        video_id: str | None = None,
        job_id: str | None = None,
        event_id: str | None = None,
        evidence_type: str | None = None,
    ) -> tuple[list[tuple[Evidence, str]], int]:
        conditions = []
        if video_id is not None:
            conditions.append(ProcessingJob.video_id == video_id)
        if job_id is not None:
            conditions.append(Evidence.job_id == job_id)
        if event_id is not None:
            conditions.append(Evidence.event_id == event_id)
        if evidence_type is not None:
            conditions.append(Evidence.evidence_type == evidence_type)

        stmt = (
            select(Evidence, ProcessingJob.video_id)
            .join(ProcessingJob, ProcessingJob.id == Evidence.job_id)
            .where(*conditions)
            .order_by(desc(Evidence.created_at), asc(Evidence.id))
            .offset(offset)
            .limit(limit)
        )
        count_stmt = (
            select(func.count(Evidence.id))
            .join(ProcessingJob, ProcessingJob.id == Evidence.job_id)
            .where(*conditions)
        )
        rows = [(row[0], row[1]) for row in self.db.execute(stmt).all()]
        total = int(self.db.scalar(count_stmt) or 0)
        return rows, total


class SettingsRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def list_all(self) -> list[SystemSetting]:
        stmt = select(SystemSetting).order_by(asc(SystemSetting.key))
        return list(self.db.scalars(stmt).all())

    def get(self, key: str) -> SystemSetting | None:
        return self.db.get(SystemSetting, key)

    def upsert(self, key: str, value: Any, description: str | None = None) -> SystemSetting:
        existing = self.db.get(SystemSetting, key)
        if existing is None:
            item = SystemSetting(
                key=key,
                value_json=sanitize_for_json(value),
                description=description,
            )
            self.db.add(item)
            self.db.flush()
            return item
        existing.value_json = sanitize_for_json(value)
        if description is not None:
            existing.description = description
        existing.updated_at = utc_now()
        self.db.flush()
        return existing


class AnalyticsRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def overview(self) -> dict[str, Any]:
        total_videos = int(
            self.db.scalar(
                select(func.count(Video.id)).where(Video.upload_status != "DELETED")
            )
            or 0
        )
        job_counts_rows = self.db.execute(
            select(ProcessingJob.status, func.count(ProcessingJob.id)).group_by(ProcessingJob.status)
        ).all()
        job_counts = {status: int(cnt) for status, cnt in job_counts_rows}
        total_jobs = sum(job_counts.values())

        total_processing_duration = float(
            self.db.scalar(
                select(func.coalesce(func.sum(ProcessingJob.elapsed_seconds), 0.0)).where(
                    ProcessingJob.elapsed_seconds.is_not(None)
                )
            )
            or 0.0
        )

        total_candidate_events = int(self.db.scalar(select(func.count(Event.id))) or 0)
        historical_confirmations = int(
            self.db.scalar(
                select(func.count(Event.id)).where(Event.was_confirmed.is_(True))
            )
            or 0
        )
        review_required_events = int(
            self.db.scalar(
                select(func.count(Event.id)).where(Event.review_required.is_(True))
            )
            or 0
        )
        non_review_confirmations = int(
            self.db.scalar(
                select(func.count(Event.id)).where(
                    Event.was_confirmed.is_(True),
                    Event.review_required.is_(False),
                )
            )
            or 0
        )
        post_confirmation_degraded = int(
            self.db.scalar(
                select(func.count(Event.id)).where(
                    Event.post_confirmation_degraded.is_(True)
                )
            )
            or 0
        )

        return {
            "total_uploaded_videos": total_videos,
            "total_processing_jobs": total_jobs,
            "completed_jobs": job_counts.get("COMPLETED", 0),
            "failed_jobs": job_counts.get("FAILED", 0),
            "cancelled_jobs": job_counts.get("CANCELLED", 0),
            "queued_jobs": job_counts.get("QUEUED", 0),
            "active_jobs": sum(job_counts.get(s, 0) for s in ACTIVE_JOB_STATUSES),
            "total_candidate_events": total_candidate_events,
            "historical_confirmations": historical_confirmations,
            "non_review_confirmations": non_review_confirmations,
            "review_required_events": review_required_events,
            "post_confirmation_degraded_events": post_confirmation_degraded,
            "total_processing_duration_seconds": round(total_processing_duration, 3),
        }

    def videos_breakdown(self) -> list[dict[str, Any]]:
        rows = self.db.execute(
            select(
                Video.id,
                Video.original_filename,
                Video.duration_seconds,
                Video.upload_status,
                func.count(func.distinct(ProcessingJob.id)).label("job_count"),
                func.count(func.distinct(Event.id)).label("candidate_events"),
                func.sum(
                    case((Event.was_confirmed.is_(True), 1), else_=0)
                ).label("confirmed_events"),
                func.sum(
                    case((Event.review_required.is_(True), 1), else_=0)
                ).label("review_required_events"),
            )
            .outerjoin(ProcessingJob, ProcessingJob.video_id == Video.id)
            .outerjoin(Event, Event.video_id == Video.id)
            .where(Video.upload_status != "DELETED")
            .group_by(Video.id)
            .order_by(desc(Video.created_at))
        ).all()
        return [
            {
                "video_id": row.id,
                "original_filename": row.original_filename,
                "duration_seconds": float(row.duration_seconds),
                "upload_status": row.upload_status,
                "job_count": int(row.job_count or 0),
                "candidate_event_count": int(row.candidate_events or 0),
                "historical_confirmed_event_count": int(row.confirmed_events or 0),
                "review_required_event_count": int(row.review_required_events or 0),
            }
            for row in rows
        ]

    def events_analytics(
        self,
        *,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> dict[str, Any]:
        conditions = []
        if start_date is not None:
            conditions.append(Event.created_at >= start_date)
        if end_date is not None:
            conditions.append(Event.created_at <= end_date)

        by_state_rows = self.db.execute(
            select(Event.event_state, func.count(Event.id))
            .where(*conditions)
            .group_by(Event.event_state)
        ).all()
        by_state = {state: int(cnt) for state, cnt in by_state_rows}

        total_events = sum(by_state.values())
        confirmed_total = int(
            self.db.scalar(
                select(func.count(Event.id)).where(*conditions, Event.was_confirmed.is_(True))
            )
            or 0
        )
        review_required_total = int(
            self.db.scalar(
                select(func.count(Event.id)).where(*conditions, Event.review_required.is_(True))
            )
            or 0
        )
        degraded_total = int(
            self.db.scalar(
                select(func.count(Event.id)).where(
                    *conditions, Event.post_confirmation_degraded.is_(True)
                )
            )
            or 0
        )

        # Confidence distribution buckets: [0,0.25), [0.25,0.5), [0.5,0.75), [0.75,1.0]
        conf_buckets = [
            ("0.00-0.25", 0.0, 0.25, False),
            ("0.25-0.50", 0.25, 0.50, False),
            ("0.50-0.75", 0.50, 0.75, False),
            ("0.75-1.00", 0.75, 1.01, True),
        ]
        confidence_distribution = []
        for label, low, high, _ in conf_buckets:
            cnt = int(
                self.db.scalar(
                    select(func.count(Event.id)).where(
                        *conditions, Event.confidence >= low, Event.confidence < high
                    )
                )
                or 0
            )
            confidence_distribution.append({"bucket": label, "count": cnt})

        # Duration distribution buckets in seconds
        dur_buckets = [
            ("0-5s", 0.0, 5.0),
            ("5-10s", 5.0, 10.0),
            ("10-30s", 10.0, 30.0),
            ("30s+", 30.0, 1e12),
        ]
        duration_distribution = []
        for label, low, high in dur_buckets:
            cnt = int(
                self.db.scalar(
                    select(func.count(Event.id)).where(
                        *conditions, Event.duration_seconds >= low, Event.duration_seconds < high
                    )
                )
                or 0
            )
            duration_distribution.append({"bucket": label, "count": cnt})

        return {
            "total_candidate_events": total_events,
            "historical_confirmations": confirmed_total,
            "non_review_confirmations": max(0, confirmed_total - review_required_total),
            "review_required_events": review_required_total,
            "post_confirmation_degraded_events": degraded_total,
            "by_state": by_state,
            "confidence_distribution": confidence_distribution,
            "duration_distribution": duration_distribution,
        }

    def processing_analytics(self) -> dict[str, Any]:
        status_rows = self.db.execute(
            select(ProcessingJob.status, func.count(ProcessingJob.id)).group_by(ProcessingJob.status)
        ).all()
        by_status = {status: int(cnt) for status, cnt in status_rows}

        agg = self.db.execute(
            select(
                func.avg(ProcessingJob.processing_fps),
                func.avg(ProcessingJob.elapsed_seconds),
                func.sum(ProcessingJob.processed_frames),
            ).where(ProcessingJob.status == "COMPLETED")
        ).one()

        return {
            "jobs_by_status": by_status,
            "completed_jobs_count": by_status.get("COMPLETED", 0),
            "average_processing_fps": round(float(agg[0]), 4) if agg[0] is not None else None,
            "average_elapsed_seconds": round(float(agg[1]), 4) if agg[1] is not None else None,
            "total_processed_frames_completed_jobs": int(agg[2] or 0),
        }

    def timeline(
        self,
        *,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[dict[str, Any]]:
        conditions = []
        if start_date is not None:
            conditions.append(Event.created_at >= start_date)
        if end_date is not None:
            conditions.append(Event.created_at <= end_date)

        day_expr = func.strftime("%Y-%m-%d", Event.created_at)
        rows = self.db.execute(
            select(
                day_expr.label("day"),
                func.count(Event.id).label("total"),
                func.sum(case((Event.was_confirmed.is_(True), 1), else_=0)).label("confirmed"),
                func.sum(case((Event.review_required.is_(True), 1), else_=0)).label("review_required"),
            )
            .where(*conditions)
            .group_by(day_expr)
            .order_by(asc(day_expr))
        ).all()

        return [
            {
                "date": row.day,
                "candidate_events": int(row.total or 0),
                "historical_confirmations": int(row.confirmed or 0),
                "review_required_events": int(row.review_required or 0),
            }
            for row in rows
            if row.day is not None
        ]
