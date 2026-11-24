"""SQLAlchemy ORM models for TRINETRA persistence."""

from __future__ import annotations

from datetime import datetime
from typing import Any
import uuid

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from backend.app.utils.serialization import utc_now


JOB_STATUSES = (
    "QUEUED",
    "VALIDATING",
    "INITIALIZING",
    "PROCESSING",
    "FINALIZING",
    "COMPLETED",
    "FAILED",
    "CANCEL_REQUESTED",
    "CANCELLED",
)

ACTIVE_JOB_STATUSES = (
    "QUEUED",
    "VALIDATING",
    "INITIALIZING",
    "PROCESSING",
    "FINALIZING",
    "CANCEL_REQUESTED",
)

TERMINAL_JOB_STATUSES = ("COMPLETED", "FAILED", "CANCELLED")

LEGAL_JOB_TRANSITIONS: dict[str, set[str]] = {
    "QUEUED": {"VALIDATING", "CANCEL_REQUESTED", "CANCELLED", "FAILED"},
    "VALIDATING": {"INITIALIZING", "CANCEL_REQUESTED", "CANCELLED", "FAILED"},
    "INITIALIZING": {"PROCESSING", "CANCEL_REQUESTED", "CANCELLED", "FAILED"},
    "PROCESSING": {"FINALIZING", "CANCEL_REQUESTED", "CANCELLED", "FAILED"},
    "FINALIZING": {"COMPLETED", "CANCEL_REQUESTED", "CANCELLED", "FAILED"},
    "CANCEL_REQUESTED": {"CANCELLED", "FAILED"},
    "COMPLETED": set(),
    "FAILED": set(),
    "CANCELLED": set(),
}

EVIDENCE_TYPES = (
    "original_frame",
    "annotated_frame",
    "probability_map",
    "metadata",
    "exported_report",
)


class Base(DeclarativeBase):
    pass


class SchemaMigration(Base):
    __tablename__ = "schema_migrations"

    version: Mapped[str] = mapped_column(String(64), primary_key=True)
    description: Mapped[str] = mapped_column(String(255), nullable=False)
    applied_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    stored_filename: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    relative_storage_path: Mapped[str] = mapped_column(String(512), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)

    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    fps: Mapped[float] = mapped_column(Float, nullable=False)
    frame_count: Mapped[int] = mapped_column(Integer, nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    codec: Mapped[str | None] = mapped_column(String(64), nullable=True)
    format: Mapped[str] = mapped_column(String(32), nullable=False)

    upload_status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="UPLOADED"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    jobs: Mapped[list["ProcessingJob"]] = relationship(
        back_populates="video", cascade="save-update, merge"
    )
    events: Mapped[list["Event"]] = relationship(
        back_populates="video", cascade="save-update, merge"
    )

    __table_args__ = (
        Index("ix_videos_created_at", "created_at"),
        Index("ix_videos_upload_status", "upload_status"),
    )


class ProcessingJob(Base):
    __tablename__ = "processing_jobs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    video_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("videos.id", ondelete="RESTRICT"), nullable=False
    )

    status: Mapped[str] = mapped_column(String(32), nullable=False, default="QUEUED")
    current_stage: Mapped[str] = mapped_column(String(64), nullable=False, default="QUEUED")
    progress_percentage: Mapped[float | None] = mapped_column(Float, nullable=True, default=0.0)

    processed_frames: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_frames: Mapped[int | None] = mapped_column(Integer, nullable=True)
    processed_seconds: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    current_video_timestamp_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    processing_fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    elapsed_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    estimated_remaining_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    active_tracks_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    confirmed_events_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    review_required_events_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_candidates_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    reference_seconds: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    start_seconds: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    end_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    sampling_interval_seconds: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)

    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    configuration_snapshot: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )
    output_directory: Mapped[str] = mapped_column(String(512), nullable=False)
    summary_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    evidence_generation_status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="PENDING"
    )
    evidence_generation_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    video: Mapped["Video"] = relationship(back_populates="jobs")
    events: Mapped[list["Event"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )
    evidence_items: Mapped[list["Evidence"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )
    logs: Mapped[list["JobLog"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_processing_jobs_video_id", "video_id"),
        Index("ix_processing_jobs_status", "status"),
        Index("ix_processing_jobs_created_at", "created_at"),
    )


class Event(Base):
    __tablename__ = "events"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    job_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("processing_jobs.id", ondelete="CASCADE"), nullable=False
    )
    video_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("videos.id", ondelete="RESTRICT"), nullable=False
    )
    track_id: Mapped[int] = mapped_column(Integer, nullable=False)
    event_state: Mapped[str] = mapped_column(String(32), nullable=False)

    first_seen_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    last_seen_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False)

    observations: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    missed_observations: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_missed_observations: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    detection_consistency: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    center_deviation_pixels: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    bbox_stability: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    motion_samples: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    motion_median_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    motion_p90_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    settling_motion_samples: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    settling_motion_median_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    settling_motion_p90_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)

    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    confidence_type: Mapped[str] = mapped_column(
        String(64), nullable=False, default="heuristic_score_not_probability"
    )

    was_confirmed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    confirmation_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    review_required: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    post_confirmation_degraded: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    degradation_first_seen_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    post_confirmation_degradation_reasons: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list
    )
    boundary_candidate: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    latest_bbox_xywh: Mapped[list[int] | None] = mapped_column(JSON, nullable=True)
    state_history: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, nullable=False, default=list
    )
    evidence_timeline: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, nullable=False, default=list
    )
    description: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        default="Candidate visual scene change",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    job: Mapped["ProcessingJob"] = relationship(back_populates="events")
    video: Mapped["Video"] = relationship(back_populates="events")
    evidence_items: Mapped[list["Evidence"]] = relationship(
        back_populates="event", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("job_id", "track_id", name="uq_events_job_track"),
        Index("ix_events_job_id", "job_id"),
        Index("ix_events_video_id", "video_id"),
        Index("ix_events_was_confirmed", "was_confirmed"),
        Index("ix_events_review_required", "review_required"),
        Index("ix_events_event_state", "event_state"),
        Index("ix_events_confidence", "confidence"),
        Index("ix_events_created_at", "created_at"),
    )


class Evidence(Base):
    __tablename__ = "evidence"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    event_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("events.id", ondelete="CASCADE"), nullable=True
    )
    job_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("processing_jobs.id", ondelete="CASCADE"), nullable=False
    )
    evidence_type: Mapped[str] = mapped_column(String(64), nullable=False)
    relative_storage_path: Mapped[str] = mapped_column(String(512), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(128), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    timestamp_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    bbox_xywh: Mapped[list[int] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    event: Mapped["Event | None"] = relationship(back_populates="evidence_items")
    job: Mapped["ProcessingJob"] = relationship(back_populates="evidence_items")

    __table_args__ = (
        UniqueConstraint("job_id", "relative_storage_path", name="uq_evidence_job_relpath"),
        Index("ix_evidence_event_id", "event_id"),
        Index("ix_evidence_job_id", "job_id"),
        Index("ix_evidence_type", "evidence_type"),
    )


class JobLog(Base):
    __tablename__ = "job_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("processing_jobs.id", ondelete="CASCADE"), nullable=False
    )
    level: Mapped[str] = mapped_column(String(16), nullable=False, default="INFO")
    stage: Mapped[str | None] = mapped_column(String(64), nullable=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    details: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    job: Mapped["ProcessingJob"] = relationship(back_populates="logs")

    __table_args__ = (
        Index("ix_job_logs_job_id_id", "job_id", "id"),
    )


class SystemSetting(Base):
    __tablename__ = "system_settings"

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value_json: Mapped[Any] = mapped_column(JSON, nullable=True)
    description: Mapped[str | None] = mapped_column(String(255), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )
