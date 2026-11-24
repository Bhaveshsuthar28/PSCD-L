"""Pydantic schemas for processing jobs, progress, logs, and summaries."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.schemas.common import PaginationMeta


class ProcessingJobCreateRequest(BaseModel):
    video_id: str = Field(..., min_length=1, max_length=64)
    reference_seconds: float = Field(default=0.0, ge=0.0)
    start_seconds: float = Field(default=0.0, ge=0.0)
    end_seconds: float | None = Field(default=None, gt=0.0)
    sampling_interval_seconds: float = Field(default=1.0, gt=0.0)
    configuration: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_window(self):
        if self.end_seconds is not None and self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be strictly greater than start_seconds.")
        return self


class ProcessingJobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    video_id: str
    status: str
    current_stage: str
    progress_percentage: float | None
    processed_frames: int
    total_frames: int | None
    processed_seconds: float
    current_video_timestamp_seconds: float | None
    processing_fps: float | None
    elapsed_seconds: float | None
    estimated_remaining_seconds: float | None
    active_tracks_count: int
    confirmed_events_count: int
    review_required_events_count: int
    total_candidates_count: int
    reference_seconds: float
    start_seconds: float
    end_seconds: float | None
    sampling_interval_seconds: float
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime
    updated_at: datetime
    error_code: str | None
    error_message: str | None
    configuration_snapshot: dict[str, Any]
    evidence_generation_status: str
    evidence_generation_error: str | None


class ProcessingJobListResponse(BaseModel):
    items: list[ProcessingJobResponse]
    pagination: PaginationMeta


class JobCancelResponse(BaseModel):
    job_id: str
    status: str
    current_stage: str
    message: str


class JobLogEntryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    job_id: str
    level: str
    stage: str | None
    message: str
    details: dict[str, Any] | None
    created_at: datetime


class ProcessingJobLogsResponse(BaseModel):
    job_id: str
    items: list[JobLogEntryResponse]
    total_items: int


class ProcessingJobSummaryResponse(BaseModel):
    job_id: str
    video_id: str
    status: str
    scientific_note: str = (
        "TRINETRA detects persistent visual scene changes. It does not independently "
        "establish object identity, ownership, abandonment, or intent. Heuristic "
        "confidence is not a calibrated probability."
    )
    summary: dict[str, Any] | None
