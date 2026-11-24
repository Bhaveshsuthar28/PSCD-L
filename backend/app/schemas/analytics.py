"""Pydantic schemas for database-backed analytics endpoints."""

from __future__ import annotations

from pydantic import BaseModel


class AnalyticsOverviewResponse(BaseModel):
    total_uploaded_videos: int
    total_processing_jobs: int
    completed_jobs: int
    failed_jobs: int
    cancelled_jobs: int
    queued_jobs: int
    active_jobs: int
    total_candidate_events: int
    historical_confirmations: int
    non_review_confirmations: int
    review_required_events: int
    post_confirmation_degraded_events: int
    total_processing_duration_seconds: float
    note: str = (
        "Candidate events, historical confirmations, non-review confirmations, and "
        "review-required events are reported separately and never conflated."
    )


class VideoAnalyticsItem(BaseModel):
    video_id: str
    original_filename: str
    duration_seconds: float
    upload_status: str
    job_count: int
    candidate_event_count: int
    historical_confirmed_event_count: int
    review_required_event_count: int


class VideoAnalyticsResponse(BaseModel):
    items: list[VideoAnalyticsItem]
    count: int


class DistributionBucket(BaseModel):
    bucket: str
    count: int


class EventAnalyticsResponse(BaseModel):
    total_candidate_events: int
    historical_confirmations: int
    non_review_confirmations: int
    review_required_events: int
    post_confirmation_degraded_events: int
    by_state: dict[str, int]
    confidence_distribution: list[DistributionBucket]
    duration_distribution: list[DistributionBucket]


class ProcessingAnalyticsResponse(BaseModel):
    jobs_by_status: dict[str, int]
    completed_jobs_count: int
    average_processing_fps: float | None
    average_elapsed_seconds: float | None
    total_processed_frames_completed_jobs: int


class TimelinePoint(BaseModel):
    date: str
    candidate_events: int
    historical_confirmations: int
    review_required_events: int


class TimelineAnalyticsResponse(BaseModel):
    items: list[TimelinePoint]
    count: int
