"""Pydantic schemas for persistent scene-change events."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from backend.app.schemas.common import PaginationMeta
from backend.app.schemas.evidence import EvidenceResponse


class EventResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    job_id: str
    video_id: str
    track_id: int
    event_state: str
    first_seen_seconds: float
    last_seen_seconds: float
    duration_seconds: float
    observations: int
    missed_observations: int
    total_missed_observations: int
    detection_consistency: float
    center_deviation_pixels: float
    bbox_stability: float
    motion_samples: int
    motion_median_ratio: float | None
    motion_p90_ratio: float | None
    settling_motion_samples: int
    settling_motion_median_ratio: float | None
    settling_motion_p90_ratio: float | None
    confidence: float
    confidence_type: str
    was_confirmed: bool
    confirmation_seconds: float | None
    review_required: bool
    post_confirmation_degraded: bool
    degradation_first_seen_seconds: float | None
    post_confirmation_degradation_reasons: list[str]
    boundary_candidate: bool
    latest_bbox_xywh: list[int] | None
    description: str
    created_at: datetime
    updated_at: datetime


class EventDetailResponse(EventResponse):
    state_history: list[dict[str, Any]]
    evidence_timeline: list[dict[str, Any]]
    evidence_items: list[EvidenceResponse] = []


class EventListResponse(BaseModel):
    items: list[EventResponse]
    pagination: PaginationMeta
