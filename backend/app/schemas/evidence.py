"""Pydantic schemas for event evidence items."""

from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, ConfigDict

from backend.app.schemas.common import PaginationMeta


class EvidenceResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    event_id: str | None
    job_id: str
    video_id: str | None = None
    evidence_type: str
    original_filename: str
    mime_type: str
    file_size_bytes: int
    sha256: str
    timestamp_seconds: float | None
    width: int | None
    height: int | None
    bbox_xywh: list[int] | None
    download_url: str | None = None
    created_at: datetime


class EvidenceListResponse(BaseModel):
    event_id: str
    items: list[EvidenceResponse]
    count: int


class EvidencePaginatedResponse(BaseModel):
    items: list[EvidenceResponse]
    pagination: PaginationMeta
