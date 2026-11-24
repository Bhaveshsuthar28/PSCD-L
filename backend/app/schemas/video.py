"""Pydantic schemas for video ingestion and metadata endpoints."""

from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, ConfigDict

from backend.app.schemas.common import PaginationMeta


class VideoMetadataDetail(BaseModel):
    duration_seconds: float
    fps: float
    frame_count: int
    width: int
    height: int
    codec: str | None
    format: str
    file_size_bytes: int
    sha256: str


class VideoResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    original_filename: str
    stored_filename: str
    file_size_bytes: int
    sha256: str
    duration_seconds: float
    fps: float
    frame_count: int
    width: int
    height: int
    codec: str | None
    format: str
    upload_status: str
    created_at: datetime
    updated_at: datetime


class VideoListResponse(BaseModel):
    items: list[VideoResponse]
    pagination: PaginationMeta


class VideoDeleteResponse(BaseModel):
    success: bool = True
    video_id: str
    file_removed: bool
    record_deleted: bool
    historical_records_preserved: bool
    message: str
