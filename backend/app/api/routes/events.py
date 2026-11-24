"""Event listing, filtering, detail, and video-scoped event routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from backend.app.api.deps import AppContainer, get_container
from backend.app.schemas.event import EventDetailResponse, EventListResponse


router = APIRouter(tags=["events"])


@router.get("/events", response_model=EventListResponse)
def list_events(
    video_id: str | None = Query(default=None),
    job_id: str | None = Query(default=None),
    event_state: str | None = Query(default=None),
    was_confirmed: bool | None = Query(default=None),
    review_required: bool | None = Query(default=None),
    post_confirmation_degraded: bool | None = Query(default=None),
    min_confidence: float | None = Query(default=None, ge=0.0, le=1.0),
    start_time_seconds: float | None = Query(default=None, ge=0.0),
    end_time_seconds: float | None = Query(default=None, ge=0.0),
    min_duration: float | None = Query(default=None, ge=0.0),
    sort_by: str = Query(default="created_at"),
    sort_order: str = Query(default="desc"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    container: AppContainer = Depends(get_container),
):
    return container.event_service.list_events(
        video_id=video_id,
        job_id=job_id,
        event_state=event_state,
        was_confirmed=was_confirmed,
        review_required=review_required,
        post_confirmation_degraded=post_confirmation_degraded,
        min_confidence=min_confidence,
        start_time_seconds=start_time_seconds,
        end_time_seconds=end_time_seconds,
        min_duration_seconds=min_duration,
        sort_by=sort_by,
        sort_order=sort_order,
        page=page,
        page_size=page_size,
    )


@router.get("/events/{event_id}", response_model=EventDetailResponse)
def get_event_detail(
    event_id: str,
    container: AppContainer = Depends(get_container),
):
    return container.event_service.get_event_detail(event_id)


@router.get("/videos/{video_id}/events", response_model=EventListResponse)
def list_video_events(
    video_id: str,
    job_id: str | None = Query(default=None),
    event_state: str | None = Query(default=None),
    was_confirmed: bool | None = Query(default=None),
    review_required: bool | None = Query(default=None),
    post_confirmation_degraded: bool | None = Query(default=None),
    min_confidence: float | None = Query(default=None, ge=0.0, le=1.0),
    min_duration: float | None = Query(default=None, ge=0.0),
    sort_by: str = Query(default="created_at"),
    sort_order: str = Query(default="desc"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    container: AppContainer = Depends(get_container),
):
    return container.event_service.list_events(
        video_id=video_id,
        job_id=job_id,
        event_state=event_state,
        was_confirmed=was_confirmed,
        review_required=review_required,
        post_confirmation_degraded=post_confirmation_degraded,
        min_confidence=min_confidence,
        min_duration_seconds=min_duration,
        sort_by=sort_by,
        sort_order=sort_order,
        page=page,
        page_size=page_size,
        verify_video_exists=True,
    )
