"""Database-backed analytics API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from backend.app.api.deps import AppContainer, get_container
from backend.app.schemas.analytics import (
    AnalyticsOverviewResponse,
    EventAnalyticsResponse,
    ProcessingAnalyticsResponse,
    TimelineAnalyticsResponse,
    VideoAnalyticsResponse,
)


router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/overview", response_model=AnalyticsOverviewResponse)
def analytics_overview(container: AppContainer = Depends(get_container)):
    return container.analytics_service.overview()


@router.get("/videos", response_model=VideoAnalyticsResponse)
def analytics_videos(container: AppContainer = Depends(get_container)):
    return container.analytics_service.videos()


@router.get("/events", response_model=EventAnalyticsResponse)
def analytics_events(
    start_date: str | None = Query(default=None),
    end_date: str | None = Query(default=None),
    container: AppContainer = Depends(get_container),
):
    return container.analytics_service.events(start_date=start_date, end_date=end_date)


@router.get("/processing", response_model=ProcessingAnalyticsResponse)
def analytics_processing(container: AppContainer = Depends(get_container)):
    return container.analytics_service.processing()


@router.get("/timeline", response_model=TimelineAnalyticsResponse)
def analytics_timeline(
    start_date: str | None = Query(default=None),
    end_date: str | None = Query(default=None),
    container: AppContainer = Depends(get_container),
):
    return container.analytics_service.timeline(start_date=start_date, end_date=end_date)
