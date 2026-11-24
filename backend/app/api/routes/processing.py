"""Processing job routes, SSE progress stream, job events, and exports."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, Header, Query, Request, status
from fastapi.responses import FileResponse, StreamingResponse

from backend.app.api.deps import AppContainer, get_container
from backend.app.core.security import validate_uuid
from backend.app.database.models import TERMINAL_JOB_STATUSES
from backend.app.schemas.event import EventListResponse
from backend.app.schemas.processing import (
    JobCancelResponse,
    ProcessingJobCreateRequest,
    ProcessingJobListResponse,
    ProcessingJobLogsResponse,
    ProcessingJobResponse,
    ProcessingJobSummaryResponse,
)
from backend.app.utils.serialization import sanitize_for_json, utc_now


router = APIRouter(prefix="/processing", tags=["processing"])


@router.post(
    "/jobs",
    response_model=ProcessingJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_processing_job(
    payload: ProcessingJobCreateRequest,
    container: AppContainer = Depends(get_container),
):
    return container.processing_service.create_job(payload)


@router.get("/jobs", response_model=ProcessingJobListResponse)
def list_processing_jobs(
    video_id: str | None = Query(default=None),
    job_status: str | None = Query(default=None, alias="status"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    sort_order: str = Query(default="desc"),
    container: AppContainer = Depends(get_container),
):
    return container.processing_service.list_jobs(
        video_id=video_id,
        status=job_status,
        page=page,
        page_size=page_size,
        sort_order=sort_order,
    )


@router.get("/jobs/{job_id}", response_model=ProcessingJobResponse)
def get_processing_job(
    job_id: str,
    container: AppContainer = Depends(get_container),
):
    return container.processing_service.get_job(job_id)


@router.post("/jobs/{job_id}/cancel", response_model=JobCancelResponse)
def cancel_processing_job(
    job_id: str,
    container: AppContainer = Depends(get_container),
):
    return container.processing_service.cancel_job(job_id)


@router.get("/jobs/{job_id}/logs", response_model=ProcessingJobLogsResponse)
def get_processing_job_logs(
    job_id: str,
    limit: int = Query(default=200, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    container: AppContainer = Depends(get_container),
):
    return container.processing_service.get_logs(job_id, limit=limit, offset=offset)


@router.get("/jobs/{job_id}/summary", response_model=ProcessingJobSummaryResponse)
def get_processing_job_summary(
    job_id: str,
    container: AppContainer = Depends(get_container),
):
    return container.processing_service.get_summary(job_id)


@router.get("/jobs/{job_id}/events", response_model=EventListResponse)
def get_processing_job_events(
    job_id: str,
    event_state: str | None = Query(default=None),
    was_confirmed: bool | None = Query(default=None),
    review_required: bool | None = Query(default=None),
    post_confirmation_degraded: bool | None = Query(default=None),
    min_confidence: float | None = Query(default=None, ge=0.0, le=1.0),
    min_duration: float | None = Query(default=None, ge=0.0),
    sort_by: str = Query(default="track_id"),
    sort_order: str = Query(default="asc"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    container: AppContainer = Depends(get_container),
):
    return container.event_service.list_events(
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
        verify_job_exists=True,
    )


@router.get("/jobs/{job_id}/export")
def export_processing_job(
    job_id: str,
    format: str = Query(default="json"),
    overwrite: bool = Query(default=False),
    container: AppContainer = Depends(get_container),
):
    file_path, filename, media_type = container.export_service.generate_export(
        job_id=job_id,
        fmt=format,
        overwrite=overwrite,
    )
    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=filename,
    )


@router.get("/jobs/{job_id}/stream")
async def stream_processing_job(
    job_id: str,
    request: Request,
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    last_event_id_query: int | None = Query(default=None, alias="last_event_id"),
    container: AppContainer = Depends(get_container),
):
    canonical_id = validate_uuid(job_id, "job_id")
    job = container.processing_service.get_job(canonical_id)

    parsed_last_id = last_event_id_query
    if parsed_last_id is None and last_event_id is not None:
        try:
            parsed_last_id = int(last_event_id.strip())
        except ValueError:
            parsed_last_id = None

    loop = asyncio.get_running_loop()
    sub, replay = container.progress_manager.subscribe(
        canonical_id,
        loop,
        last_event_id=parsed_last_id,
    )
    heartbeat_interval = container.settings.sse_heartbeat_seconds

    async def event_generator():
        try:
            snapshot_payload = sanitize_for_json(
                {
                    "job_id": job.id,
                    "status": job.status,
                    "stage": job.current_stage,
                    "progress_percentage": job.progress_percentage,
                    "processed_frames": job.processed_frames,
                    "total_frames": job.total_frames,
                    "processing_fps": job.processing_fps,
                    "elapsed_seconds": job.elapsed_seconds,
                    "estimated_remaining_seconds": job.estimated_remaining_seconds,
                    "active_tracks_count": job.active_tracks_count,
                    "confirmed_events_count": job.confirmed_events_count,
                    "review_required_events_count": job.review_required_events_count,
                    "error_code": job.error_code,
                    "error_message": job.error_message,
                    "timestamp": utc_now().isoformat(),
                }
            )
            yield f"id: 0\nevent: snapshot\ndata: {json.dumps(snapshot_payload, allow_nan=False)}\n\n"

            for item in replay:
                yield item.format_sse()
                if item.event_type in {"completed", "failed", "cancelled"}:
                    return

            if job.status in TERMINAL_JOB_STATUSES:
                terminal_event = job.status.lower()
                yield (
                    f"id: 0\nevent: {terminal_event}\n"
                    f"data: {json.dumps(snapshot_payload, allow_nan=False)}\n\n"
                )
                return

            while True:
                if await request.is_disconnected():
                    break
                try:
                    item = await asyncio.wait_for(sub.queue.get(), timeout=heartbeat_interval)
                except asyncio.TimeoutError:
                    hb = json.dumps({"job_id": canonical_id, "timestamp": utc_now().isoformat()})
                    yield f"event: heartbeat\ndata: {hb}\n\n"
                    continue

                if item is None:
                    break
                yield item.format_sse()
                if item.event_type in {"completed", "failed", "cancelled"}:
                    break
        finally:
            container.progress_manager.unsubscribe(canonical_id, sub)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
