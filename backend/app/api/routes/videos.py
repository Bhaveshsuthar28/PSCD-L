"""Video ingestion, listing, metadata, and safe deletion routes."""

from __future__ import annotations

import asyncio
import math
import re
import time

import cv2
from fastapi import APIRouter, Depends, File, Header, Query, Request, UploadFile, status
from fastapi.responses import Response, StreamingResponse

from backend.app.api.deps import AppContainer, get_container
from backend.app.core.exceptions import ValidationError
from backend.app.schemas.video import (
    VideoDeleteResponse,
    VideoListResponse,
    VideoMetadataDetail,
    VideoResponse,
)


VIDEO_MIME_TYPES = {
    "mp4": "video/mp4",
    "mov": "video/quicktime",
    "mkv": "video/x-matroska",
    "avi": "video/x-msvideo",
}


router = APIRouter(prefix="/videos", tags=["videos"])


@router.post(
    "/upload",
    response_model=VideoResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_video(
    file: UploadFile = File(...),
    container: AppContainer = Depends(get_container),
):
    return await container.video_service.ingest_upload(file)


@router.get("", response_model=VideoListResponse)
def list_videos(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    sort_by: str = Query(default="created_at"),
    sort_order: str = Query(default="desc"),
    container: AppContainer = Depends(get_container),
):
    return container.video_service.list_videos(
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        sort_order=sort_order,
    )


@router.get("/{video_id}", response_model=VideoResponse)
def get_video(
    video_id: str,
    container: AppContainer = Depends(get_container),
):
    return container.video_service.get_video(video_id)


@router.get("/{video_id}/metadata", response_model=VideoMetadataDetail)
def get_video_metadata(
    video_id: str,
    container: AppContainer = Depends(get_container),
):
    video = container.video_service.get_video(video_id)
    return VideoMetadataDetail(
        duration_seconds=video.duration_seconds,
        fps=video.fps,
        frame_count=video.frame_count,
        width=video.width,
        height=video.height,
        codec=video.codec,
        format=video.format,
        file_size_bytes=video.file_size_bytes,
        sha256=video.sha256,
    )


@router.delete("/{video_id}", response_model=VideoDeleteResponse)
def delete_video(
    video_id: str,
    container: AppContainer = Depends(get_container),
):
    return container.video_service.delete_video(video_id)


@router.get("/{video_id}/stream")
def stream_video(
    video_id: str,
    range_header: str | None = Header(default=None, alias="Range"),
    container: AppContainer = Depends(get_container),
):
    video = container.video_service.get_video(video_id)
    file_path = container.video_service.resolve_video_file(video)
    file_size = file_path.stat().st_size
    media_type = VIDEO_MIME_TYPES.get(video.format.lower(), "application/octet-stream")

    start = 0
    end = file_size - 1
    status_code = 200

    if range_header:
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
        if not match:
            raise ValidationError("Invalid Range header format.", error_code="INVALID_RANGE_HEADER")
        raw_start, raw_end = match.groups()
        if raw_start == "" and raw_end == "":
            raise ValidationError("Empty Range header.", error_code="INVALID_RANGE_HEADER")
        if raw_start != "":
            start = int(raw_start)
            end = int(raw_end) if raw_end != "" else file_size - 1
        else:
            suffix = int(raw_end)
            start = max(0, file_size - suffix)
            end = file_size - 1

        if start >= file_size or end < start:
            raise ValidationError(
                "Requested byte range is outside video file size.",
                error_code="RANGE_NOT_SATISFIABLE",
                status_code=416,
            )
        end = min(end, file_size - 1)
        status_code = 206

    content_length = (end - start) + 1

    def file_iterator(chunk_size: int = 262144):
        remaining = content_length
        with file_path.open("rb") as fh:
            fh.seek(start)
            while remaining > 0:
                chunk = fh.read(min(chunk_size, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(content_length),
        "Content-Disposition": f'inline; filename="{video.original_filename}"',
    }
    if status_code == 206:
        headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"

    return StreamingResponse(
        file_iterator(),
        status_code=status_code,
        media_type=media_type,
        headers=headers,
    )


def _resolve_fps_and_frame_count(capture: cv2.VideoCapture, fallback_fps: float) -> tuple[float, int]:
    raw_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    fps = raw_fps if math.isfinite(raw_fps) and 0.1 <= raw_fps <= 240.0 else (
        fallback_fps if math.isfinite(fallback_fps) and 0.1 <= fallback_fps <= 240.0 else 25.0
    )
    raw_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    total_frames = int(raw_count) if math.isfinite(raw_count) and raw_count > 0 else 0
    return fps, total_frames


def _encode_jpeg_frame(frame) -> bytes:
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        raise ValidationError("Failed to encode video frame as JPEG.", error_code="FRAME_ENCODE_FAILED")
    return encoded.tobytes()


def _extract_single_jpeg(file_path: str, timestamp_seconds: float, fallback_fps: float) -> bytes:
    capture = cv2.VideoCapture(file_path)
    if not capture.isOpened():
        capture.release()
        raise ValidationError("Unable to open video file for frame extraction.", error_code="VIDEO_DECODE_FAILED")
    try:
        fps, total_frames = _resolve_fps_and_frame_count(capture, fallback_fps)
        target_idx = max(0, int(round(timestamp_seconds * fps)))
        if total_frames > 0:
            target_idx = min(target_idx, total_frames - 1)

        if target_idx > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ok, frame = capture.read()

        # Near EOF some codecs fail exact-frame seek; retry one step earlier or at frame 0.
        if (not ok or frame is None) and total_frames > 0 and target_idx > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, target_idx - 2))
            ok, frame = capture.read()
        if not ok or frame is None:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = capture.read()
        if not ok or frame is None:
            raise ValidationError("No decodable frame available at requested timestamp.", error_code="FRAME_NOT_AVAILABLE")
        return _encode_jpeg_frame(frame)
    finally:
        capture.release()


def _read_next_mjpeg_part(capture: cv2.VideoCapture) -> bytes | None:
    ok, frame = capture.read()
    if not ok or frame is None:
        return None
    jpeg_bytes = _encode_jpeg_frame(frame)
    header = (
        f"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg_bytes)}\r\n\r\n"
    ).encode("ascii")
    return header + jpeg_bytes + b"\r\n"


@router.get("/{video_id}/frame")
def get_video_frame(
    video_id: str,
    t: float = Query(default=0.0, ge=0.0),
    container: AppContainer = Depends(get_container),
):
    video = container.video_service.get_video(video_id)
    file_path = container.video_service.resolve_video_file(video)
    jpeg_bytes = _extract_single_jpeg(str(file_path), t, video.fps)
    return Response(
        content=jpeg_bytes,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=60"},
    )


@router.get("/{video_id}/mjpeg")
def stream_video_mjpeg(
    video_id: str,
    request: Request,
    start: float = Query(default=0.0, ge=0.0),
    speed: float = Query(default=1.0, ge=0.25, le=4.0),
    container: AppContainer = Depends(get_container),
):
    video = container.video_service.get_video(video_id)
    file_path = container.video_service.resolve_video_file(video)

    capture = cv2.VideoCapture(str(file_path))
    if not capture.isOpened():
        capture.release()
        raise ValidationError("Unable to open video file for MJPEG streaming.", error_code="VIDEO_DECODE_FAILED")

    fps, total_frames = _resolve_fps_and_frame_count(capture, video.fps)
    start_idx = max(0, int(round(start * fps)))
    if total_frames > 0:
        start_idx = min(start_idx, total_frames - 1)
    if start_idx > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

    frame_interval = 1.0 / (fps * speed)

    async def mjpeg_generator():
        try:
            next_deadline = time.monotonic()
            while True:
                if await request.is_disconnected():
                    break
                part = await asyncio.to_thread(_read_next_mjpeg_part, capture)
                if part is None:
                    break
                yield part
                next_deadline += frame_interval
                delay = next_deadline - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(min(delay, 1.0))
                else:
                    # Resync if encoding/network fell behind so bursts don't fast-forward
                    next_deadline = time.monotonic()
        finally:
            capture.release()

    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
