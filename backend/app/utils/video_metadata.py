"""Video metadata extraction and decode verification via OpenCV."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import cv2

from backend.app.core.exceptions import VideoMetadataInvalidError


@dataclass(frozen=True)
class VideoMetadata:
    duration_seconds: float
    fps: float
    frame_count: int
    width: int
    height: int
    codec: str | None
    format: str


def _decode_fourcc(fourcc_float: float) -> str | None:
    if not math.isfinite(fourcc_float) or fourcc_float <= 0:
        return None
    val = int(fourcc_float)
    chars = [chr((val >> (8 * i)) & 0xFF) for i in range(4)]
    cleaned = "".join(c for c in chars if 32 <= ord(c) <= 126).strip()
    return cleaned or None


def extract_and_validate_video_metadata(
    video_path: Path,
    *,
    extension: str,
    max_duration_seconds: float,
    max_width: int,
    max_height: int,
) -> VideoMetadata:
    """
    Open the video with OpenCV, verify metadata, decode first frame, and
    enforce configured resolution/duration bounds. Always releases VideoCapture.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise VideoMetadataInvalidError(
            "Unable to open uploaded video with OpenCV decoder.",
            error_code="VIDEO_DECODE_FAILED",
        )

    try:
        raw_fps = float(capture.get(cv2.CAP_PROP_FPS))
        raw_frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        raw_width = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        raw_height = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        raw_fourcc = float(capture.get(cv2.CAP_PROP_FOURCC))

        if not math.isfinite(raw_fps) or raw_fps <= 0 or raw_fps > 240.0:
            raise VideoMetadataInvalidError(
                f"Video has invalid FPS metadata ({raw_fps}).",
                error_code="VIDEO_METADATA_INVALID",
            )

        if not math.isfinite(raw_frame_count) or raw_frame_count < 1:
            raise VideoMetadataInvalidError(
                "Video contains zero or invalid frame count.",
                error_code="VIDEO_METADATA_INVALID",
            )

        if not (math.isfinite(raw_width) and math.isfinite(raw_height)):
            raise VideoMetadataInvalidError("Video dimensions are non-finite.")

        frame_count = int(round(raw_frame_count))
        width = int(round(raw_width))
        height = int(round(raw_height))

        if width <= 0 or height <= 0:
            raise VideoMetadataInvalidError(
                f"Video dimensions must be positive (got {width}x{height})."
            )

        if width > max_width or height > max_height:
            raise VideoMetadataInvalidError(
                f"Video resolution {width}x{height} exceeds configured limit {max_width}x{max_height}.",
                error_code="VIDEO_DIMENSIONS_EXCEEDED",
            )

        duration_seconds = frame_count / raw_fps
        if not math.isfinite(duration_seconds) or duration_seconds <= 0:
            raise VideoMetadataInvalidError("Computed video duration is invalid.")

        if duration_seconds > max_duration_seconds:
            raise VideoMetadataInvalidError(
                f"Video duration ({duration_seconds:.2f}s) exceeds limit ({max_duration_seconds:.2f}s).",
                error_code="VIDEO_DURATION_EXCEEDED",
            )

        # Decode the first frame to verify codec readability and non-corrupted stream.
        ok, first_frame = capture.read()
        if not ok or first_frame is None or first_frame.size == 0:
            raise VideoMetadataInvalidError(
                "Could not decode the first frame of the uploaded video.",
                error_code="VIDEO_CORRUPTED_OR_UNREADABLE",
            )

        if first_frame.ndim != 3 or first_frame.shape[2] != 3:
            raise VideoMetadataInvalidError(
                f"Decoded frame has unsupported channel layout {first_frame.shape}."
            )

        actual_h, actual_w = first_frame.shape[:2]
        if actual_w != width or actual_h != height:
            width = actual_w
            height = actual_h

        return VideoMetadata(
            duration_seconds=round(duration_seconds, 6),
            fps=round(raw_fps, 6),
            frame_count=frame_count,
            width=width,
            height=height,
            codec=_decode_fourcc(raw_fourcc),
            format=extension.lstrip(".").lower(),
        )
    finally:
        capture.release()
