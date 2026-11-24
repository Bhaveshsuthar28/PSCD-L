"""Upload file validation: extension, container signature, and streaming write."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import BinaryIO

from backend.app.core.exceptions import (
    PayloadTooLargeError,
    UnsupportedMediaError,
    ValidationError,
    VideoMetadataInvalidError,
)


def validate_video_extension(filename: str, allowed_extensions: list[str]) -> str:
    ext = Path(filename).suffix.lower()
    if not ext or ext not in allowed_extensions:
        raise UnsupportedMediaError(
            f"Unsupported video extension '{ext or 'none'}'. Allowed: {sorted(allowed_extensions)}",
            error_code="UNSUPPORTED_VIDEO_EXTENSION",
            details={"allowed_extensions": sorted(allowed_extensions)},
        )
    return ext


def validate_container_signature(file_path: Path, extension: str) -> None:
    """Inspect magic bytes to reject non-video payloads disguised with a video extension."""
    try:
        with file_path.open("rb") as handle:
            header = handle.read(32)
    except OSError as exc:
        raise VideoMetadataInvalidError("Unable to read uploaded file header.") from exc

    if len(header) < 12:
        raise VideoMetadataInvalidError(
            "Uploaded file is too small or truncated to be a valid video.",
            error_code="VIDEO_TRUNCATED",
        )

    ext = extension.lower()
    if ext == ".avi":
        if not (header[:4] == b"RIFF" and header[8:12] == b"AVI "):
            raise UnsupportedMediaError(
                "File content does not match AVI container signature.",
                error_code="INVALID_VIDEO_SIGNATURE",
            )
    elif ext in {".mp4", ".mov"}:
        box_type = header[4:8]
        valid_boxes = {b"ftyp", b"moov", b"mdat", b"wide", b"free", b"skip"}
        if box_type not in valid_boxes:
            raise UnsupportedMediaError(
                f"File content does not match {ext.upper()} container signature.",
                error_code="INVALID_VIDEO_SIGNATURE",
            )
    elif ext == ".mkv":
        if header[:4] != b"\x1a\x45\xdf\xa3":
            raise UnsupportedMediaError(
                "File content does not match MKV (EBML) container signature.",
                error_code="INVALID_VIDEO_SIGNATURE",
            )


async def stream_upload_to_temp(
    upload_file,
    temp_path: Path,
    *,
    max_size_bytes: int,
    chunk_size_bytes: int,
) -> tuple[int, str]:
    """
    Stream an async UploadFile into `temp_path` while enforcing max size
    and computing SHA-256 without loading the entire file into memory.
    """
    digest = hashlib.sha256()
    total_bytes = 0
    temp_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with temp_path.open("wb") as out:
            while True:
                chunk = await upload_file.read(chunk_size_bytes)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_size_bytes:
                    raise PayloadTooLargeError(
                        f"Upload exceeds maximum allowed size of {max_size_bytes} bytes.",
                        details={"max_upload_size_bytes": max_size_bytes},
                    )
                digest.update(chunk)
                out.write(chunk)
            out.flush()
            os.fsync(out.fileno())
    except Exception:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise

    if total_bytes == 0:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise ValidationError(
            "Uploaded file is empty (0 bytes).",
            error_code="EMPTY_UPLOAD",
        )

    return total_bytes, digest.hexdigest().upper()


def stream_sync_file_to_temp(
    src: BinaryIO,
    temp_path: Path,
    *,
    max_size_bytes: int,
    chunk_size_bytes: int = 1024 * 1024,
) -> tuple[int, str]:
    """Synchronous variant for CLI/testing ingestion."""
    digest = hashlib.sha256()
    total_bytes = 0
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temp_path.open("wb") as out:
            while True:
                chunk = src.read(chunk_size_bytes)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_size_bytes:
                    raise PayloadTooLargeError("Upload exceeds maximum allowed size.")
                digest.update(chunk)
                out.write(chunk)
            out.flush()
            os.fsync(out.fileno())
    except Exception:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise

    if total_bytes == 0:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise ValidationError("Uploaded file is empty (0 bytes).", error_code="EMPTY_UPLOAD")

    return total_bytes, digest.hexdigest().upper()
