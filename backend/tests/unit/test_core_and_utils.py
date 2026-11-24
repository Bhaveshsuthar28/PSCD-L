"""Unit tests for core config, security, serialization, pagination, and video validation."""

from __future__ import annotations

from datetime import datetime, timezone
import io
import math
from pathlib import Path

import numpy as np
import pytest

from backend.app.core.config import Settings
from backend.app.core.exceptions import (
    ForbiddenPathError,
    InvalidIdentifierError,
    PayloadTooLargeError,
    UnsupportedMediaError,
    ValidationError,
    VideoMetadataInvalidError,
)
from backend.app.core.security import (
    resolve_within_root,
    sanitize_filename,
    validate_processing_configuration,
    validate_uuid,
)
from backend.app.utils.file_validation import (
    stream_sync_file_to_temp,
    validate_container_signature,
    validate_video_extension,
)
from backend.app.utils.pagination import build_pagination_meta, normalize_pagination
from backend.app.utils.serialization import safe_float, sanitize_for_json
from backend.app.utils.video_metadata import extract_and_validate_video_metadata
from backend.tests.conftest import create_synthetic_avi


def test_settings_validation_normalizes_and_rejects_bad_sha():
    cfg = Settings(
        cors_origins="http://localhost:5173, http://127.0.0.1:5173",
        allowed_video_extensions="avi, .MP4",
        expected_checkpoint_sha256="a" * 64,
    )
    assert cfg.cors_origins == ["http://localhost:5173", "http://127.0.0.1:5173"]
    assert cfg.allowed_video_extensions == [".avi", ".mp4"]
    assert cfg.expected_checkpoint_sha256 == "A" * 64

    with pytest.raises(Exception):
        Settings(expected_checkpoint_sha256="not-a-sha")


def test_uuid_validation():
    valid = "7F89793B-8A97-427E-8777-96B7846028B6"
    assert validate_uuid(valid) == valid.lower()
    for bad in ["", "not-a-uuid", "../etc/passwd", "1234"]:
        with pytest.raises(InvalidIdentifierError):
            validate_uuid(bad)


def test_sanitize_filename_handles_unicode_long_and_traversal():
    assert sanitize_filename("caméra_日本語_01.avi") == "caméra_日本語_01.avi"
    long_name = ("a" * 260) + ".mp4"
    sanitized = sanitize_filename(long_name)
    assert len(sanitized) <= 200
    assert sanitized.endswith(".mp4")
    assert sanitize_filename("CON.avi").startswith("_CON")

    for bad in ["../../secret.avi", "..\\..\\evil.mp4", "bad\x00name.avi", ""]:
        with pytest.raises(ValidationError):
            sanitize_filename(bad)


def test_resolve_within_root_prevents_escape(tmp_path: Path):
    root = tmp_path / "storage"
    root.mkdir()
    ok = resolve_within_root(root, "uploads/test.avi")
    assert str(ok).startswith(str(root.resolve()))

    with pytest.raises(ForbiddenPathError):
        resolve_within_root(root, "../outside.txt")
    with pytest.raises(ForbiddenPathError):
        resolve_within_root(root, tmp_path / "abs.txt")


def test_validate_processing_configuration_allowlist():
    valid = validate_processing_configuration(
        {
            "candidate_config": {"min_area": 120},
            "temporal_config": {"min_persistence_seconds": 8.0, "min_observations": 4},
        }
    )
    assert valid["candidate_config"]["min_area"] == 120
    assert valid["temporal_config"]["min_observations"] == 4

    with pytest.raises(ValidationError):
        validate_processing_configuration({"temporal_config": {"checkpoint_path": "/etc/passwd"}})
    with pytest.raises(ValidationError):
        validate_processing_configuration({"unknown_section": {}})
    with pytest.raises(ValidationError):
        validate_processing_configuration({"candidate_config": {"min_area": True}})
    with pytest.raises(ValidationError):
        validate_processing_configuration(
            {"temporal_config": {"min_motion_samples": 6, "settling_motion_window_samples": 3}}
        )


def test_serialization_handles_numpy_and_nonfinite():
    payload = {
        "int_val": np.int64(7),
        "float_val": np.float32(0.5),
        "nan_val": float("nan"),
        "inf_val": np.float64(math.inf),
        "bool_val": np.bool_(True),
        "arr": np.array([1.0, float("nan")]),
        "dt": datetime(2026, 1, 1, tzinfo=timezone.utc),
    }
    clean = sanitize_for_json(payload)
    assert clean["int_val"] == 7
    assert isinstance(clean["int_val"], int)
    assert clean["nan_val"] is None
    assert clean["inf_val"] is None
    assert clean["bool_val"] is True
    assert clean["arr"] == [1.0, None]
    assert clean["dt"].startswith("2026-01-01T00:00:00")
    assert safe_float(float("nan")) is None


def test_pagination_validation():
    assert normalize_pagination(2, 10) == (2, 10, 10)
    meta = build_pagination_meta(25, 2, 10)
    assert meta["total_pages"] == 3
    assert meta["has_next"] is True
    assert meta["has_previous"] is True

    with pytest.raises(ValidationError):
        normalize_pagination(0, 10)
    with pytest.raises(ValidationError):
        normalize_pagination(1, 500)


def test_file_validation_and_video_metadata(tmp_path: Path):
    assert validate_video_extension("clip.AVI", [".avi", ".mp4"]) == ".avi"
    with pytest.raises(UnsupportedMediaError):
        validate_video_extension("clip.txt", [".avi", ".mp4"])

    fake = tmp_path / "fake.avi"
    fake.write_bytes(b"NOT_A_RIFF_FILE_AT_ALL")
    with pytest.raises(UnsupportedMediaError):
        validate_container_signature(fake, ".avi")

    empty = tmp_path / "empty.part"
    with pytest.raises(ValidationError):
        stream_sync_file_to_temp(io.BytesIO(b""), empty, max_size_bytes=1024)

    oversize = tmp_path / "oversize.part"
    with pytest.raises(PayloadTooLargeError):
        stream_sync_file_to_temp(io.BytesIO(b"x" * 2048), oversize, max_size_bytes=1024)
    assert not oversize.exists()

    avi_path = create_synthetic_avi(tmp_path / "valid.avi", duration_seconds=2.0, fps=10.0)
    validate_container_signature(avi_path, ".avi")
    meta = extract_and_validate_video_metadata(
        avi_path,
        extension=".avi",
        max_duration_seconds=60.0,
        max_width=1920,
        max_height=1080,
    )
    assert meta.width == 160
    assert meta.height == 120
    assert meta.frame_count == 20
    assert meta.fps == pytest.approx(10.0)

    with pytest.raises(VideoMetadataInvalidError):
        extract_and_validate_video_metadata(
            avi_path,
            extension=".avi",
            max_duration_seconds=1.0,
            max_width=1920,
            max_height=1080,
        )
