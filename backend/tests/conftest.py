"""Shared fixtures for TRINETRA backend tests (CPU/mocked, zero GPU dependency)."""

from __future__ import annotations

from pathlib import Path
import time
import uuid

import cv2
from fastapi.testclient import TestClient
import numpy as np
import pytest

from backend.app.core.config import Settings
from backend.app.main import create_app


class MockTrinetraEngine:
    """
    Deterministic lightweight mock for TrinetraInferenceEngine.

    Emits a high-probability region wherever pixel difference against the
    reference frame exceeds a small threshold, allowing full Task 2
    candidate extraction, tracking, optical flow, temporal confirmation,
    and evidence generation to run on synthetic videos without GPU.
    """

    def __init__(self, checkpoint_path: Path | None = None, device: str | None = None) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device or "cpu"
        self.loaded = True
        self.reference_frame: np.ndarray | None = None
        self.delay_per_frame: float = 0.0

    def set_reference(self, reference_frame_bgr: np.ndarray) -> None:
        self.reference_frame = reference_frame_bgr.copy()

    def predict(self, current_frame_bgr: np.ndarray) -> np.ndarray:
        if self.delay_per_frame > 0:
            time.sleep(self.delay_per_frame)
        if self.reference_frame is None:
            raise RuntimeError("Reference frame not set.")
        diff = np.abs(
            current_frame_bgr.astype(np.float32) - self.reference_frame.astype(np.float32)
        ).mean(axis=2)
        prob = np.clip(diff / 40.0, 0.0, 1.0).astype(np.float32)
        return prob

    def clear_reference(self) -> None:
        self.reference_frame = None

    def close(self) -> None:
        self.clear_reference()
        self.loaded = False


def create_synthetic_avi(
    path: Path,
    *,
    width: int = 160,
    height: int = 120,
    fps: float = 10.0,
    duration_seconds: float = 14.0,
    static_box_from_second: float | None = 1.0,
    box_rect: tuple[int, int, int, int] = (50, 40, 30, 30),
    unstable_after_second: float | None = None,
) -> Path:
    """Write a valid MJPG AVI video with an optional static changed region."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {path}")

    total_frames = int(round(fps * duration_seconds))
    x, y, w, h = box_rect
    try:
        for idx in range(total_frames):
            t = idx / fps
            frame = np.full((height, width, 3), 40, dtype=np.uint8)
            # Subtle textured background so Farneback flow is well-conditioned.
            frame[::8, ::8] = 60
            if static_box_from_second is not None and t >= static_box_from_second:
                if unstable_after_second is not None and t >= unstable_after_second:
                    # Shift/grow box dramatically and oscillate intensity to trigger post-confirmation degradation
                    shift = int((idx % 3) * 25)
                    cv2.rectangle(
                        frame,
                        (min(width - 10, x + shift), max(0, y - 20)),
                        (min(width - 1, x + shift + w + 40), min(height - 1, y + h + 30)),
                        (230, 230, 230),
                        -1,
                    )
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (220, 220, 220), -1)
            writer.write(frame)
    finally:
        writer.release()
    return path


@pytest.fixture()
def test_settings(tmp_path: Path) -> Settings:
    storage_root = tmp_path / "storage"
    log_dir = tmp_path / "logs"
    db_path = storage_root / "test_trinetra.db"
    return Settings(
        app_env="testing",
        database_url=f"sqlite:///{db_path.as_posix()}",
        storage_root=storage_root,
        log_dir=log_dir,
        max_upload_size_bytes=50 * 1024 * 1024,
        min_free_disk_bytes=1024 * 1024,
        worker_count=1,
        gpu_job_concurrency=1,
        max_queue_size=10,
        progress_db_persist_interval_seconds=0.1,
        progress_db_persist_frame_batch=1,
        sse_heartbeat_seconds=1.0,
    )


@pytest.fixture()
def mock_engine_factory():
    def _factory(checkpoint_path: Path, device: str | None) -> MockTrinetraEngine:
        return MockTrinetraEngine(checkpoint_path=checkpoint_path, device=device)

    return _factory


@pytest.fixture()
def client(test_settings: Settings, mock_engine_factory):
    app = create_app(test_settings, engine_factory=mock_engine_factory, start_workers=True)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def sample_avi_path(tmp_path: Path) -> Path:
    return create_synthetic_avi(
        tmp_path / f"sample_{uuid.uuid4().hex[:6]}.avi",
        duration_seconds=14.0,
        fps=10.0,
        static_box_from_second=1.0,
    )
