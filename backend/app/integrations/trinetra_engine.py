"""Managed single-instance lifecycle wrapper for TrinetraInferenceEngine."""

from __future__ import annotations

import hashlib
from pathlib import Path
import threading
import time
from typing import Any, Callable

import torch

from backend.app.core.config import Settings
from backend.app.core.exceptions import ModelInitializationError
from backend.app.core.logging import get_logger


logger = get_logger("integrations.engine")


class ManagedInferenceEngine:
    """
    Thread-safe lifecycle manager for the frozen TRINETRA inference engine.

    Enforces single-model residency for 4 GB VRAM safety (RTX 3050 Laptop GPU)
    and never forces model initialization during HTTP health checks.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        engine_factory: Callable[[Path, str | None], Any] | None = None,
    ) -> None:
        self.settings = settings
        self._engine_factory = engine_factory
        self._lock = threading.RLock()
        self._inference_semaphore = threading.Semaphore(settings.gpu_job_concurrency)
        self._engine: Any | None = None
        self._load_time_seconds: float | None = None
        self._last_error: str | None = None
        self._sha256_verified: bool | None = None

    @staticmethod
    def compute_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest().upper()

    def checkpoint_exists(self) -> bool:
        return self.settings.resolved_checkpoint_path.is_file()

    def is_initialized(self) -> bool:
        with self._lock:
            return self._engine is not None and bool(getattr(self._engine, "loaded", True))

    def get_model_status(self) -> dict[str, Any]:
        with self._lock:
            cuda_available = torch.cuda.is_available()
            device_str = None
            if self._engine is not None:
                device_str = str(getattr(self._engine, "device", ""))
            elif self.settings.inference_device:
                device_str = self.settings.inference_device
            else:
                device_str = "cuda" if cuda_available else "cpu"

            return {
                "architecture": "DINOv2 ViT-B/14 + Trainable Change Detection Decoder",
                "backbone": "facebook/dinov2-base",
                "feature_dimensions": [1, 768, 37, 37],
                "output_logits_dimensions": [1, 1, 512, 512],
                "checkpoint_exists": self.checkpoint_exists(),
                "expected_sha256": self.settings.expected_checkpoint_sha256,
                "checkpoint_sha256_verified": self._sha256_verified,
                "model_initialized": self.is_initialized(),
                "device": device_str,
                "load_time_seconds": self._load_time_seconds,
                "last_error": self._last_error,
            }

    def acquire_engine(self) -> Any:
        """Load the frozen model if not already loaded and return the engine instance."""
        with self._lock:
            if self._engine is not None and getattr(self._engine, "loaded", True):
                return self._engine

            checkpoint_path = self.settings.resolved_checkpoint_path
            if self._engine_factory is None:
                if not checkpoint_path.is_file():
                    self._sha256_verified = False
                    self._last_error = "Frozen checkpoint file is missing."
                    raise ModelInitializationError(
                        "Frozen TRINETRA checkpoint file was not found.",
                        error_code="CHECKPOINT_MISSING",
                    )

                actual_hash = self.compute_sha256(checkpoint_path)
                if actual_hash != self.settings.expected_checkpoint_sha256:
                    self._sha256_verified = False
                    self._last_error = "Checkpoint SHA-256 verification failed."
                    raise ModelInitializationError(
                        "Frozen checkpoint SHA-256 verification failed.",
                        error_code="CHECKPOINT_HASH_MISMATCH",
                    )
                self._sha256_verified = True

            start_ts = time.perf_counter()
            try:
                if self._engine_factory is not None:
                    engine = self._engine_factory(
                        checkpoint_path,
                        self.settings.inference_device,
                    )
                    self._sha256_verified = True
                else:
                    from src.inference.engine import TrinetraInferenceEngine

                    engine = TrinetraInferenceEngine(
                        checkpoint_path=checkpoint_path,
                        device=self.settings.inference_device,
                    )

                self._engine = engine
                self._load_time_seconds = round(time.perf_counter() - start_ts, 4)
                self._last_error = None
                logger.info(
                    "TRINETRA inference engine initialized in %.3fs",
                    self._load_time_seconds,
                    extra={"operation": "model_init"},
                )
                return self._engine

            except ModelInitializationError:
                raise
            except torch.cuda.OutOfMemoryError as exc:
                self._last_error = "CUDA out of memory during model initialization."
                self._cleanup_cuda()
                logger.exception(
                    "CUDA OOM during model initialization",
                    extra={"operation": "model_init", "error_code": "CUDA_OOM"},
                )
                raise ModelInitializationError(
                    "Insufficient GPU VRAM to initialize TRINETRA model.",
                    error_code="CUDA_OUT_OF_MEMORY",
                ) from exc
            except Exception as exc:
                self._last_error = str(exc)
                self._cleanup_cuda()
                logger.exception(
                    "Model initialization failed: %s",
                    exc,
                    extra={"operation": "model_init", "error_code": "MODEL_INIT_FAILED"},
                )
                raise ModelInitializationError(
                    "Failed to initialize TRINETRA inference engine.",
                    error_code="MODEL_INITIALIZATION_FAILED",
                ) from exc

    def acquire_execution_slot(self, timeout: float | None = None) -> bool:
        if timeout is None:
            self._inference_semaphore.acquire()
            return True
        return self._inference_semaphore.acquire(timeout=timeout)

    def release_execution_slot(self) -> None:
        self._inference_semaphore.release()

    def clear_reference(self) -> None:
        with self._lock:
            if self._engine is not None and hasattr(self._engine, "clear_reference"):
                try:
                    self._engine.clear_reference()
                except Exception:
                    pass
            self._cleanup_cuda()

    def unload(self) -> None:
        with self._lock:
            if self._engine is not None:
                try:
                    self._engine.close()
                except Exception as exc:
                    logger.warning("Error while closing inference engine: %s", exc)
                finally:
                    self._engine = None
            self._cleanup_cuda()

    @staticmethod
    def _cleanup_cuda() -> None:
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
