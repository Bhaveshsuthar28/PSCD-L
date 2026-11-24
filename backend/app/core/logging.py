"""Structured rotating logging with request/job context binding."""

from __future__ import annotations

from contextvars import ContextVar
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any
import sys


request_id_ctx: ContextVar[str | None] = ContextVar("trinetra_request_id", default=None)
job_id_ctx: ContextVar[str | None] = ContextVar("trinetra_job_id", default=None)
video_id_ctx: ContextVar[str | None] = ContextVar("trinetra_video_id", default=None)


class ContextJsonFormatter(logging.Formatter):
    """Format log records as compact single-line JSON with correlation IDs."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", None) or request_id_ctx.get(),
            "job_id": getattr(record, "job_id", None) or job_id_ctx.get(),
            "video_id": getattr(record, "video_id", None) or video_id_ctx.get(),
            "operation": getattr(record, "operation", None),
            "error_code": getattr(record, "error_code", None),
        }
        # Omit None fields to keep logs concise.
        compact = {k: v for k, v in payload.items() if v is not None}
        if record.exc_info:
            compact["exception"] = self.formatException(record.exc_info)
        return json.dumps(compact, ensure_ascii=False, default=str)


def configure_logging(
    log_dir: Path,
    *,
    level: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Logger:
    """Configure root/trinetra structured logging with bounded rotating file output."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "trinetra_backend.log"

    logger = logging.getLogger("trinetra")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    # Clear existing handlers on reconfiguration (e.g., tests).
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    formatter = ContextJsonFormatter()

    file_handler = RotatingFileHandler(
        filename=str(log_file),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def get_logger(name: str = "trinetra") -> logging.Logger:
    """Return a child logger under the trinetra namespace."""
    if name == "trinetra" or name.startswith("trinetra."):
        return logging.getLogger(name)
    return logging.getLogger(f"trinetra.{name}")
