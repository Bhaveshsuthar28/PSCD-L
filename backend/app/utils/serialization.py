"""Safe serialization for NumPy values, non-finite floats, and datetimes."""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any

import numpy as np


def utc_now() -> datetime:
    """Return timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def ensure_utc_iso(value: datetime | None) -> str | None:
    """Format a datetime as timezone-aware ISO 8601 UTC string."""
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.isoformat()


def safe_float(value: Any) -> float | None:
    """Convert float/numpy float to Python float, mapping NaN/Inf to None."""
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def sanitize_for_json(obj: Any) -> Any:
    """Recursively convert NumPy types and non-finite floats into JSON-safe values."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, np.ndarray):
        return [sanitize_for_json(item) for item in obj.tolist()]
    if isinstance(obj, datetime):
        return ensure_utc_iso(obj)
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_for_json(item) for item in obj]
    return str(obj)
