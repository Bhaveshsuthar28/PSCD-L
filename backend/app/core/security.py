"""Security, identifier validation, and path containment utilities."""

from __future__ import annotations

import os
from pathlib import Path
import re
import unicodedata
import uuid

from backend.app.core.exceptions import (
    ForbiddenPathError,
    InvalidIdentifierError,
    ValidationError,
)


MAX_FILENAME_LENGTH = 200
WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}

# Explicit allowlist of user-configurable inference/temporal parameters.
# Note: Thresholds and checkpoint paths are NOT arbitrarily exposed unless safe.
ALLOWLISTED_CANDIDATE_KEYS: dict[str, tuple[type, float, float]] = {
    "min_area": (int, 20, 100000),
    "opening_kernel": (int, 1, 15),
    "closing_kernel": (int, 1, 21),
    "boundary_margin": (int, 0, 50),
}

ALLOWLISTED_TEMPORAL_KEYS: dict[str, tuple[type, float, float]] = {
    "min_persistence_seconds": (float, 1.0, 3600.0),
    "min_observations": (int, 2, 1000),
    "settling_min_seconds": (float, 0.0, 600.0),
    "min_motion_samples": (int, 1, 100),
    "settling_motion_window_samples": (int, 1, 100),
    "max_motion_history": (int, 1, 100000),
}

FORBIDDEN_CONFIG_KEYS = {
    "checkpoint",
    "checkpoint_path",
    "model_path",
    "output_dir",
    "video_path",
    "device",
    "__class__",
    "__dict__",
}


def validate_uuid(value: str, label: str = "id") -> str:
    """Validate and return a canonical UUID4/UUID string."""
    if not isinstance(value, str) or not value.strip():
        raise InvalidIdentifierError(f"Invalid {label}: expected non-empty UUID.")
    cleaned = value.strip()
    try:
        parsed = uuid.UUID(cleaned)
    except ValueError as exc:
        raise InvalidIdentifierError(
            f"Invalid {label} '{cleaned}': must be a valid UUID."
        ) from exc
    return str(parsed)


def sanitize_filename(raw_filename: str | None) -> str:
    """
    Sanitize an untrusted upload filename while preserving safe Unicode characters.

    Rejects explicit traversal payloads and neutralizes path separators and
    control characters. Never uses user filenames directly for storage paths.
    """
    if not raw_filename or not isinstance(raw_filename, str):
        raise ValidationError("Filename is missing.", error_code="INVALID_FILENAME")

    # Reject raw null bytes or traversal segments before stripping.
    if "\x00" in raw_filename:
        raise ValidationError(
            "Filename contains illegal null bytes.",
            error_code="INVALID_FILENAME",
        )

    normalized_raw = raw_filename.replace("\\", "/")
    parts = [p for p in normalized_raw.split("/") if p]
    if any(part in {"..", "."} for part in parts[:-1]) or ".." in normalized_raw:
        raise ValidationError(
            "Filename contains illegal path traversal sequence.",
            error_code="INVALID_FILENAME",
        )

    leaf = parts[-1] if parts else ""
    leaf = unicodedata.normalize("NFKC", leaf).strip()
    # Remove control characters and characters unsafe on Windows/POSIX filesystems.
    leaf = re.sub(r'[\x00-\x1f\x7f<>:"/\\|?*]', "_", leaf)
    leaf = leaf.strip(". ")

    if not leaf or leaf in {".", ".."}:
        raise ValidationError("Filename is empty after sanitization.", error_code="INVALID_FILENAME")

    stem, ext = os.path.splitext(leaf)
    if stem.upper() in WINDOWS_RESERVED_NAMES:
        stem = f"_{stem}"

    max_stem_len = max(1, MAX_FILENAME_LENGTH - len(ext))
    if len(stem) > max_stem_len:
        stem = stem[:max_stem_len]

    sanitized = f"{stem}{ext}"
    if len(sanitized) > MAX_FILENAME_LENGTH:
        sanitized = sanitized[:MAX_FILENAME_LENGTH]
    return sanitized


def resolve_within_root(root: Path, relative_path: str | Path) -> Path:
    """
    Resolve `relative_path` under `root` and verify it never escapes `root`.
    """
    root_resolved = root.resolve()
    rel = Path(relative_path)
    if rel.is_absolute():
        raise ForbiddenPathError("Absolute paths are not permitted.")

    if ".." in rel.parts:
        raise ForbiddenPathError("Path traversal segments are forbidden.")

    candidate = (root_resolved / rel).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise ForbiddenPathError(
            "Resolved path escapes the configured storage root."
        ) from exc

    return candidate


def validate_processing_configuration(
    config_payload: dict | None,
) -> dict[str, dict]:
    """
    Validate user-supplied processing configuration against a strict allowlist.

    Returns normalized {'candidate_config': {...}, 'temporal_config': {...}}.
    Rejects unknown keys, forbidden keys, non-dict payloads, or out-of-range values.
    """
    if config_payload is None:
        return {"candidate_config": {}, "temporal_config": {}}

    if not isinstance(config_payload, dict):
        raise ValidationError(
            "Configuration must be a JSON object.",
            error_code="INVALID_CONFIGURATION",
        )

    allowed_top_keys = {"candidate_config", "temporal_config"}
    unknown_top = set(config_payload.keys()) - allowed_top_keys
    if unknown_top:
        raise ValidationError(
            f"Unsupported configuration section(s): {sorted(unknown_top)}",
            error_code="UNSUPPORTED_CONFIGURATION_KEY",
            details={"unsupported_keys": sorted(unknown_top)},
        )

    def _validate_section(
        section_name: str,
        raw_section: object,
        allowlist: dict[str, tuple[type, float, float]],
    ) -> dict:
        if raw_section is None:
            return {}
        if not isinstance(raw_section, dict):
            raise ValidationError(
                f"'{section_name}' must be a JSON object.",
                error_code="INVALID_CONFIGURATION",
            )
        validated: dict = {}
        for key, value in raw_section.items():
            if key in FORBIDDEN_CONFIG_KEYS or key not in allowlist:
                raise ValidationError(
                    f"Unsupported configuration key '{section_name}.{key}'.",
                    error_code="UNSUPPORTED_CONFIGURATION_KEY",
                )
            expected_type, min_val, max_val = allowlist[key]
            if isinstance(value, bool):
                raise ValidationError(
                    f"Invalid boolean value for '{section_name}.{key}'.",
                    error_code="INVALID_CONFIGURATION",
                )
            if expected_type is int:
                if not isinstance(value, int):
                    raise ValidationError(
                        f"'{section_name}.{key}' must be an integer.",
                        error_code="INVALID_CONFIGURATION",
                    )
                coerced = int(value)
            else:
                if not isinstance(value, (int, float)):
                    raise ValidationError(
                        f"'{section_name}.{key}' must be numeric.",
                        error_code="INVALID_CONFIGURATION",
                    )
                coerced = float(value)

            if coerced < min_val or coerced > max_val:
                raise ValidationError(
                    f"'{section_name}.{key}' must be between {min_val} and {max_val}.",
                    error_code="INVALID_CONFIGURATION",
                )
            validated[key] = coerced
        return validated

    candidate_cfg = _validate_section(
        "candidate_config",
        config_payload.get("candidate_config"),
        ALLOWLISTED_CANDIDATE_KEYS,
    )
    temporal_cfg = _validate_section(
        "temporal_config",
        config_payload.get("temporal_config"),
        ALLOWLISTED_TEMPORAL_KEYS,
    )

    # Cross-field temporal consistency check
    min_obs = temporal_cfg.get("min_observations")
    min_motion = temporal_cfg.get("min_motion_samples")
    window = temporal_cfg.get("settling_motion_window_samples")
    max_hist = temporal_cfg.get("max_motion_history")
    effective_min_motion = min_motion if min_motion is not None else 3
    effective_window = window if window is not None else 5
    if effective_window < effective_min_motion:
        raise ValidationError(
            "settling_motion_window_samples cannot be less than min_motion_samples.",
            error_code="INVALID_CONFIGURATION",
        )
    if max_hist is not None and max_hist < effective_min_motion:
        raise ValidationError(
            "max_motion_history cannot be less than min_motion_samples.",
            error_code="INVALID_CONFIGURATION",
        )
    if max_hist is not None and effective_window > max_hist:
        raise ValidationError(
            "settling_motion_window_samples cannot exceed max_motion_history.",
            error_code="INVALID_CONFIGURATION",
        )
    if min_obs is not None and min_obs < 2:
        raise ValidationError(
            "min_observations must be at least 2.",
            error_code="INVALID_CONFIGURATION",
        )

    return {
        "candidate_config": candidate_cfg,
        "temporal_config": temporal_cfg,
    }
