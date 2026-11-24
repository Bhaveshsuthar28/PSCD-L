"""Validated pagination and sorting helpers."""

from __future__ import annotations

import math

from backend.app.core.exceptions import ValidationError


def normalize_pagination(
    page: int = 1,
    page_size: int = 20,
    *,
    max_page_size: int = 100,
) -> tuple[int, int, int]:
    """Validate page/page_size and return (page, page_size, offset)."""
    if page < 1:
        raise ValidationError("page must be >= 1.", error_code="INVALID_PAGINATION")
    if page_size < 1 or page_size > max_page_size:
        raise ValidationError(
            f"page_size must be between 1 and {max_page_size}.",
            error_code="INVALID_PAGINATION",
        )
    offset = (page - 1) * page_size
    return page, page_size, offset


def build_pagination_meta(total_items: int, page: int, page_size: int) -> dict:
    total_pages = max(1, math.ceil(total_items / page_size)) if total_items > 0 else 0
    return {
        "page": page,
        "page_size": page_size,
        "total_items": total_items,
        "total_pages": total_pages,
        "has_next": page < total_pages,
        "has_previous": page > 1 and total_pages > 0,
    }
