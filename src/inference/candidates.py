from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CandidateConfig:
    probability_threshold: float = 0.50
    min_area: int = 100
    opening_kernel: int = 3
    closing_kernel: int = 5
    boundary_margin: int = 2


def extract_candidates(
    probability_map: np.ndarray,
    config: CandidateConfig | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """
    Extract changed-region candidates from a TRINETRA probability map.

    Args:
        probability_map: Float probability map [H, W], values in [0, 1].
        config: Candidate extraction configuration.

    Returns:
        Binary mask [H, W], uint8 with values 0 or 255.
        List of candidate dictionaries in original video coordinates.
    """

    if config is None:
        config = CandidateConfig()

    if not isinstance(probability_map, np.ndarray):
        raise TypeError("Expected probability_map as a NumPy array.")

    if probability_map.ndim != 2:
        raise ValueError(
            f"Expected a 2D probability map, got {probability_map.shape}."
        )

    if probability_map.size == 0:
        raise ValueError("Probability map cannot be empty.")

    if not np.issubdtype(probability_map.dtype, np.number):
        raise TypeError("Probability map must contain numeric values.")

    if not np.isfinite(probability_map).all():
        raise ValueError("Probability map contains NaN or infinite values.")

    if probability_map.min() < 0 or probability_map.max() > 1:
        raise ValueError("Probability values must be between 0 and 1.")

    if not 0 < config.probability_threshold <= 1:
        raise ValueError("Probability threshold must be in (0, 1].")

    if config.min_area < 1:
        raise ValueError("Minimum area must be at least 1.")

    if config.opening_kernel < 1 or config.closing_kernel < 1:
        raise ValueError("Morphology kernel sizes must be positive.")

    probability_map = probability_map.astype(np.float32, copy=False)

    image_height, image_width = probability_map.shape

    # Initial binary mask from TRINETRA probabilities.
    mask = (
        probability_map >= config.probability_threshold
    ).astype(np.uint8) * 255

    # Remove isolated noise.
    if config.opening_kernel > 1:
        opening = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (config.opening_kernel, config.opening_kernel),
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, opening
        )

    # Fill small gaps inside changed regions.
    if config.closing_kernel > 1:
        closing = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (config.closing_kernel, config.closing_kernel),
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, closing
        )

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
        ltype=cv2.CV_32S,
    )

    candidates = []

    for component_id in range(1, count):
        x, y, width, height, area = stats[component_id]

        if area < config.min_area:
            continue

        component_mask = labels == component_id
        component_probabilities = probability_map[component_mask]

        center_x, center_y = centroids[component_id]

        right = int(x + width)
        bottom = int(y + height)

        margin = config.boundary_margin

        touches_boundary = (
            x <= margin
            or y <= margin
            or right >= image_width - margin
            or bottom >= image_height - margin
        )

        candidates.append({
            "bbox": [
                int(x),
                int(y),
                int(width),
                int(height),
            ],
            "center": [
                round(float(center_x), 2),
                round(float(center_y), 2),
            ],
            "area": int(area),
            "mean_probability": round(
                float(component_probabilities.mean()), 5
            ),
            "max_probability": round(
                float(component_probabilities.max()), 5
            ),
            "boundary": bool(touches_boundary),
        })

    candidates.sort(
        key=lambda candidate: candidate["area"],
        reverse=True,
    )

    return mask, candidates
