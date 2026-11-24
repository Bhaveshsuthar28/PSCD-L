from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class MotionConfig:
    pyramid_scale: float = 0.5
    levels: int = 3
    window_size: int = 15
    iterations: int = 3
    polynomial_size: int = 5
    polynomial_sigma: float = 1.2
    surrounding_margin: int = 25
    epsilon: float = 1e-6


def _to_grayscale(frame: np.ndarray) -> np.ndarray:
    if not isinstance(frame, np.ndarray):
        raise TypeError("Frame must be a NumPy array.")

    if frame.size == 0:
        raise ValueError("Frame cannot be empty.")

    if frame.ndim == 2:
        gray = frame
    elif frame.ndim == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(
            f"Expected grayscale or BGR frame, got {frame.shape}."
        )

    if gray.dtype != np.uint8:
        raise TypeError("Frame must have uint8 dtype.")

    return np.ascontiguousarray(gray)


class MotionAnalyzer:
    """
    Measure local optical flow inside a candidate region.

    Global median flow is subtracted to reduce dominant camera motion.
    The analyzer reports measurements only; it does not classify objects.
    """

    def __init__(self, config: MotionConfig | None = None):
        self.config = config or MotionConfig()

        if self.config.surrounding_margin < 1:
            raise ValueError("surrounding_margin must be positive.")

        if self.config.epsilon <= 0:
            raise ValueError("epsilon must be positive.")

    def analyze_pair(
        self,
        previous_frame: np.ndarray,
        current_frame: np.ndarray,
        bbox: list[int] | tuple[int, int, int, int],
    ) -> dict[str, Any]:

        previous = _to_grayscale(previous_frame)
        current = _to_grayscale(current_frame)

        if previous.shape != current.shape:
            raise ValueError("Frame dimensions must match.")

        if len(bbox) != 4:
            raise ValueError("bbox must contain x, y, width, height.")

        x, y, width, height = [int(value) for value in bbox]

        image_height, image_width = previous.shape

        if width <= 0 or height <= 0:
            raise ValueError("Bounding-box dimensions must be positive.")

        if (
            x < 0 or y < 0
            or x + width > image_width
            or y + height > image_height
        ):
            raise ValueError("Bounding box lies outside the frame.")

        if width * height < 4:
            raise ValueError("Bounding box is too small for motion analysis.")

        flow = cv2.calcOpticalFlowFarneback(
            previous,
            current,
            None,
            self.config.pyramid_scale,
            self.config.levels,
            self.config.window_size,
            self.config.iterations,
            self.config.polynomial_size,
            self.config.polynomial_sigma,
            0,
        )

        # Estimate dominant image movement.
        global_dx = float(np.median(flow[..., 0]))
        global_dy = float(np.median(flow[..., 1]))

        # Remove dominant global movement.
        compensated = flow.copy()
        compensated[..., 0] -= global_dx
        compensated[..., 1] -= global_dy

        magnitude = np.sqrt(
            compensated[..., 0] ** 2
            + compensated[..., 1] ** 2
        )

        # Candidate-region measurements.
        region = magnitude[y:y + height, x:x + width]

        # Surrounding ring.
        margin = self.config.surrounding_margin

        outer_x1 = max(0, x - margin)
        outer_y1 = max(0, y - margin)
        outer_x2 = min(image_width, x + width + margin)
        outer_y2 = min(image_height, y + height + margin)

        surrounding = magnitude[
            outer_y1:outer_y2,
            outer_x1:outer_x2,
        ]

        background_mask = np.ones(
            surrounding.shape,
            dtype=bool,
        )

        inner_x1 = x - outer_x1
        inner_y1 = y - outer_y1

        background_mask[
            inner_y1:inner_y1 + height,
            inner_x1:inner_x1 + width,
        ] = False

        background = surrounding[background_mask]

        if background.size == 0:
            raise ValueError("Insufficient surrounding background pixels.")

        region_mean = float(region.mean())
        background_mean = float(background.mean())

        return {
            "global_motion": {
                "dx": global_dx,
                "dy": global_dy,
            },
            "region": {
                "mean": region_mean,
                "median": float(np.median(region)),
                "p95": float(np.percentile(region, 95)),
            },
            "background": {
                "mean": background_mean,
                "median": float(np.median(background)),
            },
            "region_background_ratio": (
                region_mean / (background_mean + self.config.epsilon)
            ),
            "bbox": [x, y, width, height],
        }
