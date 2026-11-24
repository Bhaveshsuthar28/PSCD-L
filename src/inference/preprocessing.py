from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


IMAGE_SIZE = 512
MODEL_SIZE = 518
PATCH_SIZE = 14

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class PreprocessMetadata:
    original_width: int
    original_height: int
    resized_width: int
    resized_height: int
    top: int
    left: int
    canvas_size: int = IMAGE_SIZE
    model_size: int = MODEL_SIZE


def preprocess_frame(
    frame_bgr: np.ndarray,
) -> tuple[torch.Tensor, PreprocessMetadata]:
    """
    Convert an OpenCV BGR frame into a normalized DINOv2 input.

    Returns:
        Tensor: [1, 3, 518, 518], float32.
        Metadata: Information required to restore predictions.
    """

    if not isinstance(frame_bgr, np.ndarray):
        raise TypeError("Expected a NumPy video frame.")

    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(
            f"Expected BGR frame [H, W, 3], got {frame_bgr.shape}."
        )

    if frame_bgr.dtype != np.uint8:
        raise TypeError(
            f"Expected uint8 frame, got {frame_bgr.dtype}."
        )

    original_height, original_width = frame_bgr.shape[:2]

    if original_width <= 0 or original_height <= 0:
        raise ValueError("Frame dimensions must be positive.")

    resized_width = IMAGE_SIZE
    resized_height = round(
        original_height * IMAGE_SIZE / original_width
    )

    if resized_height <= 0 or resized_height > IMAGE_SIZE:
        raise ValueError(
            f"Unsupported aspect ratio: {original_width}x{original_height}."
        )

    # OpenCV BGR -> RGB.
    frame_rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])

    image = Image.fromarray(frame_rgb)

    image = image.resize(
        (resized_width, resized_height),
        Image.Resampling.BILINEAR,
    )

    image_array = np.asarray(image, dtype=np.float32) / 255.0

    image_tensor = torch.from_numpy(
        np.ascontiguousarray(image_array)
    ).permute(2, 0, 1)

    mean = torch.tensor(
        IMAGENET_MEAN, dtype=torch.float32
    ).view(3, 1, 1)

    std = torch.tensor(
        IMAGENET_STD, dtype=torch.float32
    ).view(3, 1, 1)

    image_tensor = (image_tensor - mean) / std

    top = (IMAGE_SIZE - resized_height) // 2
    left = (IMAGE_SIZE - resized_width) // 2

    # Zero padding is applied in normalized space.
    canvas = torch.zeros(
        (3, IMAGE_SIZE, IMAGE_SIZE),
        dtype=torch.float32,
    )

    canvas[
        :,
        top:top + resized_height,
        left:left + resized_width,
    ] = image_tensor

    # 512 -> 518. Both dimensions are divisible by DINOv2 patch size 14.
    model_input = F.pad(
        canvas,
        (0, 6, 0, 6),
        mode="constant",
        value=0.0,
    )

    if model_input.shape != (3, MODEL_SIZE, MODEL_SIZE):
        raise RuntimeError(
            f"Unexpected model input shape: {tuple(model_input.shape)}"
        )

    metadata = PreprocessMetadata(
        original_width=original_width,
        original_height=original_height,
        resized_width=resized_width,
        resized_height=resized_height,
        top=top,
        left=left,
    )

    return model_input.unsqueeze(0), metadata


def restore_probability_map(
    probability_map: torch.Tensor,
    metadata: PreprocessMetadata,
) -> torch.Tensor:
    """
    Restore a [512, 512] probability map to original frame dimensions.

    Returns:
        Tensor: [original_height, original_width], float32.
    """

    if probability_map.ndim == 4:
        if probability_map.shape[:2] != (1, 1):
            raise ValueError(
                f"Expected [1, 1, H, W], got {tuple(probability_map.shape)}."
            )
        probability_map = probability_map[0, 0]

    elif probability_map.ndim != 2:
        raise ValueError(
            f"Expected [H, W] or [1, 1, H, W], "
            f"got {tuple(probability_map.shape)}."
        )

    if probability_map.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(
            f"Expected 512x512 probability map, "
            f"got {tuple(probability_map.shape)}."
        )

    # Remove the letterbox and retain only the resized image region.
    image_region = probability_map[
        metadata.top:metadata.top + metadata.resized_height,
        metadata.left:metadata.left + metadata.resized_width,
    ]

    restored = F.interpolate(
        image_region[None, None].float(),
        size=(
            metadata.original_height,
            metadata.original_width,
        ),
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    return restored
