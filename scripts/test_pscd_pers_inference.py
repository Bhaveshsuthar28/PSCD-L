
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from scripts.evaluate_pscd_pers import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_INDEX,
    DINOv2Backbone,
    DIRECTIONS,
    load_binary_mask,
    load_trained_model,
    preprocess_image,
    resolve_dataset_path,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "visualizations"
    / "pscd_pers_smoke_test.png"
)

THUMB_WIDTH = 400
THUMB_HEIGHT = 231

COLUMNS = [
    "T0 - Reference",
    "T1 - Current",
    "Ground Truth",
    "Valid Region",
    "Prediction",
    "GT / Prediction Overlay",
]


def create_overlay(
    image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
) -> Image.Image:

    result = image.astype(np.float32).copy()

    # Ground truth: green.
    result[ground_truth] = (
        result[ground_truth] * 0.45
        + np.array([0, 255, 0]) * 0.55
    )

    # Prediction: red.
    result[prediction] = (
        result[prediction] * 0.45
        + np.array([255, 0, 0]) * 0.55
    )

    # Overlap: yellow.
    overlap = ground_truth & prediction

    result[overlap] = (
        image[overlap].astype(np.float32) * 0.25
        + np.array([255, 255, 0]) * 0.75
    )

    return Image.fromarray(
        np.clip(result, 0, 255).astype(np.uint8)
    )


def make_thumbnail(image: Image.Image) -> Image.Image:

    return image.resize(
        (THUMB_WIDTH, THUMB_HEIGHT),
        Image.Resampling.LANCZOS,
    )


def main() -> None:

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    with DEFAULT_INDEX.open(
        "r",
        encoding="utf-8",
    ) as file:
        records = json.load(file)

    selected_records = {}

    for direction in DIRECTIONS:

        matches = [
            record
            for record in records
            if record["sample_id"].endswith(f"_{direction}")
        ]

        if not matches:
            raise RuntimeError(
                f"No sample found for direction {direction}"
            )

        selected_records[direction] = matches[0]

    print("\nSelected samples:")

    for direction, record in selected_records.items():
        print(direction, record["sample_id"])

    print("\nLoading checkpoint...")

    decoder = load_trained_model(
        DEFAULT_CHECKPOINT,
        device,
    )

    print("Loading DINOv2...")

    backbone = DINOv2Backbone(
        model_name="facebook/dinov2-base",
        device=str(device),
    )

    backbone.eval()

    calibration_masks = {}

    for direction in DIRECTIONS:

        calibration_path = (
            DEFAULT_DATASET_ROOT
            / "cmask"
            / f"mask_sensor_{direction}.png"
        )

        calibration_masks[direction] = load_binary_mask(
            calibration_path
        )

    canvas_width = THUMB_WIDTH * len(COLUMNS)
    header_height = 45
    row_height = THUMB_HEIGHT + 55

    canvas = Image.new(
        "RGB",
        (
            canvas_width,
            header_height + row_height * len(DIRECTIONS),
        ),
        "white",
    )

    draw = ImageDraw.Draw(canvas)

    for column_index, title in enumerate(COLUMNS):

        x = column_index * THUMB_WIDTH + 8

        draw.text(
            (x, 15),
            title,
            fill="black",
        )

    print("\nRunning inference...")

    with torch.inference_mode():

        for row_index, direction in enumerate(DIRECTIONS):

            record = selected_records[direction]
            sample_id = record["sample_id"]

            t0_path = resolve_dataset_path(
                record["t0_path"],
                DEFAULT_DATASET_ROOT,
            )

            t1_path = resolve_dataset_path(
                record["t1_path"],
                DEFAULT_DATASET_ROOT,
            )

            mask_path = resolve_dataset_path(
                record["mask_path"],
                DEFAULT_DATASET_ROOT,
            )

            original_t0 = Image.open(t0_path).convert("RGB")
            original_t1 = Image.open(t1_path).convert("RGB")

            ground_truth = load_binary_mask(mask_path)
            valid_region = calibration_masks[direction]

            t0 = preprocess_image(t0_path, device)
            t1 = preprocess_image(t1_path, device)

            features_t0 = backbone(t0).float()
            features_t1 = backbone(t1).float()

            logits = decoder(features_t0, features_t1)

            probabilities = torch.sigmoid(logits)[0, 0].float()

            # Recover the original image region from the letterbox.
            image_region = probabilities[108:404, 0:512]

            original_probability = torch.nn.functional.interpolate(
                image_region[None, None],
                size=(1920, 3325),
                mode="bilinear",
                align_corners=False,
            )[0, 0]

            prediction = (
                original_probability.cpu().numpy() >= 0.5
            )

            overlay = create_overlay(
                np.asarray(original_t1),
                ground_truth,
                prediction,
            )

            visual_items = [
                original_t0,
                original_t1,
                Image.fromarray(
                    ground_truth.astype(np.uint8) * 255
                ),
                Image.fromarray(
                    valid_region.astype(np.uint8) * 255
                ),
                Image.fromarray(
                    prediction.astype(np.uint8) * 255
                ),
                overlay,
            ]

            row_y = header_height + row_index * row_height

            draw.text(
                (8, row_y),
                f"Direction {direction} | {sample_id}",
                fill="black",
            )

            image_y = row_y + 22

            for column_index, visual in enumerate(visual_items):

                thumbnail = make_thumbnail(visual)

                x = column_index * THUMB_WIDTH

                canvas.paste(
                    thumbnail,
                    (x, image_y),
                )

            print(
                f"{direction}: completed | "
                f"prediction changed pixels="
                f"{prediction.sum():,} | "
                f"ground truth changed pixels="
                f"{ground_truth.sum():,}"
            )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canvas.save(
        OUTPUT_PATH,
        quality=95,
    )

    print("\nSMOKE TEST COMPLETED")
    print("Visualization:", OUTPUT_PATH)
    print("Samples tested:", len(DIRECTIONS))


if __name__ == "__main__":
    main()
