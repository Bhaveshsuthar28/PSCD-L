from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import normalize, to_tensor

from src.models.backbone import DINOv2Backbone
from src.lightning_module import TrinetraLightningModule

from scripts.evaluate_pscd import (
    resolve_dataset_path,
    validate_index,
    load_binary_mask,
    load_trained_model,
    calculate_metrics,
    DEFAULT_INDEX,
    DEFAULT_DATASET_ROOT,
    DEFAULT_CHECKPOINT,
)

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

WIDTH = 4096
HEIGHT = 1152

TILE_SIZE = 1152
MODEL_SIZE = 518
IMAGE_SIZE = 512

THRESHOLD = 0.5
MASK_THRESHOLD = 128

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

DEFAULT_OUTPUT = (
    Path("outputs/evaluation/pscd_tiled_exploratory")
)

# ---------------------------------------------------------
# TILE GEOMETRY
# ---------------------------------------------------------

def get_tile_positions():
    count = math.ceil(
        (WIDTH - TILE_SIZE) / (TILE_SIZE * 0.65)
    ) + 1

    positions = np.linspace(
        0,
        WIDTH - TILE_SIZE,
        count,
    )

    return sorted(set(int(round(x)) for x in positions))


# ---------------------------------------------------------
# IMAGE PREPROCESSING
# ---------------------------------------------------------


def preprocess_tile(image: Image.Image, device):
    image = image.convert("RGB")

    if image.size != (TILE_SIZE, TILE_SIZE):
        raise ValueError(
            f"Invalid tile dimensions: {image.size}"
        )

    # Resize the square tile to the training image dimensions.
    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR,
    )

    tensor = to_tensor(image)

    tensor = normalize(
        tensor,
        MEAN,
        STD,
    )

    # Match training preprocessing exactly.
    tensor = F.pad(
        tensor,
        (0, 6, 0, 6),
        mode="constant",
        value=0.0,
    )

    if tensor.shape != (3, MODEL_SIZE, MODEL_SIZE):
        raise RuntimeError(
            f"Unexpected tensor shape: {tensor.shape}"
        )

    return tensor.unsqueeze(0).to(device)

# ---------------------------------------------------------
# TILED PROBABILITY INFERENCE
# ---------------------------------------------------------

def tiled_inference(
    image_t0,
    image_t1,
    backbone,
    model,
    device,
):
    positions = get_tile_positions()

    # Accumulate on CPU to avoid unnecessary GPU memory use.
    probability_sum = torch.zeros(
        (HEIGHT, WIDTH),
        dtype=torch.float32,
    )

    weight_sum = torch.zeros(
        (HEIGHT, WIDTH),
        dtype=torch.float32,
    )

    # Smooth positive window for overlapping tile blending.
    window_1d = torch.hann_window(
        TILE_SIZE,
        periodic=False,
        dtype=torch.float32,
    ).clamp(min=0.05)

    weight_window = (
        window_1d[:, None] * window_1d[None, :]
    )

    for x in positions:

        crop_t0 = image_t0.crop(
            (x, 0, x + TILE_SIZE, HEIGHT)
        )

        crop_t1 = image_t1.crop(
            (x, 0, x + TILE_SIZE, HEIGHT)
        )

        tensor_t0 = preprocess_tile(crop_t0, device)
        tensor_t1 = preprocess_tile(crop_t1, device)

        features_t0 = backbone(tensor_t0).float()
        features_t1 = backbone(tensor_t1).float()

        logits = model(features_t0, features_t1)

        probabilities = torch.sigmoid(logits.float())

        if tuple(probabilities.shape) != (
            1, 1, IMAGE_SIZE, IMAGE_SIZE
        ):
            raise ValueError(
                f"Unexpected decoder output: {probabilities.shape}"
            )

        # Restore tile probability map to original tile resolution.
        tile_probability = F.interpolate(
            probabilities,
            size=(TILE_SIZE, TILE_SIZE),
            mode="bilinear",
            align_corners=False,
        )[0, 0].cpu()

        probability_sum[:, x:x + TILE_SIZE] += (
            tile_probability * weight_window
        )

        weight_sum[:, x:x + TILE_SIZE] += weight_window

    if torch.any(weight_sum <= 0):
        raise RuntimeError(
            "Some panorama pixels received no tile coverage."
        )

    return probability_sum / weight_sum


# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

def evaluate(args):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and not args.cpu
        else "cpu"
    )

    print("=" * 65)
    print("TRINETRA - PSCD TILED EXPLORATORY EVALUATION")
    print("=" * 65)

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("Checkpoint:", args.checkpoint)
    print("Threshold:", THRESHOLD)

    records = validate_index(args.index)

    if args.limit is not None:
    	records = records[:args.limit]
    	print(f"TEST MODE: Evaluating only {len(records)} samples.")

    if not args.dataset_root.is_dir():
        raise FileNotFoundError(args.dataset_root)

    print("\nLoading trained model...")

    model = load_trained_model(
        args.checkpoint,
        device,
    )

    print("Model loaded.")

    print("\nLoading frozen DINOv2 backbone...")

    backbone = DINOv2Backbone(
        model_name="facebook/dinov2-base",
        device=str(device),
    )

    backbone.eval()

    print("Backbone loaded.")

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_dir = args.output / "predictions"

    if args.save_predictions:
        prediction_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    global_counts = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
    }

    per_sample_results = []

    macro_keys = (
        "precision",
        "recall",
        "f1",
        "iou",
        "dice",
        "accuracy",
    )

    macro_sum = {
        key: 0.0 for key in macro_keys
    }

    positions = get_tile_positions()

    print("\nTile positions:", positions)
    print("Tile count:", len(positions))
    print("\nStarting evaluation...\n")

    with torch.inference_mode():

        for index, record in enumerate(records, start=1):

            sample_id = record["sample_id"]

            t0_path = resolve_dataset_path(
                record["t0_path"],
                args.dataset_root,
            )

            t1_path = resolve_dataset_path(
                record["t1_path"],
                args.dataset_root,
            )

            mask_path = resolve_dataset_path(
                record["mask_path"],
                args.dataset_root,
            )

            with Image.open(t0_path) as image:
                image_t0 = image.convert("RGB")

            with Image.open(t1_path) as image:
                image_t1 = image.convert("RGB")

            if (
                image_t0.size != (WIDTH, HEIGHT)
                or image_t1.size != (WIDTH, HEIGHT)
            ):
                raise ValueError(
                    f"Unexpected image dimensions: {sample_id}"
                )

            ground_truth = load_binary_mask(mask_path)

            probability = tiled_inference(
                image_t0,
                image_t1,
                backbone,
                model,
                device,
            )

            prediction = (
                probability.numpy() >= THRESHOLD
            )

            tp = int(np.count_nonzero(
                prediction & ground_truth
            ))

            fp = int(np.count_nonzero(
                prediction & ~ground_truth
            ))

            fn = int(np.count_nonzero(
                ~prediction & ground_truth
            ))

            tn = int(np.count_nonzero(
                ~prediction & ~ground_truth
            ))

            metrics = calculate_metrics(
                tp, fp, fn, tn
            )

            for key, value in zip(
                ("tp", "fp", "fn", "tn"),
                (tp, fp, fn, tn),
            ):
                global_counts[key] += value

            for key in macro_keys:
                macro_sum[key] += metrics[key]

            per_sample_results.append({
                "sample_id": sample_id,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                **metrics,
            })

            if args.save_predictions:

                output_mask = (
                    prediction.astype(np.uint8) * 255
                )

                Image.fromarray(
                    output_mask,
                    mode="L",
                ).save(
                    prediction_dir / f"{sample_id}.png"
                )

            if index % 10 == 0 or index == len(records):
                print(
                    f"Processed {index}/{len(records)} "
                    f"({index / len(records) * 100:.1f}%)"
                )

    micro_metrics = calculate_metrics(
        global_counts["tp"],
        global_counts["fp"],
        global_counts["fn"],
        global_counts["tn"],
    )

    macro_metrics = {
        key: macro_sum[key] / len(records)
        for key in macro_keys
    }

    # Save CSV.
    csv_path = args.output / "per_sample_metrics.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(per_sample_results[0].keys()),
        )

        writer.writeheader()
        writer.writerows(per_sample_results)

    # Save JSON.
    results = {
        "dataset": "PSCD",
        "evaluation_type": "exploratory_tiled_inference",
        "samples": len(records),
        "checkpoint": str(args.checkpoint.resolve()),
        "prediction_threshold": THRESHOLD,
        "tile_size": TILE_SIZE,
        "tile_positions": positions,
        "tile_count": len(positions),
        "overlap_pixels": [
            positions[i] + TILE_SIZE - positions[i + 1]
            for i in range(len(positions) - 1)
        ],
        "blending": "hann_weighted_average",
        "preprocessing": {
            "tile_size": [TILE_SIZE, TILE_SIZE],
            "resized_size": [IMAGE_SIZE, IMAGE_SIZE],
            "model_input_size": [MODEL_SIZE, MODEL_SIZE],
            "normalization": "ImageNet",
            "bottom_right_padding": 6,
            "interpolation": "bilinear",
        },
        "global_confusion_counts": global_counts,
        "micro_metrics": micro_metrics,
        "macro_metrics": macro_metrics,
    }

    json_path = args.output / "evaluation_results.json"

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(results, file, indent=4)

    print("\n" + "=" * 65)
    print("PSCD TILED EXPLORATORY RESULTS")
    print("=" * 65)

    print("\nMICRO METRICS")

    for key, value in micro_metrics.items():
        print(f"{key.upper():12s}: {value:.6f}")

    print("\nMACRO METRICS")

    for key, value in macro_metrics.items():
        print(f"{key.upper():12s}: {value:.6f}")

    print("\nConfusion counts:")

    for key, value in global_counts.items():
        print(f"{key.upper():12s}: {value:,}")

    print("\nSaved JSON:", json_path)
    print("Saved CSV:", csv_path)

    if args.save_predictions:
        print("Saved predictions:", prediction_dir)

    print("\nTILED EXPLORATORY EVALUATION COMPLETED.")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description="PSCD tiled exploratory evaluation"
    )

    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX,
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--save-predictions",
        action="store_true",
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
    )

    parser.add_argument(
    	"--limit",
    	type=int,
    	default=None,
    	help="Evaluate only the first N samples for testing.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
