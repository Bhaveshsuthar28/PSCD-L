
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]

INDEX = ROOT / "outputs/index/vl_cmu_train_val_split.json"

CACHE = ROOT / "data/processed/vl_cmu_cd_v2/train"

OUTPUT = ROOT / "outputs/diagnostics/feature_signal"

SAMPLE_IDS = [
    "131_1_01_0",
    "131_1_03_0",
    "131_1_04_0",
    "001_1_01_0",
    "003_1_04_0",
    "004_1_00_0",
]

FEATURE_DIM = 768
FEATURE_SIZE = 37
PATCH_SIZE = 14
PADDED_SIZE = 518


def load_mask(record):

    from data.datasets import resolve_mask_path

    path = resolve_mask_path(
        record["mask_path"],
        ROOT / "Dataset" / "VL-CMU-CD-binary255",
    )

    mask = np.asarray(
        Image.open(path).convert("L")
    )

    if mask.shape != (512, 512):
        raise ValueError(
            f"Unexpected mask shape: {mask.shape}"
        )

    return (mask >= 128).astype(np.float32)


def load_features(sample_id):

    path = CACHE / f"{sample_id}.pt"

    if not path.is_file():
        raise FileNotFoundError(path)

    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if payload.get("feature_model") != "facebook/dinov2-base":
        raise ValueError(
            f"Unexpected feature model: {sample_id}"
        )

    if payload.get("feature_dim") != FEATURE_DIM:
        raise ValueError(
            f"Unexpected feature dimension: {sample_id}"
        )

    if payload.get("patch_size") != PATCH_SIZE:
        raise ValueError(
            f"Unexpected patch size: {sample_id}"
        )

    t0 = payload["t0"].float()
    t1 = payload["t1"].float()

    expected_shape = (
        1,
        FEATURE_DIM,
        FEATURE_SIZE,
        FEATURE_SIZE,
    )

    if tuple(t0.shape) != expected_shape:
        raise ValueError(
            f"Unexpected T0 feature shape: {tuple(t0.shape)}"
        )

    if tuple(t1.shape) != expected_shape:
        raise ValueError(
            f"Unexpected T1 feature shape: {tuple(t1.shape)}"
        )

    if not torch.isfinite(t0).all():
        raise ValueError("Non-finite T0 features.")

    if not torch.isfinite(t1).all():
        raise ValueError("Non-finite T1 features.")

    return t0, t1


def create_patch_occupancy(mask):

    tensor = torch.from_numpy(mask)[None, None]

    # Match the backbone's bottom/right padding:
    # 512 -> 518 -> 37 patches of 14 pixels.
    padded = F.pad(
        tensor,
        (0, 6, 0, 6),
        mode="constant",
        value=0.0,
    )

    occupancy = F.avg_pool2d(
        padded,
        kernel_size=PATCH_SIZE,
        stride=PATCH_SIZE,
    )

    if tuple(occupancy.shape) != (1, 1, 37, 37):
        raise RuntimeError(
            f"Unexpected occupancy shape: {occupancy.shape}"
        )

    return occupancy[0, 0]


def calculate_weighted_mean(values, weights):

    total_weight = float(weights.sum())

    if total_weight <= 0:
        return None

    return float(
        (values * weights).sum() / total_weight
    )


def calculate_signal_metrics(
    cosine_distance,
    absolute_difference,
    occupancy,
):

    inside_weights = occupancy
    outside_weights = 1.0 - occupancy

    inside_cosine = calculate_weighted_mean(
        cosine_distance,
        inside_weights,
    )

    outside_cosine = calculate_weighted_mean(
        cosine_distance,
        outside_weights,
    )

    inside_absolute = calculate_weighted_mean(
        absolute_difference,
        inside_weights,
    )

    outside_absolute = calculate_weighted_mean(
        absolute_difference,
        outside_weights,
    )

    occupied_cells = int(
        (occupancy > 0).sum()
    )

    majority_occupied_cells = int(
        (occupancy >= 0.5).sum()
    )

    result = {
        "inside_mean_cosine_distance": inside_cosine,
        "outside_mean_cosine_distance": outside_cosine,
        "cosine_inside_outside_ratio": (
            inside_cosine / outside_cosine
            if outside_cosine and inside_cosine is not None
            else None
        ),
        "inside_mean_absolute_feature_difference": inside_absolute,
        "outside_mean_absolute_feature_difference": outside_absolute,
        "absolute_difference_inside_outside_ratio": (
            inside_absolute / outside_absolute
            if outside_absolute and inside_absolute is not None
            else None
        ),
        "occupied_patch_cells": occupied_cells,
        "patch_cells_with_at_least_50_percent_mask": (
            majority_occupied_cells
        ),
        "maximum_patch_occupancy": float(
            occupancy.max()
        ),
    }

    return result


def save_feature_visualization(
    sample_id,
    occupancy,
    cosine_distance,
    absolute_difference,
):

    def normalize(array):

        array = array.astype(np.float32)

        minimum = float(array.min())
        maximum = float(array.max())

        if maximum <= minimum:
            return np.zeros_like(array, dtype=np.uint8)

        normalized = (
            (array - minimum) / (maximum - minimum)
        )

        return (normalized * 255).astype(np.uint8)

    panels = [
        (
            "Ground-truth patch occupancy",
            normalize(occupancy.numpy()),
        ),
        (
            "DINO cosine distance",
            normalize(cosine_distance.numpy()),
        ),
        (
            "Mean absolute feature difference",
            normalize(absolute_difference.numpy()),
        ),
    ]

    panel_size = 370
    label_height = 42
    margin = 15

    canvas = Image.new(
        "RGB",
        (
            len(panels) * panel_size
            + (len(panels) + 1) * margin,
            panel_size + label_height + 2 * margin,
        ),
        "#171b25",
    )

    draw = ImageDraw.Draw(canvas)

    draw.text(
        (margin, 8),
        f"FEATURE SIGNAL | {sample_id}",
        fill="white",
    )

    for index, (label, array) in enumerate(panels):

        x = margin + index * (panel_size + margin)
        y = label_height + margin

        draw.text(
            (x, y - label_height),
            label,
            fill="white",
        )

        image = Image.fromarray(
            array,
            mode="L",
        ).convert("RGB")

        image = image.resize(
            (panel_size, panel_size),
            Image.Resampling.NEAREST,
        )

        canvas.paste(image, (x, y))

    canvas.save(
        OUTPUT / f"{sample_id}_feature_signal.png"
    )


def main():

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with INDEX.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    if len(manifest["val"]) != 372:
        raise ValueError(
            "Expected 372 validation records."
        )

    lookup = {
        record["sample_id"]: record
        for record in manifest["val"]
    }

    rows = []

    print("=" * 70)
    print("TRINETRA DINOv2 FEATURE SIGNAL ANALYSIS")
    print("=" * 70)

    print("Feature model: facebook/dinov2-base")
    print("Feature shape: [768, 37, 37]")
    print("Patch size:", PATCH_SIZE)

    for sample_id in SAMPLE_IDS:

        if sample_id not in lookup:
            raise ValueError(
                f"Sample not in validation split: {sample_id}"
            )

        record = lookup[sample_id]

        mask = load_mask(record)

        t0, t1 = load_features(sample_id)

        # Calculate feature difference maps.
        t0_features = t0[0]
        t1_features = t1[0]

        cosine_distance = 1.0 - F.cosine_similarity(
            t0_features,
            t1_features,
            dim=0,
            eps=1e-8,
        )

        absolute_difference = (
            t1_features - t0_features
        ).abs().mean(dim=0)

        occupancy = create_patch_occupancy(mask)

        metrics = calculate_signal_metrics(
            cosine_distance,
            absolute_difference,
            occupancy,
        )

        metrics["sample_id"] = sample_id
        metrics["original_mask_pixels"] = int(mask.sum())

        rows.append(metrics)

        save_feature_visualization(
            sample_id,
            occupancy,
            cosine_distance,
            absolute_difference,
        )

        print(
            f"\n{sample_id}"
            f"\n  Mask pixels: {metrics['original_mask_pixels']}"
            f"\n  Occupied patches: {metrics['occupied_patch_cells']}"
            f"\n  Patches >=50% mask: "
            f"{metrics['patch_cells_with_at_least_50_percent_mask']}"
            f"\n  Cosine inside/outside: "
            f"{metrics['cosine_inside_outside_ratio']}"
            f"\n  Absolute difference inside/outside: "
            f"{metrics['absolute_difference_inside_outside_ratio']}"
        )

    fields = list(rows[0].keys())

    with (
        OUTPUT / "feature_signal_report.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)

    with (
        OUTPUT / "feature_signal_report.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            rows,
            file,
            indent=4,
        )

    print("\n" + "=" * 70)
    print("FEATURE SIGNAL ANALYSIS COMPLETED")
    print("=" * 70)

    print("Samples analyzed:", len(rows))
    print("Output directory:", OUTPUT.resolve())


if __name__ == "__main__":
    main()
