
from __future__ import annotations

import csv
import json
from pathlib import Path, PureWindowsPath

import numpy as np
from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]

INDEX = ROOT / "outputs/index/vl_cmu_train_val_split.json"

DATASET_ROOT = ROOT / "Dataset/VL-CMU-CD-binary255"

OUTPUT = ROOT / "outputs/diagnostics/failed_samples"

GROUPS = [
    "131_1_01",
    "131_1_03",
    "131_1_04",
]

ROTATIONS = ["0", "90", "180", "270"]

THRESHOLD = 128


def resolve_path(path_string):

    parts = PureWindowsPath(str(path_string)).parts

    folder = "VL-CMU-CD-binary255"

    index = next(
        (
            i for i, part in enumerate(parts)
            if part.casefold() == folder.casefold()
        ),
        None,
    )

    if index is None:
        raise ValueError(
            f"Cannot resolve dataset path: {path_string}"
        )

    path = DATASET_ROOT.joinpath(*parts[index + 1:])

    if not path.is_file():
        raise FileNotFoundError(path)

    return path


def load_image(path, mode):

    image = Image.open(path).convert(mode)

    if image.size != (512, 512):
        raise ValueError(
            f"Unexpected image size {image.size}: {path}"
        )

    return image


def get_bbox(mask):

    ys, xs = np.where(mask)

    if len(xs) == 0:
        return None

    return [
        int(xs.min()),
        int(ys.min()),
        int(xs.max()),
        int(ys.max()),
    ]


def make_difference_map(diff):

    diff_uint8 = np.clip(
        diff * 255,
        0,
        255,
    ).astype(np.uint8)

    return ImageOps.colorize(
        Image.fromarray(diff_uint8, mode="L"),
        black="#101020",
        white="#ffffcc",
    )


def make_mask_overlay(image, mask):

    base = image.convert("RGBA")

    red = Image.new(
        "RGBA",
        image.size,
        (255, 30, 30, 120),
    )

    overlay = Image.new(
        "RGBA",
        image.size,
        (0, 0, 0, 0),
    )

    overlay.paste(
        red,
        (0, 0),
        Image.fromarray(mask.astype(np.uint8) * 255),
    )

    return Image.alpha_composite(
        base,
        overlay,
    ).convert("RGB")


def make_comparison(
    sample_id,
    t0,
    t1,
    mask,
    difference,
    overlay,
):

    tile = 512
    margin = 12
    label_height = 35
    header_height = 65

    canvas = Image.new(
        "RGB",
        (
            2 * tile + 3 * margin,
            header_height + 2 * (tile + label_height + margin),
        ),
        "#171b25",
    )

    draw = ImageDraw.Draw(canvas)

    draw.text(
        (margin, 15),
        f"DATA AUDIT | {sample_id}",
        fill="white",
    )

    tiles = [
        ("T0 - Reference", t0),
        ("T1 - Current", t1),
        ("Ground-truth mask", mask),
        ("Absolute RGB difference", difference),
        ("Ground-truth overlay on T1", overlay),
    ]

    positions = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 0),
    ]

    # Use a 2-column layout; fifth panel spans the bottom.
    for (label, image), (row, column) in zip(tiles, positions):

        x = margin + column * (tile + margin)

        y = (
            header_height
            + margin
            + row * (tile + label_height + margin)
        )

        draw.text(
            (x, y - label_height),
            label,
            fill="white",
        )

        canvas.paste(
            image.convert("RGB").resize((tile, tile)),
            (x, y),
        )

    # Add a separate full-width difference overlay panel
    # by saving the fifth image independently as well.

    return canvas


def main():

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with INDEX.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    records = manifest["val"]

    lookup = {
        record["sample_id"]: record
        for record in records
    }

    sample_ids = [
        f"{group}_{rotation}"
        for group in GROUPS
        for rotation in ROTATIONS
    ]

    missing = [
        sample_id
        for sample_id in sample_ids
        if sample_id not in lookup
    ]

    if missing:
        raise ValueError(
            f"Samples missing from validation index: {missing}"
        )

    report_rows = []

    print("=" * 65)
    print("TRINETRA FAILED-SAMPLE DATA AUDIT")
    print("=" * 65)

    for sample_id in sample_ids:

        record = lookup[sample_id]

        t0 = load_image(
            resolve_path(record["t0_path"]),
            "RGB",
        )

        t1 = load_image(
            resolve_path(record["t1_path"]),
            "RGB",
        )

        raw_mask = load_image(
            resolve_path(record["mask_path"]),
            "L",
        )

        t0_array = np.asarray(t0).astype(np.float32)
        t1_array = np.asarray(t1).astype(np.float32)

        raw_mask_array = np.asarray(raw_mask)

        binary_mask = raw_mask_array >= THRESHOLD

        # Per-pixel mean absolute RGB difference,
        # normalized to [0, 1].
        difference = np.abs(
            t1_array - t0_array
        ).mean(axis=2) / 255.0

        mask_pixels = int(binary_mask.sum())
        total_pixels = int(binary_mask.size)

        inside = difference[binary_mask]
        outside = difference[~binary_mask]

        mean_inside = (
            float(inside.mean())
            if inside.size else 0.0
        )

        mean_outside = (
            float(outside.mean())
            if outside.size else 0.0
        )

        bbox = get_bbox(binary_mask)

        unique_values, value_counts = np.unique(
            raw_mask_array,
            return_counts=True,
        )

        raw_mask_values = {
            str(int(value)): int(count)
            for value, count in zip(
                unique_values,
                value_counts,
            )
        }

        difference_image = make_difference_map(
            difference
        )

        mask_image = Image.fromarray(
            binary_mask.astype(np.uint8) * 255,
            mode="L",
        )

        overlay = make_mask_overlay(
            t1,
            binary_mask,
        )

        comparison = make_comparison(
            sample_id,
            t0,
            t1,
            mask_image,
            difference_image,
            overlay,
        )

        comparison.save(
            OUTPUT / f"{sample_id}_audit.png"
        )

        overlay.save(
            OUTPUT / f"{sample_id}_mask_overlay.png"
        )

        difference_image.save(
            OUTPUT / f"{sample_id}_difference.png"
        )

        row = {
            "sample_id": sample_id,
            "mask_pixels": mask_pixels,
            "mask_area_percent": (
                100.0 * mask_pixels / total_pixels
            ),
            "mask_bbox": json.dumps(bbox),
            "raw_mask_values": json.dumps(raw_mask_values),
            "mean_difference_inside_mask": mean_inside,
            "mean_difference_outside_mask": mean_outside,
            "inside_outside_difference_ratio": (
                mean_inside / mean_outside
                if mean_outside > 0 else None
            ),
            "t0_size": str(t0.size),
            "t1_size": str(t1.size),
            "mask_size": str(raw_mask.size),
        }

        report_rows.append(row)

        print(
            f"{sample_id} | "
            f"Mask pixels={mask_pixels} | "
            f"Mask area={row['mask_area_percent']:.4f}% | "
            f"Inside diff={mean_inside:.4f} | "
            f"Outside diff={mean_outside:.4f}"
        )

    # Save CSV.
    csv_path = OUTPUT / "failed_samples_audit.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(report_rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(report_rows)

    # Save JSON.
    json_path = OUTPUT / "failed_samples_audit.json"

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report_rows,
            file,
            indent=4,
        )

    print("\n" + "=" * 65)
    print("DATA AUDIT COMPLETED")
    print("=" * 65)

    print("Samples audited:", len(report_rows))
    print("CSV:", csv_path.resolve())
    print("JSON:", json_path.resolve())
    print("Images:", OUTPUT.resolve())


if __name__ == "__main__":
    main()
