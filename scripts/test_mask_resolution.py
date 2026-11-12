
from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

INDEX = ROOT / "outputs/index/vl_cmu_train_val_split.json"

DATASET_ROOT = ROOT / "Dataset/VL-CMU-CD-binary255"

OUTPUT = ROOT / "outputs/diagnostics/mask_resolution"

SAMPLE_IDS = [
    "131_1_01_0",
    "131_1_03_0",
    "131_1_04_0",
    "001_1_01_0",
    "003_1_04_0",
    "004_1_00_0",
]


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
        raise ValueError(path_string)

    path = DATASET_ROOT.joinpath(*parts[index + 1:])

    if not path.is_file():
        raise FileNotFoundError(path)

    return path


def main():

    with INDEX.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    lookup = {
        record["sample_id"]: record
        for record in manifest["val"]
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)

    results = []

    print("=" * 65)
    print("MASK FEATURE-RESOLUTION ANALYSIS")
    print("=" * 65)

    for sample_id in SAMPLE_IDS:

        if sample_id not in lookup:
            raise ValueError(
                f"Missing validation sample: {sample_id}"
            )

        record = lookup[sample_id]

        mask_path = resolve_path(record["mask_path"])

        mask = np.asarray(
            Image.open(mask_path).convert("L")
        )

        binary = (mask >= 128).astype(np.float32)

        tensor = torch.from_numpy(binary)[None, None]

        nearest = F.interpolate(
            tensor,
            size=(37, 37),
            mode="nearest",
        )

        area = F.interpolate(
            tensor,
            size=(37, 37),
            mode="area",
        )

        bilinear = F.interpolate(
            tensor,
            size=(37, 37),
            mode="bilinear",
            align_corners=False,
        )

        nearest_positive = int(
            (nearest >= 0.5).sum().item()
        )

        area_positive = int(
            (area >= 0.5).sum().item()
        )

        bilinear_positive = int(
            (bilinear >= 0.5).sum().item()
        )

        original_positive = int(binary.sum())

        row = {
            "sample_id": sample_id,
            "original_positive_pixels": original_positive,
            "original_area_percent": (
                original_positive / binary.size * 100
            ),
            "nearest_positive_cells": nearest_positive,
            "area_positive_cells": area_positive,
            "bilinear_positive_cells": bilinear_positive,
            "nearest_preservation_percent": (
                nearest_positive / original_positive * 100
            ),
            "area_preservation_percent": (
                area_positive / original_positive * 100
            ),
        }

        results.append(row)

        # Save feature-resolution mask visualizations.
        for name, reduced in [
            ("nearest", nearest),
            ("area", area),
            ("bilinear", bilinear),
        ]:

            array = (
                reduced[0, 0].numpy() * 255
            ).clip(0, 255).astype(np.uint8)

            Image.fromarray(array).resize(
                (512, 512),
                Image.Resampling.NEAREST,
            ).save(
                OUTPUT / f"{sample_id}_{name}_37x37.png"
            )

        print(
            f"{sample_id} | "
            f"Original={original_positive} | "
            f"Nearest cells={nearest_positive} | "
            f"Area cells={area_positive} | "
            f"Bilinear cells={bilinear_positive}"
        )

    with (
        OUTPUT / "mask_resolution_report.json"
    ).open("w", encoding="utf-8") as file:

        json.dump(results, file, indent=4)

    print("\nAnalysis completed.")
    print("Output:", OUTPUT.resolve())


if __name__ == "__main__":
    main()
