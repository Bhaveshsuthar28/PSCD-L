
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path, PureWindowsPath

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

INDEX_PATH = ROOT / "outputs/index/pscd_pers.json"
DATASET_ROOT = ROOT / "Dataset/PSCD_pers"
OUTPUT_DIR = ROOT / "outputs/diagnostics/pscd_pers_audit"

REQUIRED_DIRS = [
    "t0",
    "t1",
    "mask",
    "cmask",
]

INSPECT_COUNT = 12


def resolve_path(path_string: str) -> Path:
    """Resolve indexed Windows paths on the current machine."""

    parts = PureWindowsPath(str(path_string)).parts

    folder = "PSCD_pers"

    index = next(
        (
            i for i, part in enumerate(parts)
            if part.casefold() == folder.casefold()
        ),
        None,
    )

    if index is None:
        raise ValueError(
            f"Cannot resolve PSCD-PERS path: {path_string}"
        )

    path = DATASET_ROOT.joinpath(*parts[index + 1:])

    return path


def inspect_image(path: Path) -> dict:

    if not path.is_file():
        return {
            "exists": False,
            "path": str(path),
        }

    with Image.open(path) as image:

        result = {
            "exists": True,
            "path": str(path),
            "size": list(image.size),
            "mode": image.mode,
            "format": image.format,
        }

        array = np.asarray(image)

        result["shape"] = list(array.shape)
        result["dtype"] = str(array.dtype)

        if array.ndim == 2:
            values, counts = np.unique(
                array,
                return_counts=True,
            )

            result["unique_values"] = {
                str(int(value)): int(count)
                for value, count in zip(values, counts)
            }

            result["minimum"] = int(array.min())
            result["maximum"] = int(array.max())

        return result


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("TRINETRA PSCD-PERS DATASET AUDIT")
    print("=" * 72)

    if not INDEX_PATH.is_file():
        raise FileNotFoundError(INDEX_PATH)

    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(DATASET_ROOT)

    with INDEX_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise TypeError("Expected a JSON list of records.")

    print("\nINDEX")
    print("Records:", len(records))
    print("Dataset root:", DATASET_ROOT.resolve())

    # Check required directories.
    print("\nDIRECTORY COUNTS")

    directory_counts = {}

    for name in REQUIRED_DIRS:

        directory = DATASET_ROOT / name

        if not directory.is_dir():
            raise FileNotFoundError(directory)

        count = sum(
            1 for path in directory.glob("*.png")
        )

        directory_counts[name] = count

        print(f"{name:10s}: {count}")

    # Verify all indexed image paths and expected cmask files.
    print("\nFULL FILE-EXISTENCE CHECK")

    missing = []
    checked = Counter()

    for record in records:

        sample_id = record["sample_id"]

        for key in ("t0_path", "t1_path", "mask_path"):

            path = resolve_path(record[key])

            checked[key] += 1

            if not path.is_file():
                missing.append({
                    "sample_id": sample_id,
                    "field": key,
                    "path": str(path),
                })

        cmask_path = (
            DATASET_ROOT / "cmask" / f"{sample_id}.png"
        )

        checked["cmask_path"] += 1

        if not cmask_path.is_file():
            missing.append({
                "sample_id": sample_id,
                "field": "cmask_path",
                "path": str(cmask_path),
            })

    print("Files checked:", sum(checked.values()))

    for key, count in checked.items():
        print(f"{key}: {count}")

    print("Missing files:", len(missing))

    # Inspect representative samples.
    print("\nSAMPLE IMAGE INSPECTION")

    inspected = []

    for record in records[:INSPECT_COUNT]:

        sample_id = record["sample_id"]

        print(f"\nSample: {sample_id}")

        paths = {
            "t0": resolve_path(record["t0_path"]),
            "t1": resolve_path(record["t1_path"]),
            "mask": resolve_path(record["mask_path"]),
            "cmask": (
                DATASET_ROOT / "cmask" / f"{sample_id}.png"
            ),
        }

        sample_result = {
            "sample_id": sample_id,
            "images": {},
        }

        for name, path in paths.items():

            details = inspect_image(path)

            sample_result["images"][name] = details

            print(f"\n{name.upper()}")

            if not details["exists"]:
                print("MISSING")
                continue

            print("Size:", details["size"])
            print("Mode:", details["mode"])
            print("Shape:", details["shape"])

            if "unique_values" in details:
                print(
                    "Unique values:",
                    details["unique_values"],
                )

        inspected.append(sample_result)

    # Check sample ID uniqueness.
    sample_ids = [
        record["sample_id"]
        for record in records
    ]

    duplicate_ids = [
        sample_id
        for sample_id, count in Counter(sample_ids).items()
        if count > 1
    ]

    report = {
        "dataset": "PSCD_pers",
        "record_count": len(records),
        "directory_counts": directory_counts,
        "checked_file_counts": dict(checked),
        "missing_files": missing,
        "duplicate_sample_ids": duplicate_ids,
        "inspected_samples": inspected,
    }

    report_path = (
        OUTPUT_DIR / "pscd_pers_audit.json"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
        )

    print("\n" + "=" * 72)
    print("AUDIT SUMMARY")
    print("=" * 72)

    print("Records:", len(records))
    print("Missing files:", len(missing))
    print("Duplicate sample IDs:", len(duplicate_ids))

    print("JSON report:", report_path.resolve())

    if missing:
        print("\nWARNING: Missing files detected.")

    if duplicate_ids:
        print("\nWARNING: Duplicate sample IDs detected.")

    if not missing and not duplicate_ids:
        print("\nFILE-EXISTENCE AND UNIQUENESS CHECK PASSED.")


if __name__ == "__main__":
    main()
