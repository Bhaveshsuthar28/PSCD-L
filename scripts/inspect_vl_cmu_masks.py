
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INDEX_FILES = [
    PROJECT_ROOT / "outputs/index/vl_cmu_train.json",
    PROJECT_ROOT / "outputs/index/vl_cmu_test.json",
]

MAX_SAMPLES = 20


def main() -> None:
    print("=" * 65)
    print("TRINETRA — VL-CMU MASK INSPECTION")
    print("=" * 65)

    all_values = Counter()
    unique_patterns = Counter()
    total_masks = 0

    for index_path in INDEX_FILES:
        if not index_path.exists():
            print(f"\nIndex not found: {index_path}")
            continue

        records = json.loads(
            index_path.read_text(encoding="utf-8")
        )

        print(f"\nINDEX: {index_path.name}")
        print(f"Records: {len(records)}")

        for record in records[:MAX_SAMPLES]:
            mask_path = Path(record["mask_path"])

            if not mask_path.is_absolute():
                mask_path = PROJECT_ROOT / mask_path

            if not mask_path.exists():
                print(f"MISSING: {mask_path}")
                continue

            with Image.open(mask_path) as image:
                mask = np.array(image)

            if mask.ndim != 2:
                print(
                    f"Unexpected mask dimensions: "
                    f"{mask.shape} — {mask_path}"
                )
                continue

            values, counts = np.unique(
                mask,
                return_counts=True,
            )

            value_distribution = {
                int(value): int(count)
                for value, count in zip(values, counts)
            }

            all_values.update(value_distribution)
            unique_patterns.update(values.tolist())
            total_masks += 1

            print(f"\nSample: {record['sample_id']}")
            print(f"Shape: {mask.shape}")
            print(f"Dtype: {mask.dtype}")
            print(f"Unique values: {values.tolist()}")
            print(f"Pixel counts: {value_distribution}")
            print(
                "Nonzero proportion: "
                f"{np.count_nonzero(mask) / mask.size:.6f}"
            )

    print("\n" + "=" * 65)
    print("AGGREGATED RESULTS")
    print("=" * 65)
    print(f"Masks inspected: {total_masks}")
    print(f"Pixel-value frequencies: {dict(sorted(all_values.items()))}")
    print(f"Values encountered: {sorted(unique_patterns)}")


if __name__ == "__main__":
    main()
