from pathlib import Path
import json

from config import OUTPUT_ROOT


INDEX_DIR = OUTPUT_ROOT / "index"

INDEXES = [
    "pscd",
    "pscd_pers",
    "vl_cmu_train",
    "vl_cmu_test",
]


def inspect_index(name):
    path = INDEX_DIR / f"{name}.json"

    with open(path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    print(f"\n{name}")
    print(f"count: {len(samples)}")

    for sample in samples[:2]:
        print(f"  id:   {sample['sample_id']}")
        print(f"  t0:   {Path(sample['t0_path']).name}")
        print(f"  t1:   {Path(sample['t1_path']).name}")
        print(f"  mask: {Path(sample['mask_path']).name}")

        if "split" in sample:
            print(f"  split: {sample['split']}")


def main():
    for name in INDEXES:
        inspect_index(name)

    print("\nSTEP 5C: PASS")


if __name__ == "__main__":
    main()