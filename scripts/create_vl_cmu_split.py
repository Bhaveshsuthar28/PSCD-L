
import json
import random
from collections import defaultdict
from pathlib import Path

SEED = 42
VALIDATION_RATIO = 0.10

SOURCE = Path("outputs/index/vl_cmu_train.json")
TRAIN_OUTPUT = Path("outputs/index/vl_cmu_train_split.json")
VAL_OUTPUT = Path("outputs/index/vl_cmu_val.json")


def main():
    with SOURCE.open("r", encoding="utf-8") as f:
        samples = json.load(f)

    groups = defaultdict(list)

    for sample in samples:
        parts = sample["sample_id"].split("_")

        if len(parts) != 4:
            raise ValueError(
                f"Unexpected sample ID: {sample['sample_id']}"
            )

        group_id = "_".join(parts[:3])
        rotation = parts[3]

        groups[group_id].append((sample, rotation))

    # Verify that every original pair has exactly four rotations.
    expected_rotations = {"0", "90", "180", "270"}

    for group_id, records in groups.items():
        rotations = {rotation for _, rotation in records}

        if len(records) != 4 or rotations != expected_rotations:
            raise ValueError(
                f"Invalid variants for group {group_id}: "
                f"{rotations}, count={len(records)}"
            )

    group_ids = sorted(groups.keys())

    rng = random.Random(SEED)
    rng.shuffle(group_ids)

    val_group_count = round(len(group_ids) * VALIDATION_RATIO)

    val_groups = set(group_ids[:val_group_count])
    train_groups = set(group_ids[val_group_count:])

    train_samples = [
        sample
        for group_id in sorted(train_groups)
        for sample, _ in groups[group_id]
    ]

    val_samples = [
        sample
        for group_id in sorted(val_groups)
        for sample, _ in groups[group_id]
    ]

    # Verify no group appears in both splits.
    assert train_groups.isdisjoint(val_groups)
    assert len(train_samples) + len(val_samples) == len(samples)

    TRAIN_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with TRAIN_OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(train_samples, f, indent=2)

    with VAL_OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(val_samples, f, indent=2)

    print("VL-CMU GROUP-AWARE SPLIT COMPLETE")
    print(f"Random seed: {SEED}")
    print(f"Total original pairs: {len(groups)}")
    print(f"Training groups: {len(train_groups)}")
    print(f"Validation groups: {len(val_groups)}")
    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Training percentage: {len(train_samples) / len(samples) * 100:.2f}%")
    print(f"Validation percentage: {len(val_samples) / len(samples) * 100:.2f}%")
    print(f"Group overlap: {len(train_groups & val_groups)}")
    print(f"Train index: {TRAIN_OUTPUT}")
    print(f"Validation index: {VAL_OUTPUT}")


if __name__ == "__main__":
    main()
