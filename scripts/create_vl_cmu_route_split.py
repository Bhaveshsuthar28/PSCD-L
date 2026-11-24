
import json
import random
from collections import defaultdict
from pathlib import Path

SEED = 42
VALIDATION_RATIO = 0.10

SOURCE = Path("outputs/index/vl_cmu_train.json")
TRAIN_OUTPUT = Path("outputs/index/vl_cmu_train_route_split.json")
VAL_OUTPUT = Path("outputs/index/vl_cmu_val_route.json")


def route_group(sample_id):
    parts = sample_id.split("_")

    if len(parts) != 4:
        raise ValueError(f"Unexpected sample ID: {sample_id}")

    return "_".join(parts[:2])


def location_group(sample_id):
    return "_".join(sample_id.split("_")[:3])


def select_validation_groups(groups, target):
    """Find a deterministic subset of complete route groups near target."""
    group_ids = sorted(groups)
    random.Random(SEED).shuffle(group_ids)

    # Dynamic programming: sample count -> selected route groups.
    reachable = {0: ()}

    for group_id in group_ids:
        group_size = len(groups[group_id])

        for current_size in sorted(list(reachable), reverse=True):
            new_size = current_size + group_size

            if new_size not in reachable:
                reachable[new_size] = (
                    reachable[current_size] + (group_id,)
                )

    best_size = min(
        reachable,
        key=lambda size: (abs(size - target), size > target, size)
    )

    return set(reachable[best_size]), best_size


def main():
    with SOURCE.open("r", encoding="utf-8") as f:
        samples = json.load(f)

    if not samples:
        raise ValueError("Training index is empty.")

    sample_ids = [sample["sample_id"] for sample in samples]

    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate sample IDs found.")

    groups = defaultdict(list)

    for sample in samples:
        groups[route_group(sample["sample_id"])].append(sample)

    # Verify four rotations for every location.
    expected_rotations = {"0", "90", "180", "270"}

    for group_id, records in groups.items():
        locations = defaultdict(set)

        for sample in records:
            parts = sample["sample_id"].split("_")
            location_id = location_group(sample["sample_id"])
            locations[location_id].add(parts[3])

        for location_id, rotations in locations.items():
            if rotations != expected_rotations:
                raise ValueError(
                    f"Invalid rotations in {location_id}: {rotations}"
                )

            matching = [
                sample for sample in records
                if location_group(sample["sample_id"]) == location_id
            ]

            if len(matching) != 4:
                raise ValueError(
                    f"Expected four samples for {location_id}, "
                    f"found {len(matching)}"
                )

    target = round(len(samples) * VALIDATION_RATIO)

    val_groups, val_count = select_validation_groups(groups, target)
    train_groups = set(groups) - val_groups

    train_samples = [
        sample
        for group_id in sorted(train_groups)
        for sample in groups[group_id]
    ]

    val_samples = [
        sample
        for group_id in sorted(val_groups)
        for sample in groups[group_id]
    ]

    # Integrity checks.
    train_ids = {sample["sample_id"] for sample in train_samples}
    val_ids = {sample["sample_id"] for sample in val_samples}
    original_ids = set(sample_ids)

    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids == original_ids
    assert len(train_samples) + len(val_samples) == len(samples)
    assert train_groups.isdisjoint(val_groups)

    train_locations = {
        location_group(sample_id) for sample_id in train_ids
    }
    val_locations = {
        location_group(sample_id) for sample_id in val_ids
    }

    assert train_locations.isdisjoint(val_locations)

    TRAIN_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # Refuse to overwrite existing files.
    for output in (TRAIN_OUTPUT, VAL_OUTPUT):
        if output.exists():
            raise FileExistsError(
                f"Output already exists; refusing to overwrite: {output}"
            )

    with TRAIN_OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(train_samples, f, indent=2)

    with VAL_OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(val_samples, f, indent=2)

    print("\nROUTE-INDEPENDENT VL-CMU SPLIT COMPLETE")
    print(f"Seed: {SEED}")
    print(f"Original samples: {len(samples)}")
    print(f"Total route/pass groups: {len(groups)}")
    print(f"Target validation samples: {target}")
    print(f"Actual validation samples: {len(val_samples)}")
    print(f"Training samples: {len(train_samples)}")
    print(f"Validation percentage: {len(val_samples) / len(samples) * 100:.2f}%")
    print(f"Training route groups: {len(train_groups)}")
    print(f"Validation route groups: {len(val_groups)}")
    print(f"Route overlap: {len(train_groups & val_groups)}")
    print(f"Location overlap: {len(train_locations & val_locations)}")
    print(f"Sample ID overlap: {len(train_ids & val_ids)}")
    print(f"Coverage: {len(train_ids | val_ids)}/{len(original_ids)}")
    print(f"Training index: {TRAIN_OUTPUT}")
    print(f"Validation index: {VAL_OUTPUT}")


if __name__ == "__main__":
    main()
