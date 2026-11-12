from pathlib import Path
import json

from config import OUTPUT_ROOT


INDEX_DIR = OUTPUT_ROOT / "index"


EXPECTED = {
    "pscd": 770,
    "pscd_pers": 3080,
    "vl_cmu_train": 3732,
    "vl_cmu_test": 429,
}


def validate_index(name, expected_count):
    path = INDEX_DIR / f"{name}.json"

    if not path.exists():
        print(f"[FAIL] {name}: index missing")
        return False

    with open(path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if len(samples) != expected_count:
        print(
            f"[FAIL] {name}: "
            f"expected={expected_count}, actual={len(samples)}"
        )
        return False

    required = {
        "sample_id",
        "dataset",
        "t0_path",
        "t1_path",
        "mask_path",
    }

    if "vl_cmu" in name:
        required.add("split")

    ids = set()

    for sample in samples:
        if not required.issubset(sample):
            print(f"[FAIL] {name}: missing fields")
            return False

        sample_id = sample["sample_id"]

        if sample_id in ids:
            print(f"[FAIL] {name}: duplicate ID {sample_id}")
            return False

        ids.add(sample_id)

        for key in ["t0_path", "t1_path", "mask_path"]:
            if not Path(sample[key]).is_file():
                print(
                    f"[FAIL] {name}: "
                    f"missing file {sample[key]}"
                )
                return False

    print(f"[PASS] {name}: {len(samples)} valid samples")
    return True


def main():
    passed = True

    for name, expected_count in EXPECTED.items():
        if not validate_index(name, expected_count):
            passed = False

    print()

    if passed:
        print("STEP 5B: PASS")
    else:
        print("STEP 5B: FAIL")


if __name__ == "__main__":
    main()