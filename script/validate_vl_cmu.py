from pathlib import Path


ROOT = (
    Path(__file__).resolve().parent.parent
    / "Dataset"
    / "VL-CMU-CD-binary255"
)


def get_files(folder):
    return {
        p.stem: p
        for p in folder.rglob("*.png")
        if p.is_file() and not p.name.startswith("._")
    }


def validate_split(name):
    root = ROOT / name

    t0 = get_files(root / "t0")
    t1 = get_files(root / "t1")
    mask = get_files(root / "mask")

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(f"t0       : {len(t0):,}")
    print(f"t1       : {len(t1):,}")
    print(f"mask     : {len(mask):,}")

    all_ids = set(t0) | set(t1) | set(mask)
    common_ids = set(t0) & set(t1) & set(mask)

    missing_t0 = all_ids - set(t0)
    missing_t1 = all_ids - set(t1)
    missing_mask = all_ids - set(mask)

    print(f"Common   : {len(common_ids):,}")
    print(f"Expected : {len(all_ids):,}")

    print(f"Missing t0   : {len(missing_t0):,}")
    print(f"Missing t1   : {len(missing_t1):,}")
    print(f"Missing mask : {len(missing_mask):,}")

    invalid = (
        len(missing_t0) > 0
        or len(missing_t1) > 0
        or len(missing_mask) > 0
    )

    if invalid:
        print("\nStatus: FAIL")

        for sample_id in sorted(missing_t0)[:10]:
            print(f"  Missing t0   : {sample_id}")

        for sample_id in sorted(missing_t1)[:10]:
            print(f"  Missing t1   : {sample_id}")

        for sample_id in sorted(missing_mask)[:10]:
            print(f"  Missing mask : {sample_id}")

        return False

    print("\nStatus: PASS")
    return True


def main():
    train_ok = validate_split("train")
    test_ok = validate_split("test")

    print("\n" + "=" * 60)
    print("VL-CMU VALIDATION")
    print("=" * 60)

    print(f"Train : {'PASS' if train_ok else 'FAIL'}")
    print(f"Test  : {'PASS' if test_ok else 'FAIL'}")

    if train_ok and test_ok:
        print("Overall: PASS")
    else:
        print("Overall: FAIL")


if __name__ == "__main__":
    main()