from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent / "Dataset"


def png_files(folder):
    return {
        p.stem: p
        for p in folder.glob("*.png")
        if not p.name.startswith("._")
    }


def validate_pscd():
    root = ROOT / "PSCD"

    folders = {
        "t0": png_files(root / "t0"),
        "t1": png_files(root / "t1"),
        "mask": png_files(root / "mask"),
    }

    ids = set(folders["t0"])

    print("\n" + "=" * 60)
    print("PSCD")
    print("=" * 60)

    for name, files in folders.items():
        print(f"{name:<8}: {len(files):,}")

    missing = {}

    for name, files in folders.items():
        missing[name] = sorted(ids - set(files))

    common = set.intersection(*(set(files) for files in folders.values()))

    print(f"Common   : {len(common):,}")
    print(f"Expected : {len(ids):,}")

    for name, values in missing.items():
        print(f"Missing {name:<2}: {len(values):,}")

    return len(common) == len(ids)


def validate_pscd_pers():
    root = ROOT / "PSCD_pers"

    folders = {
        "t0": png_files(root / "t0"),
        "t1": png_files(root / "t1"),
        "mask": png_files(root / "mask"),
    }

    print("\n" + "=" * 60)
    print("PSCD_pers")
    print("=" * 60)

    for name, files in folders.items():
        print(f"{name:<8}: {len(files):,}")

    all_ids = set(folders["t0"])

    for name, files in folders.items():
        missing = sorted(all_ids - set(files))
        print(f"Missing {name:<2}: {len(missing):,}")

    views = {"F", "B", "L", "R"}

    scene_ids = set()

    for file_id in all_ids:
        parts = file_id.rsplit("_", 1)

        if len(parts) != 2:
            continue

        scene_id, view = parts

        if view in views:
            scene_ids.add(scene_id)

    expected = len(scene_ids) * 4

    print(f"Scenes   : {len(scene_ids):,}")
    print(f"Views    : {len(all_ids):,}")
    print(f"Expected : {expected:,}")

    invalid_views = [
        file_id
        for file_id in all_ids
        if file_id.rsplit("_", 1)[-1] not in views
    ]

    print(f"Bad views: {len(invalid_views):,}")

    cmask = {
        p.stem.replace("mask_sensor_", "")
        for p in (root / "cmask").glob("*.png")
    }

    missing_cmask = sorted(views - cmask)

    print(f"cmask    : {len(cmask):,}")
    print(f"Missing cmask: {len(missing_cmask):,}")

    return (
        all(
            len(set(folders[name])) == len(all_ids)
            for name in folders
        )
        and len(invalid_views) == 0
        and len(all_ids) == expected
        and not missing_cmask
    )


def main():
    pscd_ok = validate_pscd()
    pers_ok = validate_pscd_pers()

    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    print(f"PSCD      : {'PASS' if pscd_ok else 'FAIL'}")
    print(f"PSCD_pers : {'PASS' if pers_ok else 'FAIL'}")


if __name__ == "__main__":
    main()