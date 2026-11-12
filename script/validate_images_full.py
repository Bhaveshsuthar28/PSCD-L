from pathlib import Path
from PIL import Image
import numpy as np


DATASET_ROOT = Path(
    r"C:\Users\bhave\OneDrive\Desktop\AI Projects\PSCD&L\Dataset"
)


FOLDERS = {
    "PSCD/t0": DATASET_ROOT / "PSCD/t0",
    "PSCD/t1": DATASET_ROOT / "PSCD/t1",
    "PSCD/mask": DATASET_ROOT / "PSCD/mask",

    "PSCD_pers/t0": DATASET_ROOT / "PSCD_pers/t0",
    "PSCD_pers/t1": DATASET_ROOT / "PSCD_pers/t1",
    "PSCD_pers/mask": DATASET_ROOT / "PSCD_pers/mask",

    "VL-CMU/train/t0":
        DATASET_ROOT / "VL-CMU-CD-binary255/train/t0",
    "VL-CMU/train/t1":
        DATASET_ROOT / "VL-CMU-CD-binary255/train/t1",
    "VL-CMU/train/mask":
        DATASET_ROOT / "VL-CMU-CD-binary255/train/mask",

    "VL-CMU/test/t0":
        DATASET_ROOT / "VL-CMU-CD-binary255/test/t0",
    "VL-CMU/test/t1":
        DATASET_ROOT / "VL-CMU-CD-binary255/test/t1",
    "VL-CMU/test/mask":
        DATASET_ROOT / "VL-CMU-CD-binary255/test/mask",
}


def get_png_files(folder):
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() == ".png"
        and not p.name.startswith("._")
    )


def validate_file(path):
    try:
        with Image.open(path) as img:
            img.verify()

        with Image.open(path) as img:
            array = np.asarray(img)

            return {
                "size": img.size,
                "mode": img.mode,
                "format": img.format,
                "shape": array.shape,
                "dtype": str(array.dtype),
            }

    except Exception as e:
        return str(e)


def validate_folder(name, folder):
    files = get_png_files(folder)

    if not files:
        print(f"[FAIL] {name}: no PNG files")
        return False

    reference = None
    errors = []
    inconsistent = []

    for index, path in enumerate(files, 1):
        result = validate_file(path)

        if isinstance(result, str):
            errors.append((path.name, result))
            continue

        properties = (
            result["size"],
            result["mode"],
            result["format"],
            result["shape"],
            result["dtype"],
        )

        if reference is None:
            reference = properties
        elif properties != reference:
            inconsistent.append((path.name, properties))

        if index % 500 == 0:
            print(f"  {name}: {index}/{len(files)}")

    if errors:
        print(f"[FAIL] {name}: {len(errors)} corrupted")
        for filename, error in errors[:5]:
            print(f"  {filename}: {error}")
        return False

    if inconsistent:
        print(f"[FAIL] {name}: {len(inconsistent)} inconsistent")
        for filename, properties in inconsistent[:5]:
            print(f"  {filename}: {properties}")
        return False

    print(f"[PASS] {name}: {len(files)} files")
    return True


def main():
    passed = True

    for name, folder in FOLDERS.items():
        if not validate_folder(name, folder):
            passed = False

    print()

    if passed:
        print("STEP 3B: PASS")
    else:
        print("STEP 3B: FAIL")


if __name__ == "__main__":
    main()