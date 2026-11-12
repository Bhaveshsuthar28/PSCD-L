from pathlib import Path
from PIL import Image
import numpy as np


DATASET_ROOT = Path(r"C:\Users\bhave\OneDrive\Desktop\AI Projects\PSCD&L\Dataset")


FOLDERS = {
    "PSCD/t0": DATASET_ROOT / "PSCD/t0",
    "PSCD/t1": DATASET_ROOT / "PSCD/t1",
    "PSCD/mask": DATASET_ROOT / "PSCD/mask",

    "PSCD_pers/t0": DATASET_ROOT / "PSCD_pers/t0",
    "PSCD_pers/t1": DATASET_ROOT / "PSCD_pers/t1",
    "PSCD_pers/mask": DATASET_ROOT / "PSCD_pers/mask",

    "VL-CMU/train/t0": DATASET_ROOT / "VL-CMU-CD-binary255/train/t0",
    "VL-CMU/train/t1": DATASET_ROOT / "VL-CMU-CD-binary255/train/t1",
    "VL-CMU/train/mask": DATASET_ROOT / "VL-CMU-CD-binary255/train/mask",

    "VL-CMU/test/t0": DATASET_ROOT / "VL-CMU-CD-binary255/test/t0",
    "VL-CMU/test/t1": DATASET_ROOT / "VL-CMU-CD-binary255/test/t1",
    "VL-CMU/test/mask": DATASET_ROOT / "VL-CMU-CD-binary255/test/mask",
}


def get_png_files(folder):
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() == ".png"
        and not p.name.startswith("._")
    )


def get_sample(files, n=6):
    if len(files) <= n:
        return files

    mid = len(files) // 2

    return [
        files[0],
        files[1],
        files[mid - 1],
        files[mid],
        files[-2],
        files[-1],
    ]


def inspect_file(path):
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
                "dtype": array.dtype,
                "min": array.min().item(),
                "max": array.max().item(),
                "unique": len(np.unique(array)),
            }

    except Exception as e:
        return {"error": str(e)}


def validate_folder(name, folder):
    if not folder.exists():
        print(f"[FAIL] {name}: folder missing")
        return False

    files = get_png_files(folder)

    if not files:
        print(f"[FAIL] {name}: no PNG files")
        return False

    samples = get_sample(files)

    results = []

    for path in samples:
        result = inspect_file(path)

        if "error" in result:
            print(f"[FAIL] {name}: {path.name}")
            print(f"       {result['error']}")
            return False

        results.append(result)

    properties = {
        (
            r["size"],
            r["mode"],
            r["format"],
            tuple(r["shape"]),
            str(r["dtype"]),
        )
        for r in results
    }

    print(
        f"[PASS] {name}: "
        f"{len(files)} files | "
        f"sample={len(samples)} | "
        f"properties={len(properties)}"
    )

    for i, result in enumerate(results, 1):
        print(
            f"  {i}: "
            f"{result['size']} "
            f"{result['mode']} "
            f"{result['dtype']} "
            f"range={result['min']}-{result['max']} "
            f"unique={result['unique']}"
        )

    return True


def main():
    passed = True

    for name, folder in FOLDERS.items():
        if not validate_folder(name, folder):
            passed = False

    print()

    if passed:
        print("STEP 3A: PASS")
    else:
        print("STEP 3A: FAIL")


if __name__ == "__main__":
    main()