from pathlib import Path
from collections import Counter
from PIL import Image
import numpy as np


DATASET_ROOT = Path(
    r"C:\Users\bhave\OneDrive\Desktop\AI Projects\PSCD&L\Dataset"
)

MASK_FOLDERS = {
    "train": DATASET_ROOT / "VL-CMU-CD-binary255/train/mask",
    "test": DATASET_ROOT / "VL-CMU-CD-binary255/test/mask",
}


def get_png_files(folder):
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() == ".png"
        and not p.name.startswith("._")
    )


def analyze_folder(name, folder):
    files = get_png_files(folder)

    value_counts = Counter()
    total_pixels = 0

    for index, path in enumerate(files, 1):
        with Image.open(path) as img:
            array = np.asarray(img)

        values, counts = np.unique(array, return_counts=True)

        for value, count in zip(values, counts):
            value_counts[int(value)] += int(count)

        total_pixels += array.size

        if index % 500 == 0:
            print(f"  {name}: {index}/{len(files)}")

    print(f"\n{name}")
    print(f"files: {len(files)}")
    print(f"pixels: {total_pixels}")
    print(f"unique values: {len(value_counts)}")

    for value in sorted(value_counts):
        count = value_counts[value]
        percentage = count / total_pixels * 100

        print(
            f"  {value:3d}: "
            f"{count:12,d} "
            f"({percentage:8.4f}%)"
        )


def main():
    for name, folder in MASK_FOLDERS.items():
        analyze_folder(name, folder)

    print("\nSTEP 3C: COMPLETE")


if __name__ == "__main__":
    main()