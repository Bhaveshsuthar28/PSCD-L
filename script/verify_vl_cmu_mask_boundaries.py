from pathlib import Path
from PIL import Image
import numpy as np


ROOT = Path(
    r"C:\Users\bhave\OneDrive\Desktop\AI Projects\PSCD&L\Dataset"
) / "VL-CMU-CD-binary255"


MASK_FOLDERS = {
    "train": ROOT / "train/mask",
    "test": ROOT / "test/mask",
}


def get_png_files(folder):
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() == ".png"
        and not p.name.startswith("._")
    )


def analyze_mask(path):
    with Image.open(path) as img:
        mask = np.asarray(img)

    foreground = mask == 255
    intermediate = (mask > 0) & (mask < 255)

    if not intermediate.any():
        return 0, 0

    padded = np.pad(
        foreground,
        1,
        mode="constant",
        constant_values=False
    )

    neighbors = (
        padded[:-2, :-2] |
        padded[:-2, 1:-1] |
        padded[:-2, 2:] |
        padded[1:-1, :-2] |
        padded[1:-1, 2:] |
        padded[2:, :-2] |
        padded[2:, 1:-1] |
        padded[2:, 2:]
    )

    intermediate_count = int(intermediate.sum())
    boundary_count = int((intermediate & neighbors).sum())

    return intermediate_count, boundary_count


def analyze_folder(name, folder):
    files = get_png_files(folder)

    total_intermediate = 0
    total_boundary = 0
    masks_with_intermediate = 0

    for index, path in enumerate(files, 1):
        intermediate, boundary = analyze_mask(path)

        total_intermediate += intermediate
        total_boundary += boundary

        if intermediate > 0:
            masks_with_intermediate += 1

        if index % 500 == 0:
            print(f"  {name}: {index}/{len(files)}")

    if total_intermediate:
        percentage = (
            total_boundary / total_intermediate
        ) * 100
    else:
        percentage = 100.0

    print(f"\n{name}")
    print(f"files: {len(files)}")
    print(f"masks with intermediate pixels: {masks_with_intermediate}")
    print(f"intermediate pixels: {total_intermediate:,}")
    print(f"touching 255 region: {total_boundary:,}")
    print(f"boundary percentage: {percentage:.4f}%")


def main():
    for name, folder in MASK_FOLDERS.items():
        analyze_folder(name, folder)

    print("\nSTEP 3E: COMPLETE")


if __name__ == "__main__":
    main()