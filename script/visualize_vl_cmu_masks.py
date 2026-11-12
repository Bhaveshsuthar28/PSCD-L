from pathlib import Path
import random

from PIL import Image, ImageDraw


ROOT = Path(
    r"C:\Users\bhave\OneDrive\Desktop\AI Projects\PSCD&L\Dataset"
) / "VL-CMU-CD-binary255"

OUTPUT = Path(
    r"C:\Users\bhave\OneDrive\Desktop\AI Projects\PSCD&L\outputs"
) / "vl_cmu_mask_analysis.png"


def get_png_files(folder):
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() == ".png"
        and not p.name.startswith("._")
    )


def make_panel(t0_path, t1_path, mask_path, size=(320, 320)):
    t0 = Image.open(t0_path).convert("RGB")
    t1 = Image.open(t1_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    t0.thumbnail(size)
    t1.thumbnail(size)
    mask.thumbnail(size)

    mask_rgb = Image.merge("RGB", (mask, mask, mask))

    panel = Image.new("RGB", (size[0] * 3, size[1] + 40), "white")

    panel.paste(t0, (0, 40))
    panel.paste(t1, (size[0], 40))
    panel.paste(mask_rgb, (size[0] * 2, 40))

    draw = ImageDraw.Draw(panel)
    draw.text((10, 10), "t0", fill="black")
    draw.text((size[0] + 10, 10), "t1", fill="black")
    draw.text((size[0] * 2 + 10, 10), "mask", fill="black")

    return panel


def main():
    train_t0 = get_png_files(ROOT / "train/t0")
    train_t1 = get_png_files(ROOT / "train/t1")
    train_mask = get_png_files(ROOT / "train/mask")

    test_t0 = get_png_files(ROOT / "test/t0")
    test_t1 = get_png_files(ROOT / "test/t1")
    test_mask = get_png_files(ROOT / "test/mask")

    random.seed(42)

    train_indices = random.sample(range(len(train_t0)), 4)
    test_indices = random.sample(range(len(test_t0)), 2)

    panels = []

    for i in train_indices:
        panels.append(
            make_panel(
                train_t0[i],
                train_t1[i],
                train_mask[i],
            )
        )

    for i in test_indices:
        panels.append(
            make_panel(
                test_t0[i],
                test_t1[i],
                test_mask[i],
            )
        )

    width = 960
    height = 360 * len(panels)

    output = Image.new("RGB", (width, height), "white")

    for i, panel in enumerate(panels):
        output.paste(panel, (0, i * 360))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output.save(OUTPUT)

    print(f"[PASS] Saved: {OUTPUT}")
    print(f"Panels: {len(panels)}")


if __name__ == "__main__":
    main()