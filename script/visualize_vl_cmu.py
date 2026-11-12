from pathlib import Path

from PIL import Image, ImageDraw


ROOT = (
    Path(__file__).resolve().parent.parent
    / "Dataset"
    / "VL-CMU-CD-binary255"
)

OUTPUT = (
    Path(__file__).resolve().parent.parent
    / "outputs"
    / "vl_cmu_inspection.png"
)


def get_files(folder):
    return sorted(
        p for p in folder.rglob("*.png")
        if not p.name.startswith("._")
    )


def create_sheet(files):
    thumb_size = 180
    label_height = 35
    columns = 4
    rows = len(files) // columns + (len(files) % columns > 0)

    sheet = Image.new(
        "RGB",
        (
            columns * thumb_size,
            rows * (thumb_size + label_height)
        ),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    for index, path in enumerate(files):
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((thumb_size, thumb_size))

            x = (index % columns) * thumb_size
            y = (index // columns) * (thumb_size + label_height)

            sheet.paste(image, (x, y))

            draw.text(
                (x + 3, y + thumb_size + 3),
                path.name,
                fill="black"
            )

    return sheet


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    train = get_files(ROOT / "train")
    test = get_files(ROOT / "test")

    selected = []

    train_names = [
        "001_1_00_0.png",
        "001_1_00_90.png",
        "001_1_00_180.png",
        "001_1_00_270.png",
        "001_1_01_0.png",
        "001_1_01_90.png",
        "002_1_00_0.png",
        "002_1_00_90.png",
    ]

    test_names = [
        "000_1_00_0.png",
        "000_1_01_0.png",
        "000_1_02_0.png",
        "000_1_03_0.png",
        "006_1_00_0.png",
        "007_1_00_0.png",
        "007_1_01_0.png",
        "009_1_00_0.png",
    ]

    train_map = {p.name: p for p in train}
    test_map = {p.name: p for p in test}

    for name in train_names:
        if name in train_map:
            selected.append(train_map[name])

    for name in test_names:
        if name in test_map:
            selected.append(test_map[name])

    sheet = create_sheet(selected)
    sheet.save(OUTPUT)

    print(f"Saved: {OUTPUT}")
    print(f"Images: {len(selected)}")


if __name__ == "__main__":
    main()