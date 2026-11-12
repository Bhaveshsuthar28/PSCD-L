from pathlib import Path

from PIL import Image


ROOT = (
    Path(__file__).resolve().parent.parent
    / "Dataset"
    / "VL-CMU-CD-binary255"
)


def get_files(folder):
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() == ".png"
        and not p.name.startswith("._")
    )


def inspect_file(path):
    with Image.open(path) as image:
        grayscale = image.convert("L")
        values = set(grayscale.getdata())

        print(f"\n{path.name}")
        print(f"  Size   : {image.size}")
        print(f"  Mode   : {image.mode}")
        print(f"  Format : {image.format}")
        print(f"  Values : {sorted(values)[:20]}")
        print(f"  Unique : {len(values)}")


def inspect_split(name, count=12):
    folder = ROOT / name
    files = get_files(folder)

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)
    print(f"Files: {len(files):,}")

    for file in files[:count]:
        inspect_file(file)


def main():
    inspect_split("train")
    inspect_split("test")

    print("\n" + "=" * 60)
    print("Inspection complete")
    print("=" * 60)


if __name__ == "__main__":
    main()