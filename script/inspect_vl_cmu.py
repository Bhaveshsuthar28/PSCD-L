from pathlib import Path
from collections import Counter


ROOT = Path(__file__).resolve().parent.parent / "Dataset" / "VL-CMU-CD-binary255"


def inspect_split(name):
    folder = ROOT / name

    files = sorted(
        p for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() == ".png"
        and not p.name.startswith("._")
    )

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(f"Folder: {folder}")
    print(f"Files : {len(files):,}")

    print("\nFirst 20 files:")

    for file in files[:20]:
        print(f"  {file.name}")

    parts = [file.stem.split("_") for file in files]

    lengths = Counter(len(x) for x in parts)

    print("\nFilename parts:")

    for length, count in sorted(lengths.items()):
        print(f"  {length} parts: {count:,}")

    print("\nExamples:")

    for file, tokens in zip(files[:10], parts[:10]):
        print(f"  {file.name} -> {tokens}")


def main():
    print(f"Dataset: {ROOT}")

    if not ROOT.exists():
        print("DATASET NOT FOUND")
        return

    inspect_split("train")
    inspect_split("test")

    print("\n" + "=" * 60)
    print("Audit complete")
    print("=" * 60)


if __name__ == "__main__":
    main()