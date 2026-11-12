from pathlib import Path
from collections import Counter


ROOT = Path(__file__).resolve().parent.parent / "Dataset"

DATASETS = {
    "PSCD": ROOT / "PSCD",
    "PSCD_pers": ROOT / "PSCD_pers",
    "VL-CMU-CD": ROOT / "VL-CMU-CD-binary255",
    "ABODA": ROOT / "ABODA",
}


def valid_files(path):
    return [
        p for p in path.rglob("*")
        if p.is_file() and not p.name.startswith("._")
    ]


def inspect_folders(name, root):
    print(f"\n{'=' * 60}")
    print(name)
    print(f"{'=' * 60}")

    if not root.exists():
        print("NOT FOUND")
        return

    folders = sorted(
        p for p in root.iterdir()
        if p.is_dir()
    )

    for folder in folders:
        files = valid_files(folder)

        extensions = Counter(
            p.suffix.lower() if p.suffix else "[no extension]"
            for p in files
        )

        print(f"\n{folder.name}")
        print(f"  Files: {len(files):,}")

        if extensions:
            print("  Extensions:")
            for ext, count in sorted(extensions.items()):
                print(f"    {ext:<12} {count:,}")

        samples = sorted(files)[:5]

        if samples:
            print("  Samples:")
            for file in samples:
                print(f"    {file.name}")


def inspect_aboda(root):
    print(f"\n{'=' * 60}")
    print("ABODA")
    print(f"{'=' * 60}")

    videos = sorted(root.glob("*.avi"))

    for video in videos:
        print(f"  {video.name}")


def main():
    print("DATASET STRUCTURE AUDIT")
    print(f"Dataset root: {ROOT}")

    inspect_folders("PSCD", DATASETS["PSCD"])
    inspect_folders("PSCD_pers", DATASETS["PSCD_pers"])
    inspect_folders("VL-CMU-CD", DATASETS["VL-CMU-CD"])
    inspect_aboda(DATASETS["ABODA"])

    print(f"\n{'=' * 60}")
    print("Audit complete")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()