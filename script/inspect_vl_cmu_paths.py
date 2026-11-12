from pathlib import Path
from collections import defaultdict


ROOT = (
    Path(__file__).resolve().parent.parent
    / "Dataset"
    / "VL-CMU-CD-binary255"
)


def get_files(folder):
    return sorted(
        p for p in folder.rglob("*.png")
        if p.is_file()
        and not p.name.startswith("._")
    )


def inspect_split(name):
    files = get_files(ROOT / name)

    groups = defaultdict(list)

    for file in files:
        parts = file.stem.split("_")

        if len(parts) != 4:
            continue

        key = tuple(parts)

        groups[key].append(file)

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    duplicate_counts = []

    for key, paths in groups.items():
        if len(paths) > 1:
            duplicate_counts.append((key, paths))

    print(f"Unique filename IDs : {len(groups):,}")
    print(f"IDs with duplicates : {len(duplicate_counts):,}")

    print("\nFirst 10 duplicate groups:")

    for key, paths in duplicate_counts[:10]:
        print(f"\n{'_'.join(key)}")

        for path in sorted(paths):
            print(f"  {path.relative_to(ROOT)}")


def main():
    inspect_split("train")
    inspect_split("test")

    print("\n" + "=" * 60)
    print("Inspection complete")
    print("=" * 60)


if __name__ == "__main__":
    main()