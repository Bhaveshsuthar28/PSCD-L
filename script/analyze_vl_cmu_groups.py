from pathlib import Path
from collections import defaultdict, Counter


ROOT = (
    Path(__file__).resolve().parent.parent
    / "Dataset"
    / "VL-CMU-CD-binary255"
)


def get_files(folder):
    return sorted(
        p for p in folder.rglob("*.png")
        if not p.name.startswith("._")
    )


def analyze_split(name):
    files = get_files(ROOT / name)

    groups = defaultdict(list)

    for file in files:
        parts = file.stem.split("_")

        if len(parts) != 4:
            continue

        key = tuple(parts[:3])
        rotation = parts[3]

        groups[key].append(rotation)

    sizes = Counter(len(values) for values in groups.values())

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(f"Files  : {len(files):,}")
    print(f"Groups : {len(groups):,}")

    print("\nGroup sizes:")

    for size, count in sorted(sizes.items()):
        print(f"  {size} files: {count:,} groups")

    print("\nFirst 20 groups:")

    for key, rotations in list(sorted(groups.items()))[:20]:
        print(f"  {'_'.join(key)} -> {sorted(rotations)}")

    print("\nSecond-field distribution:")

    second = Counter()

    for file in files:
        parts = file.stem.split("_")

        if len(parts) == 4:
            second[parts[1]] += 1

    for value, count in sorted(second.items()):
        print(f"  {value}: {count:,}")


def main():
    analyze_split("train")
    analyze_split("test")

    print("\n" + "=" * 60)
    print("Analysis complete")
    print("=" * 60)


if __name__ == "__main__":
    main()