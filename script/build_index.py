from pathlib import Path
import json

from config import DATASETS, OUTPUT_ROOT


INDEX_DIR = OUTPUT_ROOT / "index"


def get_png_files(folder):
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() == ".png"
        and not p.name.startswith("._")
    )


def build_paired_index(dataset_name, dataset):
    root = dataset["root"]

    t0_files = {
        p.stem: p
        for p in get_png_files(root / dataset["t0"])
    }

    t1_files = {
        p.stem: p
        for p in get_png_files(root / dataset["t1"])
    }

    mask_files = {
        p.stem: p
        for p in get_png_files(root / dataset["mask"])
    }

    common_ids = sorted(
        set(t0_files) &
        set(t1_files) &
        set(mask_files)
    )

    samples = []

    for sample_id in common_ids:
        samples.append({
            "sample_id": sample_id,
            "dataset": dataset_name,
            "t0_path": str(t0_files[sample_id]),
            "t1_path": str(t1_files[sample_id]),
            "mask_path": str(mask_files[sample_id]),
        })

    return samples


def build_split_index(dataset_name, dataset, split):
    root = dataset["root"]
    split_config = dataset[split]

    t0_files = {
        p.stem: p
        for p in get_png_files(root / split_config["t0"])
    }

    t1_files = {
        p.stem: p
        for p in get_png_files(root / split_config["t1"])
    }

    mask_files = {
        p.stem: p
        for p in get_png_files(root / split_config["mask"])
    }

    common_ids = sorted(
        set(t0_files) &
        set(t1_files) &
        set(mask_files)
    )

    samples = []

    for sample_id in common_ids:
        samples.append({
            "sample_id": sample_id,
            "dataset": dataset_name,
            "split": split,
            "t0_path": str(t0_files[sample_id]),
            "t1_path": str(t1_files[sample_id]),
            "mask_path": str(mask_files[sample_id]),
        })

    return samples


def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    pscd_samples = build_paired_index(
        "pscd",
        DATASETS["pscd"]
    )

    pscd_pers_samples = build_paired_index(
        "pscd_pers",
        DATASETS["pscd_pers"]
    )

    vl_train = build_split_index(
        "vl_cmu",
        DATASETS["vl_cmu"],
        "train"
    )

    vl_test = build_split_index(
        "vl_cmu",
        DATASETS["vl_cmu"],
        "test"
    )

    indexes = {
        "pscd": pscd_samples,
        "pscd_pers": pscd_pers_samples,
        "vl_cmu_train": vl_train,
        "vl_cmu_test": vl_test,
    }

    for name, samples in indexes.items():
        output = INDEX_DIR / f"{name}.json"

        with open(output, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2)

        print(f"[PASS] {name}: {len(samples)} samples")


if __name__ == "__main__":
    main()