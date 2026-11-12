import json

from config import OUTPUT_ROOT


INDEX_DIR = OUTPUT_ROOT / "index"


FILES = {
    "pscd": INDEX_DIR / "pscd.json",
    "pscd_pers": INDEX_DIR / "pscd_pers.json",
    "vl_cmu_train": INDEX_DIR / "vl_cmu_train.json",
    "vl_cmu_test": INDEX_DIR / "vl_cmu_test.json",
}


def load_index(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    indexes = {
        name: load_index(path)
        for name, path in FILES.items()
    }

    for name, samples in indexes.items():
        ids = [sample["sample_id"] for sample in samples]

        assert len(ids) == len(set(ids))

        for sample in samples:
            assert sample["dataset"] in {
                "pscd",
                "pscd_pers",
                "vl_cmu",
            }

            assert sample["t0_path"].endswith(".png")
            assert sample["t1_path"].endswith(".png")
            assert sample["mask_path"].endswith(".png")

        print(f"[PASS] {name}: {len(samples)} samples")

    train_ids = {
        sample["sample_id"]
        for sample in indexes["vl_cmu_train"]
    }

    test_ids = {
        sample["sample_id"]
        for sample in indexes["vl_cmu_test"]
    }

    overlap = train_ids & test_ids

    assert not overlap

    print("[PASS] VL-CMU train/test IDs: no overlap")

    print("\nSTEP 8B: PASS")


if __name__ == "__main__":
    main()